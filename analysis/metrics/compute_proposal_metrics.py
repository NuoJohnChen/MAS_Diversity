import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def safe_exec(content: str) -> dict:
    # Escape problematic \uXXXX sequences by doubling backslash
    fixed = content.replace("\\u", "\\\\u")
    ns: dict = {}
    exec(fixed, {}, ns)
    return ns


def load_dir(dpath: Path):
    proposals = []
    errors = []
    for fp in sorted(dpath.glob("*_proposals.txt")):
        topic = fp.name.replace("_proposals.txt", "")
        try:
            with fp.open("r", encoding="utf-8") as f:
                content = f.read()
            ns = safe_exec(content)
            papers = ns.get("paper_txts", [])
        except Exception as e:  # pragma: no cover
            errors.append(f"{fp.name}: {e}")
            continue
        for p in papers:
            text = p.strip()
            if text:
                proposals.append((topic, text))
    return proposals, errors


def cosine_dist(a, b):
    denom = (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return 1 - np.einsum("ij,ij->i", a, b) / np.maximum(denom, 1e-12)


def main():
    base_root = Path(
        "./data/extracted_proposals"
    )
    out_root = Path("./data/tsne")
    out_root.mkdir(parents=True, exist_ok=True)

    all_dirs = [p for p in base_root.iterdir() if p.is_dir()]
    if not all_dirs:
        raise SystemExit("No proposal dirs found")

    # Try model
    model_name = "sentence-transformers:all-MiniLM-L6-v2"
    model = None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:  # pragma: no cover
        print(f"Using TF-IDF fallback because sentence-transformers unavailable: {e}")
        model = None
        model_name = "tfidf"

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from scipy.spatial.distance import pdist

    summary_rows = []

    for dpath in sorted(all_dirs, key=lambda p: p.name):
        print(f"Processing {dpath.name}...")
        proposals, errors = load_dir(dpath)
        if not proposals:
            print(f"No proposals in {dpath.name}, skip")
            continue
        texts = [t for _, t in proposals]
        topics = [topic for topic, _ in proposals]
        uniq_topics = sorted(set(topics))
        topic_to_idx = {t: i for i, t in enumerate(uniq_topics)}
        labels = np.array([topic_to_idx[t] for t in topics])

        if model_name.startswith("sentence-transformers") and model is not None:
            emb = model.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            )
        else:
            vec = TfidfVectorizer(
                max_features=4096, ngram_range=(1, 2), min_df=2
            )
            emb = vec.fit_transform(texts).toarray()
            norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norm

        centroids = np.zeros((len(uniq_topics), emb.shape[1]))
        counts = np.zeros(len(uniq_topics), dtype=int)
        for i, t in enumerate(uniq_topics):
            idxs = labels == i
            counts[i] = idxs.sum()
            centroids[i] = emb[idxs].mean(axis=0)
            nrm = np.linalg.norm(centroids[i]) + 1e-12
            centroids[i] = centroids[i] / nrm

        topic_metrics = []
        outlier_90 = 0
        outlier_95 = 0
        total = len(texts)
        for i, t in enumerate(uniq_topics):
            idxs = np.where(labels == i)[0]
            vecs = emb[idxs]
            if len(idxs) == 1:
                dists_l2 = np.array([0.0])
                dists_cos = np.array([0.0])
            else:
                diff = vecs - centroids[i]
                dists_l2 = np.linalg.norm(diff, axis=1)
                dists_cos = cosine_dist(
                    vecs,
                    np.repeat(centroids[i][None, :], len(idxs), axis=0),
                )
            p90_l2 = float(np.percentile(dists_l2, 90))
            p95_l2 = float(np.percentile(dists_l2, 95))
            outlier_90 += (dists_l2 > p90_l2).sum()
            outlier_95 += (dists_l2 > p95_l2).sum()
            topic_metrics.append(
                {
                    "topic": t,
                    "count": len(idxs),
                    "intra_mean_l2": float(dists_l2.mean()),
                    "intra_median_l2": float(np.median(dists_l2)),
                    "intra_mean_cos": float(dists_cos.mean()),
                    "p90_l2": p90_l2,
                    "p95_l2": p95_l2,
                }
            )

        inter_l2 = (
            pdist(centroids, metric="euclidean")
            if len(uniq_topics) > 1
            else np.array([0.0])
        )
        inter_cos = (
            pdist(centroids, metric="cosine")
            if len(uniq_topics) > 1
            else np.array([0.0])
        )

        n = len(texts)
        max_pairs = 200000
        rng = np.random.default_rng(42)

        def pair_dists():
            idx1 = rng.integers(0, n, size=max_pairs)
            idx2 = rng.integers(0, n, size=max_pairs)
            mask = idx1 != idx2
            idx1 = idx1[mask]
            idx2 = idx2[mask]
            diff = emb[idx1] - emb[idx2]
            l2 = np.linalg.norm(diff, axis=1)
            cos = cosine_dist(emb[idx1], emb[idx2])
            same = labels[idx1] == labels[idx2]
            return l2, cos, same

        l2_all, cos_all, same_mask = pair_dists()
        same_l2 = l2_all[same_mask]
        diff_l2 = l2_all[~same_mask]
        same_cos = cos_all[same_mask]
        diff_cos = cos_all[~same_mask]

        silhouette = float("nan")
        db = float("nan")
        ch = float("nan")
        try:
            if len(uniq_topics) > 1:
                silhouette = float(
                    silhouette_score(emb, labels, metric="euclidean")
                )
        except Exception as e:  # pragma: no cover
            print(f"{dpath.name} silhouette error: {e}")
        try:
            if len(uniq_topics) > 1:
                db = float(davies_bouldin_score(emb, labels))
        except Exception as e:  # pragma: no cover
            print(f"{dpath.name} DB error: {e}")
        try:
            if len(uniq_topics) > 1:
                ch = float(calinski_harabasz_score(emb, labels))
        except Exception as e:  # pragma: no cover
            print(f"{dpath.name} CH error: {e}")

        summary_rows.append(
            {
                "dataset": dpath.name,
                "n_samples": n,
                "n_topics": len(uniq_topics),
                "embedding": model_name,
                "intra_mean_l2": float(
                    np.mean([m["intra_mean_l2"] for m in topic_metrics])
                ),
                "intra_median_l2": float(
                    np.mean([m["intra_median_l2"] for m in topic_metrics])
                ),
                "intra_mean_cos": float(
                    np.mean([m["intra_mean_cos"] for m in topic_metrics])
                ),
                "inter_centroid_mean_l2": float(inter_l2.mean())
                if inter_l2.size
                else float("nan"),
                "inter_centroid_min_l2": float(inter_l2.min())
                if inter_l2.size
                else float("nan"),
                "inter_centroid_mean_cos": float(inter_cos.mean())
                if inter_cos.size
                else float("nan"),
                "inter_centroid_min_cos": float(inter_cos.min())
                if inter_cos.size
                else float("nan"),
                "same_mean_l2": float(same_l2.mean()),
                "diff_mean_l2": float(diff_l2.mean()),
                "same_mean_cos": float(same_cos.mean()),
                "diff_mean_cos": float(diff_cos.mean()),
                "same_diff_ratio_l2": float(
                    same_l2.mean() / diff_l2.mean()
                )
                if diff_l2.size
                else float("nan"),
                "silhouette": silhouette,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "outlier_frac_gt_p90": float(outlier_90 / total),
                "outlier_frac_gt_p95": float(outlier_95 / total),
                "errors": "; ".join(errors) if errors else "",
            }
        )

        topic_csv = out_root / f"metrics_{dpath.name}_topics.csv"
        with topic_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "topic",
                    "count",
                    "intra_mean_l2",
                    "intra_median_l2",
                    "intra_mean_cos",
                    "p90_l2",
                    "p95_l2",
                ],
            )
            writer.writeheader()
            writer.writerows(topic_metrics)
        print(f"Wrote topic metrics to {topic_csv}")

    summary_csv = out_root / "metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary to {summary_csv}")


if __name__ == "__main__":
    main()


