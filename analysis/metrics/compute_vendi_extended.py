import os
import math
import csv
import time
import pickle
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


BASE_ROOT = Path("./data/extracted_proposals")
OUT_DIR = Path("./data/tsnenew")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Content-only settings
# -----------------------------

_TOKEN_PATTERN = re.compile(r"[a-z]+")  # fast + stable

# Basic stopwords (avoid nltk download overhead)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "have", "had", "he", "her", "him",
    "i", "in", "is", "it", "its", "may", "might", "more", "most",
    "of", "on", "or", "our", "ours", "she", "so", "some", "such",
    "than", "that", "the", "their", "them", "these", "they", "this",
    "to", "was", "we", "were", "what", "when", "which", "who", "will",
    "with", "would", "you", "your",
}

# Optional: remove common academic boilerplate words to reduce template effects
# (You can tune this list or auto-mine from corpus later.)
_BOILERPLATE = {
    "paper", "work", "study", "approach", "method", "methods", "framework",
    "model", "models", "propose", "proposed", "novel", "new",
    "evaluate", "evaluation", "experiment", "experiments", "results", "performance",
    "dataset", "datasets", "task", "tasks", "analysis", "system", "systems",
}


def safe_exec(content: str) -> dict:
    # 防止文本里出现裸 \uXXXX 被 Python 解析为转义
    fixed = content.replace("\\u", "\\\\u")
    ns: dict = {}
    exec(fixed, {}, ns)
    return ns


def load_dataset(dpath: Path):
    proposals = []
    for fp in sorted(dpath.glob("*_proposals.txt")):
        topic = fp.name.replace("_proposals.txt", "")
        try:
            with fp.open("r", encoding="utf-8") as f:
                content = f.read()
            ns = safe_exec(content)
            papers = ns.get("paper_txts", [])
        except Exception as e:  # pragma: no cover
            print(f"Failed to parse {fp}: {e}")
            continue
        for p in papers:
            text = p.strip()
            if text:
                proposals.append((topic, text))
    return proposals


def get_openai_embeddings_cached(texts, cache_file, model="text-embedding-3-large", use_cache=True):
    """
    Return L2-normalized embeddings using OpenAI text-embedding-3-large.
    - 当 use_cache=True 时：尝试从磁盘缓存加载 / 保存
    - 当 use_cache=False 时：完全忽略缓存，每次都重新请求（但仍可重用此函数的批处理与chunking逻辑）
    """
    if use_cache and cache_file.exists():
        print(f"    [Cache Hit] Loading embeddings from {cache_file.name}...")
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, dict):
                cached_texts_hash = cached_data.get("texts_hash")
                cached_emb = cached_data.get("embeddings")
                current_texts_hash = hash(tuple(texts))
                if cached_texts_hash == current_texts_hash and cached_emb is not None:
                    if len(cached_emb) == len(texts):
                        print(f"    [Cache Hit] Loaded {len(texts)} embeddings from cache")
                        return np.array(cached_emb)
            elif isinstance(cached_data, (list, np.ndarray)):
                if len(cached_data) == len(texts):
                    print(f"    [Cache Hit] Loaded {len(texts)} embeddings from cache (old format)")
                    return np.array(cached_data)
        except Exception as e:
            print(f"    [Cache Miss] Failed to load cache: {e}")

    print(f"    [API Call] Fetching embeddings for {len(texts)} texts (use_cache={use_cache})...")
    from openai import OpenAI

    # NOTE: You hardcoded a key in your original script.
    # Consider using env var in real usage:
    # api_key = os.environ["OPENAI_API_KEY"]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    MAX_TOKENS_PER_BATCH = 200000
    MAX_TOKENS_PER_REQUEST = 300000

    print("    Using fast character-based token estimation (1 token ≈ 4 chars)...")

    def count_tokens(text):
        return len(text) // 4

    def process_batch(batch, batch_num, processed_count, total_count, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model, input=batch)
                batch_emb = np.array([item.embedding for item in response.data], dtype=np.float32)
                processed_count += len(batch)
                print(f"    Processed batch {batch_num}: {processed_count}/{total_count} texts")
                return batch_emb, processed_count
            except Exception as e:
                error_msg = str(e)

                if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                    raise ValueError(f"Invalid OpenAI API key. Original error: {error_msg}")
                if "429" in error_msg or "insufficient_quota" in error_msg.lower():
                    raise ValueError(f"API quota exceeded. Original error: {error_msg}")

                is_retryable = (
                    "connection" in error_msg.lower()
                    or "timeout" in error_msg.lower()
                    or "rate_limit" in error_msg.lower()
                    or "503" in error_msg
                    or "502" in error_msg
                    or "500" in error_msg
                )
                if attempt < max_retries - 1 and is_retryable:
                    wait_time = (attempt + 1) * 2
                    print(f"    Batch {batch_num} failed (attempt {attempt+1}/{max_retries}): {error_msg}")
                    print(f"    Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(
                    f"Failed to get embeddings for batch {batch_num} after {attempt+1} attempts: {error_msg}"
                )

    # Helper: 对单个超长文本做chunking并取平均embedding
    def embed_long_text_with_chunking(text, max_retries=3, chars_per_chunk=32000):
        """
        对超过 MAX_TOKENS_PER_REQUEST 的长文本进行切片，每片单独请求 embedding，
        然后对所有片的向量取平均。这样可以在不截断内容的前提下避免上下文超限。
        """
        # 基础清理
        text = text.replace("\n", " ").strip()
        if not text:
            return np.zeros((1, 3072), dtype=np.float32)

        # 按字符切片（约 4 chars ≈ 1 token，32000 chars ≈ 8000 tokens）
        chunks = [text[i : i + chars_per_chunk] for i in range(0, len(text), chars_per_chunk)]
        chunk_embs = []

        for idx, chunk in enumerate(chunks):
            for attempt in range(max_retries):
                try:
                    resp = client.embeddings.create(model=model, input=[chunk])
                    emb_vec = np.array(resp.data[0].embedding, dtype=np.float32)
                    chunk_embs.append(emb_vec)
                    break
                except Exception as e:
                    error_msg = str(e)
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"    Chunk {idx+1}/{len(chunks)} failed (attempt {attempt+1}/{max_retries}): {error_msg}")
                        print(f"    Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"    Warning: Failed to get embedding for chunk {idx+1}/{len(chunks)}: {error_msg}")
                        # 用零向量占位，保持长度一致
                        chunk_embs.append(np.zeros(3072, dtype=np.float32))
                        break

        if not chunk_embs:
            return np.zeros((1, 3072), dtype=np.float32)

        avg_emb = np.mean(chunk_embs, axis=0, dtype=np.float32)
        return avg_emb.reshape(1, -1)

    all_embeddings = []
    current_batch = []
    current_tokens = 0
    total_texts = len(texts)
    processed_texts = 0
    batch_num = 0

    print(f"    Starting batch processing for {total_texts} texts...")
    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0:
            print(f"    Processing: {i + 1}/{total_texts} texts, current batch: {len(current_batch)} texts")

        text_tokens = count_tokens(text)
        if text_tokens > MAX_TOKENS_PER_REQUEST:
            # 使用chunking方式处理超长文本，而不是直接跳过
            print(f"    Warning: Text {i+1} has estimated {text_tokens} tokens, exceeding API limit.")
            print(f"    Processing this long text with chunked embeddings...")
            long_emb = embed_long_text_with_chunking(text)
            all_embeddings.append(long_emb)
            processed_texts += 1
            continue

        if current_batch and (current_tokens + text_tokens > MAX_TOKENS_PER_BATCH):
            batch_num += 1
            batch_emb, processed_texts = process_batch(current_batch, batch_num, processed_texts, total_texts)
            all_embeddings.append(batch_emb)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += text_tokens

    if current_batch:
        batch_num += 1
        batch_emb, processed_texts = process_batch(current_batch, batch_num, processed_texts, total_texts)
        all_embeddings.append(batch_emb)

    if not all_embeddings:
        raise ValueError("No embeddings were generated")

    emb = np.vstack(all_embeddings)

    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norm

    # 仅当 use_cache=True 时才写入缓存
    if use_cache:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {"texts_hash": hash(tuple(texts)), "embeddings": emb.tolist()}
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"    [Cache Saved] Saved embeddings to {cache_file.name}")
        except Exception as e:
            print(f"    [Warning] Failed to save cache: {e}")

    return emb


def vendi_score(emb: np.ndarray) -> float:
    try:
        from vendi_score import vendi
    except ImportError:
        raise ImportError("vendi_score package not found. Please install it with: pip install vendi-score")

    n = emb.shape[0]
    if n == 0:
        return float("nan")

    K = emb @ emb.T
    K = (K + K.T) / 2.0
    score = vendi.score_K(K)
    return float(score)


def order_parameter(emb: np.ndarray) -> float:
    if emb.shape[0] == 0:
        return float("nan")
    v_avg = emb.mean(axis=0)
    nrm = np.linalg.norm(v_avg) + 1e-12
    v_avg = v_avg / nrm
    dots = emb @ v_avg
    return float(dots.mean())


def pairwise_cosine_distance(emb: np.ndarray) -> float:
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    cosine_similarities = emb @ emb.T
    triu_indices = np.triu_indices(n, k=1)
    similarities = cosine_similarities[triu_indices]
    distances = 1.0 - similarities
    return float(distances.mean())


# -----------------------------
# Content-only WDistinct-n (GLOBAL IDF)
# -----------------------------

def get_content_words(text: str):
    """
    Content-only token stream:
    - lower
    - regex tokenize [a-z]+
    - remove stopwords + optional boilerplate
    """
    text = text.lower()
    toks = _TOKEN_PATTERN.findall(text)
    return [t for t in toks if t not in _STOPWORDS and t not in _BOILERPLATE]


def build_global_content_ngram_idf(all_datasets, n=3):
    """
    Build global IDF across ALL datasets (full corpus) to ensure comparability.
    Streaming DF accumulation (doesn't store all texts).
    """
    df = defaultdict(int)
    valid_docs = 0

    for dpath in all_datasets:
        proposals = load_dataset(dpath)
        if not proposals:
            continue
        texts = [t for _, t in proposals]

        for text in texts:
            words = get_content_words(text)
            if len(words) < n:
                continue

            valid_docs += 1
            grams = set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
            for g in grams:
                df[g] += 1

    if valid_docs == 0:
        return {}, 0

    # Non-negative smoothed IDF: log((1+N)/(1+df))
    idf = {g: float(math.log((1.0 + valid_docs) / (1.0 + dfg))) for g, dfg in df.items()}
    return idf, valid_docs


def content_only_wdistinct_n(texts, idf, n=3):
    """
    WDistinct-n over content-only tokens using GLOBAL IDF table.
    WDistinct-n = sum_idf(unique ngrams) / sum_idf(all ngrams)
    """
    all_ngrams = []
    for text in texts:
        words = get_content_words(text)
        if len(words) < n:
            continue
        all_ngrams.extend(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    if not all_ngrams:
        return float("nan")

    sum_idf_all = 0.0
    sum_idf_unique = 0.0
    seen = set()

    for g in all_ngrams:
        w = idf.get(g, 0.0)
        sum_idf_all += w
        if g not in seen:
            sum_idf_unique += w
            seen.add(g)

    return float(sum_idf_unique / (sum_idf_all + 1e-12))


# -----------------------------
# Existing utilities (self-BLEU etc.) kept as-is
# -----------------------------

def self_bleu(texts):
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        print("nltk not available, falling back to basic BLEU implementation")
        return compute_self_bleu_basic(texts)

    if len(texts) < 2:
        return float("nan")

    smoothing = SmoothingFunction().method1
    bleu_scores = []

    for i, hypothesis in enumerate(texts):
        hyp_tokens = hypothesis.split()
        references = [ref.split() for j, ref in enumerate(texts) if j != i]
        if not references:
            continue
        score = sentence_bleu(references, hyp_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)

    if not bleu_scores:
        return float("nan")

    return float(np.mean(bleu_scores))


def compute_self_bleu_basic(texts):
    if len(texts) < 2:
        return float("nan")

    def ngram_overlap(hyp, refs, n=4):
        hyp_ngrams = [tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1)]
        if not hyp_ngrams:
            return 0.0

        max_overlaps = 0
        for ref in refs:
            ref_ngrams = [tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)]
            ref_counts = {}
            for ng in ref_ngrams:
                ref_counts[ng] = ref_counts.get(ng, 0) + 1

            overlaps = 0
            hyp_counts = {}
            for ng in hyp_ngrams:
                hyp_counts[ng] = hyp_counts.get(ng, 0) + 1

            for ng in hyp_ngrams:
                if ng in ref_counts:
                    overlaps += min(hyp_counts[ng], ref_counts[ng])

            max_overlaps = max(max_overlaps, overlaps)

        return max_overlaps / len(hyp_ngrams)

    bleu_scores = []
    for i, hypothesis in enumerate(texts):
        hyp_tokens = hypothesis.split()
        references = [ref.split() for j, ref in enumerate(texts) if j != i]
        if not references or not hyp_tokens:
            continue

        precisions = [ngram_overlap(hyp_tokens, references, n) for n in range(1, 5)]
        bp = min(1.0, len(hyp_tokens) / max(len(r) for r in references))
        score = bp * (np.prod(precisions) ** (1.0 / len(precisions))) if precisions else 0.0
        bleu_scores.append(score)

    return float(np.mean(bleu_scores)) if bleu_scores else float("nan")


def distinct_n(texts, n=2):
    if len(texts) == 0:
        return float("nan")

    all_ngrams = []
    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i:i+n]))

    if len(all_ngrams) == 0:
        return float("nan")

    return float(len(set(all_ngrams)) / len(all_ngrams))


def load_existing_results(csv_path):
    processed = set()

    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset = row["dataset"]
                    has_content_wdistinct = (
                        "content_only_wdistinct_3" in row
                        and row["content_only_wdistinct_3"]
                        and row["content_only_wdistinct_3"].strip()
                        and row["content_only_wdistinct_3"] != "nan"
                    )
                    has_all_fields = (
                        "vendi_score" in row and row["vendi_score"] and row["vendi_score"].strip() and row["vendi_score"] != "nan"
                        and "order_phi" in row and row["order_phi"] and row["order_phi"].strip() and row["order_phi"] != "nan"
                        and "pcd" in row and row["pcd"] and row["pcd"].strip() and row["pcd"] != "nan"
                        and has_content_wdistinct
                    )
                    if has_all_fields:
                        processed.add(dataset)
                    else:
                        print(f"  Dataset {dataset} exists but missing required metrics, will recompute")
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
    return processed


def main():
    datasets = [p for p in BASE_ROOT.iterdir() if p.is_dir()]
    out_path = OUT_DIR / "metrics_vendi_order.csv"

    processed_datasets = load_existing_results(out_path)

    # Load existing rows
    rows = []
    rows_to_update = {}
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                required_fields = ["vendi_score", "order_phi", "pcd", "content_only_wdistinct_3"]
                for i, row in enumerate(rows):
                    dataset = row["dataset"]
                    has_all_fields = all(
                        field in row and row[field] and row[field].strip() and row[field] != "nan"
                        for field in required_fields
                    )
                    if not has_all_fields:
                        rows_to_update[dataset] = (i, row.copy())
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
            rows = []

    # Decide which datasets we will actually process this run
    datasets_to_process = []
    for dpath in sorted(datasets, key=lambda p: p.name):
        if dpath.name in processed_datasets:
            continue
        datasets_to_process.append(dpath)

    # Build GLOBAL IDF once (critical for cross-mode comparability)
    # UPDATED: build over ALL datasets, not just datasets_to_process
    print("Building GLOBAL content-only IDF for WDistinct-3 over ALL datasets ...")
    global_idf, global_N = build_global_content_ngram_idf(datasets, n=3)
    print(f"  Global IDF built: {len(global_idf)} unique 3-grams over {global_N} valid documents (ALL datasets)")

    for dpath in sorted(datasets, key=lambda p: p.name):
        if dpath.name in processed_datasets:
            print(f"Skipping {dpath.name} (already processed)")
            continue

        print(f"Processing {dpath.name} ...")
        print("  Loading dataset...")
        proposals = load_dataset(dpath)
        if not proposals:
            print("  No proposals, skip.")
            continue

        texts = [t for _, t in proposals]
        n = len(texts)
        print(f"  Loaded {n} proposals")

        if n < 2:
            print("  Need at least 2 samples for metrics, skip.")
            continue

        cache_file = dpath / "embeddings_cache_v3large.pkl"
        print(f"  Starting to get embeddings (IGNORING cache this run: {cache_file.name})...")
        try:
            # 本次运行显式关闭缓存：每个dataset都重新计算embeddings
            emb = get_openai_embeddings_cached(texts, cache_file, model="text-embedding-3-large", use_cache=False)
            model_name = "text-embedding-3-large"
            print(f"  Got embeddings: {emb.shape[0]} texts, {emb.shape[1]} dimensions")
        except (ValueError, RuntimeError) as e:
            print(f"  Failed to get embeddings: {e}")
            continue

        print("  Computing Vendi Score...")
        try:
            v_score = vendi_score(emb)
        except Exception as e:
            print(f"  Failed to compute Vendi Score: {e}")
            v_score = float("nan")

        print("  Computing Order Parameter...")
        phi = order_parameter(emb)

        print("  Computing PCD...")
        try:
            pcd = pairwise_cosine_distance(emb)
        except Exception as e:
            print(f"  Failed to compute PCD: {e}")
            pcd = float("nan")

        # Content-only WDistinct-3 using GLOBAL IDF
        print("  Computing Content-only WDistinct-3 (GLOBAL IDF)...")
        try:
            content_wdistinct_3 = content_only_wdistinct_n(texts, global_idf, n=3)
        except Exception as e:
            print(f"  Failed to compute Content-only WDistinct-3: {e}")
            content_wdistinct_3 = float("nan")

        print(
            f"  n={n}, model={model_name}, "
            f"Vendi={v_score:.3f}, "
            f"Order(phi)={phi:.3f}, "
            f"PCD={pcd:.3f}, "
            f"Content-only WDistinct-3={content_wdistinct_3:.3f}"
        )
        print("  Note: For diversity, higher is better for Vendi, PCD, Content-only WDistinct-3.")
        print("        Lower is better for Order(phi) (use 1 - Order for visualization).")

        new_row = {
            "dataset": dpath.name,
            "n_samples": str(n),
            "embedding": model_name,
            "vendi_score": str(v_score) if not math.isnan(v_score) else "nan",
            "order_phi": str(phi) if not math.isnan(phi) else "nan",
            "pcd": str(pcd) if not math.isnan(pcd) else "nan",
            "content_only_wdistinct_3": str(content_wdistinct_3) if not math.isnan(content_wdistinct_3) else "nan",
        }

        if dpath.name in rows_to_update:
            idx, _existing_row = rows_to_update[dpath.name]
            for old_field in ["keyphrase_wdistinct_3", "kp_wdistinct_3", "content_wdistinct_3"]:
                if old_field in rows[idx]:
                    del rows[idx][old_field]
            for key, value in new_row.items():
                rows[idx][key] = value
            print(f"  Updated existing row for {dpath.name}")
        else:
            rows.append(new_row)

    if rows:
        fieldnames = [
            "dataset",
            "n_samples",
            "embedding",
            "vendi_score",
            "order_phi",
            "pcd",
            "content_only_wdistinct_3",
        ]
        cleaned_rows = []
        for row in rows:
            cleaned_row = {field: row.get(field, "nan") for field in fieldnames}
            cleaned_rows.append(cleaned_row)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)
        print(f"Wrote {out_path}")
    else:
        print("No datasets processed; nothing written.")


if __name__ == "__main__":
    main()


# import math
# import csv
# import time
# import pickle
# import re
# from pathlib import Path

# import numpy as np


# BASE_ROOT = Path("./data/extracted_proposals")
# OUT_DIR = Path("./data/tsne")
# OUT_DIR.mkdir(parents=True, exist_ok=True)


# def safe_exec(content: str) -> dict:
#     # 防止文本里出现裸 \uXXXX 被 Python 解析为转义
#     fixed = content.replace("\\u", "\\\\u")
#     ns: dict = {}
#     exec(fixed, {}, ns)
#     return ns


# def load_dataset(dpath: Path):
#     proposals = []
#     for fp in sorted(dpath.glob("*_proposals.txt")):
#         topic = fp.name.replace("_proposals.txt", "")
#         try:
#             with fp.open("r", encoding="utf-8") as f:
#                 content = f.read()
#             ns = safe_exec(content)
#             papers = ns.get("paper_txts", [])
#         except Exception as e:  # pragma: no cover
#             print(f"Failed to parse {fp}: {e}")
#             continue
#         for p in papers:
#             text = p.strip()
#             if text:
#                 proposals.append((topic, text))
#     return proposals


# def get_openai_embeddings_cached(texts, cache_file, model="text-embedding-3-large"):
#     """Return L2-normalized embeddings using OpenAI text-embedding-3-large with caching.
    
#     Unified embedding model for all metrics (Vendi, Order, PCD).
#     Uses batched processing to handle large datasets that exceed token limits.
#     Caches embeddings to avoid redundant API calls.
#     """
#     # Try to load from cache first
#     if cache_file.exists():
#         print(f"    [Cache Hit] Loading embeddings from {cache_file.name}...")
#         try:
#             with open(cache_file, 'rb') as f:
#                 cached_data = pickle.load(f)
#             # Check if cached data matches current texts
#             if isinstance(cached_data, dict):
#                 # New format: dict with texts_hash and embeddings
#                 cached_texts_hash = cached_data.get('texts_hash')
#                 cached_emb = cached_data.get('embeddings')
#                 current_texts_hash = hash(tuple(texts))
#                 if cached_texts_hash == current_texts_hash and cached_emb is not None:
#                     if len(cached_emb) == len(texts):
#                         print(f"    [Cache Hit] Loaded {len(texts)} embeddings from cache")
#                         return np.array(cached_emb)
#             elif isinstance(cached_data, (list, np.ndarray)):
#                 # Old format: just embeddings array
#                 if len(cached_data) == len(texts):
#                     print(f"    [Cache Hit] Loaded {len(texts)} embeddings from cache (old format)")
#                     return np.array(cached_data)
#         except Exception as e:
#             print(f"    [Cache Miss] Failed to load cache: {e}")
    
#     # Cache miss - fetch from API
#     print(f"    [API Call] Fetching embeddings for {len(texts)} texts...")
#     from openai import OpenAI

#     api_key = os.environ.get("OPENAI_API_KEY", "")
#     base_url = "https://api.openai.com/v1"
#     client = OpenAI(api_key=api_key, base_url=base_url)
    
#     # OpenAI API limit: 300,000 tokens per request
#     # We'll use a conservative limit of 200,000 tokens per batch to leave safety margin
#     MAX_TOKENS_PER_BATCH = 200000
#     MAX_TOKENS_PER_REQUEST = 300000  # Hard limit from API
    
#     # Use fast character-based estimation instead of tiktoken for speed
#     # tiktoken is accurate but slow for large datasets
#     print(f"    Using fast character-based token estimation (1 token ≈ 4 chars)...")
#     def count_tokens(text):
#         # Fast estimation: 1 token ≈ 4 characters (conservative estimate)
#         # This is faster than tiktoken and sufficient for batching
#         return len(text) // 4
    
#     def process_batch(batch, batch_num, processed_count, total_count, max_retries=3):
#         """Process a batch of texts and return embeddings with retry logic."""
#         for attempt in range(max_retries):
#             try:
#                 response = client.embeddings.create(
#                     model="text-embedding-3-large", input=batch
#                 )
#                 batch_emb = np.array([item.embedding for item in response.data], dtype=np.float32)
#                 processed_count += len(batch)
#                 print(f"    Processed batch {batch_num}: {processed_count}/{total_count} texts")
#                 return batch_emb, processed_count
#             except Exception as e:
#                 error_msg = str(e)
                
#                 # Don't retry for these errors
#                 if "401" in error_msg or "invalid_api_key" in error_msg.lower():
#                     raise ValueError(
#                         f"Invalid OpenAI API key. Please check your API key. "
#                         f"Original error: {error_msg}"
#                     )
#                 elif "429" in error_msg or "insufficient_quota" in error_msg.lower():
#                     raise ValueError(
#                         f"API quota exceeded. Please check your billing. "
#                         f"Original error: {error_msg}"
#                     )
                
#                 # Retry for connection errors and rate limits
#                 is_retryable = (
#                     "connection" in error_msg.lower() or
#                     "timeout" in error_msg.lower() or
#                     "rate_limit" in error_msg.lower() or
#                     "503" in error_msg or
#                     "502" in error_msg or
#                     "500" in error_msg
#                 )
                
#                 if attempt < max_retries - 1 and is_retryable:
#                     wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
#                     print(f"    Batch {batch_num} failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
#                     print(f"    Retrying in {wait_time} seconds...")
#                     time.sleep(wait_time)
#                     continue
#                 else:
#                     # Last attempt failed or non-retryable error
#                     raise RuntimeError(
#                         f"Failed to get OpenAI embeddings for batch {batch_num} after {attempt + 1} attempts: {error_msg}"
#                     )
    
#     # Batch texts to avoid exceeding token limit
#     all_embeddings = []
#     current_batch = []
#     current_tokens = 0
#     total_texts = len(texts)
#     processed_texts = 0
#     batch_num = 0
    
#     print(f"    Starting batch processing for {total_texts} texts...")
#     for i, text in enumerate(texts):
#         # Show progress every 50 texts
#         if (i + 1) % 50 == 0:
#             print(f"    Processing: {i + 1}/{total_texts} texts, current batch: {len(current_batch)} texts")
#         text_tokens = count_tokens(text)
        
#         # Check if single text exceeds hard limit
#         if text_tokens > MAX_TOKENS_PER_REQUEST:
#             print(f"    Warning: Text {i+1} has {text_tokens} tokens, exceeding API limit. Skipping.")
#             # Create a dummy embedding (zeros) for this text to maintain alignment
#             dummy_emb = np.zeros((1, 3072), dtype=np.float32)  # text-embedding-3-large has 3072 dims
#             all_embeddings.append(dummy_emb)
#             processed_texts += 1
#             continue
        
#         # If adding this text would exceed the limit, process current batch first
#         if current_batch and (current_tokens + text_tokens > MAX_TOKENS_PER_BATCH):
#             batch_num += 1
#             batch_emb, processed_texts = process_batch(
#                 current_batch, batch_num, processed_texts, total_texts
#             )
#             all_embeddings.append(batch_emb)
            
#             # Reset batch
#             current_batch = []
#             current_tokens = 0
        
#         # Add text to current batch
#         current_batch.append(text)
#         current_tokens += text_tokens
    
#     # Process remaining batch
#     if current_batch:
#         batch_num += 1
#         batch_emb, processed_texts = process_batch(
#             current_batch, batch_num, processed_texts, total_texts
#         )
#         all_embeddings.append(batch_emb)
    
#     # Concatenate all batches
#     if not all_embeddings:
#         raise ValueError("No embeddings were generated")
    
#     emb = np.vstack(all_embeddings)
    
#     # Ensure L2 normalization
#     norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
#     emb = emb / norm
    
#     # Save to cache
#     try:
#         cache_file.parent.mkdir(parents=True, exist_ok=True)
#         cache_data = {
#             'texts_hash': hash(tuple(texts)),
#             'embeddings': emb.tolist()  # Convert to list for JSON-like storage
#         }
#         with open(cache_file, 'wb') as f:
#             pickle.dump(cache_data, f)
#         print(f"    [Cache Saved] Saved embeddings to {cache_file.name}")
#     except Exception as e:
#         print(f"    [Warning] Failed to save cache: {e}")
    
#     return emb


# def vendi_score(emb: np.ndarray) -> float:
#     """Compute Vendi Score using vendi_score package.
    
#     Uses vendi.score_K(K) where K is the cosine similarity kernel matrix.
#     K_ij = cos(x_i, x_j) = <e_i, e_j> where emb rows are L2 normalized.
#     """
#     try:
#         from vendi_score import vendi
#     except ImportError:
#         raise ImportError(
#             "vendi_score package not found. Please install it with: pip install vendi-score"
#         )
    
#     n = emb.shape[0]
#     if n == 0:
#         return float("nan")
    
#     # Compute cosine similarity kernel matrix
#     K = emb @ emb.T  # (n, n)
#     # Ensure symmetry and numerical stability
#     K = (K + K.T) / 2.0
    
#     # Use vendi_score package
#     score = vendi.score_K(K)
#     return float(score)


# def order_parameter(emb: np.ndarray) -> float:
#     """Order parameter phi = average cos(v_i, v_avg)."""
#     if emb.shape[0] == 0:
#         return float("nan")
#     v_avg = emb.mean(axis=0)
#     nrm = np.linalg.norm(v_avg) + 1e-12
#     v_avg = v_avg / nrm
#     dots = emb @ v_avg
#     return float(dots.mean())




# def pairwise_cosine_distance(emb: np.ndarray) -> float:
#     """Compute average pairwise cosine distance.
    
#     Cosine distance = 1 - cosine similarity
#     For L2-normalized embeddings, cosine similarity = dot product.
#     """
#     n = emb.shape[0]
#     if n < 2:
#         return float("nan")
#     # Compute cosine similarities (dot products for L2-normalized vectors)
#     cosine_similarities = emb @ emb.T  # (n, n)
#     # Get upper triangle (excluding diagonal) to avoid duplicates
#     triu_indices = np.triu_indices(n, k=1)
#     similarities = cosine_similarities[triu_indices]
#     # Convert to distances
#     distances = 1.0 - similarities
#     return float(distances.mean())


# def self_bleu(texts):
#     """Compute Self-BLEU score.
    
#     For each sentence i, treat it as hypothesis and all other sentences as references.
#     Compute BLEU score, then average over all sentences.
#     """
#     try:
#         from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#     except ImportError:
#         print("nltk not available, falling back to basic BLEU implementation")
#         return compute_self_bleu_basic(texts)
    
#     if len(texts) < 2:
#         return float("nan")
    
#     smoothing = SmoothingFunction().method1
#     bleu_scores = []
    
#     for i, hypothesis in enumerate(texts):
#         # Tokenize hypothesis
#         hyp_tokens = hypothesis.split()
#         # All other sentences as references
#         references = [ref.split() for j, ref in enumerate(texts) if j != i]
        
#         if not references:
#             continue
            
#         # Compute BLEU score
#         score = sentence_bleu(
#             references, hyp_tokens, smoothing_function=smoothing
#         )
#         bleu_scores.append(score)
    
#     if not bleu_scores:
#         return float("nan")
    
#     return float(np.mean(bleu_scores))


# def compute_self_bleu_basic(texts):
#     """Basic BLEU implementation without nltk."""
#     if len(texts) < 2:
#         return float("nan")
    
#     def ngram_overlap(hyp, refs, n=4):
#         """Compute n-gram overlap between hypothesis and references."""
#         hyp_ngrams = []
#         for i in range(len(hyp) - n + 1):
#             hyp_ngrams.append(tuple(hyp[i:i+n]))
        
#         if not hyp_ngrams:
#             return 0.0
        
#         max_overlaps = 0
#         for ref in refs:
#             ref_ngrams = []
#             for i in range(len(ref) - n + 1):
#                 ref_ngrams.append(tuple(ref[i:i+n]))
#             ref_ngram_counts = {}
#             for ng in ref_ngrams:
#                 ref_ngram_counts[ng] = ref_ngram_counts.get(ng, 0) + 1
            
#             overlaps = 0
#             hyp_ngram_counts = {}
#             for ng in hyp_ngrams:
#                 hyp_ngram_counts[ng] = hyp_ngram_counts.get(ng, 0) + 1
            
#             for ng in hyp_ngrams:
#                 if ng in ref_ngram_counts:
#                     overlaps += min(hyp_ngram_counts[ng], ref_ngram_counts[ng])
            
#             max_overlaps = max(max_overlaps, overlaps)
        
#         return max_overlaps / len(hyp_ngrams) if hyp_ngrams else 0.0
    
#     bleu_scores = []
#     for i, hypothesis in enumerate(texts):
#         hyp_tokens = hypothesis.split()
#         references = [ref.split() for j, ref in enumerate(texts) if j != i]
        
#         if not references or not hyp_tokens:
#             continue
        
#         # Simplified BLEU: average of 1-4 gram precisions
#         precisions = []
#         for n in range(1, 5):
#             prec = ngram_overlap(hyp_tokens, references, n)
#             precisions.append(prec)
        
#         # BLEU = geometric mean of precisions * brevity penalty
#         bp = min(1.0, len(hyp_tokens) / max(len(r) for r in references) if references else 1.0)
#         score = bp * (np.prod(precisions) ** (1.0 / len(precisions))) if precisions else 0.0
#         bleu_scores.append(score)
    
#     return float(np.mean(bleu_scores)) if bleu_scores else float("nan")


# def distinct_n(texts, n=2):
#     """Compute Distinct-n score: ratio of unique n-grams to total n-grams.
    
#     Distinct-n = |unique n-grams| / |total n-grams|
    
#     Higher values indicate more diversity.
#     Range: [0, 1]
#     """
#     if len(texts) == 0:
#         return float("nan")
    
#     all_ngrams = []
#     for text in texts:
#         tokens = text.split()
#         if len(tokens) < n:
#             continue
#         # Extract n-grams
#         for i in range(len(tokens) - n + 1):
#             ngram = tuple(tokens[i:i+n])
#             all_ngrams.append(ngram)
    
#     if len(all_ngrams) == 0:
#         return float("nan")
    
#     unique_ngrams = set(all_ngrams)
#     distinct_score = len(unique_ngrams) / len(all_ngrams)
#     return float(distinct_score)


# def idf_weighted_distinct_n(texts, n=2):
#     """Compute IDF-weighted Distinct-n score.
    
#     WDistinct-n = sum(IDF(g) for g in unique n-grams) / sum(IDF(g) for g in all n-grams)
    
#     This gives lower weight to common phrases (like "we propose", "we evaluate")
#     and higher weight to novel word combinations.
    
#     Higher values indicate more diversity.
#     """
#     if len(texts) == 0:
#         return float("nan")
    
#     # Step 1: Extract all n-grams from all texts
#     all_ngrams_list = []
#     ngrams_per_text = []
    
#     for text in texts:
#         tokens = text.split()
#         if len(tokens) < n:
#             ngrams_per_text.append([])
#             continue
#         text_ngrams = []
#         for i in range(len(tokens) - n + 1):
#             ngram = tuple(tokens[i:i+n])
#             text_ngrams.append(ngram)
#             all_ngrams_list.append(ngram)
#         ngrams_per_text.append(text_ngrams)
    
#     if len(all_ngrams_list) == 0:
#         return float("nan")
    
#     # Step 2: Compute IDF for each n-gram
#     # IDF(g) = log(N / (1 + df(g))) where N is number of texts, df is document frequency
#     ngram_to_doc_freq = {}
#     for text_ngrams in ngrams_per_text:
#         unique_ngrams_in_text = set(text_ngrams)
#         for ngram in unique_ngrams_in_text:
#             ngram_to_doc_freq[ngram] = ngram_to_doc_freq.get(ngram, 0) + 1
    
#     N = len([t for t in texts if len(t.split()) >= n])
#     if N == 0:
#         return float("nan")
    
#     # Step 3: Compute IDF for each n-gram
#     ngram_to_idf = {}
#     for ngram, df in ngram_to_doc_freq.items():
#         idf = np.log(N / (1 + df))
#         ngram_to_idf[ngram] = idf
    
#     # Step 4: Compute weighted distinct score
#     unique_ngrams = set(all_ngrams_list)
    
#     # Sum IDF for unique n-grams
#     sum_idf_unique = sum(ngram_to_idf.get(ngram, 0) for ngram in unique_ngrams)
    
#     # Sum IDF for all n-grams
#     sum_idf_all = sum(ngram_to_idf.get(ngram, 0) for ngram in all_ngrams_list)
    
#     if sum_idf_all == 0:
#         return float("nan")
    
#     weighted_distinct = sum_idf_unique / sum_idf_all
#     return float(weighted_distinct)


# def extract_keyphrases(text, method="tfidf", top_k=10):
#     """Extract keyphrases from a text using KeyBERT, YAKE, or TF-IDF.
    
#     Returns a list of keyphrases (strings).
#     """
#     if method == "keybert":
#         try:
#             from keybert import KeyBERT
#             # Use a lightweight model for speed
#             kw_model = KeyBERT(model="all-MiniLM-L6-v2")
#             keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_k)
#             return [kw[0] for kw in keywords]
#         except ImportError:
#             print("    Warning: KeyBERT not available, falling back to TF-IDF")
#             method = "tfidf"
    
#     if method == "yake":
#         try:
#             import yake
#             kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=top_k)
#             keywords = kw_extractor.extract_keywords(text)
#             return [kw[1] for kw in keywords]
#         except ImportError:
#             print("    Warning: YAKE not available, falling back to TF-IDF")
#             method = "tfidf"
    
#     # Default: TF-IDF based extraction
#     if method == "tfidf":
#         try:
#             from sklearn.feature_extraction.text import TfidfVectorizer
#             from sklearn.feature_extraction.text import CountVectorizer
            
#             # Extract unigrams and bigrams
#             vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=top_k*2, stop_words='english')
#             try:
#                 tfidf_matrix = vectorizer.fit_transform([text])
#             except ValueError:
#                 # If text is too short, use count vectorizer
#                 vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=top_k)
#                 tfidf_matrix = vectorizer.fit_transform([text])
            
#             feature_names = vectorizer.get_feature_names_out()
#             scores = tfidf_matrix.toarray()[0]
            
#             # Get top keyphrases
#             top_indices = np.argsort(scores)[-top_k:][::-1]
#             keyphrases = [feature_names[i] for i in top_indices if scores[i] > 0]
#             return keyphrases
#         except Exception as e:
#             print(f"    Warning: TF-IDF extraction failed: {e}")
#             # Fallback: return first few words
#             tokens = text.split()[:top_k]
#             return tokens
    
#     return []


# def keyphrase_distinct_n(texts, n=2, method="tfidf", top_k=10):
#     """Compute Keyphrase Distinct-n score.
    
#     First extracts keyphrases from each text, then computes distinct-n on the keyphrases.
    
#     Higher values indicate more diversity in keyphrases.
#     """
#     if len(texts) == 0:
#         return float("nan")
    
#     # Extract keyphrases from each text
#     all_keyphrase_ngrams = []
#     for text in texts:
#         keyphrases = extract_keyphrases(text, method=method, top_k=top_k)
#         if not keyphrases:
#             continue
        
#         # Convert keyphrases to tokens and extract n-grams
#         for keyphrase in keyphrases:
#             tokens = keyphrase.split()
#             if len(tokens) < n:
#                 continue
#             for i in range(len(tokens) - n + 1):
#                 ngram = tuple(tokens[i:i+n])
#                 all_keyphrase_ngrams.append(ngram)
    
#     if len(all_keyphrase_ngrams) == 0:
#         return float("nan")
    
#     unique_ngrams = set(all_keyphrase_ngrams)
#     distinct_score = len(unique_ngrams) / len(all_keyphrase_ngrams)
#     return float(distinct_score)


# def get_content_words(text):
#     """Extract content words (non-stopwords) from text.
    
#     Filters out stopwords and punctuation, keeping only meaningful content words.
#     """
#     try:
#         import nltk
#         try:
#             from nltk.corpus import stopwords
#             stop_words = set(stopwords.words('english'))
#         except LookupError:
#             # Download stopwords if not available
#             nltk.download('stopwords', quiet=True)
#             from nltk.corpus import stopwords
#             stop_words = set(stopwords.words('english'))
#     except ImportError:
#         # Fallback: basic English stopwords
#         stop_words = {
#             'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
#             'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
#             'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
#             'had', 'what', 'said', 'each', 'which', 'their', 'if', 'up', 'out',
#             'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make',
#             'like', 'into', 'him', 'has', 'two', 'more', 'very', 'after', 'words',
#             'long', 'than', 'first', 'been', 'call', 'who', 'oil', 'sit', 'now',
#             'find', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
#         }
    
#     # Tokenize and filter
#     tokens = text.lower().split()
#     # Remove punctuation and stopwords
#     content_words = []
#     for token in tokens:
#         # Remove punctuation
#         token = re.sub(r'[^\w\s]', '', token)
#         if token and len(token) > 1 and token not in stop_words:
#             content_words.append(token)
    
#     return content_words


# def content_wdistinct_n(texts, n=3):
#     """Compute IDF-weighted Distinct-n score on content words only.
    
#     Process:
#     1. Remove templates, stopwords from each text
#     2. Keep only content words (nouns, verbs, adjectives)
#     3. Extract n-grams from the content word sequence
#     4. Compute IDF-weighted Distinct-n on these n-grams
    
#     This focuses on semantic diversity by measuring diversity of meaningful content
#     rather than all words including stopwords and templates.
#     """
#     if len(texts) == 0:
#         return float("nan")
    
#     # Step 1: Extract content words from each text and extract n-grams
#     all_ngrams_list = []
#     ngrams_per_text = []
    
#     for text in texts:
#         # Extract content words (removes stopwords, templates, keeps nouns/verbs/adjectives)
#         content_words = get_content_words(text)
#         if len(content_words) < n:
#             ngrams_per_text.append([])
#             continue
        
#         # Extract n-grams from content word sequence
#         text_ngrams = []
#         for i in range(len(content_words) - n + 1):
#             ngram = tuple(content_words[i:i+n])
#             text_ngrams.append(ngram)
#             all_ngrams_list.append(ngram)
#         ngrams_per_text.append(text_ngrams)
    
#     if len(all_ngrams_list) == 0:
#         return float("nan")
    
#     # Step 2: Compute IDF for each n-gram
#     # IDF(g) = log(N / (1 + df(g))) where N is number of texts, df is document frequency
#     ngram_to_doc_freq = {}
#     for text_ngrams in ngrams_per_text:
#         unique_ngrams_in_text = set(text_ngrams)
#         for ngram in unique_ngrams_in_text:
#             ngram_to_doc_freq[ngram] = ngram_to_doc_freq.get(ngram, 0) + 1
    
#     N = len([ngrams for ngrams in ngrams_per_text if len(ngrams) > 0])
#     if N == 0:
#         return float("nan")
    
#     # Step 3: Compute IDF for each n-gram
#     ngram_to_idf = {}
#     for ngram, df in ngram_to_doc_freq.items():
#         idf = np.log(N / (1 + df))
#         ngram_to_idf[ngram] = idf
    
#     # Step 4: Compute weighted distinct score
#     unique_ngrams = set(all_ngrams_list)
    
#     # Sum IDF for unique n-grams
#     sum_idf_unique = sum(ngram_to_idf.get(ngram, 0) for ngram in unique_ngrams)
    
#     # Sum IDF for all n-grams
#     sum_idf_all = sum(ngram_to_idf.get(ngram, 0) for ngram in all_ngrams_list)
    
#     if sum_idf_all == 0:
#         return float("nan")
    
#     weighted_distinct = sum_idf_unique / sum_idf_all
#     return float(weighted_distinct)


# def load_existing_results(csv_path):
#     """Load existing results from CSV to skip already processed datasets.
    
#     Returns a set of datasets that have all required metrics computed.
#     """
#     processed = set()
    
#     if csv_path.exists():
#         try:
#             with csv_path.open("r", encoding="utf-8") as f:
#                 reader = csv.DictReader(f)
#                 for row in reader:
#                     dataset = row["dataset"]
#                     # Check for content_only_wdistinct_3 field
#                     has_content_wdistinct = (
#                         "content_only_wdistinct_3" in row and row["content_only_wdistinct_3"] and 
#                         row["content_only_wdistinct_3"].strip() and row["content_only_wdistinct_3"] != "nan"
#                     )
#                     has_all_fields = (
#                         "vendi_score" in row and row["vendi_score"] and row["vendi_score"].strip() and row["vendi_score"] != "nan" and
#                         "order_phi" in row and row["order_phi"] and row["order_phi"].strip() and row["order_phi"] != "nan" and
#                         "pcd" in row and row["pcd"] and row["pcd"].strip() and row["pcd"] != "nan" and
#                         has_content_wdistinct
#                     )
#                     if has_all_fields:
#                         processed.add(dataset)
#                     else:
#                         print(f"  Dataset {dataset} exists but missing required metrics, will recompute")
#         except Exception as e:
#             print(f"Warning: Could not read existing CSV: {e}")
#     return processed


# def main():
#     datasets = [p for p in BASE_ROOT.iterdir() if p.is_dir()]
#     out_path = OUT_DIR / "metrics_vendi_order.csv"
    
#     # Load existing results to skip already processed datasets
#     processed_datasets = load_existing_results(out_path)
    
#     # Load existing rows from CSV
#     # We'll update rows that need recomputation and keep rows that are complete
#     rows = []
#     rows_to_update = {}  # Map dataset name to (row_index, existing_row_data)
#     if out_path.exists():
#         try:
#             with out_path.open("r", encoding="utf-8") as f:
#                 reader = csv.DictReader(f)
#                 rows = list(reader)
#                 # Track which rows need updating and what fields they already have
#                 required_fields = ["vendi_score", "order_phi", "pcd", "content_only_wdistinct_3"]
#                 for i, row in enumerate(rows):
#                     dataset = row["dataset"]
#                     has_all_fields = all(
#                         field in row and row[field] and row[field].strip() and row[field] != "nan"
#                         for field in required_fields
#                     )
#                     if not has_all_fields:
#                         rows_to_update[dataset] = (i, row.copy())
#         except Exception as e:
#             print(f"Warning: Could not read existing CSV: {e}")
#             rows = []

#     for dpath in sorted(datasets, key=lambda p: p.name):
#         # Skip if already processed
#         if dpath.name in processed_datasets:
#             print(f"Skipping {dpath.name} (already processed)")
#             continue
            
#         print(f"Processing {dpath.name} ...")
#         print(f"  Loading dataset...")
#         proposals = load_dataset(dpath)
#         if not proposals:
#             print("  No proposals, skip.")
#             continue
#         texts = [t for _, t in proposals]
#         n = len(texts)
#         print(f"  Loaded {n} proposals")
        
#         # Note: Order Parameter and PCD are averages (sample-size independent)
#         # Vendi Score and Self-BLEU are sample-size dependent
#         if n < 2:
#             print(f"  Need at least 2 samples for metrics, skip.")
#             continue
        
#         # Get embeddings using text-embedding-3-large (unified for all metrics)
#         # Cache file location: in the dataset directory
#         cache_file = dpath / "embeddings_cache_v3large.pkl"
#         print(f"  Starting to get embeddings (cache: {cache_file.name})...")
#         try:
#             emb = get_openai_embeddings_cached(texts, cache_file, model="text-embedding-3-large")
#             model_name = "text-embedding-3-large"
#             print(f"  Got embeddings: {emb.shape[0]} texts, {emb.shape[1]} dimensions")
#         except (ValueError, RuntimeError) as e:
#             print(f"  Failed to get embeddings: {e}")
#             print(f"  Note: All metrics require a valid OpenAI API key. Set OPENAI_API_KEY environment variable.")
#             continue
        
#         # Compute all metrics using the same embeddings
#         print(f"  Computing Vendi Score...")
#         try:
#             v_score = vendi_score(emb)
#         except Exception as e:
#             print(f"  Failed to compute Vendi Score: {e}")
#             v_score = float("nan")
        
#         print(f"  Computing Order Parameter...")
#         phi = order_parameter(emb)
        
#         print(f"  Computing PCD...")
#         try:
#             pcd = pairwise_cosine_distance(emb)
#         except Exception as e:
#             print(f"  Failed to compute PCD: {e}")
#             pcd = float("nan")
        
#         # Compute Content-only WDistinct-3 (on content words only)
#         print(f"  Computing Content-only WDistinct-3...")
#         try:
#             content_wdistinct_3 = content_wdistinct_n(texts, n=3)
#         except Exception as e:
#             print(f"  Failed to compute Content-only WDistinct-3: {e}")
#             content_wdistinct_3 = float("nan")

#         print(
#             f"  n={n}, model={model_name}, "
#             f"Vendi={v_score:.3f}, "
#             f"Order(phi)={phi:.3f}, "
#             f"PCD={pcd:.3f}, "
#             f"Content-only WDistinct-3={content_wdistinct_3:.3f}"
#         )
#         print(f"  Note: For diversity, higher is better for Vendi, PCD, Content-only WDistinct-3.")
#         print(f"        Lower is better for Order(phi) (use 1 - Order for visualization).")

#         # Add new result (convert to string for CSV compatibility)
#         new_row = {
#             "dataset": dpath.name,
#             "n_samples": str(n),
#             "embedding": model_name,
#             "vendi_score": str(v_score) if not math.isnan(v_score) else "nan",
#             "order_phi": str(phi) if not math.isnan(phi) else "nan",
#             "pcd": str(pcd) if not math.isnan(pcd) else "nan",
#             "content_only_wdistinct_3": str(content_wdistinct_3) if not math.isnan(content_wdistinct_3) else "nan",
#         }
        
#         # Update existing row or append new one
#         if dpath.name in rows_to_update:
#             # Update existing row with new metrics
#             idx, existing_row = rows_to_update[dpath.name]
#             # Remove old field names if they exist (for backward compatibility)
#             for old_field in ["keyphrase_wdistinct_3", "kp_wdistinct_3", "content_wdistinct_3"]:
#                 if old_field in rows[idx]:
#                     del rows[idx][old_field]
#             # Update all fields
#             for key, value in new_row.items():
#                 rows[idx][key] = value
#             print(f"  Updated existing row for {dpath.name}")
#         else:
#             # Append new row
#             rows.append(new_row)

#     # Write all results to CSV
#     if rows:
#         # Clean up rows: remove old field names and keep only valid fields
#         fieldnames = [
#             "dataset",
#             "n_samples",
#             "embedding",
#             "vendi_score",
#             "order_phi",
#             "pcd",
#                     "content_only_wdistinct_3",
#         ]
#         cleaned_rows = []
#         for row in rows:
#             cleaned_row = {}
#             # Only keep fields that are in fieldnames
#             for field in fieldnames:
#                 cleaned_row[field] = row.get(field, "nan")
#             cleaned_rows.append(cleaned_row)
        
#         with out_path.open("w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(cleaned_rows)
#         print(f"Wrote {out_path}")
#     else:
#         print("No datasets processed; nothing written.")


# if __name__ == "__main__":
#     main()
