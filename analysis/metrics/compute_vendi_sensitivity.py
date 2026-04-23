import os
import math
import csv
import time
import pickle
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


BASE_ROOT = Path("./data/extracted_proposals")
OUT_DIR = Path("./data/tsne")
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


def get_embeddings_cached(texts, cache_file, model="bge-large-en-v1.5"):
    """Return L2-normalized embeddings using specified model with caching.
    Supports both OpenAI API models and local models (bge-large-en-v1.5 via sentence-transformers).
    """
    if cache_file.exists():
        print(f"    [Cache Hit] Loading embeddings from {cache_file.name}...")
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, dict):
                cached_texts_hash = cached_data.get("texts_hash")
                cached_model = cached_data.get("model")
                cached_emb = cached_data.get("embeddings")
                current_texts_hash = hash(tuple(texts))
                if (cached_texts_hash == current_texts_hash and 
                    cached_model == model and 
                    cached_emb is not None):
                    if len(cached_emb) == len(texts):
                        print(f"    [Cache Hit] Loaded {len(texts)} embeddings from cache")
                        return np.array(cached_emb)
            elif isinstance(cached_data, (list, np.ndarray)):
                # Old format: just embeddings array (check length only)
                if len(cached_data) == len(texts):
                    print(f"    [Cache Hit] Loaded {len(texts)} embeddings from cache (old format)")
                    return np.array(cached_data)
        except Exception as e:
            print(f"    [Cache Miss] Failed to load cache: {e}")

    print(f"    [Computing] Generating embeddings for {len(texts)} texts using {model}...")
    
    if model == "bge-large-en-v1.5":
        # Use sentence-transformers for BGE model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not found. Please install it with: pip install sentence-transformers")
        
        print(f"    Loading {model} model (this may take a moment on first run)...")
        encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_emb = encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(batch_emb)
            if (i + batch_size) % 100 == 0 or i + batch_size >= total:
                print(f"    Processed: {min(i + batch_size, total)}/{total} texts")
        
        emb = np.vstack(all_embeddings).astype(np.float32)
        
    else:
        # Use OpenAI API for other models
        from openai import OpenAI
        
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
                print(f"    Warning: Text {i+1} has {text_tokens} tokens, exceeding API limit. Skipping.")
                dummy_emb = np.zeros((1, 3072), dtype=np.float32)
                all_embeddings.append(dummy_emb)
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

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {"texts_hash": hash(tuple(texts)), "model": model, "embeddings": emb.tolist()}
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
    Supports n=2, 3, 4.
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
    """Compute raw token distinct-n (unweighted, no content filtering)."""
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
    """Check which datasets are already fully processed."""
    processed = set()

    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset = row["dataset"]
                    # Check if this row has all required fields
                    required_fields = [
                        "vendi_score_openai", "order_phi_openai", "pcd_openai",
                        "vendi_score_bge", "order_phi_bge", "pcd_bge",
                        "raw_distinct_3",
                        "content_wdistinct_2", "content_wdistinct_3", "content_wdistinct_4"
                    ]
                    has_all_fields = all(
                        field in row and row[field] and row[field].strip() and row[field] != "nan"
                        for field in required_fields
                    )
                    if has_all_fields:
                        processed.add(dataset)
                    else:
                        print(f"  Dataset {dataset} exists but missing required metrics, will recompute")
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
    return processed


def compute_spearman_correlations_for_model(rows, model_suffix="openai"):
    """Compute Spearman correlations between PCD, Vendi, and 1-phi for a specific model."""
    # Extract valid values
    pcd_values = []
    vendi_values = []
    one_minus_phi_values = []
    dataset_names = []
    
    pcd_key = f"pcd_{model_suffix}"
    vendi_key = f"vendi_score_{model_suffix}"
    phi_key = f"order_phi_{model_suffix}"
    
    for row in rows:
        try:
            pcd = float(row.get(pcd_key, "nan"))
            vendi = float(row.get(vendi_key, "nan"))
            phi = float(row.get(phi_key, "nan"))
            
            if not (math.isnan(pcd) or math.isnan(vendi) or math.isnan(phi)):
                pcd_values.append(pcd)
                vendi_values.append(vendi)
                one_minus_phi_values.append(1.0 - phi)
                dataset_names.append(row.get("dataset", "unknown"))
        except (ValueError, TypeError):
            continue
    
    print(f"  Found {len(pcd_values)} valid data points for {model_suffix} correlation analysis")
    
    if len(pcd_values) < 3:
        print(f"  Warning: Not enough valid data points for Spearman correlation (need at least 3, got {len(pcd_values)})")
        return {}
    
    # Compute rankings
    pcd_ranks = np.argsort(np.argsort(pcd_values))
    vendi_ranks = np.argsort(np.argsort(vendi_values))
    phi_ranks = np.argsort(np.argsort(one_minus_phi_values))
    
    # Check if ranks are identical (which would cause ρ=1.0)
    ranks_identical_pcd_vendi = np.array_equal(pcd_ranks, vendi_ranks)
    ranks_identical_pcd_phi = np.array_equal(pcd_ranks, phi_ranks)
    ranks_identical_vendi_phi = np.array_equal(vendi_ranks, phi_ranks)
    
    if ranks_identical_pcd_vendi or ranks_identical_pcd_phi or ranks_identical_vendi_phi:
        print(f"  ⚠️  Note: Some metrics have identical rankings across all {len(pcd_values)} datasets")
        print(f"     This results in perfect Spearman correlation (ρ=1.0) even if values differ.")
        if ranks_identical_pcd_vendi:
            print(f"     → PCD and Vendi rankings are identical")
        if ranks_identical_pcd_phi:
            print(f"     → PCD and (1-φ) rankings are identical")
        if ranks_identical_vendi_phi:
            print(f"     → Vendi and (1-φ) rankings are identical")
        print(f"     (This is expected if metrics measure similar aspects of diversity)")
    
    correlations = {}
    
    # PCD vs Vendi
    try:
        corr, pval = spearmanr(pcd_values, vendi_values)
        correlations["pcd_vendi"] = {"correlation": corr, "pvalue": pval}
    except Exception as e:
        print(f"  Error computing PCD vs Vendi correlation: {e}")
        correlations["pcd_vendi"] = {"correlation": float("nan"), "pvalue": float("nan")}
    
    # PCD vs (1-phi)
    try:
        corr, pval = spearmanr(pcd_values, one_minus_phi_values)
        correlations["pcd_one_minus_phi"] = {"correlation": corr, "pvalue": pval}
    except Exception as e:
        print(f"  Error computing PCD vs (1-phi) correlation: {e}")
        correlations["pcd_one_minus_phi"] = {"correlation": float("nan"), "pvalue": float("nan")}
    
    # Vendi vs (1-phi)
    try:
        corr, pval = spearmanr(vendi_values, one_minus_phi_values)
        correlations["vendi_one_minus_phi"] = {"correlation": corr, "pvalue": pval}
    except Exception as e:
        print(f"  Error computing Vendi vs (1-phi) correlation: {e}")
        correlations["vendi_one_minus_phi"] = {"correlation": float("nan"), "pvalue": float("nan")}
    
    return correlations


def compute_spearman_correlations(rows):
    """Compute Spearman correlations between PCD, Vendi, and 1-phi."""
    # Extract valid values
    pcd_values = []
    vendi_values = []
    one_minus_phi_values = []
    dataset_names = []
    
    for row in rows:
        try:
            pcd = float(row.get("pcd", "nan"))
            vendi = float(row.get("vendi_score", "nan"))
            phi = float(row.get("order_phi", "nan"))
            
            if not (math.isnan(pcd) or math.isnan(vendi) or math.isnan(phi)):
                pcd_values.append(pcd)
                vendi_values.append(vendi)
                one_minus_phi_values.append(1.0 - phi)
                dataset_names.append(row.get("dataset", "unknown"))
        except (ValueError, TypeError):
            continue
    
    print(f"  Found {len(pcd_values)} valid data points for correlation analysis")
    
    if len(pcd_values) < 3:
        print(f"  Warning: Not enough valid data points for Spearman correlation (need at least 3, got {len(pcd_values)})")
        if len(pcd_values) > 0:
            print(f"  Sample values:")
            print(f"    PCD: {pcd_values[:min(5, len(pcd_values))]}")
            print(f"    Vendi: {vendi_values[:min(5, len(vendi_values))]}")
            print(f"    1-φ: {one_minus_phi_values[:min(5, len(one_minus_phi_values))]}")
        return {}
    
    # Compute rankings
    pcd_ranks = np.argsort(np.argsort(pcd_values))
    vendi_ranks = np.argsort(np.argsort(vendi_values))
    phi_ranks = np.argsort(np.argsort(one_minus_phi_values))
    
    # Check if ranks are identical (which would cause ρ=1.0)
    ranks_identical_pcd_vendi = np.array_equal(pcd_ranks, vendi_ranks)
    ranks_identical_pcd_phi = np.array_equal(pcd_ranks, phi_ranks)
    ranks_identical_vendi_phi = np.array_equal(vendi_ranks, phi_ranks)
    
    if ranks_identical_pcd_vendi or ranks_identical_pcd_phi or ranks_identical_vendi_phi:
        print(f"  ⚠️  Note: Some metrics have identical rankings across all {len(pcd_values)} datasets")
        print(f"     This results in perfect Spearman correlation (ρ=1.0) even if values differ.")
        if ranks_identical_pcd_vendi:
            print(f"     → PCD and Vendi rankings are identical")
        if ranks_identical_pcd_phi:
            print(f"     → PCD and (1-φ) rankings are identical")
        if ranks_identical_vendi_phi:
            print(f"     → Vendi and (1-φ) rankings are identical")
        print(f"     (This is expected if metrics measure similar aspects of diversity)")
    
    correlations = {}
    
    # PCD vs Vendi
    try:
        corr, pval = spearmanr(pcd_values, vendi_values)
        correlations["pcd_vendi"] = {"correlation": corr, "pvalue": pval}
    except Exception as e:
        print(f"  Error computing PCD vs Vendi correlation: {e}")
        correlations["pcd_vendi"] = {"correlation": float("nan"), "pvalue": float("nan")}
    
    # PCD vs (1-phi)
    try:
        corr, pval = spearmanr(pcd_values, one_minus_phi_values)
        correlations["pcd_one_minus_phi"] = {"correlation": corr, "pvalue": pval}
    except Exception as e:
        print(f"  Error computing PCD vs (1-phi) correlation: {e}")
        correlations["pcd_one_minus_phi"] = {"correlation": float("nan"), "pvalue": float("nan")}
    
    # Vendi vs (1-phi)
    try:
        corr, pval = spearmanr(vendi_values, one_minus_phi_values)
        correlations["vendi_one_minus_phi"] = {"correlation": corr, "pvalue": pval}
    except Exception as e:
        print(f"  Error computing Vendi vs (1-phi) correlation: {e}")
        correlations["vendi_one_minus_phi"] = {"correlation": float("nan"), "pvalue": float("nan")}
    
    return correlations


def main():
    datasets = [p for p in BASE_ROOT.iterdir() if p.is_dir()]
    out_path = OUT_DIR / "metrics_vendi_order_sensitivity.csv"

    processed_datasets = load_existing_results(out_path)

    # Load existing rows
    rows = []
    rows_to_update = {}
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                required_fields = [
                    "vendi_score_openai", "order_phi_openai", "pcd_openai",
                    "vendi_score_bge", "order_phi_bge", "pcd_bge",
                    "raw_distinct_3",
                    "content_wdistinct_2", "content_wdistinct_3", "content_wdistinct_4"
                ]
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

    # Build GLOBAL IDF for content-only wdistinct-2, 3, 4
    print("Building GLOBAL content-only IDF for WDistinct-2,3,4 over ALL datasets ...")
    global_idf_2, global_N_2 = build_global_content_ngram_idf(datasets, n=2)
    global_idf_3, global_N_3 = build_global_content_ngram_idf(datasets, n=3)
    global_idf_4, global_N_4 = build_global_content_ngram_idf(datasets, n=4)
    print(f"  Global IDF-2: {len(global_idf_2)} unique 2-grams over {global_N_2} valid documents")
    print(f"  Global IDF-3: {len(global_idf_3)} unique 3-grams over {global_N_3} valid documents")
    print(f"  Global IDF-4: {len(global_idf_4)} unique 4-grams over {global_N_4} valid documents")

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

        # Get embeddings with text-embedding-3-large
        model_openai = "text-embedding-3-large"
        cache_file_openai = dpath / "embeddings_cache_v3large.pkl"
        print(f"  Getting embeddings with {model_openai} (cache: {cache_file_openai.name})...")
        try:
            emb_openai = get_embeddings_cached(texts, cache_file_openai, model=model_openai)
            print(f"  Got {model_openai} embeddings: {emb_openai.shape[0]} texts, {emb_openai.shape[1]} dimensions")
        except (ValueError, RuntimeError, ImportError) as e:
            print(f"  Failed to get {model_openai} embeddings: {e}")
            emb_openai = None

        # Get embeddings with bge-large-en-v1.5
        model_bge = "bge-large-en-v1.5"
        cache_file_bge = dpath / f"embeddings_cache_{model_bge.replace('-', '_').replace('.', '_')}.pkl"
        print(f"  Getting embeddings with {model_bge} (cache: {cache_file_bge.name})...")
        try:
            emb_bge = get_embeddings_cached(texts, cache_file_bge, model=model_bge)
            print(f"  Got {model_bge} embeddings: {emb_bge.shape[0]} texts, {emb_bge.shape[1]} dimensions")
        except (ValueError, RuntimeError, ImportError) as e:
            print(f"  Failed to get {model_bge} embeddings: {e}")
            emb_bge = None

        # Compute metrics with text-embedding-3-large
        v_score_openai = float("nan")
        phi_openai = float("nan")
        pcd_openai = float("nan")
        if emb_openai is not None:
            print(f"  Computing metrics with {model_openai}...")
            try:
                v_score_openai = vendi_score(emb_openai)
            except Exception as e:
                print(f"    Failed to compute Vendi Score: {e}")
            try:
                phi_openai = order_parameter(emb_openai)
            except Exception as e:
                print(f"    Failed to compute Order Parameter: {e}")
            try:
                pcd_openai = pairwise_cosine_distance(emb_openai)
            except Exception as e:
                print(f"    Failed to compute PCD: {e}")

        # Compute metrics with bge-large-en-v1.5
        v_score_bge = float("nan")
        phi_bge = float("nan")
        pcd_bge = float("nan")
        if emb_bge is not None:
            print(f"  Computing metrics with {model_bge}...")
            try:
                v_score_bge = vendi_score(emb_bge)
            except Exception as e:
                print(f"    Failed to compute Vendi Score: {e}")
            try:
                phi_bge = order_parameter(emb_bge)
            except Exception as e:
                print(f"    Failed to compute Order Parameter: {e}")
            try:
                pcd_bge = pairwise_cosine_distance(emb_bge)
            except Exception as e:
                print(f"    Failed to compute PCD: {e}")

        # Raw token distinct-3 (unweighted)
        print("  Computing Raw Distinct-3 (unweighted)...")
        raw_distinct_3 = distinct_n(texts, n=3)

        # Content-only wdistinct-2, 3, 4 (weighted with global IDF)
        print("  Computing Content-only WDistinct-2,3,4 (GLOBAL IDF)...")
        content_wdistinct_2 = content_only_wdistinct_n(texts, global_idf_2, n=2)
        content_wdistinct_3 = content_only_wdistinct_n(texts, global_idf_3, n=3)
        content_wdistinct_4 = content_only_wdistinct_n(texts, global_idf_4, n=4)

        print(
            f"  n={n}, "
            f"OpenAI: Vendi={v_score_openai:.3f}, Order={phi_openai:.3f}, PCD={pcd_openai:.3f}, "
            f"BGE: Vendi={v_score_bge:.3f}, Order={phi_bge:.3f}, PCD={pcd_bge:.3f}, "
            f"Raw Distinct-3={raw_distinct_3:.3f}, "
            f"Content WDistinct-2={content_wdistinct_2:.3f}, "
            f"Content WDistinct-3={content_wdistinct_3:.3f}, "
            f"Content WDistinct-4={content_wdistinct_4:.3f}"
        )

        new_row = {
            "dataset": dpath.name,
            "n_samples": str(n),
            "vendi_score_openai": str(v_score_openai) if not math.isnan(v_score_openai) else "nan",
            "order_phi_openai": str(phi_openai) if not math.isnan(phi_openai) else "nan",
            "pcd_openai": str(pcd_openai) if not math.isnan(pcd_openai) else "nan",
            "vendi_score_bge": str(v_score_bge) if not math.isnan(v_score_bge) else "nan",
            "order_phi_bge": str(phi_bge) if not math.isnan(phi_bge) else "nan",
            "pcd_bge": str(pcd_bge) if not math.isnan(pcd_bge) else "nan",
            "raw_distinct_3": str(raw_distinct_3) if not math.isnan(raw_distinct_3) else "nan",
            "content_wdistinct_2": str(content_wdistinct_2) if not math.isnan(content_wdistinct_2) else "nan",
            "content_wdistinct_3": str(content_wdistinct_3) if not math.isnan(content_wdistinct_3) else "nan",
            "content_wdistinct_4": str(content_wdistinct_4) if not math.isnan(content_wdistinct_4) else "nan",
        }

        if dpath.name in rows_to_update:
            idx, _existing_row = rows_to_update[dpath.name]
            for key, value in new_row.items():
                rows[idx][key] = value
            print(f"  Updated existing row for {dpath.name}")
        else:
            rows.append(new_row)

    if rows:
        fieldnames = [
            "dataset",
            "n_samples",
            "vendi_score_openai",
            "order_phi_openai",
            "pcd_openai",
            "vendi_score_bge",
            "order_phi_bge",
            "pcd_bge",
            "raw_distinct_3",
            "content_wdistinct_2",
            "content_wdistinct_3",
            "content_wdistinct_4",
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
        
        # Compute Spearman correlations for OpenAI model
        print("\nComputing Spearman correlations for OpenAI embedding-based metrics...")
        correlations_openai = compute_spearman_correlations_for_model(cleaned_rows, "openai")
        if correlations_openai:
            print("\nSpearman Correlation Results (OpenAI):")
            print(f"  PCD vs Vendi: ρ = {correlations_openai['pcd_vendi']['correlation']:.4f}, p = {correlations_openai['pcd_vendi']['pvalue']:.4f}")
            print(f"  PCD vs (1-φ): ρ = {correlations_openai['pcd_one_minus_phi']['correlation']:.4f}, p = {correlations_openai['pcd_one_minus_phi']['pvalue']:.4f}")
            print(f"  Vendi vs (1-φ): ρ = {correlations_openai['vendi_one_minus_phi']['correlation']:.4f}, p = {correlations_openai['vendi_one_minus_phi']['pvalue']:.4f}")
        
        # Compute Spearman correlations for BGE model
        print("\nComputing Spearman correlations for BGE embedding-based metrics...")
        correlations_bge = compute_spearman_correlations_for_model(cleaned_rows, "bge")
        if correlations_bge:
            print("\nSpearman Correlation Results (BGE):")
            print(f"  PCD vs Vendi: ρ = {correlations_bge['pcd_vendi']['correlation']:.4f}, p = {correlations_bge['pcd_vendi']['pvalue']:.4f}")
            print(f"  PCD vs (1-φ): ρ = {correlations_bge['pcd_one_minus_phi']['correlation']:.4f}, p = {correlations_bge['pcd_one_minus_phi']['pvalue']:.4f}")
            print(f"  Vendi vs (1-φ): ρ = {correlations_bge['vendi_one_minus_phi']['correlation']:.4f}, p = {correlations_bge['vendi_one_minus_phi']['pvalue']:.4f}")
    else:
        print("No datasets processed; nothing written.")


if __name__ == "__main__":
    main()
