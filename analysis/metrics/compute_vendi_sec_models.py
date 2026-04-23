import math
import csv
import time
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from compute_vendi_extended import (
    safe_exec,
    get_openai_embeddings_cached,
    vendi_score,
    order_parameter,
    pairwise_cosine_distance,
)

# Define the directories to process
DATASETS = [
    Path("./data/sec_models/extracted_proposals_dsv3"),
    Path("./data/sec_models/extracted_proposals_gpt51"),
    Path("./data/sec_models/extracted_proposals_grok4"),
    Path("./data/sec_models/extracted_proposals_o1mini"),
]

# Output directory
OUT_DIR = Path("./data/sec_modelsnew")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(dpath: Path):
    """Load all proposals from all topic files in a dataset.
    
    Returns a list of (topic, text) tuples for all proposals.
    This matches the logic in compute_vendi_and_order_includeselfbleunew.py
    """
    proposals = []
    for fp in sorted(dpath.glob("*_proposals.txt")):
        topic = fp.name.replace("_proposals.txt", "")
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            ns = safe_exec(content)
            papers = ns.get("paper_txts", [])
        except Exception as e:
            print(f"  Failed to parse {fp}: {e}")
            continue
        for p in papers:
            text = p.strip()
            if text:
                proposals.append((topic, text))
    return proposals


def load_existing_results(csv_path):
    """Load existing results from CSV to skip already processed datasets."""
    processed = set()
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Use dataset name as key (aggregated format, no topic column)
                    dataset = row.get('dataset', '')
                    if dataset:
                        processed.add(dataset)
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
    return processed


def main(no_cache=False):
    out_path = OUT_DIR / "metrics_vendi_order.csv"
    
    # Load existing results to skip already processed datasets
    if no_cache:
        print("Cache disabled: will recompute all datasets")
        processed_datasets = set()
        # Load existing rows but will update them instead of skipping
        existing_rows_dict = {}
        if out_path.exists():
            try:
                with out_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dataset = row.get('dataset', '')
                        if dataset:
                            existing_rows_dict[dataset] = row
            except Exception as e:
                print(f"Warning: Could not read existing CSV: {e}")
                existing_rows_dict = {}
    else:
        processed_datasets = load_existing_results(out_path)
        existing_rows_dict = {}
    
    # Load existing rows from CSV (for no_cache mode, we'll update them)
    rows = []
    if not no_cache and out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
            rows = []

    for dpath in DATASETS:
        # Get dataset name from path
        dataset_name = dpath.name
        
        if not dpath.exists():
            print(f"Warning: Directory {dpath} does not exist, skipping.")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Check if already processed
        if not no_cache and dataset_name in processed_datasets:
            print(f"Skipping {dataset_name} (already processed)")
            continue
        
        # In no_cache mode, note if recomputing
        if no_cache and dataset_name in existing_rows_dict:
            print(f"Recomputing {dataset_name} (cache disabled)")
        
        # Load all proposals from all topics (aggregated)
        print("  Loading all proposals from all topics...")
        proposals = load_dataset(dpath)
        if not proposals:
            print(f"  No proposals found in {dpath}, skipping.")
            continue
        
        texts = [t for _, t in proposals]
        n = len(texts)
        print(f"  Loaded {n} proposals across all topics")
        
        if n < 2:
            print("  Need at least 2 samples for metrics, skip.")
            continue
        
        # Get embeddings using text-embedding-3-large with caching and chunking support
        cache_file = dpath / "embeddings_cache_v3large.pkl"
        print(f"  Starting to get embeddings (cache: {cache_file.name})...")
        try:
            emb = get_openai_embeddings_cached(texts, cache_file, model="text-embedding-3-large", use_cache=not no_cache)
            model_name = "text-embedding-3-large"
            print(f"  Got embeddings: {emb.shape[0]} texts, {emb.shape[1]} dimensions")
        except (ValueError, RuntimeError) as e:
            print(f"  Failed to get embeddings: {e}")
            continue
        
        # Compute all metrics using the same embeddings
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

        print(
            f"  Results: n={n}, model={model_name}, Vendi={v_score:.3f}, "
            f"Order(phi)={phi:.3f}, PCD={pcd:.3f}"
        )

        # Add new result (convert to string for CSV compatibility)
        # Note: No 'topic' column - this is aggregated by dataset
        new_row = {
            "dataset": dataset_name,
            "n_samples": str(n),
            "embedding": model_name,
            "vendi_score": str(v_score) if not math.isnan(v_score) else "nan",
            "order_phi": str(phi) if not math.isnan(phi) else "nan",
            "pcd": str(pcd) if not math.isnan(pcd) else "nan",
        }
        
        if no_cache:
            # Update existing entry or add new one
            existing_rows_dict[dataset_name] = new_row
        else:
            rows.append(new_row)

    # Write all results to CSV
    if no_cache:
        # In no_cache mode, write all rows (updated + existing from other datasets)
        # existing_rows_dict already contains all updated entries
        all_rows = list(existing_rows_dict.values())
        
        if all_rows:
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "dataset",
                        "n_samples",
                        "embedding",
                        "vendi_score",
                        "order_phi",
                        "pcd",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_rows)
            updated_count = sum(1 for dpath in DATASETS if dpath.exists())
            print(f"Wrote {out_path} (updated {updated_count} datasets, total {len(all_rows)} entries)")
        else:
            print("No datasets processed; nothing written.")
    else:
        # Normal mode: merge new rows with existing rows (avoid duplicates)
        existing_rows = []
        existing_datasets = set()
        if out_path.exists():
            try:
                with out_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_datasets.add(row.get('dataset', ''))
                        existing_rows.append(row)  # Keep existing rows
            except Exception as e:
                print(f"Warning: Could not read existing CSV: {e}")
        
        # Combine existing rows with new rows (only add new datasets)
        all_rows = existing_rows + [r for r in rows if r.get('dataset', '') not in existing_datasets]
        
        if all_rows:
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "dataset",
                        "n_samples",
                        "embedding",
                        "vendi_score",
                        "order_phi",
                        "pcd",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_rows)
            new_count = len(all_rows) - len(existing_rows)
            print(f"Wrote {out_path} ({new_count} new datasets added, total {len(all_rows)} datasets)")
        else:
            print("No datasets processed; nothing written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Vendi scores and order parameters for proposal datasets")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache: recompute all datasets even if they exist in CSV"
    )
    args = parser.parse_args()
    
    main(no_cache=args.no_cache)

