import os
import matplotlib
import re
import sys
import pickle
import warnings
from pathlib import Path

# Set backend to Agg for non-interactive environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Suppress UMAP warnings
warnings.filterwarnings("ignore")

# --- OpenAI API Setup ---
try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library not installed.")
    sys.exit(1)

# 请替换为你的 Key
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- CONFIGURATION ---
LEGEND_NAMES = {
    "leader": "Leader-Led",
    "x": "Interdisciplinary",
    "mix": "Vertical",
    "young": "Horizontal",
    "rec": "Naive",
}

# 保持与密度图一致的配色
COLOR_MAP = {
    "Leader-Led": "#d62728",      # Red
    "Interdisciplinary": "#9467bd", # Purple
    "Vertical": "#2ca02c",        # Green
    "Horizontal": "#17becf",      # Cyan
    "Naive": "#7f7f7f",           # Grey
}

# Sort order for plots
PLOT_ORDER = ["Naive", "Leader-Led", "Horizontal", "Interdisciplinary", "Vertical"]

def resolve_info(dir_name):
    short_name = dir_name.replace("ai_researcher_multi_topic_dsv3_", "").replace("_final", "")
    final_name = short_name
    
    # Exact match first
    if short_name in LEGEND_NAMES:
        final_name = LEGEND_NAMES[short_name]
    else:
        # Fuzzy match
        for key, val in LEGEND_NAMES.items():
            if key in short_name:
                final_name = val
                break
    return final_name

def get_openai_embeddings_cached(texts, cache_file, model="text-embedding-3-large", batch_size=50):
    """Fetch embeddings with local caching."""
    if cache_file.exists():
        print(f"      [Cache Hit] Loading from {cache_file.name}...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if len(cached_data) == len(texts):
                return np.array(cached_data)
        except Exception:
            pass

    embeddings = []
    total = len(texts)
    print(f"      [API Call] Fetching for {total} texts...")
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch = [t.replace("\n", " ") for t in batch]
            response = client.embeddings.create(input=batch, model=model)
            batch_embs = [data.embedding for data in response.data]
            embeddings.extend(batch_embs)
        except Exception as e:
            print(f"\nError: {e}")
            raise e
            
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
    except Exception:
        pass

    return np.array(embeddings)

def main():
    base_root = Path("./data/extracted_proposals")
    out_root = Path("./data/tsne")
    out_root.mkdir(parents=True, exist_ok=True)

    subdirs = [
        p for p in base_root.iterdir()
        if p.is_dir() and p.name.startswith("ai_researcher_multi_topic_dsv3_")
    ]
    if not subdirs:
        raise SystemExit("No targets to process")

    try:
        import umap
    except ImportError:
        print("Error: umap-learn not installed. pip install umap-learn")
        return

    # --- 1. DATA COLLECTION ---
    all_embeddings = []
    all_labels = []
    all_topics = []
    
    print("Loading data and embeddings...")
    for subdir in sorted(subdirs):
        label = resolve_info(subdir.name)
        
        # Load Proposals
        proposals = []
        for fp in sorted(subdir.glob("*_proposals.txt")):
            topic = fp.name.replace("_proposals.txt", "")
            try:
                ns: dict = {}
                with fp.open("r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                content = re.sub(r'\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])', r'?', content)
                code = compile(content, str(fp), "exec")
                exec(code, {}, ns)
                papers = ns.get("paper_txts", [])
            except Exception:
                continue
            for p in papers:
                if p.strip():
                    proposals.append((topic, p.strip()))

        if not proposals:
            continue

        texts = [t for _, t in proposals]
        topics = [topic for topic, _ in proposals]
        
        # Load Embeddings
        cache_file = subdir / "embeddings_cache_v3large.pkl"
        try:
            emb = get_openai_embeddings_cached(texts, cache_file)
            
            all_embeddings.append(emb)
            all_labels.extend([label] * len(texts))
            all_topics.extend(topics)
            print(f"  Loaded {len(texts)} samples for {label}")
        except Exception as e:
            print(f"  Skipping {subdir.name}: {e}")

    if not all_embeddings:
        return

    X = np.vstack(all_embeddings)
    y_labels = np.array(all_labels)
    y_topics = np.array(all_topics)

    print(f"Total samples: {X.shape}")

    # --- 2. PREPROCESSING: PER-TOPIC WHITENING (StandardScaler) ---
    # This is crucial! It removes the "topic cluster" effect and leaves only 
    # the "structural spread" effect.
    print("Applying Per-Topic Whitening...")
    X_whitened = np.zeros_like(X)
    unique_topics = np.unique(y_topics)
    
    for topic in unique_topics:
        mask = y_topics == topic
        if np.sum(mask) > 1:
            scaler = StandardScaler()
            # Fit transform on this topic's data across all structures
            X_whitened[mask] = scaler.fit_transform(X[mask])
        else:
            X_whitened[mask] = 0 # Handle singletons

    # --- 3. RUN UMAP ---
    # Run once on the entire dataset so everyone is in the same space
    print("Running UMAP on full dataset...")
    reducer = umap.UMAP(
        n_neighbors=30,      # Slightly higher for global structure
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42      # Reproducibility
    )
    embedding_2d = reducer.fit_transform(X_whitened)

    # --- 4. PLOTTING: SMALL MULTIPLES ---
    print("Generating Faceted Plots...")
    
    # Setup 1x5 plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharex=True, sharey=True)
    
    # Background points (All data in light grey)
    bg_color = "#e0e0e0"
    
    for i, target_label in enumerate(PLOT_ORDER):
        if i >= len(axes): break
        ax = axes[i]
        
        # Plot Background (Context)
        ax.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1], 
            c=bg_color, 
            s=5, 
            alpha=0.3,
            edgecolors='none',
            label='_nolegend_'
        )
        
        # Plot Foreground (Target Structure)
        mask = y_labels == target_label
        target_color = COLOR_MAP.get(target_label, 'blue')
        
        ax.scatter(
            embedding_2d[mask, 0], 
            embedding_2d[mask, 1], 
            c=target_color, 
            s=15, 
            alpha=0.7,
            edgecolors='none',
            label=target_label
        )
        
        # Styling
        ax.set_title(target_label, fontsize=12, fontweight='bold', color='#333333')
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove box/spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Optional: Add centroid of the foreground
        if np.sum(mask) > 0:
            centroid = np.median(embedding_2d[mask], axis=0)
            ax.plot(centroid[0], centroid[1], '+', color='black', markersize=15, markeredgewidth=2)

    plt.suptitle("Topological Fingerprints of Collaborative Structures (UMAP)", fontsize=14, y=1.05)
    plt.tight_layout()
    
    out_path = out_root / "umap_faceted_structures.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()