import os
import matplotlib
import re
import time
import sys

# Set backend to Agg for non-interactive environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde, kurtosis

# --- OpenAI API Setup ---
try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library not installed. Please install it!")
    print("pip install openai")
    sys.exit(1)

# 配置你的 API Key 和 Base URL
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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

def safe_exec(content: str) -> dict:
    # 防止文本里出现裸 \uXXXX 被 Python 解析为转义
    fixed = content.replace("\\u", "\\\\u")
    ns: dict = {}
    exec(fixed, {}, ns)
    return ns

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



def get_openai_embeddings(texts, model="text-embedding-3-large", batch_size=50):
    """
    Fetch embeddings from OpenAI API in batches.
    """
    embeddings = []
    total = len(texts)
    print(f"    - Fetching OpenAI embeddings for {total} texts (Batch size: {batch_size})...")
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            # Replace newlines to avoid negative effects on performance for some models
            batch = [t.replace("\n", " ") for t in batch]
            
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            # Ensure order is preserved
            batch_embs = [data.embedding for data in response.data]
            embeddings.extend(batch_embs)
            
            # Optional: simple progress indicator
            print(f"      Processed {min(i + batch_size, total)}/{total}...", end='\r')
            
        except Exception as e:
            print(f"\nError fetching embeddings: {e}")
            # Retry logic could be added here, but for now we raise
            raise e
            
    print("") # New line after progress
    return np.array(embeddings)

def cosine_distance(vec1, vec2):
    """Compute cosine distance between vectors."""
    # Ensure vectors are normalized
    norm1 = np.linalg.norm(vec1, axis=-1, keepdims=True) + 1e-12
    norm2 = np.linalg.norm(vec2, axis=-1, keepdims=True) + 1e-12
    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2
    # Cosine similarity
    cosine_sim = np.sum(vec1_norm * vec2_norm, axis=-1)
    # Cosine distance
    cosine_dist = 1.0 - cosine_sim
    # Clip to avoid numerical errors slightly < 0
    return np.clip(cosine_dist, 0.0, 2.0)

def main():
    base_root = Path("./data/extracted_proposals")
    out_root = Path("./data/tsne")
    out_root.mkdir(parents=True, exist_ok=True)

    # Filter targets
    subdirs = [
        p for p in base_root.iterdir()
        if p.is_dir() and p.name.startswith("ai_researcher_multi_topic_dsv3_")
    ]
    if not subdirs:
        raise SystemExit("No targets to process")
    
    print("Targets found:", len(subdirs))

    # --- CONFIGURATION ---
    # Legend Name Mapping
    legend_names = {
        "leader": "Leader-Led Collaboration",       # Echo Chamber (Red)
        "x": "Interdisciplinary Collaboration",     # High Value (Purple)
        "mix": "Vertical Collaboration (Ours)",     # Balanced (Green)
        "young": "Horizontal Collaboration",        # Divergent (Cyan)
        "rec": "Naive Collaboration",               # Baseline (Grey)
    }
    
    # Color Mapping
    color_map = {
        "Leader-Led Collaboration": "#d62728",      # Red
        "Interdisciplinary Collaboration": "#9467bd", # Purple
        "Vertical Collaboration (Ours)": "#2ca02c", # Green
        "Horizontal Collaboration": "#17becf",      # Cyan
        "Naive Collaboration": "#7f7f7f",           # Grey
    }

    # Helper to resolve names
    def resolve_info(dir_name):
        short_name = dir_name.replace("ai_researcher_multi_topic_dsv3_", "").replace("_final", "")
        final_name = short_name
        for key, val in legend_names.items():
            if key in short_name:
                final_name = val
                break
        return final_name, color_map.get(final_name, "black")

    # Store distances
    all_distances_dict = {}
    
    for subdir in sorted(subdirs):
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
                text = p.strip()
                if text:
                    proposals.append((topic, text))

        if not proposals:
            continue

        texts = [t for _, t in proposals]
        topics = [topic for topic, _ in proposals]
        
        print(f"Processing {subdir.name}: {len(texts)} texts...")

        # --- Compute Embeddings using OpenAI API ---
        # Note: This will consume API credits.
        try:
            embeddings = get_openai_embeddings(texts, model="text-embedding-3-large")
        except Exception as e:
            print(f"Skipping {subdir.name} due to API error: {e}")
            continue

        # Compute centroid distances
        uniq_topics = sorted(set(topics))
        distances = []
        for topic in uniq_topics:
            topic_indices = [i for i, t in enumerate(topics) if t == topic]
            if not topic_indices:
                continue
            topic_embeddings = embeddings[topic_indices]
            
            # Centroid
            centroid = np.mean(topic_embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            
            # Distances
            for emb in topic_embeddings:
                d = cosine_distance(emb.reshape(1, -1), centroid.reshape(1, -1))[0]
                distances.append(d)

        if distances:
            all_distances_dict[subdir.name] = np.array(distances)

    if not all_distances_dict:
        print("No distances computed.")
        return

    # --- PLOTTING ---
    print("Plotting...")
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # OpenAI text-embedding-3-large distances might be slightly different range than SBERT
    # But usually < 0.6 is safe for semantically related content.
    x = np.linspace(0, 0.6, 1000) 
    
    kde_data = {}
    kurtosis_values = {}
    
    for dir_name, distances_array in sorted(all_distances_dict.items()):
        label, color = resolve_info(dir_name)
        
        # Calculate KDE
        try:
            kde = gaussian_kde(distances_array)
            density = kde(x)
            kde_data[label] = (x, density, color)
            
            # Kurtosis
            kurt_val = kurtosis(distances_array, fisher=True)
            kurtosis_values[label] = kurt_val
            
            # Line style
            lw = 2.5 if "Ours" in label or "Leader" in label else 1.5
            zorder = 10 if "Ours" in label else 5
            
            plt.plot(x, density, linewidth=lw, color=color, label=label, alpha=0.9, zorder=zorder)
            plt.fill_between(x, density, alpha=0.1, color=color, zorder=zorder-1)
            
        except Exception as e:
            print(f"Skipping plot for {label}: {e}")

    # --- ANNOTATIONS ---
    # 1. Homogenization Zone
    ax.axvspan(0.0, 0.1, alpha=0.1, color='red', zorder=0)
    ax.text(0.05, ax.get_ylim()[1] * 0.96, 'Echo Chamber\nRisk', 
            ha='center', va='top', fontsize=9, color='darkred', weight='bold')
    
    # 2. Divergence Zone
    ax.axvspan(0.35, 0.6, alpha=0.05, color='blue', zorder=0)
    ax.text(0.48, ax.get_ylim()[1] * 0.05, 'Zone of Divergence', 
            ha='center', va='bottom', fontsize=9, color='darkblue')
    
    # 3. Highlight Ours
    if "Vertical Collaboration (Ours)" in kde_data:
        x_o, y_o, c_o = kde_data["Vertical Collaboration (Ours)"]
        peak_idx = np.argmax(y_o)
        ax.annotate('Optimal Balance', 
                   xy=(x_o[peak_idx], y_o[peak_idx]), 
                   xytext=(x_o[peak_idx] + 0.08, y_o[peak_idx]),
                   arrowprops=dict(arrowstyle='->', color=c_o, lw=2),
                   fontsize=10, ha='left', va='center', color=c_o, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))

    # --- FORMATTING ---
    plt.xlabel("Semantic Distance to Centroid", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 0.6) 
    
    # Legend
    plt.legend(loc='center right', fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
    
    # --- Kurtosis Table ---
    table_data = []
    sorted_kurt = sorted(kurtosis_values.items(), key=lambda item: item[1], reverse=True)
    
    for name, k_val in sorted_kurt:
        display_name = name.split(" (")[0] if "(" in name else name
        table_data.append([display_name, f"{k_val:.2f}"])
    
    if table_data:
        row_height = 0.05
        table_height = (len(table_data) + 1) * row_height
        
        table = ax.table(cellText=table_data,
                        colLabels=['Structure', 'Kurtosis'],
                        cellLoc='left',
                        loc='upper right',
                        bbox=[0.68, 0.55, 0.30, table_height], # Adjusted bbox slightly down
                        edges='horizontal')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')

    plt.tight_layout()
    
    png_path = out_root / "density_social_structure_OpenAI_v3large.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Wrote plot to {png_path}")

if __name__ == "__main__":
    main()