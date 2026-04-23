import os
import matplotlib
import re
import sys
import pickle
from collections import defaultdict

# Set backend to Agg for non-interactive environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde, kurtosis, skew

# --- OpenAI API Setup ---
try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library not installed.")
    sys.exit(1)

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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
            print(f"        Progress: {min(i + batch_size, total)}/{total}...", end='\r')
        except Exception as e:
            print(f"\nError: {e}")
            raise e
            
    print("")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
    except Exception:
        pass

    return np.array(embeddings)

def cosine_distance(vec1, vec2):
    norm1 = np.linalg.norm(vec1, axis=-1, keepdims=True) + 1e-12
    norm2 = np.linalg.norm(vec2, axis=-1, keepdims=True) + 1e-12
    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2
    cosine_sim = np.sum(vec1_norm * vec2_norm, axis=-1)
    return np.clip(1.0 - cosine_sim, 0.0, 2.0)

def draw_top_bracket(ax, x_start, x_end, text, color='black', y_offset=0.05):
    """Draw a bracket style annotation at the top of the plot."""
    # Get current y-axis limit to place text above data
    y_max = ax.get_ylim()[1]
    y_line = y_max * (1 + y_offset/2)
    y_text = y_max * (1 + y_offset)
    
    # Draw the horizontal line
    ax.plot([x_start, x_end], [y_line, y_line], color=color, lw=1.5, clip_on=False)
    
    # Draw vertical ticks at ends
    tick_height = y_max * 0.03
    ax.plot([x_start, x_start], [y_line - tick_height, y_line], color=color, lw=1.5, clip_on=False)
    ax.plot([x_end, x_end], [y_line - tick_height, y_line], color=color, lw=1.5, clip_on=False)
    
    # Add text
    ax.text((x_start + x_end) / 2, y_text, text, 
            ha='center', va='bottom', fontsize=10, weight='bold', color=color)

def main():
    base_root = Path("./data/extracted_proposals")
    out_root = Path("./data/tsne")
    out_root.mkdir(parents=True, exist_ok=True)

    subdirs = [
        p for p in base_root.iterdir()
        if p.is_dir() and p.name.startswith("ai_researcher_multi_topic_dsv3_")
    ]
    
    # --- NEUTRAL CONFIGURATION ---
    # Legend Name Mapping (Exploratory / Neutral Tone)
    legend_names = {
            "mix": "Vertical Collaboration",        # 必须在 "x" 之前！因为 "mix" 包含 "x"
            "leader": "Leader-Led Collaboration",       
            "young": "Horizontal Collaboration",        
            "rec": "Naive Collaboration",               
            "x": "Interdisciplinary Collaboration", # 放到最后作为兜底
        }
    
    # Color Mapping (Consistent but semantic)
    color_map = {
        "Leader-Led Collaboration": "#d62728",      # Red (Constraint)
        "Interdisciplinary Collaboration": "#9467bd", # Purple (Complex)
        "Vertical Collaboration": "#2ca02c",        # Green (Balanced)
        "Horizontal Collaboration": "#17becf",      # Cyan (Open)
        "Naive Collaboration": "#7f7f7f",           # Grey (Baseline)
    }

    def resolve_info(dir_name):
            # 1. 提取短名 (e.g., "mix", "x", "leader_experience2")
            short_name = dir_name.replace("ai_researcher_multi_topic_dsv3_", "").replace("_final", "")
            
            final_name = short_name # Default
            
            # 2. 优先尝试精确匹配 (Exact Match) - 最安全
            if short_name in legend_names:
                final_name = legend_names[short_name]
            else:
                # 3. 如果没精确匹配，再进行模糊查找 (Fuzzy Search)
                # 此时因为我们在字典里把 "mix" 放到了 "x" 前面，所以会先匹配到 mix
                for key, val in legend_names.items():
                    if key in short_name:
                        final_name = val
                        break
            
            return final_name

    # --- DATA COLLECTION ---
    data_by_label = defaultdict(list)
    
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
        
        print(f"Processing {subdir.name} -> Label: [{label}]")

        cache_file = subdir / "embeddings_cache_v3large.pkl"
        try:
            embeddings = get_openai_embeddings_cached(texts, cache_file, model="text-embedding-3-large")
        except Exception as e:
            print(f"Skipping {subdir.name}: {e}")
            continue

        uniq_topics = sorted(set(topics))
        for topic in uniq_topics:
            topic_indices = [i for i, t in enumerate(topics) if t == topic]
            if not topic_indices:
                continue
            topic_embeddings = embeddings[topic_indices]
            
            centroid = np.mean(topic_embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            
            for emb in topic_embeddings:
                d = cosine_distance(emb.reshape(1, -1), centroid.reshape(1, -1))[0]
                data_by_label[label].append(d)

    if not data_by_label:
        print("No distances computed.")
        return

    # --- PLOTTING ---
    print("Generating Plot...")
    plt.figure(figsize=(11, 7)) # Increased height for top brackets
    ax = plt.gca()
    
    x = np.linspace(0, 0.25, 1000) 
    kurtosis_values = {}
    skewness_values = {}
    peak_values = {}
    
    # Sort order: Naive -> Leader -> Others
    sort_keys = list(data_by_label.keys())
    # Simple logic to push Naive/Leader to background
    sort_keys.sort(key=lambda k: 0 if "Naive" in k else (1 if "Leader" in k else 2))

    max_density = 0

    for label in sort_keys:
        distances_array = np.array(data_by_label[label])
        color = color_map.get(label, "black")
        
        try:
            kde = gaussian_kde(distances_array)
            density = kde(x)
            max_density = max(max_density, np.max(density))
            
            # Compute statistics
            kurt_val = kurtosis(distances_array, fisher=True)
            skew_val = skew(distances_array)
            # Peak value: x position where density is maximum
            peak_idx = np.argmax(density)
            peak_x = x[peak_idx]
            
            kurtosis_values[label] = kurt_val
            skewness_values[label] = skew_val
            peak_values[label] = peak_x
            
            lw = 2.5
            zorder = 5
            if "Naive" in label or "Leader" in label:
                zorder = 2 # Background
            
            plt.plot(x, density, linewidth=lw, color=color, label=label, alpha=0.9, zorder=zorder)
            plt.fill_between(x, density, alpha=0.15, color=color, zorder=zorder-1)
            
        except Exception as e:
            print(f"Skipping plot for {label}: {e}")

    # --- ANNOTATIONS (BRACKET STYLE) ---
    # Set y-limit to fixed value
    ax.set_ylim(0, 35)
    
    # 1. Echo Chamber Risk Bracket (0.0 - 0.07)
    # Interpretation: High Semantic Redundancy / Repetition
    draw_top_bracket(ax, 0.0, 0.07, "Echo Chamber Risk\n(Semantic Redundancy)", 
                    color='#8B0000', y_offset=0.02)
    
    # 2. Zone of Divergence Bracket (0.15 - 0.25)
    # Interpretation: Exploration / High Entropy
    draw_top_bracket(ax, 0.15, 0.25, "Zone of Divergence\n(Exploration)", 
                    color='#00008B', y_offset=0.02)
    
    # --- FORMATTING ---
    plt.xlabel("Semantic Distance to Centroid", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 0.25)
    
    # Legend - upper right but slightly shifted toward center
    legend = plt.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=0.85, 
                       bbox_to_anchor=(0.98, 0.92))  # Slightly shifted from edge
    
    # --- STATISTICS TABLE ---
    table_data = []
    sorted_kurt = sorted(kurtosis_values.items(), key=lambda item: item[1], reverse=True)
    
    for name, k_val in sorted_kurt:
        display_name = name.replace(" Collaboration", "")
        skew_val = skewness_values.get(name, 0.0)
        peak_val = peak_values.get(name, 0.0)
        table_data.append([display_name, f"{k_val:.2f}", f"{skew_val:.2f}", f"{peak_val:.3f}"])
    
    if table_data:
        row_height = 0.06  # Increased row height
        total_height = max(0.25, (len(table_data) + 1) * row_height)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Structure', 'Kurtosis', 'Skewness', 'Peak'],
                        cellLoc='center',  # Center alignment for all columns
                        loc='lower right',
                        bbox=[0.55, 0.15, 0.43, total_height], # Wider to accommodate all columns
                        edges='horizontal')
        table.auto_set_font_size(False)
        table.set_fontsize(12)  # Slightly smaller to fit 4 columns
        table.scale(1.0, 1.3)  # Scale table vertically for better spacing
        
        # Adjust column widths: make all columns wider to prevent overlap
        for (row, col), cell in table.get_celld().items():
            if col == 0:  # Structure column
                cell.set_width(0.22)  # Wider for Structure
            elif col == 1:  # Kurtosis column
                cell.set_width(0.18)  # Wider for Kurtosis to prevent overlap
            elif col == 2:  # Skewness column
                cell.set_width(0.15)  # Wider for Skewness to prevent overlap
            elif col == 3:  # Peak column
                cell.set_width(0.12)  # Slightly wider for Peak
        
        # Style header - all cells are already center-aligned
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')

    plt.tight_layout()
    
    png_path = out_root / "density_social_structure_bracket_style.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Wrote plot to {png_path}")

if __name__ == "__main__":
    main()