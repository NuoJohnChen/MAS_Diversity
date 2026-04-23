import os
import matplotlib
import re
import sys
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import rbf_kernel # 引入 RBF 核用于计算 MMD

# Set backend to Agg for non-interactive environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# --- OpenAI API Setup ---
try:
    from openai import OpenAI
except ImportError:
    print("Warning: 'openai' library not found. Will rely on cache only.")
    pass 

# !!! 请在此处填入你的 API Key !!!
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"


try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except:
    client = None

# --- CONFIGURATION ---
ROUNDS_ORDER = ["round_0", "round_1", "round_2", "round_3", "final_round"]
ROUND_LABELS_MAP = {
    "round_0": "R0",
    "round_1": "R1",
    "round_2": "R2",
    "round_3": "R3",
    "final_round": "Final"
}

def get_openai_embeddings_cached(texts, cache_file, model="text-embedding-3-large", batch_size=50):
    """读取缓存的 Embeddings，如果不存在则调用 API。"""
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if len(cached_data) == len(texts):
                return np.array(cached_data)
        except Exception:
            pass

    if not texts:
        return np.array([])
        
    embeddings = []
    total = len(texts)
    
    if client is None:
        raise ValueError("OpenAI Client not initialized and no cache found.")

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch = [t.replace("\n", " ") for t in batch]
            response = client.embeddings.create(input=batch, model=model)
            batch_embs = [data.embedding for data in response.data]
            embeddings.extend(batch_embs)
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise e
            
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return np.array(embeddings)

def calc_dispersion(embeddings, centroid):
    """计算离散度：所有点到质心的平均 Cosine Distance"""
    if len(embeddings) == 0: return 0.0
    dists = [cosine(emb, centroid) for emb in embeddings]
    return np.mean(dists)

def calc_mmd(X, Y, gamma=None):
    """
    计算两个集合 X 和 Y 之间的 Maximum Mean Discrepancy (MMD)。
    使用 RBF (Gaussian) Kernel。
    MMD^2(X,Y) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    
    X: (n1, dim)
    Y: (n2, dim)
    gamma: RBF kernel width. 如果为 None，sklearn 默认为 1/n_features
    """
    if len(X) == 0 or len(Y) == 0:
        return 0.0
        
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    
    # 简单的无偏估计
    return XX.mean() + YY.mean() - 2 * XY.mean()

def main():
    # --- 请修改你的路径 ---
    base_root = Path("./data/extracted_proposals/extracted_proposals_by_round")
    out_root = Path("./data/high_dim_stats_mmd") # 新文件夹
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. LOAD DATA
    data_store = {}
    print(">>> Step 1: Loading High-Dimensional Embeddings...")
    
    for r_idx, r_name in enumerate(ROUNDS_ORDER):
        r_path = base_root / r_name
        if not r_path.exists(): continue
        
        files = sorted(list(r_path.glob("*_proposals.txt")))
        round_texts = []
        file_map = [] 
        
        for fp in files:
            topic = fp.name.replace("_proposals.txt", "")
            try:
                ns = {}
                with fp.open("r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                content = re.sub(r'\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])', r'?', content)
                code = compile(content, str(fp), "exec")
                exec(code, {}, ns)
                papers = ns.get("paper_txts", [])
                papers = [p.strip() for p in papers if p.strip()]
                
                if papers:
                    start = len(round_texts)
                    round_texts.extend(papers)
                    end = len(round_texts)
                    file_map.append((topic, start, end))
            except Exception as e:
                print(f"Err parsing {fp.name}: {e}")

        if not round_texts: continue
        
        cache_file = r_path / "embeddings_cache.pkl"
        try:
            emb_matrix = get_openai_embeddings_cached(round_texts, cache_file)
            for topic, s, e in file_map:
                if topic not in data_store: data_store[topic] = {}
                data_store[topic][r_name] = emb_matrix[s:e]
        except Exception as e:
            print(f"Failed to get embeddings for {r_name}: {e}")

    # 2. CALCULATE METRICS
    print(">>> Step 2: Calculating Metrics (Drift, Dispersion, MMD)...")
    
    drift_records = []      # 记录 (Topic, Round_Transition, Velocity)
    dispersion_records = [] # 记录 (Topic, Round, Dispersion)
    mmd_records = []        # 记录 (Topic, Round_Transition, MMD)
    
    for topic, rounds_data in data_store.items():
        centroids = {}
        
        # A. 计算每一轮的 Dispersion 和 Centroid
        for r_name in ROUNDS_ORDER:
            if r_name not in rounds_data: continue
            embs = rounds_data[r_name]
            if len(embs) == 0: continue
            
            cent = np.mean(embs, axis=0)
            centroids[r_name] = cent
            
            disp = calc_dispersion(embs, cent)
            dispersion_records.append({
                "Topic": topic,
                "Round": ROUND_LABELS_MAP[r_name],
                "Dispersion": disp
            })
            
        # B. 计算 Transition Metrics (Drift & MMD)
        for i in range(len(ROUNDS_ORDER) - 1):
            curr_r = ROUNDS_ORDER[i]
            next_r = ROUNDS_ORDER[i+1]
            
            if curr_r in rounds_data and next_r in rounds_data:
                # Data for MMD
                X = rounds_data[curr_r]
                Y = rounds_data[next_r]
                
                # 1. MMD (Distribution Shift)
                # 使用默认 gamma，或者你可以指定 gamma=1.0
                mmd_val = calc_mmd(X, Y)
                
                label = f"{ROUND_LABELS_MAP[curr_r]} → {ROUND_LABELS_MAP[next_r]}"
                mmd_records.append({
                    "Topic": topic,
                    "Transition": label,
                    "MMD": mmd_val
                })
                
                # 2. Drift Velocity (Centroid Shift)
                if curr_r in centroids and next_r in centroids:
                    dist = cosine(centroids[curr_r], centroids[next_r])
                    drift_records.append({
                        "Topic": topic,
                        "Transition": label,
                        "Velocity": dist
                    })

    df_drift = pd.DataFrame(drift_records)
    df_disp = pd.DataFrame(dispersion_records)
    df_mmd = pd.DataFrame(mmd_records)

    if df_drift.empty:
        print("Error: No data computed.")
        return

    # 3. DIAGNOSTICS (Updated with MMD)
    print("\n" + "="*40)
    print("DIAGNOSTICS REPORT")
    print("="*40)
    
    final_trans = "R3 → Final"
    
    # Check MMD Outliers (Distribution changed most)
    mmd_final = df_mmd[df_mmd["Transition"] == final_trans]
    if not mmd_final.empty:
        print(f"\n[Top 5 Topics by Distribution Shift (MMD) in {final_trans}]:")
        # MMD 越大，分布差异越大
        top_mmd = mmd_final.sort_values(by="MMD", ascending=False).head(5)
        for _, row in top_mmd.iterrows():
            print(f"  - {row['Topic']}: {row['MMD']:.6f}")
    
    print("="*40 + "\n")

    # 4. PLOTTING (3 Subplots)
    print(">>> Step 3: Generating Plots...")
    # 设置风格
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 定义x轴顺序
    transition_order = ["R0 → R1", "R1 → R2", "R2 → R3", "R3 → Final"]
    
    # 变成 1行3列
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # Plot A: Drift (Centroid)
    sns.boxplot(data=df_drift, x="Transition", y="Velocity", ax=axes[0], 
                palette="Reds", showfliers=False, width=0.6, linewidth=1.5,
                order=transition_order)
    sns.stripplot(data=df_drift, x="Transition", y="Velocity", ax=axes[0], 
                  color=".2", alpha=0.5, size=4, jitter=True, order=transition_order)
    axes[0].set_title("A. Centroid Drift (Mean Shift)", fontweight='bold', pad=15)
    axes[0].set_ylabel("Cosine Distance", fontweight='bold')
    axes[0].set_xlabel("Debate Stage Transition", fontweight='bold')
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Plot B: MMD (Distribution)
    sns.boxplot(data=df_mmd, x="Transition", y="MMD", ax=axes[1], 
                palette="Purples", showfliers=False, width=0.6, linewidth=1.5,
                order=transition_order)
    sns.stripplot(data=df_mmd, x="Transition", y="MMD", ax=axes[1], 
                  color=".2", alpha=0.5, size=4, jitter=True, order=transition_order)
    axes[1].set_title("B. Distribution Shift (MMD)", fontweight='bold', pad=15)
    axes[1].set_ylabel("Max Mean Discrepancy", fontweight='bold')
    axes[1].set_xlabel("Debate Stage Transition", fontweight='bold')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Plot C: Dispersion
    sns.boxplot(data=df_disp, x="Round", y="Dispersion", ax=axes[2], 
                palette="Blues", showfliers=False, width=0.6, linewidth=1.5)
    sns.stripplot(data=df_disp, x="Round", y="Dispersion", ax=axes[2], 
                  color=".2", alpha=0.5, size=4, jitter=True)
    axes[2].set_title("C. Semantic Diversity (Dispersion)", fontweight='bold', pad=15)
    axes[2].set_ylabel("Avg. Distance to Centroid", fontweight='bold')
    axes[2].set_xlabel("Debate Round", fontweight='bold')
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(pad=3.0)
    out_path = out_root / "high_dim_stats_with_mmd.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Plot to: {out_path}")
    
    # Save CSVs
    df_drift.to_csv(out_root / "drift.csv", index=False)
    df_mmd.to_csv(out_root / "mmd.csv", index=False)
    df_disp.to_csv(out_root / "dispersion.csv", index=False)
    print("Saved CSVs.")

if __name__ == "__main__":
    main()