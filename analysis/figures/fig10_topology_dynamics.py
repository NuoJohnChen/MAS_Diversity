import os
import re
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import threading

# --- 1. 配置与初始化 ---

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

DATA_DIRS = {
    "Standard": "./data/extracted_proposals/extracted_proposals_recursive",
    "NGT": "./data/extracted_proposals/extracted_proposals_ngt",
    "Subgroups": "./data/extracted_proposals/extracted_proposals_subgroup"
}

# 并行设置
MAX_WORKERS = 10 

# --- 2. 核心辅助函数 ---

def get_embedding(text):
    """
    科学的 Embedding 获取方式：
    如果文本超过 Token 限制，进行切片(Chunking)并取平均(Average)，
    而不是简单截断。
    """
    # text-embedding-3-large 限制约 8191 tokens。
    # 为了安全，我们按字符切分，每块约 12000 字符 (约 3000-4000 tokens)，留足余量
    CHUNK_SIZE = 12000
    
    # 移除换行，减少噪音
    text = text.replace("\n", " ")
    
    if not text.strip():
        return np.zeros(3072)

    # 如果文本较短，直接请求
    if len(text) < CHUNK_SIZE:
        for _ in range(3): # 重试机制
            try:
                resp = client.embeddings.create(input=[text], model="text-embedding-3-large")
                return np.array(resp.data[0].embedding)
            except Exception as e:
                time.sleep(1)
        return np.zeros(3072)
    
    # 如果文本过长，进行切片
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    chunk_embeddings = []
    
    for chunk in chunks:
        for _ in range(3):
            try:
                resp = client.embeddings.create(input=[chunk], model="text-embedding-3-large")
                chunk_embeddings.append(resp.data[0].embedding)
                break
            except Exception:
                time.sleep(1)
    
    if not chunk_embeddings:
        return np.zeros(3072)
    
    # 核心步骤：对所有切片的向量取平均值 (Mean Pooling)
    # 这样能代表整段长文本的语义重心
    avg_embedding = np.mean(chunk_embeddings, axis=0)
    
    # 重新归一化 (Embedding 通常需要是单位向量)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
        
    return avg_embedding 

def judge_strict_critique(prev_context, current_text):
    prompt = f"""
    Compare Speaker B's statement to Speaker A's context.
    Speaker A: "{prev_context[-400:]}..."
    Speaker B: "{current_text}"
    Task: Rate level of DISAGREEMENT/NOVELTY (1-10).
    Strict Scoring:
    - 1-4: Echo/Additive (Safe)
    - 5-6: Minor Detail
    - 7-8: Soft Critique/Refinement
    - 9-10: Major Disruption
    Output integer only.
    """
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.0
            )
            content = response.choices[0].message.content
            match = re.search(r'\d+', content)
            if match:
                return int(match.group())
        except Exception:
            time.sleep(1)
    return 4 

def parse_chat_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        pattern = re.compile(r'(Participant \d+):(.*?)(?=Participant \d+:|$)', re.DOTALL)
        matches = pattern.findall(content)
        return [{"speaker": s.strip(), "text": t.strip()} for s, t in matches]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

# --- 3. 单文件处理逻辑 ---

def process_single_file(args):
    mode_name, file_path = args
    results = []
    
    turns = parse_chat_file(file_path)
    if not turns or len(turns) < 2:
        return []
    
    anchor_vec = np.array(get_embedding(turns[0]['text'])).reshape(1, -1)
    
    for i, turn in enumerate(turns):
        if i == 0: continue 
        
        vec = np.array(get_embedding(turn['text'])).reshape(1, -1)
        if np.all(anchor_vec == 0) or np.all(vec == 0):
            sim = 1.0 
        else:
            sim = cosine_similarity(vec, anchor_vec)[0][0]
            
        divergence = 1.0 - sim
        
        prev_text = turns[i-1]['text']
        raw_score = judge_strict_critique(prev_text, turn['text'])
        
        # Threshold calculation happens here, but we hide it in the plot label
        is_high_conflict = 1 if raw_score >= 7 else 0
        
        results.append({
            "Mode": mode_name,
            "Turn": i,
            "Divergence": divergence,
            "Score": raw_score,
            "HighConflict": is_high_conflict
        })
        
    return results

# --- 4. 主程序 ---

if __name__ == "__main__":
    all_tasks = []
    for mode_name, dir_path in DATA_DIRS.items():
        path_obj = Path(dir_path)
        files = list(path_obj.glob("*.txt"))
        print(f"Found {len(files)} files for mode: {mode_name}")
        for f in files:
            all_tasks.append((mode_name, f))
    
    if not all_tasks:
        print("Error: No files found.")
        exit()

    print(f"\nStarting parallel processing of {len(all_tasks)} files...")

    all_data = []
    completed_count = 0
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_single_file, task): task for task in all_tasks}
        
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                data = future.result()
                if data:
                    all_data.extend(data)
                
                with lock:
                    completed_count += 1
                    if completed_count % 10 == 0:
                        print(f"Progress: {completed_count}/{len(all_tasks)} files processed.")
                        
            except Exception as exc:
                print(f"Task generated an exception: {exc}")

    # --- 5. 绘图 (Visual Refinement) ---
    
    df = pd.DataFrame(all_data)
    if df.empty: exit()
    df = df[df["Turn"] <= 5]

    df_agg = df.groupby(["Mode", "Turn"]).agg({
        "Divergence": ["mean", "sem"],
        "HighConflict": ["mean", "sem"]
    }).reset_index()
    df_agg.columns = ["Mode", "Turn", "Div_mean", "Div_sem", "Conf_mean", "Conf_sem"]

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = {"Standard": "#d62728", "NGT": "#2ca02c", "Subgroups": "#17becf"}
    markers = {"Standard": "o", "NGT": "s", "Subgroups": "^"}
    # 更加学术的 Legend
    labels_map = {"Standard": "Standard", "NGT": "NGT", "Subgroups": "Subgroups"}

    # === 左图: Semantic Diversity (原 Divergence) ===
    for mode in colors.keys():
        if mode not in df_agg["Mode"].values: continue
        subset = df_agg[df_agg["Mode"] == mode]
        
        ax1.plot(subset["Turn"], subset["Div_mean"], marker=markers[mode], 
                 color=colors[mode], linewidth=3, markersize=9, label=labels_map[mode])
        # 使用0.5倍标准误，使误差带更窄
        ax1.fill_between(subset["Turn"], 
                         subset["Div_mean"] - subset["Div_sem"] * 0.7, 
                         subset["Div_mean"] + subset["Div_sem"] * 0.7, 
                         color=colors[mode], alpha=0.15)

    # 标题和轴标签修改：强调 Diversity
    ax1.set_title("Evolution of Semantic Diversity", fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel("Discussion Turn", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Semantic Diversity", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_xticks(range(1, 6))
    ax1.set_xticklabels([f"T{i}" for i in range(1, 6)])

    # === 右图: Dialectical Diversity (原 Critique) ===
    for mode in colors.keys():
        if mode not in df_agg["Mode"].values: continue
        subset = df_agg[df_agg["Mode"] == mode]
        
        ax2.plot(subset["Turn"], subset["Conf_mean"], marker=markers[mode], 
                 color=colors[mode], linewidth=3, markersize=9, label=None)
        # 使用0.5倍标准误，使误差带更窄
        ax2.fill_between(subset["Turn"], 
                         subset["Conf_mean"] - subset["Conf_sem"] * 0.7, 
                         subset["Conf_mean"] + subset["Conf_sem"] * 0.7, 
                         color=colors[mode], alpha=0.15)

    # 标题和轴标签修改：概念化，去掉数学符号
    ax2.set_title("Density of Constructive Conflict", fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlabel("Discussion Turn", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Constructive Conflict Ratio", fontsize=12, fontweight='bold')
    
    # # 增加一个小 Note 解释 Metric，不抢主视觉
    # ax2.text(0.95, 0.95, "Note: Ratio of interactions with\nLLM Critique Score $\geq$ 7", 
    #          transform=ax2.transAxes, fontsize=9, color='#555555', va='top', ha='right',
    #          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_xticks(range(1, 6))
    ax2.set_xticklabels([f"T{i}" for i in range(1, 6)])
    ax2.set_ylim(0, 1.0) 

    lines, labels = ax1.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=16, frameon=False)

    plt.tight_layout()
    output_path = "diversity_dynamics_final.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Plot saved to {output_path}")