import os
import re
import time
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from openai import OpenAI
import concurrent.futures
import threading
import sys

metrics_path = Path(__file__).parent.parent / "metrics"
sys.path.insert(0, str(metrics_path))
from compute_vendi_extended import safe_exec, get_openai_embeddings_cached

# --- 1. 配置 ---

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

DATA_DIRS = {
    3: "./data/extracted_proposals/extracted_proposals_groupsize3",
    4: "./data/extracted_proposals/extracted_proposals_groupsize4",
    5: "./data/extracted_proposals/extracted_proposals_groupsize5",
    6: "./data/extracted_proposals/extracted_proposals_groupsize6",
    7: "./data/extracted_proposals/extracted_proposals_groupsize7",
    8: "./data/extracted_proposals/extracted_proposals_groupsize8"
}

MAX_WORKERS = 10 

# --- 2. 核心算法 (含长文本切片) ---

def get_embedding_safe(text):
    """科学的 Embedding 获取方式：切片平均"""
    CHUNK_SIZE = 12000
    text = text.replace("\n", " ")
    if not text.strip(): return np.zeros(3072)

    if len(text) < CHUNK_SIZE:
        for _ in range(3):
            try:
                resp = client.embeddings.create(input=[text], model="text-embedding-3-large")
                return np.array(resp.data[0].embedding)
            except Exception: time.sleep(1)
        return np.zeros(3072)
    
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    chunk_embeddings = []
    for chunk in chunks:
        for _ in range(3):
            try:
                resp = client.embeddings.create(input=[chunk], model="text-embedding-3-large")
                chunk_embeddings.append(resp.data[0].embedding)
                break
            except Exception: time.sleep(1)
    
    if not chunk_embeddings: return np.zeros(3072)
    avg_embedding = np.mean(chunk_embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0: avg_embedding = avg_embedding / norm
    return avg_embedding

def compute_vendi_score(embeddings):
    """Vendi Score 使用 vendi_score 包计算，与 CSV 数据生成代码一致"""
    try:
        from vendi_score import vendi
    except ImportError:
        raise ImportError("vendi_score package not found. Please install it with: pip install vendi-score")
    
    if len(embeddings) < 2:
        return 1.0
    
    emb = np.array(embeddings)
    n = emb.shape[0]
    if n == 0:
        return float("nan")
    
    # 确保L2归一化（安全起见，即使已经归一化也再次归一化）
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms
    
    # 计算内积矩阵（与 compute_vendi_and_order.py 一致）
    # 对于L2归一化的向量，内积 = 余弦相似度
    K = emb @ emb.T
    # 确保对称性（数值稳定性）
    K = (K + K.T) / 2.0
    
    # 使用 vendi_score 包
    score = vendi.score_K(K)
    return float(score)

def parse_agents(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        pattern = re.compile(r'(Participant \d+):(.*?)(?=Participant \d+:|$)', re.DOTALL)
        matches = pattern.findall(content)
        agent_texts = {}
        for speaker, text in matches:
            speaker = speaker.strip()
            if speaker not in agent_texts: agent_texts[speaker] = ""
            agent_texts[speaker] += " " + text.strip()
        return agent_texts
    except: return {}

# --- 3. 处理流程 ---

def process_file_vendi(args):
    g_size, file_path = args
    agent_texts = parse_agents(file_path)
    if len(agent_texts) < 2:
        return None
    
    # 不在这里做 8000 字符截断，直接使用完整文本；
    # 超长文本由 get_embedding_safe 内部的 chunking 处理
    embeddings = []
    for txt in agent_texts.values():
        emb = get_embedding_safe(txt)
        if not np.all(emb == 0):
            embeddings.append(emb)
    if not embeddings:
        return None

    score = compute_vendi_score(embeddings)
    return {"Group Size": g_size, "Vendi Score": score}

# --- 4. 主程序 ---

def get_openai_embeddings_batch(texts, batch_size=50, model="text-embedding-3-large"):
    """批量获取embeddings，支持批处理以提高效率"""
    embeddings = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=model)
            batch_embs = [np.array(data.embedding) for data in resp.data]
            embeddings.extend(batch_embs)
            if (i + batch_size) % 200 == 0 or (i + batch_size) >= total:
                print(f"      Processed {min(i + batch_size, total)}/{total} texts...")
        except Exception as e:
            print(f"    Error fetching embeddings for batch {i//batch_size + 1}: {e}")
            # 如果批量失败，回退到单个处理
            for txt in batch:
                emb = get_embedding_safe(txt)
                embeddings.append(emb)
    
    return embeddings

"""
--- 4. 主程序 ---
- 单位：participant（跨 topic 聚合）
- 不做 8000 字符截断（仅依赖 get_embedding_safe 内部的 chunking 处理极长文本）
- 使用 get_openai_embeddings_cached 做磁盘缓存，避免重复调用 API
"""

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Compute Vendi scores for different group sizes")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache: recompute all embeddings even if cache exists"
    )
    args = parser.parse_args()
    
    use_cache = not args.no_cache
    if not use_cache:
        print("Cache disabled: will recompute all embeddings")
    
    # 使用 participants 分析：group size = participant 数量
    results = []
    
    for g_size, dir_path in DATA_DIRS.items():
        path_obj = Path(dir_path)
        files = sorted(list(path_obj.glob("*_proposals.txt")))
        print(f"Processing Group Size {g_size}: Found {len(files)} topic files")
        
        # 合并所有 topics 的所有 participants
        # 每个 topic 的 proposals 按顺序对应 participants（第1个proposal = Participant 1，第2个 = Participant 2，...）
        all_agent_texts = {}  # {participant_idx: combined_text}
        
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()
                
                # 尝试解析 paper_txts 格式
                if 'paper_txts' in content:
                    ns = safe_exec(content)
                    papers = ns.get("paper_txts", [])
                    # 每个proposal按顺序对应一个participant（索引从0开始）
                    for idx, p in enumerate(papers):
                        text = p.strip()
                        if text:
                            participant_idx = idx % g_size  # 确保索引在[0, g_size-1]范围内
                            if participant_idx not in all_agent_texts:
                                all_agent_texts[participant_idx] = ""
                            all_agent_texts[participant_idx] += " " + text.strip()
                else:
                    # 尝试解析 Participant X: 格式
                    agent_texts = parse_agents(f)
                    if agent_texts:
                        # 如果有participant格式，合并每个participant的文本
                        for participant, text in agent_texts.items():
                            # 提取participant编号
                            match = re.search(r'Participant (\d+)', participant)
                            if match:
                                participant_idx = int(match.group(1)) - 1  # 转换为0-based索引
                                if participant_idx not in all_agent_texts:
                                    all_agent_texts[participant_idx] = ""
                                all_agent_texts[participant_idx] += " " + text.strip()
            except Exception as e:
                print(f"  Warning: Failed to parse file {f.name}: {e}")
                continue
        
        if len(all_agent_texts) < 2:
            print(f"  Warning: Group Size {g_size} has less than 2 participants, skipping")
            continue
        
        print(f"  Total unique participants across all topics: {len(all_agent_texts)}")
        
        # 不截断，只做换行清理
        # 按participant索引排序，确保顺序一致
        texts_clean = [all_agent_texts[idx].replace("\n", " ").strip() 
                       for idx in sorted(all_agent_texts.keys()) 
                       if all_agent_texts[idx].strip()]
        
        if len(texts_clean) < 2:
            print(f"  Warning: Group Size {g_size} has less than 2 non-empty texts, skipping")
            continue
        
        # 分离正常长度和超长文本
        # text-embedding-3-large 限制约8192 tokens，留余量使用8000 tokens ≈ 32000 chars
        MAX_CHARS_PER_TEXT = 32000
        normal_texts = []
        long_texts = []
        text_indices = []  # 记录每个文本的原始索引
        
        for idx, txt in enumerate(texts_clean):
            if len(txt) > MAX_CHARS_PER_TEXT:
                long_texts.append((idx, txt))
            else:
                normal_texts.append(txt)
                text_indices.append(idx)
        
        if len(normal_texts) + len(long_texts) < 2:
            print(f"  Warning: Group Size {g_size} has less than 2 non-empty texts, skipping")
            continue
        
        # 初始化embeddings列表
        all_embeddings = [None] * len(texts_clean)
        
        # 处理正常长度的文本（使用批处理和缓存）
        if normal_texts:
            cache_file = path_obj / "embeddings_cache_v3large_participants.pkl"
            cache_status = "disabled" if not use_cache else f"{cache_file.name}"
            print(f"  Fetching embeddings for {len(normal_texts)} normal-length participants (cache: {cache_status})...")
            try:
                if use_cache:
                    emb_array = get_openai_embeddings_cached(normal_texts, cache_file, model="text-embedding-3-large")
                else:
                    # 禁用 cache：删除缓存文件（如果存在）然后重新计算
                    if cache_file.exists():
                        cache_file.unlink()
                        print(f"    Cache file removed, will recompute embeddings")
                    emb_array = get_openai_embeddings_cached(normal_texts, cache_file, model="text-embedding-3-large")
                    # 计算完后删除缓存文件（因为我们不想保存）
                    if cache_file.exists():
                        cache_file.unlink()
                
                # 将结果放回正确位置
                for i, emb in enumerate(emb_array):
                    all_embeddings[text_indices[i]] = emb
            except Exception as e:
                print(f"  Error fetching embeddings with cache: {e}")
                print(f"  Falling back to individual processing for normal texts...")
                for i, txt in enumerate(normal_texts):
                    emb = get_embedding_safe(txt)
                    if not np.all(emb == 0):
                        all_embeddings[text_indices[i]] = emb
        
        # 单独处理超长文本（使用get_embedding_safe的chunking机制）
        if long_texts:
            print(f"  Processing {len(long_texts)} long participants with chunking...")
            for orig_idx, txt in long_texts:
                emb = get_embedding_safe(txt)
                if not np.all(emb == 0):
                    all_embeddings[orig_idx] = emb
        
        # 过滤None和零向量
        embeddings = [emb for emb in all_embeddings if emb is not None and not np.all(emb == 0)]
        
        if len(embeddings) < 2:
            print(f"  Warning: Group Size {g_size} got less than 2 embeddings, skipping")
            continue
        
        # 计算 Vendi Score
        score = compute_vendi_score(embeddings)
        print(f"  Group Size {g_size}: Vendi Score = {score:.4f} (n={len(embeddings)})\n")
        
        results.append({
            "Group Size": g_size,
            "Vendi Score": score,
            "Num_Participants": len(embeddings),
        })

    # --- 5. 数据计算 ---
    
    df = pd.DataFrame(results)
    if df.empty: exit()
    
    # 聚合（现在每个group size只有一个值，因为已经合并了所有topics）
    df_agg = df[["Group Size", "Vendi Score"]].copy()
    df_agg["mean"] = df_agg["Vendi Score"]
    df_agg["sem"] = 0.0  # 只有一个值，所以SEM为0
    
    # 计算 Utilization Ratio (利用率)
    # Ratio = Actual Vendi / Theoretical Max (N)
    # 这是一个 [0, 1] 的无量纲指标，可以安全地比较
    df_agg["Utilization"] = df_agg["mean"] / df_agg["Group Size"]
    df_agg["Utilization_sem"] = df_agg["sem"] / df_agg["Group Size"] # 简单的误差传递

    print("\n=== Data Analysis Table ===")
    print(df_agg)

    # --- 6. 绘图：Ideal vs Actual (Reviewer-Proof) ---
    
    sns.set_theme(style="white", rc={"axes.grid": False})
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_actual = "#1f77b4"  # 蓝线 (Actual)
    color_ideal = "#7f7f7f"   # 灰线 (Ideal)
    color_util = "#d62728"    # 红柱 (Utilization)

    # === 左轴: Scaling Gap Analysis ===
    ax1.set_xlabel("Group Size ($N$)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Effective Diversity (Vendi Score)", fontsize=14, fontweight='bold', color=color_actual)
    
    # 1. 画理想线 (Theoretical Limit: y = x)
    # 这是一个完美的物理参照系
    x_range = np.linspace(2.8, 8.2, 20)
    ax1.plot(x_range, x_range, color=color_ideal, linestyle='--', linewidth=2, alpha=0.6, label="Theoretical Max ($y=N$)")
    
    # 2. 画实际观测线
    ax1.errorbar(df_agg["Group Size"], df_agg["mean"], yerr=df_agg["sem"], 
                 fmt='-o', color=color_actual, linewidth=3, markersize=10, capsize=5, 
                 label="Observed Diversity")
    
    # 3. 标注 Gap
    # 在最大 groupsize 处画一条垂直线展示差距
    max_size = df_agg["Group Size"].max()
    if max_size in df_agg["Group Size"].values:
        actual_max = df_agg[df_agg["Group Size"]==max_size]["mean"].values[0]
        ax1.plot([max_size, max_size], [actual_max, max_size], color='black', linestyle=':', alpha=0.5)
        ax1.text(max_size + 0.05, (actual_max + max_size)/2, "Redundancy Gap", rotation=90, va='center', fontsize=10, color='black')

    ax1.tick_params(axis='y', labelcolor=color_actual)
    ax1.set_xticks(sorted(df_agg["Group Size"].unique()))
    
    # === 右轴: Utilization Ratio (归一化指标) ===
    ax2 = ax1.twinx()
    ax2.set_ylabel("Diversity Utilization Ratio ($Vendi / N$)", fontsize=14, fontweight='bold', color=color_util)
    
    # 画柱状图
    bars = ax2.bar(df_agg["Group Size"], df_agg["Utilization"], 
            width=0.3, color=color_util, alpha=0.3, label="Utilization Efficiency")
    
    # 标注数值
    for rect in bars:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', color=color_util, fontweight='bold')

    # 添加趋势箭头
    ax2.plot(df_agg["Group Size"], df_agg["Utilization"], 
             color=color_util, linestyle='-', marker='', alpha=0.6)

    ax2.tick_params(axis='y', labelcolor=color_util)
    
    # 调整范围
    max_size = df_agg["Group Size"].max()
    y_min, y_max = df_agg["mean"].min(), max_size + 0.2 # Max 要能包住最大 N
    ax1.set_ylim(y_min * 0.9, max_size + 0.5) # 左轴要给 Ideal Line 留空间
    
    ax2.set_ylim(0, 1.0) # Utilization 是 0-1

    # === 合并图例 ===
    # 手动创建图例对象以获得完美控制
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], color=color_ideal, linestyle='--', label='Theoretical Max (Perfect Orthogonality)'),
        Line2D([0], [0], color=color_actual, marker='o', label='Observed Diversity'),
        Patch(facecolor=color_util, alpha=0.3, label='Utilization Ratio (Efficiency)')
    ]
    
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=10)

    plt.title("Scaling Efficiency: The Divergence from Theory", fontsize=16, fontweight='bold', y=1.15)
    plt.tight_layout()
    
    output_path = "scaling_gap_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Plot saved to {output_path}")