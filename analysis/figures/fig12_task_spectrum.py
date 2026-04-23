import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
import re

# --- 1. 配置 ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 文件夹路径
TASK_DIRS = {
    "Physics\n(Hard Constraint)": "./data/extracted_proposals/newtopic/Multi_Collaboration_newtopic/extracted_proposals",
    "Policy\n(Soft Constraint)": "./data/extracted_proposals/newtopic/Multi_Collaboration_newtopic2/extracted_proposals",
    "Creative\n(No Constraint)": "./data/extracted_proposals/newtopic/Multi_Collaboration_newtopic3/extracted_proposals",
    # 这里的 AI 作为一个 High Entropy 的代表
    "AI Research": "./data/sec_models/dsv3_naive_recursive/extracted_proposals_representationlearning" 
}

# ICLR 风格配色 (Academic & Comfortable)
# 使用 Seaborn Deep/Muted 变体，更沉稳
PALETTE = {
    "Physics\n(Hard Constraint)": "#4C72B0",      # Deep Blue (稳重，代表收敛)
    "Policy\n(Soft Constraint)": "#DD8452",       # Deep Orange (温暖，代表人类共识)
    "Creative\n(No Constraint)": "#55A868",       # Deep Green (自然，代表生长/发散)
    "AI Research": "#C44E52" # Deep Red (醒目，代表复杂/高维)
}

# 排序：从收敛 (Physics) 到 极度发散 (AI)
ORDER = [
    "Physics\n(Hard Constraint)", 
    "Policy\n(Soft Constraint)", 
    "Creative\n(No Constraint)",
    "AI Research"
]

# --- 2. 核心函数：对 proposals 的“正确”计算逻辑（不截断内容，支持长文本chunking） ---

def get_embedding_single(text, max_retries=3):
    """
    获取单个文本的embedding，支持长文本chunking。
    不对文本做内容截断，只在内部按块切分以满足API上下文限制。
    """
    # text-embedding-3-large 上下文限制约 8192 tokens
    # 保守按字符切分，每块 ~12000 字符（约 3000-4000 tokens）
    CHUNK_SIZE = 12000

    # 基本清理：去掉换行，收紧首尾空格
    text = text.replace("\n", " ").strip()
    if not text:
        return np.zeros(3072)  # text-embedding-3-large 维度

    # 短文本：直接请求
    if len(text) <= CHUNK_SIZE:
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(input=[text], model="text-embedding-3-large")
                return np.array(resp.data[0].embedding)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"      Warning: Failed to get embedding after {max_retries} attempts: {e}")
                    return np.zeros(3072)

    # 长文本：按块切分，然后对所有块的embedding取平均
    chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    chunk_embeddings = []

    for chunk in chunks:
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(input=[chunk], model="text-embedding-3-large")
                chunk_embeddings.append(np.array(resp.data[0].embedding))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"      Warning: Failed to get chunk embedding: {e}")
                    # 某个chunk失败时，用零向量占位，避免完全丢弃
                    chunk_embeddings.append(np.zeros(3072))

    if not chunk_embeddings:
        return np.zeros(3072)

    avg_embedding = np.mean(chunk_embeddings, axis=0)
    # 重新归一化
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
    return avg_embedding


def get_embeddings(texts, batch_size=50):
    """
    对所有 proposals 计算 embeddings：
    - 不过滤/截断文本（除了基本的换行清理）
    - 对正常长度文本做批处理以提高效率
    - 对超长文本用 get_embedding_single 做chunking
    """
    # 清理文本（但不丢弃短文本）
    processed = [t.replace("\n", " ").strip() for t in texts if t is not None]
    if not processed:
        return []

    CHUNK_SIZE = 12000
    short_texts = []      # (idx, text)
    long_text_indices = []  # indices into processed

    for i, txt in enumerate(processed):
        if len(txt) <= CHUNK_SIZE:
            short_texts.append((i, txt))
        else:
            long_text_indices.append(i)

    embeddings = [None] * len(processed)

    # 批量处理短文本
    if short_texts:
        short_only = [t for _, t in short_texts]
        total_short = len(short_only)
        for i in range(0, total_short, batch_size):
            batch = short_only[i : i + batch_size]
            try:
                resp = client.embeddings.create(input=batch, model="text-embedding-3-large")
                batch_embs = [np.array(d.embedding) for d in resp.data]
                for j, emb in enumerate(batch_embs):
                    orig_idx = short_texts[i + j][0]
                    embeddings[orig_idx] = emb
                if (i + batch_size) % 200 == 0 or (i + batch_size) >= total_short:
                    print(f"      Processed {min(i + batch_size, total_short)}/{total_short} short texts...")
            except Exception as e:
                print(f"    Error in short-text batch {i//batch_size + 1}: {e}")
                # 回退到单个处理
                for j in range(i, min(i + batch_size, total_short)):
                    orig_idx = short_texts[j][0]
                    txt = short_texts[j][1]
                    embeddings[orig_idx] = get_embedding_single(txt)

    # 单独处理长文本
    if long_text_indices:
        print(f"      Processing {len(long_text_indices)} long texts with chunking...")
        for k, idx in enumerate(long_text_indices):
            txt = processed[idx]
            embeddings[idx] = get_embedding_single(txt)
            if (k + 1) % 10 == 0 or (k + 1) == len(long_text_indices):
                print(f"        Chunked {k + 1}/{len(long_text_indices)} long texts...")

    # 保底：把 None 替换成零向量，避免后续出错
    final_embeddings = []
    for emb in embeddings:
        if emb is None:
            final_embeddings.append(np.zeros(3072))
        else:
            final_embeddings.append(emb)

    return final_embeddings

def compute_pcd_distribution(embeddings):
    X = np.array(embeddings)
    if len(X) < 2: return []
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim_matrix = X @ X.T
    mask = np.triu_indices_from(sim_matrix, k=1)
    return 1.0 - sim_matrix[mask]

def compute_vendi_scalar(embeddings):
    X = np.array(embeddings)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    K = X @ X.T
    evals = np.linalg.eigvalsh(K) / n
    evals = evals[evals > 1e-10]
    return np.exp(-np.sum(evals * np.log(evals)))

def bootstrap_vendi(embeddings, n_boot=50, sample_size=None):
    X = np.array(embeddings)
    if len(X) < 5: return [0]
    if sample_size is None: sample_size = len(X)
    scores = []
    for _ in range(n_boot):
        X_resampled = resample(X, n_samples=sample_size)
        scores.append(compute_vendi_scalar(X_resampled))
    return scores

# --- 3. 数据处理 ---
pcd_data = []
vendi_boot_data = []

print("Starting Intrinsic Entropy Spectrum Analysis...")

for label, dir_path in TASK_DIRS.items():
    print(f"Processing {label}...")
    path_obj = Path(dir_path)
    files = sorted(list(path_obj.glob("*_proposals.txt")))
    
    texts = []
    for f in files:
        try:
            ns = {}
            with f.open("r", encoding="utf-8", errors="replace") as file:
                content = file.read()
            content = re.sub(r'\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])', r'?', content)
            code = compile(content, str(f), "exec")
            exec(code, {}, ns)
            papers = ns.get("paper_txts", [])
            papers = [p.strip() for p in papers if p.strip()]
            texts.extend(papers)
        except: continue
    
    if len(texts) < 5: 
        print(f"  Warning: Not enough texts for {label}")
        continue
    
    # 限制数量保持平衡 (AI 数据可能多，随机采 50 个)
    if len(texts) > 50:
        texts = resample(texts, n_samples=50, random_state=42)
        
    embs = get_embeddings(texts)
    
    if len(embs) < 2:
        print(f"  Warning: Got less than 2 embeddings for {label}, skipping")
        continue
    
    print(f"  Got {len(embs)} embeddings")
    
    # 1. PCD
    dists = compute_pcd_distribution(embs)
    if len(dists) > 0:
        for d in dists:
            pcd_data.append({"Task": label, "PCD": d})
        
        # 2. Bootstrapped Vendi
        vendi_scores = bootstrap_vendi(embs, n_boot=50, sample_size=int(len(embs)*0.8))
        if len(vendi_scores) > 0:
            for v in vendi_scores:
                vendi_boot_data.append({"Task": label, "Vendi": v})
    else:
        print(f"  Warning: No PCD values computed for {label}")

df_pcd = pd.DataFrame(pcd_data)
df_vendi = pd.DataFrame(vendi_boot_data)

# Check if we have data to plot
if df_pcd.empty or df_vendi.empty:
    print("\nError: No data to plot. All tasks were skipped or failed.")
    print(f"PCD data: {len(df_pcd)} rows")
    print(f"Vendi data: {len(df_vendi)} rows")
    exit(1)

print(f"\nData summary:")
print(f"  PCD data: {len(df_pcd)} rows, {df_pcd['Task'].nunique()} tasks")
print(f"  Vendi data: {len(df_vendi)} rows, {df_vendi['Task'].nunique()} tasks")

# --- 4. 绘图 (High-Tier Publication Quality) ---

sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

# Filter ORDER to only include tasks that exist in df
available_tasks_pcd = set(df_pcd["Task"].unique())
available_tasks_vendi = set(df_vendi["Task"].unique())
plot_order = [task for task in ORDER if task in available_tasks_pcd and task in available_tasks_vendi]
plot_palette = {task: PALETTE[task] for task in plot_order if task in PALETTE}

if not plot_order:
    print("Error: No tasks available for plotting after filtering")
    exit(1)

# Plot A: Geometric Constraint Spectrum (Violin)
sns.violinplot(
    data=df_pcd,
    x="Task",
    y="PCD",
    order=plot_order,
    palette=plot_palette,
    ax=axes[0],
    inner="quartile",
    linewidth=1.2,
    alpha=0.9,
    cut=0
)
axes[0].set_title("Semantic Dispersion", fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel("Pairwise Distance", fontsize=13, fontweight='bold')
axes[0].set_xlabel("")
axes[0].set_ylim(0, 0.35)
axes[0].tick_params(axis='x', rotation=10, labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)

# 标注 Convergence / Divergence
# if len(plot_order) > 0 and plot_order[0] in PALETTE:
#     axes[0].text(0, 0.32, "Natural Convergence\n(Truth Seeking)", ha='center', color=PALETTE[plot_order[0]], fontweight='bold', fontsize=10)
# if len(plot_order) > 3 and plot_order[-1] in PALETTE:
#     axes[0].text(len(plot_order)-1, 0.32, "Combinatorial Explosion\n(High Entropy)", ha='center', color=PALETTE[plot_order[-1]], fontweight='bold', fontsize=10)

# Plot B: Effective Diversity Capacity (Boxplot)
sns.boxplot(
    data=df_vendi,
    x="Task",
    y="Vendi",
    order=plot_order,
    palette=plot_palette,
    ax=axes[1],
    width=0.5,
    showfliers=False,
    linewidth=1.2
)
# 连接均值线
means = df_vendi.groupby("Task")["Vendi"].mean().reindex(plot_order)
axes[1].plot(range(len(plot_order)), means.values, color='gray', linestyle='--', marker='o', alpha=0.5, markersize=5)

axes[1].set_title("Effective Diversity", fontsize=14, fontweight='bold', pad=15)
axes[1].set_ylabel("Vendi Score", fontsize=13, fontweight='bold')
axes[1].set_xlabel("")
axes[1].tick_params(axis='x', rotation=10, labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)

# 高亮 AI 区域 (Target Domain) - 如果AI Research在plot_order中
target_task = "AI Research"
if target_task in plot_order:
    target_idx = plot_order.index(target_task)
    for ax in axes:
        ax.axvspan(target_idx - 0.5, target_idx + 0.5, color=PALETTE[target_task], alpha=0.08, zorder=0, lw=0)
        # 在底部添加标注
        label_y = ax.get_ylim()[0] 
        # ax.text(target_idx, label_y, "ICLR Topic", 
        #         ha='center', va='bottom', color=PALETTE[target_task], fontweight='bold', fontsize=9, 
        #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

plt.tight_layout()
plt.savefig("intrinsic_task_spectrum_iclrnew.png", dpi=300, bbox_inches='tight')
print("Saved intrinsic_task_spectrum_iclrnew.png")
plt.show()