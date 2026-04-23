import os
import re
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.patches import FancyArrowPatch

# --- 1. 配置 ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 定义路径 (根据你提供的信息)
PATHS = {
    # --- 主角组 (The Extremes) ---
    "GPT-5.1 + Recursive (Interdisciplinary)": "./data/sec_models/gpt5_1_Interdisciplinary_recursive",
    "GPT-5.1 + NGT (Interdisciplinary)": "./data/sec_models/gpt5_1_Interdisciplinary_ngt",
    
    "o1-mini + Recursive (Horizontal)": "./data/sec_models/o1_mini_horizontal_recursive",
    "o1-mini + Subgroup (Horizontal)": "./data/sec_models/o1_mini_horizontal_subgroup",

    # --- 参照组 (The Baseline: DeepSeek V3) ---
    # 利用 Naive 组展示标准反应
    "DSV3 + Recursive (Naive)": "./data/sec_models/dsv3_naive_recursive", 
    "DSV3 + NGT (Naive)": "./data/sec_models/dsv3_naive_ngt",
    "DSV3 + Subgroup (Naive)": "./data/sec_models/dsv3_naive_subgroup",
    
    # 额外的数据点 (作为背景点缀)
    "DSV3 + Recursive (Inter)": "./data/sec_models/dsv3_Interdisciplinary_recursive",
    "DSV3 + Recursive (Horizontal)": "./data/sec_models/dsv3_horizontal_recursive"
}

# 颜色定义
PALETTE = {
    "GPT-5.1": "#d62728",   # Red
    "o1-mini": "#1f77b4", # Blue
    "DSV3": "gray"        # Gray (Baseline)
}

# 缓存目录
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# --- 2. 计算逻辑 (带缓存) ---
def get_batch_embeddings(texts, cache_key=None):
    """
    获取embeddings，支持缓存和批处理（避免token限制）。
    cache_key: 用于生成缓存文件名的唯一标识
    """
    valid = [t.replace("\n", " ") for t in texts if t.strip()]
    if not valid: return []
    
    # 生成缓存文件名
    if cache_key is None:
        # 使用文本内容的hash作为key
        texts_str = "|||".join(valid)
        cache_key = hashlib.md5(texts_str.encode()).hexdigest()
    
    cache_file = CACHE_DIR / f"embeddings_{cache_key}.pkl"
    
    # 尝试从缓存加载
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, dict):
                cached_texts_hash = cached_data.get('texts_hash')
                cached_emb = cached_data.get('embeddings')
                current_texts_hash = hash(tuple(valid))
                if cached_texts_hash == current_texts_hash and cached_emb is not None:
                    if len(cached_emb) == len(valid):
                        print(f"    [Cache Hit] Loaded {len(valid)} embeddings from cache")
                        return cached_emb
            elif isinstance(cached_data, list):
                if len(cached_data) == len(valid):
                    print(f"    [Cache Hit] Loaded {len(valid)} embeddings from cache (old format)")
                    return cached_data
        except Exception as e:
            print(f"    [Cache Miss] Failed to load cache: {e}")
    
    # 缓存未命中，调用API（带批处理）
    print(f"    [API Call] Fetching embeddings for {len(valid)} texts...")
    
    # text-embedding-3-large 模型限制：最大8192 tokens per input
    MAX_TOKENS_PER_INPUT = 8000  # 模型限制（留点余量）
    MAX_TOKENS_PER_BATCH = 200000  # 批次总token限制（API限制）
    CHUNK_SIZE = 30000  # 单个chunk的大小（字符），约7500 tokens，留余量
    
    def count_tokens(text):
        """快速token估算：1 token ≈ 4 chars"""
        return len(text) // 4
    
    def embed_long_text_with_chunking(text, max_retries=3):
        """对超长文本进行chunking并取平均embedding"""
        text = text.replace("\n", " ").strip()
        if not text:
            return np.zeros(3072)
        
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        chunk_embs = []
        
        for idx, chunk in enumerate(chunks):
            for attempt in range(max_retries):
                try:
                    resp = client.embeddings.create(input=[chunk], model="text-embedding-3-large")
                    chunk_embs.append(np.array(resp.data[0].embedding))
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        print(f"    Warning: Failed to get embedding for chunk {idx+1}/{len(chunks)}: {e}")
                        chunk_embs.append(np.zeros(3072))
                        break
        
        if not chunk_embs:
            return np.zeros(3072)
        
        avg_emb = np.mean(chunk_embs, axis=0)
        # 重新归一化
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm
        return avg_emb
    
    def process_batch(batch, batch_num, processed_count, total_count, max_retries=3):
        """处理一个批次，确保每个文本不超过模型限制"""
        # 检查批次中每个文本的长度，分离超长文本
        safe_batch = []
        long_text_embs = []
        for text in batch:
            text_tokens = count_tokens(text)
            if text_tokens > MAX_TOKENS_PER_INPUT:
                # 如果批次中的文本也超长，单独处理
                print(f"    Warning: Text in batch {batch_num} has ~{text_tokens} tokens, processing separately with chunking...")
                long_emb = embed_long_text_with_chunking(text)
                long_text_embs.append(long_emb)
                processed_count += 1
            else:
                safe_batch.append(text)
        
        # 如果有超长文本，先返回它们的embeddings
        if not safe_batch:
            # 如果批次中所有文本都需要单独处理，直接返回
            return long_text_embs, processed_count
        
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(input=safe_batch, model="text-embedding-3-large")
                embeddings = [np.array(d.embedding) for d in resp.data]
                processed_count += len(safe_batch)
                print(f"    Processed batch {batch_num}: {processed_count}/{total_count} texts")
                # 合并超长文本的embeddings和批次embeddings
                return long_text_embs + embeddings, processed_count
            except Exception as e:
                error_msg = str(e)
                # 检查是否是单个文本太长导致的错误
                if "maximum context length" in str(e) or "8192" in str(e):
                    # 如果是因为文本太长，尝试逐个处理
                    print(f"    Batch {batch_num} contains texts that are too long, processing individually...")
                    individual_embs = []
                    for text in safe_batch:
                        text_tokens = count_tokens(text)
                        if text_tokens > MAX_TOKENS_PER_INPUT:
                            individual_embs.append(embed_long_text_with_chunking(text))
                        else:
                            try:
                                resp = client.embeddings.create(input=[text], model="text-embedding-3-large")
                                individual_embs.append(np.array(resp.data[0].embedding))
                            except Exception as e2:
                                print(f"    Warning: Failed to process individual text: {e2}")
                                individual_embs.append(np.zeros(3072))
                    processed_count += len(safe_batch)
                    print(f"    Processed batch {batch_num} individually: {processed_count}/{total_count} texts")
                    return individual_embs, processed_count
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"    Batch {batch_num} failed (attempt {attempt+1}/{max_retries}): {error_msg}")
                    print(f"    Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"    [Error] Batch {batch_num} failed after {max_retries} attempts: {error_msg}")
                    raise
    
    try:
        all_embeddings = []
        current_batch = []
        current_tokens = 0
        total_texts = len(valid)
        processed_texts = 0
        batch_num = 0
        
        for i, text in enumerate(valid):
            text_tokens = count_tokens(text)
            
            # 如果单个文本超过模型限制（8192 tokens），使用chunking
            if text_tokens > MAX_TOKENS_PER_INPUT:
                print(f"    Warning: Text {i+1} has ~{text_tokens} tokens (exceeds {MAX_TOKENS_PER_INPUT}), using chunking...")
                long_emb = embed_long_text_with_chunking(text)
                all_embeddings.append(long_emb)
                processed_texts += 1
                continue
            
            # 确保单个文本不超过模型限制（即使估算可能不准确，也要保守处理）
            # 如果文本看起来接近限制，也使用chunking
            if text_tokens > MAX_TOKENS_PER_INPUT or len(text) > CHUNK_SIZE:
                print(f"    Warning: Text {i+1} is too long (~{text_tokens} tokens, {len(text)} chars), using chunking...")
                long_emb = embed_long_text_with_chunking(text)
                all_embeddings.append(long_emb)
                processed_texts += 1
                continue
            
            # 如果添加当前文本会超过批次限制，先处理当前批次
            if current_batch and (current_tokens + text_tokens > MAX_TOKENS_PER_BATCH):
                batch_num += 1
                batch_embs, processed_texts = process_batch(current_batch, batch_num, processed_texts, total_texts)
                all_embeddings.extend(batch_embs)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(text)
            current_tokens += text_tokens
        
        # 处理剩余的批次
        if current_batch:
            batch_num += 1
            batch_embs, processed_texts = process_batch(current_batch, batch_num, processed_texts, total_texts)
            all_embeddings.extend(batch_embs)
        
        # 保存到缓存
        try:
            cache_data = {
                'texts_hash': hash(tuple(valid)),
                'embeddings': all_embeddings
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"    [Cache Saved] Saved {len(all_embeddings)} embeddings to cache")
        except Exception as e:
            print(f"    [Warning] Failed to save cache: {e}")
        
        return all_embeddings
    except Exception as e:
        print(f"    [Error] API call failed: {e}")
        return []

def compute_vendi(embs):
    if not embs: return 0.0
    X = np.array(embs)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    evals = np.linalg.eigvalsh(X @ X.T) / X.shape[0]
    return np.exp(-np.sum(evals[evals > 1e-10] * np.log(evals[evals > 1e-10])))

def compute_density(path_str, label=""):
    # 检查是否有缓存的结果
    cache_key = hashlib.md5(str(path_str).encode()).hexdigest()
    cache_file = CACHE_DIR / f"density_{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, dict) and 'density' in cached_data:
                # 检查路径是否改变
                if cached_data.get('path') == str(path_str):
                    print(f"    [Cache Hit] Loaded density from cache: {cached_data['density']:.4f}")
                    return cached_data['density']
        except Exception as e:
            print(f"    [Cache Miss] Failed to load density cache: {e}")
    
    # 鲁棒的路径寻找逻辑
    p = Path(path_str)
    # 尝试多种可能的 outputs 位置
    candidates = [
        p / "outputs",
        p.parent / "outputs",
        Path(str(p).replace("extracted_proposals", "outputs")), # 假设平行结构
        Path(str(p).replace("extracted_proposals", "sec_models").replace("dsv3", "outputs")) # 乱猜
    ]
    
    # 如果路径里本身就是实验根目录 (包含outputs子目录)
    if (p / "outputs").exists():
        chat_dir = p / "outputs"
    # 如果路径是 extracted_proposals 里的子文件夹
    elif "extracted_proposals" in str(p):
        # 尝试回溯去找 outputs
        # 假设 dsv3_naive_recursive 对应的 output 在 ../../outputs/dsv3_naive_recursive 
        # 这里需要你根据实际情况微调，如果 outputs 都在同一个大文件夹下
        # 暂时用 "当前文件夹下的txt" 作为 fallback，假设你把 log 也放进去了
        chat_dir = p 
    else:
        chat_dir = p

    files = list(chat_dir.glob("*.txt"))
    if not files: 
        print(f"    [Warning] No .txt files found in {chat_dir}")
        result = np.nan
    else:
        print(f"    Found {len(files)} files in {chat_dir}")
        densities = []
        skipped_no_turns = 0
        skipped_no_embs = 0
        processed = 0
        
        for i, f in enumerate(files):
            try:
                with open(f, 'r', errors='replace') as F: content = F.read()
                # 支持两种格式：Participant \d+ 和 PhD Student [A-Z]
                pattern1 = re.findall(r'(Participant \d+):(.*?)(?=Participant \d+:|$)', content, re.DOTALL)
                pattern2 = re.findall(r'(PhD Student [A-Z]):(.*?)(?=PhD Student [A-Z]:|$)', content, re.DOTALL)
                if pattern1:
                    turns = pattern1
                elif pattern2:
                    turns = pattern2
                else:
                    # 如果都不匹配，尝试更宽松的模式
                    turns = []
                texts = [t[1].strip() for t in turns]
                if len(texts) < 2:
                    skipped_no_turns += 1
                    continue
                # 使用文件路径和内容生成cache key
                file_cache_key = hashlib.md5((str(f) + content).encode()).hexdigest()
                embs = get_batch_embeddings(texts, cache_key=f"{label}_density_{i}_{file_cache_key}")
                if len(embs) < 2:
                    skipped_no_embs += 1
                    continue
                sim = cosine_similarity(embs)
                mask = np.triu_indices_from(sim, k=1)
                densities.append(np.mean(sim[mask]))
                processed += 1
            except Exception as e:
                print(f"    [Error] Failed to process file {f.name}: {e}")
                continue
        
        if processed == 0:
            print(f"    [Warning] No files were successfully processed (skipped {skipped_no_turns} with <2 turns, {skipped_no_embs} with <2 embs)")
        else:
            print(f"    Processed {processed} files successfully (skipped {skipped_no_turns} with <2 turns, {skipped_no_embs} with <2 embs)")
        
        result = np.mean(densities) if densities else np.nan
    
    # 保存结果到缓存
    try:
        cache_data = {
            'path': str(path_str),
            'density': result
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"    [Warning] Failed to save density cache: {e}")
    
    return result

def compute_final_vendi(path_str, label=""):
    # 检查是否有缓存的结果
    cache_key = hashlib.md5(str(path_str).encode()).hexdigest()
    cache_file = CACHE_DIR / f"vendi_{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, dict) and 'vendi' in cached_data:
                # 检查路径是否改变
                if cached_data.get('path') == str(path_str):
                    print(f"    [Cache Hit] Loaded vendi from cache: {cached_data['vendi']:.4f}")
                    return cached_data['vendi']
        except Exception as e:
            print(f"    [Cache Miss] Failed to load vendi cache: {e}")
    
    p = Path(path_str)
    # 如果路径本身就是 extracted_proposals 目录（路径名包含 extracted_proposals）
    if "extracted_proposals" in str(p) and p.name not in ["extracted_proposals"]:
        # 如果路径是 extracted_proposals 下的子目录，直接使用
        prop_dir = p
    elif (p / "extracted_proposals").exists():
        # 如果路径下有 extracted_proposals 子目录，使用它
        prop_dir = p / "extracted_proposals"
    else:
        # 否则假设路径本身就是提案目录
        prop_dir = p
        
    if not prop_dir.exists(): 
        result = np.nan
    else:
        files = list(prop_dir.glob("*.txt"))
        texts = []
        
        # 解析Python格式的文件（包含paper_txts列表）
        def safe_exec(content: str) -> dict:
            """安全执行Python代码，提取paper_txts"""
            # 防止文本里出现裸 \uXXXX 被 Python 解析为转义
            content = content.replace('\\u', '\\\\u')
            ns = {}
            try:
                code = compile(content, '<string>', 'exec')
                exec(code, {}, ns)
            except Exception as e:
                print(f"    [Warning] Failed to parse file: {e}")
            return ns
        
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8', errors='replace') as F:
                    content = F.read()
                
                # 检查是否是Python格式（包含paper_txts）
                if 'paper_txts' in content:
                    ns = safe_exec(content)
                    papers = ns.get("paper_txts", [])
                    # 提取每个提案文本
                    for p in papers:
                        text = p.strip()
                        if text:
                            texts.append(text)
                else:
                    # 普通文本文件，直接使用
                    if content.strip():
                        texts.append(content.strip())
            except Exception as e:
                print(f"    [Warning] Failed to read file {f.name}: {e}")
                continue
        
        if not texts:
            print(f"    [Warning] No texts extracted from {prop_dir}")
            result = np.nan
        else:
            print(f"    Extracted {len(texts)} proposal texts")
            # 使用路径和标签生成cache key
            texts_hash = hashlib.md5(("|||".join(texts)).encode()).hexdigest()
            embs = get_batch_embeddings(texts, cache_key=f"{label}_vendi_{texts_hash}")
            if len(embs) < 2:
                print(f"    [Warning] Got {len(embs)} embeddings (need >=2), returning NaN")
                result = np.nan
            else:
                result = compute_vendi(embs)
    
    # 保存结果到缓存
    try:
        cache_data = {
            'path': str(path_str),
            'vendi': result
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"    [Warning] Failed to save vendi cache: {e}")
    
    return result

# --- 3. 运行计算 ---
data = []
print("Computing metrics across all settings...")

for label, path in PATHS.items():
    print(f"Processing: {label}")
    if "GPT-5.1" in label: fam = "GPT-5.1"
    elif "o1-mini" in label: fam = "o1-mini"
    else: fam = "DSV3"
    
    v = compute_final_vendi(path, label=label)
    d = compute_density(path, label=label) # 注意：如果找不到 outputs，这里会返回 NaN
    
    # 打印调试信息
    if np.isnan(v):
        print(f"  Warning: Vendi is NaN for {label}")
    if np.isnan(d):
        print(f"  Warning: Density is NaN for {label} (no chat logs found or empty files)")
    
    # 只有当两个值都有效时才添加到数据中
    if not np.isnan(v) and not np.isnan(d):
        data.append({"Label": label, "Family": fam, "Vendi": v, "Density": d})
        print(f"  ✓ Successfully computed: Vendi={v:.6f}, Density={d:.6f}")
    else:
        print(f"  ✗ Skipping {label} due to missing data (Vendi: {v}, Density: {d})")

df = pd.DataFrame(data)
print(df)

# --- 4. 绘图 (Contextual Landscape) ---
if df.empty:
    print("No valid data found.")
    exit()

# 为绘图创建一个副本，并移除不需要在图中显示的点
df_plot = df[df["Label"] != "DSV3 + Recursive (Inter)"].copy()
if df_plot.empty:
    print("No valid data left for plotting after filtering.")
    exit()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 5))
ax = plt.gca()

# 画点 - 为每个label分配独特的颜色和标记组合
# 定义丰富的颜色调色板
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 为每个label分配独特的颜色和标记
label_styles = {
    "GPT-5.1 + Recursive (Interdisciplinary)": {"color": "#d62728", "marker": "s"},  # 红色方块
    "GPT-5.1 + NGT (Interdisciplinary)": {"color": "#ff7f0e", "marker": "s"},  # 橙色方块
    "o1-mini + Recursive (Horizontal)": {"color": "#1f77b4", "marker": "o"},  # 蓝色圆圈
    "o1-mini + Subgroup (Horizontal)": {"color": "#2ca02c", "marker": "o"},  # 绿色圆圈
    "DSV3 + Recursive (Naive)": {"color": "#7f7f7f", "marker": "X"},  # 灰色X
    "DSV3 + NGT (Naive)": {"color": "#9467bd", "marker": "^"},  # 紫色三角
    "DSV3 + Subgroup (Naive)": {"color": "#8c564b", "marker": "v"},  # 棕色倒三角
    "DSV3 + Recursive (Inter)": {"color": "#e377c2", "marker": "D"},  # 粉色菱形
    "DSV3 + Recursive (Horizontal)": {"color": "#bcbd22", "marker": "p"},  # 黄绿色五角形
}

# 为每个数据点单独绘制，使用独特的颜色和标记
for idx, row in df_plot.iterrows():
    label = row["Label"]
    style = label_styles.get(label, {"color": "black", "marker": "o"})
    plt.scatter(row["Density"], row["Vendi"], 
               c=style["color"], marker=style["marker"],
               s=150, zorder=5, label=label, edgecolors='black', linewidths=1)

# 动态添加箭头
def add_arrow(start_key, end_key, color, label=None, alpha=0.6):
    try:
        p1 = df_plot[df_plot["Label"].str.contains(start_key, regex=False)].iloc[0]
        p2 = df_plot[df_plot["Label"].str.contains(end_key, regex=False)].iloc[0]
        
        arrow = FancyArrowPatch(
            (p1["Density"], p1["Vendi"]), (p2["Density"], p2["Vendi"]),
            arrowstyle="->", mutation_scale=15, color=color, linewidth=1.5, 
            linestyle='--', zorder=3, alpha=alpha
        )
        ax.add_patch(arrow)
    except (IndexError, KeyError): 
        pass

# 自动生成所有合作关系箭头 - 显示完整的model*topology*group_dynamic网络
# 定义所有可能的连接关系

# 1. 同一模型、同一group_dynamic，不同topology之间的连接（主要干预效果）
# 箭头颜色使用起点和终点颜色的混合，或使用起点的颜色
def get_arrow_color(start_label, end_label):
    """获取箭头颜色，使用起点颜色"""
    start_style = label_styles.get(start_label, {"color": "gray"})
    return start_style["color"]

connections = [
    # GPT-5.1 (Interdisciplinary)
    ("GPT-5.1 + Recursive (Interdisciplinary)", "GPT-5.1 + NGT (Interdisciplinary)", 
     get_arrow_color("GPT-5.1 + Recursive (Interdisciplinary)", "GPT-5.1 + NGT (Interdisciplinary)"), 
     "Jailbreak", 0.8),
    
    # o1-mini (Horizontal)
    ("o1-mini + Recursive (Horizontal)", "o1-mini + Subgroup (Horizontal)", 
     get_arrow_color("o1-mini + Recursive (Horizontal)", "o1-mini + Subgroup (Horizontal)"), 
     "Anti-Collapse", 0.8),
    
    # DSV3 (Naive) - 完整的topology网络
    ("DSV3 + Recursive (Naive)", "DSV3 + NGT (Naive)", 
     get_arrow_color("DSV3 + Recursive (Naive)", "DSV3 + NGT (Naive)"), None, 0.6),
    ("DSV3 + Recursive (Naive)", "DSV3 + Subgroup (Naive)", 
     get_arrow_color("DSV3 + Recursive (Naive)", "DSV3 + Subgroup (Naive)"), None, 0.6),
    ("DSV3 + NGT (Naive)", "DSV3 + Subgroup (Naive)", 
     get_arrow_color("DSV3 + NGT (Naive)", "DSV3 + Subgroup (Naive)"), None, 0.6),
]

# 2. 同一模型、同一topology，不同group_dynamic之间的连接（group_dynamic效应）
group_dynamic_connections = [
    # DSV3 Recursive: 不同group_dynamic
    ("DSV3 + Recursive (Naive)", "DSV3 + Recursive (Inter)", 
     get_arrow_color("DSV3 + Recursive (Naive)", "DSV3 + Recursive (Inter)"), None, 0.4),
    ("DSV3 + Recursive (Naive)", "DSV3 + Recursive (Horizontal)", 
     get_arrow_color("DSV3 + Recursive (Naive)", "DSV3 + Recursive (Horizontal)"), None, 0.4),
    ("DSV3 + Recursive (Inter)", "DSV3 + Recursive (Horizontal)", 
     get_arrow_color("DSV3 + Recursive (Inter)", "DSV3 + Recursive (Horizontal)"), None, 0.4),
]

# 绘制所有连接
for start, end, color, label, alpha in connections:
    add_arrow(start, end, color, label, alpha)

for start, end, color, label, alpha in group_dynamic_connections:
    add_arrow(start, end, color, label, alpha)

# 装饰
plt.xlabel("Interaction Density (Consensus Strength)", fontsize=11, fontweight='bold')
plt.ylabel("Semantic Diversity (Vendi Score)", fontsize=11, fontweight='bold')

# Auto Zoom
x_min, x_max = df_plot["Density"].min(), df_plot["Density"].max()
y_min, y_max = df_plot["Vendi"].min(), df_plot["Vendi"].max()
margin_x = (x_max - x_min) * 0.15
margin_y = (y_max - y_min) * 0.15
plt.xlim(x_min - margin_x, x_max + margin_x)
plt.ylim(y_min - margin_y, y_max + margin_y)

# 创建详细的legend，显示所有label，放在左下角，并缩小图例标记尺寸
plt.legend(loc='lower left',
           bbox_to_anchor=(0.02, 0.02),
           title="Model × Topology × Group Dynamic",
           fontsize=8,
           framealpha=0.9,
           markerscale=0.7)
plt.tight_layout()
_out_base = "./outputs/interaction_landscape_with_naive_fixed"
plt.savefig(_out_base + ".png", dpi=300)
plt.savefig(_out_base + ".pdf", bbox_inches="tight")
print(f"Saved plot to {_out_base}.png and {_out_base}.pdf")
plt.show()

# import os
# import re
# import time
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from pathlib import Path
# from openai import OpenAI
# from sklearn.metrics.pairwise import cosine_similarity
# from matplotlib.patches import FancyArrowPatch
# import concurrent.futures
# import threading

# # --- 1. 配置 ---
# API_KEY = os.environ.get("OPENAI_API_KEY", "")
# BASE_URL = "https://api.openai.com/v1"

# client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# # 并行设置
# MAX_WORKERS = 10  # 可以根据API限制调整

# # 定义四种设定路径
# # 请确保路径指向包含 extracted_proposals 和 outputs 的父目录，或者具体调整下方逻辑
# PATHS = {
#     "GPT-5 + Recursive\n(Rigid Baseline)": "./data/sec_models/gpt5_1_Interdisciplinary_recursive",
#     "GPT-5 + NGT\n(Process Intervention)": "./data/sec_models/gpt5_1_Interdisciplinary_ngt",
#     "o1-mini + Horizontal\n(Wild Baseline)": "./data/sec_models/o1_mini_horizontal_recursive",
#     "o1-mini + Subgroup\n(Process Intervention)": "./data/sec_models/o1_mini_horizontal_subgroup"
# }

# # 颜色映射
# COLORS = {
#     "GPT-5 + Recursive\n(Rigid Baseline)": "#d62728", # Red (Danger/Rigid)
#     "GPT-5 + NGT\n(Process Intervention)": "#ff7f0e", # Orange (Fixed)
#     "o1-mini + Horizontal\n(Wild Baseline)": "#1f77b4", # Blue (Wild)
#     "o1-mini + Subgroup\n(Process Intervention)": "#2ca02c" # Green (Balanced)
# }

# # --- 2. 核心计算函数 ---

# def get_embedding_safe(text):
#     """
#     科学的 Embedding 获取方式：
#     如果文本超过 Token 限制，进行切片(Chunking)并取平均(Average)，
#     而不是简单截断。
#     """
#     CHUNK_SIZE = 12000  # 约 3000-4000 tokens，留足余量
    
#     # 移除换行，减少噪音
#     text = text.replace("\n", " ")
    
#     if not text.strip():
#         return np.zeros(3072)

#     # 如果文本较短，直接请求
#     if len(text) < CHUNK_SIZE:
#         for _ in range(3):  # 重试机制
#             try:
#                 resp = client.embeddings.create(input=[text], model="text-embedding-3-large")
#                 return np.array(resp.data[0].embedding)
#             except Exception as e:
#                 time.sleep(1)
#         return np.zeros(3072)
    
#     # 如果文本过长，进行切片
#     chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
#     chunk_embeddings = []
    
#     for chunk in chunks:
#         for _ in range(3):
#             try:
#                 resp = client.embeddings.create(input=[chunk], model="text-embedding-3-large")
#                 chunk_embeddings.append(resp.data[0].embedding)
#                 break
#             except Exception:
#                 time.sleep(1)
    
#     if not chunk_embeddings:
#         return np.zeros(3072)
    
#     # 核心步骤：对所有切片的向量取平均值 (Mean Pooling)
#     avg_embedding = np.mean(chunk_embeddings, axis=0)
    
#     # 重新归一化 (Embedding 通常需要是单位向量)
#     norm = np.linalg.norm(avg_embedding)
#     if norm > 0:
#         avg_embedding = avg_embedding / norm
        
#     return avg_embedding

# def get_batch_embeddings(texts):
#     """批量获取 Embeddings，使用完整文本（支持chunking）"""
#     valid_texts = [t.strip() for t in texts if t.strip()]
#     if not valid_texts: 
#         return []
    
#     embeddings = []
#     for text in valid_texts:
#         emb = get_embedding_safe(text)
#         if not np.all(emb == 0):
#             embeddings.append(emb)
    
#     return embeddings

# def compute_vendi_score(embeddings):
#     """计算 Vendi Score (Diversity)"""
#     if not embeddings: return 0.0
#     X = np.array(embeddings)
#     # Normalize
#     X = X / np.linalg.norm(X, axis=1, keepdims=True)
#     # Kernel Matrix
#     K = X @ X.T
#     n = K.shape[0]
#     # Eigenvalues
#     evals = np.linalg.eigvalsh(K) / n
#     evals = evals[evals > 1e-10] # Filter distinct 0
#     # Entropy
#     entropy = -np.sum(evals * np.log(evals))
#     return np.exp(entropy)

# def process_single_file_density(f_path):
#     """处理单个文件，计算其density"""
#     try:
#         with open(f_path, 'r', encoding='utf-8', errors='replace') as f:
#             content = f.read()
        
#         # 提取对话内容：支持两种格式
#         pattern1 = re.compile(r'(Participant \d+):(.*?)(?=Participant \d+:|$)', re.DOTALL)
#         pattern2 = re.compile(r'(PhD Student [A-Z]):(.*?)(?=PhD Student [A-Z]:|$)', re.DOTALL)
        
#         matches1 = pattern1.findall(content)
#         matches2 = pattern2.findall(content)
        
#         if matches1:
#             turns = [m[1].strip() for m in matches1]
#         elif matches2:
#             turns = [m[1].strip() for m in matches2]
#         else:
#             return None
        
#         if len(turns) < 2:
#             return None
        
#         # 计算该场对话的 Embeddings
#         embs = get_batch_embeddings(turns)
#         if len(embs) < 2:
#             return None
        
#         # 计算两两相似度矩阵
#         sim_matrix = cosine_similarity(embs)
#         # 取上三角部分的平均值 (不包含对角线)
#         mask = np.triu_indices_from(sim_matrix, k=1)
#         avg_sim = np.mean(sim_matrix[mask])
#         return avg_sim
#     except Exception as e:
#         return None

# def compute_interaction_density(chat_dir, debug=False):
#     """
#     计算 Interaction Density (Consensus Strength).
#     逻辑：读取 outputs 里的对话，计算所有发言两两之间的平均相似度。
#     相似度越高 = Consensus 越强 (Rigid)。
#     相似度越低 = Interaction 越发散 (Wild)。
#     """
#     path_obj = Path(chat_dir)
#     files = list(path_obj.glob("*.txt"))  # 全量处理所有文件
    
#     if debug:
#         print(f"  Found {len(files)} files in {chat_dir}")
    
#     if len(files) == 0:
#         if debug:
#             print(f"  Warning: No .txt files found in {chat_dir}")
#         return 0.0
    
#     # 并行处理文件
#     densities = []
#     files_processed = 0
#     files_skipped = 0
#     lock = threading.Lock()
    
#     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         future_to_file = {executor.submit(process_single_file_density, f): f for f in files}
        
#         for future in concurrent.futures.as_completed(future_to_file):
#             result = future.result()
#             with lock:
#                 if result is not None:
#                     densities.append(result)
#                     files_processed += 1
#                 else:
#                     files_skipped += 1
#                     if debug and files_skipped <= 3:
#                         f_path = future_to_file[future]
#                         print(f"  Skipped {f_path.name}")
    
#     if debug:
#         print(f"  Processed {files_processed} files, skipped {files_skipped} files")
#         if len(densities) > 0:
#             print(f"  Density range: {min(densities):.4f} - {max(densities):.4f}")
#         else:
#             print(f"  Warning: No valid densities computed!")
#             if len(files) > 0:
#                 with open(files[0], 'r', encoding='utf-8', errors='replace') as f:
#                     sample = f.read()[:500]
#                     print(f"  Sample content from first file:\n{sample}")
        
#     return np.mean(densities) if densities else 0.0

# def process_single_proposal_file(f_path):
#     """处理单个proposal文件"""
#     try:
#         with open(f_path, 'r', encoding='utf-8', errors='replace') as f:
#             text = f.read()
#         return text
#     except Exception:
#         return None

# def load_proposals_get_vendi(prop_dir):
#     """读取最终 proposal 计算 Vendi（并行处理）"""
#     path_obj = Path(prop_dir)
#     files = list(path_obj.glob("*.txt"))  # 全量处理所有文件
    
#     if len(files) == 0:
#         return 0.0
    
#     # 并行读取文件
#     texts = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         future_to_file = {executor.submit(process_single_proposal_file, f): f for f in files}
#         for future in concurrent.futures.as_completed(future_to_file):
#             text = future.result()
#             if text:
#                 texts.append(text)
    
#     if not texts:
#         return 0.0
    
#     # 批量计算embeddings（get_batch_embeddings内部已经是逐个处理，这里可以进一步优化）
#     embs = get_batch_embeddings(texts)
#     return compute_vendi_score(embs)

# # --- 3. 主程序 ---

# results = []

# print("Starting Analysis...")

# for label, base_path in PATHS.items():
#     print(f"Processing {label.splitlines()[0]}...")
    
#     # 1. 计算 Diversity (基于 extracted_proposals)
#     prop_path = Path(base_path) / "extracted_proposals"
#     if not prop_path.exists():
#         print(f"Warning: {prop_path} not found, trying parent.")
#         prop_path = Path(base_path) # Fallback
        
#     vendi = load_proposals_get_vendi(prop_path)
    
#     # 2. 计算 Interaction Density (基于 outputs)
#     chat_path = Path(base_path) / "outputs"
#     if not chat_path.exists():
#         print(f"Warning: {chat_path} not found.")
#         interaction = 0.5 # Default fallback
#     else:
#         # 对 o1-mini 启用调试模式
#         debug_mode = "o1-mini" in label
#         if debug_mode:
#             print(f"  Computing Interaction Density for {label.splitlines()[0]}...")
#         interaction = compute_interaction_density(chat_path, debug=debug_mode)
        
#     results.append({
#         "Setting": label,
#         "Group": "GPT-5 (Rigid)" if "GPT-5" in label else "o1-mini (Wild)",
#         "Vendi Score": vendi,
#         "Interaction Density": interaction # 越高代表越趋同/回音室
#     })

# df = pd.DataFrame(results)
# print("\nResults Computed:")
# print(df)

# # --- 4. 绘图 (Interaction Plot) ---

# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(10, 8))
# ax = plt.gca()

# # 画点
# sns.scatterplot(
#     data=df, 
#     x="Interaction Density", 
#     y="Vendi Score", 
#     hue="Setting", 
#     palette=COLORS, 
#     s=300, 
#     style="Group", # 不同的形状区分模型
#     markers={"GPT-5 (Rigid)": "s", "o1-mini (Wild)": "o"},
#     zorder=5
# )

# # 手动添加"干预轨迹"箭头
# # 直接使用Setting的完整值进行匹配，更可靠
# setting_gpt_recursive = "GPT-5 + Recursive\n(Rigid Baseline)"
# setting_gpt_ngt = "GPT-5 + NGT\n(Process Intervention)"
# setting_o1_horizontal = "o1-mini + Horizontal\n(Wild Baseline)"
# setting_o1_subgroup = "o1-mini + Subgroup\n(Process Intervention)"

# # 1. GPT-5: Recursive -> NGT
# if setting_gpt_recursive in df["Setting"].values and setting_gpt_ngt in df["Setting"].values:
#     p1 = df[df["Setting"] == setting_gpt_recursive].iloc[0]
#     p2 = df[df["Setting"] == setting_gpt_ngt].iloc[0]
    
#     arrow_gpt = FancyArrowPatch(
#         (p1["Interaction Density"], p1["Vendi Score"]), 
#         (p2["Interaction Density"], p2["Vendi Score"]),
#         arrowstyle='->', mutation_scale=20, color='gray', linestyle='--', linewidth=2, zorder=3
#     )
#     ax.add_patch(arrow_gpt)
#     plt.text((p1["Interaction Density"]+p2["Interaction Density"])/2, (p1["Vendi Score"]+p2["Vendi Score"])/2 + 0.1, 
#              "NGT Breaks Alignment", fontsize=10, color='gray', ha='center')
# else:
#     print(f"Warning: Could not find GPT-5 points for arrow.")

# # 2. o1-mini: Horizontal -> Subgroup
# if setting_o1_horizontal in df["Setting"].values and setting_o1_subgroup in df["Setting"].values:
#     p3 = df[df["Setting"] == setting_o1_horizontal].iloc[0]
#     p4 = df[df["Setting"] == setting_o1_subgroup].iloc[0]
    
#     arrow_o1 = FancyArrowPatch(
#         (p3["Interaction Density"], p3["Vendi Score"]), 
#         (p4["Interaction Density"], p4["Vendi Score"]),
#         arrowstyle='->', mutation_scale=20, color='gray', linestyle='--', linewidth=2, zorder=3
#     )
#     ax.add_patch(arrow_o1)
#     plt.text((p3["Interaction Density"]+p4["Interaction Density"])/2, (p3["Vendi Score"]+p4["Vendi Score"])/2 - 0.2, 
#              "Subgroups Add Structure", fontsize=10, color='gray', ha='center')
# else:
#     print(f"Warning: Could not find o1-mini points for arrow.")

# # 装饰
# plt.title("Interaction Landscape: Model Alignment vs. Social Structure", fontsize=15, fontweight='bold', pad=20)
# plt.xlabel("Interaction Density (Consensus Strength)\nLow = Divergent, High = Echo Chamber", fontsize=12)
# plt.ylabel("Semantic Diversity (Vendi Score)", fontsize=12)

# # 划分区域背景
# plt.axvline(x=df["Interaction Density"].mean(), color='gray', linestyle=':', alpha=0.3)
# plt.axhline(y=df["Vendi Score"].mean(), color='gray', linestyle=':', alpha=0.3)

# # 标注象限含义
# x_lims = ax.get_xlim()
# y_lims = ax.get_ylim()
# plt.text(x_lims[1]*0.95, y_lims[0]*1.05, "Echo Chamber Zone\n(Rigid + Recursive)", color='red', ha='right', alpha=0.5)
# plt.text(x_lims[0]*1.05, y_lims[1]*0.95, "Ideal Innovation Zone\n(High Diversity)", color='green', ha='left', alpha=0.5)

# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.tight_layout()

# output_path = "interaction_diversity_analysis.png"
# plt.savefig(output_path, dpi=300)
# print(f"Plot saved to {output_path}")
# plt.show()