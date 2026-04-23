import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.patches import FancyArrowPatch

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

DATA_DIRS = {
    "Standard (Recursive)": "./data/extracted_proposals/extracted_proposals_recursive",
    "NGT": "./data/extracted_proposals/extracted_proposals_ngt",
    "Subgroups": "./data/extracted_proposals/extracted_proposals_subgroup"
}
# MAX_FILES = 8 # 已改为全量处理，不再限制文件数量

# --- 核心函数 ---

def get_embedding(text):
    text = text.replace("\n", " ")[:8000]
    try:
        resp = client.embeddings.create(input=[text], model="text-embedding-3-large")
        return resp.data[0].embedding
    except:
        return np.zeros(3072)

def judge_strict_critique(prev_context, current_text):
    """
    [严厉版判官]
    默认给低分。只有真正的“对抗性”观点才能拿高分。
    """
    prompt = f"""
    Compare Speaker B's statement to Speaker A's context.
    
    Speaker A: "{prev_context[-400:]}..."
    Speaker B: "{current_text}"
    
    Task: Rate the level of DISAGREEMENT and NOVELTY on a scale of 1-10.
    
    Strict Scoring Rubric:
    - 1-3 (Echo/Safe): Agrees, repeats, or adds minor "fluff" (e.g., "I agree", "Building on that", "Crucially...").
    - 4-6 (Additive): Adds specific details but stays within A's logical framework. No conflict.
    - 7-8 (Refinement): Points out a gap, limitation, or edge case in A's logic. (Soft critique).
    - 9-10 (Disruption): Fundamentally challenges A's premise, proposes a competing paradigm, or steers the topic to a completely new dimension.
    
    Be harsh. Most cooperative dialogues should score 3-5. Only rate >7 for real conflict.
    
    Output ONLY the integer score.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        score = int(re.search(r'\d+', response.choices[0].message.content).group())
        return min(max(score, 1), 10)
    except:
        return 4 # Default to additive

def parse_chat_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    # 提取 Participant X: ... 的内容
    pattern = re.compile(r'(Participant \d+):(.*?)(?=Participant \d+:|$)', re.DOTALL)
    matches = pattern.findall(content)
    return [{"speaker": s.strip(), "text": t.strip()} for s, t in matches]

# --- 主循环 ---
all_data = []

print("Running Divergence Analysis...")

for mode_name, dir_path in DATA_DIRS.items():
    path_obj = Path(dir_path)
    files = list(path_obj.glob("*.txt"))  # 全量处理所有文件
    print(f"Processing {len(files)} files for {mode_name}...")
    
    for file_path in files:
        turns = parse_chat_file(file_path)
        if not turns: continue
        
        # Anchor: 第一轮发言
        anchor_vec = np.array(get_embedding(turns[0]['text'])).reshape(1, -1)
        
        for i, turn in enumerate(turns):
            if i == 0: continue # 跳过 Anchor
            
            # X轴: 语义离散度 (1 - Cosine Similarity)
            vec = np.array(get_embedding(turn['text'])).reshape(1, -1)
            sim = cosine_similarity(vec, anchor_vec)[0][0]
            divergence = 1.0 - sim 
            
            # Y轴: 严厉批判分
            prev_text = turns[i-1]['text']
            raw_score = judge_strict_critique(prev_text, turn['text'])
            norm_score = (raw_score - 1) / 9.0 # 0-1 Scale
            
            all_data.append({
                "Mode": mode_name,
                "Turn": i,
                "Divergence": divergence,
                "Critique": norm_score
            })

df = pd.DataFrame(all_data)

# --- 绘图 (改进版：显示误差和原始数据) ---
if df.empty:
    print("No data.")
    exit()

# 只看前 5 轮 (避免后续噪音)
df = df[df["Turn"] <= 5]

# 聚合取均值和标准差
df_traj = df.groupby(["Mode", "Turn"]).agg({
    "Divergence": ["mean", "std"],
    "Critique": ["mean", "std"]
}).reset_index()
df_traj.columns = ["Mode", "Turn", "Divergence_mean", "Divergence_std", "Critique_mean", "Critique_std"]

# 计算标准误 (SEM = std / sqrt(n))
df_counts = df.groupby(["Mode", "Turn"]).size().reset_index(name="n")
df_traj = df_traj.merge(df_counts, on=["Mode", "Turn"])
df_traj["Divergence_sem"] = df_traj["Divergence_std"] / np.sqrt(df_traj["n"])
df_traj["Critique_sem"] = df_traj["Critique_std"] / np.sqrt(df_traj["n"])

# 创建图形：主图 + 统计摘要
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.3)

# === 主图：轨迹图 ===
ax1 = fig.add_subplot(gs[0, 0])

colors = {"Standard (Recursive)": "#d62728", "NGT": "#2ca02c", "Subgroups": "#17becf"}
markers = {"Standard (Recursive)": "o", "NGT": "s", "Subgroups": "^"}

# 先画原始数据点（半透明，显示分布）
for mode in colors.keys():
    if mode not in df["Mode"].values: continue
    subset_raw = df[df["Mode"] == mode]
    ax1.scatter(subset_raw["Divergence"], subset_raw["Critique"], 
                c=colors[mode], alpha=0.15, s=20, zorder=1, label=None)

# 自动确定坐标轴范围 (基于原始数据)
x_min, x_max = df["Divergence"].min(), df["Divergence"].max()
y_min, y_max = df["Critique"].min(), df["Critique"].max()
x_margin = (x_max - x_min) * 0.15
y_margin = (y_max - y_min) * 0.15
ax1.set_xlim(max(0, x_min - x_margin), min(1, x_max + x_margin))
ax1.set_ylim(max(0, y_min - y_margin), min(1, y_max + y_margin))

# 画均值和误差棒
for mode in colors.keys():
    if mode not in df_traj["Mode"].values: continue
    
    subset = df_traj[df_traj["Mode"] == mode].sort_values("Turn")
    x_mean = subset["Divergence_mean"].values
    y_mean = subset["Critique_mean"].values
    x_err = subset["Divergence_sem"].values * 1.96  # 95% CI
    y_err = subset["Critique_sem"].values * 1.96
    
    # 画轨迹线
    ax1.plot(x_mean, y_mean, c=colors[mode], alpha=0.7, linewidth=2.5, zorder=3, label=mode)
    
    # 画误差椭圆（简化版：用误差棒）
    for i in range(len(x_mean)):
        ax1.errorbar(x_mean[i], y_mean[i], 
                    xerr=x_err[i], yerr=y_err[i],
                    fmt='none', ecolor=colors[mode], alpha=0.4, capsize=3, capthick=1.5, zorder=2)
    
    # 画点（用不同marker区分）
    ax1.scatter(x_mean, y_mean, c=colors[mode], s=120, marker=markers[mode], 
                label=mode, zorder=5, edgecolors='white', linewidths=2)
    
    # 箭头
    for i in range(len(x_mean)-1):
        arrow = FancyArrowPatch((x_mean[i], y_mean[i]), (x_mean[i+1], y_mean[i+1]), 
                                arrowstyle='->', mutation_scale=20, 
                                color=colors[mode], alpha=0.9, zorder=4, linewidth=2)
        ax1.add_patch(arrow)
        
    # 标注 T1, T2...
    for i, t_val in enumerate(subset["Turn"].values):
        ax1.text(x_mean[i], y_mean[i]+0.008, f"T{t_val}", 
                fontsize=9, color=colors[mode], fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=colors[mode], linewidth=1),
                zorder=6)

ax1.set_title("Trajectory of Discourse: Semantic Divergence vs. Critical Stance\n(with 95% Confidence Intervals)", 
              fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel("Semantic Divergence from Anchor (1 - Similarity)", fontsize=11)
ax1.set_ylabel("Critical Contribution Score (Strict)", fontsize=11)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# 象限标记
ax1.text(0.95, 0.05, "Low Critique\n(Agreement)", transform=ax1.transAxes, 
         color='gray', ha='right', va='bottom', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax1.text(0.05, 0.95, "High Critique\n(Debate)", transform=ax1.transAxes, 
         color='gray', ha='left', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# === 副图：统计摘要 ===
ax2 = fig.add_subplot(gs[0, 1])

# 计算每个Turn的组间差异统计
turn_stats = []
for turn in sorted(df["Turn"].unique()):
    turn_data = df[df["Turn"] == turn]
    for mode in colors.keys():
        mode_data = turn_data[turn_data["Mode"] == mode]
        if len(mode_data) > 0:
            turn_stats.append({
                "Turn": turn,
                "Mode": mode,
                "Divergence_mean": mode_data["Divergence"].mean(),
                "Critique_mean": mode_data["Critique"].mean(),
                "Divergence_std": mode_data["Divergence"].std(),
                "Critique_std": mode_data["Critique"].std(),
                "n": len(mode_data)
            })

df_stats = pd.DataFrame(turn_stats)

# 绘制每个Turn的均值对比（带误差棒）
turns = sorted(df_stats["Turn"].unique())
x_pos = np.arange(len(turns))
width = 0.25

for idx, mode in enumerate(colors.keys()):
    mode_stats = df_stats[df_stats["Mode"] == mode].sort_values("Turn")
    if len(mode_stats) == 0:
        continue
    
    means = mode_stats["Critique_mean"].values
    stds = mode_stats["Critique_std"].values / np.sqrt(mode_stats["n"].values)  # SEM
    
    ax2.bar(x_pos + idx*width, means, width, label=mode, color=colors[mode], alpha=0.7)
    ax2.errorbar(x_pos + idx*width, means, yerr=stds*1.96, fmt='none', 
                color='black', capsize=3, capthick=1, alpha=0.6)

ax2.set_xlabel("Turn", fontsize=11)
ax2.set_ylabel("Critical Contribution Score (Mean ± 95% CI)", fontsize=10)
ax2.set_title("Critique Score by Turn and Method", fontsize=11, fontweight='bold')
ax2.set_xticks(x_pos + width)
ax2.set_xticklabels([f"T{t}" for t in turns])
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig("trajectory_analysis_zoomed.png", dpi=300, bbox_inches='tight')
print("Saved improved plot to trajectory_analysis_zoomed.png")

# 打印统计摘要
print("\n=== Statistical Summary ===")
print("\nMean values by Mode and Turn:")
summary = df.groupby(["Mode", "Turn"])[["Divergence", "Critique"]].agg(["mean", "std", "count"])
print(summary)
print("\nOverall means by Mode:")
overall = df.groupby("Mode")[["Divergence", "Critique"]].mean()
print(overall)