import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import csv
from pathlib import Path

# --- 1. Setup & Data Loading ---
csv_path = Path("./data/tsne/metrics_vendi_order.csv")
out_path = Path("./data/tsne/bar_chart_metrics_final.png")

# Dataset name mapping
dataset_mapping = {
    "ai_researcher_multi_topic_dsv3_rec_final": "Naive",
    "ai_researcher_multi_topic_dsv3_leader_experience2": "Leader-Led",
    "ai_researcher_multi_topic_dsv3_mix_final": "Vertical",
    "ai_researcher_multi_topic_dsv3_x_final": "Interdisciplinary",
    "ai_researcher_multi_topic_dsv3_young_final": "Horizontal"
}

# # Colors
# colors = {
#     "Naive": "#7f7f7f",            # Grey
#     "Leader-Led": "#d62728",       # Red
#     "Horizontal": "#17becf",       # Cyan
#     "Interdisciplinary": "#9467bd",# Purple
#     "Vertical": "#2ca02c"          # Green
# }
# colors = {
#     "Naive": "#B0B0B0",            # Neutral grey
#     "Leader-Led": "#F5A3A3",       # Muted red
#     "Horizontal": "#9CCEDB",       # Muted cyan
#     "Interdisciplinary": "#C6B7DD",# Muted purple
#     "Vertical": "#A8C8A8"          # Muted green
# }
colors = {
    "Naive": "#C4C4C4",
    "Leader-Led": "#E7A6A1",
    "Horizontal": "#B7D9E5",
    "Interdisciplinary": "#D4C8E6",
    "Vertical": "#C9DFC9"
}


# Data Container
data_store = {}

if csv_path.exists():
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            if dataset in dataset_mapping:
                name = dataset_mapping[dataset]
                try:
                    def safe_float(val):
                        if val and val.strip() and val != "nan":
                            return float(val)
                        return 0.0
                    
                    order_val = safe_float(row["order_phi"])
                    data_store[name] = {
                        "wdistinct": safe_float(row.get("content_only_wdistinct_3", "nan")),
                        "disorder": 1.0 - order_val, # 1 - Order
                        "pcd": safe_float(row["pcd"]),
                        "vendi": safe_float(row["vendi_score"]),
                    }
                except Exception as e:
                    print(f"Warning parsing {name}: {e}")
else:
    print(f"Error: CSV not found")
    exit(1)

# --- 2. Plotting Configuration ---

# Metrics Order (Left to Right)
metrics_config = [
    {"key": "wdistinct", "title": "Lexical Uniqueness\n(WDistinct-3)"},  # 表象：词汇层
    {"key": "disorder",  "title": "Structural Disorder\n(1 - Order $\phi$)"}, # 物理层：系统状态
    {"key": "pcd",       "title": "Semantic Dispersion\n(Pairwise Distance)"}, # 几何层：空间分布
    {"key": "vendi",     "title": "Effective Diversity\n(Vendi Score)"}      # 信息层：有效容量
]

# Figure Size: Wider and Shorter (Compact)
fig, axes = plt.subplots(1, 4, figsize=(18, 3.5), sharey=False)
# Tighter spacing between subplots
plt.subplots_adjust(wspace=0.2, top=0.8, bottom=0.15, left=0.05, right=0.95)

bar_width = 0.5  # Thinner bars relative to figure size

# --- 3. Rendering Loop ---

for i, (ax, metric) in enumerate(zip(axes, metrics_config)):
    key = metric["key"]
    title = metric["title"]
    
    # Prepare Data
    plot_data = []
    for name, metrics in data_store.items():
        val = metrics[key]
        col = colors[name]
        plot_data.append((name, val, col))
    
    # Sort Descending (Big -> Small)
    plot_data.sort(key=lambda x: x[1], reverse=True)
    
    sorted_names = [x[0] for x in plot_data]
    sorted_values = [x[1] for x in plot_data]
    sorted_colors = [x[2] for x in plot_data]
    
    # Transparency
    alphas = [1.0 if name == "Vertical" else 0.85 for name in sorted_names]
    indices = np.arange(len(sorted_names))
    
    # Plot Bars
    bars = ax.bar(indices, sorted_values, width=bar_width, color=sorted_colors, alpha=0.9)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    
    # Y-Axis Settings (Start from 0)
    y_max = max(sorted_values)
    ax.set_ylim(0, y_max * 1.25) # Start from 0, add top headroom
    
    # Value Labels
    for bar, val in zip(bars, sorted_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (y_max * 0.02),
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    
    # Styling
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#333333')
    
    # X-Axis Labels
    ax.set_xticks(indices)
    ax.set_xticklabels(sorted_names, rotation=25, ha='right', fontsize=9)
    
    # Grid & Spines
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

# --- 4. Global Annotation (Single Large Arrow) ---

# Draw a global arrow at the top left of the FIGURE (spanning first plot area)
# Coordinates are in Figure fraction (0,0 bottom-left to 1,1 top-right)
# Adjust these numbers to position the arrow exactly where you want
arrow_y = 0.97
arrow_x_start = 0.37
arrow_x_end = 0.08  # Pointing Left

fig.add_artist(lines.Line2D(
    [arrow_x_start, arrow_x_end], [arrow_y, arrow_y],
    transform=fig.transFigure, figure=fig,
    color='#444444', linewidth=1.5,
    marker='<', markersize=8, markeredgecolor='#444444',
    markevery=[-1] # <--- 关键修改：只标记终点（左侧）
))

# Add text label for the arrow
fig.text(arrow_x_start + 0.04, arrow_y, "Higher Value = More Diverse", 
         transform=fig.transFigure, ha='left', va='center', 
         fontsize=11, fontweight='bold', color='#444444', style='italic')
# Save
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Final polished chart generated at: {out_path}")