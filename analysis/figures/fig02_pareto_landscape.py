import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe
from pathlib import Path
import re

# ================= CONFIGURATION =================
# 定义你想要的严格边界
X_LIMITS = (1.0, 4.5)
Y_LIMITS = (7.9, 8.6)

sns.set_theme(style="white", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

CSV_PATH = Path("./data/sec_models/metrics_vendi_order.csv")
OUTPUT_DIR = Path("./data/sec_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "extracted_proposals_dsv3": {
        "model_name": "DeepSeek-V3",
        "quality_file": Path("./data/sec_models/ai_researcher_multi_topic_dsv3_rec_final/_OVERALL_AVERAGES.txt"),
        "color": "#185ADB",
    },
    "extracted_proposals_gpt51": {
        "model_name": "GPT-5.1",
        "quality_file": Path("./data/sec_models/ai_researcher_multi_topic_gpt51/_OVERALL_AVERAGES.txt"),
        "color": "#10A37F",
    },
    "extracted_proposals_grok4": {
        "model_name": "Grok-4",
        "quality_file": Path("./data/sec_models/Multi_Collaboration_grok4/_OVERALL_AVERAGES.txt"),
        "color": "#2A2A2A",
    },
    "extracted_proposals_o1mini": {
        "model_name": "o1-mini",
        "quality_file": Path("./data/sec_models/ai_researcher_multi_topic_o1mini/_OVERALL_AVERAGES.txt"),
        "color": "#F57C00",
    },
}

# ================= UTILS =================

def parse_quality_file(file_path: Path):
    topic_qualities = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        breakdown_match = re.search(r'Detailed breakdown by topic:(.*)', content, re.DOTALL)
        if not breakdown_match:
            return topic_qualities
        breakdown_text = breakdown_match.group(1)
        matches = re.findall(r'([^:\n]+)_proposals:\s*\n\s*Overall Quality:\s*([\d.]+)', breakdown_text)
        for topic_name, quality_str in matches:
            topic_qualities[topic_name.strip()] = float(quality_str.strip())
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return topic_qualities

def confidence_ellipse(x, y, ax, n_std=1.96, facecolor='none', **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# ================= MAIN PLOT =================

def main():
    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    all_data = []

    for dataset_name, config in MODEL_CONFIGS.items():
        if not config["quality_file"].exists():
            continue
        topic_qualities = parse_quality_file(config["quality_file"])
        dataset_df = df[df['dataset'] == dataset_name].copy()
        
        for _, row in dataset_df.iterrows():
            topic = row['topic']
            quality = topic_qualities.get(topic)
            if quality is None:
                for t_key, q_val in topic_qualities.items():
                    if t_key.replace("_proposals", "").strip() == topic.strip():
                        quality = q_val
                        break
            if quality is not None:
                all_data.append({
                    'Model': config["model_name"],
                    'Diversity': float(row['vendi_score']),
                    'Quality': quality
                })

    plot_df = pd.DataFrame(all_data)
    if len(plot_df) == 0: return

    palette = {cfg['model_name']: cfg['color'] for cfg in MODEL_CONFIGS.values() if cfg['model_name'] in plot_df['Model'].unique()}

    # 1. SETUP GRID
    # space=0.02 进一步减小边缘图和主图的距离
    g = sns.JointGrid(data=plot_df, x="Diversity", y="Quality", height=10, ratio=5, space=0.02)

    # 2. SCATTER PLOT
    sns.scatterplot(
        data=plot_df, x="Diversity", y="Quality", hue="Model", palette=palette,
        s=30, alpha=0.6, edgecolor=None, ax=g.ax_joint, legend=False
    )

    # 3. LABEL CONFIG & CENTROIDS
    LABEL_CONFIG = {
        "DeepSeek-V3": ( 100, -40, "left", "top"),    
        "GPT-5.1":     (-30,  50, "right", "bottom"), 
        "Grok-4":      ( 43,  22, "left", "bottom"),  
        "o1-mini":     ( 160, -35, "right", "top")     
    }

    centroids = plot_df.groupby('Model')[['Diversity', 'Quality']].mean()
    
    for model_name, row in centroids.iterrows():
        color = palette[model_name]
        g.ax_joint.scatter(
            row['Diversity'], row['Quality'],
            c=color, s=250, edgecolors='white', linewidth=2, zorder=10
        )
        
        offset_x, offset_y, ha_val, va_val = LABEL_CONFIG.get(model_name, (30, 30, "left", "bottom"))
        
        text_obj = g.ax_joint.annotate(
            model_name, 
            xy=(row['Diversity'], row['Quality']), 
            xytext=(offset_x, offset_y),           
            textcoords='offset points',
            fontsize=15,    # 稍微再大一点
            fontweight='bold', 
            color=color,
            ha=ha_val, 
            va=va_val,
            zorder=30,
            arrowprops=dict(
                arrowstyle="-", 
                color=color, 
                alpha=0.6, 
                linewidth=1.5,
                shrinkA=5, shrinkB=5
            )
        )
        
        # 文字白色描边
        text_obj.set_path_effects([
            pe.withStroke(linewidth=4, foreground='white'),
            pe.Normal()
        ])

    # 4. CONFIDENCE ELLIPSES
    for model_name, color in palette.items():
        subset = plot_df[plot_df['Model'] == model_name]
        if len(subset) > 2:
            face_color_with_alpha = to_rgba(color, alpha=0.1)
            confidence_ellipse(
                subset['Diversity'], subset['Quality'], g.ax_joint,
                n_std=1.96,                 
                facecolor=face_color_with_alpha,
                edgecolor=color,            
                linewidth=1.5,              
                linestyle='--',             
                zorder=5                    
            )

    # 5. MARGINAL PLOTS (CRITICAL UPDATE)
    # 顶部 (Diversity): clip限制在 X_LIMITS 范围内
    sns.kdeplot(
        data=plot_df, x="Diversity", hue="Model", palette=palette,
        fill=True, common_norm=False, alpha=0.2, legend=False,
        linewidth=2.5, bw_adjust=1.5, 
        clip=X_LIMITS,  # 【关键】：只计算这个范围内的密度，防止尾巴拖出去
        ax=g.ax_marg_x
    )

    # 右侧 (Quality): clip限制在 Y_LIMITS 范围内
    sns.kdeplot(
        data=plot_df, y="Quality", hue="Model", palette=palette,
        fill=True, common_norm=False, alpha=0.2, legend=False,
        linewidth=2.5, bw_adjust=1.5, 
        clip=Y_LIMITS,  # 【关键】：只计算这个范围内的密度
        ax=g.ax_marg_y
    )

    # 6. AESTHETICS & AXES
    ax = g.ax_joint
    
    ax.set_xlabel("Semantic Diversity (Vendi Score)", fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel("Idea Quality (Expert Rating)", fontsize=16, fontweight='bold', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 【终极收紧步骤】
    # 1. 设置Margins为0，消除内部padding
    ax.margins(0)
    g.ax_marg_x.margins(0)
    g.ax_marg_y.margins(0)

    # 2. 强制锁定坐标轴范围
    # 注意：JointGrid中，marg_x共享x轴，marg_y共享y轴。
    # 只要在主图 set_xlim，边缘图也会跟着变。
    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)
    
    # Save
    out_png = OUTPUT_DIR / "pareto_landscape_final.png"
    out_pdf = OUTPUT_DIR / "pareto_landscape_final.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved visualizations to {out_png}")

if __name__ == "__main__":
    main()