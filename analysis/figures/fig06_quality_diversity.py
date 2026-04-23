#!/usr/bin/env python3
"""
Generate a Quality-Diversity tradeoff figure for Section 4.
Left panel: grouped bar chart of Overall Quality across 5 personas.
Right panel: grouped bar chart of Vendi Score (from existing metrics).
Both panels share the same x-axis (persona names) for direct visual comparison.
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── Load quality scores ──────────────────────────────────
RESULTS_FILE = os.path.expanduser(
    "~/AI-Scientist/proposals/tsne_code/quality_persona_results.json"
)
with open(RESULTS_FILE) as f:
    all_results = json.load(f)

# Group by persona
persona_quality = {}
for r in all_results:
    name = r["persona"]
    if name not in persona_quality:
        persona_quality[name] = []
    persona_quality[name].append(r["scores"])

ordered = ["Horizontal", "Naive", "Vertical", "Leader-Led", "Interdisciplinary"]
short_labels = ["Horizontal", "Naive", "Vertical", "Leader-Led", "Interdisc."]

# Quality dimensions to show
dims_quality = ["Overall Quality", "Novelty", "Workability"]
dims_short = ["Overall\nQuality", "Novelty", "Workability"]

# ── Vendi Scores (from paper: Figure 3 data) ────────────
# These are the Vendi Scores reported in the paper for each persona
vendi_scores = {
    "Horizontal": 8.08,
    "Naive": 5.567,
    "Vertical": 6.082,
    "Leader-Led": 6.932,
    "Interdisciplinary": 4.647,
}

# ── Compute quality means and stds ──────────────────────
quality_means = {}
quality_stds = {}
for dim in dims_quality:
    quality_means[dim] = []
    quality_stds[dim] = []
    for name in ordered:
        vals = [s[dim] for s in persona_quality[name] if dim in s]
        quality_means[dim].append(np.mean(vals))
        quality_stds[dim].append(np.std(vals))

# ── Color scheme ─────────────────────────────────────────
colors = {
    "Horizontal": "#2196F3",      # blue
    "Naive": "#9E9E9E",           # gray
    "Vertical": "#FF9800",        # orange
    "Leader-Led": "#F44336",      # red
    "Interdisciplinary": "#9C27B0", # purple
}
bar_colors = [colors[n] for n in ordered]

# ── Create figure ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.6, 1]})

# Left panel: Quality dimensions (grouped bars)
x = np.arange(len(ordered))
width = 0.25
offsets = [-width, 0, width]

for i, (dim, dim_label) in enumerate(zip(dims_quality, dims_short)):
    bars = ax1.bar(
        x + offsets[i],
        quality_means[dim],
        width,
        yerr=quality_stds[dim],
        label=dim_label,
        color=[plt.cm.Set2(i)] * len(ordered),
        edgecolor='white',
        linewidth=0.5,
        capsize=3,
        error_kw={'linewidth': 1},
    )
    # Add value labels
    for j, bar in enumerate(bars):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + quality_stds[dim][j] + 0.08,
            f'{quality_means[dim][j]:.1f}',
            ha='center', va='bottom', fontsize=7,
        )

ax1.set_xticks(x)
ax1.set_xticklabels(short_labels, fontsize=9)
ax1.set_ylabel('Score (1–10)', fontsize=10)
ax1.set_title('(a) Proposal Quality by Persona', fontsize=11, fontweight='bold')
ax1.set_ylim(5.5, 10.0)
ax1.legend(fontsize=8, loc='lower right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right panel: Vendi Score (single bars, colored by persona)
vendi_vals = [vendi_scores[n] for n in ordered]
bars2 = ax2.bar(x, vendi_vals, 0.6, color=bar_colors, edgecolor='white', linewidth=0.5)
for j, bar in enumerate(bars2):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f'{vendi_vals[j]:.1f}',
        ha='center', va='bottom', fontsize=9, fontweight='bold',
    )

ax2.set_xticks(x)
ax2.set_xticklabels(short_labels, fontsize=9)
ax2.set_ylabel('Vendi Score', fontsize=10)
ax2.set_title('(b) Semantic Diversity by Persona', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 10.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add annotation arrow showing the asymmetry
ax2.annotate(
    'Quality Δ = 0.6\nDiversity Δ = 3.4',
    xy=(0.5, 0.85), xycoords='axes fraction',
    fontsize=8, ha='center',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
)

plt.tight_layout()

outpath = os.path.expanduser(
    "~/AI-Scientist/690714e0483d3f8c204905fd/img/quality_diversity_tradeoff.png"
)
os.makedirs(os.path.dirname(outpath), exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved to {outpath}")

# Also save a PDF version
plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved PDF version")
