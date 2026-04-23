#!/usr/bin/env python3
"""
Create dual-panel figure:
  (a) GPT-5.1 cross-topology per-topic Vendi Scores
  (b) Cross-model comparison: DSV3 vs GPT-5.1 at Standard topology
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

# Load data
gpt51 = pd.read_csv('./data/tsne_code/pertopic_vendi_gpt51_topologies.csv')
dsv3 = pd.read_csv('./data/tsne_code/pertopic_vendi_topologies.csv')

# Short topic names
def shorten(t):
    mapping = {
        'applications_to_neuroscience_&_cognitive_science': 'Neuro/CogSci',
        'applications_to_physical_sciences_(physics,_chemistry,_biology,_etc.)': 'PhysSci',
        'applications_to_robotics,_autonomy,_planning': 'Robotics',
        'causal_reasoning': 'Causal',
        'datasets_and_benchmarks': 'Data/Bench',
        'deep_generative_models_and_autoencoders': 'GenModels',
        'general_machine_learning_(i.e.,_none_of_the_above)': 'General ML',
        'graph_neural_networks_and_graph-based_methods': 'Graphs/Topo',
        'infrastructure_(hardware,_software,_design_patterns)': 'Infra/SW',
        'learning_theory': 'LearnTheory',
        'metric_learning,_entity_embeddings,_and_kernel_methods': 'Metric/Kernel',
        'neurosymbolic_&_hybrid_ai_systems': 'NeuroSymb',
        'optimization_for_deep_networks': 'Optim',
        'probabilistic_methods_(e.g.,_variational_inference,_bayesian_deep_learning)': 'Prob/Bayes',
        'reinforcement_learning': 'RL',
        'representation_learning': 'RepLearn',
        'self-supervised_and_unsupervised_learning': 'SSL/Unsup',
        'societal_considerations_(e.g.,_fairness,_safety,_privacy)': 'Society/Fair',
        'transfer_learning_and_meta-learning': 'Transfer/Meta',
        'visualization_or_interpretability': 'Viz/Interp',
    }
    for k, v in mapping.items():
        if k in t:
            return v
    return t[:12]

# Build a topic-to-short mapping from unique topics
all_topics = sorted(gpt51['topic'].unique())
topic_short = {t: shorten(t) for t in all_topics}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Panel (a): GPT-5.1 cross-topology ──
colors_topo = {'GPT5.1-Standard': '#d62728', 'GPT5.1-NGT': '#ff7f0e', 'GPT5.1-Recursive': '#2ca02c'}
labels_topo = {'GPT5.1-Standard': 'Standard', 'GPT5.1-NGT': 'NGT', 'GPT5.1-Recursive': 'Recursive'}

# Sort topics by Standard vendi for consistent ordering
std_sub = gpt51[gpt51['condition'] == 'GPT5.1-Standard'].sort_values('vendi')
topic_order = std_sub['topic'].tolist()  # use full topic names as keys
short_labels = [topic_short[t] for t in topic_order]

x = np.arange(len(topic_order))
width = 0.25

for i, cond in enumerate(['GPT5.1-Standard', 'GPT5.1-NGT', 'GPT5.1-Recursive']):
    sub = gpt51[gpt51['condition'] == cond].set_index('topic').loc[topic_order]
    ax1.barh(x + (i - 1) * width, sub['vendi'].values, height=width,
             color=colors_topo[cond], label=labels_topo[cond], alpha=0.85, edgecolor='white', linewidth=0.5)

ax1.set_yticks(x)
ax1.set_yticklabels(short_labels, fontsize=8)
ax1.set_xlabel('Per-Topic Vendi Score')
ax1.set_title('(a) GPT-5.1: Topology Effect on Diversity')
ax1.legend(loc='lower right', framealpha=0.9)
ax1.axvline(x=gpt51[gpt51['condition'] == 'GPT5.1-Standard']['vendi'].mean(), color='#d62728',
            linestyle='--', alpha=0.5, linewidth=0.8)
ax1.axvline(x=gpt51[gpt51['condition'] == 'GPT5.1-Recursive']['vendi'].mean(), color='#2ca02c',
            linestyle='--', alpha=0.5, linewidth=0.8)
ax1.set_xlim(0, 3.8)

# ── Panel (b): Cross-model at Standard topology ──
dsv3_std = dsv3[dsv3['condition'] == 'Standard'].set_index('topic')
gpt51_std = gpt51[gpt51['condition'] == 'GPT5.1-Standard'].set_index('topic')

x2 = np.arange(len(topic_order))

dsv3_vals = [dsv3_std.loc[t, 'vendi'] for t in topic_order]
gpt51_vals = [gpt51_std.loc[t, 'vendi'] for t in topic_order]

ax2.barh(x2 + 0.15, dsv3_vals, height=0.3, color='#1f77b4', label='DeepSeek-V3', alpha=0.85, edgecolor='white', linewidth=0.5)
ax2.barh(x2 - 0.15, gpt51_vals, height=0.3, color='#d62728', label='GPT-5.1', alpha=0.85, edgecolor='white', linewidth=0.5)

ax2.set_yticks(x2)
ax2.set_yticklabels(short_labels, fontsize=8)
ax2.set_xlabel('Per-Topic Vendi Score')
ax2.set_title('(b) Cross-Model: Standard Topology')
ax2.legend(loc='lower right', framealpha=0.9)

# Add annotation for the gap
mean_dsv3 = np.mean(dsv3_vals)
mean_gpt51 = np.mean(gpt51_vals)
ax2.annotate(f'$\\Delta$ = {mean_dsv3 - mean_gpt51:.2f}\n(d = 6.97)',
             xy=(2.4, len(topic_order) - 2), fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
ax2.set_xlim(0, 4.2)

plt.tight_layout()
plt.savefig('./outputs/gpt51_cross_topology.png',
            dpi=300, bbox_inches='tight')
plt.savefig('./outputs/gpt51_cross_topology.pdf',
            bbox_inches='tight')
print("Saved figures/gpt51_cross_topology.{png,pdf}")
