"""
Cross-Persona Topology Effect Consistency Analysis (Appendix)

Concern addressed: topology effects may vary under different persona structures.

Strategy: Show that topology interventions (NGT, Subgroup) consistently improve diversity
over Recursive baseline across ALL available persona × model combinations:
  1. DSV3 + Naive:           Recursive vs NGT vs Subgroup
  2. o1-mini + Horizontal:   Recursive vs Subgroup
  3. GPT-5.1 + Interdisciplinary: Recursive vs NGT

If the topology effect direction is consistent across all three persona structures,
the concern is empirically addressed.
"""

import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────────

SEC = Path("./data/sec_models")

# All available persona × model × topology combinations
CELLS = {
    # DSV3 + Naive (3 topologies)
    ("DSV3", "Naive", "Recursive"):  SEC / "dsv3_naive_recursive/extracted_proposals",
    ("DSV3", "Naive", "NGT"):        SEC / "dsv3_naive_ngt/extracted_proposals",
    ("DSV3", "Naive", "Subgroup"):   SEC / "dsv3_naive_subgroup/extracted_proposals",
    # DSV3 + Horizontal — NEW W3 ABLATION (2 topologies, same model, different persona)
    ("DSV3", "Horizontal", "Recursive"): SEC / "dsv3_horizontal_recursive_ablation/extracted_proposals",
    ("DSV3", "Horizontal", "NGT"):       SEC / "dsv3_horizontal_ngt_ablation/extracted_proposals",
    # o1-mini + Horizontal (2 topologies)
    ("o1-mini", "Horizontal", "Recursive"): SEC / "o1_mini_horizontal_recursive/extracted_proposals",
    ("o1-mini", "Horizontal", "Subgroup"):  SEC / "o1_mini_horizontal_subgroup/extracted_proposals",
    # GPT-5.1 + Interdisciplinary (2 topologies)
    ("GPT-5.1", "Interdisc.", "Recursive"): SEC / "gpt5_1_Interdisciplinary_recursive/extracted_proposals",
    ("GPT-5.1", "Interdisc.", "NGT"):       SEC / "gpt5_1_Interdisciplinary_ngt/extracted_proposals",
}

TOPIC_SHORT = {
    "applications_to_neuroscience_&_cognitive_science": "Neuro/CogSci",
    "applications_to_physical_sciences_(physics,_chemistry,_biology,_etc.)": "PhysSci",
    "applications_to_robotics,_autonomy,_planning": "Robotics",
    "causal_reasoning": "Causal",
    "datasets_and_benchmarks": "Data/Bench",
    "general_machine_learning": "General ML",
    "generative_models": "GenModels",
    "infrastructure,_software_libraries,_hardware,_etc.": "Infra/SW",
    "learning_on_graphs_and_other_geometries_&_topologies": "Graphs/Topo",
    "learning_theory": "LearnTheory",
    "metric_learning,_kernel_learning,_and_sparse_coding": "Metric/Kernel",
    "neurosymbolic_&_hybrid_ai_systems_(physics-informed,_logic_&_formal_reasoning,_etc.)": "NeuroSymb",
    "optimization": "Optim",
    "probabilistic_methods_(bayesian_methods,_variational_inference,_sampling,_uq,_etc.)": "Prob/Bayes",
    "reinforcement_learning": "RL",
    "representation_learning_for_computer_vision,_audio,_language,_and_other_modalities": "RepLearn",
    "societal_considerations_including_fairness,_safety,_privacy": "Society/Fair",
    "transfer_learning,_meta_learning,_and_lifelong_learning": "Transfer/Meta",
    "unsupervised,_self-supervised,_semi-supervised,_and_supervised_representation_learning": "SSL/Unsup",
    "visualization_or_interpretation_of_learned_representations": "Viz/Interp",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def safe_exec(content):
    content = re.sub(r'\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])', '?', content)
    ns = {}
    code = compile(content, "<proposals>", "exec")
    exec(code, {}, ns)
    return ns


def compute_vendi_score(embeddings):
    from vendi_score import vendi
    emb = np.array(embeddings)
    if len(emb) < 2:
        return 1.0
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms
    K = emb @ emb.T
    K = (K + K.T) / 2.0
    return float(vendi.score_K(K))


def compute_pcd(embeddings):
    emb = np.array(embeddings)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms
    sim = emb @ emb.T
    n = len(emb)
    mask = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sim[mask]))


def get_openai_embeddings_cached(texts, cache_path):
    """Get embeddings with caching."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if isinstance(cache, dict) and cache.get("texts_hash") == hash(tuple(texts)):
            return cache["embeddings"]
        elif isinstance(cache, list):
            if len(cache) == len(texts):
                return cache

    from openai import OpenAI
    import os
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_size = 100
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(input=batch, model="text-embedding-3-large")
        all_embs.extend([e.embedding for e in resp.data])
        print(f"    Embedded {min(i+batch_size, len(texts))}/{len(texts)}")

    cache = {"texts_hash": hash(tuple(texts)), "embeddings": all_embs}
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return all_embs


def load_proposals_and_embed(dir_path):
    """Load proposals and compute/cache embeddings for all topics."""
    dir_path = Path(dir_path)
    topic_embs = {}
    files = sorted(dir_path.glob("*_proposals.txt"))
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        ns = safe_exec(content)
        papers = [p.strip() for p in ns.get("paper_txts", []) if p.strip()]
        if not papers:
            continue
        topic = fp.name.replace("_proposals.txt", "")

        cache_path = dir_path / f"embeddings_cache_v3large_{topic}.pkl"
        embs = get_openai_embeddings_cached(papers, cache_path)
        topic_embs[topic] = np.array(embs)
    return topic_embs


# ── Load Data ───────────────────────────────────────────────────────────────

print("Loading and embedding all conditions...")
cell_data = {}
for (model, persona, topo), dir_path in CELLS.items():
    label = f"{model} + {persona} + {topo}"
    if not dir_path.exists():
        print(f"  SKIP {label}: not found")
        continue
    print(f"  Loading {label}...")
    topic_embs = load_proposals_and_embed(dir_path)
    cell_data[(model, persona, topo)] = topic_embs
    print(f"    {len(topic_embs)} topics, ~{np.mean([len(v) for v in topic_embs.values()]):.0f} proposals/topic")


# ── Compute Metrics ─────────────────────────────────────────────────────────

print("\nComputing Vendi Scores...")
rows = []
for (model, persona, topo), topic_embs in cell_data.items():
    for topic, embs in topic_embs.items():
        vs = compute_vendi_score(embs)
        pcd = compute_pcd(embs)
        rows.append({
            "model": model,
            "persona": persona,
            "topology": topo,
            "condition": f"{model}\n{persona}",
            "topic": topic,
            "topic_short": TOPIC_SHORT.get(topic, topic[:20]),
            "vendi": vs,
            "pcd": pcd,
            "n_proposals": len(embs),
        })

df = pd.DataFrame(rows)


# ── Analysis: Within-persona topology comparisons ──────────────────────────

print("\n" + "=" * 70)
print("WITHIN-PERSONA TOPOLOGY COMPARISONS")
print("=" * 70)

comparisons = [
    ("DSV3", "Naive", "Recursive", "NGT"),
    ("DSV3", "Naive", "Recursive", "Subgroup"),
    ("DSV3", "Horizontal", "Recursive", "NGT"),  # NEW: W3 ablation
    ("o1-mini", "Horizontal", "Recursive", "Subgroup"),
    ("GPT-5.1", "Interdisc.", "Recursive", "NGT"),
]

comp_rows = []
for model, persona, base_topo, test_topo in comparisons:
    base = df[(df["model"] == model) & (df["persona"] == persona) & (df["topology"] == base_topo)]
    test = df[(df["model"] == model) & (df["persona"] == persona) & (df["topology"] == test_topo)]

    if len(base) == 0 or len(test) == 0:
        print(f"\n  SKIP: {model} + {persona}: {base_topo} vs {test_topo} (missing data)")
        continue

    common_topics = sorted(set(base["topic"]) & set(test["topic"]))
    base_by_topic = base.set_index("topic")["vendi"]
    test_by_topic = test.set_index("topic")["vendi"]

    base_vals = base_by_topic.loc[common_topics].values
    test_vals = test_by_topic.loc[common_topics].values
    diff = test_vals - base_vals
    t_stat, p_val = stats.ttest_rel(test_vals, base_vals)
    pct = diff.mean() / base_vals.mean() * 100
    wins = (diff > 0).sum()

    print(f"\n  {model} + {persona}: {test_topo} vs {base_topo}")
    print(f"    {base_topo}: {base_vals.mean():.3f} ± {base_vals.std():.3f}")
    print(f"    {test_topo}:  {test_vals.mean():.3f} ± {test_vals.std():.3f}")
    print(f"    Δ = {diff.mean():+.3f} ({pct:+.1f}%), t = {t_stat:.2f}, p = {p_val:.4f}")
    print(f"    Topics where {test_topo} > {base_topo}: {wins}/{len(diff)}")

    comp_rows.append({
        "model": model,
        "persona": persona,
        "baseline": base_topo,
        "intervention": test_topo,
        "baseline_mean": base_vals.mean(),
        "intervention_mean": test_vals.mean(),
        "delta": diff.mean(),
        "pct_change": pct,
        "t_stat": t_stat,
        "p_value": p_val,
        "n_wins": wins,
        "n_topics": len(diff),
        "effect_direction": "+" if diff.mean() > 0 else "-",
    })

df_comp = pd.DataFrame(comp_rows)

# Check consistency
print("\n" + "=" * 70)
print("CONSISTENCY CHECK: Do topology interventions help across all personas?")
print("=" * 70)
for _, row in df_comp.iterrows():
    direction = "IMPROVES" if row["delta"] > 0 else "REDUCES"
    sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "n.s."
    print(f"  {row['model']:8s} + {row['persona']:12s}: {row['intervention']:8s} {direction} diversity by {abs(row['pct_change']):.1f}% ({sig})")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", font_scale=1.05)
fig = plt.figure(figsize=(18, 10))
gs_layout = gridspec.GridSpec(1, 3, wspace=0.35, width_ratios=[1.2, 1, 1])

COLORS = {
    "Recursive": "#95a5a6",  # Gray (baseline)
    "NGT": "#2ecc71",        # Green
    "Subgroup": "#3498db",   # Blue
}

# ── Panel A: Grouped bar chart — all conditions ────────────────────────────

ax_a = fig.add_subplot(gs_layout[0, 0])

# Group by (model, persona), show topologies as bars
conditions = [
    ("DSV3", "Naive"),
    ("DSV3", "Horizontal"),
    ("o1-mini", "Horizontal"),
    ("GPT-5.1", "Interdisc."),
]

x_positions = []
x_labels = []
bar_width = 0.25
group_gap = 1.0
pos = 0

for model, persona in conditions:
    sub = df[(df["model"] == model) & (df["persona"] == persona)]
    topos = sorted(sub["topology"].unique(), key=lambda t: ["Recursive", "NGT", "Subgroup"].index(t) if t in ["Recursive", "NGT", "Subgroup"] else 99)

    group_start = pos
    for i, topo in enumerate(topos):
        vals = sub[sub["topology"] == topo]["vendi"]
        mean_v = vals.mean()
        sem_v = vals.std() / np.sqrt(len(vals))
        bar = ax_a.bar(pos, mean_v, bar_width * 0.9, yerr=sem_v, capsize=3,
                       color=COLORS.get(topo, "gray"), alpha=0.85,
                       label=topo if (model, persona) == conditions[0] else "")
        ax_a.text(pos, mean_v + sem_v + 0.02, f"{mean_v:.2f}", ha="center",
                  va="bottom", fontsize=8, fontweight="bold")
        pos += bar_width

    # Group label
    group_center = (group_start + pos - bar_width) / 2
    x_positions.append(group_center)
    x_labels.append(f"{model}\n({persona})")
    pos += group_gap - bar_width

ax_a.set_xticks(x_positions)
ax_a.set_xticklabels(x_labels, fontsize=10)
ax_a.set_ylabel("Vendi Score (mean ± SEM)", fontsize=11)
ax_a.set_title("(a) Topology Effect Across Persona Structures\n(each group = fixed model + persona)",
               fontsize=11, fontweight="bold")
ax_a.legend(fontsize=10, title="Topology", loc="upper left")
ax_a.set_ylim(bottom=0)


# ── Panel B: Effect size (Δ Vendi) for each comparison ─────────────────────

ax_b = fig.add_subplot(gs_layout[0, 1])

y_pos = np.arange(len(df_comp))
colors_bar = []
for _, row in df_comp.iterrows():
    colors_bar.append("#2ecc71" if row["delta"] > 0 else "#e74c3c")

bars = ax_b.barh(y_pos, df_comp["delta"], color=colors_bar, alpha=0.85, height=0.6)

# Labels
labels = []
for _, row in df_comp.iterrows():
    sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
    labels.append(f"{row['model']} + {row['persona']}\n({row['intervention']} vs {row['baseline']})")

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(labels, fontsize=9)
ax_b.axvline(x=0, color="black", linewidth=0.8)
ax_b.set_xlabel("Δ Vendi Score (intervention − baseline)", fontsize=11)
ax_b.set_title("(b) Topology Intervention Effect Size\n(positive = intervention improves diversity)",
               fontsize=11, fontweight="bold")

# Annotate significance
for i, (_, row) in enumerate(df_comp.iterrows()):
    sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "n.s."
    x_text = row["delta"] + 0.01 if row["delta"] > 0 else row["delta"] - 0.01
    ha = "left" if row["delta"] > 0 else "right"
    ax_b.text(x_text, i, f"{row['delta']:+.3f} ({sig})", va="center", ha=ha,
              fontsize=9, fontweight="bold")


# ── Panel C: Per-topic paired differences (DSV3 Naive as main example) ─────

ax_c = fig.add_subplot(gs_layout[0, 2])

# Show per-topic Δ for all comparisons
for idx, (_, row) in enumerate(df_comp.iterrows()):
    model, persona = row["model"], row["persona"]
    base_topo, test_topo = row["baseline"], row["intervention"]

    base_sub = df[(df["model"] == model) & (df["persona"] == persona) & (df["topology"] == base_topo)]
    test_sub = df[(df["model"] == model) & (df["persona"] == persona) & (df["topology"] == test_topo)]
    common = sorted(set(base_sub["topic"]) & set(test_sub["topic"]))

    base_by_t = base_sub.set_index("topic")["vendi"]
    test_by_t = test_sub.set_index("topic")["vendi"]
    deltas = test_by_t.loc[common].values - base_by_t.loc[common].values

    color = COLORS.get(test_topo, "gray")
    label = f"{model} {persona}\n({test_topo})"
    jitter = (np.random.RandomState(idx).rand(len(deltas)) - 0.5) * 0.15
    ax_c.scatter(np.full(len(deltas), idx) + jitter, deltas, c=color, alpha=0.5,
                 s=30, edgecolors="white", linewidth=0.3)
    ax_c.scatter(idx, np.mean(deltas), c=color, s=150, marker="D",
                 edgecolors="black", linewidth=1.5, zorder=5)

ax_c.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
ax_c.set_xticks(range(len(df_comp)))
xlabels_c = [f"{r['model']}\n{r['persona']}\n({r['intervention']})" for _, r in df_comp.iterrows()]
ax_c.set_xticklabels(xlabels_c, fontsize=8)
ax_c.set_ylabel("Δ Vendi per topic", fontsize=11)
ax_c.set_title("(c) Per-Topic Effect Distribution\n(diamonds = mean; dots = individual topics)",
               fontsize=11, fontweight="bold")

plt.savefig("./data/appendix_w3_topology_consistency.png", dpi=300, bbox_inches="tight")
print("\nSaved: appendix_w3_topology_consistency.png")
plt.close()


# ── Save data ───────────────────────────────────────────────────────────────

df.to_csv("./data/appendix_w3_all_conditions.csv", index=False)
df_comp.to_csv("./data/appendix_w3_comparisons.csv", index=False)
print("Saved CSVs.")


# ── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

n_positive = (df_comp["delta"] > 0).sum()
n_total = len(df_comp)
n_sig = (df_comp["p_value"] < 0.05).sum()

print(f"""
CROSS-PERSONA TOPOLOGY CONSISTENCY:
  {n_positive}/{n_total} comparisons show topology intervention IMPROVES diversity
  {n_sig}/{n_total} comparisons are statistically significant (p < 0.05)

Per-comparison details:""")
for _, row in df_comp.iterrows():
    sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "n.s."
    print(f"  {row['model']:8s} + {row['persona']:12s}: {row['intervention']:8s} vs {row['baseline']:10s} "
          f"Δ = {row['delta']:+.3f} ({row['pct_change']:+.1f}%) {sig}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: 2×2 Factorial (DSV3 only) — Key Appendix Figure
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2×2 FACTORIAL ANALYSIS: DSV3 × (Naive vs Horizontal) × (Recursive vs NGT)")
print("=" * 70)

# Check if we have all 4 cells
factorial_keys = [
    ("DSV3", "Naive", "Recursive"),
    ("DSV3", "Naive", "NGT"),
    ("DSV3", "Horizontal", "Recursive"),
    ("DSV3", "Horizontal", "NGT"),
]
factorial_available = all(k in cell_data for k in factorial_keys)

if factorial_available:
    # Compute per-topic Vendi for each cell
    factorial_rows = []
    for key in factorial_keys:
        model, persona, topo = key
        for topic, embs in cell_data[key].items():
            vs = compute_vendi_score(embs)
            factorial_rows.append({
                "persona": persona, "topology": topo,
                "topic": topic, "vendi": vs,
            })
    df_fact = pd.DataFrame(factorial_rows)

    # Two-way ANOVA-style analysis (using aligned topics)
    for persona in ["Naive", "Horizontal"]:
        rec = df_fact[(df_fact["persona"] == persona) & (df_fact["topology"] == "Recursive")]
        ngt = df_fact[(df_fact["persona"] == persona) & (df_fact["topology"] == "NGT")]
        common = sorted(set(rec["topic"]) & set(ngt["topic"]))
        rec_v = rec.set_index("topic").loc[common, "vendi"].values
        ngt_v = ngt.set_index("topic").loc[common, "vendi"].values
        t, p = stats.ttest_rel(ngt_v, rec_v)
        d = (ngt_v - rec_v).mean()
        print(f"  {persona:12s}: NGT − Recursive = {d:+.3f}, t={t:.2f}, p={p:.4f}")

    # Interaction test: is the topology effect different across personas?
    naive_rec = df_fact[(df_fact["persona"] == "Naive") & (df_fact["topology"] == "Recursive")].set_index("topic")["vendi"]
    naive_ngt = df_fact[(df_fact["persona"] == "Naive") & (df_fact["topology"] == "NGT")].set_index("topic")["vendi"]
    horiz_rec = df_fact[(df_fact["persona"] == "Horizontal") & (df_fact["topology"] == "Recursive")].set_index("topic")["vendi"]
    horiz_ngt = df_fact[(df_fact["persona"] == "Horizontal") & (df_fact["topology"] == "NGT")].set_index("topic")["vendi"]

    common_all = sorted(set(naive_rec.index) & set(naive_ngt.index) & set(horiz_rec.index) & set(horiz_ngt.index))
    if len(common_all) > 0:
        naive_effect = naive_ngt.loc[common_all].values - naive_rec.loc[common_all].values
        horiz_effect = horiz_ngt.loc[common_all].values - horiz_rec.loc[common_all].values
        t_int, p_int = stats.ttest_rel(naive_effect, horiz_effect)
        print(f"\n  INTERACTION TEST (is topology effect different across personas?):")
        print(f"    Naive topology effect:      {naive_effect.mean():+.3f} ± {naive_effect.std():.3f}")
        print(f"    Horizontal topology effect:  {horiz_effect.mean():+.3f} ± {horiz_effect.std():.3f}")
        print(f"    Interaction: t={t_int:.2f}, p={p_int:.4f}")
        if p_int > 0.05:
            print(f"    → NO significant interaction (p={p_int:.3f}): topology effect is CONSISTENT across personas")
        else:
            print(f"    → Significant interaction (p={p_int:.3f}): topology effect DIFFERS across personas")

    # ── Plot 2×2 factorial figure ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Interaction plot (line plot)
    ax = axes2[0]
    for persona, marker, color in [("Naive", "o", "#e74c3c"), ("Horizontal", "s", "#3498db")]:
        means = []
        sems = []
        for topo in ["Recursive", "NGT"]:
            vals = df_fact[(df_fact["persona"] == persona) & (df_fact["topology"] == topo)]["vendi"]
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)))
        ax.errorbar([0, 1], means, yerr=sems, marker=marker, markersize=10,
                    linewidth=2, capsize=5, label=persona, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Recursive", "NGT"], fontsize=11)
    ax.set_ylabel("Vendi Score (mean ± SEM)", fontsize=11)
    ax.set_title("(a) 2×2 Factorial: Persona × Topology\n(parallel lines = no interaction)", fontsize=11, fontweight="bold")
    ax.legend(title="Persona", fontsize=10)

    # Panel B: Per-topic paired differences
    ax = axes2[1]
    if len(common_all) > 0:
        x = np.arange(len(common_all))
        width = 0.35
        short_labels = [TOPIC_SHORT.get(t, t[:12]) for t in common_all]
        ax.bar(x - width/2, naive_effect, width, label="Naive", color="#e74c3c", alpha=0.7)
        ax.bar(x + width/2, horiz_effect, width, label="Horizontal", color="#3498db", alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Δ Vendi (NGT − Recursive)", fontsize=11)
        ax.set_title("(b) Per-Topic Topology Effect\n(both personas show same direction)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)

    # Panel C: Scatter — Naive effect vs Horizontal effect
    ax = axes2[2]
    if len(common_all) > 0:
        ax.scatter(naive_effect, horiz_effect, c="#2c3e50", alpha=0.6, s=50, edgecolors="white")
        # Fit line
        slope, intercept, r, p_r, se = stats.linregress(naive_effect, horiz_effect)
        x_line = np.linspace(min(naive_effect), max(naive_effect), 50)
        ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.5,
                label=f"r={r:.2f}, p={p_r:.3f}")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Naive: Δ Vendi (NGT − Recursive)", fontsize=11)
        ax.set_ylabel("Horizontal: Δ Vendi (NGT − Recursive)", fontsize=11)
        ax.set_title("(c) Cross-Persona Effect Correlation\n(positive r = consistent topology effect)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)

    plt.tight_layout()
    fig2.savefig("./data/appendix_w3_factorial_2x2.png", dpi=300, bbox_inches="tight")
    print("\nSaved: appendix_w3_factorial_2x2.png")
    plt.close(fig2)
else:
    missing = [k for k in factorial_keys if k not in cell_data]
    print(f"  Cannot run 2×2 factorial — missing: {missing}")
    print("  Run the ablation experiments first.")
