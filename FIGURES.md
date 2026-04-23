# Figure ↔ Script ↔ Data Mapping

Every paper figure is listed with the script that produced it and the input data the script reads. All data paths below are relative to `./data/` (set `./data` as a symlink to your proposals root, or place data under that directory).

## Main paper

| Fig | File in paper | Script (this repo) | Input data |
|-----|---------------|--------------------|------------|
| 1   | `img/main_fig.pdf` | **hand-authored diagram** (no script) | — |
| 2   | `img/pareto_landscape_final.pdf` | `analysis/figures/fig02_pareto_landscape.py` | `sec_models/metrics_vendi_order.csv`, quality averages under `sec_models/ai_researcher_multi_topic_*/_OVERALL_AVERAGES.txt` |
| 3   | `img/bar_chart_metrics_final.pdf` | `analysis/figures/fig03_persona_bar.py` | `tsne/metrics_vendi_order.csv` |
| 4   | `img/umap.png` | `analysis/figures/fig04_umap.py` | `extracted_proposals/ai_researcher_multi_topic_dsv3_*_final/` + embedding caches |
| 5   | `img/semantic.png` | `analysis/figures/fig05_distance_density.py` | Same 5 persona proposal directories as Fig. 4 |
| 6   | `img/quality_diversity_tradeoff.pdf` | `analysis/figures/fig06_quality_diversity.py` | Persona-level Vendi + quality scores |
| 7   | `img/scaling_gap_analysis.png` | `analysis/figures/fig07_scaling_gap.py` | `extracted_proposals/extracted_proposals_groupsize{3..8}/` |
| 8   | `img/high_dim_stats_with_mmd.pdf` | `analysis/figures/fig08_high_dim_mmd.py` | `extracted_proposals/extracted_proposals_by_round/` — writes `.png`; paper figure was converted to `.pdf` |
| 9   | `img/trajectory.pdf` | `analysis/figures/fig09_trajectory.py` | `extracted_proposals/extracted_proposals_{recursive,ngt,subgroup}/` |
| 10  | `img/diversity_dynamics_final.pdf` | `analysis/figures/fig10_topology_dynamics.py` | `extracted_proposals/extracted_proposals_{recursive,ngt,subgroup}/` |
| 11  | `img/interaction_landscape_with_naive_fixed.pdf` | `analysis/figures/fig11_interaction_landscape.py` | Cross-model × persona × topology proposal dirs under `sec_models/` |
| 12  | `img/intrinsic_task_spectrum_iclr.pdf` | `analysis/figures/fig12_task_spectrum.py` | `extracted_proposals/newtopic/Multi_Collaboration_newtopic{,2,3}/` + `sec_models/dsv3_naive_recursive/extracted_proposals_representationlearning/` |

## Appendix

| Fig | File in paper | Script (this repo) | Input data |
|-----|---------------|--------------------|------------|
| A1  | `figures/gpt51_cross_topology.pdf` | `analysis/rebuttal/gpt51_cross_topology.py` | GPT-5.1 per-topic Vendi CSVs (`tsne_code/pertopic_vendi_gpt51_topologies.csv`) |
| A2  | `img/rebuttal_topic_complexity_v3.pdf` | `analysis/rebuttal/topic_complexity.py` | Group-size extracted proposals (same as Fig. 7) |
| A3  | `img/rebuttal_w3_factorial_2x2.pdf` | `analysis/rebuttal/w3_factorial.py` | W3 factorial CSVs (top-level `rebuttal_w3_*.csv`) |

## Simulation config ↔ paper condition

| Paper condition | Config file |
|-----------------|-------------|
| DSV3 + Naive + Recursive | `simulation/configs/dsv3_naive_recursive.yaml` |
| DSV3 + Naive + NGT | `simulation/configs/dsv3_naive_ngt.yaml` |
| DSV3 + Naive + Subgroup | `simulation/configs/dsv3_naive_subgroup.yaml` |
| DSV3 + Horizontal + Recursive | `simulation/configs/dsv3_horizontal_recursive.yaml` |
| DSV3 + Interdisciplinary + Recursive | `simulation/configs/dsv3_interdisciplinary_recursive.yaml` |
| GPT-5.1 + Interdisciplinary + Recursive | `simulation/configs/gpt5_1_interdisciplinary_recursive.yaml` |
| GPT-5.1 + Interdisciplinary + NGT | `simulation/configs/gpt5_1_interdisciplinary_ngt.yaml` |
| o1-mini + Horizontal + Recursive | `simulation/configs/o1_mini_horizontal_recursive.yaml` |
| o1-mini + Horizontal + Subgroup | `simulation/configs/o1_mini_horizontal_subgroup.yaml` |

The 8-way identical standard runner (`run_dynamic_topic.py`) handles Recursive / NGT topologies; Subgroup uses the dedicated `run_dynamic_topic_subgroup.py`. Topology selection is driven by the `rule` field in each config.

## Metric scripts used for multiple figures

| Script | Produces | Used by |
|--------|----------|---------|
| `analysis/metrics/compute_vendi_and_order.py` | Vendi, Order Parameter, PCD per (persona × topic) | Figs 3, 5, appendix |
| `analysis/metrics/compute_vendi_extended.py` | + Self-BLEU, WDistinct-n | Extended metric table in appendix |
| `analysis/metrics/compute_vendi_sec_models.py` | Per-model metrics (DSV3, GPT-5.1, o1-mini, Grok-4) | Fig 2 |
| `analysis/metrics/compute_vendi_sensitivity.py` | OpenAI-vs-BGE embedding sensitivity | Sensitivity appendix |
| `analysis/metrics/compute_proposal_metrics.py` | Proposal-level summaries | Aggregate tables |
