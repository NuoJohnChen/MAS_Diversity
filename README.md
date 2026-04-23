# Diversity Collapse in Multi-Agent LLM Systems

Code for the paper **"Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation."**

> Multi-agent systems (MAS) are increasingly used for open-ended idea generation, driven by the expectation that collective interaction will broaden exploration diversity. However, when and why such collaboration truly expands the solution space remains unclear. We present a systematic empirical study of diversity in MAS-based ideation across three bottom-up levels: model intelligence, agent cognition, and system dynamics. At the model level, we identify a compute efficiency paradox, where stronger, highly aligned models yield diminishing marginal diversity despite higher per-sample quality. At the cognition level, authority-driven dynamics suppress semantic diversity compared to junior-dominated groups. At the system level, group-size scaling yields diminishing returns and dense communication topologies accelerate premature convergence. We characterize these outcomes as *collective failures* emerging from *structural coupling*, a process where interaction inadvertently contracts agent exploration and triggers *diversity collapse*.

## Repository layout

```
MAS_Diversity/
├── simulation/          # MAS pipeline: runners, configs, proposal extraction
│   ├── launch_scientist.py
│   ├── run_dynamic_topic.py            # Standard / NGT / Recursive
│   ├── run_dynamic_topic_subgroup.py   # Subgroup variant
│   ├── extract_txt.py                  # Proposal extraction
│   └── configs/                        # 9 YAML configs (model × persona × topology)
├── analysis/
│   ├── metrics/                        # Vendi, Order Parameter, PCD, Self-BLEU, etc.
│   └── figures/                        # fig02…fig12 (main) + figA1…figA3 (appendix)
├── docs/
│   └── persona_mapping.md              # Paper persona names ↔ config suffixes
├── ASSETS.md                           # Figure → script → required data mapping
├── requirements.txt
└── LICENSE                             # MIT
```

## Setup

```bash
pip install -r requirements.txt

# API keys — metric + embedding scripts need OpenAI; simulation uses provider keys per config
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=...             # if running DeepSeek-V3 configs
# ... other provider keys as needed
```

### External dependencies for simulation

The simulation runners build on two external projects; they are required only if you want to regenerate proposals from scratch. The analysis scripts under `analysis/` do not need them.

- **AgentVerse** (`https://github.com/OpenBMB/AgentVerse`): provides the `agentverse.simulation` primitives that `simulation/run_dynamic_topic.py` imports. Point `MAS_AGENTVERSE_ROOT` at your AgentVerse checkout, or clone it as a sibling of this repo.
- **AI-Scientist** (`https://github.com/SakanaAI/AI-Scientist`): `simulation/launch_scientist.py` delegates to `ai_scientist.generate_ideas` and friends. Make the package importable via `PYTHONPATH` or install it in the same environment.

### Data paths

Scripts resolve data paths relative to `./data/`. Either place generated data under `./data/` or create a symlink (`ln -s /path/to/your/proposals ./data`). Output figures default to `./outputs/`.

## Reproduction workflow

End-to-end reproduction runs in three stages. Intermediate data is not shipped — follow each stage if you want to regenerate from scratch.

### 1. Generate proposals (per config)

```bash
python simulation/launch_scientist.py --config simulation/configs/dsv3_naive_recursive.yaml
python simulation/extract_txt.py --input outputs/dsv3_naive_recursive --output data/extracted_proposals/recursive
```

Each config in `simulation/configs/` specifies one (model × persona × topology) cell from the paper. Run the configs you need for the figures you want to reproduce; `ASSETS.md` lists the input requirement per figure.

### 2. Compute metrics

```bash
# Core Section 2.3 metrics (Vendi Score, Order Parameter, PCD)
python analysis/metrics/compute_vendi_and_order.py

# Extended (+ Self-BLEU, WDistinct-n)
python analysis/metrics/compute_vendi_extended.py

# Section 3 per-model comparison
python analysis/metrics/compute_vendi_sec_models.py

# Appendix sensitivity (OpenAI vs BGE embeddings)
python analysis/metrics/compute_vendi_sensitivity.py
```

These scripts write aggregated CSVs under `./data/tsne/`, `./data/sec_models/`, etc.

### 3. Generate figures

```bash
python analysis/figures/fig02_pareto_landscape.py
python analysis/figures/fig03_persona_bar.py
# ... one script per paper figure
```

Full script-to-figure mapping: see [ASSETS.md](./ASSETS.md).

## What's excluded

To keep the repository lightweight, the following are **not** shipped:
- Raw proposal `.txt` directories (~10,000 proposals across all configurations)
- OpenAI / BGE embedding `.pkl` caches
- LLM-as-judge quality evaluation `.jsonl` caches
- Aggregated metric CSVs (regenerated from stage 2)
- LLM-as-judge batch evaluation scripts (paper-specific infrastructure)

`ASSETS.md` documents what each figure expects under `./data/`.

## Citation

```bibtex
@inproceedings{chen2026masdiversity,
  title={Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation},
  author={Nuo Chen and Yicheng Tong and Yuzhe Yang and Yufei He and Xueyi Zhang and Qingyun Zou and Qian Wang and Bingsheng He},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026},
  note={Available on arXiv:2604.18005}
}
```

## License

MIT. See [LICENSE](./LICENSE).
