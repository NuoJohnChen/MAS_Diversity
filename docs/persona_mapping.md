# Persona ↔ Directory Suffix Mapping

The paper uses five persona structures. Internally, proposal directories use abbreviated suffixes.

| Paper name | Directory suffix | Description |
|------------|------------------|-------------|
| Naive | `rec_final` | No role assignment — baseline |
| Horizontal | `young_final` | Junior peer-to-peer (all first-year PhD) |
| Vertical | `mix_final` | Hierarchical mix (senior + junior) |
| Leader-Led | `leader_experience2` | One designated leader assigns directions |
| Interdisciplinary | `x_final` | Senior experts from different fields |

So for example a Naive × DSV3 run produces proposals under `ai_researcher_multi_topic_dsv3_rec_final/`, and Horizontal × DSV3 produces `ai_researcher_multi_topic_dsv3_young_final/`.

The analysis scripts under `analysis/figures/` and `analysis/metrics/` consume these directory names directly — when adapting the code to new persona variants, keep the suffix naming consistent or update the dataset list at the top of each script.
