# Stage 3 Report: Expert Comparability and Oracle Routing Upper Bound

## 1. Goal

Stage 3 evaluates whether the three Stage 2 anomaly-inspection experts are
comparable under the same few-normal-shot setting, and whether there is enough
complementarity to justify a later routing Agent.

The stage builds an evaluator-only routing matrix from expert predictions,
computes true expert quality with labels and masks kept outside Agent input, and
estimates Oracle upper bounds. The Oracle results are not Agent-visible; they
only quantify the maximum possible gain from choosing experts with evaluator
knowledge.

## 2. Inputs

Stage 3 uses:

- Stage 1 manifests: `data/manifests/mvtec_agent_input.csv` and
  `data/manifests/mvtec_evaluator.csv`
- Stage 1 support sets: `data/support_sets/mvtec_k{1,2,4,8}_seed{0..4}.csv`
- Stage 2 expert predictions under `outputs/stage2`
- evaluator-only labels and masks from `data/manifests/mvtec_evaluator.csv`

The current MVTec Stage 3 artifacts cover 34,500 sample/support rows in wide
format and 103,500 sample/expert rows in long format.

## 3. No-Leakage Boundary

Agent-visible files may contain query identity, dataset/category metadata,
support-set identity, shot/seed, expert cards, and expert scores. They must not
contain ground-truth or Oracle fields:

- `label`
- `mask_path`
- `defect_type`
- `anomaly_type`
- `oracle_best_expert`

Evaluator-only files may contain labels and masks, but they are restricted to
`outputs/stage3/evaluator_only`, `outputs/stage3/oracle`, and report summaries.
The exported Agent-visible routing tasks are stored in
`outputs/stage3/routing_matrix/agent_routing_tasks.jsonl`, with expert metadata
in `outputs/stage3/routing_matrix/expert_cards.json`.

## 4. Routing Matrix

`routing_matrix_long.csv` has one row per sample/expert prediction. It includes
the shared sample keys, `expert_name`, prediction score/decision, failure status,
and evaluator-only `label` and `mask_path`.

`routing_matrix_wide.csv` has one row per sample/support-set combination. It
pivots the three expert scores into:

- `patchcore_score`
- `winclip_score`
- `anomalydino_score`

The wide matrix is the source for Agent-visible routing tasks after dropping
evaluator-only fields.

## 5. Expert Quality

`reports/stage3/expert_quality_summary.csv` summarizes true evaluator-only
quality by expert, category, and k-shot. Averaged over the 60 category/shot
groups, the image-level results are:

| Expert | Image AUROC | Image AP | Image F1 max |
| --- | ---: | ---: | ---: |
| AnomalyDINO | 0.633134 | 0.829202 | 0.858314 |
| PatchCore | 0.623977 | 0.827245 | 0.853089 |
| WinCLIP | 0.482622 | 0.741115 | 0.840822 |

AnomalyDINO is the best single expert on average, but PatchCore is very close in
AUROC/AP and wins several categories. WinCLIP is weaker overall, but still wins
specific low-shot/category regimes.

## 6. Complementarity

`reports/stage3/complementarity_summary.csv` shows that the best expert changes
by category. Category-level winners are:

- AnomalyDINO: 10 categories: cable, capsule, grid, hazelnut, pill, screw, tile,
  toothbrush, transistor, wood
- PatchCore: 4 categories: bottle, leather, metal_nut, zipper
- WinCLIP: 1 category: carpet

By k-shot, AnomalyDINO is the best aggregate expert at all evaluated shots, and
its mean AUROC increases from 0.589201 at k=1 to 0.668541 at k=8. However, the
category winners and per-run selection counts vary, so the experts are not
interchangeable.

## 7. Oracle Upper Bound

`reports/stage3/oracle_summary.csv` reports evaluator-only upper bounds:

| Method | Image AUROC | Image AP | Image F1 max |
| --- | ---: | ---: | ---: |
| Best single expert | 0.633134 | 0.829202 | 0.858314 |
| Run-level Oracle | 0.680013 | 0.852690 | 0.862408 |
| Sample-level Oracle | 0.988257 | 0.995662 | 0.988787 |

The run-level Oracle improves image AUROC by 0.046878 over the best single
expert. Across 300 run selections, the Oracle chooses AnomalyDINO 155 times,
PatchCore 109 times, and WinCLIP 36 times. This confirms that a fixed global
expert leaves measurable performance on the table.

The sample-level Oracle is much higher, but it uses ground-truth sample labels
and is only an upper bound. It must not be interpreted as an achievable Agent
policy.

## 8. Routing Implication

Stage 3 supports building an Agent/Router. The useful target is not to beat the
sample-level Oracle, but to close part of the gap between the best single expert
and the run-level Oracle using only Agent-visible information.

The strongest near-term signal is category/shot-aware routing: category winners
vary, k-shot affects relative performance, and the run-level Oracle repeatedly
selects multiple experts. A router should start simple and auditable before any
learned policy is introduced.

## 9. Next Stage

Stage 4 should implement and compare:

- rule-based Agent
- fixed policy router
- lightweight learned router
- budget-aware tool selection

All Stage 4 policies must consume only Agent-visible inputs and must continue to
save config, seed, git commit, environment, predictions, and failures.
