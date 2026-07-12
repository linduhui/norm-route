# Stage 3 Routing Matrix Schema

Stage 3 outputs are evaluator-only artifacts. They join Stage 2 expert
predictions with `data/manifests/mvtec_evaluator.csv` labels and masks only
inside `outputs/stage3/evaluator_only`.

These files must never be used as Agent-visible input because they contain
ground-truth `label` and `mask_path` fields.

## Files

- `outputs/stage3/evaluator_only/routing_matrix_long.csv`
- `outputs/stage3/evaluator_only/routing_matrix_wide.csv`
- `outputs/stage3/agent_visible/agent_routing_tasks.jsonl`
- `outputs/stage3/agent_visible/expert_cards.json`

## Long Format

Each row is one sample-expert prediction.

Columns:

- `image_id`
- `dataset`
- `category`
- `support_set_id`
- `k_shot`
- `seed`
- `expert_name`
- `final_score`
- `final_decision`
- `status`
- `error_message`
- `anomaly_map_path`
- `pixel_score_path`
- `label`
- `mask_path`
- `run_dir`

## Wide Format

Each row is one sample/support-set combination, with one score column per
expert.

Columns:

- `image_id`
- `dataset`
- `category`
- `support_set_id`
- `k_shot`
- `seed`
- `label`
- `mask_path`
- `patchcore_score`
- `winclip_score`
- `anomalydino_score`

## Agent-Visible Routing Tasks

`agent_routing_tasks.jsonl` is exported from `routing_matrix_wide.csv` after
dropping evaluator-only fields. Each line is one Agent routing task.

Fields:

- `task_id`
- `sample_id`
- `dataset`
- `category`
- `support_set_id`
- `k_shot`
- `seed`
- `expert_scores`

`expert_scores` contains:

- `PatchCore`
- `WinCLIP`
- `AnomalyDINO`

The file must not contain `label`, `mask_path`, `defect_type`,
`anomaly_type`, or `oracle_best_expert`.

## Expert Cards

`expert_cards.json` contains one card each for PatchCore, WinCLIP, and
AnomalyDINO. Each card contains:

- `method_description`
- `input_requirements`
- `compute_cost`

## Leakage Boundary

Stage 2 `predictions.csv` files are rejected if they contain any of:

- `label`
- `mask_path`
- `defect_type`
- `anomaly_type`

The evaluator-only join reads `label` and `mask_path` from
`data/manifests/mvtec_evaluator.csv` after Stage 2 prediction files have passed
the no-leakage audit.

Oracle artifacts are evaluator-only. Oracle CSV files must include
`evaluator_only=True` and must not be copied into Agent-visible directories.

The three expert columns in `routing_matrix_wide.csv` must be present for every
row. In `routing_matrix_long.csv`, PatchCore, WinCLIP, and AnomalyDINO must have
the same `sample_id`/`image_id` set for each comparison.
