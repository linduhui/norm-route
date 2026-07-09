# Stage 3 Routing Matrix Schema

Stage 3 outputs are evaluator-only artifacts. They join Stage 2 expert
predictions with `data/manifests/mvtec_evaluator.csv` labels and masks only
inside `outputs/stage3/evaluator_only`.

These files must never be used as Agent-visible input because they contain
ground-truth `label` and `mask_path` fields.

## Files

- `outputs/stage3/evaluator_only/routing_matrix_long.csv`
- `outputs/stage3/evaluator_only/routing_matrix_wide.csv`

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

## Leakage Boundary

Stage 2 `predictions.csv` files are rejected if they contain any of:

- `label`
- `mask_path`
- `defect_type`
- `anomaly_type`

The evaluator-only join reads `label` and `mask_path` from
`data/manifests/mvtec_evaluator.csv` after Stage 2 prediction files have passed
the no-leakage audit.

