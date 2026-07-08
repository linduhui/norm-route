# Stage 2 Report: Baseline Expert Reproduction

## Goal

Stage 2 connects three baseline visual experts under one few-normal-shot protocol:
PatchCore, WinCLIP, and AnomalyDINO. The objective is to produce comparable
per-run artifacts for the same dataset/category/k-shot/seed/support-set grid.

The current implementations are local no-download wrappers that exercise the
Stage 2 protocol and output contract without modifying external baseline core
algorithms or downloading data/weights.

## Protocol

- dataset: MVTec AD (`mvtec`)
- categories: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
  `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`,
  `transistor`, `wood`, `zipper`
- k_shot: `1`, `2`, `4`, `8`
- seeds: `0`, `1`, `2`, `3`, `4`
- support_set_id: `mvtec_{category}_k{k_shot}_seed{seed}`
- budget: `1` tool call per expert prediction

All experts consume the same support-set id for the same
dataset/category/k-shot/seed combination.

## Expert Implementations

PatchCore builds a support patch memory bank from train/good images and scores
query patches by nearest-neighbor distance.

WinCLIP combines support-image global features with fixed category-name normal
and anomaly prompt templates.

AnomalyDINO builds a support token bank from train/good images and scores query
tokens by nearest-neighbor distance. It also writes token-level pixel score CSVs.

## Output Format

Every run directory must contain:

- `predictions.csv`
- `metrics.json`
- `failures.json`
- `run_metadata.json`
- `anomaly_maps/`

AnomalyDINO also writes `pixel_scores/`.

The shared validator checks prediction columns, required files, failure counts,
forbidden leakage columns, and support-set consistency across experts.

## No-Leakage Check

Expert-visible input is limited to image/query metadata and support paths:

- `image_id`
- `query_path`
- `dataset`
- `category`
- `support_set_id`
- `k_shot`
- `seed`
- `budget`
- `support_paths`

The expert interface rejects evaluator-only fields, including `label`,
`mask_path`, `defect_type`, and `anomaly_type`. Support paths are validated to
come from the official `train/good` split.

## Results

Canonical comparison table:

- `reports/stage2/stage2_summary.csv`

The canonical summary now covers all three experts on the full MVTec grid:

| expert | rows | predictions | failed | avg runtime ms |
|---|---:|---:|---:|---:|
| anomalydino | 300 | 34,500 | 0 | 69.61 |
| patchcore | 300 | 34,500 | 0 | 54.58 |
| winclip | 300 | 34,500 | 0 | 78.31 |

This is a run-completion and runtime comparison. The currently saved
`metrics.json` files do not yet include evaluator-only quality metrics such as
AUROC, AP, F1, pixel AUROC, or PRO.

## Failure Cases

`reports/stage2/mvtec_full_failed_combos.json` is an empty list.

| failed category | failure reason | affects summary table |
|---|---|---|
| none | no failed combinations recorded | no |

## Observations

All three experts completed the same MVTec grid with zero recorded failures and
zero abstention. PatchCore is fastest on average in this local run, followed by
AnomalyDINO and WinCLIP.

Category-level expert strength and complementarity cannot be concluded from the
current accounting-only metrics. That analysis should be done by a
label-isolated evaluator that joins predictions with evaluator-only labels after
expert inference is complete.

## Next Step

Stage 2 is ready to feed Stage 3 routing experiments:

- Oracle expert selection
- Agent baseline
- Router upper bound

The Stage 3 evaluator must keep labels, masks, defect types, and anomaly types
out of expert and Agent inputs.
