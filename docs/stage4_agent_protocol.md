# Stage 4 Agent Task and Seed-CV Protocol

This document freezes the Stage 4 pre-route task boundary, five-fold seed
split, and replay-only Agent interface. The replay executor selects exactly
one expert per task and reads that expert's immutable Stage 2
`predictions.csv`; it does not run or modify an expert algorithm.

## Inputs and outputs

The task builder reads:

- `outputs/stage3/routing_matrix/agent_routing_tasks.jsonl`

It writes:

- `outputs/stage4/tasks/pre_route_tasks.jsonl`

The split builder reads that Stage 4 task file and
`configs/stage4/seed_cv.yaml`. It writes:

- `outputs/stage4/splits/fold_manifest.csv`
- `outputs/stage4/splits/split_audit.json`

The checked-in config is the source of truth for the split. The audit embeds
the parsed config, input hashes, git commit, Python version, platform, counts,
and an explicit empty `failures` list. A malformed or failed input row aborts
the build; rows are never silently skipped.

## Pre-route task schema

Each JSONL row has exactly these fields:

| Field | Purpose | Policy feature? |
| --- | --- | --- |
| `protocol_version` | Schema version (`stage4.pre_route.v1`) | No |
| `task_id` | Unique run-task identity | No |
| `sample_id` | Query identity for locating and recording the run | No |
| `dataset` | Dataset identity | Yes |
| `category` | Target class | Yes |
| `k_shot` | Number of fixed normal supports | Yes |
| `seed` | Support realization and split provenance | No |
| `support_set_id` | Fixed support-set locator shared by methods | No |
| `candidate_experts` | Fixed ordered routing action vocabulary | No |
| `policy_features` | The complete policy-readable mapping | Boundary |

For example:

```json
{
  "protocol_version": "stage4.pre_route.v1",
  "task_id": "example-task",
  "sample_id": "example-sample",
  "dataset": "mvtec",
  "category": "bottle",
  "k_shot": 1,
  "seed": 0,
  "support_set_id": "mvtec_bottle_k1_seed0",
  "candidate_experts": ["PatchCore", "WinCLIP", "AnomalyDINO"],
  "policy_features": {"dataset": "mvtec", "category": "bottle", "k_shot": 1}
}
```

`candidate_experts` names legal actions; it contains no expert outcome. Stage 3
`expert_scores` is intentionally discarded during conversion and is not part
of the Stage 4 schema.

## Policy feature boundary

The exhaustive policy feature allowlist is:

- `dataset`
- `category`
- `k_shot`

A routing policy must receive only the `policy_features` object, not the full
task row. In particular, `seed` and `support_set_id` may be used by the runner
to assign a fold, locate the fixed official `train/good` support set, and record
reproducibility metadata. They are provenance fields and are forbidden from the
policy feature allowlist. `sample_id`, `task_id`, and `protocol_version` are
also excluded so that the policy cannot memorize query identities.

## Forbidden fields

The following keys are rejected at every nesting depth in a pre-route task:

- `label`
- `mask_path`
- `defect_type`
- `anomaly_type`
- `oracle_best_expert`
- `patchcore_score`
- `winclip_score`
- `anomalydino_score`

The exact task schema additionally rejects all unrecognized top-level keys,
including Stage 3 `expert_scores`. Ground-truth labels, masks, defect types,
Oracle decisions, and realized expert scores must remain outside the Agent and
policy boundary. Target abnormal samples must not be used to tune routing
rules, thresholds, prompts, budgets, or hyperparameters.

## Fixed five-fold seed split

The split unit is the complete support-selection `seed`, not an individual
task. Every task with a given seed receives the same split within a fold.

| Fold | Train seeds | Validation seed | Test seed |
| --- | --- | --- | --- |
| `fold0` | 2, 3, 4 | 1 | 0 |
| `fold1` | 0, 3, 4 | 2 | 1 |
| `fold2` | 0, 1, 4 | 3 | 2 |
| `fold3` | 0, 1, 2 | 4 | 3 |
| `fold4` | 1, 2, 3 | 0 | 4 |

Within every fold, the train, validation, and test seed sets must be pairwise
disjoint and must cover seeds 0 through 4 exactly once. Across the five folds,
each seed is used as test exactly once and validation exactly once.

Because this protocol splits support realizations, the same `sample_id` can
occur with different seeds in different partitions. Identity fields are not
policy features. Any future learned router must preserve that boundary and may
use evaluator-only outcomes only from the current fold's training partition;
validation is for selection and test is for final reporting only.

## Manifest and audit schemas

`fold_manifest.csv` contains one row per `(fold, task_id)` with:
`fold`, `split`, `task_id`, `sample_id`, `dataset`, `category`, `k_shot`,
`seed`, and `support_set_id`. A task appears exactly once in each fold.

`split_audit.json` records the configured seeds, observed seeds, per-split task
counts, pairwise seed intersections, unassigned and duplicate assignment
counts, hashes, and provenance. `all_folds_valid` must be `true` before the
manifest may be used.

## Commands

Run from the repository root:

```powershell
python -m src.normroute.cli.build_stage4_tasks
python -m src.normroute.cli.build_stage4_splits
python scripts/calibrate_policy.py --policy category_shot_prior
python -m src.normroute.cli.run_agent --policy always_anomalydino --fold fold0 --split test
python -m src.normroute.cli.run_agent --policy category_shot_prior --policy-artifact outputs/stage4/policies/category_shot_prior/fold0/policy_artifact.json --fold fold0 --split test
pytest
```

## Fold-specific rule policies

`calibrate_policy.py` joins Stage 3 `expert_quality_by_run.csv` to the frozen
fold manifest by run identity and parses quality/runtime values only for the
requested fold's `train` rows. `category_prior` selects the best mean-training
expert for `(dataset, category)`. `category_shot_prior` first uses
`(dataset, category, k_shot)`, then falls back to category and finally the
global train-fold best. The default optimization metric is `image_auroc`;
`image_ap` is selectable. Metric ties use lower mean training runtime, followed
only when runtime is also tied by the frozen candidate-expert order.
The runtime tie-break reads `average_runtime_ms`, propagated into the Stage 3
per-run quality table from immutable Stage 2 prediction runtimes.

Each fold writes a compact `policy_artifact.json` containing selected rules,
train seeds, metric, tie-break contract, git commit, and hashes computed only
from training rows. It contains no labels, masks, test quality, or raw quality
values.

## Replay outputs

Each replay run writes `route_decisions.csv`, `selected_predictions.csv`,
`failures.json`, `run_metadata.json`, and `budget_summary.json`. The metadata
captures the complete replay config, seeds, git commit, environment, and input
hashes. Missing Stage 2 runs, missing/duplicate prediction matches, failed
predictions, mixed-expert Stage 2 run files, and budget violations are recorded
as explicit task failures; none are silently skipped.
