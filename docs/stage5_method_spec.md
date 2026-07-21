# Stage 5 NORM-Route Category-Held-Out Protocol

This document freezes the Stage 5 protocol skeleton. It does not implement a
dataset loader, model, Agent, expert, or experiment result. Stage 5 inherits
the Stage 1 input/evaluator boundary, the Stage 2 shared-support contract, the
Stage 3 evaluator-only Oracle boundary, and the Stage 4 auditable routing and
failure-recording rules.

## Objective

Stage 5 measures transfer to categories that are absent from router training
and validation. The split unit is the complete category. It replaces the Stage
4 seed-held-out protocol for Stage 5 experiments; it does not change or
reinterpret any Stage 1--4 result.

The checked-in source of truth is
`configs/stage5/category_cv_v2.yaml`, with protocol version
`stage5.category_cv.v2`.

## Frozen category groups

| Group | Categories |
| --- | --- |
| `G0` | `carpet`, `bottle`, `cable` |
| `G1` | `grid`, `capsule`, `hazelnut` |
| `G2` | `leather`, `metal_nut`, `pill` |
| `G3` | `tile`, `screw`, `toothbrush` |
| `G4` | `wood`, `transistor`, `zipper` |

For fold `i`, test is `Gi`, validation is `G(i+1 mod 5)`, and training is the
other three groups.

| Fold | Train groups | Validation group | Test group |
| --- | --- | --- | --- |
| `fold0` | `G2`, `G3`, `G4` | `G1` | `G0` |
| `fold1` | `G0`, `G3`, `G4` | `G2` | `G1` |
| `fold2` | `G0`, `G1`, `G4` | `G3` | `G2` |
| `fold3` | `G0`, `G1`, `G2` | `G4` | `G3` |
| `fold4` | `G1`, `G2`, `G3` | `G0` | `G4` |

Within a fold, category membership alone determines the split. Consequently,
every row for that category remains together, including all `k_shot` values,
seeds, query identities, and `support_set_id` values. Splitting, sampling, or
deduplicating individual rows after category assignment is forbidden.

## Inputs and split artifacts

`build_stage5_splits.py` consumes a complete inference-visible JSONL or CSV
task table. Each row must provide:

- `task_id`
- `dataset`
- `category`
- `k_shot`
- `seed`
- `support_set_id`
- one query identity field: `sample_id`, `image_id`, `query_id`, or
  `query_path`

The builder fails on blank/malformed rows, duplicate task ids, missing or
unexpected categories, inconsistent support-set identities, or forbidden
inference fields. It writes:

- `outputs/stage5/splits/fold_manifest.csv`
- `outputs/stage5/splits/split_audit.json`

The manifest contains only `fold`, `split`, task/query identity, dataset,
category, K, seed, and support-set provenance. The audit records the frozen
config and its hash, input hash, git commit, Python/platform environment,
counts, category intersections, duplicate/unassigned counts, category-bundle
consistency, and explicit failures.

## Training, validation, and test use

- Training categories may fit model parameters.
- Validation categories may select a frozen hyperparameter, threshold, prompt,
  stop strategy, or budget policy.
- Test categories are used only after all parameters and choices for the fold
  are frozen.
- Target abnormal test samples must never tune any parameter or policy.
- Evaluator-only outcomes may score a completed validation or test run, but
  they must not be joined into an inference-visible row.

Support images remain few-normal-shot inputs for their target category. They
must come only from the official `train/good` split. Every method in one
comparison must receive the identical `support_set_id`; category-held-out
means that router training is held out, not that normal supports are removed
at inference.

## Inference boundary

Before an inference-visible artifact is used, it must pass
`audit_stage5_inputs.py` and the rules in `docs/stage5_leakage_policy.md`.
Inference-visible files must not contain labels, masks, defect/anomaly types,
target test statistics, realized expert scores, Oracle information, or teacher
utility. This rule applies recursively to nested JSON and to CSV/YAML headers.

The split manifest is runner provenance, not a policy feature bundle. A model
or Agent may receive only the fields explicitly defined for its frozen Stage 5
input schema. Fold/split labels, task identity, query file-system names, seed,
and `support_set_id` must not be promoted into learned features merely because
they are present in runner metadata.

## Reproducibility and failure handling

Every future Stage 5 run must save config, seed, git commit, environment,
predictions, and failures. Every target query must yield either one prediction
record or one explicit failure record. Missing inputs, parse errors, support
failures, and audit failures are fatal; no row may be silently skipped.

No script in this skeleton downloads data or weights, creates fake results, or
changes an external baseline algorithm.

## Frozen visual backbone

Stage 5 router image features use
`src/normroute/router/feature_provider.py`. The preferred lightweight encoder
is DINOv2-S/14 (`dinov2_vits14`, 518-pixel input); an explicitly configured
equivalent lightweight `timm` architecture may be used for an ablation. The
provider always constructs `timm` models with `pretrained=False`, loads only a
local checkpoint, switches the model to evaluation mode, and disables gradients
for every parameter.

The effective backbone config must record `checkpoint_path`, `sha256`,
`architecture`, `input_size`, and `frozen: true`. Relative checkpoint paths are
resolved from the config file directory. Copy
`configs/stage5/router_backbone.example.yaml`, point it at the exact local
checkpoint, and replace the sentinel hash with that file's SHA-256. Neither the
feature provider nor the audit script downloads missing weights. A missing file,
hash mismatch, incompatible state dict, train-mode model, or trainable parameter
is a hard failure.

Install the optional local environment with `pip install -e ".[stage5]"` only
after the required packages and checkpoint are available under the project's
no-automatic-download policy. Before a router run, save the audit emitted by
`audit_router_backbone.py` alongside the run config and other required
reproducibility artifacts.

## Commands

From the repository root:

```powershell
python scripts/build_stage5_splits.py --input outputs/stage4/tasks/pre_route_tasks.jsonl
python scripts/audit_stage5_inputs.py outputs/stage5/splits/fold_manifest.csv
python scripts/audit_router_backbone.py configs/stage5/router_backbone.yaml
pytest
```
