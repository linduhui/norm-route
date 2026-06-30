# NORM-Route Data Leakage Policy

This policy freezes the V1 separation between Agent-visible inputs and
evaluator-only ground truth.

## Separation Boundary

- `agent_input` and evaluator state must be separated.
- The Agent may read only the fields defined by `InspectionInput`.
- The evaluator may read `label` and `mask` for metric computation.
- The evaluator must not pass `label`, `mask`, `mask_path`, `defect_type`, or
  aggregate target test statistics into Agent-visible inputs.

## Agent and Expert Restrictions

- Experts and Agents must not read target test `label`.
- Experts and Agents must not read target test `mask` or `mask_path`.
- Experts and Agents must not read target test `defect_type`.
- Expert outputs used by the Agent must be computed without evaluator-only
  ground truth.

## Support Set Rules

- Support images must come only from the official `train/good` split.
- All methods in the same comparison must use the same `support_set_id`.
- Support selection must be reproducible from the recorded config and seed.
- Target test samples, including abnormal samples, must never be included in a
  support set.

## Tuning Restrictions

Target abnormal samples must not be used for:

- Threshold selection.
- Prompt selection or prompt tuning.
- Stop strategy selection.
- Hyperparameter selection.
- Budget policy selection.

Any tuning split or validation procedure added in later versions must preserve
the Agent/evaluator separation defined here.
