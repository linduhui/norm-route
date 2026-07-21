# Stage 5 Leakage Policy

This policy is additive to `docs/data_leakage_policy.md`. If two rules differ,
the stricter boundary applies.

## Artifact classes

Stage 5 uses three non-overlapping artifact classes:

1. **Inference-visible**: data serialized for, passed to, or readable by a
   model, router, Agent, expert call, tool prompt, or stop/budget policy.
2. **Runner-only provenance**: task and query locators, fold/split assignment,
   seed, support-set identity, hashes, and execution metadata used to locate
   inputs and reproduce a run. These fields are not automatically model
   features.
3. **Evaluator/training-target only**: ground truth and realized outcomes used
   after inference for evaluation, or on the current fold's permitted training
   partition for constructing supervised targets. These artifacts must be
   stored separately and never copied into inference-visible files.

## Forbidden inference-visible information

Inference-visible files must not contain, at any nesting depth:

- labels or ground truth;
- masks, mask paths, or mask-derived features;
- `defect_type`, `anomaly_type`, or target-test aggregate statistics;
- realized per-expert scores, expert outcomes, or score vectors;
- Oracle identities, selections, scores, regret, or utility;
- teacher scores, targets, or teacher utility.

The prohibition is semantic and case-insensitive. Renaming `label` to
`target_label`, `expert_scores` to `expert_utilities`, or `oracle_best_expert`
to another Oracle-prefixed key does not make it legal. Known score columns such
as `patchcore_score`, `winclip_score`, and `anomalydino_score` are forbidden.

Category, K, budget, and query-derived features are permitted only when the
frozen method specification explicitly allowlists them. Query file-system
paths must not expose a defect directory name to the model; use image bytes or
an opaque/sanitized locator at the model boundary.

## Category isolation

The category is the indivisible Stage 5 split unit. Within one fold:

- train, validation, and test category sets must be pairwise disjoint;
- their union must equal the 15 frozen MVTec categories;
- all K values, seeds, query identities, and support sets for one category must
  receive exactly one split;
- the same query or `support_set_id` must not cross category identities;
- every category must be test exactly once and validation exactly once across
  the five folds.

No outcome from a validation or test category may be used to fit the router.
Validation may select among frozen choices; test is reporting-only. A target
abnormal sample must never be used to tune thresholds or hyperparameters.

## Supports and comparisons

Every support image must come from the official `train/good` split of its
target category. Target test images, including abnormal samples, must never
enter a support set. For a fixed dataset/category/K/seed signature, exactly one
`support_set_id` is allowed, and all compared methods must use it.

## Required audit

`audit_stage5_inputs.py` must run before inference. It recursively checks JSON,
JSONL, CSV, and the dependency-free YAML subset used by this repository. The
audit fails on forbidden field names, evaluator/Oracle path placement,
malformed or blank records, missing files, or an empty scan. It emits a
machine-readable report with input hashes, environment, git commit, counts,
and explicit failures.

An audit pass establishes schema-level separation only. It does not prove that
an image, embedding, free-text prompt, or opaque binary lacks ground-truth
information; producers remain responsible for those semantic checks.

## Evaluation ordering

Evaluation may join immutable predictions with labels and masks only after the
inference output has been finalized. Oracle and teacher analyses must remain
under evaluator/training-target-only storage and must never be used as a
realizable Stage 5 policy result. Failed samples must be retained and reported,
not removed before metric computation.
