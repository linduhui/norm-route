# NORM-Route V1 Experiment Plan

This document freezes the first formal experiment protocol for NORM-Route. V1
is a protocol-only milestone: it adds documents, configuration, and invariant
tests without implementing data loading, models, Agents, or fake results.

## Protocol Overview

- Each run evaluates target test images through the `InspectionInput` contract
  defined in `docs/task_spec.md`.
- Agent-visible input is limited to the query image identity/path, dataset,
  category, fixed `support_set_id`, `k_shot`, seed, and budget.
- Ground-truth labels, masks, defect types, and target test statistics are
  evaluator-only data.
- The action vocabulary is reserved for later implementation and currently
  serves as a stable protocol contract.

## Fold and Support Policy

- MVTec AD category evaluation uses the fixed 5-fold split in
  `configs/folds.yaml`.
- Every method in the same comparison must use the same `support_set_id`.
- Support images must come only from the official `train/good` split.
- Target abnormal samples must not influence thresholds, prompts, stop
  strategies, budget policies, or hyperparameters.

## Required Run Artifacts

Every future executable run must save:

- Config.
- Seed.
- Git commit.
- Environment.
- Predictions.
- Failures.

Prediction records must follow the schema in `docs/task_spec.md`. Failed
samples must be recorded explicitly and must not be silently skipped.

## Current Non-Goals

This V1 freeze does not:

- Implement dataset reading.
- Implement models.
- Implement Agents.
- Create experiment results.
- Access local or server data directories.
- Modify external baseline core algorithms.
