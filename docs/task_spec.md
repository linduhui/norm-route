# NORM-Route Task Specification

This document freezes the first formal task interface for NORM-Route experiments.
It defines the information an inspection Agent may receive, fields that are
forbidden from Agent input, the allowed later action space, and the prediction
record emitted for evaluation.

No data loading, model behavior, or Agent implementation is defined here.

## InspectionInput

`InspectionInput` is the complete public input contract for an Agent on one
target image. It may contain only:

- `image_id`: Stable identifier for the target test image.
- `query_path`: Path or URI to the target query image.
- `dataset`: Dataset name, such as `mvtec_ad`.
- `category`: Target object category.
- `support_set_id`: Identifier of the fixed normal support set shared by all
  methods in the same comparison.
- `k_shot`: Number of normal support examples available to the Agent.
- `seed`: Reproducibility seed for support selection and stochastic Agent
  behavior.
- `budget`: Maximum allowed tool budget for the Agent run.

## Agent Forbidden Fields

The following fields are forbidden from model input, Agent input, expert input,
tool prompts, routing prompts, stop policies, and any Agent-visible context:

- `label`
- `mask_path`
- `defect_type`
- `test_statistics`

Target test label, mask, and defect type are evaluator-only information. They
must never be exposed to the Agent or to any expert module used by the Agent.

## Later Action Space

The V1 protocol reserves the following later actions. These names define the
allowed action vocabulary for future Agent implementations, but this document
does not implement them:

- `PATCHCORE_GLOBAL`
- `WINCLIP_SEMANTIC`
- `ANOMALYDINO_HIGHRES`
- `TILED_RESCAN`
- `STOP_NORMAL`
- `STOP_ANOMALY`
- `ABSTAIN`

`STOP_NORMAL`, `STOP_ANOMALY`, and `ABSTAIN` are terminal actions.

## Prediction Output Schema

Each evaluated target image must produce exactly one prediction record with the
following fields:

- `image_id`: Stable identifier matching the input target image.
- `final_score`: Final anomaly score used for metric computation.
- `final_decision`: Final discrete decision, such as normal, anomaly, or
  abstain.
- `anomaly_map_path`: Path to the saved anomaly map, or null when no map is
  produced.
- `actions`: Ordered list of Agent actions taken.
- `tool_calls`: Number of tool calls consumed.
- `runtime_ms`: End-to-end runtime in milliseconds for the sample.
- `status`: Execution status for the sample, including success or explicit
  failure state.

Failed samples must be recorded with a non-success `status`; they must not be
silently skipped.
