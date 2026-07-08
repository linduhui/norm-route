# Stage 2 Report: Baseline Expert Reproduction

## 1. Goal

本阶段目标是接入 PatchCore、WinCLIP、AnomalyDINO 三个视觉专家，并在统一 few-normal-shot 协议下输出可比较的 Stage 2 运行产物。

当前仓库中的三个专家实现是 no-download、support-guided 的本地 wrapper，用于验证 Stage 2 协议、输入隔离、输出格式和批量运行流程。它们没有自动下载数据或权重，也没有修改外部 baseline 核心算法。

## 2. Protocol

- dataset: MVTec AD (`mvtec`)
- categories: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`
- k_shot: `1`, `2`, `4`, `8`
- seeds: `0`, `1`, `2`, `3`, `4`
- support_set_id: `mvtec_{category}_k{k_shot}_seed{seed}`
- metrics: current saved metrics are run-accounting metrics: `num_predictions`, `num_success`, `num_failed`, `average_tool_calls`, `average_runtime_ms`, `abstention_rate`

Each run consumes:

- `data/manifests/mvtec_agent_input.csv`
- one fixed `data/support_sets/mvtec_k{k_shot}_seed{seed}.csv`
- the target query image path
- the category, seed, k-shot, support-set id, and tool budget

## 3. Expert Implementations

### PatchCore

PatchCore is implemented as a local PatchCore-style nearest-neighbor wrapper. It builds a support memory bank from train/good support image patches, scores query patches by nearest support distance, writes `anomaly_maps/`, and emits one prediction row per query image.

Key config:

- `image_size: 64`
- `patch_size: 16`
- `stride: 8`
- `anomaly_threshold: 0.5`
- `budget: 1`

### WinCLIP

WinCLIP is implemented as a local WinCLIP-style prompt and support wrapper. It combines support-image global features with category-name normal/anomaly prompt templates, writes `anomaly_maps/`, and emits one prediction row per query image.

Key config:

- `image_size: 64`
- `grid_size: 8`
- `anomaly_threshold: 0.5`
- `budget: 1`
- category prompts are rendered from the fixed class-name map and prompt templates in `configs/stage2/winclip_mvtec.yaml`

### AnomalyDINO

AnomalyDINO is implemented as a local AnomalyDINO-style token nearest-neighbor wrapper. It builds a support token bank from train/good support images, scores query tokens against the support bank, writes `anomaly_maps/` and `pixel_scores/`, and emits one prediction row per query image.

Key config:

- `image_size: 64`
- `token_grid_size: 8`
- `anomaly_threshold: 0.5`
- `image_score_top_fraction: 0.1`
- `budget: 1`

## 4. Output Format

Each run directory is expected to contain:

- `predictions.csv`
- `metrics.json`
- `failures.json`
- `run_metadata.json`
- `anomaly_maps/`

AnomalyDINO additionally writes:

- `pixel_scores/`

The Stage 2 summary CSV records one row per expert/dataset/category/k-shot/seed/support-set run and points back to each run's `metrics.json`.

## 5. No-leakage Check

专家输入中没有 `label`, `mask_path`, `defect_type`, `anomaly_type`。

The shared expert input contract only exposes:

- `image_id`
- `query_path`
- `dataset`
- `category`
- `support_set_id`
- `k_shot`
- `seed`
- `budget`
- `support_paths`

The expert interface rejects evaluator-only fields before fit or predict. Support paths are also validated to come from `train/good`.

## 6. Results

Canonical summary file:

- `reports/stage2/stage2_summary.csv`

At the time of this report, the canonical `stage2_summary.csv` contains PatchCore rows only:

| expert | rows | predictions | failed | avg runtime ms |
|---|---:|---:|---:|---:|
| patchcore | 300 | 34,500 | 0 | 54.66 |

Full local reproduction candidate:

- `reports/stage2/mvtec_full_summary.csv`

This file contains all three experts:

| expert | rows | predictions | failed | avg runtime ms |
|---|---:|---:|---:|---:|
| anomalydino | 300 | 34,500 | 0 | 69.61 |
| patchcore | 300 | 34,500 | 0 | 54.58 |
| winclip | 300 | 34,500 | 0 | 78.31 |

Important limitation: current `metrics.json` files do not contain label-dependent quality metrics such as AUROC, AP, F1, pixel AUROC, or PRO. The table above is therefore a run-completion and runtime summary, not a detection-quality comparison.

## 7. Failure Cases

`reports/stage2/mvtec_full_failed_combos.json` is an empty list.

Observed failure summary:

| failed category | failure reason | affects summary table |
|---|---|---|
| none | no failed combos recorded | no |

## 8. Observations

The current artifacts support accounting-level observations only:

- PatchCore completed all 300 full-grid runs and is the fastest of the three local wrappers on average.
- AnomalyDINO completed all 300 full-grid runs and writes both anomaly maps and token-level pixel-score CSVs.
- WinCLIP completed all 300 full-grid runs and is the slowest of the three local wrappers on average in this local run.
- All three experts use the same support-set id convention for each category, k-shot, and seed.

The following questions cannot be answered from the current saved metrics without adding label-isolated evaluation outputs:

- 哪些类别 PatchCore 强？
- 哪些类别 WinCLIP 强？
- 哪些类别 AnomalyDINO 强？
- 是否存在明显 expert complementarity？

To answer complementarity without leakage, evaluation should happen after prediction export by joining `predictions.csv` with evaluator-only manifests outside the expert input path, then writing aggregate quality metrics separately.

## 9. Next Step

进入阶段三：

- Oracle expert selection
- Agent baseline
- Router upper bound

Before Stage 3 routing experiments, the Stage 2 quality evaluation should be materialized in a label-isolated summary so oracle and router comparisons can optimize against evaluator-only metrics without exposing labels, masks, defect types, or anomaly types to experts or Agents.
