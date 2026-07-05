# NORM-Route Project Rules

This repository studies few-normal-shot industrial anomaly inspection agents.
The agent input is limited to a query image, K normal support images from the
target class, and a tool budget.

## Data Leakage Rules

- Target test label, mask, and defect_type must never enter model or Agent input.
- Support images must come only from the official train/good split.
- All methods in the same comparison must use the same support_set_id.
- Target abnormal samples must never be used to tune thresholds or hyperparameters.

## Required Rules

- target test label、mask、defect_type不得进入模型或Agent输入。
- support只能来自官方train/good。
- 所有方法必须使用相同support_set_id。
- 不得使用目标异常样本调阈值或超参数。
- 不得自动下载数据或权重。
- 不得静默跳过失败样本。
- 所有运行必须保存config、seed、git commit、environment、predictions、failures。
- 每次修改后必须运行pytest。
- Codex不得擅自修改外部baseline核心算法。

## Reproducibility Rules

- Do not automatically download datasets or model weights.
- Do not silently skip failed samples.
- Every run must save config, seed, git commit, environment, predictions, and failures.
- After every change, run pytest before reporting the change as verified.

## Baseline Rules

- Codex must not modify the core algorithms of external baselines unless the user
  explicitly asks for that change.

## Local Development Notes

- Current local development happens in the Windows Codex App workspace.
- The experiment server project path is `/data/gauss/ldh/projects/norm-route`.
- This repository must not create fake experiment results.
- This repository must not implement data loading, models, or Agents until those
  tasks are explicitly requested.
