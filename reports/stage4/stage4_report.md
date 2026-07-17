# Stage 4 最终 Gate 报告

> Gate 状态：**PASS**。本报告仅总结当前 Stage 4 协议和本地 wrapper 实验，**不构成最终 NORM-Route SOTA 声明**。

## 1. 结论与范围

- Expert implementation mode：Local no-download, support-guided PatchCore-style, WinCLIP-style, and AnomalyDINO-style wrappers used to validate the protocol and routing stack; these are not the official external baseline implementations.
- 当前数值用于验证输入隔离、五折路由、预算、回放、评估和 live wiring；不得与官方实现结果或最终 SOTA 等同。
- 选择口径：五折 test 的 `micro` `auroc_mean` 均值；Oracle 是每折独立重算的 evaluator-only run-level reference。

## 2. Gate 校验

| Check | 状态 | 摘要 |
| --- | --- | --- |
| `pre_route_tasks_no_forbidden_fields` | PASS | validated 34500 pre-route tasks |
| `policy_feature_manifest_no_leakage` | PASS | validated 10 learned-policy feature manifests |
| `train_val_test_isolation` | PASS | validated 172500 assignments across 5 folds |
| `route_decisions_one_expert_per_run` | PASS | validated exactly one expert decision for 345000 routing tasks across 50 run directories |
| `tool_calls_at_most_one` | PASS | validated tool_calls <= 1 across 345000 decisions and 345000 predictions |
| `selected_predictions_match_selected_expert` | PASS | aligned 345000 selected predictions to route decisions |
| `failures_match_metrics` | PASS | reconciled 0 recorded failures with run counts |
| `reproducibility_artifacts` | PASS | validated required artifacts and provenance for 50 runs |
| `failures_match_evaluation_metrics` | PASS | reconciled 50 metric rows and evaluator failure counts |
| `evaluator_after_route_decision` | PASS | validated evaluator ordering for 50 routed inputs |
| `live_smoke` | PASS | validated latest live smoke run with 1 prediction(s) |
| `stage4_report_inputs` | PASS | validated 36 policy-summary rows, 180 cost-quality rows, and five-fold config |

## 3. 五折协议

- split unit：`seed`；fold 数：5。
- 同一 fold 的 train/val/test seed 两两不重叠；每个 seed 恰好一次作为 test、一次作为 validation。

| Fold | Train seeds | Val seed | Test seed |
| --- | --- | --- | --- |
| `fold0` | 2, 3, 4 | 1 | 0 |
| `fold1` | 0, 3, 4 | 2 | 1 |
| `fold2` | 0, 1, 4 | 3 | 2 |
| `fold3` | 0, 1, 2 | 4 | 3 |
| `fold4` | 1, 2, 3 | 0 | 4 |

## 4. 五折结果摘要

| 组别 | 最佳方法 | AUROC | AP | F1 | 估计耗时/样本 (ms) | Oracle AUROC gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed baseline | `fastest_expert` | 0.514017 | 0.733303 | 0.120048 | 53.667231 | 0.012154 |
| 规则 Agent | `cost_aware` | 0.522179 | 0.756288 | 0.244223 | 64.812957 | 0.003992 |
| Learned metadata baseline | `decision_tree_metadata` | 0.519345 | 0.753313 | 0.265928 | 66.257066 | 0.006826 |
| Fold-matched run-level Oracle | `run_level_oracle` | 0.526171 | 0.759199 | 0.244962 | 64.879045 | 0.000000 |

- 最佳 fixed baseline：`fastest_expert`。
- 最佳规则 Agent：`cost_aware`。
- 最佳 learned metadata baseline：`decision_tree_metadata`。该方法是诊断 baseline，不是最终 NORM-Route 模型。
- 最佳可实现策略到 fold-matched Oracle 的 AUROC 差距：0.003992（`cost_aware` 对 `run_level_oracle`）。

## 5. Cost-quality tradeoff

- 低成本端：`fastest_expert`，AUROC=0.514017，估计 53.667231 ms/样本。
- 高质量端：`cost_aware`，AUROC=0.522179，估计 64.812957 ms/样本。
- 从低成本端切换到高质量端：AUROC +0.008162，估计耗时 +11.145726 ms/样本（+20.768216%）。回放耗时是 Stage 2 历史估计值，不是本次回放 wall-clock。
- 可实现策略的均值 Pareto frontier：`fastest_expert` (0.514017, 53.667231 ms), `cost_aware` (0.522179, 64.812957 ms)。

## 6. Live smoke

- 状态：**PASS**；policy=`category_shot_prior`；selected expert=`PatchCore`。
- 输入：dataset=`mvtec`，category=`bottle`，K=1，seed=4。
- 决策/预测：1/1；actual subprocess runtime=205.400667 ms。
- evaluator 输入哈希与 routed selected_predictions 一致，并在 live run 执行窗口内完成；单样本 smoke 只证明链路闭环。

## 7. 当前限制

- All quality numbers come from local no-download support-guided wrappers, not official external baseline implementations or a final NORM-Route model.
- The current Stage 4 full-CV report covers MVTec; it is not a multi-dataset SOTA comparison.
- Replay costs are historical estimated runtimes copied from immutable Stage 2 outputs; only the one-sample live smoke records actual subprocess runtime.
- F1 uses saved wrapper decisions and frozen thresholds; target abnormal test samples were not used for threshold or hyperparameter tuning.
- The live smoke contains one routed sample, so it validates wiring and ordering rather than production latency or quality.

## 8. Stage 5 方向

- Integrate official, version-pinned baseline adapters and user-provided weights without automatic downloads; preserve external core algorithms.
- Freeze a leakage-safe learned router with query-visible features only, then repeat fold-isolated calibration and testing.
- Keep exactly one auditable expert decision per routing task and extend routing tests to future query-visible features.
- Expand live execution beyond a one-sample smoke and report actual latency, memory, failures, and quality with uncertainty.
- Add VisA and other held-out datasets under the same support_set_id and evaluator-isolation protocol.

## 9. 可复现性

- Gate protocol：`stage4.gate.v1`
- Generated at UTC：`2026-07-17T04:19:57.729238+00:00`
- Git commit：`4b879105958e7706c0dbb9e4e9dfa716250aad54`
- Runs root：`outputs\stage4\runs\full_cv_20260716_060523`
- Evaluation root：`outputs\stage4\evaluation\full_cv_20260716_060523`
- 记录的输入 SHA-256：10 个。
