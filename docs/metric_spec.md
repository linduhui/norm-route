# NORM-Route Metric Specification

This document freezes the V1 metric set for formal NORM-Route experiments.
It defines what must be reported; it does not prescribe an implementation.

## Detection Metrics

- **Image AUROC**: Area under the receiver operating characteristic curve using
  one final anomaly score per target image.
- **Image AP**: Average precision using one final anomaly score per target
  image.
- **Pixel AUROC**: Area under the receiver operating characteristic curve using
  pixel-level anomaly scores and evaluator-only ground-truth masks.
- **Pixel AP**: Average precision using pixel-level anomaly scores and
  evaluator-only ground-truth masks.
- **AUPRO**: Area under the per-region overlap curve for pixel-level anomaly
  localization, computed from anomaly maps and evaluator-only masks.

## Efficiency and Behavior Metrics

- **Average tool calls**: Mean number of tool calls consumed per evaluated
  target image.
- **Average latency**: Mean end-to-end runtime per evaluated target image.
- **P95 latency**: 95th percentile end-to-end runtime per evaluated target
  image.
- **Abstention rate**: Fraction of evaluated target images whose final decision
  is `ABSTAIN` or an equivalent abstain decision.
- **Oracle regret**: Difference between the Agent outcome and the best outcome
  available from the evaluated action/tool set under evaluator-only hindsight.

All metrics must be computed from saved predictions and evaluator-only labels or
masks. Failed samples must remain visible in run artifacts and must not be
silently removed from metric inputs.
