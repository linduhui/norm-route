# Stage 5 Soft Teacher and Expert Capability Profile Bank

This pipeline keeps evaluator supervision physically separate from Router
inference features. The teacher accepts only a `routing_matrix*` file below an
`evaluator_only` directory, serializes train-category rows only, and filters
test categories before parsing label, score, status, runtime, or expert fields.

## Fold pipeline

The Stage 3 long matrix is the authoritative per-query latency source. Rebuild
it from the same Stage 2 prediction root that produced the Stage 2 summary;
legacy matrices without a complete `runtime_ms` column are intentionally
rejected. Write the rebuilt matrix to a new directory so prior results remain
recoverable.

```bash
python -m normroute.cli.build_routing_matrix \
  --stage2-root outputs/stage2_mvtec_full \
  --evaluator-csv data/manifests/mvtec_evaluator.csv \
  --output-dir outputs/stage3/runtime_v3/evaluator_only
```

```bash
python -m normroute.cli.build_teacher_targets \
  --routing-matrix outputs/stage3/runtime_v3/evaluator_only/routing_matrix_long.csv \
  --fold-manifest outputs/stage5/splits/fold_manifest.csv \
  --fold fold0 \
  --output-dir outputs/stage5/runtime_v3/evaluator_only/fold0

python -m normroute.cli.build_expert_capability_bank \
  --teacher-data outputs/stage5/runtime_v3/evaluator_only/fold0/teacher.parquet \
  --router-features outputs/stage5/fbdp_router_cv_v2_20260805_054813/materialized/fold0/normal_bir_fbdp/router_features.jsonl \
  --stage2-summary reports/stage2/stage2_summary.csv \
  --fold fold0 \
  --output-dir outputs/stage5/runtime_v3/capability_bank/fold0

python -m normroute.cli.audit_teacher_artifacts \
  --teacher-data outputs/stage5/runtime_v3/evaluator_only/fold0/teacher.parquet \
  --capability-bank outputs/stage5/runtime_v3/capability_bank/fold0/capability_bank.json \
  --fold-manifest outputs/stage5/splits/fold_manifest.csv \
  --fold fold0
```

`build_teacher_targets --fold all` builds the five teacher folds beneath the
given evaluator-only output root. Capability banks remain separate commands
because each fold has a distinct Router feature bundle.

## Implemented research contract

- Expert score calibration is weighted Platt calibration with
  leave-one-train-category-out cross-fitting.
- Every category and opaque query receives equal aggregate influence; repeated
  K/seed variants receive inverse-frequency weight.
- Teacher risk is negative log calibrated correctness probability. Runtime and
  explicit failure penalties form a risk-cost objective, which is converted to
  a soft expert distribution.
- Every selected train row must carry a finite, non-negative per-query
  `runtime_ms`. The teacher records train-only runtime coverage and range in its
  metadata; a missing latency is an error rather than an implicit zero.
- Teacher v3 first fits a dimensionless objective-gap scale as the grouped,
  weighted median positive regret over category-cross-fitted **train** rows.
  It then forms Boltzmann targets from normalized regret. Validation rows
  jointly select temperature and a strictly positive per-expert probability
  floor against an independently defined empirical expert winner: each
  expert's validation score is transformed by its train-only calibration,
  thresholded at 0.5, then ranked by zero-one error plus normalized runtime
  and the explicit failure penalty. Raw scores are never compared across
  experts, so the selection is invariant to independent positive affine score
  transforms. This
  removes the v2 near-identity in which normalized correctness and
  `softmax(log(correctness) / T)` forced `T=1` and nearly uniform targets.
- Test outcomes are never parsed by the teacher and remain unavailable to the
  Router until label-free predictions have been persisted.
- ECPB reports boundary, foreground/background, low-shot, and available texture
  skills; K-shot curves; latency p50/p95; failure rate/type; difficulty-quantile
  curves; and category-bootstrap confidence intervals.
- ECPB latency and failure statistics come from exact train-category teacher
  rows. `reports/stage2/stage2_summary.csv` is category-filtered before numeric
  parsing and must exactly match their condition coverage, counts, failures,
  and mean runtimes. Test and validation summary rows never enter a profile.
- Router training accepts the soft distributions and sample weights. Optional
  ECPB routing minimizes Router risk plus conditional capability risk,
  uncertainty, and cost; the three weights are selected on validation only.
  The default uncertainty is query-adaptive and expert-specific: normalized
  Router predictive entropy gates the risk gap from an active-skill point
  estimate to its train-category-bootstrap lower confidence bound, plus any
  failure-rate upper-bound excess. The old absolute interval-width term is
  retained only as an explicit legacy ablation.
- Router validation and evaluator reporting reuse the same train-only weighted
  Platt protocol. Expert raw scores are never compared across methods. The
  fixed validation loss is zero-one error plus `0.05 * normalized_runtime`;
  it selects capability, uncertainty, and cost weights without structurally
  preferring the zero-cost coefficient. Hard and soft supervision use the
  same category/query inverse-frequency weights.
- Teacher metadata records held-out empirical NLL, multiclass Brier, ECE,
  entropy, effective class count, top-1 mass, selected temperature, and
  probability floor. The three-seed summary additionally emits all 120
  selected policies and 45 route-change rows for capability, uncertainty,
  and cost ablations. These diagnostics are evaluator-only evidence, never
  inference features.

The artifact audit requires complete teacher runtime, non-null finite expert
latency p50/p95 values, train-only runtime provenance, and a successful Stage 2
runtime cross-check. Old capability-bank artifacts are therefore not compatible
with the strengthened acceptance checks and must be rebuilt.

The current ablation and required-report matrix is frozen in
`configs/stage5/teacher_ecpb_v3.json`; v2 is retained only for historical
reproducibility. Configuration is not an experiment result. The v3 claim gate
requires three seeds over five folds for core, capability, uncertainty, and
cost ablations (plus the legacy-target comparison), followed by paired
cross-seed aggregation.

The executable closure is `scripts/run_teacher_ecpb_v3_closure.sh`. It runs
the complete test suite as a hard JUnit gate, rebuilds both v3 and legacy
teachers plus five ECPBs, executes 120 Router runs, produces per-seed and
cross-seed summaries, and finishes with
`final_acceptance_v3.json`. A completed run must also contain
`selected_capability_policies.csv`, `route_change_diagnostics.csv`, and a
commit/hash-bound `COMPLETE.txt`.
