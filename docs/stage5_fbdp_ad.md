# Stage 5 FBDP-AD v2

This document freezes the implemented Foreground-Background Decoupled Normal
Prototypes branch and its evaluation contract. It describes executable code,
not a claim that the method improves real-data results. Such a claim requires
the complete category-held-out experiments in
`configs/stage5/fbdp_ad_experiments.json`.

## 1. Scope and leakage boundary

FBDP-AD is a Router evidence branch. It does not alter PatchCore, WinCLIP,
AnomalyDINO, or another external expert. Its inference-visible inputs are only:

- the current query image's frozen patch features;
- K support images from the same task's official `train/good` split;
- patch-grid geometry and frozen hyperparameters.

Target test labels, masks, defect/anomaly types, realized expert outcomes,
Oracle identities, and category lookup tables are neither accepted nor stored
in FBDP signatures. Candidate selection, prototypes, FBC, reliability, and the
gate are frozen before the query is composed. The batch pipeline reuses the
audited Stage 5 support resolver and rejects non-`train/good`, mismatched K,
mismatched category/dataset, duplicate support hashes, and incomplete tasks.

## 2. Support-derived candidates

Every patch feature is explicitly L2-normalized. Border patches first form an
initial background bank. For support patch `z`, the implemented objectness is
the product of three support-derived terms:

```text
feature_foregroundness(z) = clip(1 - max_b cos(z, p_border_b), 0, 1)
objectness(z) = feature_foregroundness(z)
              * (0.5 + 0.5 * spatial_centrality(z))
              * (0.5 + 0.5 * cross_support_consistency(z))
```

The default background candidates are the union of all border patches and
patches below the pooled support-objectness quantile. `border_only` and
`low_objectness_only` exist only as mechanism controls.

The foreground pool excludes border/low-objectness patches and requires both:

- cross-support consistency above a support-derived quantile;
- cosine distance from the final background prototypes above a
  support-derived quantile.

The default consistency compares each patch with the best patch in a local
window of every other support image. It tolerates small translations while
remaining spatially constrained. Strict same-position and unrestricted
nearest-neighbour modes are ablations. K=1 and the explicit no-consistency
ablation use neutral consistency values plus a false validity bit; they do not
pretend that cross-support evidence was observed.

If strict foreground selection is empty, the maximally stable/far/object-like
patch is retained to make failure observable rather than silently dropping the
task. The signature sets `foreground_candidate_fallback=true`, and the gate is
capped by `fallback_gate_cap`.

## 3. Prototype banks, FBC, and support reliability

The default bank builder is small deterministic spherical k-means with
canonical input/prototype order and deterministic farthest initialization.
Prototype pooling is available as an ablation. Requested bank size is capped
at the number of candidates, so both banks are always non-empty.

Foreground-background confusion is symmetric positive cosine overlap:

```text
FBC = 0.5 * [
  mean_f max_b clip(cos(p_fg_f, p_bg_b), 0, 1)
  + mean_b max_f clip(cos(p_bg_b, p_fg_f), 0, 1)
]
```

High FBC means the two support-derived banks are not reliably separable.
Support reliability is the geometric mean of four independent checks:

1. foreground/background assignment confidence, `1 - mean(binary_entropy)`;
2. within-bank prototype compactness;
3. the fraction of support images contributing foreground candidates;
4. leave-one-support-out reconstruction margin, mapped to `[0, 1]`.

The leave-one-out term is invalid at K=1 and has an explicit validity bit.
This makes one-shot uncertainty visible to the Router.

## 4. Support-derived objectness gate

Let separation be `1 - FBC` (or one in the `no_fbc` ablation), contrast be
the normalized foreground/background objectness gap, stability be mean
foreground consistency, and reliability be the independent reliability above.
The default gate is:

```text
g_obj = clip(
  sqrt(separation * contrast * stability) * sqrt(reliability),
  0, 1
)
```

For K=1, the default `shrink` policy multiplies the gate by 0.5. `neutral` and
`disable` are explicit sensitivity settings. A fallback foreground pool caps
the gate at 0.25. A texture-like/non-bimodal support set tends toward high FBC,
low contrast, uncertain assignments, or low compactness and therefore weakens
the branch continuously. There is no manual texture category list.

## 5. Query-decoupled evidence

For query patch `q`:

```text
s_fg = max_f cos(q, p_fg_f)
s_bg = max_b cos(q, p_bg_b)
margin = s_fg - s_bg
pi_fg = sigmoid(margin / temperature)

r_fg = pi_fg * (1 - s_fg)
r_bg = (1 - pi_fg) * (1 - s_bg)
s_decoupled = pi_fg * s_fg + (1 - pi_fg) * s_bg
r = r_fg + r_bg = 1 - s_decoupled
r_gated = g_obj * r
```

The batch signature stores q50/q90/q95/q99, mean, and maximum for raw and
gated residuals; margin quantiles/statistics; and normalized binary assignment
entropy quantiles/statistics. The `single_bank` ablation replaces decoupled
mixing with the nearest prototype in the union bank.

## 6. Artifacts and commands

An isolated cached-array diagnostic, matching the function shown in the design
discussion, is:

```bash
python -m normroute.cli.diagnose_fbdp_ad \
  --query-patches query.npy \
  --support-patches support_0.npy \
  --support-patches support_1.npy \
  --grid-shape 37 37 \
  --output-dir outputs/stage5/diagnostics/fbdp_ad
```

The project-integrated batch command is:

```bash
python scripts/build_fbdp_ad.py \
  --tasks outputs/stage5/tasks/bir_ad_tasks.jsonl \
  --supports outputs/stage5/supports/bir_ad_supports.csv \
  --backbone-config configs/stage5/router_backbone.yaml \
  --feature-cache outputs/stage5/features \
  --normal-signatures outputs/stage5/normal_domain/normal_signatures.parquet \
  --grid-shape 37 37 \
  --ablation full \
  --diagnostics-limit 100 \
  --output-dir outputs/stage5/fbdp_ad_ablations_v1/full
```

It writes immutable signatures, a flat statistics CSV, optional patch/prototype
PNG diagnostics, Router features, the cache manifest, explicit failures, the
ablation record, and a run record with config, seed, git commit, environment,
predictions, and failures. It downloads neither data nor weights.

Run the label-free acceptance audit before using signatures:

```bash
python scripts/audit_fbdp_ad.py \
  --tasks outputs/stage5/tasks/bir_ad_tasks.jsonl \
  --signatures outputs/stage5/fbdp_ad_ablations_v1/full/fbdp_ad_signatures.jsonl \
  --failures outputs/stage5/fbdp_ad_ablations_v1/full/fbdp_ad_failures.json \
  --output-dir outputs/stage5/fbdp_ad_ablations_v1/full/audit
```

The audit checks exact task coverage, protocol/schema, provenance, finite
values, candidate/prototype counts, grid/count identities, monotone residual
quantiles, gate scaling, gate saturation, fallback rate, K validity, and
grouped diagnostics. It never opens an evaluator artifact.

## 7. Router views and provenance

`stage5.router_feature_bundle.v3` supports four strict views:

| View | Numeric prefixes |
| --- | --- |
| `normal_only` | `normal_` |
| `normal_bir` | `normal_`, `bir_` |
| `normal_fbdp` | `normal_`, `fbdp_` |
| `normal_bir_fbdp` | `normal_`, `bir_`, `fbdp_` |

The reader rejects missing prefix families, schema drift, non-finite values,
forbidden feature names, task/provenance mismatch, and mixed feature views or
ablation sources. BIR and FBDP ablation provenance are independent. Combined
views can be reconstructed from immutable signatures with
`scripts/materialize_fbdp_ad_features.py`; the materializer verifies that the
signature's compute controls match the requested FBDP ablation.

## 8. Required ablations and evidence

The frozen registry includes `single_bank`, `no_fbc`, `no_objectness_gate`,
`border_only_background`, `low_objectness_only_background`,
`no_cross_support_consistency`, `pooling`, `same_position_consistency`,
`raw_residual_only`, and `full`. Run all ten with:

```bash
GPUS=0,1,2,3,4,5 \
PROJECT_ROOT=/data/gauss/ldh/projects/norm-route \
bash scripts/run_fbdp_ad_ablations.sh
```

Then run mechanism comparisons and the four-view complementarity study:

```bash
GPUS=0,1,2,3,4,5 \
FBDP_ROOT=outputs/stage5/fbdp_ad_ablations_v1 \
BIR_ROOT=outputs/stage5/bir_ad_ablations_gpu_v2 \
OUTPUT_ROOT=outputs/stage5/fbdp_router_cv_v1 \
bash scripts/run_fbdp_router_cv.sh
```

FBDP has no fold-fitted parameters, so its signatures are computed once and
reused. BIR/FBDP combined artifacts are materialized separately for each BIR
fold. The Router still trains and evaluates on all five category-held-out
folds. The script freezes label-free predictions before evaluator joins and
produces same-fold paired effects for FBDP, BIR, their complementarity, and
every FBDP mechanism control.

## 9. Acceptance and claim boundary

Engineering acceptance requires all tests and all label-free audits to pass.
Scientific acceptance additionally requires complete five-fold results with
identical tasks/supports/optimizer across views, validation-only selection,
paired effect sizes, fold consistency, runtime, and evaluator-only anomaly
metrics. Gate/fallback statistics must be reported by K and category to expose
failure modes. With five folds, exact sign-flip p-values are coarse; report
effect magnitude and signs rather than relying on p-value alone.

Until those real-data artifacts exist, the defensible statement is that
FBDP-AD v2 is implemented, leakage-safe by interface, and experimentally
falsifiable. It is not yet evidence of a performance improvement.
