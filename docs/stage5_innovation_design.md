# Stage 5 Innovation Design Baseline

This document preserves the Stage 5 research concepts that future requirements,
Goals, implementations, and manuscript text must follow.  It is a method-design
baseline, not evidence that every named component has already been implemented
or experimentally validated.

## 1. Research position

NORM-Route is not another anomaly-score ensemble.  Under one query image, K
official normal support images from the target category, and a one-call tool
budget, it predicts the risk, uncertainty, and execution cost of heterogeneous
anomaly-detection experts and invokes exactly one expert.

The intended doctoral contribution is the complete support-conditioned,
query-corrected, risk-cost routing problem and its leakage-safe category-held-out
evaluation.  Prototype means, cosine distances, patch nearest neighbours, or a
single NIV scalar are useful ingredients but are not by themselves the central
novelty claim.

The method has two information branches:

1. **Support-level prior**: depends only on normal support images and is
   cacheable before a query arrives.
2. **Query-level correction**: depends on the current query relative to that
   frozen support description.

Both branches are matched against expert capability representations before the
risk-cost decision.  They never consume target labels, masks, defect types,
realized expert outcomes, Oracle answers, or teacher utilities at inference.

## 2. Frozen terminology

| Acronym | Frozen English name | Role | Current maturity |
| --- | --- | --- | --- |
| `NDSE` | Normal-Domain Signature Encoder | Builds the support prior and query-relative signature from frozen visual features. | Core v2 statistics implemented; later feature families remain planned. |
| `NIV` | Normal Intra-support Variation | Structured description of variation observed inside the finite normal support set. | Core v2 vector implemented. |
| `NSVS` | Normal Support Variation Signature | Preferred long name for the complete NIV vector; avoids claiming that K samples estimate the entire domain. | Same artifact as NIV. |
| `BIR-AD` | Boundary-Informed Reweighting for Anomaly Detection | Produces boundary clarity/ambiguity evidence for routing; it does not modify external expert algorithms. | Planned; formula must follow Section 5. |
| `BAI-S` | Support Boundary Ambiguity Index | Support-only boundary ambiguity summary. | Planned. |
| `BAI-Q` | Query Boundary Ambiguity Index | Query-relative boundary ambiguity summary. | Planned. |
| `FBDP-AD` | Foreground-Background Decoupled Normal Prototypes for Anomaly Detection | Describes whether foreground and background normal patterns are separable and whether query patches are explained by either prototype bank. | Planned; formula must follow Section 6. |
| `FBC` | Foreground-Background Confusion | Support-only overlap/confusion between foreground and background prototype banks. | Planned. |
| `LGD` | Local-Global Deviation | Query deviation that contrasts global residuals with local patch residuals. | Planned derived query feature. |
| `ECPB` | Expert Capability Profile Bank | Fold-specific, training-category-only representations of expert strengths, costs, and failure behaviour. | Planned. |
| `RCR` | Risk-Cost Routing Head | Predicts expert risk/uncertainty/cost and selects one expert under the frozen budget. | Planned evolution of Stage 4 cost-aware routing. |

Do not use `Normal Intra-domain Variation` as the expanded name of NIV in
future documents.  K in {1,2,4,8} is a finite support set, not a reliable
estimate of the complete target domain.

## 3. NDSE and NIV v2

For cached global feature `g` and patch feature `z`, explicit L2 normalization
is mandatory even when the backbone already returns LayerNorm outputs:

```text
g_hat = g / (||g||_2 + eps)
z_hat = z / (||z||_2 + eps)
```

The normal global prototype and diagonal population variance are computed in
this normalized space:

```text
mu = mean_i(g_hat_i)
var = mean_i((g_hat_i - mu)^2)       # ddof=0
```

For K=1, `var` is algebraically zero but is not a reliable variance estimate.
The signature therefore records `normal_diagonal_variance_valid = false`.

### 3.1 Global support variation

For K >= 2:

```text
V_G = 2 / (K * (K - 1)) * sum_{i<j}(1 - g_hat_i^T g_hat_j)
```

### 3.2 Local support variation

For support patch sets `Z_i` and `Z_j`, define symmetric nearest-neighbour
cosine Chamfer distance:

```text
D_L(Z_i, Z_j) = 0.5 * [
    mean_p min_q(1 - z_hat_i,p^T z_hat_j,q)
  + mean_q min_p(1 - z_hat_j,q^T z_hat_i,p)
]

V_L = 2 / (K * (K - 1)) * sum_{i<j} D_L(Z_i, Z_j)
```

### 3.3 Single-image structural complexity

The K=1-compatible component is:

```text
V_S = mean_i [1 - ||mean_p(z_hat_i,p)||_2]
```

This is the stable form of mean patch-to-mean-direction dispersion and remains
defined when the mean patch direction approaches zero.  It is structural
complexity, not cross-support variation.

### 3.4 NIV vector and missingness

The raw model-ready vector is ordered as:

```text
NIV = [V_G*, V_L*, V_S, log2(K), m_G, m_L]
m_G = m_L = I[K >= 2]
```

`V_G*` and `V_L*` are zero-imputed only in the vector when unestimable.  Their
scalar artifact fields are null, and the masks distinguish missing estimates
from genuine zero variation.  A weighted scalar NIV may be reported only as an
ablation; it is not the primary Router input.

`global_rms_spread = sqrt(sum(var))` preserves the useful Stage 5 v1 statistic
under an unambiguous name.  It must not be presented as the complete NIV.

Before learned routing, continuous NIV components are robustly scaled using
statistics fitted only on the current fold's training categories.  The default
future transform is per-K training-fold Q05/Q95 clipping with a pooled-training
fallback when a K stratum is too small.  Validation can choose transform
hyperparameters; test categories never fit them.

## 4. Query-level correction

The query branch remains separate from NIV:

```text
r_global = g_hat_query - mu
d_patch(p) = min_m(1 - z_hat_query,p^T z_hat_support,m)
```

The frozen patch summaries are q50, q90, q95, q99, mean, and maximum.  Future
LGD features may combine these local residual summaries with global cosine,
L2, and shrinkage-standardized residual summaries.  Query patch nearest
distance is anomaly-relevant routing evidence, not part of support variation
and not a replacement for V_L.

When diagonal variance is used for standardization, a training-fold-only
shrinkage/floor is required; K=1 zero variance must never be interpreted as
infinite certainty.

## 5. BIR-AD design boundary

Do not define boundary evidence from only `Std(feature_channels)` or feature
L2 norm.  DINOv2 normalized tokens make these terms close to constants.

The supported design combines:

- pixel edge energy aligned to the patch grid (for example Sobel or LoG);
- spatial feature discontinuity between neighbouring patch tokens;
- cross-support boundary-response consistency;
- disagreement between pixel and feature boundaries.

`BAI-S` summarizes boundary ambiguity across normal supports.  `BAI-Q`
summarizes whether the query's boundary evidence is compatible with that
normal boundary prior.  Clear and ambiguous pooled tokens may be used as Router
features, but BIR-AD must not edit PatchCore, WinCLIP, or AnomalyDINO internals.

## 6. FBDP-AD design boundary

Foreground/background candidates must be derived from support images only,
using an auditable combination of spatial prior, cross-support stability,
feature clustering, or frozen objectness.  For each query patch:

```text
s_fg = max_m cos(q, p_fg_m)
s_bg = max_n cos(q, p_bg_n)
pi_fg = softmax([s_fg, s_bg] / tau)[foreground]

r_fg = pi_fg * (1 - s_fg)
r_bg = (1 - pi_fg) * (1 - s_bg)
margin_fg_bg = s_fg - s_bg
```

Do not use `1 - (s_fg - alpha * s_bg)` as the only residual: a well-explained
normal background patch would then incorrectly receive a larger residual.

`FBC` measures support foreground/background prototype overlap.  A
support-derived objectness gate `g_obj` must weaken or disable FBDP-AD for
texture-like or otherwise non-bimodal support sets; no test label or manually
encoded test-category list may control this gate.

## 7. ECPB

One expert profile may include boundary skill, foreground/background skill,
low-shot robustness, texture skill, latency, memory, and failure rate.  These
are learned or aggregated only from the current fold's training categories.
Validation may select profile granularity; test outcomes never update it.

Expert names are identifiers, not hand-written capability labels.  The
preferred implementation is a learned expert embedding conditioned on NDSE
features, with an auditable training-derived profile table retained for
interpretation.

## 8. RCR and learning target

For expert `e`, the final head predicts expected error risk `R_e`, uncertainty
`U_e`, and end-to-end cost `C_e`.  The frozen selection rule is:

```text
e* = argmin_e [R_e + lambda_u * U_e + lambda_c * normalized(C_e)]
```

Hard runtime/memory budgets filter infeasible experts before this comparison.
Exactly one expert may be invoked.  Router backbone time for a new query is
part of end-to-end cost; offline feature-cache reuse must not be reported as
production query latency.

Stage 4 run-level best-expert labels are a diagnostic baseline but cannot train
query-level correction.  Future query-level risk targets must be constructed
evaluator-side on training categories, for example from frozen per-expert score
calibration and proper per-query loss.  Teacher labels/utilities may be used in
the isolated training pipeline but must never appear in inference-visible
artifacts.

Uncertainty should represent unseen-category and finite-support uncertainty;
a category/bootstrap ensemble is preferred to an uncalibrated single variance
head.  Validation categories calibrate uncertainty and lambda values.

## 9. Recommended contribution hierarchy

The manuscript should present three primary contributions rather than five
independent acronyms:

1. normal-support-conditioned variation signature (NDSE/NIV);
2. support-prior plus query-correction expert capability matching (including
   BIR/FBDP feature families when validated);
3. category-generalized, uncertainty-aware risk-cost one-call routing
   (ECPB/RCR) under strict leakage controls.

Every component requires an ablation against Stage 4 metadata/cost-aware
baselines.  Current local expert wrappers validate the protocol but cannot
support final SOTA claims; final experiments require version-pinned official
or faithfully isolated expert adapters.

## 10. Protocol and artifact versioning

- Stage 5 v1 normal-signature artifacts use the legacy scalar `niv` meaning and
  remain engineering evidence only.
- Stage 5 v2 changes the statistical space, NIV type, K=1 semantics, and patch
  distance metric.  It must write a new protocol version and be regenerated.
- Never overwrite v1 smoke evidence and describe it as v2 output.
- Future Goal text that conflicts with this baseline must explicitly request a
  protocol revision and document the scientific reason.
