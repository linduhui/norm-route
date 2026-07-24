# Stage 5 BIR-AD implementation

This document freezes the implemented BIR-AD protocol. It is an
inference-side Router feature module, not an anomaly detector and not a
modification of PatchCore, WinCLIP, AnomalyDINO, or any other expert.

## Information boundary

BIR-AD may consume only the query image, its frozen patch tokens, and the K
official `train/good` supports belonging to the task's `support_set_id`.
Target labels, masks, defect types, expert outcomes, Oracle data, and fold
labels are not accepted as learned features. Fold labels are used only by the
offline normalization builder to select training tasks.

## Strict pixel-token geometry

`FrozenVisualBackboneProvider.align_images_for_patches()` applies the exact
same deterministic resize/crop transform used for token extraction. It then
inverts only channel normalization, preserving the transformed pixel geometry.
Each `PatchAlignedImage` records:

- the patch grid;
- the source-image SHA-256;
- a fingerprint of the frozen spatial transform;
- the alignment protocol version.

The batch encoder checks image hash, token count, grid size, and transform
fingerprint. A task cannot combine features and pixels from different images
or transforms. For DINOv2-S/14 at 518 pixels, the expected grid is 37 by 37.

## Patch evidence and ambiguity

For patch token `f_p`, BIR-AD computes channel standard deviation and RMS L2
norm. Potential feature boundaries are the mean cosine discontinuity to four
or eight spatial neighbours. On the aligned pixel region it computes Sobel
edge energy, structure-tensor orientation coherence, and two-sided feature
contrast sampled along the image-gradient normal.

Every signal is standardized with statistics fitted only on the current
fold's training-category normal supports. Let the normalized signals be
`sigma_tilde`, `n_tilde`, `e_tilde`, `rho_tilde`, and `d_tilde`. Pixel-feature
boundary disagreement is explicitly measured as:

```text
D_p = |sigmoid(b_tilde_p) - sigmoid(e_tilde_p)|
```

Patch clarity is:

```text
c_p = sigmoid(
    alpha_sigma * sigma_tilde_p
  + alpha_n * n_tilde_p
  + alpha_e * e_tilde_p
  + alpha_rho * rho_tilde_p
  + alpha_d * d_tilde_p
  - lambda_D * D_p
)
```

The coefficients are non-negative and normalized to sum to one. Structural
boundary weights and the formal mask-free ambiguity index are:

```text
w_p = softmax(b_tilde_p / tau_b)
BAI(x) = sum_p w_p * (1 - c_p)
```

Clear and ambiguous patch weights are separately normalized:

```text
w_clear_p = normalize(w_p * c_p)
w_amb_p   = normalize(w_p * (1 - c_p))
```

They pool L2-normalized tokens into the clear and ambiguous representations.
All softmax, sigmoid, norm, pooling, and reduction paths have finite-value and
normalization checks.

## Support/query decomposition

For canonicalized support results:

```text
BAI-S     = mean_k BAI(x_support_k)
StdBAI-S  = sqrt(mean_k (BAI(x_support_k) - BAI-S)^2)
BAI-Q     = BAI(x_query)
DeltaBAI  = BAI-Q - BAI-S
```

The formal Router BAI vector is:

```text
[BAI-S, StdBAI-S, BAI-Q, DeltaBAI, |DeltaBAI|]
```

Patch-level cross-support consistency does not assume coordinate equality.
For each source patch, BIR-AD finds the feature-nearest patch in the other
normal supports and compares a bounded response vector containing feature
boundary, pixel boundary, clarity, orientation coherence, and two-sided
contrast. Query-to-support consistency uses the same frozen normal patch bank.
K=1 support consistency is represented as missing plus an explicit validity
mask, never as measured zero consistency. Byte-identical supports are rejected.

## Fold normalization closure

`fit_fold_bir_ad_normalization()`:

1. validates task and support provenance;
2. selects only `split=train` tasks from one fold manifest;
3. uses only their official `train/good` support images;
4. deduplicates byte-identical training images by SHA-256;
5. fits signal locations/scales;
6. records fold, train categories, support set ids, image hashes, backbone
   fingerprint, and alignment fingerprint.

`BIRADTaskEncoder` can receive this complete artifact. It refuses to encode if
the current feature cache or spatial transform disagrees with it.

## Router feature contract

`build_router_feature_bundles()` joins the normal-domain signature and BIR-AD
signature only when task id, dataset, category, K, seed, support set, encoder,
query hash, and support hashes all agree. The learned vector contains:

- NIV and query residual features;
- BAI-S, BAI-Q, signed/absolute shift, and reliability;
- pixel-feature disagreement;
- patch-level support/query consistency and its validity mask;
- optional clear and ambiguous representations.

Identifiers, category, seed, paths, support-set id, hashes, and fold membership
remain provenance and do not enter the numeric learned vector. One output file
cannot mix feature schemas, ablations, encoders, normalizations, or transforms.

## Required ablations

`configs/stage5/bir_ad_ablations.json` freezes this sequence:

1. `sigma_l2`;
2. `plus_sobel`;
3. `plus_structural_boundary`;
4. `plus_directional_evidence`;
5. `plus_cross_modal_disagreement`;
6. `plus_support_consistency`;
7. `representations_only`;
8. `full`.

The first six form a nested evidence chain. `representations_only` tests
whether the pooled tokens alone explain the gain. Comparing `full` against
`plus_support_consistency` isolates the contribution of the two pooled
representations. All variants must share tasks, fold, backbone, and
`support_set_id`; selection is validation-category-only.

## Batch command

Fit one fold, encode tasks, save diagnostic maps for a subset, and optionally
join existing normal-domain signatures:

```powershell
python scripts/build_bir_ad.py `
  --tasks outputs/stage4/tasks/pre_route_tasks.jsonl `
  --supports outputs/stage1/support_sets.csv `
  --fold-manifest outputs/stage5/splits/fold_manifest.csv `
  --fold fold0 `
  --backbone-config configs/stage5/router_backbone.yaml `
  --feature-cache outputs/stage5/features `
  --normal-signatures outputs/stage5/normal_signatures.parquet `
  --ablation full `
  --diagnostics-limit 100 `
  --output-dir outputs/stage5/bir_ad/fold0
```

No data or weights are downloaded. The command saves the frozen normalization,
ablation record, BIR-AD signatures, optional Router feature bundle, cache
manifest, explicit failures, diagnostics, and a run record containing config,
seed, git commit, and environment.
