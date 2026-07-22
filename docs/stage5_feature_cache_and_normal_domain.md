# Stage 5 Feature Cache and Normal-Domain Signatures

This document freezes the implementation contract for
`src/normroute/router/feature_cache.py` and
`src/normroute/router/normal_domain.py`. It is additive to
`stage5_method_spec.md` and `stage5_leakage_policy.md`.

## Inference boundary

The encoder receives only query image bytes, normal support image bytes, and
frozen backbone configuration. Target labels, masks, defect/anomaly types,
expert outcomes, Oracle data, and teacher data are rejected and must never be
present in task or support records.

Support rows must originate from the official `train/good` split. A task is
identified for runner provenance by dataset, category, K, seed, and
`support_set_id`; seed and support-set identity are not learned router
features. The exact query and support inputs used by a run must be saved and
audited before encoding.

## Frozen backbone output

`FrozenVisualBackboneProvider.encode_global_and_patches()` returns:

- one global feature vector `[D]` per image;
- one patch-token matrix `[P, D]` per image.

The model is constructed by timm with `pretrained=False`, loaded only from the
configured local checkpoint with strict state-dict matching, frozen, placed in
evaluation mode, and executed under inference mode. Missing or incompatible
weights are fatal; no download fallback is permitted.

## Content-addressed cache

The cache identity is derived from:

1. SHA-256 of the exact image bytes;
2. an encoder fingerprint derived from the complete frozen backbone config;
3. protocol version `stage5.feature_cache.v1`.

K, seed, task id, image path, and `support_set_id` are deliberately absent
from the cache key. Consequently, byte-identical queries shared across K/seed
settings are encoded once.

Global and patch arrays are stored as deterministic NumPy `.npy` files with
dtype `float16`. Each cache entry contains:

- `global.npy`;
- `patches.npy`;
- `metadata.json` with image identity, encoder identity, shapes, dtype, and
  feature-file SHA-256 values.

Writes are atomic. An existing manifest is the resume integrity anchor: every
manifest row, metadata record, feature hash, dtype, and shape is verified
before reuse. Corrupt or partial entries fail explicitly and are never
silently skipped or replaced.

`feature_cache_manifest.csv` has one row per unique image SHA-256 in the
current encoder namespace. Paths in this manifest are relative to the cache
root, not necessarily to the manifest directory.

## Normal-domain definitions

Let normal support global features be `s_i`, query global feature be `q`,
query patches be `p_j`, and the union of all support patches be `M`.

The normal global prototype is

```text
mu = (1 / K) * sum_i s_i
```

The diagonal population variance (`ddof=0`) is

```text
var = (1 / K) * sum_i (s_i - mu)^2
```

For K=1, the diagonal variance is exactly zero. Normal intra-support variation
(NIV) is

```text
NIV = sqrt(sum_d var_d)
```

which is equivalent to the root mean squared Euclidean support distance from
the prototype. The signed query-global residual is

```text
r = q - mu
```

and `query_global_residual_l2` is `||r||_2`.

For every query patch, the nearest-normal distance is

```text
d_j = min_{m in M} ||p_j - m||_2
```

The frozen patch summaries are q50, q90, q95, and q99 using linear quantile
interpolation, plus mean and maximum distance. Distance evaluation is chunked
to bound working memory.

Support global and patch rows are sorted by their feature bytes before
floating-point reductions. This makes outputs exactly invariant to support
row permutation rather than merely numerically close.

## Required artifacts

`build_normal_signatures()` performs cache population and signature creation
in one pass and writes:

- `feature_cache_manifest.csv`;
- `normal_signatures.parquet`;
- the `feature_cache/entries/` tree when the default cache directory is used.

The Parquet file contains only allowlisted signature data and runner
provenance. Its numerical fields include the prototype, diagonal variance,
NIV, query residual, patch-distance quantiles, patch counts, and support image
hashes. It does not contain image paths, labels, masks, defect types, or
evaluator outcomes.

Calling the builder again with the same inputs and output/cache directories is
the resume operation; there is no separate `--resume` flag. Cache entry files
must remain unchanged, and the manifest and Parquet file must be byte
reproducible.

## Verification

The mandatory tests are in `tests/test_stage5_feature_cache.py` and cover:

- fp16 cache format;
- image-hash deduplication and K/seed query reuse;
- resume without re-encoding;
- cached feature and manifest tamper detection;
- support permutation invariance;
- exact normal statistics;
- required manifest and Parquet output;
- byte-identical repeated output.

After any implementation or protocol change, run the complete `pytest` suite.
