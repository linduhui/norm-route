# Stage 5 NDSE/NIV v2 Feature Smoke Acceptance

## Status

**PASS**

This report records the real remote smoke run for
`stage5.normal_domain_signature.v2`. The synchronized run is preserved under:

```text
outputs/stage5/smoke_v2_20260722_113444/
```

The historical v1 evidence in `outputs/stage5/smoke/` remains unchanged. The
v2 run used a separate output directory and therefore did not reinterpret or
overwrite the legacy scalar-NIV artifact.

## Scope and non-claims

The smoke verifies real frozen-backbone execution, content-addressed fp16
feature caching, hash-checked resume, NDSE/NIV v2 statistics, artifact schema,
and leakage controls on a small MVTec AD bottle subset.

It is not an anomaly-detection or Router performance result. All four smoke
queries are inference-visible `test/good` images, no labels or masks enter the
encoder, and no threshold, hyperparameter, or expert policy is tuned from
these outputs. The numerical values below are descriptive acceptance evidence,
not category-level scientific conclusions.

## Synchronized artifact integrity

- Transfer archive:
  `smoke_v2_20260722_113444.tgz`
- Archive SHA-256:
  `0ba61f15028d569cd847aaeb07d750f904b2b0472fbe8a247a2663d94b098b4c`
- Archive size: 13,660,748 bytes
- Files verified from the internal `repro/SHA256SUMS`: 65
- Cached `.npy` arrays present after synchronization: 28
- Cache entry files present: 42 (14 entries, three files per entry)
- Internal checksum verification: PASS

The synchronized archive SHA-256 matched its sidecar, and every file listed in
the internal `SHA256SUMS` matched after extraction.

## Source and environment provenance

- Git commit:
  `fd9d98a010b30993fa0919112ecccb04a77bb12a`
- Commit subject:
  `feat(stage5): implement NIV v2 normal-domain signatures`
- Remote project:
  `/data/gauss/ldh/projects/norm-route`
- Device: `cuda:0`
- GPU: NVIDIA A40, 46,068 MiB
- NVIDIA driver: `535.104.12`
- Reported CUDA runtime: `12.2`
- Platform: Linux 5.4.0-91-generic, x86_64, glibc 2.31
- Python: 3.10.20
- NumPy: 2.2.6
- PyArrow: 25.0.0
- Torch: 2.7.1+cu118
- torchvision: 0.22.1+cu118
- timm: 1.0.28
- Pillow: 12.3.0

The run metadata, input audit, and backbone audit all report the same Git
commit. The remote worktree also contained pre-existing modified dataset
manifests/support CSVs, one Stage 3 report, and untracked editable-install
metadata. No `src/normroute/` implementation file was reported modified.
Consequently, the immutable smoke input snapshots and their hashes, rather
than the general remote data worktree state, define this run.

## Frozen backbone audit

- Architecture: `dinov2_vits14`
- Input size: 518
- Feature dimension: 384
- Patch grid output: 1,369 patch tokens per image
- Checkpoint:
  `weights/dinov2_vits14_pretrain_timm.pth`
- Checkpoint size: 88,284,339 bytes
- Expected and observed checkpoint SHA-256:
  `82d4346964e211cc5c1ac44c7f2d15d2039f41c2aa86de2c0250ef2a30e12d1e`
- Hash match: true
- Model loaded: true
- All parameters frozen: true
- Evaluation mode: true
- Backbone audit failures: 0

The backbone audit status is PASS.

## Inference-visible inputs

- Dataset/category: MVTec AD / bottle
- Unique query images: 4
- Query location class: `test/good`
- Support sets: 5
- Support rows: 10
- Support source: official `train/good`
- Task rows: 20
- K/task distribution: K=1: 8, K=2: 8, K=4: 4
- Input audit files scanned: 2
- Input audit failures: 0

Support sets:

- `mvtec_bottle_k1_seed0`
- `mvtec_bottle_k1_seed1`
- `mvtec_bottle_k2_seed0`
- `mvtec_bottle_k2_seed1`
- `mvtec_bottle_k4_seed0`

Immutable input hashes:

- `tasks.jsonl`:
  `9e3645ac9592e1c5ee6d67fd9334a7d88c28198c51b75afab9c71fdfb7e46843`
- `supports.csv`:
  `c6ac83934b796d29d7b8ac3fe0aba3550e58a5dff1fe496d796d7b1122955ffc`

The Stage 5 input audit rejected none of the audited files and found no
inference-visible label, ground truth, mask, defect/anomaly type, expert
outcome, Oracle, or teacher field.

## Required artifacts

- `feature_cache_manifest.csv`: present
- `normal_signatures.parquet`: present
- `feature_cache/entries/`: present
- `run_metadata.json`: present
- `failures.json`: present and equal to `[]`
- `validation.txt`: present and records PASS

Observed artifact properties:

- Unique feature-cache entries: 14
- Cache dtype: fp16 for every entry
- Global feature shape: `[384]`
- Patch feature shape: `[1369, 384]`
- Signature rows: 20
- Unique query SHA-256 values: 4
- Normal-signature protocol:
  `stage5.normal_domain_signature.v2`
- Parquet columns: 34, including the complete structured NIV fields,
  query-global residual, and patch nearest-distance summaries

## Real NIV v2 support statistics

The following fields were read directly from the synchronized Parquet. `V_G`
and `V_L` are null for K=1, with their validity masks false; they are not
reported as measured zero variation. `V_S` remains defined for K=1.

| Support set | K | Seed | `V_G` | `V_L` | `V_S` | Global RMS spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mvtec_bottle_k1_seed0` | 1 | 0 | null | null | 0.335410 | 0.000000 |
| `mvtec_bottle_k1_seed1` | 1 | 1 | null | null | 0.341977 | 0.000000 |
| `mvtec_bottle_k2_seed0` | 2 | 0 | 0.019477 | 0.029508 | 0.347605 | 0.098684 |
| `mvtec_bottle_k2_seed1` | 2 | 1 | 0.018035 | 0.032313 | 0.351121 | 0.094961 |
| `mvtec_bottle_k4_seed0` | 4 | 0 | 0.025060 | 0.033281 | 0.344049 | 0.137095 |

For K=1, the model-ready NIV vector zero-imputes the first two positions only
and records `global_valid=0` and `local_valid=0`. For K>=2, both scalar fields
are present and both masks equal one. Diagonal variance validity follows the
same K sufficiency rule.

The support-only prototype, diagonal variance, global RMS spread, and complete
NIV vector were exactly constant across the four queries sharing each
`support_set_id`. This confirms that support-level statistics did not acquire
query dependence.

## Real query-relative summaries

Each row below averages the four query-relative results associated with the
given support set. Patch values are nearest-normal cosine-distance quantiles.

| Support set | Mean global residual L2 | Mean patch q50 | Mean patch q95 | Mean patch q99 |
| --- | ---: | ---: | ---: | ---: |
| `mvtec_bottle_k1_seed0` | 0.272813 | 0.032521 | 0.127655 | 0.180087 |
| `mvtec_bottle_k1_seed1` | 0.255498 | 0.028422 | 0.108597 | 0.159985 |
| `mvtec_bottle_k2_seed0` | 0.162105 | 0.019312 | 0.063980 | 0.109441 |
| `mvtec_bottle_k2_seed1` | 0.178087 | 0.020573 | 0.072127 | 0.125259 |
| `mvtec_bottle_k4_seed0` | 0.177181 | 0.018903 | 0.068405 | 0.115071 |

All per-row q50/q90/q95/q99 sequences were nondecreasing. These smoke
statistics demonstrate valid computation and serialization only; the five
support sets are too small to infer a monotonic K effect or routing advantage.

## Resume and byte reproducibility

Feature-cache entry hashes and entry filesystem state were captured after the
first run and compared after an identical second invocation.

- Cache state before/after: identical
- Cache entry hashes before/after: identical
- Required artifact hashes before/after: identical
- Repeat feature encoding or cache-entry rewrite observed: no

Required artifact hashes on both runs:

- `feature_cache_manifest.csv`:
  `cc950a947c29449091cc863ec9e443318857962945e8d131e78d6d0545bb428c`
- `normal_signatures.parquet`:
  `9e921585edb3cce617330f57d5a7bc49b893455a9d39bcdf0b559eca0ff9ccf0`

The manifest and Parquet artifacts are therefore byte reproducible under the
recorded environment, and the second run reused the verified content-addressed
cache.

## Acceptance

This run accepts the Stage 5 feature cache and Normal-Domain Signature Encoder
v2 for the tested smoke scope:

- real frozen DINOv2 backbone execution;
- query/support deduplication by image SHA-256;
- fp16 global and patch feature storage;
- verified resume without cache-entry re-encoding;
- normal global prototype and diagonal variance;
- structured NIV/NSVS with `V_G`, `V_L`, `V_S`, log2(K), and K=1 masks;
- normalized query-global residual;
- nearest-normal patch cosine-distance quantiles;
- required manifest and Parquet outputs;
- zero explicit failures and no silent sample skipping;
- no inference-visible target labels, masks, defect types, expert outcomes,
  Oracle values, or teacher values.

The result closes the Stage 5 NDSE/NIV v2 engineering smoke gate. It does not
close the later Router learning, category-held-out generalization, expert
capability, risk-cost calibration, or anomaly-detection performance gates.
