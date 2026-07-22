# Stage 5 Feature Cache Smoke Acceptance

## Scope

This report records the real remote smoke run for the Stage 5 feature cache
and Normal-Domain Signature Encoder. It is not a router prediction result and
does not fabricate a `predictions.csv` artifact.

## Provenance

- Project path: `/data/gauss/ldh/projects/norm-route`
- Git commit: `af1f6ce8bb6f29167bc262f862ac8bd8063efa96`
- Device: `cuda:0`
- Python: `3.10.20`
- Torch: `2.7.1+cu118`
- torchvision: `0.22.1+cu118`
- timm: `1.0.28`
- NumPy: `2.2.6`
- PyArrow: `25.0.0`
- Pillow: `12.3.0`
- Backbone architecture: `dinov2_vits14`
- Converted timm checkpoint SHA-256:
  `82d4346964e211cc5c1ac44c7f2d15d2039f41c2aa86de2c0250ef2a30e12d1e`

The exact remote backbone config is preserved at
`outputs/stage5/smoke/repro/router_backbone.yaml`. The portable project config
uses the equivalent relative checkpoint path.

## Inputs

- Dataset/category: MVTec AD / bottle
- Unique queries: 4
- Support sets: 5
- Task rows: 20
- Support rows: 10
- Support ids:
  - `mvtec_bottle_k1_seed0`
  - `mvtec_bottle_k1_seed1`
  - `mvtec_bottle_k2_seed0`
  - `mvtec_bottle_k2_seed1`
  - `mvtec_bottle_k4_seed0`

The same four query images were repeated across the five K/seed settings. The
saved task and support snapshots were audited before feature extraction.

## Results

- Backbone audit: PASS
- Stage 5 inference-input audit: PASS
- Explicit failures: 0
- Unique cache entries: 14
- Cache dtype: fp16 for all global and patch arrays
- Global shape: `[384]`
- Patch shape: `[1369, 384]`
- Normal signature rows: 20
- Unique query SHA-256 values in signatures: 4
- Resume cache-entry comparison: identical
- Resume manifest/Parquet comparison: identical

Artifact hashes from the completed smoke:

- `feature_cache_manifest.csv`:
  `f5548d25fb329678297a922d2eb99c47723b62f026209a078a1fe08e95da1ad8`
- `normal_signatures.parquet`:
  `8e432cd9c2b289e3715d9e75d653dba37759cc0a34bc624e2fd8e174a967f58e`
- task snapshot:
  `9e3645ac9592e1c5ee6d67fd9334a7d88c28198c51b75afab9c71fdfb7e46843`
- support snapshot:
  `c6ac83934b796d29d7b8ac3fe0aba3550e58a5dff1fe496d796d7b1122955ffc`

All synchronized smoke files are covered by
`outputs/stage5/smoke/repro/SHA256SUMS`. The original and converted checkpoint
hashes plus conversion-record hash are preserved in
`outputs/stage5/smoke/audits/weight_files.sha256`.

## Reproducibility note

The recorded source commit matches the local Stage 5 implementation commit.
The remote worktree was not globally clean: dataset manifests/support CSVs,
one Stage 3 report, the runtime backbone config, and editable-install metadata
were reported as modified or untracked. No Python source file was dirty.
Exact smoke tasks and supports were therefore saved as immutable snapshots and
their hashes were checked by the Stage 5 input audit. A future formal run
should start from a clean source/data worktree or record independent hashes for
all upstream manifests before task selection.

## Acceptance

The smoke accepts the Stage 5 feature-cache and normal-signature behavior for
the tested MVTec bottle subset: content-addressed deduplication, fp16 storage,
hash-checked resume, normal statistics, patch nearest-distance summaries,
support permutation invariance (unit-tested), required artifact emission, and
zero silent failures are all covered.
