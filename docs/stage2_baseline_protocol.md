# Stage 2 Baseline Protocol

Stage 2 connects real visual anomaly experts under the Stage 1 protocol.

Experts:
- PatchCore
- WinCLIP
- AnomalyDINO

Each expert must use:
- dataset
- category
- k_shot
- seed
- support_set_id
- support images
- test image paths

Each expert must not read:
- test label
- mask_path
- defect_type
- anomaly_type

Each run must output:
- predictions.csv
- anomaly_maps/
- run_metadata.json
- failures.json
- metrics.json
