#!/usr/bin/env bash
set -Eeuo pipefail

# Compute each FBDP mechanism variant once. FBDP contains no fold-fitted
# parameters, so immutable signatures are shared by all category folds.

PROJECT_ROOT="${PROJECT_ROOT:-/data/gauss/ldh/projects/norm-route}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS="${TASKS:-outputs/stage5/tasks/bir_ad_tasks.jsonl}"
SUPPORTS="${SUPPORTS:-outputs/stage5/supports/bir_ad_supports.csv}"
BACKBONE_CONFIG="${BACKBONE_CONFIG:-configs/stage5/router_backbone.yaml}"
FEATURE_CACHE="${FEATURE_CACHE:-outputs/stage5/features}"
NORMAL_SIGNATURES="${NORMAL_SIGNATURES:-outputs/stage5/normal_domain/normal_signatures.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage5/fbdp_ad_ablations_v1}"
GPU_LIST="${GPUS:-0,1,2,3,4,5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRID_ROWS="${GRID_ROWS:-37}"
GRID_COLUMNS="${GRID_COLUMNS:-37}"
DIAGNOSTICS_LIMIT="${DIAGNOSTICS_LIMIT:-100}"
SEED="${SEED:-0}"

ABLATIONS=(
  single_bank
  no_fbc
  no_objectness_gate
  border_only_background
  low_objectness_only_background
  no_cross_support_consistency
  pooling
  same_position_consistency
  raw_residual_only
)

cd "$PROJECT_ROOT"
IFS=',' read -r -a GPUS_ARRAY <<< "$GPU_LIST"
if (( ${#GPUS_ARRAY[@]} == 0 )); then
  echo "GPUS must contain at least one physical GPU id" >&2
  exit 2
fi
required_files=(
  "$TASKS"
  "$SUPPORTS"
  "$BACKBONE_CONFIG"
  "$NORMAL_SIGNATURES"
  "$FEATURE_CACHE/feature_cache_manifest.csv"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Required artifact does not exist: $path" >&2
    exit 2
  fi
done
mkdir -p "$OUTPUT_ROOT/logs"

# Complete and verify the shared feature cache in one foreground job before
# concurrent variants read it. This avoids concurrent cache population or
# manifest replacement while preserving one immutable full-method artifact.
preflight_gpu="${GPUS_ARRAY[0]}"
preflight_log="$OUTPUT_ROOT/logs/full_preflight_gpu_${preflight_gpu}.log"
export CUDA_VISIBLE_DEVICES="$preflight_gpu"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="$SEED"
"$PYTHON_BIN" scripts/build_fbdp_ad.py \
  --tasks "$TASKS" \
  --supports "$SUPPORTS" \
  --backbone-config "$BACKBONE_CONFIG" \
  --feature-cache "$FEATURE_CACHE" \
  --normal-signatures "$NORMAL_SIGNATURES" \
  --feature-view normal_fbdp \
  --device cuda:0 \
  --batch-size "$BATCH_SIZE" \
  --grid-shape "$GRID_ROWS" "$GRID_COLUMNS" \
  --ablation full \
  --diagnostics-limit "$DIAGNOSTICS_LIMIT" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_ROOT/full" > "$preflight_log" 2>&1
"$PYTHON_BIN" scripts/audit_fbdp_ad.py \
  --tasks "$TASKS" \
  --signatures "$OUTPUT_ROOT/full/fbdp_ad_signatures.jsonl" \
  --failures "$OUTPUT_ROOT/full/fbdp_ad_failures.json" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_ROOT/full/audit" >> "$preflight_log" 2>&1

run_worker() {
  local worker_index="$1"
  local physical_gpu="$2"
  local log_path="$OUTPUT_ROOT/logs/gpu_${physical_gpu}.log"
  local job_index
  local ablation
  local output_dir

  export CUDA_VISIBLE_DEVICES="$physical_gpu"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export PYTHONHASHSEED="$SEED"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
  {
    echo "physical_gpu=$physical_gpu logical_device=cuda:0"
    "$PYTHON_BIN" -c \
      'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
  } > "$log_path" 2>&1

  for ((job_index=worker_index; job_index<${#ABLATIONS[@]}; job_index+=${#GPUS_ARRAY[@]})); do
    ablation="${ABLATIONS[$job_index]}"
    output_dir="$OUTPUT_ROOT/$ablation"
    echo "GPU $physical_gpu computing FBDP: $ablation" | tee -a "$log_path"
    "$PYTHON_BIN" scripts/build_fbdp_ad.py \
      --tasks "$TASKS" \
      --supports "$SUPPORTS" \
      --backbone-config "$BACKBONE_CONFIG" \
      --feature-cache "$FEATURE_CACHE" \
      --normal-signatures "$NORMAL_SIGNATURES" \
      --feature-view normal_fbdp \
      --device cuda:0 \
      --batch-size "$BATCH_SIZE" \
      --grid-shape "$GRID_ROWS" "$GRID_COLUMNS" \
      --ablation "$ablation" \
      --diagnostics-limit "$DIAGNOSTICS_LIMIT" \
      --seed "$SEED" \
      --output-dir "$output_dir" >> "$log_path" 2>&1
    "$PYTHON_BIN" scripts/audit_fbdp_ad.py \
      --tasks "$TASKS" \
      --signatures "$output_dir/fbdp_ad_signatures.jsonl" \
      --failures "$output_dir/fbdp_ad_failures.json" \
      --seed "$SEED" \
      --output-dir "$output_dir/audit" >> "$log_path" 2>&1
  done
  echo "GPU $physical_gpu FBDP worker complete; log=$log_path"
}

pids=()
for index in "${!GPUS_ARRAY[@]}"; do
  run_worker "$index" "${GPUS_ARRAY[$index]}" &
  pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "GPU worker ${GPUS_ARRAY[$index]} failed; inspect its log" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi
echo "All ten FBDP ablations and label-free audits completed under $OUTPUT_ROOT"
