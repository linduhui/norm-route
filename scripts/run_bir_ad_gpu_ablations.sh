#!/usr/bin/env bash
set -Eeuo pipefail

# Five-fold BIR-AD ablation runner with one sequential worker per physical GPU.
# The source feature cache and fold normalizations must already exist. Override
# any path or tuning value through an environment variable before invoking it.

PROJECT_ROOT="${PROJECT_ROOT:-/data/gauss/ldh/projects/norm-route}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS="${TASKS:-outputs/stage5/tasks/bir_ad_tasks.jsonl}"
SUPPORTS="${SUPPORTS:-outputs/stage5/supports/bir_ad_supports.csv}"
NORMALIZATION_ROOT="${NORMALIZATION_ROOT:-outputs/stage5/bir_ad}"
BACKBONE_CONFIG="${BACKBONE_CONFIG:-configs/stage5/router_backbone.yaml}"
FEATURE_CACHE="${FEATURE_CACHE:-outputs/stage5/features}"
NORMAL_SIGNATURES="${NORMAL_SIGNATURES:-outputs/stage5/normal_domain/normal_signatures.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage5/bir_ad_ablations_gpu_v2}"
GPU_LIST="${GPUS:-0,1,2,3,4,5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CONSISTENCY_CHUNK_SIZE="${CONSISTENCY_CHUNK_SIZE:-4096}"
GRID_ROWS="${GRID_ROWS:-37}"
GRID_COLUMNS="${GRID_COLUMNS:-37}"

FOLDS=(fold0 fold1 fold2 fold3 fold4)
COMPUTE_ABLATIONS=(
  sigma_l2
  plus_sobel
  plus_structural_boundary
  plus_directional_evidence
  full
)
MATERIALIZED_ABLATIONS=(
  plus_cross_modal_disagreement
  plus_support_consistency
  representations_only
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
for fold in "${FOLDS[@]}"; do
  required_files+=(
    "$NORMALIZATION_ROOT/$fold/bir_ad_fold_normalization.json"
  )
done
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Required artifact does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT/logs"

run_compute_job() {
  local fold="$1"
  local ablation="$2"
  local physical_gpu="$3"
  local log_path="$4"
  local normalization="$NORMALIZATION_ROOT/$fold/bir_ad_fold_normalization.json"
  local output_dir="$OUTPUT_ROOT/$fold/$ablation"

  echo "[$fold] GPU $physical_gpu computing $ablation"
  "$PYTHON_BIN" scripts/build_bir_ad.py \
    --tasks "$TASKS" \
    --supports "$SUPPORTS" \
    --normalization-artifact "$normalization" \
    --backbone-config "$BACKBONE_CONFIG" \
    --feature-cache "$FEATURE_CACHE" \
    --normal-signatures "$NORMAL_SIGNATURES" \
    --device cuda:0 \
    --bir-consistency-backend torch \
    --bir-device cuda:0 \
    --bir-consistency-dtype float64 \
    --consistency-chunk-size "$CONSISTENCY_CHUNK_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --grid-shape "$GRID_ROWS" "$GRID_COLUMNS" \
    --ablation "$ablation" \
    --output-dir "$output_dir" >> "$log_path" 2>&1
}

run_gpu_worker() {
  local worker_index="$1"
  local physical_gpu="$2"
  local log_path="$OUTPUT_ROOT/logs/gpu_${physical_gpu}.log"
  local job_index
  local fold
  local ablation

  export CUDA_VISIBLE_DEVICES="$physical_gpu"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
  {
    echo "physical_gpu=$physical_gpu logical_device=cuda:0"
    "$PYTHON_BIN" -c \
      'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
  } > "$log_path" 2>&1

  for ((job_index=worker_index; job_index<${#JOB_FOLDS[@]}; job_index+=${#GPUS_ARRAY[@]})); do
    fold="${JOB_FOLDS[$job_index]}"
    ablation="${JOB_ABLATIONS[$job_index]}"
    run_compute_job "$fold" "$ablation" "$physical_gpu" "$log_path"
  done
  echo "GPU $physical_gpu compute worker complete; log=$log_path"
}

materialize_fold() {
  local fold="$1"
  local log_path="$OUTPUT_ROOT/logs/${fold}_materialize.log"
  local ablation
  local output_dir

  : > "$log_path"
  for ablation in "${MATERIALIZED_ABLATIONS[@]}"; do
    output_dir="$OUTPUT_ROOT/$fold/$ablation"
    echo "[$fold] materializing $ablation from full"
    "$PYTHON_BIN" scripts/materialize_bir_ad_ablation.py \
      --source-output "$OUTPUT_ROOT/$fold/full" \
      --normal-signatures "$NORMAL_SIGNATURES" \
      --ablation "$ablation" \
      --output-dir "$output_dir" >> "$log_path" 2>&1
  done
  echo "[$fold] materialization complete; log=$log_path"
}

JOB_FOLDS=()
JOB_ABLATIONS=()
for fold in "${FOLDS[@]}"; do
  for ablation in "${COMPUTE_ABLATIONS[@]}"; do
    JOB_FOLDS+=("$fold")
    JOB_ABLATIONS+=("$ablation")
  done
done

pids=()
for index in "${!GPUS_ARRAY[@]}"; do
  run_gpu_worker "$index" "${GPUS_ARRAY[$index]}" &
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

materialize_pids=()
for fold in "${FOLDS[@]}"; do
  materialize_fold "$fold" &
  materialize_pids+=("$!")
done
for index in "${!materialize_pids[@]}"; do
  if ! wait "${materialize_pids[$index]}"; then
    echo "${FOLDS[$index]} materialization failed" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

echo "All five folds and eight ablations completed under $OUTPUT_ROOT"
