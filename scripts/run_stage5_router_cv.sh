#!/usr/bin/env bash
set -Eeuo pipefail

# Train/evaluate the same linear Router across all strict BIR-AD feature views.
# One sequential worker is assigned to each visible physical GPU.

PROJECT_ROOT="${PROJECT_ROOT:-/data/gauss/ldh/projects/norm-route}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_ROOT="${FEATURE_ROOT:-outputs/stage5/bir_ad_ablations_strict_v1}"
FOLD_MANIFEST="${FOLD_MANIFEST:-outputs/stage5/splits/fold_manifest.csv}"
ROUTING_MATRIX="${ROUTING_MATRIX:-outputs/stage3/evaluator_only/runtime_refresh/routing_matrix_long.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage5/router_cv_strict_v1}"
GPU_LIST="${GPUS:-0,1,2,3,4,5}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
L2_GRID="${L2_GRID:-0 0.0001 0.001}"
SEED="${SEED:-0}"

FOLDS=(fold0 fold1 fold2 fold3 fold4)
VARIANTS=(
  normal_only
  sigma_l2
  plus_sobel
  plus_structural_boundary
  plus_directional_evidence
  plus_cross_modal_disagreement
  plus_support_consistency
  representations_only
  full
)

cd "$PROJECT_ROOT"
IFS=',' read -r -a GPUS_ARRAY <<< "$GPU_LIST"
if (( ${#GPUS_ARRAY[@]} == 0 )); then
  echo "GPUS must contain at least one physical GPU id" >&2
  exit 2
fi
for path in "$FOLD_MANIFEST" "$ROUTING_MATRIX"; do
  if [[ ! -f "$path" ]]; then
    echo "Required artifact does not exist: $path" >&2
    exit 2
  fi
done
for fold in "${FOLDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    source_variant="$variant"
    if [[ "$variant" == "normal_only" ]]; then
      source_variant="sigma_l2"
    fi
    path="$FEATURE_ROOT/$fold/$source_variant/router_features.jsonl"
    if [[ ! -f "$path" ]]; then
      echo "Required Router features do not exist: $path" >&2
      exit 2
    fi
  done
done

mkdir -p "$OUTPUT_ROOT/logs"
JOB_FOLDS=()
JOB_VARIANTS=()
for fold in "${FOLDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    JOB_FOLDS+=("$fold")
    JOB_VARIANTS+=("$variant")
  done
done

run_gpu_worker() {
  local worker_index="$1"
  local physical_gpu="$2"
  local log_path="$OUTPUT_ROOT/logs/gpu_${physical_gpu}.log"
  local job_index
  local fold
  local variant
  local source_variant
  local feature_view

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

  for ((job_index=worker_index; job_index<${#JOB_FOLDS[@]}; job_index+=${#GPUS_ARRAY[@]})); do
    fold="${JOB_FOLDS[$job_index]}"
    variant="${JOB_VARIANTS[$job_index]}"
    source_variant="$variant"
    feature_view="all"
    if [[ "$variant" == "normal_only" ]]; then
      source_variant="sigma_l2"
      feature_view="normal_only"
    fi
    echo "[$fold] GPU $physical_gpu training Router: $variant" | tee -a "$log_path"
    # L2_GRID is intentionally shell-split into numeric argparse values.
    # shellcheck disable=SC2086
    "$PYTHON_BIN" scripts/run_stage5_router.py \
      --router-features "$FEATURE_ROOT/$fold/$source_variant/router_features.jsonl" \
      --fold-manifest "$FOLD_MANIFEST" \
      --routing-matrix "$ROUTING_MATRIX" \
      --fold "$fold" \
      --variant "$variant" \
      --feature-view "$feature_view" \
      --device cuda:0 \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --l2-grid $L2_GRID \
      --seed "$SEED" \
      --output-dir "$OUTPUT_ROOT/$fold/$variant" >> "$log_path" 2>&1
  done
  echo "GPU $physical_gpu Router worker complete; log=$log_path"
}

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

"$PYTHON_BIN" scripts/summarize_stage5_router.py \
  --input-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/summary"

echo "All strict five-fold Router runs completed under $OUTPUT_ROOT"
