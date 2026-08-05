#!/usr/bin/env bash
set -Eeuo pipefail

# Run FBDP mechanism ablations and the four primary Router feature views over
# identical five-fold manifests. Test evaluator data remains isolated inside
# run_stage5_router.py until label-free predictions have been written.

PROJECT_ROOT="${PROJECT_ROOT:-/data/gauss/ldh/projects/norm-route}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FBDP_ROOT="${FBDP_ROOT:-outputs/stage5/fbdp_ad_ablations_v1}"
BIR_ROOT="${BIR_ROOT:-outputs/stage5/bir_ad_ablations_gpu_v2}"
NORMAL_SIGNATURES="${NORMAL_SIGNATURES:-outputs/stage5/normal_domain/normal_signatures.parquet}"
FOLD_MANIFEST="${FOLD_MANIFEST:-outputs/stage5/splits/fold_manifest.csv}"
ROUTING_MATRIX="${ROUTING_MATRIX:-outputs/stage3/evaluator_only/runtime_refresh/routing_matrix_long.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage5/fbdp_router_cv_v1}"
GPU_LIST="${GPUS:-0,1,2,3,4,5}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
L2_GRID="${L2_GRID:-0 0.0001 0.001}"
SEED="${SEED:-0}"

FOLDS=(fold0 fold1 fold2 fold3 fold4)
FBDP_ABLATIONS=(
  single_bank
  no_fbc
  no_objectness_gate
  border_only_background
  low_objectness_only_background
  no_cross_support_consistency
  pooling
  same_position_consistency
  raw_residual_only
  full
)

cd "$PROJECT_ROOT"
IFS=',' read -r -a GPUS_ARRAY <<< "$GPU_LIST"
if (( ${#GPUS_ARRAY[@]} == 0 )); then
  echo "GPUS must contain at least one physical GPU id" >&2
  exit 2
fi
for path in "$NORMAL_SIGNATURES" "$FOLD_MANIFEST" "$ROUTING_MATRIX"; do
  if [[ ! -f "$path" ]]; then
    echo "Required artifact does not exist: $path" >&2
    exit 2
  fi
done
for ablation in "${FBDP_ABLATIONS[@]}"; do
  for name in fbdp_ad_signatures.jsonl router_features.jsonl fbdp_ad_failures.json; do
    path="$FBDP_ROOT/$ablation/$name"
    if [[ ! -f "$path" ]]; then
      echo "Required FBDP artifact does not exist: $path" >&2
      exit 2
    fi
  done
done
for fold in "${FOLDS[@]}"; do
  path="$BIR_ROOT/$fold/full/bir_ad_signatures.jsonl"
  if [[ ! -f "$path" ]]; then
    echo "Required BIR artifact does not exist: $path" >&2
    exit 2
  fi
  "$PYTHON_BIN" scripts/materialize_fbdp_ad_features.py \
    --normal-signatures "$NORMAL_SIGNATURES" \
    --fbdp-signatures "$FBDP_ROOT/full/fbdp_ad_signatures.jsonl" \
    --fbdp-ablation full \
    --bir-signatures "$path" \
    --bir-ablation full \
    --feature-view normal_bir_fbdp \
    --seed "$SEED" \
    --output-dir "$OUTPUT_ROOT/materialized/$fold/normal_bir_fbdp"
done

mkdir -p "$OUTPUT_ROOT/logs"
JOB_FOLDS=()
JOB_VARIANTS=()
JOB_VIEWS=()
JOB_FEATURES=()
JOB_FBDP_ABLATIONS=()
for fold in "${FOLDS[@]}"; do
  joint_features="$OUTPUT_ROOT/materialized/$fold/normal_bir_fbdp/router_features.jsonl"
  JOB_FOLDS+=("$fold" "$fold")
  JOB_VARIANTS+=(normal_only normal_bir)
  JOB_VIEWS+=(normal_only normal_bir)
  JOB_FEATURES+=("$joint_features" "$joint_features")
  JOB_FBDP_ABLATIONS+=(full full)
  for ablation in "${FBDP_ABLATIONS[@]}"; do
    JOB_FOLDS+=("$fold")
    JOB_VARIANTS+=("normal_fbdp__$ablation")
    JOB_VIEWS+=(normal_fbdp)
    JOB_FEATURES+=("$FBDP_ROOT/$ablation/router_features.jsonl")
    JOB_FBDP_ABLATIONS+=("$ablation")
  done
  JOB_FOLDS+=("$fold")
  JOB_VARIANTS+=(normal_bir_fbdp)
  JOB_VIEWS+=(normal_bir_fbdp)
  JOB_FEATURES+=("$joint_features")
  JOB_FBDP_ABLATIONS+=(full)
done

run_worker() {
  local worker_index="$1"
  local physical_gpu="$2"
  local log_path="$OUTPUT_ROOT/logs/gpu_${physical_gpu}.log"
  local job_index
  local fold
  local variant
  local feature_view
  local feature_path
  local fbdp_ablation

  export CUDA_VISIBLE_DEVICES="$physical_gpu"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export PYTHONHASHSEED="$SEED"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
  : > "$log_path"
  for ((job_index=worker_index; job_index<${#JOB_FOLDS[@]}; job_index+=${#GPUS_ARRAY[@]})); do
    fold="${JOB_FOLDS[$job_index]}"
    variant="${JOB_VARIANTS[$job_index]}"
    feature_view="${JOB_VIEWS[$job_index]}"
    feature_path="${JOB_FEATURES[$job_index]}"
    fbdp_ablation="${JOB_FBDP_ABLATIONS[$job_index]}"
    echo "[$fold] GPU $physical_gpu Router: $variant" | tee -a "$log_path"
    # L2_GRID is intentionally shell-split into numeric argparse values.
    # shellcheck disable=SC2086
    "$PYTHON_BIN" scripts/run_stage5_router.py \
      --router-features "$feature_path" \
      --fold-manifest "$FOLD_MANIFEST" \
      --routing-matrix "$ROUTING_MATRIX" \
      --fold "$fold" \
      --variant "$variant" \
      --feature-view "$feature_view" \
      --bir-ablation full \
      --fbdp-ablation "$fbdp_ablation" \
      --device cuda:0 \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --l2-grid $L2_GRID \
      --seed "$SEED" \
      --output-dir "$OUTPUT_ROOT/$fold/$variant" >> "$log_path" 2>&1
  done
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

SUMMARY_VARIANTS=(normal_only normal_bir)
for ablation in "${FBDP_ABLATIONS[@]}"; do
  SUMMARY_VARIANTS+=("normal_fbdp__$ablation")
done
SUMMARY_VARIANTS+=(normal_bir_fbdp)
summary_args=(
  --input-root "$OUTPUT_ROOT"
  --output-dir "$OUTPUT_ROOT/summary"
  --variants "${SUMMARY_VARIANTS[@]}"
  --reference-variant normal_bir_fbdp
  --paired-comparison normal_only normal_fbdp__full add_fbdp
  --paired-comparison normal_only normal_bir add_bir
  --paired-comparison normal_bir normal_bir_fbdp add_fbdp_to_bir
  --paired-comparison normal_fbdp__full normal_bir_fbdp add_bir_to_fbdp
)
for ablation in "${FBDP_ABLATIONS[@]}"; do
  if [[ "$ablation" != "full" ]]; then
    summary_args+=(
      --paired-comparison "normal_fbdp__$ablation" normal_fbdp__full "${ablation}_to_full"
    )
  fi
done
"$PYTHON_BIN" scripts/summarize_stage5_router.py "${summary_args[@]}"
echo "All FBDP five-fold Router runs completed under $OUTPUT_ROOT"
