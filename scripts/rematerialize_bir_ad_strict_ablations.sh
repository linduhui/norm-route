#!/usr/bin/env bash
set -Eeuo pipefail

# Rebuild only the Router feature views under the strict v2 ablation contract.
# Existing BIR signatures are reused only when their compute settings exactly
# match the requested target.  Source experiment artifacts are never modified.

PROJECT_ROOT="${PROJECT_ROOT:-/data/gauss/ldh/projects/norm-route}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/stage5/bir_ad_ablations_gpu_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage5/bir_ad_ablations_strict_v1}"
NORMAL_SIGNATURES="${NORMAL_SIGNATURES:-outputs/stage5/normal_domain/normal_signatures.parquet}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
SEED="${SEED:-0}"

FOLDS=(fold0 fold1 fold2 fold3 fold4)
ABLATIONS=(
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
if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "$NORMAL_SIGNATURES" ]]; then
  echo "Required normal signatures do not exist: $NORMAL_SIGNATURES" >&2
  exit 2
fi
for fold in "${FOLDS[@]}"; do
  for ablation in "${ABLATIONS[@]}"; do
    source_dir="$SOURCE_ROOT/$fold/$ablation"
    for name in \
      bir_ad_ablation.json \
      bir_ad_build_run.json \
      bir_ad_failures.json \
      bir_ad_fold_normalization.json \
      bir_ad_signatures.jsonl; do
      if [[ ! -f "$source_dir/$name" ]]; then
        echo "Required source artifact does not exist: $source_dir/$name" >&2
        exit 2
      fi
    done
  done
done

mkdir -p "$OUTPUT_ROOT/logs"
JOB_FOLDS=()
JOB_ABLATIONS=()
for fold in "${FOLDS[@]}"; do
  for ablation in "${ABLATIONS[@]}"; do
    JOB_FOLDS+=("$fold")
    JOB_ABLATIONS+=("$ablation")
  done
done

run_worker() {
  local worker_index="$1"
  local job_index
  local fold
  local ablation
  local log_path="$OUTPUT_ROOT/logs/materialize_worker_${worker_index}.log"
  : > "$log_path"
  for ((job_index=worker_index; job_index<${#JOB_FOLDS[@]}; job_index+=MAX_PARALLEL)); do
    fold="${JOB_FOLDS[$job_index]}"
    ablation="${JOB_ABLATIONS[$job_index]}"
    echo "[$fold] strict materialization: $ablation" | tee -a "$log_path"
    "$PYTHON_BIN" scripts/materialize_bir_ad_ablation.py \
      --source-output "$SOURCE_ROOT/$fold/$ablation" \
      --normal-signatures "$NORMAL_SIGNATURES" \
      --ablation "$ablation" \
      --seed "$SEED" \
      --output-dir "$OUTPUT_ROOT/$fold/$ablation" >> "$log_path" 2>&1
  done
}

pids=()
for ((worker=0; worker<MAX_PARALLEL; worker+=1)); do
  run_worker "$worker" &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Strict materialization worker $index failed; inspect logs" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

echo "All strict fold/ablation feature artifacts completed under $OUTPUT_ROOT"

