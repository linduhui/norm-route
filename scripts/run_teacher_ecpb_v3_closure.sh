#!/usr/bin/env bash
set -Eeuo pipefail

# Rebuild teacher/ECPB v3 and run the minimum innovation-closure matrix:
# 3 seeds x 5 folds x 8 variants = 120 real Router runs. The script is
# resumable, writes one bounded log per operation, and stops on every failure.

PROJECT_ROOT="${PROJECT_ROOT:-/data/gauss/ldh/projects/norm-route}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-teacher_ecpb_v3_closure_$(date +%Y%m%d_%H%M%S)}"
ROUTING_MATRIX="${ROUTING_MATRIX:-$PROJECT_ROOT/outputs/stage3/teacher_ecpb_latency_v3_20260819_064125/evaluator_only/routing_matrix_long.csv}"
FOLD_MANIFEST="${FOLD_MANIFEST:-$PROJECT_ROOT/outputs/stage5/splits/fold_manifest.csv}"
FEATURE_ROOT="${FEATURE_ROOT:-$PROJECT_ROOT/outputs/stage5/fbdp_router_cv_v2_20260805_054813/materialized}"
STAGE2_SUMMARY="${STAGE2_SUMMARY:-$PROJECT_ROOT/outputs/stage5/teacher_ecpb_latency_v3_20260819_064125/stage2_stage3_ready_summary.csv}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/stage5/$RUN_TAG}"
GPU_LIST="${GPUS:-0}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
L2_GRID="${L2_GRID:-0 0.0001 0.001}"

FOLDS=(fold0 fold1 fold2 fold3 fold4)
SEEDS=(0 1 2)
VARIANTS=(
  hard_oracle
  soft_teacher_legacy_target
  soft_teacher_no_bank
  soft_teacher_static
  soft_teacher_full
  soft_no_capability
  soft_no_uncertainty
  soft_no_cost
)

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Resolve the routing source through the same code boundary used by the
# teacher before any input existence check, content read, or hash.
ROUTING_MATRIX="$("$PYTHON_BIN" -c \
  'import sys; from normroute.router.teacher import ensure_evaluator_only_routing_matrix; print(ensure_evaluator_only_routing_matrix(sys.argv[1]))' \
  "$ROUTING_MATRIX")" || {
  echo "Routing matrix evaluator-only boundary check failed" >&2
  exit 2
}
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "GPUS must name at least one physical GPU" >&2
  exit 2
fi
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid physical GPU id: $gpu" >&2
    exit 2
  fi
done

for required in "$ROUTING_MATRIX" "$FOLD_MANIFEST" "$STAGE2_SUMMARY"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done
for fold in "${FOLDS[@]}"; do
  feature="$FEATURE_ROOT/$fold/normal_bir_fbdp/router_features.jsonl"
  if [[ ! -f "$feature" ]]; then
    echo "Missing Router features: $feature" >&2
    exit 2
  fi
done
hash_file() {
  "$PYTHON_BIN" -c \
    'import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); h=hashlib.sha256(); f=p.open("rb"); [h.update(c) for c in iter(lambda: f.read(1048576), b"")]; f.close(); print(h.hexdigest())' \
    "$1"
}
MATRIX_SHA256="$(hash_file "$ROUTING_MATRIX")"
FOLD_MANIFEST_SHA256="$(hash_file "$FOLD_MANIFEST")"
declare -A FEATURE_SHA256
for fold in "${FOLDS[@]}"; do
  feature="$FEATURE_ROOT/$fold/normal_bir_fbdp/router_features.jsonl"
  FEATURE_SHA256["$fold"]="$(hash_file "$feature")"
done
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Worktree (including untracked files) must be clean before experiment provenance is frozen" >&2
  exit 2
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
TEACHER_ROOT="$RUN_ROOT/evaluator_only"
LEGACY_TEACHER_ROOT="$RUN_ROOT/ablations/legacy_target/evaluator_only"
BANK_ROOT="$RUN_ROOT/capability_bank"
ROUTER_ROOT="$RUN_ROOT/router"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT" "$ROUTER_ROOT"
printf 'status=in_progress\ngit_commit=%s\n' "$CURRENT_COMMIT" >"$RUN_ROOT/STATUS"

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"
  echo "RUN $(basename "$log_path")"
  set +e
  "$@" >"$log_path" 2>&1
  local rc=$?
  set -e
  if (( rc != 0 )); then
    echo "FAILED rc=$rc log=$log_path" >&2
    tail -n 120 "$log_path" >&2
    return "$rc"
  fi
  echo "PASS $(basename "$log_path")"
}

run_logged "$LOG_ROOT/preflight_dependencies.log" \
  "$PYTHON_BIN" -c \
  'import pyarrow, numpy, torch; assert torch.cuda.is_available(); print(pyarrow.__version__, numpy.__version__, torch.__version__)'

# Complete pytest is a hard gate for the exact committed code used below.
run_logged "$LOG_ROOT/pytest_full.log" \
  "$PYTHON_BIN" -m pytest --junitxml "$RUN_ROOT/pytest_full.xml"

run_logged "$LOG_ROOT/build_teacher_v3.log" \
  "$PYTHON_BIN" -m normroute.cli.build_teacher_targets \
  --routing-matrix "$ROUTING_MATRIX" \
  --fold-manifest "$FOLD_MANIFEST" \
  --fold all \
  --output-dir "$TEACHER_ROOT" \
  --temperatures 0.25 0.5 1 2 4 \
  --minimum-probabilities 0.005 0.01 0.025 \
  --cost-weight 0.05 \
  --failure-penalty 2 \
  --calibration-strategy leave_one_train_category_out \
  --repeat-weighting equal_category_equal_query_inverse_variant_frequency \
  --sharpness-strategy train_robust_gap_validation_oracle_nll \
  --seed 0

run_logged "$LOG_ROOT/build_teacher_legacy_target.log" \
  "$PYTHON_BIN" -m normroute.cli.build_teacher_targets \
  --routing-matrix "$ROUTING_MATRIX" \
  --fold-manifest "$FOLD_MANIFEST" \
  --fold all \
  --output-dir "$LEGACY_TEACHER_ROOT" \
  --temperatures 0.05 0.1 0.25 0.5 1 \
  --minimum-probabilities 0 \
  --cost-weight 0.05 \
  --failure-penalty 2 \
  --calibration-strategy leave_one_train_category_out \
  --repeat-weighting equal_category_equal_query_inverse_variant_frequency \
  --sharpness-strategy legacy_raw_objective_soft_ce \
  --seed 0

for fold in "${FOLDS[@]}"; do
  feature="$FEATURE_ROOT/$fold/normal_bir_fbdp/router_features.jsonl"
  run_logged "$LOG_ROOT/build_bank_${fold}.log" \
    "$PYTHON_BIN" -m normroute.cli.build_expert_capability_bank \
    --teacher-data "$TEACHER_ROOT/$fold/teacher.parquet" \
    --router-features "$feature" \
    --stage2-summary "$STAGE2_SUMMARY" \
    --fold "$fold" \
    --output-dir "$BANK_ROOT/$fold" \
    --bootstrap-replicates 1000 \
    --seed 0
  run_logged "$LOG_ROOT/audit_teacher_bank_${fold}.log" \
    "$PYTHON_BIN" -m normroute.cli.audit_teacher_artifacts \
    --teacher-data "$TEACHER_ROOT/$fold/teacher.parquet" \
    --capability-bank "$BANK_ROOT/$fold/capability_bank.json" \
    --fold-manifest "$FOLD_MANIFEST" \
    --fold "$fold" \
    --output "$TEACHER_ROOT/$fold/dataset_audit.json"
done

is_valid_run() {
  local run_dir="$1"
  local fold="$2"
  local variant="$3"
  local seed="$4"
  local teacher_file="$5"
  local bank_file="$6"
  local feature_file="$7"
  local supervision="$8"
  local capability_mode="$9"
  local capability_grid_csv="${10}"
  local uncertainty_grid_csv="${11}"
  local cost_grid_csv="${12}"
  "$PYTHON_BIN" - \
    "$run_dir" "$fold" "$variant" "$seed" "$CURRENT_COMMIT" \
    "$teacher_file" "$bank_file" "$feature_file" "$supervision" \
    "$capability_mode" "$capability_grid_csv" "$uncertainty_grid_csv" \
    "$cost_grid_csv" "$FOLD_MANIFEST" "$ROUTING_MATRIX" \
    "${FEATURE_SHA256[$fold]}" "$FOLD_MANIFEST_SHA256" "$MATRIX_SHA256" \
    "$EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" "$L2_GRID" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    root, fold, variant, seed, commit, teacher, bank, feature,
    supervision, capability_mode, capability_csv, uncertainty_csv, cost_csv,
    manifest, matrix, feature_hash, manifest_hash, matrix_hash,
    epochs, batch_size, learning_rate, l2_grid,
) = sys.argv[1:]
root = Path(root)
required = [
    root / "run.json",
    root / "failures.json",
    root / "router_model.json",
    root / "predictions.jsonl",
    root / "evaluator_only" / "metrics.json",
    root / "evaluator_only" / "per_task_metrics.csv",
]
if not all(path.is_file() for path in required):
    raise SystemExit(1)
run = json.loads(required[0].read_text())
failures = json.loads(required[1].read_text())
config = run.get("config", {})
def numbers(value):
    return [float(item) for item in value]
def csv_numbers(value):
    return [float(item) for item in value.split(",") if item]
def same_path(observed, expected):
    if expected == "NONE":
        return observed is None
    return observed is not None and Path(observed).resolve() == Path(expected).resolve()
if not (
    run.get("ok") is True
    and run.get("failures") == []
    and failures == []
    and run.get("git_commit") == commit
    and config.get("fold") == fold
    and config.get("variant") == variant
    and int(config.get("seed", -1)) == int(seed)
    and config.get("supervision") == supervision
    and config.get("supervision_target") == supervision
    and config.get("feature_view") == "normal_bir_fbdp"
    and config.get("bir_ablation") == "full"
    and config.get("fbdp_ablation") == "full"
    and config.get("capability_mode") == capability_mode
    and config.get("capability_skills") == ["boundary", "fgbg", "lowshot", "texture"]
    and numbers(config.get("capability_weight_grid", [])) == csv_numbers(capability_csv)
    and numbers(config.get("uncertainty_weight_grid", [])) == csv_numbers(uncertainty_csv)
    and numbers(config.get("cost_weight_grid", [])) == csv_numbers(cost_csv)
    and config.get("uncertainty_mode") == "entropy_scaled_lcb"
    and float(config.get("validation_runtime_tradeoff", -1)) == 0.05
    and config.get("validation_selection_metric") == "train_calibrated_zero_one_error_plus_normalized_runtime"
    and config.get("repeat_weighting") == "equal_category_equal_query_inverse_variant_frequency"
    and int(config.get("epochs", -1)) == int(epochs)
    and int(config.get("batch_size", -1)) == int(batch_size)
    and float(config.get("learning_rate", -1)) == float(learning_rate)
    and numbers(config.get("l2_grid", [])) == numbers(l2_grid.split())
    and config.get("device") == "cuda:0"
    and config.get("model_kind") == "class_balanced_linear_softmax_soft_targets"
    and config.get("standardization") == "train_only_feature_mean_std"
    and config.get("test_prediction_contract") == "label_free_and_frozen_before_evaluator_join"
):
    raise SystemExit(1)
if not all((
    same_path(config.get("teacher_data"), teacher),
    same_path(config.get("capability_bank"), bank),
    same_path(config.get("router_features"), feature),
    same_path(config.get("fold_manifest"), manifest),
    same_path(config.get("routing_matrix"), matrix),
)):
    raise SystemExit(1)
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
expected_hashes = {
    "router_features": feature_hash,
    "fold_manifest": manifest_hash,
    "routing_matrix": matrix_hash,
}
if teacher != "NONE":
    expected_hashes["teacher_data"] = sha256(teacher)
if bank != "NONE":
    expected_hashes["capability_bank"] = sha256(bank)
if run.get("input_hashes") != expected_hashes:
    raise SystemExit(1)
expected_outputs = {
    "model": root / "router_model.json",
    "predictions": root / "predictions.jsonl",
    "metrics": root / "evaluator_only" / "metrics.json",
    "per_task_metrics": root / "evaluator_only" / "per_task_metrics.csv",
    "failures": root / "failures.json",
}
if set(run.get("output_hashes", {})) != set(expected_outputs):
    raise SystemExit(1)
for name, expected_path in expected_outputs.items():
    path = run.get("outputs", {}).get(name)
    if not same_path(path, str(expected_path)):
        raise SystemExit(1)
    if sha256(expected_path) != run["output_hashes"][name]:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_router_job() {
  local fold="$1"
  local variant="$2"
  local seed="$3"
  local physical_gpu="$4"
  local feature_file="$FEATURE_ROOT/$fold/normal_bir_fbdp/router_features.jsonl"
  local teacher_file="$TEACHER_ROOT/$fold/teacher.parquet"
  local bank_file="$BANK_ROOT/$fold/capability_bank.json"
  local supervision="soft_teacher"
  local capability_mode="conditional"
  local capability_grid=(0 0.25 0.5)
  local uncertainty_grid=(0 0.25 0.5 1 2)
  local cost_grid=(0 0.05 0.1)

  case "$variant" in
    hard_oracle)
      supervision="hard_oracle"
      teacher_file="NONE"
      bank_file="NONE"
      ;;
    soft_teacher_legacy_target)
      teacher_file="$LEGACY_TEACHER_ROOT/$fold/teacher.parquet"
      bank_file="NONE"
      ;;
    soft_teacher_no_bank)
      bank_file="NONE"
      ;;
    soft_teacher_static)
      capability_mode="static"
      ;;
    soft_teacher_full)
      ;;
    soft_no_capability)
      capability_grid=(0)
      ;;
    soft_no_uncertainty)
      uncertainty_grid=(0)
      ;;
    soft_no_cost)
      cost_grid=(0)
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      return 2
      ;;
  esac

  local capability_grid_csv uncertainty_grid_csv cost_grid_csv
  capability_grid_csv="$(IFS=,; echo "${capability_grid[*]}")"
  uncertainty_grid_csv="$(IFS=,; echo "${uncertainty_grid[*]}")"
  cost_grid_csv="$(IFS=,; echo "${cost_grid[*]}")"

  local output_dir="$ROUTER_ROOT/seed${seed}/$fold/$variant"
  local log_file="$ROUTER_ROOT/seed${seed}/logs/${fold}_${variant}.log"
  mkdir -p "$output_dir" "$(dirname "$log_file")"
  if is_valid_run \
    "$output_dir" "$fold" "$variant" "$seed" "$teacher_file" \
    "$bank_file" "$feature_file" "$supervision" "$capability_mode" \
    "$capability_grid_csv" "$uncertainty_grid_csv" "$cost_grid_csv"; then
    echo "SKIP valid seed=$seed fold=$fold variant=$variant"
    return 0
  fi

  local command=(
    "$PYTHON_BIN" -m normroute.cli.run_stage5_router
    --router-features "$feature_file"
    --fold-manifest "$FOLD_MANIFEST"
    --routing-matrix "$ROUTING_MATRIX"
    --fold "$fold"
    --variant "$variant"
    --feature-view normal_bir_fbdp
    --bir-ablation full
    --fbdp-ablation full
    --supervision "$supervision"
    --capability-mode "$capability_mode"
    --capability-skills boundary fgbg lowshot texture
    --capability-weight-grid "${capability_grid[@]}"
    --uncertainty-mode entropy_scaled_lcb
    --uncertainty-weight-grid "${uncertainty_grid[@]}"
    --cost-weight-grid "${cost_grid[@]}"
    --validation-runtime-tradeoff 0.05
    --device cuda:0
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --l2-grid $L2_GRID
    --seed "$seed"
    --output-dir "$output_dir"
  )
  if [[ "$teacher_file" != "NONE" ]]; then
    command+=(--teacher-data "$teacher_file")
  fi
  if [[ "$bank_file" != "NONE" ]]; then
    command+=(--capability-bank "$bank_file")
  fi

  echo "RUN gpu=$physical_gpu seed=$seed fold=$fold variant=$variant"
  set +e
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  PYTHONHASHSEED="$seed" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "${command[@]}" >"$log_file" 2>&1
  local rc=$?
  set -e
  if (( rc != 0 )); then
    echo "FAILED gpu=$physical_gpu seed=$seed fold=$fold variant=$variant rc=$rc" >&2
    tail -n 120 "$log_file" >&2
    return "$rc"
  fi
  if ! is_valid_run \
    "$output_dir" "$fold" "$variant" "$seed" "$teacher_file" \
    "$bank_file" "$feature_file" "$supervision" "$capability_mode" \
    "$capability_grid_csv" "$uncertainty_grid_csv" "$cost_grid_csv"; then
    echo "FAILED post-run validation: $output_dir" >&2
    return 1
  fi
  echo "PASS gpu=$physical_gpu seed=$seed fold=$fold variant=$variant"
}

JOB_SEEDS=()
JOB_FOLDS=()
JOB_VARIANTS=()
for seed in "${SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      JOB_SEEDS+=("$seed")
      JOB_FOLDS+=("$fold")
      JOB_VARIANTS+=("$variant")
    done
  done
done

run_worker() {
  local worker_index="$1"
  local physical_gpu="$2"
  local job_index
  for ((job_index=worker_index; job_index<${#JOB_SEEDS[@]}; job_index+=${#GPU_ARRAY[@]})); do
    run_router_job \
      "${JOB_FOLDS[$job_index]}" \
      "${JOB_VARIANTS[$job_index]}" \
      "${JOB_SEEDS[$job_index]}" \
      "$physical_gpu" || return $?
  done
}

pids=()
for index in "${!GPU_ARRAY[@]}"; do
  run_worker "$index" "${GPU_ARRAY[$index]}" &
  pids+=("$!")
done
worker_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    worker_failed=1
  fi
done
if (( worker_failed != 0 )); then
  echo "At least one GPU worker failed; inspect $ROUTER_ROOT/seed*/logs" >&2
  exit 1
fi

summary_pairs=(
  --paired-comparison hard_oracle soft_teacher_full hard_to_full
  --paired-comparison soft_teacher_legacy_target soft_teacher_no_bank sharpness_gain
  --paired-comparison soft_teacher_no_bank soft_teacher_full ecpb_gain
  --paired-comparison soft_teacher_static soft_teacher_full conditional_profile_gain
  --paired-comparison soft_no_capability soft_teacher_full capability_gain
  --paired-comparison soft_no_uncertainty soft_teacher_full uncertainty_gain
  --paired-comparison soft_no_cost soft_teacher_full latency_cost_gain
)

for seed in "${SEEDS[@]}"; do
  run_logged "$ROUTER_ROOT/seed${seed}/summary_v3.log" \
    "$PYTHON_BIN" -m normroute.cli.summarize_stage5_router \
    --input-root "$ROUTER_ROOT/seed${seed}" \
    --output-dir "$ROUTER_ROOT/seed${seed}/summary_v3" \
    --folds "${FOLDS[@]}" \
    --variants "${VARIANTS[@]}" \
    --experiment-family teacher_ecpb \
    --reference-variant soft_teacher_full \
    "${summary_pairs[@]}"
done

run_logged "$ROUTER_ROOT/summary_multiseed_v3.log" \
  "$PYTHON_BIN" -m normroute.cli.summarize_stage5_router_multiseed \
  --input-root "$ROUTER_ROOT" \
  --output-dir "$ROUTER_ROOT/summary_multiseed_v3" \
  --seeds "${SEEDS[@]}" \
  --folds "${FOLDS[@]}" \
  --variants "${VARIANTS[@]}" \
  "${summary_pairs[@]}" \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 0

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ERROR: worktree changed while the closure experiment was running" >&2
  exit 1
fi

printf 'status=auditing\ngit_commit=%s\n' "$CURRENT_COMMIT" >"$RUN_ROOT/STATUS"
run_logged "$LOG_ROOT/final_acceptance_v3.log" \
  "$PYTHON_BIN" -m normroute.cli.audit_teacher_ecpb_v3_closure \
  --run-root "$RUN_ROOT" \
  --routing-matrix "$ROUTING_MATRIX" \
  --output "$RUN_ROOT/final_acceptance_v3.json"

AUDIT_SHA256="$(hash_file "$RUN_ROOT/final_acceptance_v3.json")"
SUMMARY_SHA256="$(hash_file "$ROUTER_ROOT/summary_multiseed_v3/multiseed_paired_deltas.csv")"
{
  printf 'run_tag=%s\n' "$RUN_TAG"
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'git_commit=%s\n' "$CURRENT_COMMIT"
  printf 'router_runs=%s\n' "${#JOB_SEEDS[@]}"
  printf 'audit_sha256=%s\n' "$AUDIT_SHA256"
  printf 'multiseed_paired_deltas_sha256=%s\n' "$SUMMARY_SHA256"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
} >"$RUN_ROOT/COMPLETE.txt"
printf 'status=complete\ngit_commit=%s\n' "$CURRENT_COMMIT" >"$RUN_ROOT/STATUS"

echo "Teacher/ECPB v3 innovation closure PASS"
echo "RUN_ROOT=$RUN_ROOT"
