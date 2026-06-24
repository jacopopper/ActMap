#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT/artifacts/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/features/actmap}"
GENERATION_ROOT="${GENERATION_ROOT:-$ROOT/artifacts/generations}"
LOG_ROOT="${LOG_ROOT:-$ROOT/artifacts/logs/sprint3_actmaps}"

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SKIP_INPUT_CHECK="${SKIP_INPUT_CHECK:-0}"
OVERWRITE="${OVERWRITE:-0}"
LIMIT="${LIMIT:-}"
SAVE_EVERY="${SAVE_EVERY:-100}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SMALL_MAX_MODEL_LEN="${SMALL_MAX_MODEL_LEN:-4096}"
QWEN32_MAX_MODEL_LEN="${QWEN32_MAX_MODEL_LEN:-4096}"

GPUS=(${GPUS:-0 1 2 3})
DATASETS=(
  triviaqa_no_context
  nq_open
  web_questions
  gsm8k_rationale
  cnn_dailymail_3_0_0
)
SMALL_MODELS=(
  "Qwen/Qwen3-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)
QWEN32_MODEL="Qwen/Qwen3-32B"

if [[ "$SMOKE" == "1" ]]; then
  DATASETS=(triviaqa_no_context)
  SMALL_MODELS=("Qwen/Qwen3-8B")
  LIMIT="${LIMIT:-1}"
fi

mkdir -p "$LOG_ROOT"

require_four_gpus() {
  if [[ "${#GPUS[@]}" -lt 4 ]]; then
    echo "Qwen3-32B phase expects 4 GPU ids in GPUS; got: ${GPUS[*]}" >&2
    exit 2
  fi
}

model_slug() {
  "$PYTHON" - "$1" <<'PY'
import re
import sys
print(re.sub(r"[^A-Za-z0-9_.-]+", "__", sys.argv[1]).strip("_"))
PY
}

max_new_tokens() {
  case "$1" in
    gsm8k_rationale) echo 128 ;;
    cnn_dailymail_3_0_0) echo 256 ;;
    *) echo 32 ;;
  esac
}

check_inputs() {
  if [[ "$SKIP_INPUT_CHECK" == "1" || "$DRY_RUN" == "1" ]]; then
    return
  fi
  for dataset in "${DATASETS[@]}"; do
    for required in source_records.parquet splits.parquet; do
      local path="$DATA_ROOT/$dataset/$required"
      if [[ ! -f "$path" ]]; then
        echo "missing required Sprint 2 artifact: $path" >&2
        exit 2
      fi
    done
  done
}

run_pair() {
  local cuda_devices="$1"
  local dataset="$2"
  local model="$3"
  local tensor_parallel_size="$4"
  local max_model_len="$5"
  local slug
  slug="$(model_slug "$model")"
  local log="$LOG_ROOT/${dataset}__${slug}.log"
  local cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$cuda_devices"
    "$PYTHON" -m src.generate_actmaps
    --dataset-id "$dataset"
    --model "$model"
    --data-root "$DATA_ROOT"
    --output-root "$OUTPUT_ROOT"
    --generation-root "$GENERATION_ROOT"
    --max-new-tokens "$(max_new_tokens "$dataset")"
    --max-model-len "$max_model_len"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$tensor_parallel_size"
    --save-every "$SAVE_EVERY"
  )
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf ' > %q 2>&1\n' "$log"
    return
  fi
  echo "[$(date -Is)] start dataset=$dataset model=$model gpus=$cuda_devices tp=$tensor_parallel_size log=$log"
  "${cmd[@]}" >"$log" 2>&1
  echo "[$(date -Is)] done  dataset=$dataset model=$model"
}

run_small_model_phase() {
  local jobs=()
  for dataset in "${DATASETS[@]}"; do
    for model in "${SMALL_MODELS[@]}"; do
      jobs+=("$dataset|$model")
    done
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    for i in "${!jobs[@]}"; do
      IFS='|' read -r dataset model <<<"${jobs[$i]}"
      local gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
      run_pair "$gpu" "$dataset" "$model" 1 "$SMALL_MAX_MODEL_LEN"
    done
    return
  fi

  local pids=()
  for lane in "${!GPUS[@]}"; do
    local gpu="${GPUS[$lane]}"
    (
      for ((i=lane; i<${#jobs[@]}; i+=${#GPUS[@]})); do
        IFS='|' read -r dataset model <<<"${jobs[$i]}"
        run_pair "$gpu" "$dataset" "$model" 1 "$SMALL_MAX_MODEL_LEN"
      done
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

run_qwen32_phase() {
  require_four_gpus
  local all_gpus
  all_gpus="$(IFS=,; echo "${GPUS[*]:0:4}")"
  for dataset in "${DATASETS[@]}"; do
    run_pair "$all_gpus" "$dataset" "$QWEN32_MODEL" 4 "$QWEN32_MAX_MODEL_LEN"
  done
}

echo "Sprint 3 ActMap runner"
echo "root=$ROOT"
echo "data_root=$DATA_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "generation_root=$GENERATION_ROOT"
echo "log_root=$LOG_ROOT"
echo "gpus=${GPUS[*]} dry_run=$DRY_RUN smoke=$SMOKE limit=${LIMIT:-all}"

"$PYTHON" -m src.generate_actmaps --help >/dev/null
check_inputs
run_small_model_phase
run_qwen32_phase
echo "Sprint 3 ActMap runner finished"
