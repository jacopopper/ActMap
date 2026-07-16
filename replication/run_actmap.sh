#!/usr/bin/env bash
# Public ActMap replication runner.
# Stages: prepare, generate, label-cnndm, balance, train, all.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
REGISTRY="${REGISTRY:-$ROOT/replication/actmap_registry.json}"
ARTIFACT_ROOT="${ACTMAP_ARTIFACT_ROOT:-$ROOT/artifacts}"
DATA_ROOT="${ACTMAP_DATA_ROOT:-$ARTIFACT_ROOT/data}"
GENERATION_ROOT="${ACTMAP_GENERATION_ROOT:-$ARTIFACT_ROOT/generations}"
ACTMAP_ROOT="${ACTMAP_FEATURE_ROOT:-$ARTIFACT_ROOT/features/actmap}"
DIRECT_GENERATION_ROOT="${ACTMAP_DIRECT_GENERATION_ROOT:-$ARTIFACT_ROOT/generations_gsm8k_direct}"
DIRECT_ACTMAP_ROOT="${ACTMAP_DIRECT_FEATURE_ROOT:-$ARTIFACT_ROOT/features/actmap_gsm8k_direct}"
BALANCED_ROOT="${ACTMAP_BALANCED_ROOT:-$ARTIFACT_ROOT/balanced_indices}"
DIRECT_BALANCED_ROOT="${ACTMAP_DIRECT_BALANCED_ROOT:-$ARTIFACT_ROOT/balanced_indices_gsm8k_direct}"
CHECKPOINT_ROOT="${ACTMAP_CHECKPOINT_ROOT:-$ARTIFACT_ROOT/checkpoints/actmap}"
PREDICTION_ROOT="${ACTMAP_PREDICTION_ROOT:-$ARTIFACT_ROOT/predictions/actmap}"
METRICS_ROOT="${ACTMAP_METRICS_ROOT:-$ARTIFACT_ROOT/results/actmap}"
REPORT_ROOT="${ACTMAP_REPORT_ROOT:-$ARTIFACT_ROOT/reports}"
FACTUALITY_ROOT="${ACTMAP_FACTUALITY_ROOT:-$ARTIFACT_ROOT/results/factuality/minicheck}"
GPU_ID="${GPU_ID:-0}"
DATASETS_STRING="${DATASETS:-triviaqa_no_context nq_open gsm8k_rationale cnn_dailymail_3_0_0}"
MODELS_STRING="${MODELS:-Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3}"
SEEDS_STRING="${SEEDS:-42 123 456}"
read -r -a DATASETS <<< "$DATASETS_STRING"
read -r -a MODELS <<< "$MODELS_STRING"
read -r -a SEEDS <<< "$SEEDS_STRING"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HOME="${HF_HOME:-$ARTIFACT_ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
stage="${1:-help}"
usage() {
  cat <<'EOF'
Usage: bash replication/run_actmap.sh <stage>

Stages:
  smoke         Validate the registry and command wiring without downloading data or using a GPU.
  prepare       Download source datasets and create deterministic source/split artifacts.
  generate      Generate hidden-state ActMaps for every selected model and dataset.
  label-cnndm   Label generated CNN/DailyMail summaries with MiniCheck Flan-T5 Large.
  balance       Build equal-correct/incorrect train, validation, and test indexes.
  train         Train and evaluate the compact ActMap ViT on the balanced indexes.
  all           Run all stages sequentially. The full run requires a GPU and a HF token.

Environment overrides:
  DATASETS, MODELS, SEEDS, GPU_ID, PYTHON, ACTMAP_ARTIFACT_ROOT, HF_HOME.
  Set MINICHECK_MODEL_PATH when MiniCheck is already downloaded locally.
EOF
}
run_smoke() {
  local smoke_root
  smoke_root="${TMPDIR:-/tmp}/actmap_smoke_${USER:-user}_$$"
  mkdir -p "$smoke_root"
  "$PYTHON" -m src.experiment_registry --registry "$REGISTRY"
  "$PYTHON" -m src.generate_actmaps --dry-run --registry "$REGISTRY" --dataset-id triviaqa_no_context --model Qwen/Qwen3-8B --data-root "$smoke_root/data" --output-root "$smoke_root/features" --generation-root "$smoke_root/generations"
  "$PYTHON" -m src.generate_actmaps --dry-run --registry "$REGISTRY" --dataset-id gsm8k_rationale --model Qwen/Qwen3-8B --gsm8k-output-mode direct --data-root "$smoke_root/data" --output-root "$smoke_root/features" --generation-root "$smoke_root/generations"
  "$PYTHON" -m src.score_actmap_vit --help >/dev/null
  echo "ActMap smoke test passed."
}

run_prepare() {
  mkdir -p "$DATA_ROOT" "$REPORT_ROOT"
  for dataset in "${DATASETS[@]}"; do
    limit=""
    case "$dataset" in
      triviaqa_no_context|nq_open|cnn_dailymail_3_0_0) limit="--limit 50000" ;;
    esac
    # shellcheck disable=SC2086
    "$PYTHON" -m src.data_split_labels build --dataset "$dataset" --registry "$REGISTRY" --output-root "$DATA_ROOT" --report "$REPORT_ROOT/data_split_label_report.md" --split-seed 42 $limit
  done
}
run_generate() {
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      args=(-m src.generate_actmaps --dataset-id "$dataset" --model "$model" --registry "$REGISTRY" --data-root "$DATA_ROOT" --temperature 0.0 --tensor-parallel-size 1 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" --batch-size 1 --actmap-dtype float16 --normalize-actmap --no-think --save-every 100)
      if [[ "$dataset" == gsm8k_rationale ]]; then
        args+=(--gsm8k-output-mode direct --max-new-tokens 32 --output-root "$DIRECT_ACTMAP_ROOT" --generation-root "$DIRECT_GENERATION_ROOT")
      else
        args+=(--output-root "$ACTMAP_ROOT" --generation-root "$GENERATION_ROOT")
        if [[ "$dataset" == cnn_dailymail_3_0_0 ]]; then args+=(--max-new-tokens 384); else args+=(--max-new-tokens 32); fi
      fi
      if [[ "${OVERWRITE:-0}" == 1 ]]; then args+=(--overwrite); fi
      "$PYTHON" "${args[@]}"
    done
  done
}
run_label_cnndm() {
  if [[ ! " ${DATASETS[*]} " =~ [[:space:]]cnn_dailymail_3_0_0[[:space:]] ]]; then
    echo "CNN/DailyMail not selected; skipping MiniCheck labeling."
    return 0
  fi
  model_args=()
  for model in "${MODELS[@]}"; do model_args+=(--model "$model"); done
  args=(-m src.label_cnndm_minicheck run-shard --data-root "$DATA_ROOT" --generation-root "$GENERATION_ROOT" --output-root "$FACTUALITY_ROOT" --report-path "$REPORT_ROOT/cnndm_factuality_report.md" --shard-index "${MINICHECK_SHARD_INDEX:-0}" --num-shards "${MINICHECK_NUM_SHARDS:-1}" --batch-size "${MINICHECK_BATCH_SIZE:-64}" --record-group-size "${MINICHECK_RECORD_GROUP_SIZE:-32}" --minicheck-model-id "${MINICHECK_MODEL_ID:-lytang/MiniCheck-Flan-T5-Large}" --max-model-len 2048 --dtype bfloat16)
  if [[ -n "${MINICHECK_MODEL_PATH:-}" ]]; then args+=(--minicheck-model-path "$MINICHECK_MODEL_PATH"); else args+=(--no-local-files-only); fi
  "$PYTHON" "${args[@]}" "${model_args[@]}"
  "$PYTHON" -m src.label_cnndm_minicheck merge --data-root "$DATA_ROOT" --generation-root "$GENERATION_ROOT" --output-root "$FACTUALITY_ROOT" --report-path "$REPORT_ROOT/cnndm_factuality_report.md" "${model_args[@]}"
}
run_balance() {
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      if [[ "$dataset" == cnn_dailymail_3_0_0 ]]; then
        "$PYTHON" -m src.build_cnndm_minicheck_balanced_indices --registry "$REGISTRY" --data-root "$DATA_ROOT" --label-path "$FACTUALITY_ROOT/../minicheck.parquet" --output-root "$BALANCED_ROOT" --report-path "$REPORT_ROOT/cnndm_minicheck_balanced_indices_report.md" --dataset "$dataset" --model "$model"
      else
        pair_gen_root="$GENERATION_ROOT"; pair_actmap_root="$ACTMAP_ROOT"; pair_index_root="$BALANCED_ROOT"
        if [[ "$dataset" == gsm8k_rationale ]]; then pair_gen_root="$DIRECT_GENERATION_ROOT"; pair_actmap_root="$DIRECT_ACTMAP_ROOT"; pair_index_root="$DIRECT_BALANCED_ROOT"; fi
        "$PYTHON" -m src.balanced_indices build --registry "$REGISTRY" --data-root "$DATA_ROOT" --generation-root "$pair_gen_root" --actmap-root "$pair_actmap_root" --output-root "$pair_index_root" --report-path "$REPORT_ROOT/balanced_indices_report.md" --dataset "$dataset" --model "$model"
      fi
    done
  done
}
run_train() {
  args=(-m src.score_actmap_vit --registry "$REGISTRY" --data-root "$DATA_ROOT" --actmap-root "$ACTMAP_ROOT" --balanced-index-root "$BALANCED_ROOT" --gsm8k-direct-actmap-root "$DIRECT_ACTMAP_ROOT" --gsm8k-direct-balanced-index-root "$DIRECT_BALANCED_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --prediction-root "$PREDICTION_ROOT" --metrics-root "$METRICS_ROOT" --report-path "$REPORT_ROOT/actmap_report.md" --arch vit2d --patch-h 4 --patch-w 16 --embed-dim 192 --num-heads 6 --num-layers 6 --mlp-ratio 3.0 --attn-drop 0.1 --drop-path-rate 0.05 --epochs 80 --batch-size 64 --lr 1e-3 --weight-decay 0.05 --dropout 0.3 --patience 20 --noise-std 0.08 --warmup-epochs 5 --mixup-alpha 0.2 --num-workers "${NUM_WORKERS:-4}" --ece-bins 10 --deterministic --require-cuda)
  for dataset in "${DATASETS[@]}"; do args+=(--dataset "$dataset"); done
  for model in "${MODELS[@]}"; do args+=(--model "$model"); done
  for seed in "${SEEDS[@]}"; do args+=(--seed "$seed"); done
  "$PYTHON" "${args[@]}"
}
case "$stage" in
  smoke) run_smoke ;;
  prepare) run_prepare ;;
  generate) run_generate ;;
  label-cnndm) run_label_cnndm ;;
  balance) run_balance ;;
  train) run_train ;;
  all) run_prepare; run_generate; run_label_cnndm; run_balance; run_train ;;
  help|-h|--help) usage ;;
  *) echo "Unknown stage: $stage" >&2; usage >&2; exit 2 ;;
esac
