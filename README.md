# ActMap

This repository contains the reproducible implementation of **ActMap**, a
single-pass uncertainty detector built from generation-time hidden-state
activation maps. The public replication path is deliberately limited to ActMap
feature generation, balanced split construction, and ActMap detector training.
Generated activations, model checkpoints, predictions, caches, logs, and paper
artifacts are not part of the repository.

## Replication Scope

The primary paper results use three instruction-tuned models on four datasets:

| Resource | Hugging Face link | Notes |
|---|---|---|
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Primary model |
| Llama 3.1 8B Instruct | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Gated repository; access approval and `HF_TOKEN` are required |
| Mistral 7B Instruct v0.3 | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Primary model |
| Qwen3-32B | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Optional scaling experiment |
| TriviaQA | [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | `rc.nocontext` configuration |
| NQ-Open | [google-research-datasets/nq_open](https://huggingface.co/datasets/google-research-datasets/nq_open) | Open-domain QA |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `main` configuration; direct numeric-answer mode |
| CNN/DailyMail | [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) | `3.0.0` configuration |
| MiniCheck | [lytang/MiniCheck-Flan-T5-Large](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large) | CNN/DailyMail factuality labels |

The exact identifiers, configurations, source limits, split seed, and model
URLs are recorded in [`replication/actmap_registry.json`](replication/actmap_registry.json).

## Setup

Use Python 3.12 and a CUDA environment compatible with the installed PyTorch
and vLLM versions. The Llama repository requires an approved Hugging Face
account and token.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
hf auth login
```

For a separate data disk, set the artifact and Hugging Face cache locations
before running:

```bash
export ACTMAP_ARTIFACT_ROOT=/data_disk/$USER/ActMap/artifacts
export HF_HOME=/data_disk/$USER/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

## Reproduce ActMap

First verify the installation and command wiring. This smoke test does not
use a GPU or download model weights or datasets.

```bash
bash replication/run_actmap.sh smoke
```

The full runner uses one visible GPU by default and is restartable because
every stage writes checkpoints and row-level artifacts.

```bash
bash replication/run_actmap.sh prepare
bash replication/run_actmap.sh generate
bash replication/run_actmap.sh label-cnndm
bash replication/run_actmap.sh balance
bash replication/run_actmap.sh train
```

`generate` uses greedy decoding with thinking disabled, 32 new tokens for QA
and direct GSM8K, and 384 new tokens for CNN/DailyMail. The default detector
uses the paper configuration: a `12 x 32 x 128` ActMap, compact ViT2D, three
seeds (`42 123 456`), balanced train/validation/test splits, and 10-bin ECE.

To run only a subset, override the space-separated variables:

```bash
DATASETS="triviaqa_no_context nq_open" \
MODELS="Qwen/Qwen3-8B" \
GPU_ID=0 \
bash replication/run_actmap.sh all
```

For CNN/DailyMail, `label-cnndm` runs MiniCheck before `balance`. It can be
sharded across GPUs by setting `MINICHECK_SHARD_INDEX` and
`MINICHECK_NUM_SHARDS`, then merging once after all shards finish.

## Outputs

The runner writes only under `ACTMAP_ARTIFACT_ROOT`:

- source records and deterministic source-disjoint splits;
- generation JSONL and normalized `12 x 32 x 128` ActMaps;
- balanced correctness indexes;
- per-seed ActMap checkpoints and predictions;
- AUROC, AUPRC, and 10-bin ECE reports.

No repository commit or push is performed by the replication commands.
