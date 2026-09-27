# ActMap

**Single-Pass Uncertainty Quantification from Generation-Time Activation Maps**

ActMap estimates whether an LLM-generated answer is correct using the model's
hidden states captured during a single generation. It compresses the activation
trajectory into a fixed-size map, then uses a compact classifier to predict
answer correctness. Scoring a captured map requires no additional LLM calls.

This repository provides the ActMap pipeline: activation capture, correctness
labeling, balanced split construction, and detector training and evaluation.

## Method

1. **Capture.** Collect hidden states across transformer layers during decoding.
2. **Compress.** Summarize the trajectory with 12 temporal-statistic channels,
   including segment means, variability, and trends. Pool across depth and hidden
   coordinates to produce a `12 × 32 × 128` tensor: 96 KiB in float16.
3. **Score.** Train a compact Vision Transformer (ViT2D) on maps with binary
   correctness labels to estimate the probability that an answer is correct.

The detector reads activation maps without using answer text or token
probabilities as classifier inputs. Its score can support abstention, routing,
or selective verification.

The primary evaluation is **supervised and in-domain**, with a separate detector
for each model–dataset pair. Scores learned on balanced splits may need
recalibration when the correctness base rate changes; a shared map shape does
not imply transfer across models or tasks.

## Experimental scope

The primary experiments cover three instruction-tuned models and four datasets,
for twelve model–dataset configurations.

| Model | Role |
| --- | --- |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Primary experiments |
| [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Primary experiments; gated model access required |
| [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Primary experiments |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Optional scaling experiment |

| Dataset | Task | Configuration |
| --- | --- | --- |
| [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | Short-answer question answering | `rc.nocontext` |
| [NQ-Open](https://huggingface.co/datasets/google-research-datasets/nq_open) | Open-domain question answering | Default |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Direct-answer mathematics | `main`; final numeric answer only |
| [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) | Summarization factuality | `3.0.0`; labels from [MiniCheck](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large) |

The [experiment registry](replication/actmap_registry.json) records model and
dataset identifiers, source limits, label definitions, and split settings.
The runner retains the internal ID `gsm8k_rationale` for compatibility but uses
**direct-answer mode** for the primary GSM8K experiments.

## Installation

Use Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA environment compatible
with PyTorch and vLLM. Run these commands from the repository root:

```bash
uv sync --python 3.12
source .venv/bin/activate
```

For Llama, first obtain access to the model on Hugging Face, then authenticate
with `hf auth login` or set `HF_TOKEN` in your environment.

Verify the registry and command wiring without downloading datasets or model
weights, or using a GPU:

```bash
bash replication/run_actmap.sh smoke
```

## Reproduction

The runner defaults to the three primary models, all four datasets, one visible
GPU (`GPU_ID=0`), and detector seeds `42`, `123`, and `456`.

Run the stages in order:

```bash
bash replication/run_actmap.sh prepare
bash replication/run_actmap.sh generate
bash replication/run_actmap.sh label-cnndm
bash replication/run_actmap.sh balance
bash replication/run_actmap.sh train
```

| Stage | Purpose |
| --- | --- |
| `prepare` | Download source datasets and build deterministic, source-disjoint splits. |
| `generate` | Generate answers and capture normalized ActMaps. |
| `label-cnndm` | Label CNN/DailyMail summaries with MiniCheck; skipped when this dataset is not selected. |
| `balance` | Build train, validation, and test indexes with equal numbers of correct and incorrect examples. |
| `train` | Train the detector and report AUROC, AUPRC, and 10-bin expected calibration error (ECE). |

Use `bash replication/run_actmap.sh all` to run the complete sequence. For a
smaller experiment, select a subset with space-separated environment variables:

```bash
DATASETS="triviaqa_no_context nq_open" \
MODELS="Qwen/Qwen3-8B" \
GPU_ID=0 \
bash replication/run_actmap.sh all
```

Generation uses greedy decoding with Qwen thinking disabled, up to 32 new tokens
for question answering and direct-answer GSM8K, and up to 384 for CNN/DailyMail.
The default detector is a ViT2D with six transformer blocks, embedding width
192, and `4 × 16` patches, trained for up to 80 epochs with early stopping on
validation AUROC.

### Artifact storage

By default, outputs and Hugging Face caches are stored under `artifacts/`.
To use a separate data disk, set these paths before running:

```bash
export ACTMAP_ARTIFACT_ROOT=/path/to/actmap/artifacts
export HF_HOME=/path/to/huggingface/cache
```

Outputs include source records and splits, generation JSONL, activation tensors,
balanced indexes, detector checkpoints, per-seed predictions, and metric reports.
Generated artifacts and local paper sources are excluded from version control.

## Activation dataset

The [ActMap corpus on Hugging Face](https://huggingface.co/datasets/jacopopper/ActMap)
is **private until publication**. Access is required for the example below.

The corpus contains 476,372 successful captures from the twelve primary
model–dataset configurations, stored as `12 × 32 × 128` float16 maps with binary
correctness labels. The 317,212 rows marked `paper_balanced` identify the subset
used in the primary experiments.

```python
from datasets import load_dataset

dataset = load_dataset("jacopopper/ActMap", "qwen3-8b__triviaqa", token=True)
paper_train = dataset["train"].filter(lambda row: row["paper_balanced"])
```

Precomputed maps support detector experiments without rerunning LLM generation.
The replication runner above builds its own local artifacts from the source
datasets and does not require access to this private corpus.

## Code overview

| Path | Purpose |
| --- | --- |
| [`replication/run_actmap.sh`](replication/run_actmap.sh) | Entry point for the replication stages |
| [`replication/actmap_registry.json`](replication/actmap_registry.json) | Experiment configuration and resource identifiers |
| [`src/generate_actmaps.py`](src/generate_actmaps.py) | Generation, hidden-state capture, and map construction |
| [`src/data_split_labels.py`](src/data_split_labels.py) | Source splits and correctness labels |
| [`src/label_cnndm_minicheck.py`](src/label_cnndm_minicheck.py) | MiniCheck factuality labeling for summaries |
| [`src/balanced_indices.py`](src/balanced_indices.py) | Balanced indexes for question answering and mathematics |
| [`src/build_cnndm_minicheck_balanced_indices.py`](src/build_cnndm_minicheck_balanced_indices.py) | Balanced indexes for summarization |
| [`src/methods/actmap.py`](src/methods/actmap.py) | Detector architectures and training utilities |
| [`src/score_actmap_vit.py`](src/score_actmap_vit.py) | Detector training and evaluation entry point |
