# Experiments Needed

## Fixed Setup

- Split each dataset into train, validation, and test. Methods that do not need training, such as black-box methods, use only the test set.
- Detector seeds: `42`, `123`, `456`.
- Primary ActMap: activation maps built from generation-time hidden states, classified with a compact ViT2D.
- ActMap normalization: normalize maps to mean 0 and standard deviation 1.
- Calibration: use the best temperature per model.
- Metrics: AUROC, AUPRC with prevalence baseline, and 10-bin ECE.

## Dataset x Model Matrix

Run the full per-pair experiment suite for each of these 20 pairs.

- TriviaQA no-context x `Qwen/Qwen3-8B`
- TriviaQA no-context x `Qwen/Qwen3-32B`
- TriviaQA no-context x `meta-llama/Llama-3.1-8B-Instruct`
- TriviaQA no-context x `mistralai/Mistral-7B-Instruct-v0.3`

- NQ-Open x `Qwen/Qwen3-8B`
- NQ-Open x `Qwen/Qwen3-32B`
- NQ-Open x `meta-llama/Llama-3.1-8B-Instruct`
- NQ-Open x `mistralai/Mistral-7B-Instruct-v0.3`

- WebQuestions x `Qwen/Qwen3-8B`
- WebQuestions x `Qwen/Qwen3-32B`
- WebQuestions x `meta-llama/Llama-3.1-8B-Instruct`
- WebQuestions x `mistralai/Mistral-7B-Instruct-v0.3`

- GSM8K rationale x `Qwen/Qwen3-8B`
- GSM8K rationale x `Qwen/Qwen3-32B`
- GSM8K rationale x `meta-llama/Llama-3.1-8B-Instruct`
- GSM8K rationale x `mistralai/Mistral-7B-Instruct-v0.3`

- CNN/DailyMail 3.0.0 x `Qwen/Qwen3-8B`
- CNN/DailyMail 3.0.0 x `Qwen/Qwen3-32B`
- CNN/DailyMail 3.0.0 x `meta-llama/Llama-3.1-8B-Instruct`
- CNN/DailyMail 3.0.0 x `mistralai/Mistral-7B-Instruct-v0.3`

## Methods

- Main method:
  - ActMap.
- Black-box methods:
  - [Semantic Entropy](https://arxiv.org/abs/2302.09664);
  - [LUQ](https://arxiv.org/abs/2403.20279) for CNN/DailyMail.
- Grey-box methods:
  - perplexity;
  - MTE;
  - [`P(True)`](https://arxiv.org/abs/2207.05221).
- White-box methods:
  - [Factoscope](https://arxiv.org/abs/2312.16374);
  - [TAD](https://arxiv.org/abs/2408.10692);
  - [RAUQ](https://arxiv.org/abs/2505.20045);
  - [HalluGuard](https://arxiv.org/abs/2601.18753), or [HARP](https://arxiv.org/abs/2509.11536) fallback if HalluGuard is unavailable or incompatible;
  - [EigenScore/INSIDE](https://arxiv.org/abs/2402.03744).

## Per-Pair Suite

Run all listed methods for every dataset/task x model pair above.

## Global Tests

- Check that every dataset/model pair has the expected number of generations, labels, features, and predictions.
- Evaluate CNN/DailyMail factuality with [MiniCheck](https://arxiv.org/abs/2404.10774) and [AlignScore](https://arxiv.org/abs/2305.16739).
- Add human CNN/DailyMail factuality labels only if they are available.
- Report the single-generation leaderboard by dataset and model.
- Report the sampling-allowed leaderboard by dataset and model.
- Report averages by dataset, by model, and overall.
- Report Qwen3-32B results separately.

## Transfer Tests

- Cross-dataset transfer for each supervised method: source train/dev/calibration only, target test only.
- Strict cross-model transfer:
  - `Qwen/Qwen3-8B` to `Qwen/Qwen3-32B`;
  - `Qwen/Qwen3-32B` to `Qwen/Qwen3-8B`;
  - all 7B/8B models to `Qwen/Qwen3-32B`;
  - leave each model out in turn.
- Same-question model-only transfer diagnostic, supplement only.
- Short-to-long transfer: short-answer detector to CNN/DailyMail factuality, supplement only.
- Long-to-short transfer: CNN/DailyMail factuality detector to short-answer correctness, supplement only.

## Shortcut And Leakage Controls

- Check that ActMap is not just learning answer length.
- Check that ActMap is not just learning model identity.
- Check that train, validation, and test examples do not leak across splits.
- Run a random-label sanity check.
- Report results separately by dataset and model.

## Ablations

Run ablations only on the selected test pair: TriviaQA no-context x `Qwen/Qwen3-8B`.

- CV model choice: compare compact ViT2D against other image/CV classifiers for ActMap.
- Ensemble: combine the best ActMap models and compare against the best single model.
- Classifier capacity: ActMap + ViT, ActMap + logistic regression, ActMap + MLP.
- Single-statistic histogram: show AUROC for each individual ActMap statistic.
- Hidden-axis assumption: contiguous pooling versus hidden-coordinate permutation and Gaussian projection.
- Resolution sanity: main `12 x 32 x 128` map versus one smaller map and one larger map.
- Output-length slices: short, medium, and long generations.

## Efficiency And Reporting

- Report runtime per method.
- Report peak GPU memory per method.
- Report extra generated tokens or extra forward passes per method.
- Produce final tables and figures.
