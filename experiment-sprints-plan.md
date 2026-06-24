# Plan: Experiment Sprints

**Generated**: 2026-06-24
**Estimated Complexity**: High

## Overview

This plan turns `EXPERIMENTS_NEEDED.md` into independent coding-agent work units. The sprints are dependency-based, not time-based. Each sprint should leave behind runnable commands, machine-readable artifacts, and a short status report so another agent can continue without guessing.

The main rule: every stage must write explicit outputs under `artifacts/` and a short report saying what succeeded, what failed, and what remains blocked.

## Global Prerequisites

- `EXPERIMENTS_NEEDED.md` is the source of truth for datasets, models, methods, metrics, ablations, and reporting.
- All generated artifacts must be reproducible from code, not manual edits.
- Every dataset is split into train, validation, and test.
- Methods that do not train use only the test split.
- ActMap maps are normalized to mean 0 and standard deviation 1.
- Detector seeds are `42`, `123`, and `456`.
- Metrics are AUROC, AUPRC with prevalence baseline, and 10-bin ECE.

## Sprint 1: Experiment Registry

**Goal**: Create one explicit machine-readable registry for datasets, models, methods, splits, metrics, and the 20 dataset/model pairs.

**Prerequisites**:
- `EXPERIMENTS_NEEDED.md`.

**Deliverables**:
- `configs/experiments/experiment_registry.json`
- `artifacts/orchestration/experiment_matrix.json`
- `artifacts/orchestration/sprint_1_registry_report.md`

**Tasks**:

1. Define dataset entries for TriviaQA no-context, NQ-Open, WebQuestions, GSM8K rationale, and CNN/DailyMail 3.0.0.
2. Define model entries for Qwen3-8B, Qwen3-32B, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3.
3. Define method entries grouped as ActMap, black-box, grey-box, and white-box.
4. Generate the 20 dataset/model pairs from the registry.
5. Add a validation command that fails if the registry does not match `EXPERIMENTS_NEEDED.md`.

**Validation**:
- A command prints exactly 20 dataset/model pairs.
- A command prints every listed method once.
- Unit test verifies no removed GSM8K direct pair exists.

## Sprint 2: Data Splits And Labels

**Goal**: Build train/validation/test splits and labels for every dataset.

**Prerequisites**:
- Sprint 1 registry.
- Dataset download access.

**Deliverables**:
- `artifacts/data/{dataset}/source_records.parquet`
- `artifacts/data/{dataset}/splits.parquet`
- `artifacts/data/{dataset}/labels.parquet`
- `artifacts/reports/data_split_label_report.md`

**Tasks**:

1. Implement source record builders for all five datasets.
2. Split every dataset into train, validation, and test.
3. Ensure the same source question never appears in more than one split.
4. Implement correctness labels for TriviaQA, NQ-Open, WebQuestions, and GSM8K rationale.
5. Implement CNN/DailyMail factuality-label placeholders for MiniCheck and AlignScore outputs.
6. Add row-count and label-prevalence reports.

**Validation**:
- Tests prove no train/validation/test leakage.
- Tests prove black-box methods can select test rows only.
- Reports show row counts and label prevalence per dataset.

## Sprint 3: Generation And Feature Capture

**Goal**: Generate model outputs and capture the features required by all methods.

**Prerequisites**:
- Sprint 1 registry.
- Sprint 2 splits.
- GPU environment for model inference.

**Deliverables**:
- `artifacts/generations/{dataset}/{model}/records.parquet`
- `artifacts/features/actmap/{dataset}/{model}/`
- `artifacts/features/grey_box/{dataset}/{model}/`
- `artifacts/features/white_box/{dataset}/{model}/`
- `artifacts/features/black_box_samples/{dataset}/{model}/`
- `artifacts/reports/generation_feature_report.md`

**Tasks**:

1. Implement canonical generation for every dataset/model pair.
2. Capture generation-time hidden states for ActMap.
3. Build ActMap tensors and normalize them to mean 0 and standard deviation 1.
4. Capture grey-box quantities needed for perplexity, MTE, and `P(True)`.
5. Capture white-box quantities needed for Factoscope, TAD, RAUQ, HalluGuard/HARP, and EigenScore/INSIDE.
6. Generate sampled responses needed by Semantic Entropy, LUQ, and EigenScore/INSIDE.
7. Record runtime, peak GPU memory, extra tokens, and extra forward passes for every method family.

**Validation**:
- Every dataset/model pair has generation records.
- Every valid generation has feature references required by each applicable method.
- ActMap tensors have expected shape, finite values, mean near 0, and standard deviation near 1.
- Report lists missing or failed rows explicitly.

## Sprint 4: Method Implementations

**Goal**: Implement all listed methods behind one prediction interface.

**Prerequisites**:
- Sprint 3 feature artifacts.

**Deliverables**:
- `src/methods/actmap.py`
- `src/methods/black_box.py`
- `src/methods/grey_box.py`
- `src/methods/white_box.py`
- `artifacts/predictions/{method}/{dataset}/{model}/predictions.parquet`
- `artifacts/reports/method_implementation_report.md`

**Tasks**:

1. Implement ActMap training and prediction for seeds `42`, `123`, and `456`.
2. Implement black-box methods: Semantic Entropy and LUQ.
3. Implement grey-box methods: perplexity, MTE, and `P(True)`.
4. Implement white-box methods: Factoscope, TAD, RAUQ, HalluGuard or HARP fallback, and EigenScore/INSIDE.
5. Standardize every method output to one prediction schema.
6. Make unsupported methods fail explicitly with a reason, not silently disappear.

**Validation**:
- Every method writes predictions for every applicable dataset/model pair.
- Every prediction row joins to a valid generation row.
- Prediction files include method name, dataset, model, split, score, label, and seed where applicable.
- Unsupported HalluGuard must trigger the HARP fallback path.

## Sprint 5: Metrics And Leaderboards

**Goal**: Compute metrics and produce the main result tables.

**Prerequisites**:
- Sprint 4 predictions.

**Deliverables**:
- `artifacts/results/metrics/by_pair.parquet`
- `artifacts/results/metrics/by_dataset.parquet`
- `artifacts/results/metrics/by_model.parquet`
- `artifacts/results/metrics/qwen32_only.parquet`
- `artifacts/tables/single_generation_leaderboard.md`
- `artifacts/tables/sampling_allowed_leaderboard.md`
- `artifacts/reports/metrics_report.md`

**Tasks**:

1. Implement AUROC, AUPRC, prevalence baseline, and 10-bin ECE.
2. Compute metrics for every dataset/model/method.
3. Compute averages by dataset, by model, and overall.
4. Produce a separate Qwen3-32B result table.
5. Separate single-generation and sampling-allowed leaderboards.
6. Add checks that every expected pair has labels, features, predictions, and metrics.

**Validation**:
- Metrics command fails on missing predictions unless the missing method is explicitly unsupported.
- Leaderboards include ActMap and all applicable baselines.
- Reports list row counts used for every metric.

## Sprint 6: Transfer Tests

**Goal**: Run the required transfer experiments.

**Prerequisites**:
- Sprint 4 trainable methods.
- Sprint 5 metric code.

**Deliverables**:
- `artifacts/results/transfer/cross_dataset.parquet`
- `artifacts/results/transfer/cross_model.parquet`
- `artifacts/results/transfer/short_long.parquet`
- `artifacts/figures/transfer_results.*`
- `artifacts/reports/transfer_report.md`

**Tasks**:

1. Run cross-dataset transfer for each supervised method.
2. Run Qwen3-8B to Qwen3-32B transfer.
3. Run Qwen3-32B to Qwen3-8B transfer.
4. Run all 7B/8B models to Qwen3-32B transfer.
5. Run leave-one-model-out transfer.
6. Run same-question model-only diagnostic and mark it diagnostic only.
7. Run short-to-long and long-to-short transfer as supplement-only results.

**Validation**:
- Transfer reports prove no train/test leakage.
- Transfer metrics use the same metric code as Sprint 5.
- Diagnostic-only results are clearly labeled.

## Sprint 7: Controls And Ablations

**Goal**: Run sanity controls and the reduced ablation set.

**Prerequisites**:
- Sprint 4 methods.
- Sprint 5 metric code.

**Deliverables**:
- `artifacts/results/controls/shortcut_controls.parquet`
- `artifacts/results/ablations/triviaqa_qwen8b.parquet`
- `artifacts/figures/single_statistic_histogram.*`
- `artifacts/reports/controls_ablations_report.md`

**Tasks**:

1. Check that ActMap is not just learning answer length.
2. Check that ActMap is not just learning model identity.
3. Run split-leakage checks.
4. Run random-label sanity check.
5. Run ablations only on TriviaQA no-context x Qwen/Qwen3-8B.
6. Compare ActMap CV classifiers.
7. Compare best single ActMap model against an ensemble.
8. Compare ActMap + ViT, ActMap + logistic regression, and ActMap + MLP.
9. Produce AUROC histogram for each individual ActMap statistic.
10. Compare normal hidden layout against shuffled hidden dimensions and Gaussian projection.
11. Compare main map resolution against one smaller and one larger map.
12. Report output-length slices.

**Validation**:
- Controls report says whether answer length or model identity explains ActMap.
- Random-label sanity check is near chance.
- Ablations run only on the selected test pair.

## Sprint 8: CNN/DailyMail Factuality Evaluation

**Goal**: Evaluate summarization factuality and connect it to method metrics.

**Prerequisites**:
- CNN/DailyMail generations from Sprint 3.
- Sprint 5 metric code.

**Deliverables**:
- `artifacts/results/factuality/minicheck.parquet`
- `artifacts/results/factuality/alignscore.parquet`
- `artifacts/results/factuality/human_labels.parquet` if available
- `artifacts/reports/cnndm_factuality_report.md`

**Tasks**:

1. Run MiniCheck on CNN/DailyMail generations.
2. Run AlignScore on CNN/DailyMail generations.
3. Ingest human labels only if available.
4. Compute method metrics against MiniCheck and AlignScore labels.
5. Compute human-label metrics only when real human labels exist.

**Validation**:
- CNN/DailyMail report states which factuality labels are present.
- Human-label results are omitted, not faked, when labels are unavailable.

## Sprint 9: Efficiency And Final Reporting

**Goal**: Produce final tables, figures, and an autonomous completion report.

**Prerequisites**:
- Sprints 5 through 8.

**Deliverables**:
- `artifacts/results/efficiency/by_method.parquet`
- `artifacts/tables/final_results.md`
- `artifacts/figures/final_results_dashboard.*`
- `artifacts/figures/efficiency_summary.*`
- `artifacts/reports/final_experiment_report.md`

**Tasks**:

1. Aggregate runtime per method.
2. Aggregate peak GPU memory per method.
3. Aggregate extra generated tokens and extra forward passes per method.
4. Produce final tables for all methods, datasets, and models.
5. Produce final figures for main results, transfer, ablations, factuality, and efficiency.
6. Write a final report with completed work, failed methods, missing artifacts, and exact commands to rerun.

**Validation**:
- Final report links every table and figure to machine-readable inputs.
- Any missing method or dataset/model pair has an explicit reason.
- A fresh agent can rerun the final aggregation command from the report.

## Testing Strategy

- Unit tests for split integrity, label parsing, metric calculations, feature shape checks, and prediction schema.
- Integration tests for one small dataset/model pair before full matrix execution.
- Smoke run for each method family before full method execution.
- End-to-end check that every method prediction joins to a generation row and a label row.
- Final completeness check against `EXPERIMENTS_NEEDED.md`.

## Potential Risks And Mitigations

- **GPU memory failures**: run Qwen3-32B separately and record failed attempts explicitly.
- **Unavailable official method code**: use documented fallback and mark the original method unsupported.
- **Silent row loss**: every stage must report expected rows, produced rows, failed rows, and reasons.
- **Data leakage**: split checks must run before training and before transfer metrics.
- **Metric inconsistency**: all metrics must use one shared implementation.
- **CNN/DailyMail human labels missing**: omit human results and state that automatic factuality labels were used.

## Rollback Plan

- Keep generated artifacts under versioned `artifacts/` subdirectories.
- Do not overwrite previous predictions or metrics without writing a new run ID.
- If a sprint fails, preserve its report and rerun only that sprint after fixing the blocker.
