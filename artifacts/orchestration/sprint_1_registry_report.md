# Sprint 1 Registry Report

Status: succeeded

## Outputs

- `configs/experiments/experiment_registry.json`
- `artifacts/orchestration/experiment_matrix.json`
- `artifacts/orchestration/sprint_1_registry_report.md`

## Summary

- Datasets: 5
- Models: 4
- Dataset/model pairs: 20
- Methods: 12
- Metrics: 3

## Validation Commands

```bash
python3 -m src.experiment_registry validate
python3 -m src.experiment_registry list-pairs
python3 -m src.experiment_registry list-methods
python3 -m unittest discover
```

## Validation Result

- Registry matches `EXPERIMENTS_NEEDED.md` for dataset/model pairs and methods.
- The generated matrix contains exactly 20 dataset/model pairs.
- The registry contains train, validation, and test splits.
- The removed GSM8K direct dataset/pair is absent; only GSM8K rationale is registered.

## Remaining Blockers

- None for Sprint 1.
