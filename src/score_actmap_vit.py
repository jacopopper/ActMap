from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .experiment_registry import DEFAULT_REGISTRY_PATH, REPO_ROOT, read_json
from .generate_actmaps import load_parquet_rows, model_slug
from .methods import actmap


DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts"
DEFAULT_DATA_ROOT = DEFAULT_ARTIFACT_ROOT / "data"
DEFAULT_ACTMAP_ROOT = DEFAULT_ARTIFACT_ROOT / "features" / "actmap"
DEFAULT_BALANCED_INDEX_ROOT = DEFAULT_ARTIFACT_ROOT / "balanced_indices"
DEFAULT_PREDICTION_ROOT = DEFAULT_ARTIFACT_ROOT / "predictions"
DEFAULT_CHECKPOINT_ROOT = DEFAULT_ARTIFACT_ROOT / "checkpoints" / "actmap"
DEFAULT_METRICS_ROOT = DEFAULT_ARTIFACT_ROOT / "results" / "actmap"
DEFAULT_REPORT_PATH = DEFAULT_ARTIFACT_ROOT / "reports" / "actmap_metrics_report.md"
DEFAULT_DIRECT_GSM8K_ACTMAP_ROOT = DEFAULT_ARTIFACT_ROOT / "features" / "actmap_gsm8k_direct"
DEFAULT_DIRECT_GSM8K_BALANCED_INDEX_ROOT = DEFAULT_ARTIFACT_ROOT / "balanced_indices_gsm8k_direct"
DEFAULT_DATASETS = (
    "triviaqa_no_context",
    "nq_open",
    "gsm8k_rationale",
    "cnn_dailymail_3_0_0",
)
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_SMALL_MODELS = (
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
METHOD = actmap.ACTMAP_METHOD


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _print(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prediction_path(prediction_root: Path, method: str, dataset_id: str, model: str) -> Path:
    return prediction_root / method / dataset_id / model_slug(model) / "predictions.parquet"


def metrics_path(metrics_root: Path, method: str, dataset_id: str, model: str) -> Path:
    return metrics_root / method / dataset_id / model_slug(model) / "metrics.json"


def save_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def collect_metrics(metrics_root: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted((metrics_root / METHOD).glob("*/*/metrics.json")):
        try:
            payloads.append(read_json(path))
        except Exception as exc:
            _print(f"warning: could not read metrics {path}: {exc}")
    return payloads


def write_aggregate_outputs(metrics_root: Path, report_path: Path) -> None:
    payloads = collect_metrics(metrics_root)
    if payloads:
        import pandas as pd

        frame = pd.DataFrame(payloads)
        frame.to_parquet(metrics_root / "actmap_metrics.parquet", index=False)
        frame.to_csv(metrics_root / "actmap_metrics.csv", index=False)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ActMap Metrics",
        "",
        "Metrics use the shared balanced test rows for each dataset and generator.",
        "",
        "| dataset | model | seeds | n | AUROC | AUPRC | ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(value: Any) -> str:
        return "" if value is None else f"{float(value):.4f}"

    for item in sorted(payloads, key=lambda value: (value["dataset_id"], value["model"])):
        lines.append(
            "| {dataset} | {model} | {seeds} | {n} | {auroc} | {auprc} | {ece} |".format(
                dataset=item["dataset_id"],
                model=item["model"],
                seeds=item.get("seed_count", ""),
                n=item.get("n", ""),
                auroc=fmt(item.get("auroc")),
                auprc=fmt(item.get("auprc")),
                ece=fmt(item.get("ece")),
            )
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    registry = read_json(args.registry)
    dataset_ids = {dataset["id"] for dataset in registry["datasets"]}
    model_ids = {model["hf_id"] for model in registry["models"]}
    unknown_datasets = set(args.datasets) - dataset_ids
    unknown_models = set(args.models) - model_ids
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {sorted(unknown_datasets)}")
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.limit_per_split is not None and args.limit_per_split < 2:
        raise ValueError("--limit-per-split must be >= 2")
    if not args.seeds:
        raise ValueError("at least one --seed is required")


def dataset_roots(args: argparse.Namespace, dataset_id: str) -> tuple[Path, Path, str]:
    if dataset_id == "gsm8k_rationale" and args.use_gsm8k_direct_roots:
        return (
            args.gsm8k_direct_actmap_root,
            args.gsm8k_direct_balanced_index_root,
            "gsm8k_direct",
        )
    return args.actmap_root, args.balanced_index_root, "main"


def balanced_index_path(index_root: Path, dataset_id: str, model: str, split: str) -> Path:
    return index_root / dataset_id / model_slug(model) / f"{split}.parquet"


def pair_dir(root: Path, dataset_id: str, model: str) -> Path:
    return root / dataset_id / model_slug(model)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing ActMap manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def label_value(row: dict[str, Any]) -> int:
    value = row.get("label", row.get("is_correct"))
    if value is None:
        raise ValueError(f"balanced row is missing label/is_correct: {row}")
    return int(bool(value))


def maybe_limit_balanced_rows(rows: list[dict[str, Any]], limit: int | None, *, seed: int) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    per_label = max(1, limit // 2)
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[label_value(row)].append(row)
    take = min(per_label, len(by_label[0]), len(by_label[1]))
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for label in (0, 1):
        indices = np.arange(len(by_label[label]))
        rng.shuffle(indices)
        selected.extend(by_label[label][int(idx)] for idx in indices[:take])
    selected.sort(key=lambda row: (str(row.get("split") or ""), int(row.get("selection_rank", 0)), str(row.get("source_record_id") or "")))
    return selected


def load_records(records_path: Path) -> list[dict[str, Any]]:
    if not records_path.exists():
        raise FileNotFoundError(f"missing ActMap records: {records_path}")
    rows = torch.load(records_path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(rows, list):
        raise TypeError(f"expected {records_path} to contain a list, got {type(rows).__name__}")
    return rows


def check_tensor(row: dict[str, Any], *, source: str) -> None:
    tensor = row.get("actmap")
    if not hasattr(tensor, "shape"):
        raise TypeError(f"{source}: missing tensor-like actmap")
    if tuple(tensor.shape) != actmap.EXPECTED_ACTMAP_SHAPE:
        raise ValueError(f"{source}: bad ActMap shape {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{source}: non-finite ActMap tensor")


def materialize_split(
    *,
    record_map: dict[str, dict[str, Any]],
    index_rows: list[dict[str, Any]],
    dataset_id: str,
    model: str,
    split: str,
    validate_tensors: bool,
) -> list[actmap.ActMapRecord]:
    materialized: list[actmap.ActMapRecord] = []
    missing = 0
    wrong_split = 0
    for index_row in index_rows:
        source_record_id = str(index_row.get("source_record_id") or "")
        record = record_map.get(source_record_id)
        if record is None:
            missing += 1
            continue
        row_split = str(record.get("split") or index_row.get("split") or "")
        if row_split != split:
            wrong_split += 1
        if validate_tensors:
            check_tensor(record, source=f"{dataset_id}/{model}/{source_record_id}")
        materialized.append(
            actmap.ActMapRecord(
                source_record_id=source_record_id,
                dataset_id=dataset_id,
                model=model,
                split=split,
                actmap=record["actmap"],
                label=label_value(index_row),
                generated_token_count=(
                    int(index_row["generated_token_count"])
                    if index_row.get("generated_token_count") is not None
                    else (
                        int(record["generated_token_count"])
                        if record.get("generated_token_count") is not None
                        else None
                    )
                ),
                generation=str(record.get("generation") or ""),
            )
        )
    if missing:
        raise RuntimeError(f"{dataset_id} {model} {split}: {missing} balanced ids missing from records.pt")
    if wrong_split:
        raise RuntimeError(f"{dataset_id} {model} {split}: {wrong_split} rows have wrong split in records.pt")
    return materialized


def split_rows(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    dataset_id: str,
    model: str,
    index_root: Path,
) -> tuple[list[actmap.ActMapRecord], list[actmap.ActMapRecord], list[actmap.ActMapRecord], dict[str, Any]]:
    record_map: dict[str, dict[str, Any]] = {}
    duplicate_ids = 0
    for record in records:
        source_record_id = str(record.get("source_record_id") or "")
        if source_record_id in record_map:
            duplicate_ids += 1
        record_map[source_record_id] = record
    if duplicate_ids:
        raise RuntimeError(f"{dataset_id} {model}: duplicate source_record_id count={duplicate_ids}")

    result: dict[str, list[actmap.ActMapRecord]] = {}
    index_counts: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        path = balanced_index_path(index_root, dataset_id, model, split)
        if not path.exists():
            raise FileNotFoundError(f"missing balanced index: {path}")
        rows = load_parquet_rows(path)
        rows = maybe_limit_balanced_rows(rows, args.limit_per_split, seed=args.seed_for_limit)
        materialized = materialize_split(
            record_map=record_map,
            index_rows=rows,
            dataset_id=dataset_id,
            model=model,
            split=split,
            validate_tensors=args.validate_tensors,
        )
        labels = [row.label for row in materialized]
        index_counts[split] = {
            "rows": len(materialized),
            "n_positive": int(sum(labels)),
            "n_negative": int(len(labels) - sum(labels)),
            "path": str(path),
        }
        if index_counts[split]["n_positive"] != index_counts[split]["n_negative"]:
            raise RuntimeError(f"{dataset_id} {model} {split}: split is not balanced: {index_counts[split]}")
        result[split] = materialized

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = {row.source_record_id for row in result[left]} & {row.source_record_id for row in result[right]}
        if overlap:
            raise RuntimeError(f"{dataset_id} {model}: {left}/{right} overlap={len(overlap)}")
    return result["train"], result["validation"], result["test"], index_counts


def seed_prediction_dir(prediction_root: Path, dataset_id: str, model: str, seed: int) -> Path:
    return prediction_root / METHOD / dataset_id / model_slug(model) / f"seed_{seed}"


def seed_prediction_path(prediction_root: Path, dataset_id: str, model: str, seed: int) -> Path:
    return seed_prediction_dir(prediction_root, dataset_id, model, seed) / "predictions.parquet"


def seed_metrics_path(metrics_root: Path, dataset_id: str, model: str, seed: int) -> Path:
    return metrics_root / METHOD / dataset_id / model_slug(model) / f"metrics_seed_{seed}.json"


def checkpoint_dir(checkpoint_root: Path, dataset_id: str, model: str, seed: int) -> Path:
    return checkpoint_root / dataset_id / model_slug(model) / f"seed_{seed}"


def float_values(seed_metrics: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for item in seed_metrics:
        value = item.get(key)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return values


def mean_payload(seed_metrics: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = float_values(seed_metrics, key)
    if not values:
        return {key: None, f"{key}_std": None, f"{key}_ci95": None}
    arr = np.array(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {key: float(arr.mean()), f"{key}_std": std, f"{key}_ci95": ci}


def aggregate_prediction_rows(seed_results: list[actmap.TrainResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in seed_results:
        for row in result.prediction_rows:
            grouped.setdefault(str(row["source_record_id"]), []).append(row)
    output = []
    for source_record_id, rows in sorted(grouped.items()):
        scores = np.array([float(row["score"]) for row in rows], dtype=np.float64)
        probabilities = np.array([float(row["probability"]) for row in rows], dtype=np.float64)
        first = rows[0]
        output.append(
            {
                "method": METHOD,
                "method_group": "actmap",
                "dataset_id": first["dataset_id"],
                "model": first["model"],
                "source_record_id": source_record_id,
                "split": first["split"],
                "label": int(first["label"]),
                "score": float(scores.mean()),
                "probability": float(probabilities.mean()),
                "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "probability_std": float(probabilities.std(ddof=1)) if len(probabilities) > 1 else 0.0,
                "seed_count": int(len(rows)),
                "seeds": json.dumps([int(row["seed"]) for row in rows]),
                "balance_mode": first["balance_mode"],
                "generated_token_count": first.get("generated_token_count"),
            }
        )
    return output


def save_pair_aggregate(
    *,
    args: argparse.Namespace,
    dataset_id: str,
    model: str,
    seed_results: list[actmap.TrainResult],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    seed_metrics = [result.metrics for result in seed_results]
    aggregate_rows = aggregate_prediction_rows(seed_results)
    pred_path = prediction_path(args.prediction_root, METHOD, dataset_id, model)
    save_prediction_rows(pred_path, aggregate_rows)

    metrics = {
        "method": METHOD,
        "method_group": "actmap",
        "dataset_id": dataset_id,
        "model": model,
        "model_slug": model_slug(model),
        "balance_mode": args.balance_mode,
        "n": int(seed_metrics[0].get("n_test", len(aggregate_rows))) if seed_metrics else len(aggregate_rows),
        "n_positive": int(seed_metrics[0].get("n_positive", 0)) if seed_metrics else 0,
        "n_negative": int(seed_metrics[0].get("n_negative", 0)) if seed_metrics else 0,
        "seed_count": len(seed_results),
        "seeds": [result.seed for result in seed_results],
        "prediction_path": str(pred_path),
        "metrics_path": str(metrics_path(args.metrics_root, METHOD, dataset_id, model)),
        "checkpoint_root": str(args.checkpoint_root / dataset_id / model_slug(model)),
        "probability_note": "ActMap probability is sigmoid(logit). Main metrics are mean across seeds; *_ci95 is 1.96*std/sqrt(n_seeds).",
        "metadata": metadata,
        "seed_metrics": seed_metrics,
    }
    for key in ("auroc", "auprc", "ece", "test_accuracy", "accuracy_rate", "best_val_auroc"):
        metrics.update(mean_payload(seed_metrics, key))
    write_json(metrics_path(args.metrics_root, METHOD, dataset_id, model), metrics)
    return metrics


def run_pair(args: argparse.Namespace, dataset_id: str, model: str) -> dict[str, Any] | None:
    actmap_root, index_root, root_note = dataset_roots(args, dataset_id)
    pdir = pair_dir(actmap_root, dataset_id, model)
    manifest_path = pdir / "manifest.json"
    records_path = pdir / "records.pt"
    if not manifest_path.exists() or not records_path.exists():
        message = f"missing ActMap artifacts for {dataset_id} {model}: {pdir}"
        if args.strict:
            raise FileNotFoundError(message)
        _print(f"skipped: {message}")
        return None
    manifest = load_manifest(manifest_path)
    if manifest.get("actmap_shape") != list(actmap.EXPECTED_ACTMAP_SHAPE):
        raise RuntimeError(f"{dataset_id} {model}: bad manifest shape {manifest.get('actmap_shape')}")
    if manifest.get("actmap_normalized") is not True:
        raise RuntimeError(f"{dataset_id} {model}: ActMaps are not marked normalized")

    aggregate_metric_path = metrics_path(args.metrics_root, METHOD, dataset_id, model)
    completed_seeds = []
    for seed in args.seeds:
        if seed_metrics_path(args.metrics_root, dataset_id, model, seed).exists():
            completed_seeds.append(seed)
    if aggregate_metric_path.exists() and len(completed_seeds) == len(args.seeds) and not args.overwrite:
        _print(f"exists: {aggregate_metric_path}")
        return read_json(aggregate_metric_path)

    _print(f"loading records {records_path}")
    records = load_records(records_path)
    train_rows, validation_rows, test_rows, index_counts = split_rows(
        args=args,
        records=records,
        dataset_id=dataset_id,
        model=model,
        index_root=index_root,
    )
    metadata = {
        "root_note": root_note,
        "actmap_root": str(actmap_root),
        "balanced_index_root": str(index_root),
        "manifest_path": str(manifest_path),
        "records_path": str(records_path),
        "manifest": manifest,
        "balanced_counts": index_counts,
    }
    if args.dry_run:
        _print(
            f"dry-run {dataset_id} {model} root={root_note} "
            f"train={len(train_rows)} validation={len(validation_rows)} test={len(test_rows)} seeds={args.seeds}"
        )
        return {
            "method": METHOD,
            "dataset_id": dataset_id,
            "model": model,
            "dry_run": True,
            "metadata": metadata,
        }

    seed_results: list[actmap.TrainResult] = []
    for seed in args.seeds:
        out_dir = checkpoint_dir(args.checkpoint_root, dataset_id, model, seed)
        per_seed_metrics_path = seed_metrics_path(args.metrics_root, dataset_id, model, seed)
        per_seed_prediction_path = seed_prediction_path(args.prediction_root, dataset_id, model, seed)
        if per_seed_metrics_path.exists() and per_seed_prediction_path.exists() and not args.overwrite:
            _print(f"seed exists: {per_seed_metrics_path}")
            metrics = read_json(per_seed_metrics_path)
            prediction_rows = load_parquet_rows(per_seed_prediction_path)
            seed_results.append(
                actmap.TrainResult(
                    seed=seed,
                    output_dir=out_dir,
                    checkpoint_path=out_dir / "best_model.pt",
                    prediction_rows=prediction_rows,
                    metrics=metrics,
                )
            )
            continue
        result = actmap.train_one_seed(
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            output_dir=out_dir,
            seed=seed,
            dataset_id=dataset_id,
            model_id=model,
            balance_mode=args.balance_mode,
            metadata=metadata,
            arch=args.arch,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
            noise_std=args.noise_std,
            warmup_epochs=args.warmup_epochs,
            mixup_alpha=args.mixup_alpha,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            mlp_ratio=args.mlp_ratio,
            attn_drop=args.attn_drop,
            drop_path_rate=args.drop_path_rate,
            num_workers=args.num_workers,
            require_cuda=args.require_cuda,
            deterministic=args.deterministic,
            allow_tf32=args.allow_tf32,
            ece_bins=args.ece_bins,
        )
        save_prediction_rows(per_seed_prediction_path, result.prediction_rows)
        write_json(per_seed_metrics_path, result.metrics)
        seed_results.append(result)
    aggregate = save_pair_aggregate(
        args=args,
        dataset_id=dataset_id,
        model=model,
        seed_results=seed_results,
        metadata=metadata,
    )
    _print(
        f"done {dataset_id} {model}: auroc={aggregate.get('auroc')} "
        f"auprc={aggregate.get('auprc')} ece={aggregate.get('ece')}"
    )
    return aggregate


def run(args: argparse.Namespace) -> int:
    payloads = []
    for dataset_id in args.datasets:
        for model in args.models:
            payload = run_pair(args, dataset_id, model)
            if payload is not None:
                payloads.append(payload)
    if not args.dry_run:
        write_aggregate_outputs(args.metrics_root, args.report_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate ActMap ViT2D on current artifacts.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--actmap-root", type=Path, default=DEFAULT_ACTMAP_ROOT)
    parser.add_argument("--balanced-index-root", type=Path, default=DEFAULT_BALANCED_INDEX_ROOT)
    parser.add_argument("--gsm8k-direct-actmap-root", type=Path, default=DEFAULT_DIRECT_GSM8K_ACTMAP_ROOT)
    parser.add_argument("--gsm8k-direct-balanced-index-root", type=Path, default=DEFAULT_DIRECT_GSM8K_BALANCED_INDEX_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--dataset", dest="datasets", action="append", default=[])
    parser.add_argument("--model", dest="models", action="append", default=[])
    parser.add_argument("--seed", dest="seeds", type=int, action="append", default=[])
    parser.add_argument("--balance-mode", choices=["balanced"], default="balanced")
    parser.add_argument("--use-gsm8k-direct-roots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--seed-for-limit", type=int, default=42)
    parser.add_argument("--arch", choices=actmap.ARCH_CHOICES, default="vit2d")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--patch-h", type=int, default=4)
    parser.add_argument("--patch-w", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=3.0)
    parser.add_argument("--attn-drop", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--validate-tensors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.datasets = args.datasets or list(DEFAULT_DATASETS)
    args.models = args.models or list(DEFAULT_SMALL_MODELS)
    args.seeds = args.seeds or list(DEFAULT_SEEDS)
    validate_args(args)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
