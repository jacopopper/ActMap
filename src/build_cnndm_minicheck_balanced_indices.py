from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .balanced_indices import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_SMALL_MODELS,
    INDEX_COLUMNS,
    MANIFEST_COLUMNS,
    coerce_bool,
    read_parquet_rows,
    select_balanced_rows,
    split_seed,
    write_parquet,
)
from .experiment_registry import DEFAULT_REGISTRY_PATH, REPO_ROOT, read_json
from .generate_actmaps import model_slug


CNN_DATASET_ID = "cnn_dailymail_3_0_0"
DEFAULT_DATA_ROOT = REPO_ROOT / "artifacts" / "data"
DEFAULT_LABEL_PATH = REPO_ROOT / "artifacts" / "results" / "factuality" / "minicheck.parquet"
DEFAULT_REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "cnndm_minicheck_balanced_indices_report.md"
DEFAULT_SPLITS = ("train", "validation", "test")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _print(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _pd():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and pyarrow are required for CNN/DailyMail balanced indices") from exc
    return pd


def registry_model_ids(path: Path) -> set[str]:
    registry = read_json(path)
    return {str(model["hf_id"]) for model in registry.get("models", [])}


def resolve_models(requested: list[str], registry_path: Path) -> list[str]:
    valid = registry_model_ids(registry_path)
    if not requested or requested == ["small"] or "small" in requested:
        resolved = [model for model in DEFAULT_SMALL_MODELS if model in valid]
        resolved.extend(model for model in requested if model != "small")
    elif "all" in requested:
        resolved = sorted(valid)
    else:
        resolved = requested
    unknown = [model for model in resolved if model not in valid]
    if unknown:
        raise ValueError(f"Unknown model ids: {unknown}")
    deduped: list[str] = []
    seen: set[str] = set()
    for model in resolved:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def infer_label_path(data_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    artifact_root = data_root.parent if data_root.name == "data" else REPO_ROOT / "artifacts"
    candidate = artifact_root / "results" / "factuality" / "minicheck.parquet"
    return candidate if candidate.exists() else DEFAULT_LABEL_PATH


def load_label_rows(path: Path, dataset_id: str, model: str) -> list[dict[str, Any]]:
    pd = _pd()
    if not path.exists():
        raise FileNotFoundError(f"missing MiniCheck label parquet: {path}")
    frame = pd.read_parquet(path)
    slug = model_slug(model)
    if "dataset_id" in frame.columns:
        frame = frame[frame["dataset_id"] == dataset_id]
    if "model_slug" in frame.columns:
        frame = frame[frame["model_slug"] == slug]
    elif "model" in frame.columns:
        frame = frame[frame["model"] == model]
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        source_record_id = str(raw.get("source_record_id") or "")
        split = str(raw.get("split") or "")
        is_correct = coerce_bool(raw.get("is_correct"))
        if not source_record_id or not split or is_correct is None:
            continue
        rows.append(
            {
                "source_record_id": source_record_id,
                "split": split,
                "is_correct": is_correct,
                "label_status": raw.get("label_status") or raw.get("status") or "minicheck_labeled",
                "generated_token_count": raw.get("generated_token_count"),
            }
        )
    return rows


def build_indices(
    *,
    data_root: Path,
    label_path: Path,
    output_root: Path,
    report_path: Path,
    registry_path: Path,
    dataset_id: str,
    models: list[str],
    splits: list[str],
    seed: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    model_ids = resolve_models(models, registry_path)
    source_records = len(read_parquet_rows(data_root / dataset_id / "source_records.parquet"))
    manifest_rows: list[dict[str, Any]] = []
    for model in model_ids:
        slug = model_slug(model)
        label_rows = load_label_rows(label_path, dataset_id, model)
        rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
        skipped_split_rows = 0
        label_status_counts: dict[str, Counter[str]] = {split: Counter() for split in splits}
        for row in label_rows:
            split = str(row["split"])
            if split not in rows_by_split:
                skipped_split_rows += 1
                continue
            rows_by_split[split].append(row)
            label_status_counts[split][str(row.get("label_status") or "")] += 1
        for split in splits:
            selected_rows, n_correct, n_incorrect, selected_per_class = select_balanced_rows(
                rows=rows_by_split[split],
                dataset_id=dataset_id,
                model=model,
                split=split,
                seed=seed,
            )
            output_path = output_root / dataset_id / slug / f"{split}.parquet"
            if selected_rows:
                if not dry_run:
                    write_parquet(output_path, selected_rows, columns=INDEX_COLUMNS)
                status = "written"
                reason = "ok"
            elif not rows_by_split[split]:
                status = "skipped"
                reason = "no_labeled_rows"
            elif n_correct == 0:
                status = "skipped"
                reason = "missing_correct_class"
            elif n_incorrect == 0:
                status = "skipped"
                reason = "missing_incorrect_class"
            else:
                status = "skipped"
                reason = "no_balanced_rows"
            manifest_rows.append(
                {
                    "dataset_id": dataset_id,
                    "model": model,
                    "model_slug": slug,
                    "split": split,
                    "status": status,
                    "reason": reason,
                    "balance_mode": "balanced",
                    "sample_seed": seed,
                    "split_sample_seed": split_seed(seed, dataset_id, model, split),
                    "source_records": source_records,
                    "generation_records": len(label_rows),
                    "labeled_records": len(rows_by_split[split]),
                    "unlabeled_records": 0,
                    "unknown_source_records": 0,
                    "missing_split_records": skipped_split_rows,
                    "n_correct_total": n_correct,
                    "n_incorrect_total": n_incorrect,
                    "n_selected_per_class": selected_per_class,
                    "n_selected": len(selected_rows),
                    "manifest_records": None,
                    "manifest_failures": None,
                    "manifest_path": "",
                    "generation_path": str(label_path),
                    "output_path": str(output_path) if selected_rows else "",
                    "label_status_counts_json": json.dumps(label_status_counts[split], sort_keys=True),
                }
            )
            _print(
                f"{dataset_id} {model} {split}: status={status} selected={len(selected_rows)} "
                f"correct={n_correct} incorrect={n_incorrect}"
            )
    if not dry_run:
        write_parquet(output_root / "cnndm_minicheck_manifest.parquet", manifest_rows, columns=MANIFEST_COLUMNS)
        write_report(report_path, manifest_rows, label_path=label_path, output_root=output_root)
    return manifest_rows


def write_report(report_path: Path, rows: list[dict[str, Any]], *, label_path: Path, output_root: Path) -> None:
    lines = [
        "# CNN/DailyMail MiniCheck Balanced Indices",
        "",
        f"Label source: `{label_path}`",
        f"Output root: `{output_root}`",
        "",
        "| Model | Split | Status | Correct | Incorrect | Selected | Output |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model_slug']}` | `{row['split']}` | {row['status']} "
            f"| {row['n_correct_total']} | {row['n_incorrect_total']} | {row['n_selected']} "
            f"| `{row['output_path']}` |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced CNN/DailyMail indices from MiniCheck factuality labels.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--dataset", default=CNN_DATASET_ID)
    parser.add_argument("--model", dest="models", action="append", default=[])
    parser.add_argument("--split", dest="splits", action="append", default=[])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.models = args.models or ["small"]
    args.splits = args.splits or list(DEFAULT_SPLITS)
    args.label_path = infer_label_path(args.data_root, args.label_path)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_indices(
        data_root=args.data_root,
        label_path=args.label_path,
        output_root=args.output_root,
        report_path=args.report_path,
        registry_path=args.registry,
        dataset_id=args.dataset,
        models=args.models,
        splits=args.splits,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
