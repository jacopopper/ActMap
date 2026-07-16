from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .experiment_registry import DEFAULT_REGISTRY_PATH, REPO_ROOT, read_json
from .generate_actmaps import model_slug


DEFAULT_DATA_ROOT = REPO_ROOT / "artifacts" / "data"
DEFAULT_GENERATION_ROOT = REPO_ROOT / "artifacts" / "generations"
DEFAULT_ACTMAP_ROOT = REPO_ROOT / "artifacts" / "features" / "actmap"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "balanced_indices"
DEFAULT_REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "balanced_indices_report.md"
DEFAULT_SEED = 42
DEFAULT_SMALL_MODELS = (
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
DEFAULT_DATASETS = (
    "triviaqa_no_context",
    "nq_open",
    "gsm8k_rationale",
    "cnn_dailymail_3_0_0",
)
DEFAULT_SPLITS = ("train", "validation", "test")
INDEX_COLUMNS = [
    "dataset_id",
    "model",
    "model_slug",
    "source_record_id",
    "split",
    "is_correct",
    "label",
    "balance_mode",
    "sample_seed",
    "split_sample_seed",
    "selection_rank",
    "n_correct_total",
    "n_incorrect_total",
    "n_selected_per_class",
    "label_status",
    "generated_token_count",
]
MANIFEST_COLUMNS = [
    "dataset_id",
    "model",
    "model_slug",
    "split",
    "status",
    "reason",
    "balance_mode",
    "sample_seed",
    "split_sample_seed",
    "source_records",
    "generation_records",
    "labeled_records",
    "unlabeled_records",
    "unknown_source_records",
    "missing_split_records",
    "n_correct_total",
    "n_incorrect_total",
    "n_selected_per_class",
    "n_selected",
    "manifest_records",
    "manifest_failures",
    "manifest_path",
    "generation_path",
    "output_path",
    "label_status_counts_json",
]


class BalancedIndexError(RuntimeError):
    """Raised when balanced-index artifacts fail validation."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _print(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _pd():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and pyarrow are required for balanced-index parquet artifacts") from exc
    return pd


def _clean_value(value: Any, pd: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    pd = _pd()
    frame = pd.read_parquet(path)
    return [
        {key: _clean_value(value, pd) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def write_parquet(path: Path, rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> None:
    pd = _pd()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def split_seed(base_seed: int, dataset_id: str, model: str, split: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{dataset_id}|{model}|{split}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def registry_dataset_ids(registry: dict[str, Any]) -> list[str]:
    return [str(dataset["id"]) for dataset in registry.get("datasets", [])]


def registry_model_ids(registry: dict[str, Any]) -> list[str]:
    return [str(model["hf_id"]) for model in registry.get("models", [])]


def registry_splits(registry: dict[str, Any]) -> list[str]:
    splits = registry.get("global", {}).get("splits") or list(DEFAULT_SPLITS)
    return [str(split) for split in splits]


def resolve_datasets(requested: Iterable[str], registry: dict[str, Any]) -> list[str]:
    requested = list(requested) or ["all"]
    valid = registry_dataset_ids(registry)
    if "all" in requested:
        return valid
    unknown = [dataset_id for dataset_id in requested if dataset_id not in set(valid)]
    if unknown:
        raise ValueError(f"Unknown dataset ids: {unknown}; valid ids are {valid}")
    return requested


def resolve_models(requested: Iterable[str], registry: dict[str, Any]) -> list[str]:
    requested = list(requested) or ["small"]
    valid = registry_model_ids(registry)
    if "all" in requested:
        return valid
    if "small" in requested:
        requested = [model for model in DEFAULT_SMALL_MODELS if model in set(valid)] + [
            model for model in requested if model != "small"
        ]
    unknown = [model for model in requested if model not in set(valid)]
    if unknown:
        raise ValueError(f"Unknown model ids: {unknown}; valid ids are {valid}")
    seen: set[str] = set()
    resolved: list[str] = []
    for model in requested:
        if model not in seen:
            resolved.append(model)
            seen.add(model)
    return resolved


def read_generation_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    stats = {"lines": 0, "blank_lines": 0, "duplicates": 0, "invalid_json": 0, "missing_id": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                stats["blank_lines"] += 1
                continue
            stats["lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue
            source_record_id = str(row.get("source_record_id") or "")
            if not source_record_id:
                stats["missing_id"] += 1
                continue
            if source_record_id in rows_by_id:
                stats["duplicates"] += 1
            rows_by_id[source_record_id] = row
    return list(rows_by_id.values()), stats


def load_dataset_context(data_root: Path, dataset_id: str) -> dict[str, Any]:
    dataset_dir = data_root / dataset_id
    source_rows = read_parquet_rows(dataset_dir / "source_records.parquet")
    split_rows = read_parquet_rows(dataset_dir / "splits.parquet")
    labels_path = dataset_dir / "labels.parquet"
    label_rows = read_parquet_rows(labels_path) if labels_path.exists() else []
    split_map = {str(row["source_record_id"]): str(row["split"]) for row in split_rows}
    label_ids = {str(row["source_record_id"]) for row in label_rows}
    source_ids = {str(row["source_record_id"]) for row in source_rows}
    return {
        "source_ids": source_ids,
        "split_map": split_map,
        "label_ids": label_ids,
        "source_records": len(source_rows),
    }


def load_manifest_summary(actmap_root: Path, dataset_id: str, model: str) -> dict[str, Any]:
    path = actmap_root / dataset_id / model_slug(model) / "manifest.json"
    if not path.exists():
        return {"exists": False, "path": str(path), "records": None, "failures": None}
    payload = read_json(path)
    return {
        "exists": True,
        "path": str(path),
        "records": int(payload.get("records") or 0),
        "failures": int(payload.get("failures") or 0),
    }


def skipped_manifest_rows(
    *,
    dataset_id: str,
    model: str,
    splits: list[str],
    seed: int,
    reason: str,
    source_records: int = 0,
    generation_path: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    slug = model_slug(model)
    manifest = manifest or {}
    rows = []
    for split in splits:
        rows.append(
            {
                "dataset_id": dataset_id,
                "model": model,
                "model_slug": slug,
                "split": split,
                "status": "skipped",
                "reason": reason,
                "balance_mode": "balanced",
                "sample_seed": seed,
                "split_sample_seed": split_seed(seed, dataset_id, model, split),
                "source_records": source_records,
                "generation_records": 0,
                "labeled_records": 0,
                "unlabeled_records": 0,
                "unknown_source_records": 0,
                "missing_split_records": 0,
                "n_correct_total": 0,
                "n_incorrect_total": 0,
                "n_selected_per_class": 0,
                "n_selected": 0,
                "manifest_records": manifest.get("records"),
                "manifest_failures": manifest.get("failures"),
                "manifest_path": manifest.get("path", ""),
                "generation_path": str(generation_path) if generation_path is not None else "",
                "output_path": "",
                "label_status_counts_json": "{}",
            }
        )
    return rows


def select_balanced_rows(
    *,
    rows: list[dict[str, Any]],
    dataset_id: str,
    model: str,
    split: str,
    seed: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    correct = sorted((row for row in rows if row["is_correct"] is True), key=lambda row: row["source_record_id"])
    incorrect = sorted((row for row in rows if row["is_correct"] is False), key=lambda row: row["source_record_id"])
    selected_per_class = min(len(correct), len(incorrect))
    if selected_per_class == 0:
        return [], len(correct), len(incorrect), 0
    rng = random.Random(split_seed(seed, dataset_id, model, split))
    selected = [(True, row) for row in rng.sample(correct, selected_per_class)]
    selected.extend((False, row) for row in rng.sample(incorrect, selected_per_class))
    rng.shuffle(selected)
    index_rows: list[dict[str, Any]] = []
    for selection_rank, (label, row) in enumerate(selected):
        index_rows.append(
            {
                "dataset_id": dataset_id,
                "model": model,
                "model_slug": model_slug(model),
                "source_record_id": row["source_record_id"],
                "split": split,
                "is_correct": bool(label),
                "label": int(label),
                "balance_mode": "balanced",
                "sample_seed": seed,
                "split_sample_seed": split_seed(seed, dataset_id, model, split),
                "selection_rank": selection_rank,
                "n_correct_total": len(correct),
                "n_incorrect_total": len(incorrect),
                "n_selected_per_class": selected_per_class,
                "label_status": row.get("label_status"),
                "generated_token_count": row.get("generated_token_count"),
            }
        )
    return index_rows, len(correct), len(incorrect), selected_per_class


def build_pair_indices(
    *,
    dataset_id: str,
    model: str,
    data_context: dict[str, Any],
    generation_root: Path,
    actmap_root: Path,
    output_root: Path,
    splits: list[str],
    seed: int,
    include_partial: bool,
    require_manifest: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    slug = model_slug(model)
    generation_path = generation_root / dataset_id / slug / "records.jsonl"
    manifest = load_manifest_summary(actmap_root, dataset_id, model)
    source_records = int(data_context["source_records"])

    if require_manifest and not manifest["exists"]:
        return skipped_manifest_rows(
            dataset_id=dataset_id,
            model=model,
            splits=splits,
            seed=seed,
            reason="missing_actmap_manifest",
            source_records=source_records,
            generation_path=generation_path,
            manifest=manifest,
        )
    if (
        not include_partial
        and manifest.get("records") is not None
        and manifest.get("failures") is not None
        and source_records
        and int(manifest["records"]) + int(manifest["failures"]) < source_records
    ):
        return skipped_manifest_rows(
            dataset_id=dataset_id,
            model=model,
            splits=splits,
            seed=seed,
            reason="incomplete_actmap_manifest",
            source_records=source_records,
            generation_path=generation_path,
            manifest=manifest,
        )
    if not generation_path.exists():
        return skipped_manifest_rows(
            dataset_id=dataset_id,
            model=model,
            splits=splits,
            seed=seed,
            reason="missing_generation_records",
            source_records=source_records,
            generation_path=generation_path,
            manifest=manifest,
        )

    generation_rows, generation_stats = read_generation_jsonl(generation_path)
    split_map: dict[str, str] = data_context["split_map"]
    source_ids: set[str] = data_context["source_ids"]
    label_ids: set[str] = data_context["label_ids"]
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    unlabeled_by_split: Counter[str] = Counter()
    label_status_by_split: dict[str, Counter[str]] = {split: Counter() for split in splits}
    missing_split_records = 0
    unknown_source_records = 0

    for row in generation_rows:
        source_record_id = str(row.get("source_record_id") or "")
        if source_record_id not in source_ids:
            unknown_source_records += 1
            continue
        split = str(row.get("split") or split_map.get(source_record_id) or "")
        if split not in rows_by_split:
            missing_split_records += 1
            continue
        label_status_by_split[split][str(row.get("label_status") or "")] += 1
        label = coerce_bool(row.get("is_correct"))
        if label is None:
            unlabeled_by_split[split] += 1
            continue
        if label_ids and source_record_id not in label_ids:
            # Keep the row usable, but validation/reporting will make the join issue visible.
            pass
        rows_by_split[split].append(
            {
                "source_record_id": source_record_id,
                "is_correct": label,
                "label_status": row.get("label_status"),
                "generated_token_count": row.get("generated_token_count"),
            }
        )

    manifest_rows: list[dict[str, Any]] = []
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
                "generation_records": len(generation_rows),
                "labeled_records": len(rows_by_split[split]),
                "unlabeled_records": int(unlabeled_by_split[split]),
                "unknown_source_records": unknown_source_records + int(generation_stats["missing_id"]),
                "missing_split_records": missing_split_records,
                "n_correct_total": n_correct,
                "n_incorrect_total": n_incorrect,
                "n_selected_per_class": selected_per_class,
                "n_selected": len(selected_rows),
                "manifest_records": manifest.get("records"),
                "manifest_failures": manifest.get("failures"),
                "manifest_path": manifest.get("path", ""),
                "generation_path": str(generation_path),
                "output_path": str(output_path) if selected_rows else "",
                "label_status_counts_json": json.dumps(label_status_by_split[split], sort_keys=True),
            }
        )
    return manifest_rows


def build_balanced_indices(
    *,
    registry_path: Path,
    data_root: Path,
    generation_root: Path,
    actmap_root: Path,
    output_root: Path,
    datasets: list[str],
    models: list[str],
    seed: int,
    include_partial: bool = False,
    require_manifest: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    registry = read_json(registry_path)
    dataset_ids = resolve_datasets(datasets, registry)
    model_ids = resolve_models(models, registry)
    splits = registry_splits(registry)
    all_manifest_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        try:
            data_context = load_dataset_context(data_root, dataset_id)
        except Exception as exc:
            for model in model_ids:
                all_manifest_rows.extend(
                    skipped_manifest_rows(
                        dataset_id=dataset_id,
                        model=model,
                        splits=splits,
                        seed=seed,
                        reason=f"data_artifact_error:{exc.__class__.__name__}:{exc}",
                    )
                )
            continue
        for model in model_ids:
            pair_rows = build_pair_indices(
                dataset_id=dataset_id,
                model=model,
                data_context=data_context,
                generation_root=generation_root,
                actmap_root=actmap_root,
                output_root=output_root,
                splits=splits,
                seed=seed,
                include_partial=include_partial,
                require_manifest=require_manifest,
                dry_run=dry_run,
            )
            written = sum(1 for row in pair_rows if row["status"] == "written")
            selected = sum(int(row["n_selected"] or 0) for row in pair_rows)
            _print(f"{dataset_id} {model}: written_splits={written} selected_rows={selected}")
            all_manifest_rows.extend(pair_rows)
    return all_manifest_rows


def _manifest_path(output_root: Path) -> Path:
    return output_root / "manifest.parquet"


def _validation_error(message: str, errors: list[str]) -> None:
    errors.append(message)


def load_manifest(output_root: Path) -> list[dict[str, Any]]:
    path = _manifest_path(output_root)
    if not path.exists():
        raise BalancedIndexError(f"missing manifest: {path}")
    return read_parquet_rows(path)


def validate_balanced_indices(
    *,
    output_root: Path,
    data_root: Path,
    generation_root: Path,
    registry_path: Path,
    datasets: list[str],
    models: list[str],
) -> list[dict[str, Any]]:
    registry = read_json(registry_path)
    allowed_datasets = set(resolve_datasets(datasets, registry))
    allowed_models = set(resolve_models(models, registry))
    manifest_rows = [
        row
        for row in load_manifest(output_root)
        if row["dataset_id"] in allowed_datasets and row["model"] in allowed_models
    ]
    errors: list[str] = []
    generation_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    split_cache: dict[str, dict[str, str]] = {}
    label_cache: dict[str, set[str]] = {}
    source_cache: dict[str, set[str]] = {}
    summaries: list[dict[str, Any]] = []

    for row in manifest_rows:
        if row["status"] != "written":
            continue
        dataset_id = str(row["dataset_id"])
        model = str(row["model"])
        split = str(row["split"])
        output_path = Path(str(row["output_path"]))
        if not output_path.exists():
            _validation_error(f"missing balanced index file: {output_path}", errors)
            continue
        index_rows = read_parquet_rows(output_path)
        if not index_rows:
            _validation_error(f"empty balanced index file: {output_path}", errors)
            continue
        labels = [coerce_bool(index_row.get("is_correct")) for index_row in index_rows]
        if any(label is None for label in labels):
            _validation_error(f"non-binary labels in {output_path}", errors)
            continue
        n_correct = sum(1 for label in labels if label is True)
        n_incorrect = sum(1 for label in labels if label is False)
        if n_correct != n_incorrect:
            _validation_error(f"unbalanced labels in {output_path}: correct={n_correct} incorrect={n_incorrect}", errors)
        ids = [str(index_row["source_record_id"]) for index_row in index_rows]
        if len(ids) != len(set(ids)):
            _validation_error(f"duplicate source_record_id values in {output_path}", errors)
        if dataset_id not in split_cache:
            context = load_dataset_context(data_root, dataset_id)
            split_cache[dataset_id] = context["split_map"]
            label_cache[dataset_id] = context["label_ids"]
            source_cache[dataset_id] = context["source_ids"]
        generation_key = (dataset_id, model)
        if generation_key not in generation_cache:
            generation_path = generation_root / dataset_id / model_slug(model) / "records.jsonl"
            generation_rows, _ = read_generation_jsonl(generation_path)
            generation_cache[generation_key] = {
                str(generation_row["source_record_id"]): generation_row for generation_row in generation_rows
            }
        for index_row in index_rows:
            source_record_id = str(index_row["source_record_id"])
            if source_record_id not in source_cache[dataset_id]:
                _validation_error(f"{source_record_id} in {output_path} is not in source records", errors)
                continue
            expected_split = split_cache[dataset_id].get(source_record_id)
            if expected_split != split:
                _validation_error(
                    f"{source_record_id} in {output_path} has split={split}, expected {expected_split}",
                    errors,
                )
            if label_cache[dataset_id] and source_record_id not in label_cache[dataset_id]:
                _validation_error(f"{source_record_id} in {output_path} is not in labels.parquet", errors)
            generation_row = generation_cache[generation_key].get(source_record_id)
            if generation_row is None:
                _validation_error(f"{source_record_id} in {output_path} is not in generation records", errors)
                continue
            generation_label = coerce_bool(generation_row.get("is_correct"))
            if generation_label != coerce_bool(index_row.get("is_correct")):
                _validation_error(f"{source_record_id} in {output_path} label disagrees with generation JSONL", errors)
        summaries.append(
            {
                "dataset_id": dataset_id,
                "model": model,
                "split": split,
                "rows": len(index_rows),
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "path": str(output_path),
            }
        )
    if errors:
        raise BalancedIndexError("\n".join(errors))
    return summaries


def build_report(manifest_rows: list[dict[str, Any]], *, validation_rows: list[dict[str, Any]] | None = None) -> str:
    written_rows = [row for row in manifest_rows if row.get("status") == "written"]
    skipped_rows = [row for row in manifest_rows if row.get("status") != "written"]
    total_selected = sum(int(row.get("n_selected") or 0) for row in written_rows)
    status = "completed" if written_rows else "blocked"
    lines = [
        "# Balanced Label Index Report",
        "",
        f"Status: {status}.",
        "",
        "## Outputs",
        "",
        "- `artifacts/balanced_indices/{dataset}/{model}/{split}.parquet`",
        "- `artifacts/balanced_indices/manifest.parquet`",
        "- `artifacts/reports/balanced_indices_report.md`",
        "",
        "## Summary",
        "",
        f"- Written split indexes: {len(written_rows)}",
        f"- Skipped split indexes: {len(skipped_rows)}",
        f"- Selected balanced rows: {total_selected}",
        "- Balance mode: per dataset x model x split with equal correct and incorrect generated answers.",
        "- CNN/DailyMail requires completed MiniCheck factuality labels.",
        "",
        "## Written Indexes",
        "",
        "| Dataset | Model | Split | Rows | Correct | Incorrect | Path |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(written_rows, key=lambda item: (item["dataset_id"], item["model_slug"], item["split"])):
        lines.append(
            f"| `{row['dataset_id']}` | `{row['model_slug']}` | `{row['split']}` "
            f"| {row['n_selected']} | {row['n_selected_per_class']} | {row['n_selected_per_class']} "
            f"| `{row['output_path']}` |"
        )
    if not written_rows:
        lines.append("| none | none | none | 0 | 0 | 0 | none |")
    skip_counts = Counter(str(row.get("reason") or "unknown") for row in skipped_rows)
    lines.extend(["", "## Skipped", ""])
    if skip_counts:
        for reason, count in sorted(skip_counts.items()):
            lines.append(f"- `{reason}`: {count} split indexes")
    else:
        lines.append("- None")
    if validation_rows is not None:
        lines.extend(["", "## Validation", "", f"- Validated written indexes: {len(validation_rows)}"])
        lines.append("- Every validated index has equal correct/incorrect rows and joins to source, split, label, and generation records.")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "python3 -m src.balanced_indices build --dataset all --model small",
            "python3 -m src.balanced_indices validate --dataset all --model small",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def cmd_build(args: argparse.Namespace) -> int:
    manifest_rows = build_balanced_indices(
        registry_path=args.registry,
        data_root=args.data_root,
        generation_root=args.generation_root,
        actmap_root=args.actmap_root,
        output_root=args.output_root,
        datasets=args.datasets,
        models=args.models,
        seed=args.seed,
        include_partial=args.include_partial,
        require_manifest=args.require_manifest,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(build_report(manifest_rows), end="")
        return 0
    write_parquet(_manifest_path(args.output_root), manifest_rows, columns=MANIFEST_COLUMNS)
    try:
        validation_rows = validate_balanced_indices(
            output_root=args.output_root,
            data_root=args.data_root,
            generation_root=args.generation_root,
            registry_path=args.registry,
            datasets=args.datasets,
            models=args.models,
        )
    except BalancedIndexError as exc:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(build_report(manifest_rows), encoding="utf-8")
        print(f"ERROR: validation failed: {exc}", file=sys.stderr)
        return 1
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(build_report(manifest_rows, validation_rows=validation_rows), encoding="utf-8")
    print(f"wrote {_manifest_path(args.output_root)}")
    print(f"wrote {args.report_path}")
    return 0 if any(row["status"] == "written" for row in manifest_rows) else 1


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        validation_rows = validate_balanced_indices(
            output_root=args.output_root,
            data_root=args.data_root,
            generation_root=args.generation_root,
            registry_path=args.registry,
            datasets=args.datasets,
            models=args.models,
        )
    except BalancedIndexError as exc:
        print(f"ERROR: validation failed: {exc}", file=sys.stderr)
        return 1
    manifest_rows = load_manifest(args.output_root)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(build_report(manifest_rows, validation_rows=validation_rows), encoding="utf-8")
    print(f"validated {len(validation_rows)} balanced index files")
    print(f"wrote {args.report_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    manifest_rows = load_manifest(args.output_root)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(build_report(manifest_rows), encoding="utf-8")
    print(f"wrote {args.report_path}")
    return 0


def add_common_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    subparser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    subparser.add_argument("--generation-root", type=Path, default=DEFAULT_GENERATION_ROOT)
    subparser.add_argument("--actmap-root", type=Path, default=DEFAULT_ACTMAP_ROOT)
    subparser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    subparser.add_argument("--dataset", dest="datasets", action="append", default=[])
    subparser.add_argument("--model", dest="models", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate ActMap balanced label indexes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build balanced index parquet files and manifest.")
    add_common_args(build)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--include-partial", action="store_true")
    build.add_argument("--require-manifest", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(func=cmd_build)

    validate = subparsers.add_parser("validate", help="Validate existing balanced index files.")
    add_common_args(validate)
    validate.set_defaults(func=cmd_validate)

    report = subparsers.add_parser("report", help="Rewrite the balanced index report from an existing manifest.")
    add_common_args(report)
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
