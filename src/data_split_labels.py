from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import string
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterable

from .experiment_registry import DEFAULT_REGISTRY_PATH, REPO_ROOT, read_json


DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "data"
DEFAULT_REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "data_split_label_report.md"
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_FRAC = 0.80
DEFAULT_VALIDATION_FRAC = 0.10
DEFAULT_TEST_FRAC = 0.10


DATASET_SOURCES: dict[str, dict[str, Any]] = {
    "triviaqa_no_context": {
        "hf_path": "mandarjoshi/trivia_qa",
        "hf_name": "rc.nocontext",
        "hf_splits": ["train", "validation"],
        "builder": "triviaqa",
        "evaluator": "triviaqa_exact_or_f1_0_8",
    },
    "nq_open": {
        "hf_path": "google-research-datasets/nq_open",
        "hf_name": None,
        "hf_splits": ["train", "validation"],
        "builder": "nq_open",
        "evaluator": "nq_open_exact_or_f1_0_8",
    },
    "gsm8k_rationale": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "hf_splits": ["train", "test"],
        "builder": "gsm8k",
        "evaluator": "gsm8k_final_numeric_answer",
    },
    "cnn_dailymail_3_0_0": {
        "hf_path": "abisee/cnn_dailymail",
        "hf_name": "3.0.0",
        "hf_splits": ["train", "validation", "test"],
        "builder": "cnn_dailymail",
        "evaluator": "minicheck_flan_t5_large",
    },
}


_NUMBER_RE_TEXT = (
    r"(?<![\w])"
    r"[-+]?"
    r"(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)?"
    r"(?![\w])"
)
_GSM8K_NUMBER_RE = re.compile(_NUMBER_RE_TEXT)
_GSM8K_HASH_RE = re.compile(r"####\s*(?P<number>" + _NUMBER_RE_TEXT + r")", re.IGNORECASE)
_GSM8K_DIRECT_ANSWER_RE = re.compile(
    r"\b(?:final\s+answer|answer|result|total)\b"
    r"\s*(?:is|are|=|:)\s*"
    r"[$]?\s*(?P<number>" + _NUMBER_RE_TEXT + r")",
    re.IGNORECASE,
)
_GSM8K_LINE_ANSWER_RE = re.compile(r"^\s*(?:final\s+answer|answer)\s*[:=]\s*(?P<body>.+)$", re.IGNORECASE)
_GSM8K_EXPLANATION_SPLIT_RE = re.compile(r"\b(?:because|since|as)\b", re.IGNORECASE)
_GSM8K_ABS_TOLERANCE = Decimal("1e-6")


class MissingDependencyError(RuntimeError):
    """Raised when a command needs an optional data dependency."""


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_jsonish_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str):
        return [str(value)] if str(value).strip() else []
    if not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
    if isinstance(parsed, list | tuple):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)] if str(parsed).strip() else []


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize_split_key(text: str) -> str:
    return normalize_answer(text) or "empty"


def clean_prediction(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"(?is)<think>.*?</think>", " ", text).strip()
    text = re.sub(r"(?is)^answer\s*:\s*", "", text).strip()
    text = text.splitlines()[0].strip()
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    return text.strip(" \t\r\n\"'`*_")


def clean_gsm8k_prediction(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"(?is)<think>.*?</think>", " ", text)
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.replace("\u2212", "-")
    text = re.sub(r"<<[^>=]*=([^>]*)>>", r" \1 ", text)
    text = re.sub(r"[*_`]", " ", text)
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _decimal_from_number_text(text: str) -> Decimal | None:
    value = str(text).strip().replace(",", "").replace("$", "")
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".")
    if not value:
        return None
    try:
        with localcontext() as ctx:
            ctx.prec = 40
            if "/" in value:
                numerator, denominator = value.split("/", maxsplit=1)
                denom = Decimal(denominator)
                if denom == 0:
                    return None
                return Decimal(numerator) / denom
            return Decimal(value)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _format_decimal(value: Decimal | None) -> str:
    if value is None or not value.is_finite():
        return ""
    try:
        if value == 0:
            return "0"
        if value == value.to_integral_value():
            return format(value, "f")
        return format(value.normalize(), "f").rstrip("0").rstrip(".")
    except (InvalidOperation, ValueError, OverflowError):
        return str(value)


def _float_decimal_or_none(value: Decimal | None) -> float | None:
    if value is None or not value.is_finite():
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if result != float("inf") and result != float("-inf") else None


def _gsm8k_number_candidates(text: str) -> list[tuple[str, Decimal]]:
    candidates: list[tuple[str, Decimal]] = []
    for match in _GSM8K_NUMBER_RE.finditer(text):
        raw = match.group(0).strip()
        value = _decimal_from_number_text(raw)
        if value is not None:
            candidates.append((raw, value))
    return candidates


def extract_gsm8k_numeric_answer(text: str) -> str | None:
    cleaned = clean_gsm8k_prediction(text)
    if not cleaned:
        return None
    hash_matches = list(_GSM8K_HASH_RE.finditer(cleaned))
    if hash_matches:
        return hash_matches[-1].group("number").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines):
        line_match = _GSM8K_LINE_ANSWER_RE.match(line)
        if not line_match:
            continue
        body = _GSM8K_EXPLANATION_SPLIT_RE.split(line_match.group("body"), maxsplit=1)[0]
        candidates = _gsm8k_number_candidates(body)
        if candidates:
            return candidates[-1][0]
    direct_matches = list(_GSM8K_DIRECT_ANSWER_RE.finditer(cleaned))
    if direct_matches:
        return direct_matches[-1].group("number").strip()
    for line in reversed(lines):
        if "=" not in line:
            continue
        candidates = _gsm8k_number_candidates(line.rsplit("=", maxsplit=1)[-1])
        if candidates:
            return candidates[-1][0]
    candidates = _gsm8k_number_candidates(cleaned)
    return candidates[-1][0] if candidates else None


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    overlap = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in common)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_sequence_contains(container: str, contained: str) -> bool:
    container_tokens = normalize_answer(container).split()
    contained_tokens = normalize_answer(contained).split()
    if not container_tokens or not contained_tokens or len(contained_tokens) > len(container_tokens):
        return False
    return any(
        container_tokens[start : start + len(contained_tokens)] == contained_tokens
        for start in range(0, len(container_tokens) - len(contained_tokens) + 1)
    )


def qa_gold_aliases(row: dict[str, Any], *, include_answers_field: bool = False) -> list[str]:
    aliases = parse_jsonish_list(row.get("normalized_aliases_json") or row.get("normalized_aliases"))
    if include_answers_field:
        aliases.extend(parse_jsonish_list(row.get("answers_json") or row.get("answers")))
    aliases.extend(parse_jsonish_list(row.get("aliases_json") or row.get("aliases")))
    for key in ("answer_normalized", "answer_value"):
        value = row.get(key)
        if value is not None and str(value).strip():
            aliases.append(str(value))
    return sorted({normalize_answer(alias) for alias in aliases if normalize_answer(alias)})


def evaluate_short_answer(
    prediction: str,
    row: dict[str, Any],
    *,
    f1_threshold: float = 0.8,
    allow_containment: bool = False,
) -> dict[str, Any]:
    cleaned_prediction = clean_prediction(prediction)
    normalized_prediction = normalize_answer(cleaned_prediction)
    gold_answers = qa_gold_aliases(row, include_answers_field=allow_containment)
    exact_match = normalized_prediction in gold_answers
    containment_match = False
    if allow_containment:
        containment_match = any(
            token_sequence_contains(normalized_prediction, alias)
            or token_sequence_contains(alias, normalized_prediction)
            for alias in gold_answers
        )
    scored_aliases = [(alias, token_f1(normalized_prediction, alias)) for alias in gold_answers]
    matched_alias, best_f1 = max(scored_aliases, key=lambda item: item[1], default=("", 0.0))
    return {
        "prediction_raw": prediction,
        "prediction_clean": cleaned_prediction,
        "prediction_normalized": normalized_prediction,
        "exact_match": exact_match,
        "containment_match": containment_match,
        "f1": best_f1,
        "matched_alias": matched_alias,
        "is_correct": bool(exact_match or containment_match or best_f1 >= f1_threshold),
    }


def evaluate_gsm8k_answer(
    prediction: str,
    row: dict[str, Any],
    *,
    abs_tolerance: Decimal = _GSM8K_ABS_TOLERANCE,
) -> dict[str, Any]:
    cleaned_prediction = clean_gsm8k_prediction(prediction)
    prediction_answer = extract_gsm8k_numeric_answer(cleaned_prediction)
    gold_answer = (
        extract_gsm8k_numeric_answer(str(row.get("answer_value") or ""))
        or extract_gsm8k_numeric_answer(str(row.get("reference_solution") or ""))
    )
    prediction_value = _decimal_from_number_text(prediction_answer or "")
    gold_value = _decimal_from_number_text(gold_answer or "")
    if prediction_value is not None and not prediction_value.is_finite():
        prediction_value = None
    if gold_value is not None and not gold_value.is_finite():
        gold_value = None
    abs_error = None
    if prediction_value is not None and gold_value is not None:
        try:
            abs_error = abs(prediction_value - gold_value)
        except (InvalidOperation, ValueError, OverflowError):
            abs_error = None
    try:
        numeric_match = bool(abs_error is not None and abs_error.is_finite() and abs_error <= abs_tolerance)
    except (InvalidOperation, ValueError, OverflowError):
        numeric_match = False
    return {
        "prediction_raw": prediction,
        "prediction_clean": prediction_answer or "",
        "prediction_text_clean": cleaned_prediction,
        "prediction_normalized": _format_decimal(prediction_value),
        "gold_answer": gold_answer or "",
        "gold_normalized": _format_decimal(gold_value),
        "numeric_match": numeric_match,
        "abs_error": _float_decimal_or_none(abs_error),
        "is_correct": numeric_match,
    }


def evaluate_generation_correctness(dataset_id: str, prediction: str, source_row: dict[str, Any]) -> dict[str, Any]:
    if dataset_id in {"triviaqa_no_context", "nq_open"}:
        return evaluate_short_answer(prediction, source_row)
    if dataset_id == "gsm8k_rationale":
        return evaluate_gsm8k_answer(prediction, source_row)
    if dataset_id == "cnn_dailymail_3_0_0":
        raise ValueError("CNN/DailyMail factuality labels require MiniCheck outputs")
    raise KeyError(f"Unknown dataset_id: {dataset_id}")


def _as_string(value: Any) -> str:
    return "" if value is None else str(value)


def _answer_dict_value(answer: Any, key: str) -> Any:
    return answer.get(key) if isinstance(answer, dict) else None


def _source_record_id(dataset_id: str, source_split: str, source_index: int, native_id: Any = None) -> str:
    if native_id is not None and str(native_id).strip():
        clean_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(native_id).strip())
        return f"{dataset_id}:{clean_id}"
    return f"{dataset_id}:{source_split}:{source_index}"


def build_source_record(dataset_id: str, source_split: str, source_index: int, row: dict[str, Any]) -> dict[str, Any]:
    if dataset_id == "triviaqa_no_context":
        answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
        question = _as_string(row.get("question"))
        aliases = parse_jsonish_list(_answer_dict_value(answer, "aliases"))
        normalized_aliases = parse_jsonish_list(_answer_dict_value(answer, "normalized_aliases"))
        answer_value = _as_string(_answer_dict_value(answer, "value") or row.get("answer_value"))
        answer_normalized = _as_string(_answer_dict_value(answer, "normalized_value") or normalize_answer(answer_value))
        return {
            "source_record_id": _source_record_id(dataset_id, source_split, source_index, row.get("question_id")),
            "dataset_id": dataset_id,
            "source_split": source_split,
            "source_index": source_index,
            "source_key": normalize_split_key(question),
            "question": question,
            "input_text": question,
            "answer_value": answer_value,
            "answer_normalized": answer_normalized,
            "aliases_json": json_dumps(aliases),
            "normalized_aliases_json": json_dumps(normalized_aliases or [answer_normalized]),
            "reference_solution": "",
            "article": "",
            "reference_summary": "",
        }
    if dataset_id == "nq_open":
        question = _as_string(row.get("question"))
        answers = parse_jsonish_list(row.get("answer") or row.get("answers"))
        normalized = sorted({normalize_answer(answer) for answer in answers if normalize_answer(answer)})
        return {
            "source_record_id": _source_record_id(dataset_id, source_split, source_index, row.get("id")),
            "dataset_id": dataset_id,
            "source_split": source_split,
            "source_index": source_index,
            "source_key": normalize_split_key(question),
            "question": question,
            "input_text": question,
            "answer_value": answers[0] if answers else "",
            "answer_normalized": normalized[0] if normalized else "",
            "aliases_json": json_dumps(answers),
            "normalized_aliases_json": json_dumps(normalized),
            "reference_solution": "",
            "article": "",
            "reference_summary": "",
        }
    if dataset_id == "gsm8k_rationale":
        question = _as_string(row.get("question"))
        answer = _as_string(row.get("answer"))
        gold_answer = extract_gsm8k_numeric_answer(answer) or ""
        return {
            "source_record_id": _source_record_id(dataset_id, source_split, source_index, row.get("id")),
            "dataset_id": dataset_id,
            "source_split": source_split,
            "source_index": source_index,
            "source_key": normalize_split_key(question),
            "question": question,
            "input_text": question,
            "answer_value": gold_answer,
            "answer_normalized": _format_decimal(_decimal_from_number_text(gold_answer)),
            "aliases_json": json_dumps([gold_answer] if gold_answer else []),
            "normalized_aliases_json": json_dumps([_format_decimal(_decimal_from_number_text(gold_answer))] if gold_answer else []),
            "reference_solution": answer,
            "article": "",
            "reference_summary": "",
        }
    if dataset_id == "cnn_dailymail_3_0_0":
        article = _as_string(row.get("article"))
        summary = _as_string(row.get("highlights"))
        return {
            "source_record_id": _source_record_id(dataset_id, source_split, source_index, row.get("id")),
            "dataset_id": dataset_id,
            "source_split": source_split,
            "source_index": source_index,
            "source_key": normalize_split_key(row.get("id") or article[:500]),
            "question": "",
            "input_text": article,
            "answer_value": "",
            "answer_normalized": "",
            "aliases_json": json_dumps([]),
            "normalized_aliases_json": json_dumps([]),
            "reference_solution": "",
            "article": article,
            "reference_summary": summary,
        }
    raise KeyError(f"Unknown dataset_id: {dataset_id}")


def load_hf_source_records(dataset_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "The `datasets` package is required to build source records from Hugging Face datasets."
        ) from exc

    config = DATASET_SOURCES[dataset_id]
    records: list[dict[str, Any]] = []
    for source_split in config["hf_splits"]:
        kwargs = {"path": config["hf_path"], "split": source_split}
        if config["hf_name"] is not None:
            kwargs["name"] = config["hf_name"]
        dataset = load_dataset(**kwargs)
        for source_index, row in enumerate(dataset):
            if limit is not None and len(records) >= limit:
                return records
            records.append(build_source_record(dataset_id, source_split, source_index, dict(row)))
    return records


def load_jsonl_source_records(dataset_id: str, path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if limit is not None and len(records) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            source_split = str(row.pop("source_split", "local"))
            records.append(build_source_record(dataset_id, source_split, source_index, row))
    return records


def group_hash(group_key: str, seed: int) -> int:
    payload = f"{seed}:{group_key}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def assign_splits(
    records: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    validation_frac: float = DEFAULT_VALIDATION_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> list[dict[str, Any]]:
    if not records:
        return []
    total_frac = train_frac + validation_frac + test_frac
    if abs(total_frac - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_frac}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["source_key"])].append(record)

    ordered_groups = sorted(groups.items(), key=lambda item: group_hash(item[0], seed))
    total_rows = len(records)
    target_test = round(total_rows * test_frac)
    target_validation = round(total_rows * validation_frac)
    if total_rows >= 3:
        target_test = max(1, target_test)
        target_validation = max(1, target_validation)

    counts = {"train": 0, "validation": 0, "test": 0}
    split_rows: list[dict[str, Any]] = []
    for key, group_records in ordered_groups:
        if counts["test"] < target_test:
            split = "test"
        elif counts["validation"] < target_validation:
            split = "validation"
        else:
            split = "train"
        counts[split] += len(group_records)
        split_hash = group_hash(key, seed)
        for record in group_records:
            split_rows.append(
                {
                    "source_record_id": record["source_record_id"],
                    "dataset_id": record["dataset_id"],
                    "split": split,
                    "split_seed": seed,
                    "split_group_key": key,
                    "split_group_hash": split_hash,
                }
            )
    return sorted(split_rows, key=lambda row: row["source_record_id"])


def build_label_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for record in records:
        dataset_id = str(record["dataset_id"])
        config = DATASET_SOURCES[dataset_id]
        base = {
            "source_record_id": record["source_record_id"],
            "dataset_id": dataset_id,
            "label_target": "summary_factuality" if dataset_id == "cnn_dailymail_3_0_0" else "answer_correctness",
            "evaluator": config["evaluator"],
            "requires_generation": True,
        }
        if dataset_id == "cnn_dailymail_3_0_0":
            labels.append(
                {
                    **base,
                    "label_status": "pending_minicheck_outputs",
                    "gold_answer": "",
                    "gold_aliases_json": json_dumps([]),
                    "minicheck_label": None,
                    "human_label": None,
                }
            )
        else:
            gold_aliases = qa_gold_aliases(record)
            labels.append(
                {
                    **base,
                    "label_status": "gold_answer_available",
                    "gold_answer": record.get("answer_normalized") or record.get("answer_value") or "",
                    "gold_aliases_json": json_dumps(gold_aliases),
                    "minicheck_label": None,
                    "human_label": None,
                }
            )
    return labels


def assert_no_split_leakage(split_rows: list[dict[str, Any]]) -> None:
    split_by_key: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        split_by_key[str(row["split_group_key"])].add(str(row["split"]))
    leaked = {key: sorted(splits) for key, splits in split_by_key.items() if len(splits) > 1}
    if leaked:
        raise AssertionError(f"Found source keys assigned to multiple splits: {leaked}")


def split_counts(split_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in split_rows:
        counts[str(row["split"])] = counts.get(str(row["split"]), 0) + 1
    return counts


def label_prevalence(label_rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in label_rows:
        status = str(row["label_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "row_count": len(label_rows),
        "status_counts": status_counts,
        "binary_prevalence": None,
        "note": "Binary correctness/factuality prevalence is available after generation-time labels are evaluated.",
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Writing parquet requires `pandas` plus a parquet engine such as `pyarrow`."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Reading parquet requires `pandas` plus a parquet engine such as `pyarrow`."
        ) from exc
    return pd.read_parquet(path).to_dict(orient="records")


def build_dataset_artifacts(
    dataset_id: str,
    output_root: Path,
    *,
    input_jsonl: Path | None = None,
    limit: int | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    records = (
        load_jsonl_source_records(dataset_id, input_jsonl, limit=limit)
        if input_jsonl is not None
        else load_hf_source_records(dataset_id, limit=limit)
    )
    split_rows = assign_splits(records, seed=split_seed)
    assert_no_split_leakage(split_rows)
    label_rows = build_label_rows(records)

    dataset_dir = output_root / dataset_id
    write_parquet(dataset_dir / "source_records.parquet", records)
    write_parquet(dataset_dir / "splits.parquet", split_rows)
    write_parquet(dataset_dir / "labels.parquet", label_rows)
    return {
        "dataset_id": dataset_id,
        "source_records": len(records),
        "split_counts": split_counts(split_rows),
        "label_prevalence": label_prevalence(label_rows),
        "outputs": {
            "source_records": str(dataset_dir / "source_records.parquet"),
            "splits": str(dataset_dir / "splits.parquet"),
            "labels": str(dataset_dir / "labels.parquet"),
        },
    }


def validate_dataset_artifacts(dataset_id: str, output_root: Path) -> dict[str, Any]:
    dataset_dir = output_root / dataset_id
    source_records = read_parquet_rows(dataset_dir / "source_records.parquet")
    split_rows = read_parquet_rows(dataset_dir / "splits.parquet")
    label_rows = read_parquet_rows(dataset_dir / "labels.parquet")
    assert_no_split_leakage(split_rows)
    source_ids = {row["source_record_id"] for row in source_records}
    split_ids = {row["source_record_id"] for row in split_rows}
    label_ids = {row["source_record_id"] for row in label_rows}
    missing = {
        "splits_missing_source_ids": sorted(source_ids - split_ids),
        "labels_missing_source_ids": sorted(source_ids - label_ids),
        "unknown_split_source_ids": sorted(split_ids - source_ids),
        "unknown_label_source_ids": sorted(label_ids - source_ids),
    }
    if any(missing.values()):
        raise AssertionError(f"{dataset_id} artifact row IDs do not align: {missing}")
    return {
        "dataset_id": dataset_id,
        "source_records": len(source_records),
        "split_counts": split_counts(split_rows),
        "label_prevalence": label_prevalence(label_rows),
    }


def registry_dataset_ids(registry_path: Path) -> list[str]:
    registry = read_json(registry_path)
    return [str(dataset["id"]) for dataset in registry["datasets"]]


def resolve_dataset_ids(requested: Iterable[str], registry_path: Path) -> list[str]:
    requested = list(requested)
    if requested == ["all"]:
        return registry_dataset_ids(registry_path)
    valid = set(registry_dataset_ids(registry_path))
    unknown = [dataset_id for dataset_id in requested if dataset_id not in valid]
    if unknown:
        raise ValueError(f"Unknown dataset ids: {unknown}; valid ids are {sorted(valid)}")
    return requested


def dependency_status() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for name in ("datasets", "pandas", "pyarrow"):
        try:
            __import__(name)
        except ModuleNotFoundError:
            status[name] = False
        else:
            status[name] = True
    return status


def build_report(
    *,
    dataset_ids: list[str],
    built: list[dict[str, Any]],
    failed: list[dict[str, str]],
    validated: list[dict[str, Any]] | None = None,
) -> str:
    deps = dependency_status()
    status = "succeeded" if built and not failed else "blocked" if failed else "implemented"
    lines = [
        "# Data Split and Label Report",
        "",
        f"Status: {status}",
        "",
        "## Outputs",
        "",
        "- `artifacts/data/{dataset}/source_records.parquet`",
        "- `artifacts/data/{dataset}/splits.parquet`",
        "- `artifacts/data/{dataset}/labels.parquet`",
        "- `artifacts/reports/data_split_label_report.md`",
        "",
        "## Dependency Status",
        "",
    ]
    for name, present in deps.items():
        lines.append(f"- `{name}`: {'available' if present else 'missing'}")
    lines.extend(["", "## Dataset Status", ""])
    summaries = validated if validated is not None else built
    if summaries:
        for item in summaries:
            lines.append(
                f"- `{item['dataset_id']}`: rows={item['source_records']} "
                f"splits={json_dumps(item['split_counts'])} "
                f"labels={json_dumps(item['label_prevalence']['status_counts'])}"
            )
    else:
        for dataset_id in dataset_ids:
            lines.append(f"- `{dataset_id}`: not built in this environment")
    if failed:
        lines.extend(["", "## Blockers", ""])
        for item in failed:
            lines.append(f"- `{item['dataset_id']}`: {item['error']}")
    else:
        lines.extend(["", "## Blockers", "", "- None reported by the last run."])
    lines.extend(
        [
            "",
            "## Validation Commands",
            "",
            "```bash",
            "python3 -m src.data_split_labels build --dataset all",
            "python3 -m src.data_split_labels validate --dataset all",
            "```",
            "",
            "## Notes",
            "",
            "- Source records are split by a normalized source key so the same question or article cannot appear in more than one split.",
            "- ActMap uses the disjoint train, validation, and test splits.",
            "- CNN/DailyMail label rows remain pending until MiniCheck factuality labeling is complete.",
        ]
    )
    return "\n".join(lines) + "\n"


def cmd_build(args: argparse.Namespace) -> int:
    dataset_ids = resolve_dataset_ids(args.dataset, args.registry)
    built: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for dataset_id in dataset_ids:
        input_jsonl = None
        if args.input_jsonl_dir is not None:
            candidate = args.input_jsonl_dir / f"{dataset_id}.jsonl"
            input_jsonl = candidate if candidate.exists() else None
        try:
            summary = build_dataset_artifacts(
                dataset_id,
                args.output_root,
                input_jsonl=input_jsonl,
                limit=args.limit,
                split_seed=args.split_seed,
            )
        except Exception as exc:
            failed.append({"dataset_id": dataset_id, "error": f"{exc.__class__.__name__}: {exc}"})
            if not args.keep_going:
                break
        else:
            built.append(summary)
            print(f"built {dataset_id}: {summary['source_records']} rows")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(dataset_ids=dataset_ids, built=built, failed=failed), encoding="utf-8")
    if failed:
        for item in failed:
            print(f"ERROR: {item['dataset_id']}: {item['error']}", file=sys.stderr)
        return 1
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    dataset_ids = resolve_dataset_ids(args.dataset, args.registry)
    validated: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for dataset_id in dataset_ids:
        try:
            summary = validate_dataset_artifacts(dataset_id, args.output_root)
        except Exception as exc:
            failed.append({"dataset_id": dataset_id, "error": f"{exc.__class__.__name__}: {exc}"})
            if not args.keep_going:
                break
        else:
            validated.append(summary)
            print(f"validated {dataset_id}: {summary['source_records']} rows")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_report(dataset_ids=dataset_ids, built=[], failed=failed, validated=validated),
        encoding="utf-8",
    )
    if failed:
        for item in failed:
            print(f"ERROR: {item['dataset_id']}: {item['error']}", file=sys.stderr)
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    dataset_ids = resolve_dataset_ids(args.dataset, args.registry)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(dataset_ids=dataset_ids, built=[], failed=[]), encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate ActMap data split and label artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset", nargs="+", default=["all"])
        subparser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
        subparser.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
        subparser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
        subparser.add_argument("--keep-going", action="store_true")

    build = subparsers.add_parser("build", help="Build source, split, and label parquet artifacts.")
    add_common(build)
    build.add_argument("--input-jsonl-dir", type=Path, default=None)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    build.set_defaults(func=cmd_build)

    validate = subparsers.add_parser("validate", help="Validate built data split and label parquet artifacts.")
    add_common(validate)
    validate.set_defaults(func=cmd_validate)

    report = subparsers.add_parser("report", help="Write a data split report without building datasets.")
    add_common(report)
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
