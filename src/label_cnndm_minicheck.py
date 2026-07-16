from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .experiment_registry import REPO_ROOT
from .generate_actmaps import model_slug


CNN_DATASET_ID = "cnn_dailymail_3_0_0"
DEFAULT_DATA_ROOT = REPO_ROOT / "artifacts" / "data"
DEFAULT_GENERATION_ROOT = REPO_ROOT / "artifacts" / "generations"
DEFAULT_FACTUALITY_ROOT = REPO_ROOT / "artifacts" / "results" / "factuality"
DEFAULT_OUTPUT_ROOT = DEFAULT_FACTUALITY_ROOT / "minicheck"
DEFAULT_REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "cnndm_factuality_report.md"
DEFAULT_MODEL_PATH: Path | None = None
DEFAULT_MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"
DEFAULT_MODELS = (
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
LABEL_SOURCE = "minicheck_flan_t5_large"
THRESHOLD = 0.5
SUMMARY_COLUMNS = [
    "dataset_id", "model", "model_slug", "source_record_id", "split",
    "label_source", "label_status", "is_correct", "factuality_score",
    "min_support_prob", "mean_support_prob", "max_support_prob",
    "sentence_count", "supported_sentence_count", "unsupported_sentence_count",
    "threshold", "generated_token_count", "generation_hash", "article_hash",
    "claim_splitter", "doc_chunk_size_words", "max_model_len", "shard_index",
    "num_shards", "error",
]
SENTENCE_COLUMNS = [
    "dataset_id", "model", "model_slug", "source_record_id", "split",
    "sentence_index", "claim", "label_source", "label_status", "is_supported",
    "support_prob", "unsupported_prob", "threshold", "chunk_count",
    "best_chunk_index", "generated_token_count", "generation_hash", "article_hash",
    "shard_index", "num_shards", "error",
]
ABBREVIATIONS = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Mt.",
    "Sen.", "Rep.", "Gov.", "Gen.", "Col.", "Lt.", "Capt.", "Sgt.",
    "U.S.", "U.K.", "U.N.", "E.U.", "L.A.", "D.C.", "N.Y.", "vs.", "etc.",
    "e.g.", "i.e.", "No.", "Fig.", "Inc.", "Ltd.", "Co.", "Corp.",
)


@dataclass(frozen=True)
class GenerationRecord:
    dataset_id: str
    model: str
    model_slug: str
    source_record_id: str
    split: str
    generation: str
    generated_token_count: int | None


@dataclass(frozen=True)
class Task:
    record_index: int
    sentence_index: int
    chunk_index: int
    doc_chunk: str
    claim: str


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _print(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    protected = text
    for idx, abbr in enumerate(ABBREVIATIONS):
        token = f"<ABBR{idx}>"
        replacements[token] = abbr
        protected = protected.replace(abbr, abbr.replace(".", token))
    return protected, replacements


def restore_abbreviations(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for token, abbr in replacements.items():
        restored = restored.replace(abbr.replace(".", token), abbr)
    return restored


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text).strip())
    if not text:
        return []
    protected, replacements = protect_abbreviations(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`(])", protected)
    return [restore_abbreviations(part.strip(), replacements) for part in parts if part.strip()]


def chunk_document(document: str, *, chunk_size_words: int) -> list[str]:
    sentences = split_sentences(document) or [str(document).strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = max(1, len(sentence.split()))
        if current and current_words + words > chunk_size_words:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_words = words
        else:
            current.append(sentence)
            current_words += words
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def pd_module():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and pyarrow are required for CNN/DailyMail factuality labels") from exc
    return pd


def write_parquet(path: Path, rows: list[dict[str, Any]], *, columns: list[str]) -> None:
    pd = pd_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    pd = pd_module()
    return pd.read_parquet(path).to_dict(orient="records")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def summary_part_dir(output_root: Path, shard_index: int) -> Path:
    return output_root / "summary_shards" / f"shard_{shard_index:05d}"


def sentence_part_dir(output_root: Path, shard_index: int) -> Path:
    return output_root / "sentence_shards" / f"shard_{shard_index:05d}"


def progress_path(output_root: Path, shard_index: int) -> Path:
    return output_root / "progress" / f"shard_{shard_index:05d}.json"


def part_index(path: Path) -> int:
    match = re.search(r"part_(\d+)\.parquet$", path.name)
    return int(match.group(1)) if match else -1


def part_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("part_*.parquet"), key=part_index)


def next_part_number(output_root: Path, shard_index: int) -> int:
    indices = [part_index(path) for path in part_paths(summary_part_dir(output_root, shard_index))]
    indices = [index for index in indices if index >= 0]
    return max(indices, default=0) + 1


def record_key(model: str, source_record_id: str) -> str:
    return f"{model_slug(model)}||{source_record_id}"


def load_completed_keys(output_root: Path, shard_index: int) -> set[str]:
    completed: set[str] = set()
    for path in part_paths(summary_part_dir(output_root, shard_index)):
        for row in read_parquet_rows(path):
            completed.add(record_key(str(row["model"]), str(row["source_record_id"])))
    return completed


def load_source_articles(data_root: Path) -> dict[str, dict[str, Any]]:
    rows = read_parquet_rows(data_root / CNN_DATASET_ID / "source_records.parquet")
    return {str(row["source_record_id"]): row for row in rows}


def iter_generation_records(generation_root: Path, models: Iterable[str]) -> Iterable[GenerationRecord]:
    for model in models:
        slug = model_slug(model)
        path = generation_root / CNN_DATASET_ID / slug / "records.jsonl"
        if not path.exists():
            _print(f"missing generation records: {path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                source_record_id = str(row.get("source_record_id") or "")
                generation = str(row.get("generation") or "").strip()
                if not source_record_id or not generation:
                    continue
                generated_token_count = row.get("generated_token_count")
                yield GenerationRecord(
                    dataset_id=str(row.get("dataset_id") or CNN_DATASET_ID),
                    model=model,
                    model_slug=slug,
                    source_record_id=source_record_id,
                    split=str(row.get("split") or ""),
                    generation=generation,
                    generated_token_count=(int(generated_token_count) if generated_token_count is not None else None),
                )


def assigned_records(
    *,
    generation_root: Path,
    models: list[str],
    shard_index: int,
    num_shards: int,
    max_records: int | None,
    completed: set[str],
) -> tuple[list[GenerationRecord], int, int]:
    todo: list[GenerationRecord] = []
    assigned_count = 0
    completed_count = 0
    for record in iter_generation_records(generation_root, models):
        if stable_hash(record_key(record.model, record.source_record_id)) % num_shards != shard_index:
            continue
        assigned_count += 1
        if record_key(record.model, record.source_record_id) in completed:
            completed_count += 1
            continue
        todo.append(record)
        if max_records is not None and len(todo) >= max_records:
            break
    return todo, assigned_count, completed_count


def load_model(model_path_or_id: str, *, local_files_only: bool, dtype: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, local_files_only=local_files_only)
    torch_dtype = {
        "auto": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path_or_id,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def score_pairs(
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    pairs: list[tuple[str, str]],
    batch_size: int,
    max_model_len: int,
) -> list[tuple[float, float]]:
    import torch

    results: list[tuple[float, float]] = []
    label_token_ids = torch.tensor([3, 209], device=device)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        texts = ["predict: " + tokenizer.eos_token.join([doc, claim]) for doc, claim in batch]
        encoded = tokenizer(
            texts,
            max_length=max_model_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        decoder_input_ids = torch.zeros((encoded["input_ids"].size(0), 1), dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
        logits = outputs.logits.squeeze(1)[:, label_token_ids]
        probs = torch.softmax(logits.float(), dim=-1).detach().cpu().tolist()
        results.extend((float(prob[0]), float(prob[1])) for prob in probs)
    return results


def summary_failure_row(record: GenerationRecord, *, error: str) -> dict[str, Any]:
    return {
        "dataset_id": CNN_DATASET_ID,
        "model": record.model,
        "model_slug": record.model_slug,
        "source_record_id": record.source_record_id,
        "split": record.split,
        "label_source": LABEL_SOURCE,
        "label_status": "failed",
        "is_correct": None,
        "factuality_score": None,
        "min_support_prob": None,
        "mean_support_prob": None,
        "max_support_prob": None,
        "sentence_count": 0,
        "supported_sentence_count": 0,
        "unsupported_sentence_count": 0,
        "threshold": THRESHOLD,
        "generated_token_count": record.generated_token_count,
        "generation_hash": short_hash(record.generation),
        "article_hash": None,
        "claim_splitter": "regex_abbrev_v1",
        "doc_chunk_size_words": None,
        "max_model_len": None,
        "shard_index": None,
        "num_shards": None,
        "error": error,
    }


def build_tasks_for_records(
    records: list[GenerationRecord],
    source_map: dict[str, dict[str, Any]],
    *,
    chunk_size_words: int,
) -> tuple[list[Task], dict[int, list[str]], list[dict[str, Any]]]:
    tasks: list[Task] = []
    sentence_map: dict[int, list[str]] = {}
    failed_summaries: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        source = source_map.get(record.source_record_id)
        if source is None:
            failed_summaries.append(summary_failure_row(record, error="missing_source_article"))
            continue
        sentences = split_sentences(record.generation)
        sentence_map[record_index] = sentences
        if not sentences:
            failed_summaries.append(summary_failure_row(record, error="no_claim_sentences"))
            continue
        chunks = chunk_document(str(source.get("article") or ""), chunk_size_words=chunk_size_words)
        if not chunks:
            failed_summaries.append(summary_failure_row(record, error="empty_source_article"))
            continue
        for sentence_index, sentence in enumerate(sentences):
            for chunk_index, chunk in enumerate(chunks):
                tasks.append(Task(record_index, sentence_index, chunk_index, chunk, sentence))
    return tasks, sentence_map, failed_summaries


def aggregate_group(
    *,
    records: list[GenerationRecord],
    source_map: dict[str, dict[str, Any]],
    tasks: list[Task],
    sentence_map: dict[int, list[str]],
    scores: list[tuple[float, float]],
    threshold: float,
    shard_index: int,
    num_shards: int,
    chunk_size_words: int,
    max_model_len: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_sentence: dict[tuple[int, int], list[tuple[int, float, float]]] = defaultdict(list)
    for task, (unsupported_prob, support_prob) in zip(tasks, scores, strict=True):
        by_sentence[(task.record_index, task.sentence_index)].append((task.chunk_index, unsupported_prob, support_prob))

    summary_rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        source = source_map.get(record.source_record_id)
        if source is None:
            continue
        article = str(source.get("article") or "")
        article_hash = short_hash(article)
        generation_hash = short_hash(record.generation)
        sentence_probs: list[float] = []
        sentences = sentence_map.get(record_index, [])
        for sentence_index, sentence in enumerate(sentences):
            chunk_scores = by_sentence.get((record_index, sentence_index), [])
            if not chunk_scores:
                continue
            best_chunk_index, unsupported_prob, support_prob = max(chunk_scores, key=lambda item: item[2])
            sentence_probs.append(support_prob)
            sentence_rows.append(
                {
                    "dataset_id": CNN_DATASET_ID,
                    "model": record.model,
                    "model_slug": record.model_slug,
                    "source_record_id": record.source_record_id,
                    "split": record.split,
                    "sentence_index": sentence_index,
                    "claim": sentence,
                    "label_source": LABEL_SOURCE,
                    "label_status": "evaluated",
                    "is_supported": bool(support_prob > threshold),
                    "support_prob": support_prob,
                    "unsupported_prob": unsupported_prob,
                    "threshold": threshold,
                    "chunk_count": len(chunk_scores),
                    "best_chunk_index": best_chunk_index,
                    "generated_token_count": record.generated_token_count,
                    "generation_hash": generation_hash,
                    "article_hash": article_hash,
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "error": None,
                }
            )
        if sentence_probs:
            supported = sum(1 for prob in sentence_probs if prob > threshold)
            unsupported = len(sentence_probs) - supported
            min_prob = min(sentence_probs)
            summary_rows.append(
                {
                    "dataset_id": CNN_DATASET_ID,
                    "model": record.model,
                    "model_slug": record.model_slug,
                    "source_record_id": record.source_record_id,
                    "split": record.split,
                    "label_source": LABEL_SOURCE,
                    "label_status": "evaluated",
                    "is_correct": bool(unsupported == 0),
                    "factuality_score": min_prob,
                    "min_support_prob": min_prob,
                    "mean_support_prob": sum(sentence_probs) / len(sentence_probs),
                    "max_support_prob": max(sentence_probs),
                    "sentence_count": len(sentence_probs),
                    "supported_sentence_count": supported,
                    "unsupported_sentence_count": unsupported,
                    "threshold": threshold,
                    "generated_token_count": record.generated_token_count,
                    "generation_hash": generation_hash,
                    "article_hash": article_hash,
                    "claim_splitter": "regex_abbrev_v1",
                    "doc_chunk_size_words": chunk_size_words,
                    "max_model_len": max_model_len,
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "error": None,
                }
            )
    return summary_rows, sentence_rows


def write_part(
    *,
    output_root: Path,
    shard_index: int,
    part_number: int,
    summary_rows: list[dict[str, Any]],
    sentence_rows: list[dict[str, Any]],
) -> None:
    summary_path = summary_part_dir(output_root, shard_index) / f"part_{part_number:06d}.parquet"
    sentence_path = sentence_part_dir(output_root, shard_index) / f"part_{part_number:06d}.parquet"
    write_parquet(summary_path, summary_rows, columns=SUMMARY_COLUMNS)
    write_parquet(sentence_path, sentence_rows, columns=SENTENCE_COLUMNS)


def write_progress(*, output_root: Path, shard_index: int, payload: dict[str, Any]) -> None:
    write_json(progress_path(output_root, shard_index), payload)


def run_shard(args: argparse.Namespace) -> int:
    started = time.time()
    source_map = load_source_articles(args.data_root)
    completed = load_completed_keys(args.output_root, args.shard_index)
    records, assigned_count, completed_count = assigned_records(
        generation_root=args.generation_root,
        models=args.models,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_records=args.max_records,
        completed=completed,
    )
    _print(f"shard={args.shard_index}/{args.num_shards} assigned={assigned_count} already_completed={completed_count} todo={len(records)}")
    write_progress(
        output_root=args.output_root,
        shard_index=args.shard_index,
        payload={"status": "loading_model", "shard_index": args.shard_index, "num_shards": args.num_shards, "assigned_records": assigned_count, "completed_existing": completed_count, "todo_records": len(records), "processed_records": 0, "updated_at": _now()},
    )
    if not records:
        return 0

    if args.minicheck_model_path is not None:
        if not args.minicheck_model_path.exists():
            raise FileNotFoundError(f"MiniCheck model path does not exist: {args.minicheck_model_path}")
        model_path_or_id = str(args.minicheck_model_path)
    else:
        model_path_or_id = str(args.minicheck_model_id)
    local_files_only = bool(args.local_files_only)
    tokenizer, model, device = load_model(model_path_or_id, local_files_only=local_files_only, dtype=args.dtype)
    _print(f"loaded MiniCheck model={model_path_or_id} device={device} batch_size={args.batch_size}")

    part_number = next_part_number(args.output_root, args.shard_index)
    pending_records: list[GenerationRecord] = []
    pending_tasks: list[Task] = []
    pending_sentence_map: dict[int, list[str]] = {}
    summary_buffer: list[dict[str, Any]] = []
    sentence_buffer: list[dict[str, Any]] = []
    processed = 0
    scored_pairs = 0
    errors = 0

    def flush_group() -> None:
        nonlocal pending_records, pending_tasks, pending_sentence_map, summary_buffer, sentence_buffer, processed, scored_pairs, errors
        if not pending_records:
            return
        try:
            pairs = [(task.doc_chunk, task.claim) for task in pending_tasks]
            scores = score_pairs(tokenizer=tokenizer, model=model, device=device, pairs=pairs, batch_size=args.batch_size, max_model_len=args.max_model_len) if pairs else []
            summaries, sentences = aggregate_group(
                records=pending_records,
                source_map=source_map,
                tasks=pending_tasks,
                sentence_map=pending_sentence_map,
                scores=scores,
                threshold=args.threshold,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                chunk_size_words=args.chunk_size_words,
                max_model_len=args.max_model_len,
            )
            summary_buffer.extend(summaries)
            sentence_buffer.extend(sentences)
            scored_pairs += len(pairs)
        except Exception as exc:
            errors += len(pending_records)
            _print(f"ERROR group failed records={len(pending_records)} error={exc.__class__.__name__}: {exc}")
            for record in pending_records:
                row = summary_failure_row(record, error=f"{exc.__class__.__name__}: {exc}")
                row["shard_index"] = args.shard_index
                row["num_shards"] = args.num_shards
                row["doc_chunk_size_words"] = args.chunk_size_words
                row["max_model_len"] = args.max_model_len
                summary_buffer.append(row)
        processed += len(pending_records)
        pending_records = []
        pending_tasks = []
        pending_sentence_map = {}

    for record in records:
        tasks, sentence_map, failures = build_tasks_for_records([record], source_map, chunk_size_words=args.chunk_size_words)
        summary_buffer.extend(failures)
        if tasks:
            offset = len(pending_records)
            pending_records.append(record)
            pending_sentence_map[offset] = sentence_map.get(0, [])
            pending_tasks.extend(Task(offset, task.sentence_index, task.chunk_index, task.doc_chunk, task.claim) for task in tasks)
        else:
            processed += 1
        if len(pending_tasks) >= args.batch_size or len(pending_records) >= args.record_group_size:
            flush_group()
        if len(summary_buffer) >= args.save_every_records:
            write_part(output_root=args.output_root, shard_index=args.shard_index, part_number=part_number, summary_rows=summary_buffer, sentence_rows=sentence_buffer)
            elapsed = max(time.time() - started, 1e-6)
            rate = processed / elapsed
            _print(f"shard={args.shard_index} wrote part={part_number:06d} processed={processed}/{len(records)} rate={rate:.2f} rec/s scored_pairs={scored_pairs}")
            write_progress(
                output_root=args.output_root,
                shard_index=args.shard_index,
                payload={"status": "running", "shard_index": args.shard_index, "num_shards": args.num_shards, "assigned_records": assigned_count, "completed_existing": completed_count, "todo_records": len(records), "processed_records": processed, "scored_pairs": scored_pairs, "errors": errors, "rate_records_per_second": rate, "last_part": part_number, "updated_at": _now()},
            )
            part_number += 1
            summary_buffer = []
            sentence_buffer = []
    flush_group()
    if summary_buffer or sentence_buffer:
        write_part(output_root=args.output_root, shard_index=args.shard_index, part_number=part_number, summary_rows=summary_buffer, sentence_rows=sentence_buffer)
        _print(f"shard={args.shard_index} wrote final part={part_number:06d}")
    elapsed = max(time.time() - started, 1e-6)
    write_progress(
        output_root=args.output_root,
        shard_index=args.shard_index,
        payload={"status": "done", "shard_index": args.shard_index, "num_shards": args.num_shards, "assigned_records": assigned_count, "completed_existing": completed_count, "todo_records": len(records), "processed_records": processed, "scored_pairs": scored_pairs, "errors": errors, "elapsed_seconds": elapsed, "rate_records_per_second": processed / elapsed, "updated_at": _now()},
    )
    _print(f"shard={args.shard_index} done processed={processed} scored_pairs={scored_pairs} elapsed={elapsed:.1f}s")
    return 0


def read_parts(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("shard_*/part_*.parquet")):
        rows.extend(read_parquet_rows(path))
    return rows


def dedupe_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_key[record_key(str(row["model"]), str(row["source_record_id"]))] = row
    return sorted(by_key.values(), key=lambda row: (str(row["model_slug"]), str(row["source_record_id"])))


def dedupe_sentence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (record_key(str(row["model"]), str(row["source_record_id"])), str(row.get("generation_hash")), int(row["sentence_index"]))
        by_key[key] = row
    return sorted(by_key.values(), key=lambda row: (str(row["model_slug"]), str(row["source_record_id"]), int(row["sentence_index"])))


def build_report(summary_rows: list[dict[str, Any]], sentence_rows: list[dict[str, Any]], *, output_root: Path) -> str:
    status_counts = Counter(str(row.get("label_status")) for row in summary_rows)
    model_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in summary_rows:
        model_counts[str(row["model_slug"])][str(row.get("is_correct"))] += 1
        split_counts[str(row["split"])][str(row.get("is_correct"))] += 1
    lines = [
        "# CNN/DailyMail MiniCheck Factuality Report", "", f"Status: {'completed' if summary_rows else 'empty'}.", "",
        "## Outputs", "", f"- `{output_root / 'summary_labels.parquet'}`", f"- `{output_root / 'sentence_labels.parquet'}`", f"- `{output_root.parent / 'minicheck.parquet'}`", "",
        "## Summary", "", f"- Summary rows: {len(summary_rows)}", f"- Sentence rows: {len(sentence_rows)}", f"- Label source: `{LABEL_SOURCE}`", f"- Summary label: factual iff every generated sentence has MiniCheck support probability > {THRESHOLD}.", "",
        "## Status Counts", "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## By Model", "", "| Model | Factual | Non-Factual | Missing |", "|---|---:|---:|---:|"])
    for model, counts in sorted(model_counts.items()):
        lines.append(f"| `{model}` | {counts['True']} | {counts['False']} | {counts['None']} |")
    lines.extend(["", "## By Split", "", "| Split | Factual | Non-Factual | Missing |", "|---|---:|---:|---:|"])
    for split, counts in sorted(split_counts.items()):
        lines.append(f"| `{split}` | {counts['True']} | {counts['False']} | {counts['None']} |")
    lines.extend(["", "## Notes", "", "- Raw generation JSONL files are not modified.", "- Sentence-level rows keep the individual MiniCheck claims for audit.", ""])
    return "\n".join(lines)


def merge_outputs(args: argparse.Namespace) -> int:
    summary_rows = dedupe_summary(read_parts(args.output_root / "summary_shards"))
    sentence_rows = dedupe_sentence(read_parts(args.output_root / "sentence_shards"))
    write_parquet(args.output_root / "summary_labels.parquet", summary_rows, columns=SUMMARY_COLUMNS)
    write_parquet(args.output_root / "sentence_labels.parquet", sentence_rows, columns=SENTENCE_COLUMNS)
    write_parquet(args.output_root.parent / "minicheck.parquet", summary_rows, columns=SUMMARY_COLUMNS)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(build_report(summary_rows, sentence_rows, output_root=args.output_root), encoding="utf-8")
    print(f"wrote {args.output_root / 'summary_labels.parquet'} rows={len(summary_rows)}")
    print(f"wrote {args.output_root / 'sentence_labels.parquet'} rows={len(sentence_rows)}")
    print(f"wrote {args.output_root.parent / 'minicheck.parquet'} rows={len(summary_rows)}")
    print(f"wrote {args.report_path}")
    return 0


def status(args: argparse.Namespace) -> int:
    progress_dir = args.output_root / "progress"
    if not progress_dir.exists():
        print(f"no progress directory: {progress_dir}")
        return 1
    for path in sorted(progress_dir.glob("shard_*.json")):
        print(json.dumps(json.loads(path.read_text()), sort_keys=True))
    if (args.output_root / "summary_labels.parquet").exists():
        rows = read_parquet_rows(args.output_root / "summary_labels.parquet")
        print(f"merged_summary_rows={len(rows)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label CNN/DailyMail generated summaries with MiniCheck Flan-T5 Large.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        subparser.add_argument("--generation-root", type=Path, default=DEFAULT_GENERATION_ROOT)
        subparser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        subparser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
        subparser.add_argument("--model", dest="models", action="append", default=[])

    run = subparsers.add_parser("run-shard", help="Run one deterministic shard of MiniCheck labels.")
    common(run)
    run.add_argument("--minicheck-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    run.add_argument("--minicheck-model-id", default=DEFAULT_MODEL_ID)
    run.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--num-shards", type=int, required=True)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--record-group-size", type=int, default=32)
    run.add_argument("--save-every-records", type=int, default=512)
    run.add_argument("--max-records", type=int, default=None)
    run.add_argument("--threshold", type=float, default=THRESHOLD)
    run.add_argument("--max-model-len", type=int, default=2048)
    run.add_argument("--chunk-size-words", type=int, default=500)
    run.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    run.set_defaults(func=run_shard)

    merge = subparsers.add_parser("merge", help="Merge shard parts into final MiniCheck parquet artifacts.")
    common(merge)
    merge.set_defaults(func=merge_outputs)

    stat = subparsers.add_parser("status", help="Print shard progress JSON files.")
    common(stat)
    stat.set_defaults(func=status)

    args = parser.parse_args(argv)
    if hasattr(args, "models"):
        args.models = args.models or list(DEFAULT_MODELS)
    if getattr(args, "num_shards", 1) < 1:
        raise ValueError("--num-shards must be >= 1")
    if getattr(args, "shard_index", 0) < 0 or getattr(args, "shard_index", 0) >= getattr(args, "num_shards", 1):
        raise ValueError("--shard-index must be in [0, num_shards)")
    if getattr(args, "batch_size", 1) < 1:
        raise ValueError("--batch-size must be >= 1")
    if getattr(args, "save_every_records", 1) < 1:
        raise ValueError("--save-every-records must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
