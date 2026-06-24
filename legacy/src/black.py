from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import (
    format_gsm8k_prompt,
    format_nq_open_prompt,
    format_triviaqa_prompt,
    format_web_questions_prompt,
)
from src.vit import make_splits


MODEL_NAMES = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
DATASET_PROMPT_FNS = {
    "triviaqa": format_triviaqa_prompt,
    "nq_open": format_nq_open_prompt,
    "web_questions": format_web_questions_prompt,
    "gsm8k": format_gsm8k_prompt,
}
SCORE_SCHEMA_VERSION = 3
REQUIRED_SCORE_KEYS = {
    "score_schema_version",
    "dataset",
    "mte",
    "p_true",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute black-box UQ AUROC baselines."
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.pt"))
    parser.add_argument("--dataset", choices=tuple(DATASET_PROMPT_FNS), default="triviaqa")
    parser.add_argument("--scores", type=Path, default=Path("data/black_scores.pt"))
    parser.add_argument("--results", type=Path, default=Path("data/black_results.json"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--worker-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--prompt-logprobs", type=int, default=20)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise TypeError(f"Expected {path} to contain a list, got {type(rows).__name__}")
    return rows


def load_scores(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    scores = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(scores, list):
        raise TypeError(f"Expected {path} to contain a list, got {type(scores).__name__}")
    return scores


def save_scores(path: Path, scores: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scores, path)


def split_indices(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[int]:
    if args.split == "all":
        indices = list(range(len(rows)))
    elif args.split_file is not None:
        with args.split_file.open() as handle:
            splits = json.load(handle)
        indices = splits[args.split]
    else:
        train_idx, val_idx, test_idx = make_splits(
            rows,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        indices = {"train": train_idx, "val": val_idx, "test": test_idx}[args.split]

    if args.limit is not None:
        indices = indices[: args.limit]
    return indices


def _logprob_value(entry: Any, token_id: int) -> float | None:
    if entry is None:
        return None
    item = entry.get(token_id)
    if item is None:
        return None
    if hasattr(item, "logprob"):
        return float(item.logprob)
    return float(item)


def _entry_logprobs(entry: Any) -> list[float]:
    if entry is None:
        return []
    values = []
    for item in entry.values():
        values.append(float(item.logprob if hasattr(item, "logprob") else item))
    return values


def _topk_entropy(entry: Any) -> float | None:
    logprobs = np.array(_entry_logprobs(entry), dtype=np.float64)
    if logprobs.size == 0:
        return None
    logprobs = logprobs - logprobs.max()
    probs = np.exp(logprobs)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def _prompt_token_ids(tokenizer: Any, model_name: str, dataset: str, question: str) -> list[int]:
    from src.activations import _prepare_input

    return _prepare_input(
        tokenizer,
        DATASET_PROMPT_FNS[dataset](question),
        no_think=True,
        model_name=model_name,
    )


def _answer_token_variants(tokenizer: Any, answer: str) -> list[tuple[str, list[int]]]:
    answer = str(answer).strip()
    if not answer:
        return []
    variants = [answer, " " + answer, "\n" + answer]
    seen: set[tuple[int, ...]] = set()
    result = []
    for variant in variants:
        token_ids = [int(token_id) for token_id in tokenizer.encode(variant, add_special_tokens=False)]
        key = tuple(token_ids)
        if token_ids and key not in seen:
            seen.add(key)
            result.append((variant, list(token_ids)))
    return result


def _score_answer_tokens_from_prompt_logprobs(
    prompt_lps: list[Any],
    *,
    answer_start: int,
    answer_ids: list[int],
) -> dict[str, float]:

    token_logprobs = []
    entropies = []
    for rel_pos, token_id in enumerate(answer_ids):
        pos = answer_start + rel_pos
        if pos >= len(prompt_lps):
            continue
        entry = prompt_lps[pos]
        token_lp = _logprob_value(entry, token_id)
        entropy = _topk_entropy(entry)
        if token_lp is not None:
            token_logprobs.append(token_lp)
        if entropy is not None:
            entropies.append(entropy)

    variant_score = float(np.mean(token_logprobs)) if token_logprobs else float("nan")
    mte = float(np.mean(entropies)) if entropies else float("nan")
    return {
        "_variant_score": variant_score,
        "mte": mte,
        "answer_tokens": len(token_logprobs),
    }


def _empty_token_mte() -> dict[str, Any]:
    return {
        "_variant_score": float("nan"),
        "mte": float("nan"),
        "answer_tokens": 0,
        "answer_variant": "",
        "answer_variant_tokens": 0,
    }


def score_token_mte_batch(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    dataset: str,
    rows: list[dict[str, Any]],
    prompt_logprobs: int,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    requests: list[tuple[int, str, int, list[int], list[int]]] = []
    candidates: list[list[dict[str, Any]]] = [[] for _row in rows]
    for row_i, row in enumerate(rows):
        prompt_ids = _prompt_token_ids(tokenizer, model_name, dataset, row["question"])
        for variant, answer_ids in _answer_token_variants(tokenizer, row["answer"]):
            requests.append((row_i, variant, len(prompt_ids), answer_ids, prompt_ids + answer_ids))

    if not requests:
        return [_empty_token_mte() for _row in rows]

    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=prompt_logprobs,
        logprobs=1,
    )
    outputs = llm.generate([request[4] for request in requests], params, use_tqdm=False)
    for request, output in zip(requests, outputs):
        row_i, variant, answer_start, answer_ids, _full_ids = request
        scores = _score_answer_tokens_from_prompt_logprobs(
            output.prompt_logprobs or [],
            answer_start=answer_start,
            answer_ids=answer_ids,
        )
        scores["answer_variant"] = variant
        scores["answer_variant_tokens"] = len(answer_ids)
        candidates[row_i].append(scores)

    results = []
    for row_candidates in candidates:
        if not row_candidates:
            results.append(_empty_token_mte())
            continue
        results.append(
            max(
                row_candidates,
                key=lambda item: (
                    item["_variant_score"] if math.isfinite(item["_variant_score"]) else -float("inf"),
                    item["answer_tokens"],
                ),
            )
        )
    return results


def _single_token_ids(tokenizer: Any, variants: list[str]) -> list[int]:
    ids = []
    for text in variants:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 1:
            ids.append(int(token_ids[0]))
    return sorted(set(ids))


def _ptrue_prompt(row: dict[str, Any], dataset: str) -> str:
    if dataset.startswith("gsm8k"):
        intro = "Decide whether the proposed final numeric answer to the math problem is correct."
        question_label = "Problem"
    else:
        intro = "Decide whether the proposed answer to the question is correct."
        question_label = "Question"
    return (
        f"{intro}\n\n"
        f"{question_label}: {row['question']}\n"
        f"Proposed answer: {row['answer']}\n\n"
        "Respond with only True or False.\n"
        "Correct:"
    )


def _ptrue_from_output(output: Any, *, true_ids: list[int], false_ids: list[int]) -> float:
    logprobs = output.outputs[0].logprobs
    if not logprobs:
        return float("nan")
    entry = logprobs[0]

    true_mass = 0.0
    false_mass = 0.0
    for token_id in true_ids:
        lp = _logprob_value(entry, token_id)
        if lp is not None:
            true_mass += math.exp(lp)
    for token_id in false_ids:
        lp = _logprob_value(entry, token_id)
        if lp is not None:
            false_mass += math.exp(lp)

    denom = true_mass + false_mass
    return true_mass / denom if denom > 0 else float("nan")


def score_ptrue_batch(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    dataset: str,
    rows: list[dict[str, Any]],
) -> list[float]:
    from vllm import SamplingParams

    from src.activations import _prepare_input

    true_ids = _single_token_ids(tokenizer, [" True", " true", "True", "true"])
    false_ids = _single_token_ids(tokenizer, [" False", " false", "False", "false"])
    logprob_token_ids = true_ids + false_ids
    if not true_ids or not false_ids or not logprob_token_ids:
        return [float("nan") for _row in rows]

    text_inputs = [
        _prepare_input(tokenizer, _ptrue_prompt(row, dataset), no_think=True, model_name=model_name)
        for row in rows
    ]
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(logprob_token_ids),
        logprob_token_ids=logprob_token_ids,
    )
    outputs = llm.generate(text_inputs, params, use_tqdm=False)
    return [
        _ptrue_from_output(output, true_ids=true_ids, false_ids=false_ids)
        for output in outputs
    ]


def run_worker(args: argparse.Namespace, model_name: str) -> None:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM

    rows = load_rows(args.data)
    indices = [idx for idx in split_indices(args, rows) if rows[idx]["model"] == model_name]
    scores = load_scores(args.scores)
    required_keys = set(REQUIRED_SCORE_KEYS)
    scores = [
        score
        for score in scores
        if (
            score.get("model") != model_name
            or (
                required_keys.issubset(score)
                and score.get("score_schema_version") == SCORE_SCHEMA_VERSION
                and score.get("dataset") == args.dataset
            )
        )
    ]
    done = {
        score["train_index"]
        for score in scores
        if score.get("model") == model_name and score.get("dataset") == args.dataset
    }
    pending = [idx for idx in indices if idx not in done]

    print(f"{model_name}: split rows={len(indices)}, done={len(done)}, pending={len(pending)}")
    if not pending:
        return

    llm = LLM(
        model=model_name,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()

    def append_score(
        idx: int,
        row: dict[str, Any],
        token_mte: dict[str, Any],
        p_true: float,
    ) -> None:
        scores.append(
            {
                "score_schema_version": SCORE_SCHEMA_VERSION,
                "dataset": args.dataset,
                "train_index": idx,
                "model": model_name,
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "answer": row["answer"],
                "is_correct": bool(row["is_correct"]),
                "mte": token_mte["mte"],
                "p_true": p_true,
                "answer_tokens": token_mte["answer_tokens"],
                "answer_variant": token_mte["answer_variant"],
                "answer_variant_tokens": token_mte["answer_variant_tokens"],
            }
        )

    try:
        saved_at = len(scores)
        batch_size = max(1, int(args.score_batch_size))
        total_batches = math.ceil(len(pending) / batch_size)
        for batch_no, start in enumerate(range(0, len(pending), batch_size), start=1):
            batch_indices = pending[start : start + batch_size]
            batch_rows = [rows[idx] for idx in batch_indices]
            print(f"{model_name}: batch {batch_no}/{total_batches} rows={len(batch_rows)}")
            batch_token_mte = score_token_mte_batch(
                llm,
                tokenizer,
                model_name,
                args.dataset,
                batch_rows,
                prompt_logprobs=args.prompt_logprobs,
            )
            batch_p_true = score_ptrue_batch(llm, tokenizer, model_name, args.dataset, batch_rows)
            for idx, row, token_mte, p_true in zip(batch_indices, batch_rows, batch_token_mte, batch_p_true):
                append_score(idx, row, token_mte, p_true)
            if len(scores) - saved_at >= args.save_every:
                save_scores(args.scores, scores)
                saved_at = len(scores)
                print(f"saved {len(scores)} scores to {args.scores}")
    finally:
        del tokenizer
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    save_scores(args.scores, scores)
    print(f"{model_name}: saved {len(scores)} total scores")


def _metric_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    mask = np.isfinite(scores)
    if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
        return None
    return float(roc_auc_score(labels[mask], scores[mask]))


def _metric_auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    mask = np.isfinite(scores)
    if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
        return None
    return float(average_precision_score(labels[mask], scores[mask]))


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    scores = [
        score for score in load_scores(args.scores)
        if score.get("score_schema_version") == SCORE_SCHEMA_VERSION
        and score.get("dataset") == args.dataset
    ]
    if not scores:
        raise RuntimeError(
            f"No schema v{SCORE_SCHEMA_VERSION} scores found at {args.scores}. "
            "Rerun src.black with --overwrite."
        )
    labels = np.array([1 if row["is_correct"] else 0 for row in scores], dtype=np.int64)

    def values(name: str, *, default: float = float("nan")) -> np.ndarray:
        return np.array([row.get(name, default) for row in scores], dtype=np.float64)

    metrics = {
        "mte": -values("mte"),
        "p_true": values("p_true"),
    }

    results: dict[str, Any] = {"n": len(scores), "metrics": {}, "per_model": {}}
    for metric_name, metric_scores in metrics.items():
        results["metrics"][metric_name] = {
            "auroc": _metric_auroc(labels, metric_scores),
            "auprc": _metric_auprc(labels, metric_scores),
        }

    for model_name in sorted({row["model"] for row in scores}):
        idx = np.array([row["model"] == model_name for row in scores])
        model_labels = labels[idx]
        results["per_model"][model_name] = {"n": int(idx.sum()), "metrics": {}}
        for metric_name, metric_scores in metrics.items():
            model_scores = metric_scores[idx]
            results["per_model"][model_name]["metrics"][metric_name] = {
                "auroc": _metric_auroc(model_labels, model_scores),
                "auprc": _metric_auprc(model_labels, model_scores),
            }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w") as handle:
        json.dump(results, handle, indent=2)
    return results


def run_parent(args: argparse.Namespace) -> None:
    if args.overwrite and args.scores.exists():
        args.scores.unlink()
    for model_name in args.models:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.black",
            "--worker-model",
            model_name,
            "--data",
            str(args.data),
            "--dataset",
            args.dataset,
            "--scores",
            str(args.scores),
            "--results",
            str(args.results),
            "--split",
            args.split,
            "--seed",
            str(args.seed),
            "--val-frac",
            str(args.val_frac),
            "--test-frac",
            str(args.test_frac),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--prompt-logprobs",
            str(args.prompt_logprobs),
            "--score-batch-size",
            str(args.score_batch_size),
            "--save-every",
            str(args.save_every),
        ]
        if args.split_file is not None:
            cmd.extend(["--split-file", str(args.split_file)])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        cmd.append("--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code")
        subprocess.run(cmd, check=True)

    results = summarize(args)
    print(json.dumps(results, indent=2))
    print(f"saved scores  -> {args.scores}")
    print(f"saved results -> {args.results}")


def main() -> None:
    args = parse_args()
    if args.worker_model:
        run_worker(args, args.worker_model)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
