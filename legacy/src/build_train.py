from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.evaluation import (
    evaluate_gsm8k_answer,
    evaluate_nq_open_answer,
    evaluate_triviaqa_answer,
    evaluate_web_questions_answer,
    format_gsm8k_prompt,
    format_nq_open_prompt,
    format_triviaqa_prompt,
    format_web_questions_prompt,
)


DATASET_CONFIGS = {
    "triviaqa": {
        "prompt_fn": format_triviaqa_prompt,
        "eval_fn": evaluate_triviaqa_answer,
        "default_input": Path("data/triviaqa_10k.csv"),
        "default_output": Path("data/train.pt"),
        "default_max_new_tokens": 32,
    },
    "nq_open": {
        "prompt_fn": format_nq_open_prompt,
        "eval_fn": evaluate_nq_open_answer,
        "default_input": Path("data/nq_open.csv"),
        "default_output": Path("data/train_nq_open.pt"),
        "default_max_new_tokens": 32,
    },
    "web_questions": {
        "prompt_fn": format_web_questions_prompt,
        "eval_fn": evaluate_web_questions_answer,
        "default_input": Path("data/web_questions.csv"),
        "default_output": Path("data/train_web_questions.pt"),
        "default_max_new_tokens": 32,
    },
    "gsm8k": {
        "prompt_fn": format_gsm8k_prompt,
        "eval_fn": evaluate_gsm8k_answer,
        "default_input": Path("data/gsm8k.csv"),
        "default_output": Path("data/train_gsm8k.pt"),
        "default_max_new_tokens": 64,
    },
}

MODEL_NAMES = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ActMap training rows."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS),
        default="triviaqa",
        help="Dataset to process.",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--worker-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--actmap-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--normalize-actmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each ActMap before saving.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output file.",
    )
    return parser.parse_args()


def iter_rows(path: Path, *, offset: int = 0, limit: int | None = None):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        yielded = 0
        for index, row in enumerate(reader):
            if index < offset:
                continue
            if limit is not None and yielded >= limit:
                break
            yielded += 1
            yield index, row


def save_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)


def load_existing_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    import torch

    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise TypeError(f"Expected {path} to contain a list, found {type(rows).__name__}")
    return rows


def load_model(args: argparse.Namespace, model_name: str):
    from vllm import LLM

    llm = LLM(
        model=model_name,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def release_model(llm: Any, tokenizer: Any | None = None) -> None:
    import gc

    import torch

    try:
        from src.activations import probe_remove_hooks

        llm.apply_model(probe_remove_hooks)
    except Exception as exc:
        print(f"[warn] could not remove vLLM hooks cleanly: {exc}")

    for attr_path in (
        ("llm_engine", "shutdown"),
        ("llm_engine", "engine_core", "shutdown"),
        ("llm_engine", "engine_core", "close"),
    ):
        obj = llm
        for attr in attr_path[:-1]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        method = getattr(obj, attr_path[-1], None) if obj is not None else None
        if callable(method):
            try:
                method()
            except Exception as exc:
                print(f"[warn] vLLM teardown method {'.'.join(attr_path)} failed: {exc}")

    del tokenizer
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_worker(args: argparse.Namespace, model_name: str) -> None:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    import torch

    from src.activations import (
        build_actmappp,
        extract_generation_hidden_states_vllm,
        normalize_actmap,
        probe_setup_hooks,
    )

    actmap_dtype = torch.float16 if args.actmap_dtype == "float16" else torch.float32

    dataset = load_existing_dataset(args.output)
    existing_for_model = sum(row["model"] == model_name for row in dataset)
    target_rows = args.limit if args.limit is not None else sum(
        1 for _source_index, _row in iter_rows(args.input, offset=args.offset, limit=None)
    )
    if existing_for_model >= target_rows:
        print(f"{model_name}: already has {existing_for_model}/{target_rows} rows, skipping")
        return

    print(f"\n=== model: {model_name} ===")
    print(f"resuming {model_name} from row count {existing_for_model}/{target_rows}")
    llm = None
    tokenizer = None
    model_completed = existing_for_model
    model_failures = 0
    try:
        llm, tokenizer = load_model(args, model_name)
        num_layers = llm.apply_model(probe_setup_hooks)[0]
        print(f"{model_name}: hooked layers={num_layers}")

        for local_index, (source_index, row) in enumerate(
            iter_rows(args.input, offset=args.offset, limit=args.limit)
        ):
            if local_index < existing_for_model:
                continue

            prompt = args._prompt_fn(row["question"])

            try:
                result = extract_generation_hidden_states_vllm(
                    llm,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    no_think=True,
                    model_name=model_name,
                    logprobs=0,
                )
                if result is None:
                    raise RuntimeError("empty generation or missing hidden states")

                answer, hidden_states, _token_ids, _logprobs = result
                actmap = build_actmappp(hidden_states)
                if args.normalize_actmap:
                    actmap = normalize_actmap(actmap)
                eval_result = args._eval_fn(answer, row)
                training_row = {
                    "question": row["question"],
                    "ground_truth": eval_result.get(
                        "gold_normalized",
                        row.get("answer_value", row.get("answer_normalized", "")),
                    ),
                    "answer": eval_result["prediction_clean"],
                    "model": model_name,
                    "is_correct": eval_result["is_correct"],
                    "actmap": actmap.detach().cpu() if hasattr(actmap, "detach") else actmap,
                }
                if row.get("question_id"):
                    training_row["question_id"] = row["question_id"]
                training_row["actmap"] = training_row["actmap"].to(dtype=actmap_dtype)
                dataset.append(training_row)
                model_completed += 1
            except Exception as exc:
                model_failures += 1
                print(f"[warn] {model_name} row {source_index} failed: {exc}")
                continue

            if len(dataset) % args.save_every == 0:
                save_dataset(args.output, dataset)
                print(
                    f"saved {len(dataset)} total rows to {args.output} "
                    f"(model_completed={model_completed}, model_failures={model_failures})"
                )
    finally:
        if llm is not None:
            release_model(llm, tokenizer)

    save_dataset(args.output, dataset)
    print(
        f"finished {model_name}: completed={model_completed}, "
        f"model_failures={model_failures}, total_rows={len(dataset)}"
    )


def run_parent(args: argparse.Namespace) -> None:
    if args.overwrite and args.output.exists():
        args.output.unlink()

    target_rows_per_model = args.limit if args.limit is not None else "all"
    print(
        f"target: {target_rows_per_model} questions x {len(args.models)} models | "
        f"output: {args.output}"
    )

    for model_name in args.models:
        cmd = [
            sys.executable,
            "-m",
            "src.build_train",
            "--worker-model",
            model_name,
            "--input",
            str(args.input),
            "--output",
            str(args.output),
            "--offset",
            str(args.offset),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--save-every",
            str(args.save_every),
            "--actmap-dtype",
            args.actmap_dtype,
        ]
        cmd.append("--normalize-actmap" if args.normalize_actmap else "--no-normalize-actmap")
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        cmd.extend(["--dataset", args.dataset])
        cmd.append("--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code")

        print(f"\nstarting worker for {model_name}")
        subprocess.run(cmd, check=True)
        print(f"worker exited for {model_name}")

    dataset = load_existing_dataset(args.output)
    print(f"\ndone: saved {len(dataset)} rows to {args.output}")


def _resolve_dataset(args: argparse.Namespace) -> None:
    cfg = DATASET_CONFIGS[args.dataset]
    if args.input is None:
        args.input = cfg["default_input"]
    if args.output is None:
        args.output = cfg["default_output"]
    if args.max_new_tokens is None:
        args.max_new_tokens = int(cfg["default_max_new_tokens"])
    args._prompt_fn = cfg["prompt_fn"]
    args._eval_fn = cfg["eval_fn"]


def main() -> None:
    args = parse_args()
    _resolve_dataset(args)
    if args.worker_model:
        run_worker(args, args.worker_model)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
