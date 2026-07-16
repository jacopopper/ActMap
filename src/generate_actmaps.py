from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_split_labels import evaluate_generation_correctness
from .experiment_registry import DEFAULT_REGISTRY_PATH, REPO_ROOT, read_json


DEFAULT_DATA_ROOT = REPO_ROOT / "artifacts" / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "features" / "actmap"
DEFAULT_GENERATION_ROOT = REPO_ROOT / "artifacts" / "generations"
PROBE_PICKLE_MODULE = "src.generate_actmaps"

if __name__ == "__main__":
    sys.modules.setdefault(PROBE_PICKLE_MODULE, sys.modules[__name__])

MODEL_FAMILIES = {
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen3-32B": "qwen3",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
}


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model).strip("_")


def model_family(model: str) -> str:
    try:
        return MODEL_FAMILIES[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model id: {model!r}") from exc


def max_new_tokens_for_dataset(dataset_id: str, *, gsm8k_output_mode: str = "rationale") -> int:
    if dataset_id == "gsm8k_rationale":
        return 32 if gsm8k_output_mode == "direct" else 128
    if dataset_id == "cnn_dailymail_3_0_0":
        return 384
    return 32


def prompt_for_record(
    dataset_id: str,
    row: dict[str, Any],
    *,
    gsm8k_output_mode: str = "rationale",
) -> str:
    question = str(row.get("question") or "")
    if dataset_id == "triviaqa_no_context":
        return (
            "Answer the trivia question with the shortest correct answer. "
            "Output only the answer, not a sentence.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    if dataset_id == "nq_open":
        return (
            "Answer the question with a short, factual answer. "
            "Output only the answer, not a sentence.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    if dataset_id == "gsm8k_rationale":
        if gsm8k_output_mode == "direct":
            return (
                "Solve the grade-school math word problem. "
                "Output only the final numeric answer, without units, explanation, or punctuation.\n\n"
                f"Question: {question}\n"
                "Answer:"
            )
        return (
            "Solve the grade-school math word problem. "
            "Return the reasoning briefly, then end with 'Final answer: <number>'.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    if dataset_id == "cnn_dailymail_3_0_0":
        article = str(row.get("article") or row.get("input_text") or "")
        return (
            "Write a concise, factual summary of the news article. "
            "Do not add facts that are not supported by the article.\n\n"
            f"Article:\n{article}\n\n"
            "Summary:"
        )
    raise KeyError(f"Unknown dataset_id: {dataset_id}")


def _coerce_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping) or hasattr(tokenized, "keys"):
        keys = set(tokenized.keys())
        if "input_ids" in keys:
            tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    elif not isinstance(tokenized, list):
        tokenized = list(tokenized)
    if tokenized and hasattr(tokenized[0], "tolist"):
        tokenized = tokenized[0].tolist()
    if tokenized and isinstance(tokenized[0], (list, tuple)):
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def _prepare_input(tokenizer: Any, prompt: str, *, no_think: bool, model_name: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if no_think and model_family(model_name) == "qwen3":
        template_kwargs["enable_thinking"] = False
    return _coerce_token_ids(tokenizer.apply_chat_template(messages, **template_kwargs))


def _probe_get_layers(model: Any) -> list[Any]:
    for attr_path in ("model.layers", "model.model.layers", "transformer.h", "layers"):
        obj = model
        found = True
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                found = False
                break
        if found and hasattr(obj, "__len__") and len(obj) > 0:
            return list(obj)
    raise RuntimeError(f"Cannot find transformer layers in {type(model).__name__}")


def probe_setup_hooks(model: Any, *, batch_size: int = 1, target_dim: int = 128) -> int:
    target_dim = int(os.environ.get("ACTMAP_PROBE_TARGET_DIM", target_dim))
    if target_dim < 1:
        raise ValueError(f"ACTMAP_PROBE_TARGET_DIM must be positive, got {target_dim}")
    layers = _probe_get_layers(model)
    model._probe_steps = [[] for _ in range(len(layers))]
    model._probe_hooks = []
    model._probe_batch_size = batch_size
    model._probe_target_dim = target_dim
    for layer_idx, layer in enumerate(layers):
        def _make_hook(idx: int):
            def _hook(_module: Any, _inp: Any, out: Any) -> None:
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[0] <= model._probe_batch_size:
                    import torch.nn.functional as F

                    h = h.detach().float()
                    if h.shape[-1] != model._probe_target_dim:
                        original_shape = h.shape
                        h = F.adaptive_avg_pool1d(
                            h.reshape(-1, 1, original_shape[-1]),
                            model._probe_target_dim,
                        )
                        h = h.reshape(*original_shape[:-1], model._probe_target_dim)
                    model._probe_steps[idx].append(h.cpu())
            return _hook

        model._probe_hooks.append(layer.register_forward_hook(_make_hook(layer_idx)))
    return len(layers)


def probe_reset(model: Any) -> None:
    if hasattr(model, "_probe_steps"):
        model._probe_steps = [[] for _ in range(len(model._probe_steps))]


def probe_collect(model: Any):
    import torch

    if not hasattr(model, "_probe_steps"):
        return None
    result = []
    for layer_steps in model._probe_steps:
        if not layer_steps:
            return None
        result.append(torch.cat(layer_steps, dim=0))
    return torch.stack(result, dim=0) if result else None


def probe_remove_hooks(model: Any) -> None:
    for handle in getattr(model, "_probe_hooks", []):
        handle.remove()
    for attr in ("_probe_hooks", "_probe_steps", "_probe_batch_size", "_probe_target_dim"):
        if hasattr(model, attr):
            delattr(model, attr)


def _make_probe_functions_pickleable() -> None:
    for func in (probe_setup_hooks, probe_reset, probe_collect, probe_remove_hooks):
        func.__module__ = PROBE_PICKLE_MODULE


_make_probe_functions_pickleable()


def build_actmap(
    hidden_states: Any,
    *,
    target_dim: int = 128,
    target_layers: int = 32,
    segment_count: int = 4,
):
    import torch
    import torch.nn.functional as F

    if target_dim < 1 or target_layers < 1 or segment_count < 1:
        raise ValueError(
            "target_dim, target_layers, and segment_count must all be positive; "
            f"got {target_dim}, {target_layers}, {segment_count}"
        )
    acts = hidden_states.float()
    layers, tokens, hidden_dim = acts.shape
    if tokens < 1:
        raise ValueError("cannot build an ActMap from an empty answer trajectory")
    if hidden_dim != target_dim:
        acts = F.adaptive_avg_pool1d(acts.reshape(layers * tokens, 1, hidden_dim), target_dim)
        acts = acts.reshape(layers, tokens, target_dim)

    channels = []
    boundaries = torch.linspace(0, tokens, segment_count + 1).long().tolist()
    for idx in range(segment_count):
        start = min(int(boundaries[idx]), tokens - 1)
        end = min(max(int(boundaries[idx + 1]), start + 1), tokens)
        channels.append(acts[:, start:end, :].mean(dim=1))
    channels.append(acts[:, -1, :])
    channels.append(acts[:, max(0, tokens - 8):, :].mean(dim=1))
    channels.append(acts.std(dim=1) if tokens >= 2 else torch.zeros(layers, target_dim))
    channels.append(acts.max(dim=1).values)
    channels.append(acts[:, -1, :] - acts[:, 0, :])
    if tokens >= 2:
        t_idx = torch.arange(tokens, dtype=acts.dtype, device=acts.device)
        centered_t = t_idx - t_idx.mean()
        centered_acts = acts - acts.mean(dim=1, keepdim=True)
        channels.append(torch.einsum("ltd,t->ld", centered_acts, centered_t) / (centered_t.square().sum().clamp(min=1e-8)))
    else:
        channels.append(torch.zeros(layers, target_dim))
    denom = torch.tensor(tokens * target_dim, dtype=acts.dtype, device=acts.device).sqrt()
    layer_norm = acts.reshape(layers, -1).norm(p=2, dim=1, keepdim=True) / denom.clamp(min=1e-8)
    channels.append(layer_norm.expand(layers, target_dim))
    channels.append((acts[:, 1:, :] - acts[:, :-1, :]).abs().mean(dim=1) if tokens >= 2 else torch.zeros(layers, target_dim))

    stacked = torch.stack(channels, dim=0)
    pooled = F.adaptive_avg_pool1d(stacked.permute(0, 2, 1), target_layers)
    return pooled.permute(0, 2, 1)


def normalize_actmap(actmap: Any, *, eps: float = 1e-6):
    if actmap.ndim != 3:
        raise ValueError(f"Expected [C, L, D] ActMap, got shape {tuple(actmap.shape)}")
    actmap = actmap.float()
    mean = actmap.mean(dim=(1, 2), keepdim=True)
    std = actmap.std(dim=(1, 2), keepdim=True).clamp(min=eps)
    return (actmap - mean) / std


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and pyarrow are required to read dataset parquet artifacts") from exc
    rows = pd.read_parquet(path).to_dict(orient="records")
    cleaned = []
    for row in rows:
        cleaned.append({key: (None if pd.isna(value) else value) for key, value in row.items()})
    return cleaned


def load_split_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = load_parquet_rows(path)
    return {str(row["source_record_id"]): str(row["split"]) for row in rows}


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import torch

    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise TypeError(f"Expected {path} to contain a list, got {type(rows).__name__}")
    return rows


def save_torch_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(rows, tmp_path)
    tmp_path.replace(path)


def record_shards_dir(output_dir: Path) -> Path:
    return output_dir / "record_shards"


def record_shard_path(shards_dir: Path, shard_index: int) -> Path:
    return shards_dir / f"records_{shard_index:06d}.pt"


def record_shard_index(path: Path) -> int:
    match = re.search(r"records_(\d+)\.pt$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def iter_record_shards(shards_dir: Path) -> list[Path]:
    if not shards_dir.exists():
        return []
    return sorted(shards_dir.glob("records_*.pt"), key=record_shard_index)


def load_saved_rows(records_path: Path, shards_dir: Path) -> list[dict[str, Any]]:
    rows = load_existing_rows(records_path)
    completed = {str(row["source_record_id"]) for row in rows}
    for shard_path in iter_record_shards(shards_dir):
        for row in load_existing_rows(shard_path):
            source_record_id = str(row["source_record_id"])
            if source_record_id not in completed:
                rows.append(row)
                completed.add(source_record_id)
    return rows


def next_record_shard_index(shards_dir: Path) -> int:
    indices = [record_shard_index(path) for path in iter_record_shards(shards_dir)]
    indices = [index for index in indices if index >= 0]
    return max(indices, default=0) + 1


def save_record_shard(shards_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> Path:
    path = record_shard_path(shards_dir, shard_index)
    save_torch_rows(path, rows)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_generation(llm: Any, tokenizer: Any, prompt: str, args: argparse.Namespace):
    from vllm import SamplingParams

    token_ids = _prepare_input(tokenizer, prompt, no_think=args.no_think, model_name=args.model)
    llm.apply_model(probe_reset)
    sampling_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "skip_special_tokens": True,
        "logprobs": args.logprobs if args.logprobs > 0 else None,
    }
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed
    params = SamplingParams(**sampling_kwargs)
    outputs = llm.generate([token_ids], params, use_tqdm=False)
    output = outputs[0].outputs[0]
    text = output.text.strip()
    if not text:
        return None
    hidden_states = llm.apply_model(probe_collect)[0]
    if hidden_states is None:
        return None
    return text, hidden_states, list(output.token_ids)


def load_model(args: argparse.Namespace):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM

    llm_kwargs: dict[str, Any] = dict(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )
    if args.attention_backend != "auto":
        from vllm.config.attention import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        llm_kwargs["attention_config"] = AttentionConfig(
            backend=AttentionBackendEnum[args.attention_backend]
        )
    if args.distributed_executor_backend is not None:
        llm_kwargs["distributed_executor_backend"] = args.distributed_executor_backend

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def release_model(llm: Any) -> None:
    import gc

    import torch

    try:
        llm.apply_model(probe_remove_hooks)
    except Exception as exc:
        print(f"[warn] could not remove hooks cleanly: {exc}")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def dataset_artifact_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_dir = args.data_root / args.dataset_id
    return dataset_dir / "source_records.parquet", dataset_dir / "splits.parquet"


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    source_path, splits_path = dataset_artifact_paths(args)
    slug = model_slug(args.model)
    output_dir = args.output_root / args.dataset_id / slug
    generations_dir = args.generation_root / args.dataset_id / slug
    records_path = output_dir / "records.pt"
    shards_dir = record_shards_dir(output_dir)
    generations_path = generations_dir / "records.jsonl"
    manifest_path = output_dir / "manifest.json"

    if args.dry_run:
        payload = {
            "dataset_id": args.dataset_id,
            "model": args.model,
            "source_path": str(source_path),
            "source_path_exists": source_path.exists(),
            "splits_path": str(splits_path),
            "splits_path_exists": splits_path.exists(),
            "records_path": str(records_path),
            "record_shards_dir": str(shards_dir),
            "generations_path": str(generations_path),
            "tensor_parallel_size": args.tensor_parallel_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "records": 0,
            "new_records": 0,
            "failures": 0,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    import torch

    source_rows = load_parquet_rows(source_path)
    split_map = load_split_map(splits_path)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]

    if args.overwrite:
        import shutil

        for path in (records_path, generations_path, manifest_path):
            if path.exists():
                path.unlink()
        if shards_dir.exists():
            shutil.rmtree(shards_dir)

    records = load_saved_rows(records_path, shards_dir)
    initial_record_count = len(records)
    completed = {str(row["source_record_id"]) for row in records}
    pending_records: list[dict[str, Any]] = []
    next_shard_index = next_record_shard_index(shards_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(parents=True, exist_ok=True)
    remaining_rows = [row for row in source_rows if str(row["source_record_id"]) not in completed]

    llm = None
    tokenizer = None
    started = time.time()
    failures = 0
    if not remaining_rows:
        manifest = {
            "dataset_id": args.dataset_id,
            "model": args.model,
            "model_slug": slug,
            "source_path": str(source_path),
            "splits_path": str(splits_path),
            "records_path": str(records_path),
            "record_shards_dir": str(shards_dir),
            "generations_path": str(generations_path),
            "records": len(records),
            "new_records": 0,
            "failures": 0,
            "elapsed_seconds": time.time() - started,
            "tensor_parallel_size": args.tensor_parallel_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "actmap_shape": [12, 32, 128],
            "actmap_normalized": args.normalize_actmap,
            "gsm8k_output_mode": args.gsm8k_output_mode,
            "sampling_temperature": args.temperature,
            "sampling_seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "no_think": args.no_think,
            "store_prompts": args.store_prompts,
        }
        write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest

    try:
        llm, tokenizer = load_model(args)
        hooked_layers = llm.apply_model(probe_setup_hooks)[0]
        print(f"hooked layers={hooked_layers} model={args.model} dataset={args.dataset_id}")
        with generations_path.open("a", encoding="utf-8") as generation_handle:
            for row_index, source_row in enumerate(remaining_rows):
                source_record_id = str(source_row["source_record_id"])
                prompt = prompt_for_record(
                    args.dataset_id,
                    source_row,
                    gsm8k_output_mode=args.gsm8k_output_mode,
                )
                try:
                    result = extract_generation(llm, tokenizer, prompt, args)
                    if result is None:
                        raise RuntimeError("empty generation or missing hidden states")
                    text, hidden_states, token_ids = result
                    actmap = build_actmap(hidden_states)
                    if args.normalize_actmap:
                        actmap = normalize_actmap(actmap)
                    if args.actmap_dtype == "float16":
                        actmap = actmap.to(dtype=torch.float16)
                    eval_result = {}
                    if args.dataset_id != "cnn_dailymail_3_0_0":
                        eval_result = evaluate_generation_correctness(args.dataset_id, text, source_row)
                    record = {
                        "source_record_id": source_record_id,
                        "dataset_id": args.dataset_id,
                        "split": split_map.get(source_record_id, ""),
                        "model": args.model,
                        "prompt": prompt if args.store_prompts else "",
                        "generation": text,
                        "generated_token_count": len(token_ids),
                        "is_correct": eval_result.get("is_correct"),
                        "label_status": "pending_factuality" if args.dataset_id == "cnn_dailymail_3_0_0" else "evaluated",
                        "actmap": actmap.detach().cpu(),
                        "gsm8k_output_mode": args.gsm8k_output_mode if args.dataset_id == "gsm8k_rationale" else "",
                    }
                    records.append(record)
                    pending_records.append(record)
                    completed.add(source_record_id)
                    generation_handle.write(json.dumps({
                        "source_record_id": source_record_id,
                        "dataset_id": args.dataset_id,
                        "split": record["split"],
                        "model": args.model,
                        "generation": text,
                        "generated_token_count": len(token_ids),
                        "label_status": record["label_status"],
                        "is_correct": record["is_correct"],
                        "gsm8k_output_mode": record["gsm8k_output_mode"],
                    }, ensure_ascii=False) + "\n")
                except Exception as exc:
                    failures += 1
                    print(f"[warn] row {row_index} source_record_id={source_record_id} failed: {exc}")
                if len(pending_records) >= args.save_every:
                    shard_path = save_record_shard(shards_dir, next_shard_index, pending_records)
                    print(f"saved shard {next_shard_index} rows={len(pending_records)} total={len(records)} path={shard_path}")
                    next_shard_index += 1
                    pending_records = []
    finally:
        if llm is not None:
            release_model(llm)

    if pending_records:
        shard_path = save_record_shard(shards_dir, next_shard_index, pending_records)
        print(f"saved shard {next_shard_index} rows={len(pending_records)} total={len(records)} path={shard_path}")
    save_torch_rows(records_path, records)
    manifest = {
        "dataset_id": args.dataset_id,
        "model": args.model,
        "model_slug": slug,
        "source_path": str(source_path),
        "splits_path": str(splits_path),
        "records_path": str(records_path),
        "record_shards_dir": str(shards_dir),
        "generations_path": str(generations_path),
        "records": len(records),
        "new_records": len(records) - initial_record_count,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "actmap_shape": [12, 32, 128],
        "actmap_normalized": args.normalize_actmap,
        "gsm8k_output_mode": args.gsm8k_output_mode,
        "sampling_temperature": args.temperature,
        "sampling_seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "no_think": args.no_think,
        "store_prompts": args.store_prompts,
        "tensor_parallel_size": args.tensor_parallel_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def validate_requested_pair(args: argparse.Namespace) -> None:
    registry = read_json(args.registry)
    dataset_ids = {dataset["id"] for dataset in registry["datasets"]}
    model_ids = {model["hf_id"] for model in registry["models"]}
    if args.dataset_id not in dataset_ids:
        raise ValueError(f"Unknown dataset_id {args.dataset_id!r}; expected one of {sorted(dataset_ids)}")
    if args.model not in model_ids:
        raise ValueError(f"Unknown model {args.model!r}; expected one of {sorted(model_ids)}")
    model_family(args.model)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ActMap feature artifacts for one dataset/model pair.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generation-root", type=Path, default=DEFAULT_GENERATION_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--gsm8k-output-mode", choices=["rationale", "direct"], default="rationale")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "FLASHINFER", "FLASH_ATTN"],
        default="auto",
    )
    parser.add_argument(
        "--distributed-executor-backend",
        choices=["mp", "ray", "external_launcher", "uni"],
        default=None,
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--actmap-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--normalize-actmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-think", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-prompts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens is None:
        args.max_new_tokens = max_new_tokens_for_dataset(
            args.dataset_id,
            gsm8k_output_mode=args.gsm8k_output_mode,
        )
    if args.batch_size != 1:
        raise ValueError(
            "ActMap generation currently supports batch_size=1 only. "
            "vLLM exposes flattened active-token hidden states to hooks, so larger "
            "batches require explicit per-request activation reconstruction to avoid "
            "mixing ActMaps across examples."
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_requested_pair(args)
    manifest = run_generation(args)
    if manifest["failures"] and manifest["new_records"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
