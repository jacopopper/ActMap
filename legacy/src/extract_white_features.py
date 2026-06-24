from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODEL_NAMES = (
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)

PAPER_PROMPT = "Question: {question}\nAnswer:"
DRIFT_HF_LAYERS = (1, 5, 9, 13, 17, 21)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def load_light_rows(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise TypeError(f"Expected {path} to contain a list of row dictionaries")

    questions = [str(row["question"]) for row in rows]
    labels = np.array([1 if bool(row["is_correct"]) else 0 for row in rows], dtype=np.int64)
    models = np.array([str(row["model"]) for row in rows], dtype=str)
    del rows
    gc.collect()
    return questions, labels, models


def select_subset(
    questions: list[str],
    labels: np.ndarray,
    models: np.ndarray,
    *,
    model_names: list[str],
    max_rows_per_model: int | None,
    seed: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    if max_rows_per_model is None:
        indices = np.arange(len(labels), dtype=np.int64)
    else:
        rng = np.random.RandomState(seed)
        selected: list[np.ndarray] = []
        for model_name in model_names:
            model_idx = np.flatnonzero(models == model_name)
            if len(model_idx) == 0:
                continue
            pos_idx = model_idx[labels[model_idx] == 1]
            neg_idx = model_idx[labels[model_idx] == 0]
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            target = min(max_rows_per_model, len(model_idx))
            half = target // 2
            take_pos = min(half, len(pos_idx))
            take_neg = min(target - take_pos, len(neg_idx))
            if take_pos + take_neg < target:
                take_pos = min(target - take_neg, len(pos_idx))
            chosen = np.concatenate([pos_idx[:take_pos], neg_idx[:take_neg]])
            rng.shuffle(chosen)
            selected.append(chosen)
        if not selected:
            raise ValueError("No rows selected; check --models and --max-rows-per-model")
        indices = np.concatenate(selected).astype(np.int64)
        rng.shuffle(indices)

    subset_questions = [questions[int(i)] for i in indices]
    subset_labels = labels[indices]
    subset_models = models[indices]
    return subset_questions, subset_labels, subset_models, indices


def save_metadata(
    out_dir: Path,
    *,
    data: Path,
    feature_path: Path,
    done_path: Path,
    labels_path: Path,
    models_path: Path,
    questions_path: Path,
    source_indices_path: Path,
    models: np.ndarray,
    layers: tuple[int, ...],
    hidden_dim: int,
    feature_dtype: str,
    max_length: int,
) -> None:
    counts = {str(model): int((models == model).sum()) for model in sorted(set(models.tolist()))}
    payload = {
        "data": str(data),
        "prompt": PAPER_PROMPT,
        "feature_position": "last question token before Answer:",
        "hf_layer_indices": list(layers),
        "shape": [int(len(models)), int(len(layers)), int(hidden_dim)],
        "feature_dtype": feature_dtype,
        "max_length": max_length,
        "features": str(feature_path),
        "done": str(done_path),
        "labels": str(labels_path),
        "models": str(models_path),
        "questions": str(questions_path),
        "source_indices": str(source_indices_path),
        "model_counts": counts,
    }
    with (out_dir / "metadata.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


def first_hidden_dim(model_name: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    hidden_dim = getattr(cfg, "hidden_size", None)
    if hidden_dim is None:
        raise ValueError(f"Could not read hidden_size for {model_name}")
    return int(hidden_dim)


def ensure_memmaps(
    out_dir: Path,
    *,
    n_rows: int,
    n_layers: int,
    hidden_dim: int,
    feature_dtype: str,
) -> tuple[np.memmap, np.memmap, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "layer_features.npy"
    done_path = out_dir / "done.npy"
    shape = (n_rows, n_layers, hidden_dim)

    if feature_path.exists():
        features = np.load(feature_path, mmap_mode="r+")
        if tuple(features.shape) != shape:
            raise ValueError(f"{feature_path} has shape {features.shape}, expected {shape}")
        if str(features.dtype) != feature_dtype:
            raise ValueError(f"{feature_path} has dtype {features.dtype}, expected {feature_dtype}")
    else:
        features = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.dtype(feature_dtype),
            shape=shape,
        )

    if done_path.exists():
        done = np.load(done_path, mmap_mode="r+")
        if tuple(done.shape) != (n_rows,):
            raise ValueError(f"{done_path} has shape {done.shape}, expected {(n_rows,)}")
    else:
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_, shape=(n_rows,))
        done[:] = False
        done.flush()

    return features, done, feature_path, done_path


def find_last_question_positions(prompts: list[str], offsets: torch.Tensor) -> list[int]:
    positions: list[int] = []
    offsets_list = offsets.detach().cpu().tolist()
    for prompt, row_offsets in zip(prompts, offsets_list):
        answer_start = prompt.rfind("Answer:")
        if answer_start < 0:
            raise ValueError("Prompt does not contain Answer: marker")

        marker_token = None
        for token_i, (start, end) in enumerate(row_offsets):
            if end > start and end > answer_start:
                marker_token = token_i
                break
        if marker_token is None:
            raise ValueError("Could not locate Answer: marker in token offsets")

        pos = marker_token - 1
        while pos >= 0:
            start, end = row_offsets[pos]
            if end > start:
                positions.append(pos)
                break
            pos -= 1
        else:
            raise ValueError("Could not locate a non-special question token before Answer:")
    return positions


def extract_for_model(
    *,
    model_name: str,
    indices: np.ndarray,
    questions: list[str],
    features: np.memmap,
    done: np.memmap,
    layers: tuple[int, ...],
    batch_size: int,
    max_length: int,
    torch_dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool,
) -> None:
    pending = indices[~np.asarray(done[indices], dtype=bool)]
    if len(pending) == 0:
        print(f"{model_name}: all {len(indices)} rows already extracted")
        return

    print(f"\n=== {model_name} ===")
    print(f"pending rows: {len(pending)}/{len(indices)}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    ).to(device)
    model.eval()

    t0 = time.time()
    try:
        for start in range(0, len(pending), batch_size):
            batch_indices = pending[start : start + batch_size]
            prompts = [PAPER_PROMPT.format(question=questions[int(i)]) for i in batch_indices]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            positions = find_last_question_positions(prompts, offsets)
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = model(**encoded, output_hidden_states=True, use_cache=False)

            batch_features = []
            for batch_i, pos in enumerate(positions):
                parts = [
                    outputs.hidden_states[layer_i][batch_i, pos, :].detach().cpu().to(torch.float32).numpy()
                    for layer_i in layers
                ]
                batch_features.append(np.stack(parts, axis=0))

            features[batch_indices, :, :] = np.stack(batch_features, axis=0).astype(features.dtype)
            done[batch_indices] = True
            features.flush()
            done.flush()

            completed = start + len(batch_indices)
            if completed == len(pending) or completed % max(batch_size * 20, 1) == 0:
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                remaining = (len(pending) - completed) / max(rate, 1e-6)
                print(
                    f"  {completed}/{len(pending)} extracted "
                    f"({rate:.2f} rows/s, eta {remaining / 60:.1f} min)"
                )
    finally:
        del model
        del tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DRIFT/linear-probe hidden-state features from existing QA rows."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    parser.add_argument(
        "--max-rows-per-model",
        type=int,
        default=None,
        help="Optional stratified small-run sample size per model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=parse_int_tuple, default=DRIFT_HF_LAYERS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was specified but CUDA is not available")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.model_dtype]

    all_questions, all_labels, all_models = load_light_rows(args.data)
    questions, labels, models, source_indices = select_subset(
        all_questions,
        all_labels,
        all_models,
        model_names=args.models,
        max_rows_per_model=args.max_rows_per_model,
        seed=args.seed,
    )
    del all_questions, all_labels, all_models
    gc.collect()
    hidden_dims = {model_name: first_hidden_dim(model_name) for model_name in args.models}
    if len(set(hidden_dims.values())) != 1:
        raise ValueError(f"All requested models must share hidden size for one cache: {hidden_dims}")
    hidden_dim = next(iter(hidden_dims.values()))

    features, done, feature_path, done_path = ensure_memmaps(
        args.output_dir,
        n_rows=len(labels),
        n_layers=len(args.layers),
        hidden_dim=hidden_dim,
        feature_dtype=args.feature_dtype,
    )

    labels_path = args.output_dir / "labels.npy"
    models_path = args.output_dir / "models.npy"
    questions_path = args.output_dir / "questions.npy"
    source_indices_path = args.output_dir / "source_indices.npy"
    np.save(labels_path, labels)
    np.save(models_path, models)
    np.save(questions_path, np.array(questions, dtype=object))
    np.save(source_indices_path, source_indices)
    save_metadata(
        args.output_dir,
        data=args.data,
        feature_path=feature_path,
        done_path=done_path,
        labels_path=labels_path,
        models_path=models_path,
        questions_path=questions_path,
        source_indices_path=source_indices_path,
        models=models,
        layers=args.layers,
        hidden_dim=hidden_dim,
        feature_dtype=args.feature_dtype,
        max_length=args.max_length,
    )

    for model_name in args.models:
        model_indices = np.flatnonzero(models == model_name)
        if len(model_indices) == 0:
            print(f"{model_name}: no rows in {args.data}")
            continue
        extract_for_model(
            model_name=model_name,
            indices=model_indices,
            questions=questions,
            features=features,
            done=done,
            layers=args.layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=args.trust_remote_code,
        )

    print(f"\nfeatures -> {feature_path}")
    print(f"labels   -> {labels_path}")
    print(f"models   -> {models_path}")


if __name__ == "__main__":
    main()
