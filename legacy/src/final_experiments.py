from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .white import (
    TwoLogitMLP,
    auroc as white_auroc,
    predict_two_logit,
    set_seed as set_white_seed,
    summarize_method,
    take_rows,
)

from .vit import (
    ProbeDataset,
    _accuracy,
    _auprc,
    _auroc,
    _ece,
    _run_epoch,
    build_model,
    checkpoint_model_config,
    checkpoint_state_dict,
    load_checkpoint_payload,
    make_splits,
    select_device,
)


DEFAULT_MODELS = (
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
DEFAULT_SEEDS = (42, 123, 456)
DRIFT_LAYERS = "1,5,9,13,17,21"
COMMON32_LAYERS = ",".join(str(i) for i in range(1, 33))
BLACK_SCORE_SCHEMA_VERSION = 3


DATASETS: dict[str, dict[str, str]] = {
    "triviaqa": {
        "data": "data/train_30k.pt",
        "csv": "data/triviaqa_30k.csv",
        "build_dataset": "triviaqa",
        "display": "TriviaQA",
    },
    "nq_open": {
        "data": "data/train_nq_open_balanced.pt",
        "csv": "data/nq_open.csv",
        "build_dataset": "nq_open",
        "display": "NQ-Open",
    },
    "web_questions": {
        "data": "data/train_web_questions.pt",
        "csv": "data/web_questions.csv",
        "build_dataset": "web_questions",
        "display": "WebQuestions",
    },
    "gsm8k": {
        "data": "data/train_gsm8k_balanced.pt",
        "raw_data": "data/train_gsm8k.pt",
        "csv": "data/gsm8k.csv",
        "build_dataset": "gsm8k",
        "display": "GSM8K",
    },
}
DEFAULT_DATASETS = ("triviaqa", "nq_open", "web_questions", "gsm8k")


def parse_seed_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def dataset_names(args: argparse.Namespace) -> list[str]:
    return list(args.datasets or DEFAULT_DATASETS)


def dataset_path(name: str) -> Path:
    return Path(DATASETS[name]["data"])


def raw_dataset_path(name: str) -> Path:
    return Path(DATASETS[name].get("raw_data", DATASETS[name]["data"]))


def dataset_models(name: str, fallback: list[str]) -> list[str]:
    value = DATASETS[name].get("models")
    if not value:
        return list(fallback)
    return [part.strip() for part in value.split(",") if part.strip()]


def split_path(root: Path, dataset: str, seed: int) -> Path:
    return root / "splits" / f"{dataset}_seed{seed}.json"


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise TypeError(f"Expected {path} to contain a list, got {type(rows).__name__}")
    return rows


def labels_for(rows: list[dict[str, Any]], indices: list[int] | np.ndarray | None = None) -> np.ndarray:
    if indices is None:
        indices = range(len(rows))
    return np.array([1 if bool(rows[int(i)]["is_correct"]) else 0 for i in indices], dtype=np.int64)


def model_counts(rows: list[dict[str, Any]], indices: list[int] | np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx in indices:
        model = str(rows[int(idx)]["model"])
        counts[model] = counts.get(model, 0) + 1
    return counts


def split_summary(rows: list[dict[str, Any]], splits: dict[str, list[int]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, idx in splits.items():
        y = labels_for(rows, idx)
        out[name] = {
            "n": int(len(idx)),
            "correct": int(y.sum()),
            "incorrect": int(len(y) - y.sum()),
            "correct_rate": float(y.mean()) if len(y) else None,
            "model_counts": model_counts(rows, idx),
        }
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def run_command(cmd: list[str], *, dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def ensure_generated_dataset(args: argparse.Namespace, dataset: str) -> None:
    output = raw_dataset_path(dataset)
    if output.exists() and not args.overwrite:
        print(f"{DATASETS[dataset]['display']} ActMaps already exist: {output}")
        return
    cmd = [
        sys.executable,
        "-m",
        "src.build_train",
        "--dataset",
        DATASETS[dataset]["build_dataset"],
        "--input",
        DATASETS[dataset]["csv"],
        "--output",
        str(output),
        "--models",
        *args.models,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        "0.0",
        "--save-every",
        str(args.save_every),
        "--actmap-dtype",
        "float16",
        "--normalize-actmap",
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.overwrite:
        cmd.append("--overwrite")
    run_command(cmd, dry_run=args.dry_run)


def ensure_web_questions_csv(args: argparse.Namespace) -> None:
    output = Path(DATASETS["web_questions"]["csv"])
    if output.exists() and not args.overwrite:
        print(f"WebQuestions CSV already exists: {output}")
        return

    import csv

    from datasets import load_dataset

    from .evaluation import normalize_answer

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "question",
        "answer_value",
        "answer_normalized",
        "aliases",
        "normalized_aliases",
        "answers",
        "source_split",
        "url",
    ]
    total = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "test"):
            rows = load_dataset("stanfordnlp/web_questions", split=split)
            for idx, row in enumerate(rows):
                answers = [str(answer) for answer in row.get("answers", []) if str(answer).strip()]
                normalized = sorted({normalize_answer(answer) for answer in answers if normalize_answer(answer)})
                writer.writerow({
                    "question_id": f"webq_{split}_{idx}",
                    "question": row["question"],
                    "answer_value": answers[0] if answers else "",
                    "answer_normalized": normalized[0] if normalized else "",
                    "aliases": json.dumps(answers, ensure_ascii=False),
                    "normalized_aliases": json.dumps(normalized, ensure_ascii=False),
                    "answers": json.dumps(answers, ensure_ascii=False),
                    "source_split": split,
                    "url": row.get("url", ""),
                })
                total += 1
    print(f"saved {output} ({total} rows)")


def ensure_web_questions(args: argparse.Namespace) -> None:
    if not Path(DATASETS["web_questions"]["csv"]).exists():
        ensure_web_questions_csv(args)
    ensure_generated_dataset(args, "web_questions")


def balance_dataset_by_model(args: argparse.Namespace, dataset: str) -> None:
    input_path = raw_dataset_path(dataset)
    output_path = dataset_path(dataset)
    if input_path == output_path:
        raise ValueError(f"{dataset} has no separate raw_data path configured")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {input_path}")
    if output_path.exists() and not args.overwrite:
        print(f"balanced {DATASETS[dataset]['display']} dataset already exists: {output_path}")
        return
    if args.dry_run:
        print(f"would balance {input_path} -> {output_path}")
        return

    rows = load_rows(input_path)
    model_filter = DATASETS[dataset].get("model_filter")
    if model_filter:
        rows = [row for row in rows if str(row.get("model")) == model_filter]
        if not rows:
            raise RuntimeError(f"No rows matched model_filter={model_filter!r} in {input_path}")
    rng = np.random.default_rng(args.balance_seed)
    selected: set[int] = set()
    metadata: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "seed": args.balance_seed,
        "strategy": "per-model downsample majority class to minority class count",
        "input_rows": len(rows),
        "model_filter": model_filter,
        "per_model": {},
    }

    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        model_indices = [idx for idx, row in enumerate(rows) if str(row["model"]) == model]
        by_label = {
            False: [idx for idx in model_indices if not bool(rows[idx]["is_correct"])],
            True: [idx for idx in model_indices if bool(rows[idx]["is_correct"])],
        }
        counts = {str(label): len(indices) for label, indices in by_label.items()}
        if not by_label[False] or not by_label[True]:
            raise RuntimeError(f"Cannot balance {dataset} model={model}; class counts={counts}")
        target = min(len(by_label[False]), len(by_label[True]))
        kept_by_label: dict[bool, list[int]] = {}
        for label, indices in by_label.items():
            if len(indices) > target:
                kept = rng.choice(np.array(indices, dtype=np.int64), size=target, replace=False).tolist()
            else:
                kept = list(indices)
            kept_by_label[label] = [int(idx) for idx in kept]
            selected.update(kept_by_label[label])

        metadata["per_model"][model] = {
            "input": {"incorrect": len(by_label[False]), "correct": len(by_label[True])},
            "output": {"incorrect": len(kept_by_label[False]), "correct": len(kept_by_label[True])},
            "pruned": len(model_indices) - (len(kept_by_label[False]) + len(kept_by_label[True])),
        }

    balanced_rows = [rows[idx] for idx in sorted(selected)]
    metadata["output_rows"] = len(balanced_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(balanced_rows, output_path)
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    print(f"saved balanced {DATASETS[dataset]['display']} dataset: {output_path}")
    print(json.dumps(metadata, indent=2))


def ensure_gsm8k(args: argparse.Namespace) -> None:
    ensure_generated_dataset(args, "gsm8k")
    balance_dataset_by_model(args, "gsm8k")


def cmd_balance_gsm8k(args: argparse.Namespace) -> None:
    balance_dataset_by_model(args, "gsm8k")


def cmd_audit(args: argparse.Namespace) -> None:
    for name in dataset_names(args):
        path = dataset_path(name)
        print(f"\n{name}: {path}")
        if not path.exists():
            print("  missing")
            continue
        rows = load_rows(path)
        y = labels_for(rows)
        print(f"  rows={len(rows)} correct={int(y.sum())} incorrect={int(len(y) - y.sum())} rate={float(y.mean()):.3f}")
        for model, count in sorted(model_counts(rows, range(len(rows))).items()):
            model_y = np.array([1 if bool(row["is_correct"]) else 0 for row in rows if row["model"] == model])
            print(f"  {model:40s} n={count:6d} correct_rate={float(model_y.mean()):.3f}")


def cmd_make_splits(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for dataset in dataset_names(args):
        rows = load_rows(dataset_path(dataset))
        for seed in args.seeds:
            out = split_path(root, dataset, seed)
            if out.exists() and not args.overwrite:
                print(f"exists: {out}")
                continue
            train_idx, val_idx, test_idx = make_splits(
                rows,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=seed,
            )
            payload = {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
                "metadata": {
                    "dataset": dataset,
                    "seed": seed,
                    "data": str(dataset_path(dataset)),
                    "val_frac": args.val_frac,
                    "test_frac": args.test_frac,
                    "summary": split_summary(rows, {"train": train_idx, "val": val_idx, "test": test_idx}),
                },
            }
            write_json(out, payload)
            print(f"saved {out}")


def cmd_train_actmap(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for dataset in dataset_names(args):
        for seed in args.seeds:
            out_dir = root / "actmap" / dataset / f"seed_{seed}"
            result_path = out_dir / "test_results.json"
            if result_path.exists() and not args.overwrite:
                print(f"exists: {result_path}")
                continue
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "src.vit",
                "train",
                "--data",
                str(dataset_path(dataset)),
                "--output",
                str(out_dir),
                "--arch",
                "vit2d",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--dropout",
                str(args.dropout),
                "--seed",
                str(seed),
                "--patience",
                str(args.patience),
                "--noise-std",
                str(args.noise_std),
                "--warmup-epochs",
                str(args.warmup_epochs),
                "--mixup-alpha",
                str(args.mixup_alpha),
                "--patch-h",
                "4",
                "--patch-w",
                "16",
                "--embed-dim",
                "192",
                "--num-heads",
                "6",
                "--num-layers",
                "6",
                "--mlp-ratio",
                "3.0",
                "--attn-drop",
                "0.1",
                "--drop-path-rate",
                "0.05",
                "--split-file",
                str(split_path(root, dataset, seed)),
            ]
            if args.require_cuda:
                cmd.append("--require-cuda")
            run_command(cmd, dry_run=args.dry_run)


def load_split_indices(path: Path, split: str) -> list[int]:
    payload = read_json(path)
    return [int(i) for i in payload[split]]


def evaluate_checkpoint(
    *,
    data_path: Path,
    checkpoint_path: Path,
    split_indices: list[int],
    batch_size: int,
    device: torch.device,
    fallback_arch: str = "vit2d",
) -> dict[str, Any]:
    rows = load_rows(data_path)
    in_channels = rows[0]["actmap"].shape[0]
    payload = load_checkpoint_payload(checkpoint_path, device)
    model_config = checkpoint_model_config(
        payload,
        checkpoint_path,
        in_channels=in_channels,
        fallback_arch=fallback_arch,
    )
    model = build_model(
        model_config["arch"],
        in_channels=model_config.get("in_channels", in_channels),
        patch_h=model_config.get("patch_h", 4),
        patch_w=model_config.get("patch_w", 8),
        embed_dim=model_config.get("embed_dim", 192),
        num_heads=model_config.get("num_heads", 6),
        num_layers=model_config.get("num_layers", 6),
        mlp_ratio=model_config.get("mlp_ratio", 3.0),
        dropout=model_config.get("dropout", 0.3),
        attn_drop=model_config.get("attn_drop", 0.1),
        drop_path_rate=model_config.get("drop_path_rate", 0.05),
        spatial_h=model_config.get("spatial_h", 32),
        spatial_w=model_config.get("spatial_w", 128),
    ).to(device)
    model.load_state_dict(checkpoint_state_dict(payload))
    model.eval()

    eval_rows = [rows[i] for i in split_indices]
    ds = ProbeDataset(eval_rows, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss()
    _, logits, labels = _run_epoch(
        model,
        loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
    )
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    per_model: dict[str, Any] = {}
    for model_name in sorted({str(row["model"]) for row in eval_rows}):
        mask = np.array([str(row["model"]) == model_name for row in eval_rows], dtype=bool)
        per_model[model_name] = {
            "n": int(mask.sum()),
            "auroc": _auroc(labels[mask], probs[mask]),
            "auprc": _auprc(labels[mask], probs[mask]),
            "ece": _ece(labels[mask].astype(int), probs[mask]),
        }
    return {
        "n": len(eval_rows),
        "auroc": _auroc(labels, probs),
        "auprc": _auprc(labels, probs),
        "accuracy": _accuracy(labels.astype(int), logits),
        "ece": _ece(labels.astype(int), probs),
        "per_model": per_model,
    }


def cmd_eval_actmap_matrix(args: argparse.Namespace) -> None:
    root = Path(args.root)
    device = select_device(args.require_cuda)
    for source in dataset_names(args):
        for target in dataset_names(args):
            for seed in args.seeds:
                out = root / "actmap_matrix" / f"{source}_to_{target}" / f"seed_{seed}.json"
                if out.exists() and not args.overwrite:
                    print(f"exists: {out}")
                    continue
                checkpoint = root / "actmap" / source / f"seed_{seed}" / "best_model.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                target_split = split_path(root, target, seed)
                indices = load_split_indices(target_split, args.split)
                metrics = evaluate_checkpoint(
                    data_path=dataset_path(target),
                    checkpoint_path=checkpoint,
                    split_indices=indices,
                    batch_size=args.batch_size,
                    device=device,
                )
                result = {
                    "method": "actmap_vit2d",
                    "source": source,
                    "target": target,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "target_data": str(dataset_path(target)),
                    "target_split_file": str(target_split),
                    "target_split": args.split,
                    "metrics": metrics,
                }
                write_json(out, result)
                print(f"saved {out}: AUROC={metrics['auroc']:.4f}")


def cmd_extract_white(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for dataset in dataset_names(args):
        for kind, layers in (("drift6", DRIFT_LAYERS), ("common32", COMMON32_LAYERS)):
            out_dir = root / "white_features" / dataset / kind
            done_path = out_dir / "done.npy"
            if done_path.exists() and not args.overwrite:
                done = np.load(done_path, mmap_mode="r")
                if bool(np.all(done)):
                    print(f"exists complete: {out_dir}")
                    continue
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "src.extract_white_features",
                "--data",
                str(dataset_path(dataset)),
                "--output-dir",
                str(out_dir),
                "--models",
                *dataset_models(dataset, args.models),
                "--layers",
                layers,
                "--batch-size",
                str(args.white_batch_size),
                "--feature-dtype",
                "float32",
                "--model-dtype",
                "bfloat16",
            ]
            if args.require_cuda:
                cmd.append("--require-cuda")
            run_command(cmd, dry_run=args.dry_run)


def white_feature_paths(root: Path, dataset: str, kind: str) -> dict[str, Path]:
    base = root / "white_features" / dataset / kind
    return {
        "features": base / "layer_features.npy",
        "labels": base / "labels.npy",
        "models": base / "models.npy",
    }


def load_white_splits(path: Path) -> dict[str, np.ndarray]:
    payload = read_json(path)
    return {name: np.array(payload[name], dtype=np.int64) for name in ("train", "val", "test")}


def flatten_white_layers(features: np.ndarray) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError(f"Expected [N,L,D] features, got {features.shape}")
    return features.reshape(features.shape[0], -1)


def fit_drift_probe(
    source_layers: np.ndarray,
    source_labels: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    seed: int,
    require_cuda: bool,
) -> dict[str, Any]:
    device = select_device(require_cuda)
    set_white_seed(seed)
    source_features = flatten_white_layers(source_layers)
    x_train = take_rows(source_features, splits["train"])
    x_val = take_rows(source_features, splits["val"])
    y_train = source_labels[splits["train"]]
    y_val = source_labels[splits["val"]]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)

    model = TwoLogitMLP(x_train.shape[1], (512, 128), 0.3, activation="leaky_relu").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=64,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_acc = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, 41):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_scores = predict_two_logit(model, x_val, 64, device)
        val_acc = float(((val_scores >= 0).astype(np.int64) == y_val).mean())
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "classifier": f"{source_features.shape[1]} -> 512 -> 128 -> 2",
        "standardized": True,
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "epochs": 40,
        "patience": 8,
        "batch_size": 64,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_acc,
    }
    return {"model": model, "scaler": scaler, "metadata": metadata, "device": device}


def predict_drift_probe(fitted: dict[str, Any], target_layers: np.ndarray) -> np.ndarray:
    x_target = flatten_white_layers(target_layers).astype(np.float32)
    x_target = fitted["scaler"].transform(x_target).astype(np.float32)
    return predict_two_logit(fitted["model"], x_target, 64, fitted["device"]).astype(np.float32)


def fit_linear_probe(
    source_layers: np.ndarray,
    source_labels: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    seed: int,
    require_cuda: bool,
    layer_ids: str,
) -> dict[str, Any]:
    device = select_device(require_cuda)
    set_white_seed(seed)
    y_train = source_labels[splits["train"]]
    y_val = source_labels[splits["val"]]

    n_layers = int(source_layers.shape[1])
    hidden_dim = int(source_layers.shape[2])
    base = nn.Linear(hidden_dim, 2)
    weight = nn.Parameter(base.weight.detach().clone().t().unsqueeze(0).repeat(n_layers, 1, 1).to(device))
    bias = nn.Parameter(base.bias.detach().clone().unsqueeze(0).repeat(n_layers, 1).to(device))
    optimizer = torch.optim.AdamW([weight, bias], lr=1e-3, weight_decay=1e-4)
    rng = np.random.RandomState(seed)
    train_idx = np.array(splits["train"], dtype=np.int64)

    linear_batch_size = 2048
    for epoch in range(30):
        rng.shuffle(train_idx)
        for start in range(0, len(train_idx), linear_batch_size):
            batch_idx = train_idx[start : start + linear_batch_size]
            xb_np = np.asarray(source_layers[batch_idx], dtype=np.float32)
            yb_np = source_labels[batch_idx]
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            yb = torch.from_numpy(yb_np.astype(np.int64)).to(device, non_blocking=True)
            logits = torch.einsum("bld,ldc->blc", xb, weight) + bias.unsqueeze(0)
            expanded_y = yb[:, None].expand(-1, n_layers).reshape(-1)
            per_item = F.cross_entropy(logits.reshape(-1, 2), expanded_y, reduction="none")
            loss = per_item.reshape(len(batch_idx), n_layers).mean(dim=0).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  linear epoch {epoch + 1}/30")

    val_scores_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(splits["val"]), linear_batch_size):
            batch_idx = splits["val"][start : start + linear_batch_size]
            xb_np = np.asarray(source_layers[batch_idx], dtype=np.float32)
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            logits = torch.einsum("bld,ldc->blc", xb, weight) + bias.unsqueeze(0)
            scores = (logits[:, :, 1] - logits[:, :, 0]).detach().cpu().float().numpy()
            val_scores_chunks.append(scores)
    val_scores_all = np.concatenate(val_scores_chunks, axis=0)

    best_layer = -1
    best_val_auroc = -1.0
    layer_metrics: dict[str, Any] = {}
    for layer_i in range(n_layers):
        val_scores = val_scores_all[:, layer_i]
        val_auroc = white_auroc(y_val, val_scores)
        layer_metrics[str(layer_i)] = {"val_auroc": val_auroc}
        if val_auroc is not None and val_auroc > best_val_auroc:
            best_layer = layer_i
            best_val_auroc = val_auroc

    if best_layer < 0:
        raise RuntimeError("No linear layer produced a valid validation AUROC")
    ids = [int(part) for part in layer_ids.split(",") if part]
    metadata: dict[str, Any] = {
        "implementation": "best source-validation affine probe transferred directly to target features",
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 30,
        "batch_size": linear_batch_size,
        "selected_layer": best_layer,
        "selected_layer_val_auroc": best_val_auroc,
        "layer_metrics": layer_metrics,
        "layer_ids": ids,
        "vectorized_training": True,
    }
    if len(ids) == source_layers.shape[1]:
        metadata["selected_layer_id"] = ids[best_layer]
    return {
        "weight": weight.detach().cpu(),
        "bias": bias.detach().cpu(),
        "best_layer": best_layer,
        "metadata": metadata,
        "device": device,
    }


def predict_linear_probe(fitted: dict[str, Any], target_layers: np.ndarray) -> np.ndarray:
    layer_i = int(fitted["best_layer"])
    x_target = np.asarray(target_layers[:, layer_i, :], dtype=np.float32)
    weight = fitted["weight"][layer_i].to(fitted["device"])
    bias = fitted["bias"][layer_i].to(fitted["device"])
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_target), 512):
            xb = torch.from_numpy(x_target[start : start + 512]).to(fitted["device"], non_blocking=True)
            logits = xb @ weight + bias
            chunks.append((logits[:, 1] - logits[:, 0]).detach().cpu().float().numpy())
    return np.concatenate(chunks).astype(np.float32)


def save_white_result(
    *,
    method: str,
    method_scores: np.ndarray,
    target_labels: np.ndarray,
    target_models: np.ndarray,
    metadata: dict[str, Any],
    result_path: Path,
    scores_path: Path,
    source_label_cache: Path,
    target_label_cache: Path,
    source_split_file: Path,
    target_split_file: Path,
    target_split: str,
    target_indices: list[int],
) -> None:
    metric = summarize_method(target_labels, method_scores, target_models)
    results = {
        "source_label_cache": str(source_label_cache),
        "target_label_cache": str(target_label_cache),
        "source_split_file": str(source_split_file),
        "target_split_file": str(target_split_file),
        "target_split": target_split,
        "target_indices": target_indices,
        "target_n": int(len(target_labels)),
        "methods": [method],
        "method_metadata": {method: metadata},
        "metrics": {method: metric},
    }
    rows = []
    for i in range(len(target_labels)):
        rows.append({
            "index": i,
            "target_index": int(target_indices[i]),
            "model": str(target_models[i]),
            "is_correct": bool(target_labels[i]),
            method: float(method_scores[i]),
        })
    write_json(result_path, results)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, scores_path)
    print(f"saved {result_path}: AUROC={metric['auroc']:.4f}")


def cmd_run_white_matrix(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for method, kind, layer_ids in (
        ("drift", "drift6", DRIFT_LAYERS),
        ("best_layer_linear_probe", "common32", COMMON32_LAYERS),
    ):
        for source in dataset_names(args):
            source_paths = white_feature_paths(root, source, kind)
            source_labels = np.asarray(np.load(source_paths["labels"], mmap_mode="r"), dtype=np.int64)
            source_layers = np.load(source_paths["features"], mmap_mode="r")
            for seed in args.seeds:
                source_split_file = split_path(root, source, seed)
                missing_targets = []
                for target in dataset_names(args):
                    out_dir = root / "white_matrix" / method / f"{source}_to_{target}" / f"seed_{seed}"
                    result_path = out_dir / "results.json"
                    if result_path.exists() and not args.overwrite:
                        print(f"exists: {result_path}")
                        continue
                    missing_targets.append(target)
                if not missing_targets:
                    continue
                if args.dry_run:
                    print(f"would fit {method} source={source} seed={seed} for targets={missing_targets}")
                    continue
                splits = load_white_splits(source_split_file)
                print(f"\nfit {method}: source={source} seed={seed} targets={missing_targets}")
                if method == "drift":
                    fitted = fit_drift_probe(
                        source_layers,
                        source_labels,
                        splits,
                        seed=seed,
                        require_cuda=args.require_cuda,
                    )
                else:
                    fitted = fit_linear_probe(
                        source_layers,
                        source_labels,
                        splits,
                        seed=seed,
                        require_cuda=args.require_cuda,
                        layer_ids=layer_ids,
                    )
                for target in missing_targets:
                    target_paths = white_feature_paths(root, target, kind)
                    target_split_file = split_path(root, target, seed)
                    target_indices = load_split_indices(target_split_file, args.split)
                    target_layers_all = np.load(target_paths["features"], mmap_mode="r")
                    target_labels_all = np.asarray(np.load(target_paths["labels"], mmap_mode="r"), dtype=np.int64)
                    target_models_all = np.asarray(np.load(target_paths["models"], allow_pickle=True), dtype=object)
                    target_layers = target_layers_all[target_indices]
                    target_labels = target_labels_all[target_indices]
                    target_models = target_models_all[target_indices]
                    if method == "drift":
                        method_scores = predict_drift_probe(fitted, target_layers)
                    else:
                        method_scores = predict_linear_probe(fitted, target_layers)
                    out_dir = root / "white_matrix" / method / f"{source}_to_{target}" / f"seed_{seed}"
                    save_white_result(
                        method=method,
                        method_scores=method_scores,
                        target_labels=target_labels,
                        target_models=target_models,
                        metadata=fitted["metadata"],
                        result_path=out_dir / "results.json",
                        scores_path=out_dir / "scores.pt",
                        source_label_cache=source_paths["labels"],
                        target_label_cache=target_paths["labels"],
                        source_split_file=source_split_file,
                        target_split_file=target_split_file,
                        target_split=args.split,
                        target_indices=target_indices,
                    )
                    del target_layers_all, target_labels_all, target_models_all, target_layers
                del fitted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


def black_union_split_path(root: Path, dataset: str, split: str) -> Path:
    return root / "black" / dataset / "union" / f"{split}_union_split.json"


def build_black_union_split(root: Path, dataset: str, seeds: tuple[int, ...], split: str) -> list[int]:
    indices: set[int] = set()
    for seed in seeds:
        payload = read_json(split_path(root, dataset, seed))
        indices.update(int(i) for i in payload[split])
    ordered = sorted(indices)
    payload = {"train": [], "val": [], "test": []}
    payload[split] = ordered
    write_json(black_union_split_path(root, dataset, split), payload)
    return ordered


def summarize_black_subset(scores_path: Path, split_indices: list[int], out_path: Path, *, dataset: str) -> None:
    scores = torch.load(scores_path, map_location="cpu", weights_only=False)
    if not isinstance(scores, list):
        raise TypeError(f"Expected {scores_path} to contain a list, got {type(scores).__name__}")
    by_index = {
        int(row["train_index"]): row
        for row in scores
        if row.get("score_schema_version") == BLACK_SCORE_SCHEMA_VERSION
        and row.get("dataset") == dataset
        and "train_index" in row
    }
    missing = [idx for idx in split_indices if idx not in by_index]
    if missing:
        raise RuntimeError(f"{scores_path} is missing {len(missing)} requested rows; first missing={missing[:5]}")

    rows = [by_index[idx] for idx in split_indices]
    labels = np.array([1 if bool(row["is_correct"]) else 0 for row in rows], dtype=np.int64)

    def values(name: str) -> np.ndarray:
        return np.array([row.get(name, float("nan")) for row in rows], dtype=np.float64)

    metric_scores = {
        "mte": -values("mte"),
        "p_true": values("p_true"),
    }
    def finite_metric_summary(metric_labels: np.ndarray, metric_values: np.ndarray) -> dict[str, Any]:
        finite = np.isfinite(metric_values)
        dropped = int((~finite).sum())
        labels_f = metric_labels[finite]
        values_f = metric_values[finite]
        return {
            "n": int(finite.sum()),
            "dropped_nonfinite": dropped,
            "auroc": _auroc(labels_f, values_f),
            "auprc": _auprc(labels_f, values_f),
        }

    result: dict[str, Any] = {"n": len(rows), "metrics": {}, "per_model": {}}
    for metric_name, arr in metric_scores.items():
        result["metrics"][metric_name] = finite_metric_summary(labels, arr)

    for model_name in sorted({str(row["model"]) for row in rows}):
        mask = np.array([str(row["model"]) == model_name for row in rows], dtype=bool)
        model_labels = labels[mask]
        result["per_model"][model_name] = {"n": int(mask.sum()), "metrics": {}}
        for metric_name, arr in metric_scores.items():
            model_arr = arr[mask]
            result["per_model"][model_name]["metrics"][metric_name] = finite_metric_summary(model_labels, model_arr)

    write_json(out_path, result)
    print(f"saved {out_path}")


def black_scores_cover(scores_path: Path, split_indices: list[int], *, dataset: str) -> bool:
    if not scores_path.exists():
        return False
    scores = torch.load(scores_path, map_location="cpu", weights_only=False)
    if not isinstance(scores, list):
        return False
    done = {
        int(row["train_index"])
        for row in scores
        if row.get("score_schema_version") == BLACK_SCORE_SCHEMA_VERSION
        and row.get("dataset") == dataset
        and "train_index" in row
    }
    return all(idx in done for idx in split_indices)


def cmd_run_black(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for dataset in dataset_names(args):
        missing = [
            seed
            for seed in args.seeds
            if args.overwrite or not (root / "black" / dataset / f"seed_{seed}" / "results.json").exists()
        ]
        if not missing:
            print(f"exists: black results for {dataset}")
            continue
        union_indices = build_black_union_split(root, dataset, args.seeds, args.split)
        union_dir = root / "black" / dataset / "union"
        union_scores = union_dir / "scores.pt"
        union_results = union_dir / "results.json"
        if args.overwrite or not black_scores_cover(union_scores, union_indices, dataset=dataset):
            print(f"{dataset}: scoring {len(union_indices)} unique {args.split} rows for black baselines")
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "src.black",
                "--data",
                str(dataset_path(dataset)),
                "--dataset",
                dataset,
                "--scores",
                str(union_scores),
                "--results",
                str(union_results),
                "--split-file",
                str(black_union_split_path(root, dataset, args.split)),
                "--split",
                args.split,
                "--models",
                *dataset_models(dataset, args.models),
                "--save-every",
                str(args.save_every),
                "--score-batch-size",
                str(args.score_batch_size),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            run_command(cmd, dry_run=args.dry_run)

        if args.dry_run:
            continue
        for seed in args.seeds:
            out_dir = root / "black" / dataset / f"seed_{seed}"
            result_path = out_dir / "results.json"
            if result_path.exists() and not args.overwrite:
                print(f"exists: {result_path}")
                continue
            split_indices = load_split_indices(split_path(root, dataset, seed), args.split)
            summarize_black_subset(union_scores, split_indices, result_path, dataset=dataset)


def mean_ci(values: list[float]) -> dict[str, float | int | None]:
    arr = np.array([value for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"n": int(arr.size), "mean": float(arr.mean()), "std": std, "ci95": ci95}


def collect_metric(values: list[dict[str, Any]], method_path: list[str], metric: str) -> dict[str, Any]:
    scores: list[float] = []
    for payload in values:
        obj: Any = payload
        for key in method_path:
            obj = obj[key]
        scores.append(obj[metric])
    return mean_ci(scores)


def cmd_aggregate(args: argparse.Namespace) -> None:
    root = Path(args.root)
    datasets = dataset_names(args)
    summary: dict[str, Any] = {"seeds": list(args.seeds), "datasets": datasets, "actmap": {}, "white": {}, "black": {}}

    for source in datasets:
        for target in datasets:
            payloads = []
            for seed in args.seeds:
                path = root / "actmap_matrix" / f"{source}_to_{target}" / f"seed_{seed}.json"
                if path.exists():
                    payloads.append(read_json(path))
            if payloads:
                key = f"{source}_to_{target}"
                summary["actmap"][key] = {
                    metric: collect_metric(payloads, ["metrics"], metric)
                    for metric in ("auroc", "auprc", "ece", "accuracy")
                }

    for method in ("drift", "best_layer_linear_probe"):
        summary["white"][method] = {}
        for source in datasets:
            for target in datasets:
                payloads = []
                for seed in args.seeds:
                    path = root / "white_matrix" / method / f"{source}_to_{target}" / f"seed_{seed}" / "results.json"
                    if path.exists():
                        payloads.append(read_json(path))
                if payloads:
                    key = f"{source}_to_{target}"
                    summary["white"][method][key] = {
                        metric: collect_metric(payloads, ["metrics", method], metric)
                        for metric in ("auroc", "auprc", "ece")
                    }

    for dataset in datasets:
        payloads = []
        for seed in args.seeds:
            path = root / "black" / dataset / f"seed_{seed}" / "results.json"
            if path.exists():
                payloads.append(read_json(path))
        if payloads:
            summary["black"][dataset] = {}
            for method in ("mte", "p_true"):
                summary["black"][dataset][method] = {
                    metric: collect_metric(payloads, ["metrics", method], metric)
                    for metric in ("auroc", "auprc")
                }

    out = root / "summary" / "final_summary.json"
    write_json(out, summary)
    print(f"saved {out}")


def plot_bar(ax: Any, labels: list[str], means: list[float], errors: list[float], *, title: str, ylabel: str) -> None:
    colors = ["#1B4E8A", "#2F7F5F", "#B45F06", "#6A51A3", "#4D4D4D", "#8C564B"]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, capsize=3, color=colors[: len(labels)], edgecolor="#1A202C", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def metric_stats(item: dict[str, Any], metric: str) -> tuple[float, float]:
    stats = item.get(metric, {})
    mean = stats.get("mean")
    ci95 = stats.get("ci95")
    return (
        float(mean) if mean is not None else float("nan"),
        float(ci95) if ci95 is not None else 0.0,
    )


def plot_grouped_bars(
    ax: Any,
    *,
    group_labels: list[str],
    method_labels: list[str],
    means: np.ndarray,
    errors: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    colors = ["#1B4E8A", "#2F7F5F", "#B45F06", "#6A51A3", "#4D4D4D", "#8C564B"]
    x = np.arange(len(group_labels), dtype=np.float64)
    width = min(0.12, 0.76 / max(1, len(method_labels)))
    offsets = (np.arange(len(method_labels)) - (len(method_labels) - 1) / 2.0) * width
    for i, label in enumerate(method_labels):
        ax.bar(
            x + offsets[i],
            means[:, i],
            width,
            yerr=errors[:, i],
            label=label,
            capsize=2.5,
            color=colors[i % len(colors)],
            edgecolor="#222222",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.45, 0.95)
    ax.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--", alpha=0.65)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.75, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_transfer_heatmap(
    ax: Any,
    matrix: np.ndarray,
    *,
    labels: list[str],
    title: str,
    vmin: float = 0.45,
    vmax: float = 0.90,
    highlight_diagonal: bool = False,
) -> Any:
    import matplotlib.patches as patches

    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                text_color = "white" if value < 0.72 else "#111111"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8.5,
                )
    if highlight_diagonal:
        for i in range(matrix.shape[0]):
            rect = patches.Rectangle(
                (i - 0.5, i - 0.5),
                1.0,
                1.0,
                linewidth=2.5,
                edgecolor="#1A1A1A",
                facecolor="none",
                zorder=3,
            )
            ax.add_patch(rect)
    return im


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    root = Path(args.root)
    summary_path = root / "summary" / "final_summary.json"
    summary = read_json(summary_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    datasets = summary["datasets"]
    display = {name: DATASETS[name]["display"] for name in datasets}
    method_labels = ["ActMap", "DRIFT", "Linear", "MTE", "P(True)"]
    indomain_means = []
    indomain_errors = []
    for dataset in datasets:
        key = f"{dataset}_to_{dataset}"
        values = [
            summary["actmap"].get(key, {}).get("auroc", {}),
            summary["white"].get("drift", {}).get(key, {}).get("auroc", {}),
            summary["white"].get("best_layer_linear_probe", {}).get(key, {}).get("auroc", {}),
            summary["black"].get(dataset, {}).get("mte", {}).get("auroc", {}),
            summary["black"].get(dataset, {}).get("p_true", {}).get("auroc", {}),
        ]
        means = [float(v["mean"]) if v.get("mean") is not None else np.nan for v in values]
        errors = [float(v["ci95"]) if v.get("ci95") is not None else 0.0 for v in values]
        indomain_means.append(means)
        indomain_errors.append(errors)
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        plot_bar(ax, method_labels, means, errors, title=f"{display[dataset]} in-domain AUROC", ylabel="AUROC")
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_indomain_auroc.pdf")
        fig.savefig(out_dir / f"{dataset}_indomain_auroc.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    plot_grouped_bars(
        ax,
        group_labels=[display[d] for d in datasets],
        method_labels=method_labels,
        means=np.asarray(indomain_means, dtype=np.float64),
        errors=np.asarray(indomain_errors, dtype=np.float64),
        title="",
        ylabel="AUROC",
    )
    ax.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "indomain_auroc_comparison.pdf")
    fig.savefig(out_dir / "indomain_auroc_comparison.png")
    plt.close(fig)

    matrix = np.full((len(datasets), len(datasets)), np.nan)
    for i, source in enumerate(datasets):
        for j, target in enumerate(datasets):
            item = summary["actmap"].get(f"{source}_to_{target}", {}).get("auroc", {})
            if item.get("mean") is not None:
                matrix[i, j] = item["mean"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(matrix, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels([display[d] for d in datasets], rotation=25, ha="right")
    ax.set_yticklabels([display[d] for d in datasets])
    ax.set_xlabel("Target dataset")
    ax.set_ylabel("Source dataset")
    ax.set_title("ActMap transfer AUROC")
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if math.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "actmap_transfer_matrix.pdf")
    fig.savefig(out_dir / "actmap_transfer_matrix.png")
    plt.close(fig)

    transfer_specs = [
        ("ActMap", summary["actmap"]),
        ("DRIFT", summary["white"].get("drift", {})),
        ("Best-layer linear", summary["white"].get("best_layer_linear_probe", {})),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.25), constrained_layout=True)
    images = []
    for ax, (title, block) in zip(axes, transfer_specs):
        values = np.full((len(datasets), len(datasets)), np.nan)
        for i, source in enumerate(datasets):
            for j, target in enumerate(datasets):
                item = block.get(f"{source}_to_{target}", {})
                values[i, j], _ = metric_stats(item, "auroc")
        images.append(
            plot_transfer_heatmap(
                ax,
                values,
                labels=[display[d] for d in datasets],
                title=title,
                highlight_diagonal=True,
            )
        )
    cbar = fig.colorbar(images[-1], ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("AUROC")
    fig.savefig(out_dir / "transfer_auroc_matrices.pdf")
    fig.savefig(out_dir / "transfer_auroc_matrices.png")
    plt.close(fig)

    print(f"saved plots -> {out_dir}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="tmp/final_experiments")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=None)
    parser.add_argument("--seeds", type=parse_seed_tuple, default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reusable final ActMap experiment runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit")
    add_common_args(p)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("prepare-webquestions")
    add_common_args(p)
    p.set_defaults(func=ensure_web_questions_csv)

    p = sub.add_parser("build-webquestions")
    add_common_args(p)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--save-every", type=int, default=100)
    p.set_defaults(func=ensure_web_questions)

    p = sub.add_parser("build-gsm8k")
    add_common_args(p)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--balance-seed", type=int, default=42)
    p.set_defaults(func=ensure_gsm8k)

    p = sub.add_parser("balance-gsm8k")
    add_common_args(p)
    p.add_argument("--balance-seed", type=int, default=42)
    p.set_defaults(func=cmd_balance_gsm8k)

    p = sub.add_parser("make-splits")
    add_common_args(p)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.set_defaults(func=cmd_make_splits)

    p = sub.add_parser("train-actmap")
    add_common_args(p)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--noise-std", type=float, default=0.08)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--require-cuda", action="store_true")
    p.set_defaults(func=cmd_train_actmap)

    p = sub.add_parser("eval-actmap-matrix")
    add_common_args(p)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--require-cuda", action="store_true")
    p.set_defaults(func=cmd_eval_actmap_matrix)

    p = sub.add_parser("extract-white")
    add_common_args(p)
    p.add_argument("--white-batch-size", type=int, default=8)
    p.add_argument("--require-cuda", action="store_true")
    p.set_defaults(func=cmd_extract_white)

    p = sub.add_parser("run-white-matrix")
    add_common_args(p)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--require-cuda", action="store_true")
    p.set_defaults(func=cmd_run_white_matrix)

    p = sub.add_parser("run-black")
    add_common_args(p)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--score-batch-size", type=int, default=128)
    p.set_defaults(func=cmd_run_black)

    p = sub.add_parser("aggregate")
    add_common_args(p)
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("plot")
    add_common_args(p)
    p.add_argument("--output-dir", default="doc/figures/final")
    p.set_defaults(func=cmd_plot)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
