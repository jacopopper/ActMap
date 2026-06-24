from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .vit import make_splits


METHODS = ("best_layer_linear_probe", "drift")

DRIFT_PAPER_PROMPT = "Question: {question}\nAnswer:"
DRIFT_PAPER_HF_LAYERS = (1, 5, 9, 13, 17, 21)
DRIFT_QWEN25_ANSWER_TOKEN_ID = 16141
DRIFT_COLON_TOKEN_ID = 25


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fmt_metric(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4f}"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    mask = np.isfinite(scores)
    if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
        return None
    return float(roc_auc_score(labels[mask], scores[mask]))


def auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    mask = np.isfinite(scores)
    if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
        return None
    return float(average_precision_score(labels[mask], scores[mask]))


def ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float | None:
    mask = np.isfinite(probs)
    if mask.sum() == 0:
        return None
    labels = labels[mask]
    probs = probs[mask]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        bucket = (probs >= lo) & (probs < hi)
        if bucket.any():
            total += bucket.sum() * abs(float(labels[bucket].mean()) - float(probs[bucket].mean()))
    return float(total / len(labels))


def summarize_method(
    labels: np.ndarray,
    scores: np.ndarray,
    row_models: np.ndarray,
) -> dict[str, Any]:
    probs = sigmoid(scores)
    summary: dict[str, Any] = {
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "ece": ece(labels, probs),
    }
    per_model: dict[str, Any] = {}
    for model_name in sorted(set(row_models.tolist())):
        mask = row_models == model_name
        per_model[model_name] = {
            "n": int(mask.sum()),
            "auroc": auroc(labels[mask], scores[mask]),
            "auprc": auprc(labels[mask], scores[mask]),
            "ece": ece(labels[mask], probs[mask]),
        }
    summary["per_model"] = per_model
    return summary


def load_rows(path: str | Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise TypeError(f"Expected {path} to contain a list of row dicts, found {type(rows).__name__}")
    return rows


def load_array(path: str | Path, *, key: str | None = None, mmap: bool = True) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r" if mmap else None)
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        if key is not None:
            return payload[key]
        keys = list(payload.keys())
        if len(keys) != 1:
            raise ValueError(f"{path} contains keys {keys}; pass an explicit cache key")
        return payload[keys[0]]
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if key is None:
            array_keys = [k for k, v in obj.items() if hasattr(v, "shape")]
            if len(array_keys) != 1:
                raise ValueError(f"{path} dict keys are {list(obj)}; pass an explicit cache key")
            key = array_keys[0]
        obj = obj[key]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def stack_row_tensor(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot stack features from an empty row list")
    if key not in rows[0]:
        raise KeyError(f"Rows do not contain required key {key!r}")
    values = [row[key] for row in rows]
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values).float().numpy()
    return np.asarray(values, dtype=np.float32)


def labels_for(rows: list[dict[str, Any]] | None, args: argparse.Namespace) -> np.ndarray:
    if args.label_cache:
        return np.asarray(load_array(args.label_cache, key=args.label_key), dtype=np.int64)
    if rows is None:
        raise ValueError("No labels available. Provide --data with row dicts or --label-cache.")
    return np.array([1 if bool(row["is_correct"]) else 0 for row in rows], dtype=np.int64)


def models_for(rows: list[dict[str, Any]] | None, n: int, args: argparse.Namespace) -> np.ndarray:
    if args.model_cache:
        models = load_array(args.model_cache, key=args.model_key, mmap=False)
        if len(models) != n:
            raise ValueError(f"Model cache has {len(models)} entries, expected {n}")
        return np.asarray(models, dtype=object)
    if rows is None:
        return np.array([args.model_name] * n, dtype=object)
    return np.array([str(row.get("model", args.model_name)) for row in rows], dtype=object)


def load_split_indices(
    rows: list[dict[str, Any]] | None,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    if args.split_file:
        with Path(args.split_file).open() as handle:
            payload = json.load(handle)
        return {name: np.array(payload[name], dtype=np.int64) for name in ("train", "val", "test")}

    n = len(labels)
    if args.split_mode == "actmap_random":
        if rows is not None:
            train_idx, val_idx, test_idx = make_splits(rows, args.val_frac, args.test_frac, args.seed)
            return {
                "train": np.array(train_idx, dtype=np.int64),
                "val": np.array(val_idx, dtype=np.int64),
                "test": np.array(test_idx, dtype=np.int64),
            }
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n)
        n_test = int(n * args.test_frac)
        n_val = int(n * args.val_frac)
        return {
            "train": perm[: n - n_val - n_test],
            "val": perm[n - n_val - n_test: n - n_test],
            "test": perm[n - n_test:],
        }

    train_idx, holdout_idx = train_test_split(
        np.arange(n, dtype=np.int64),
        train_size=args.train_frac,
        random_state=args.seed,
        stratify=labels,
    )
    val_share = args.val_frac / (1.0 - args.train_frac)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=val_share,
        random_state=args.seed,
        stratify=labels[holdout_idx],
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def take_rows(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(features[indices], dtype=np.float32)


def resolve_device(args: argparse.Namespace) -> torch.device:
    if (args.require_cuda or args.device == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was specified but CUDA is not available")
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


class TwoLogitMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float, *, activation: str = "leaky_relu"):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def predict_two_logit(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device, non_blocking=True)).detach().cpu().float().numpy()
            chunks.append(logits[:, 1] - logits[:, 0])
    return np.concatenate(chunks).astype(np.float32)


def train_two_logit_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> nn.Module:
    set_seed(seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def find_drift_last_question_position(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    answer_token_id: int = DRIFT_QWEN25_ANSWER_TOKEN_ID,
    colon_token_id: int = DRIFT_COLON_TOKEN_ID,
) -> int:
    if input_ids.ndim != 1:
        raise ValueError("input_ids must be a 1D tensor for one sequence")
    for pos in range(input_ids.numel() - 2, 0, -1):
        if int(input_ids[pos].item()) == answer_token_id and int(input_ids[pos + 1].item()) == colon_token_id:
            return pos - 1
    if attention_mask is not None:
        return max(int(attention_mask.sum().item()) - 3, 0)
    return max(input_ids.numel() - 3, 0)


def extract_drift_paper_features_from_batch(
    hidden_states: tuple[torch.Tensor, ...],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layer_indices: tuple[int, ...] = DRIFT_PAPER_HF_LAYERS,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for batch_i in range(input_ids.shape[0]):
        pos = find_drift_last_question_position(input_ids[batch_i], attention_mask[batch_i])
        parts = [
            hidden_states[layer_i][batch_i, pos, :].detach().cpu().to(torch.float32).numpy()
            for layer_i in layer_indices
        ]
        rows.append(np.concatenate(parts))
    return np.stack(rows, axis=0).astype(np.float32)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_hidden_tuple(value: str) -> tuple[int, ...]:
    return parse_int_tuple(value)


def reshape_flat_layers(features: np.ndarray, layer_count: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D flat layer feature array, got shape {features.shape}")
    if features.shape[1] % layer_count != 0:
        raise ValueError(f"Feature dim {features.shape[1]} is not divisible by layer_count={layer_count}")
    hidden_dim = features.shape[1] // layer_count
    return np.asarray(features, dtype=np.float32).reshape(features.shape[0], layer_count, hidden_dim)


def load_drift_feature_matrix(
    rows: list[dict[str, Any]] | None,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.drift_feature_cache:
        features = load_array(args.drift_feature_cache, key=args.drift_feature_key)
        if features.ndim != 2:
            raise ValueError(f"DRIFT feature cache must be 2D [N,F], got {features.shape}")
        return np.asarray(features, dtype=np.float32)
    if rows is not None and "drift_features" in rows[0]:
        return stack_row_tensor(rows, "drift_features")
    if args.layer_feature_cache:
        layers = load_layer_feature_tensor(rows, args)
        layer_axis = layers.shape[1]
        if layer_axis == len(DRIFT_PAPER_HF_LAYERS):
            selected = layers
        else:
            selected = layers[:, list(DRIFT_PAPER_HF_LAYERS), :]
        return selected.reshape(selected.shape[0], -1).astype(np.float32)
    raise ValueError(
        "DRIFT requires --drift-feature-cache, rows with `drift_features`, "
        "or --layer-feature-cache from which DRIFT layers can be selected."
    )


def load_layer_feature_tensor(
    rows: list[dict[str, Any]] | None,
    args: argparse.Namespace,
) -> np.ndarray:
    source: np.ndarray | None = None
    if args.layer_feature_cache:
        source = load_array(args.layer_feature_cache, key=args.layer_feature_key)
    elif rows is not None:
        for key in ("layer_features", "hidden_features"):
            if key in rows[0]:
                source = stack_row_tensor(rows, key)
                break
        if source is None and "drift_features" in rows[0]:
            source = stack_row_tensor(rows, "drift_features")
    elif args.drift_feature_cache:
        source = load_array(args.drift_feature_cache, key=args.drift_feature_key)

    if source is None:
        raise ValueError(
            "best_layer_linear_probe requires --layer-feature-cache with shape [N,L,D]. "
            "For convenience it can also use DRIFT features [N,6*D], but that only sweeps the six DRIFT layers."
        )

    if source.ndim == 3:
        return source
    if source.ndim == 2:
        layer_count = args.layer_count or args.drift_layer_count
        return reshape_flat_layers(source, layer_count)
    raise ValueError(f"Layer feature source must have shape [N,L,D] or [N,L*D], got {source.shape}")


def fit_best_layer_linear_probe(
    layer_features: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if layer_features.ndim != 3:
        raise ValueError(f"Expected layer features [N,L,D], got {layer_features.shape}")
    device = resolve_device(args)
    y_train = labels[splits["train"]]
    y_val = labels[splits["val"]]
    best_layer = -1
    best_val_auroc = -1.0
    best_scores: np.ndarray | None = None
    layer_metrics: dict[str, Any] = {}

    for layer_i in range(layer_features.shape[1]):
        start = time.time()
        x_train = take_rows(layer_features[:, layer_i, :], splits["train"])
        x_val = take_rows(layer_features[:, layer_i, :], splits["val"])
        x_test = take_rows(layer_features[:, layer_i, :], splits[args.split])
        model = LinearProbe(x_train.shape[1])
        train_two_logit_model(
            model,
            x_train,
            y_train,
            epochs=args.linear_epochs,
            batch_size=args.linear_batch_size,
            lr=args.linear_lr,
            weight_decay=args.linear_weight_decay,
            device=device,
            seed=args.seed,
        )
        val_scores = predict_two_logit(model, x_val, args.linear_batch_size, device)
        val_auroc = auroc(y_val, val_scores)
        layer_metrics[str(layer_i)] = {
            "val_auroc": val_auroc,
            "seconds": time.time() - start,
        }
        if val_auroc is not None and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_layer = layer_i
            best_scores = predict_two_logit(model, x_test, args.linear_batch_size, device)

    if best_scores is None:
        raise RuntimeError("No linear probe layer produced a valid validation AUROC")
    metadata = {
        "paper": "Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs",
        "implementation": "validation-selected affine logistic head on one hidden vector per layer",
        "pooling": "last-answer-token in the paper; uses whatever position was used to build --layer-feature-cache",
        "split": f"stratified {args.train_frac:.2f}/{args.val_frac:.2f}/{1.0 - args.train_frac - args.val_frac:.2f}"
        if args.split_mode == "paper_stratified" else args.split_mode,
        "optimizer": "AdamW",
        "lr": args.linear_lr,
        "weight_decay": args.linear_weight_decay,
        "epochs": args.linear_epochs,
        "batch_size": args.linear_batch_size,
        "selected_layer": best_layer,
        "selected_layer_val_auroc": best_val_auroc,
        "layer_metrics": layer_metrics,
    }
    if args.layer_ids:
        if len(args.layer_ids) != layer_features.shape[1]:
            raise ValueError(
                f"--layer-ids has {len(args.layer_ids)} entries, "
                f"but layer features have {layer_features.shape[1]} layers"
            )
        metadata["layer_ids"] = list(args.layer_ids)
        metadata["selected_layer_id"] = args.layer_ids[best_layer]
    return best_scores.astype(np.float32), metadata


def fit_drift(
    drift_features: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = resolve_device(args)
    x_train = take_rows(drift_features, splits["train"])
    x_val = take_rows(drift_features, splits["val"])
    x_test = take_rows(drift_features, splits[args.split])
    y_train = labels[splits["train"]]
    y_val = labels[splits["val"]]

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    model = TwoLogitMLP(
        x_train.shape[1],
        tuple(args.drift_hidden_dims),
        args.drift_dropout,
        activation="leaky_relu",
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.drift_lr, weight_decay=args.drift_weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=args.drift_batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_acc = -1.0
    stale = 0
    best_epoch = 0
    set_seed(args.seed)
    for epoch in range(1, args.drift_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_scores = predict_two_logit(model, x_val, args.drift_batch_size, device)
        val_acc = float(((val_scores >= 0).astype(np.int64) == y_val).mean())
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.drift_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict_two_logit(model, x_test, args.drift_batch_size, device)
    metadata = {
        "paper": "DRIFT: Detecting Representational Inconsistencies for Factual Truthfulness",
        "implementation": "paper-reproduction MLP over concatenated DRIFT HF layers [1,5,9,13,17,21]",
        "prompt": DRIFT_PAPER_PROMPT,
        "feature_position": "last question token before Answer: in the reproduction cache",
        "hf_layer_indices": list(DRIFT_PAPER_HF_LAYERS),
        "standardized": True,
        "oversampling": False,
        "classifier": f"{drift_features.shape[1]} -> {' -> '.join(map(str, args.drift_hidden_dims))} -> 2",
        "optimizer": "AdamW",
        "lr": args.drift_lr,
        "weight_decay": args.drift_weight_decay,
        "epochs": args.drift_epochs,
        "patience": args.drift_patience,
        "batch_size": args.drift_batch_size,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_acc,
    }
    return scores.astype(np.float32), metadata


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    methods = METHODS if args.methods == ["all"] else tuple(args.methods)
    rows = load_rows(args.data) if args.data else None
    labels = labels_for(rows, args)
    row_models = models_for(rows, len(labels), args)
    splits = load_split_indices(rows, labels, args)
    eval_idx = splits[args.split]
    y_eval = labels[eval_idx]
    model_eval = row_models[eval_idx]

    print(f"Labels: n={len(labels)} correct={int(labels.sum())} incorrect={int((1 - labels).sum())}")
    print(f"Split: train={len(splits['train'])} val={len(splits['val'])} {args.split}={len(eval_idx)}")
    print(f"Methods: {', '.join(methods)}")

    scores: dict[str, np.ndarray] = {}
    method_metadata: dict[str, Any] = {}
    for method in methods:
        start = time.time()
        print(f"\n{method}")
        if method == "best_layer_linear_probe":
            layer_features = load_layer_feature_tensor(rows, args)
            method_scores, metadata = fit_best_layer_linear_probe(layer_features, labels, splits, args)
            if layer_features.shape[1] == args.drift_layer_count:
                metadata["feature_warning"] = (
                    "Using only reshaped DRIFT layers, not an all-layer last-answer-token cache. "
                    "Pass --layer-feature-cache [N,L,D] for the exact paper sweep."
                )
                metadata["feature_layer_ids"] = list(DRIFT_PAPER_HF_LAYERS)
                if metadata["selected_layer"] < len(DRIFT_PAPER_HF_LAYERS):
                    metadata["selected_hf_layer"] = DRIFT_PAPER_HF_LAYERS[metadata["selected_layer"]]
        elif method == "drift":
            drift_features = load_drift_feature_matrix(rows, args)
            method_scores, metadata = fit_drift(drift_features, labels, splits, args)
        else:
            raise ValueError(f"Unknown method: {method}")

        metric = summarize_method(y_eval, method_scores, model_eval)
        print(
            f"  AUROC={fmt_metric(metric['auroc'])} "
            f"AUPRC={fmt_metric(metric['auprc'])} "
            f"ECE={fmt_metric(metric['ece'])} "
            f"[{time.time() - start:.1f}s]"
        )
        scores[method] = method_scores
        method_metadata[method] = metadata

    rows_out: list[dict[str, Any]] = []
    for local_i, row_i in enumerate(eval_idx):
        item: dict[str, Any] = {
            "index": int(row_i),
            "model": str(row_models[row_i]),
            "is_correct": bool(labels[row_i]),
        }
        if rows is not None:
            row = rows[int(row_i)]
            item.update({
                "question": row.get("question"),
                "ground_truth": row.get("ground_truth"),
                "answer": row.get("answer"),
            })
        for method, values in scores.items():
            item[method] = float(values[local_i])
        rows_out.append(item)

    results = {
        "data": str(args.data) if args.data else None,
        "label_cache": str(args.label_cache) if args.label_cache else None,
        "split": args.split,
        "split_file": str(args.split_file) if args.split_file else None,
        "split_mode": args.split_mode,
        "n": int(len(eval_idx)),
        "methods": list(methods),
        "method_metadata": method_metadata,
        "metrics": {method: summarize_method(y_eval, values, model_eval) for method, values in scores.items()},
    }

    if args.scores:
        Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
        torch.save(rows_out, args.scores)
        print(f"\nsaved scores  -> {args.scores}")
    if args.results:
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.results).open("w") as handle:
            json.dump(results, handle, indent=2)
        print(f"saved results -> {args.results}")

    print("\n=== WHITE-BOX BASELINE RESULTS ===")
    for method, metric in results["metrics"].items():
        print(
            f"{method:24s} "
            f"AUROC={fmt_metric(metric['auroc'])} "
            f"AUPRC={fmt_metric(metric['auprc'])} "
            f"ECE={fmt_metric(metric['ece'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="White-box activation baselines: best-layer linear probe and DRIFT."
    )
    parser.add_argument("--data", default=None, help="Optional row .pt file containing labels and cached features")
    parser.add_argument("--label-cache", help="Label cache, e.g. tmp/drift_cache/triviaqa_val_labels.npy")
    parser.add_argument("--label-key", help="Key when --label-cache points to .npz/.pt dict")
    parser.add_argument("--model-cache", help="Optional model-name cache aligned with labels/features")
    parser.add_argument("--model-key", help="Key when --model-cache points to .npz/.pt dict")
    parser.add_argument("--model-name", default="unknown-model", help="Model label used when --data is not provided")
    parser.add_argument("--split-file")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split-mode", choices=("paper_stratified", "actmap_random"), default="paper_stratified")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=("all",) + METHODS, default=["all"])
    parser.add_argument("--scores", default="data/white_scores.pt")
    parser.add_argument("--results", default="data/white_results.json")

    parser.add_argument("--layer-feature-cache", help="Layer hidden-state cache [N,L,D] for best-layer linear probe")
    parser.add_argument("--layer-feature-key", help="Key when --layer-feature-cache points to .npz/.pt dict")
    parser.add_argument("--layer-count", type=int, help="Layer count for flat [N,L*D] layer feature caches")
    parser.add_argument("--layer-ids", type=parse_int_tuple, help="Optional original layer ids for --layer-feature-cache")
    parser.add_argument("--linear-epochs", type=int, default=30)
    parser.add_argument("--linear-batch-size", type=int, default=128)
    parser.add_argument("--linear-lr", type=float, default=1e-3)
    parser.add_argument("--linear-weight-decay", type=float, default=1e-4)

    parser.add_argument("--drift-feature-cache", help="DRIFT feature cache [N,D], typically six concatenated layer vectors")
    parser.add_argument("--drift-feature-key", help="Key when --drift-feature-cache points to .npz/.pt dict")
    parser.add_argument("--drift-layer-count", type=int, default=len(DRIFT_PAPER_HF_LAYERS))
    parser.add_argument("--drift-hidden-dims", type=parse_hidden_tuple, default=(512, 128))
    parser.add_argument("--drift-epochs", type=int, default=40)
    parser.add_argument("--drift-patience", type=int, default=8)
    parser.add_argument("--drift-batch-size", type=int, default=64)
    parser.add_argument("--drift-lr", type=float, default=1e-4)
    parser.add_argument("--drift-weight-decay", type=float, default=1e-3)
    parser.add_argument("--drift-dropout", type=float, default=0.3)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
