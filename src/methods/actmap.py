from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


ACTMAP_METHOD = "actmap"
ARCH_CHOICES = ("vit2d", "linear", "mlp", "conv2d")
SPATIAL_H = 32
SPATIAL_W = 128
EXPECTED_ACTMAP_SHAPE = (12, SPATIAL_H, SPATIAL_W)


@dataclass(frozen=True)
class ActMapRecord:
    source_record_id: str
    dataset_id: str
    model: str
    split: str
    actmap: torch.Tensor
    label: int
    generated_token_count: int | None = None
    generation: str = ""


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    logits: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class TrainResult:
    seed: int
    output_dir: Path
    checkpoint_path: Path
    prediction_rows: list[dict[str, Any]]
    metrics: dict[str, Any]


class ActMapDataset(Dataset):
    def __init__(
        self,
        rows: list[ActMapRecord],
        *,
        augment: bool = False,
        noise_std: float = 0.08,
    ) -> None:
        self.rows = rows
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        heatmap = row.actmap.float()
        label = torch.tensor(float(row.label), dtype=torch.float32)
        if self.augment:
            heatmap = heatmap + torch.randn_like(heatmap) * self.noise_std
        return heatmap, label


class _PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch_h: int, patch_w: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * mask


class _ViTBlockWithDropPath(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attn_drop: float,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = _DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.drop_path1(self.attn(normalized, normalized, normalized, need_weights=False)[0])
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ActMapViT2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        patch_h: int = 4,
        patch_w: int = 16,
        embed_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 6,
        mlp_ratio: float = 3.0,
        dropout: float = 0.3,
        attn_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        spatial_h: int = SPATIAL_H,
        spatial_w: int = SPATIAL_W,
    ) -> None:
        super().__init__()
        if spatial_h % patch_h != 0 or spatial_w % patch_w != 0:
            raise ValueError("spatial dimensions must be divisible by patch dimensions")
        self.grid_h = spatial_h // patch_h
        self.grid_w = spatial_w // patch_w

        self.patch_embed = _PatchEmbed(in_channels, patch_h, patch_w, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.row_pos = nn.Parameter(torch.zeros(1, self.grid_h, 1, embed_dim))
        self.col_pos = nn.Parameter(torch.zeros(1, 1, self.grid_w, embed_dim))
        self.pos_drop = nn.Dropout(dropout * 0.5)

        mlp_dim = int(embed_dim * mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList(
            [
                _ViTBlockWithDropPath(embed_dim, num_heads, mlp_dim, dropout, attn_drop, dpr[i])
                for i in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(
            self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.size(0), -1)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        nn.init.trunc_normal_(self.row_pos, std=0.02)
        nn.init.trunc_normal_(self.col_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.patch_embed(x)
        pos = (self.row_pos + self.col_pos).reshape(1, self.grid_h * self.grid_w, -1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls + self.cls_pos, x + pos], dim=1)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0]).squeeze(1)


    def forward_with_attentions(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return reliability logits and per-layer, per-head self-attention."""
        batch_size = x.size(0)
        x = self.patch_embed(x)
        pos = (self.row_pos + self.col_pos).reshape(1, self.grid_h * self.grid_w, -1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls + self.cls_pos, x + pos], dim=1)
        x = self.pos_drop(x)
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            normalized = block.norm1(x)
            attn_out, weights = block.attn(
                normalized,
                normalized,
                normalized,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + block.drop_path1(attn_out)
            x = x + block.drop_path2(block.mlp(block.norm2(x)))
            attentions.append(weights)
        x = self.norm(x)
        return self.head(x[:, 0]).squeeze(1), tuple(attentions)

class ActMapLinear(nn.Module):
    def __init__(self, *, in_channels: int, spatial_h: int, spatial_w: int) -> None:
        super().__init__()
        self.head = nn.Linear(in_channels * spatial_h * spatial_w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.flatten(1)).squeeze(1)


class ActMapMLP(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        spatial_h: int,
        spatial_w: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = in_channels * spatial_h * spatial_w
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.flatten(1)).squeeze(1)


class _ResidualConvBlock(nn.Module):
    """A small pre-norm residual block for the fixed ActMap grid."""

    def __init__(self, width: int, dropout: float, drop_path: float) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.activation(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(self.activation(self.norm2(x)))
        return residual + self.drop_path(x)


class ActMapConv2D(nn.Module):
    """Compact multi-scale CNN retaining both depth and temporal locality."""

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int,
        num_layers: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        width = max(32, int(embed_dim))
        groups = 8 if width % 8 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )
        n_blocks = max(1, num_layers)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        self.blocks = nn.Sequential(
            *[_ResidualConvBlock(width, dropout, dpr[i]) for i in range(n_blocks)]
        )
        self.norm = nn.GroupNorm(groups, width)
        hidden = max(64, int(width * mlp_ratio))
        self.head = nn.Sequential(
            nn.Linear(width * (4 * 8 + 1), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.blocks(self.stem(x)))
        batch, channels, height, temporal = x.shape
        if height != 32 or temporal != 128:
            raise ValueError(f"conv2d expects the canonical 32x128 ActMap grid, got {height}x{temporal}")
        local = x.reshape(batch, channels, 4, 8, 8, 16).mean(dim=(4, 5)).flatten(1)
        global_ = x.mean(dim=(2, 3))
        return self.head(torch.cat((local, global_), dim=1)).squeeze(1)

def make_model_config(
    arch: str,
    *,
    in_channels: int,
    patch_h: int,
    patch_w: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    mlp_ratio: float,
    dropout: float,
    attn_drop: float,
    drop_path_rate: float,
    hidden_dim: int | None = None,
    spatial_h: int = SPATIAL_H,
    spatial_w: int = SPATIAL_W,
) -> dict[str, Any]:
    config = {
        "arch": arch,
        "in_channels": int(in_channels),
        "patch_h": int(patch_h),
        "patch_w": int(patch_w),
        "embed_dim": int(embed_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "mlp_ratio": float(mlp_ratio),
        "dropout": float(dropout),
        "attn_drop": float(attn_drop),
        "drop_path_rate": float(drop_path_rate),
        "spatial_h": int(spatial_h),
        "spatial_w": int(spatial_w),
    }
    if arch == "mlp":
        if hidden_dim is None or hidden_dim < 1:
            raise ValueError("MLP hidden_dim must be >= 1")
        config["hidden_dim"] = int(hidden_dim)
    return config


def build_model(arch: str, **kwargs: Any) -> nn.Module:
    if arch == "vit2d":
        keys = {
            "in_channels", "patch_h", "patch_w", "embed_dim", "num_heads",
            "num_layers", "mlp_ratio", "dropout", "attn_drop", "drop_path_rate",
            "spatial_h", "spatial_w",
        }
        return ActMapViT2D(**{key: value for key, value in kwargs.items() if key in keys})
    if arch == "linear":
        return ActMapLinear(
            in_channels=int(kwargs["in_channels"]),
            spatial_h=int(kwargs["spatial_h"]),
            spatial_w=int(kwargs["spatial_w"]),
        )
    if arch == "mlp":
        return ActMapMLP(
            in_channels=int(kwargs["in_channels"]),
            spatial_h=int(kwargs["spatial_h"]),
            spatial_w=int(kwargs["spatial_w"]),
            hidden_dim=int(kwargs["hidden_dim"]),
            dropout=float(kwargs["dropout"]),
        )
    if arch == "conv2d":
        return ActMapConv2D(
            in_channels=int(kwargs["in_channels"]),
            embed_dim=int(kwargs["embed_dim"]),
            num_layers=int(kwargs["num_layers"]),
            mlp_ratio=float(kwargs["mlp_ratio"]),
            dropout=float(kwargs["dropout"]),
            drop_path_rate=float(kwargs["drop_path_rate"]),
        )
    raise ValueError(f"Unknown ActMap architecture: {arch}")


def model_label(model_config: dict[str, Any]) -> str:
    arch = model_config["arch"]
    if arch == "linear":
        return (
            f"linear {model_config['in_channels']}ch "
            f"{model_config['spatial_h']}x{model_config['spatial_w']}"
        )
    if arch == "mlp":
        return (
            f"mlp {model_config['in_channels']}ch "
            f"{model_config['spatial_h']}x{model_config['spatial_w']} "
            f"hidden={model_config['hidden_dim']}"
        )
    if arch == "conv2d":
        return (
            f"conv2d {model_config['in_channels']}ch "
            f"width={model_config['embed_dim']} {model_config['num_layers']}L "
            f"multiscale-pool mlp_ratio={model_config['mlp_ratio']}"
        )
    return (
        f"{arch} {model_config['in_channels']}ch "
        f"{model_config['spatial_h'] // model_config['patch_h']}x"
        f"{model_config['spatial_w'] // model_config['patch_w']} patches "
        f"{model_config['num_layers']}Lx{model_config['num_heads']}Hx"
        f"{model_config['embed_dim']}D mlp_ratio={model_config['mlp_ratio']} "
        f"drop_path={model_config['drop_path_rate']}"
    )


def set_reproducible(seed: int, *, deterministic: bool = True, allow_tf32: bool = False) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(require_cuda: bool = False) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(path: Path, model: nn.Module, model_config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_config": model_config}, path)


def load_checkpoint_payload(path: str | Path, device: torch.device) -> Any:
    return torch.load(path, map_location=device, weights_only=True)


def checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(average_precision_score(labels, scores))


def accuracy(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(((logits >= 0).astype(int) == labels.astype(int)).mean().item())


def ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins_val = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins_val[:-1], bins_val[1:]):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        total += mask.sum() * abs(labels[mask].mean() - probs[mask].mean())
    return float(total / len(labels)) if len(labels) > 0 else 0.0


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    mixup_alpha: float = 0.0,
) -> EpochMetrics:
    model.train(train)
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.set_grad_enabled(train):
        for heatmaps, labels_batch in loader:
            heatmaps = heatmaps.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            labels_orig = labels_batch.clone()

            if train and mixup_alpha > 0:
                heatmaps, labels_a, labels_b, lam = mixup_batch(heatmaps, labels_batch, mixup_alpha)
                logits = model(heatmaps)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(heatmaps)
                loss = criterion(logits, labels_batch)

            if train:
                if optimizer is None:
                    raise RuntimeError("optimizer is required for training")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels_batch)
            all_logits.append(logits.detach().cpu().float().numpy())
            all_labels.append(labels_orig.detach().cpu().numpy())

    logits_arr = np.concatenate(all_logits) if all_logits else np.array([], dtype=np.float32)
    labels_arr = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    loss = total_loss / max(1, len(labels_arr))
    return EpochMetrics(loss=loss, logits=logits_arr, labels=labels_arr)


def make_loader(
    rows: list[ActMapRecord],
    *,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    noise_std: float,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ActMapDataset(rows, augment=augment, noise_std=noise_std),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=False,
    )


def labels_array(rows: list[ActMapRecord]) -> np.ndarray:
    return np.array([int(row.label) for row in rows], dtype=np.int64)


def split_summary(rows: list[ActMapRecord]) -> dict[str, Any]:
    labels = labels_array(rows)
    return {
        "n": int(len(rows)),
        "n_positive": int(labels.sum()) if len(labels) else 0,
        "n_negative": int((labels == 0).sum()) if len(labels) else 0,
    }


def metric_payload(labels: np.ndarray, logits: np.ndarray, *, bins: int) -> dict[str, Any]:
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    return {
        "auroc": auroc(labels, probs),
        "auprc": auprc(labels, probs),
        "test_accuracy": accuracy(labels, logits),
        "ece": ece(labels, probs, n_bins=bins),
    }


def build_prediction_rows(
    rows: list[ActMapRecord],
    logits: np.ndarray,
    *,
    seed: int,
    dataset_id: str,
    model_id: str,
    balance_mode: str,
    split: str = "test",
) -> list[dict[str, Any]]:
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    payloads: list[dict[str, Any]] = []
    for row, logit, prob in zip(rows, logits, probs, strict=True):
        payloads.append(
            {
                "method": ACTMAP_METHOD,
                "method_group": "actmap",
                "dataset_id": dataset_id,
                "model": model_id,
                "source_record_id": row.source_record_id,
                "split": split,
                "label": int(row.label),
                "score": float(logit),
                "probability": float(prob),
                "seed": int(seed),
                "balance_mode": balance_mode,
                "generated_token_count": row.generated_token_count,
            }
        )
    return payloads


def train_one_seed(
    *,
    train_rows: list[ActMapRecord],
    validation_rows: list[ActMapRecord],
    test_rows: list[ActMapRecord],
    output_dir: Path,
    seed: int,
    dataset_id: str,
    model_id: str,
    balance_mode: str,
    metadata: dict[str, Any],
    arch: str = "vit2d",
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    dropout: float = 0.3,
    patience: int = 20,
    noise_std: float = 0.08,
    warmup_epochs: int = 5,
    mixup_alpha: float = 0.2,
    patch_h: int = 4,
    patch_w: int = 16,
    embed_dim: int = 192,
    num_heads: int = 6,
    num_layers: int = 6,
    mlp_ratio: float = 3.0,
    attn_drop: float = 0.1,
    drop_path_rate: float = 0.05,
    hidden_dim: int = 128,
    num_workers: int = 4,
    require_cuda: bool = False,
    deterministic: bool = True,
    allow_tf32: bool = False,
    ece_bins: int = 10,
) -> TrainResult:
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("train, validation, and test rows must all be nonempty")
    set_reproducible(seed, deterministic=deterministic, allow_tf32=allow_tf32)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_channels, spatial_h, spatial_w = (int(value) for value in train_rows[0].actmap.shape)
    model_config = make_model_config(
        arch,
        in_channels=in_channels,
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_drop=attn_drop,
        drop_path_rate=drop_path_rate,
        hidden_dim=hidden_dim,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
    )

    train_labels = labels_array(train_rows)
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    save_json(
        output_dir / "run_config.json",
        {
            "seed": seed,
            "dataset_id": dataset_id,
            "model": model_id,
            "balance_mode": balance_mode,
            "model_config": model_config,
            "splits": {
                "train": split_summary(train_rows),
                "validation": split_summary(validation_rows),
                "test": split_summary(test_rows),
            },
            "hyperparams": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "patience": patience,
                "noise_std": noise_std,
                "warmup_epochs": warmup_epochs,
                "mixup_alpha": mixup_alpha,
                "hidden_dim": hidden_dim,
                "num_workers": num_workers,
                "deterministic": deterministic,
                "allow_tf32": allow_tf32,
                "pos_weight": float(pos_weight.item()),
            },
            "metadata": metadata,
        },
    )
    save_json(output_dir / "model_config.json", {"model_config": model_config})

    train_loader = make_loader(
        train_rows,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        noise_std=noise_std,
        num_workers=num_workers,
        seed=seed,
    )
    validation_loader = make_loader(
        validation_rows,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        noise_std=noise_std,
        num_workers=num_workers,
        seed=seed + 1,
    )
    test_loader = make_loader(
        test_rows,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        noise_std=noise_std,
        num_workers=num_workers,
        seed=seed + 2,
    )

    device = select_device(require_cuda)
    model = build_model(arch, **{k: v for k, v in model_config.items() if k != "arch"}).to(device)
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = cosine_schedule(optimizer, warmup_epochs, epochs)

    print(
        f"ActMap seed={seed} dataset={dataset_id} model={model_id} device={device} "
        f"train={len(train_rows)} val={len(validation_rows)} test={len(test_rows)}"
    )
    print(f"Model: {model_label(model_config)} - {n_params:,} params")

    best_auroc = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best_model.pt"
    header = f"{'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>9} {'ValAUROC':>9} {'ValAUPRC':>9} {'ValAcc':>8}"
    print(f"\n{header}\n{'-' * len(header)}")

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            mixup_alpha=mixup_alpha,
        )
        val_metrics = run_epoch(
            model,
            validation_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            train=False,
        )
        scheduler.step()

        val_probs = torch.sigmoid(torch.tensor(val_metrics.logits)).numpy()
        val_labels = val_metrics.labels.astype(int)
        val_auroc = auroc(val_labels, val_probs)
        val_auprc = auprc(val_labels, val_probs)
        val_acc = accuracy(val_labels, val_metrics.logits)
        elapsed = time.time() - started
        print(
            f"{epoch:>5} {train_metrics.loss:>10.4f} {val_metrics.loss:>9.4f} "
            f"{val_auroc or 0:>9.4f} {val_auprc or 0:>9.4f} {val_acc:>8.4f} [{elapsed:.1f}s]",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "val_loss": val_metrics.loss,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
                "val_acc": val_acc,
                "lr": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": elapsed,
            }
        )
        if val_auroc is not None and val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            save_checkpoint(checkpoint_path, model, model_config)
        if epoch - best_epoch >= patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(best val AUROC {best_auroc:.4f} at epoch {best_epoch})",
                flush=True,
            )
            break

    save_json(output_dir / "history.json", history)
    if not checkpoint_path.exists():
        save_checkpoint(checkpoint_path, model, model_config)

    payload = load_checkpoint_payload(checkpoint_path, device)
    model.load_state_dict(checkpoint_state_dict(payload))
    test_epoch = run_epoch(
        model,
        test_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
    )
    test_labels = test_epoch.labels.astype(int)
    test_metrics = metric_payload(test_labels, test_epoch.logits, bins=ece_bins)
    prediction_rows = build_prediction_rows(
        test_rows,
        test_epoch.logits,
        seed=seed,
        dataset_id=dataset_id,
        model_id=model_id,
        balance_mode=balance_mode,
    )
    result = {
        "method": ACTMAP_METHOD,
        "method_group": "actmap",
        "dataset_id": dataset_id,
        "model": model_id,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_val_auroc": None if best_auroc < 0 else float(best_auroc),
        "n_train": len(train_rows),
        "n_validation": len(validation_rows),
        "n_test": len(test_rows),
        "n_positive": int(test_labels.sum()),
        "n_negative": int((test_labels == 0).sum()),
        "auroc": test_metrics["auroc"],
        "auprc": test_metrics["auprc"],
        "ece": test_metrics["ece"],
        "test_accuracy": test_metrics["test_accuracy"],
        "accuracy_rate": test_metrics["test_accuracy"],
        "checkpoint_path": str(checkpoint_path),
        "model_config": model_config,
        "hyperparams": {
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "batch_size": batch_size,
            "noise_std": noise_std,
            "warmup_epochs": warmup_epochs,
            "mixup_alpha": mixup_alpha,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "mlp_ratio": mlp_ratio,
            "attn_drop": attn_drop,
            "drop_path_rate": drop_path_rate,
            "num_workers": num_workers,
            "deterministic": deterministic,
            "allow_tf32": allow_tf32,
        },
        "metadata": metadata,
    }
    save_json(output_dir / "test_results.json", result)
    print(
        f"TEST seed={seed} AUROC={result['auroc'] if result['auroc'] is not None else 'NA'} "
        f"AUPRC={result['auprc'] if result['auprc'] is not None else 'NA'} ECE={result['ece']:.4f}",
        flush=True,
    )
    return TrainResult(
        seed=seed,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        prediction_rows=prediction_rows,
        metrics=result,
    )
