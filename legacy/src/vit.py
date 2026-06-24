from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset


ARCH_CHOICES = ("vit2d",)
SPATIAL_H = 32
SPATIAL_W = 128


class ProbeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        augment: bool = False,
        noise_std: float = 0.08,
    ):
        self.rows = rows
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        heatmap = row["actmap"].float()
        label = torch.tensor(1.0 if bool(row["is_correct"]) else 0.0)
        if self.augment:
            heatmap = heatmap + torch.randn_like(heatmap) * self.noise_std
        return heatmap, label


def make_splits(
    rows: list[dict[str, Any]],
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    indices = np.arange(len(rows))
    groups = np.array([row.get("question_id", row.get("question", "")) for row in rows])

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_rel, test_pool = next(splitter.split(indices, groups=groups))

    tv_indices = indices[train_val_rel]
    tv_groups = groups[train_val_rel]
    test_idx = indices[test_pool].tolist()

    val_frac_adj = val_frac / (1.0 - test_frac)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_adj, random_state=seed)
    train_rel, val_rel = next(splitter2.split(tv_indices, groups=tv_groups))

    train_idx = tv_indices[train_rel].tolist()
    val_idx = tv_indices[val_rel].tolist()
    return train_idx, val_idx, test_idx


def save_split_indices(
    output_dir: str | Path,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
) -> None:
    split_path = Path(output_dir) / "split_indices.json"
    with split_path.open("w") as handle:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, handle)


def load_split_indices(path: str | Path) -> tuple[list[int], list[int], list[int]]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    return list(payload["train"]), list(payload["val"]), list(payload["test"])


class _PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch_h: int, patch_w: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w),
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
        self, embed_dim: int, num_heads: int, mlp_dim: int,
        dropout: float, attn_drop: float, drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop, batch_first=True,
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
        x = x + self.drop_path1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
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
    ):
        super().__init__()
        assert spatial_h % patch_h == 0 and spatial_w % patch_w == 0
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
        self.blocks = nn.ModuleList([
            _ViTBlockWithDropPath(embed_dim, num_heads, mlp_dim, dropout, attn_drop, dpr[i])
            for i in range(num_layers)
        ])

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
        B = x.size(0)
        x = self.patch_embed(x)
        pos = (self.row_pos + self.col_pos).reshape(1, self.grid_h * self.grid_w, -1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls + self.cls_pos, x + pos], dim=1)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0]).squeeze(1)


def build_model(
    arch: str,
    *,
    in_channels: int,
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
) -> nn.Module:
    if arch == "vit2d":
        return ActMapViT2D(
            in_channels=in_channels,
            patch_h=patch_h, patch_w=patch_w,
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
            mlp_ratio=mlp_ratio, dropout=dropout, attn_drop=attn_drop,
            drop_path_rate=drop_path_rate, spatial_h=spatial_h, spatial_w=spatial_w,
        )
    raise ValueError(f"Unknown architecture: {arch}")


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
    spatial_h: int = SPATIAL_H,
    spatial_w: int = SPATIAL_W,
) -> dict[str, Any]:
    return {
        "arch": arch,
        "in_channels": in_channels,
        "patch_h": patch_h,
        "patch_w": patch_w,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_ratio": mlp_ratio,
        "dropout": dropout,
        "attn_drop": attn_drop,
        "drop_path_rate": drop_path_rate,
        "spatial_h": spatial_h,
        "spatial_w": spatial_w,
    }


def model_label(model_config: dict[str, Any]) -> str:
    arch = model_config["arch"]
    if arch in {"vit", "vit2d"}:
        return (
            f"{arch}  {model_config['in_channels']}ch  "
            f"{model_config['spatial_h']//model_config['patch_h']}x"
            f"{model_config['spatial_w']//model_config['patch_w']} patches  "
            f"{model_config['num_layers']}Lx{model_config['num_heads']}Hx"
            f"{model_config['embed_dim']}D  mlp_ratio={model_config['mlp_ratio']}  "
            f"drop_path={model_config['drop_path_rate']}"
        )
    return f"{arch}  {model_config['in_channels']}ch"


def save_checkpoint(path: Path, model: nn.Module, model_config: dict[str, Any]) -> None:
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": model_config},
        path,
    )


def load_checkpoint_payload(path: str | Path, device: torch.device) -> Any:
    return torch.load(path, map_location=device, weights_only=True)


def checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def checkpoint_model_config(
    payload: Any,
    checkpoint_path: str | Path,
    *,
    in_channels: int,
    fallback_arch: str = "vit",
) -> dict[str, Any]:
    if isinstance(payload, dict) and "model_config" in payload:
        return payload["model_config"]

    config_path = Path(checkpoint_path).with_name("model_config.json")
    if config_path.exists():
        with config_path.open() as handle:
            saved = json.load(handle)
        if "model_config" in saved:
            return saved["model_config"]
        return saved

    return make_model_config(
        fallback_arch,
        in_channels=in_channels,
        patch_h=4, patch_w=16,
        embed_dim=192, num_heads=6, num_layers=6,
        mlp_ratio=3.0, dropout=0.3, attn_drop=0.1,
        drop_path_rate=0.05,
    )


def select_device(require_cuda: bool = False) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was specified but CUDA is not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2,
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


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return roc_auc_score(labels, scores)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return average_precision_score(labels, scores)


def _accuracy(labels: np.ndarray, logits: np.ndarray) -> float:
    return ((logits >= 0).astype(int) == labels.astype(int)).mean().item()


def _ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins_val = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins_val[:-1], bins_val[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        total += mask.sum() * abs(labels[mask].mean() - probs[mask].mean())
    return total / len(labels) if len(labels) > 0 else 0.0


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    mixup_alpha: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(train)
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.set_grad_enabled(train):
        for heatmaps, labels_batch in loader:
            heatmaps = heatmaps.to(device)
            labels_batch = labels_batch.to(device)
            labels_orig = labels_batch.clone()

            if train and mixup_alpha > 0:
                heatmaps, labels_a, labels_b, lam = mixup_batch(heatmaps, labels_batch, mixup_alpha)
                logits = model(heatmaps)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(heatmaps)
                loss = criterion(logits, labels_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels_batch)
            all_logits.append(logits.detach().cpu().float().numpy())
            all_labels.append(labels_orig.cpu().numpy())

    all_logits_arr = np.concatenate(all_logits)
    all_labels_arr = np.concatenate(all_labels)
    return total_loss / len(all_labels_arr), all_logits_arr, all_labels_arr


def _cosine_schedule(
    optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    arch: str = "vit",
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    dropout: float = 0.3,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
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
    split_file: str | Path | None = None,
    require_cuda: bool = False,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_path)
    rows: list[dict[str, Any]] = torch.load(data_path, map_location="cpu", weights_only=False)
    in_channels = rows[0]["actmap"].shape[0]
    n_correct = sum(bool(r["is_correct"]) for r in rows)
    print(f"Loaded {len(rows)} rows  |  in_channels={in_channels}  |  "
          f"correct={n_correct} incorrect={len(rows) - n_correct}")

    if split_file is None:
        train_idx, val_idx, test_idx = make_splits(rows, val_frac, test_frac, seed)
    else:
        train_idx, val_idx, test_idx = load_split_indices(split_file)
        print(f"Using split file: {split_file}")
    save_split_indices(out, train_idx, val_idx, test_idx)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_rows = [rows[i] for i in idx]
        n_pos = sum(bool(r["is_correct"]) for r in split_rows)
        print(f"  {name}: correct={n_pos}, incorrect={len(idx) - n_pos}")

    train_labels_arr = np.array([1.0 if bool(rows[i]["is_correct"]) else 0.0 for i in train_idx])
    n_pos = (train_labels_arr == 1).sum()
    n_neg = (train_labels_arr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"pos_weight: {pos_weight.item():.3f}")

    train_ds = ProbeDataset([rows[i] for i in train_idx], augment=True, noise_std=noise_std)
    val_ds   = ProbeDataset([rows[i] for i in val_idx],   augment=False)
    test_ds  = ProbeDataset([rows[i] for i in test_idx],  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = select_device(require_cuda)
    print(f"Device: {device}")

    model_config = make_model_config(
        arch,
        in_channels=in_channels,
        patch_h=patch_h, patch_w=patch_w,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        mlp_ratio=mlp_ratio, dropout=dropout, attn_drop=attn_drop,
        drop_path_rate=drop_path_rate,
    )
    model = build_model(
        arch,
        in_channels=in_channels,
        patch_h=patch_h, patch_w=patch_w,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        mlp_ratio=mlp_ratio, dropout=dropout, attn_drop=attn_drop,
        drop_path_rate=drop_path_rate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_label(model_config)} - {n_params:,} params")
    with open(out / "model_config.json", "w") as f:
        json.dump({"model_config": model_config}, f, indent=2)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _cosine_schedule(optimizer, warmup_epochs, epochs)

    best_auroc = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    header = f"{'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>9} {'ValAUROC':>9} {'ValAUPRC':>9} {'ValAcc':>8}"
    print(f"\n{header}\n{'-' * len(header)}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, _, _ = _run_epoch(
            model, train_loader, optimizer=optimizer, criterion=criterion,
            device=device, train=True, mixup_alpha=mixup_alpha,
        )
        val_loss, val_logits, val_labels = _run_epoch(
            model, val_loader, optimizer=None, criterion=criterion,
            device=device, train=False,
        )
        scheduler.step()

        val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
        val_auroc = _auroc(val_labels, val_probs)
        val_auprc = _auprc(val_labels, val_probs)
        val_acc   = _accuracy(val_labels.astype(int), val_logits)

        elapsed = time.time() - t0
        print(f"{epoch:>5} {train_loss:>10.4f} {val_loss:>9.4f} "
              f"{val_auroc or 0:>9.4f} {val_auprc or 0:>9.4f} {val_acc:>8.4f}  [{elapsed:.1f}s]")

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            val_auroc=val_auroc, val_auprc=val_auprc, val_acc=val_acc))

        if val_auroc is not None and val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            save_checkpoint(out / "best_model.pt", model, model_config)

        if epoch - best_epoch >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val AUROC {best_auroc:.4f} at epoch {best_epoch})")
            break

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    checkpoint_path = out / "best_model.pt"
    if not checkpoint_path.exists():
        print("\nNo checkpoint saved (single class in val?). Saving final model.")
        save_checkpoint(checkpoint_path, model, model_config)

    print(f"\nLoading best checkpoint (epoch {best_epoch}, val AUROC {best_auroc:.4f})")
    payload = load_checkpoint_payload(checkpoint_path, device)
    model.load_state_dict(checkpoint_state_dict(payload))

    _, test_logits, test_labels = _run_epoch(
        model, test_loader, optimizer=None, criterion=criterion, device=device, train=False,
    )
    test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()

    test_auroc = _auroc(test_labels, test_probs)
    test_auprc = _auprc(test_labels, test_probs)
    test_acc   = _accuracy(test_labels.astype(int), test_logits)
    test_ece   = _ece(test_labels.astype(int), test_probs)

    per_model_results: dict[str, dict[str, Any]] = {}

    test_rows = [rows[i] for i in test_idx]
    model_set = sorted({r["model"] for r in test_rows})
    if len(model_set) > 1:
        print("\n--- Per-model breakdown ---")
        for m_name in model_set:
            m_indices = [i for i, r in enumerate(test_rows) if r["model"] == m_name]
            m_labels = test_labels[m_indices]
            m_probs = test_probs[m_indices]
            m_auroc = _auroc(m_labels, m_probs)
            m_auprc = _auprc(m_labels, m_probs)
            per_model_results[m_name] = {
                "n": len(m_indices),
                "auroc": m_auroc,
                "auprc": m_auprc,
            }
            print(
                f"  {m_name:40s}  n={len(m_indices):4d}  "
                f"AUROC={m_auroc or 0:.4f}  AUPRC={m_auprc or 0:.4f}"
            )

    results = {
        "arch": arch,
        "model_config": model_config,
        "best_epoch": best_epoch, "best_val_auroc": best_auroc,
        "test_auroc": test_auroc, "test_auprc": test_auprc,
        "test_accuracy": test_acc, "test_ece": test_ece,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "in_channels": in_channels,
        "per_model": per_model_results,
        "hyperparams": {
            "lr": lr, "weight_decay": weight_decay, "dropout": dropout,
            "batch_size": batch_size, "noise_std": noise_std,
            "warmup_epochs": warmup_epochs, "mixup_alpha": mixup_alpha,
            "patch_h": patch_h, "patch_w": patch_w, "embed_dim": embed_dim,
            "num_heads": num_heads, "num_layers": num_layers,
            "mlp_ratio": mlp_ratio, "attn_drop": attn_drop,
            "drop_path_rate": drop_path_rate,
        },
    }

    print(f"\n=== TEST RESULTS (n={len(test_labels)}) ===")
    print(f"  AUROC:    {test_auroc:.4f}" if test_auroc is not None else "  AUROC: N/A (single class)")
    print(f"  AUPRC:    {test_auprc:.4f}" if test_auprc is not None else "  AUPRC: N/A")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  ECE:      {test_ece:.4f}")

    with open(out / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved results:    {out / 'test_results.json'}")


def _cmd_train(args: argparse.Namespace) -> None:
    train(
        data_path=args.data, output_dir=args.output,
        arch=args.arch,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        val_frac=args.val_frac, test_frac=args.test_frac,
        seed=args.seed, patience=args.patience,
        noise_std=args.noise_std, warmup_epochs=args.warmup_epochs,
        mixup_alpha=args.mixup_alpha,
        patch_h=args.patch_h, patch_w=args.patch_w,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, mlp_ratio=args.mlp_ratio,
        attn_drop=args.attn_drop, drop_path_rate=args.drop_path_rate,
        split_file=args.split_file,
        require_cuda=args.require_cuda,
    )


def _cmd_test(args: argparse.Namespace) -> None:
    device = select_device(args.require_cuda)
    data_path = Path(args.data)
    rows: list[dict[str, Any]] = torch.load(data_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(rows)} rows from {data_path}")

    if args.split_file is not None:
        with Path(args.split_file).open() as handle:
            splits = json.load(handle)
        indices = splits[args.split]
        rows = [rows[i] for i in indices]
        print(f"Using split '{args.split}' from {args.split_file}: {len(rows)} rows")

    in_channels = rows[0]["actmap"].shape[0]
    payload = load_checkpoint_payload(args.checkpoint, device)
    model_config = checkpoint_model_config(
        payload, args.checkpoint, in_channels=in_channels, fallback_arch=args.arch,
    )
    model = build_model(
        model_config["arch"],
        in_channels=model_config.get("in_channels", in_channels),
        patch_h=model_config.get("patch_h", 4),
        patch_w=model_config.get("patch_w", 16),
        embed_dim=model_config.get("embed_dim", 192),
        num_heads=model_config.get("num_heads", 6),
        num_layers=model_config.get("num_layers", 6),
        mlp_ratio=model_config.get("mlp_ratio", 3.0),
        dropout=model_config.get("dropout", 0.3),
        attn_drop=model_config.get("attn_drop", 0.1),
        drop_path_rate=model_config.get("drop_path_rate", 0.05),
        spatial_h=model_config.get("spatial_h", SPATIAL_H),
        spatial_w=model_config.get("spatial_w", SPATIAL_W),
    ).to(device)
    model.load_state_dict(checkpoint_state_dict(payload))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Model: {model_label(model_config)} - {n_params:,} params")

    ds = ProbeDataset(rows, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss()

    _, logits, labels = _run_epoch(
        model, loader, optimizer=None, criterion=criterion, device=device, train=False,
    )
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    print(f"\n=== INFERENCE RESULTS (n={len(labels)}) ===")
    a = _auroc(labels, probs)
    p = _auprc(labels, probs)
    print(f"  AUROC:    {a:.4f}" if a is not None else "  AUROC: N/A")
    print(f"  AUPRC:    {p:.4f}" if p is not None else "  AUPRC: N/A")
    print(f"  Accuracy: {_accuracy(labels.astype(int), logits):.4f}")
    print(f"  ECE:      {_ece(labels.astype(int), probs):.4f}")

    model_set = sorted({r["model"] for r in rows})
    if len(model_set) > 1:
        print("\n--- Per-model breakdown ---")
        for m_name in model_set:
            m_indices = [i for i, r in enumerate(rows) if r["model"] == m_name]
            m_labels = labels[m_indices]
            m_probs = probs[m_indices]
            m_auroc = _auroc(m_labels, m_probs)
            m_auprc = _auprc(m_labels, m_probs)
            print(
                f"  {m_name:40s}  n={len(m_indices):4d}  "
                f"AUROC={m_auroc or 0:.4f}  AUPRC={m_auprc or 0:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the ActMap ViT probe")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", default="data/train.pt")
    p_train.add_argument("--output", default="checkpoints/")
    p_train.add_argument("--arch", choices=ARCH_CHOICES, default="vit2d")
    p_train.add_argument("--epochs", type=int, default=80)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=0.05)
    p_train.add_argument("--dropout", type=float, default=0.3)
    p_train.add_argument("--val-frac", type=float, default=0.10)
    p_train.add_argument("--test-frac", type=float, default=0.10)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--patience", type=int, default=20)
    p_train.add_argument("--noise-std", type=float, default=0.08)
    p_train.add_argument("--warmup-epochs", type=int, default=5)
    p_train.add_argument("--mixup-alpha", type=float, default=0.2)
    p_train.add_argument("--patch-h", type=int, default=4)
    p_train.add_argument("--patch-w", type=int, default=16)
    p_train.add_argument("--embed-dim", type=int, default=192)
    p_train.add_argument("--num-heads", type=int, default=6)
    p_train.add_argument("--num-layers", type=int, default=6)
    p_train.add_argument("--mlp-ratio", type=float, default=3.0)
    p_train.add_argument("--attn-drop", type=float, default=0.1)
    p_train.add_argument("--drop-path-rate", type=float, default=0.05)
    p_train.add_argument("--split-file", default=None)
    p_train.add_argument("--require-cuda", action="store_true")

    p_test = sub.add_parser("test")
    p_test.add_argument("--data", default="data/train.pt")
    p_test.add_argument("--checkpoint", required=True)
    p_test.add_argument("--arch", choices=ARCH_CHOICES, default="vit2d")
    p_test.add_argument("--batch-size", type=int, default=64)
    p_test.add_argument("--split-file", default=None)
    p_test.add_argument("--split", choices=["train", "val", "test"], default="test")
    p_test.add_argument("--require-cuda", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        _cmd_train(args)
    elif args.command == "test":
        _cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
