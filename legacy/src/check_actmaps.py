from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


EXPECTED_COLUMNS = ["question", "ground_truth", "answer", "model", "is_correct", "actmap"]
EXPECTED_SHAPE = (12, 32, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check ActMap training datasets.")
    parser.add_argument("paths", nargs="*", type=Path, default=sorted(Path("data").glob("*.pt")))
    return parser.parse_args()


def summarize(path: Path) -> None:
    import torch

    rows: list[dict[str, Any]] = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== {path} ===")
    print(f"rows: {len(rows)}")
    if not rows:
        return

    print(f"schema_ok: {all(list(row.keys()) == EXPECTED_COLUMNS for row in rows)}")
    print(f"models: {sorted({row['model'] for row in rows})}")
    print(f"correct: {sum(bool(row['is_correct']) for row in rows)}/{len(rows)}")

    actmaps = [row["actmap"] for row in rows]
    print(f"shapes: {sorted({tuple(actmap.shape) for actmap in actmaps})}")
    print(f"dtypes: {sorted({str(actmap.dtype) for actmap in actmaps})}")

    bad_shape = []
    nonfinite = []
    zeroish = []
    for index, actmap in enumerate(actmaps):
        actmap_f = actmap.float()
        if tuple(actmap.shape) != EXPECTED_SHAPE:
            bad_shape.append(index)
        if not torch.isfinite(actmap_f).all().item():
            nonfinite.append(index)
        if actmap_f.std().item() < 1e-6 or actmap_f.abs().mean().item() < 1e-6:
            zeroish.append(index)

    print(f"bad_shape_count: {len(bad_shape)} {bad_shape[:10]}")
    print(f"nonfinite_count: {len(nonfinite)} {nonfinite[:10]}")
    print(f"zeroish_count: {len(zeroish)} {zeroish[:10]}")

    stacked = torch.stack([actmap.float() for actmap in actmaps])
    print(
        "global mean/std/min/max:",
        stacked.mean().item(),
        stacked.std().item(),
        stacked.min().item(),
        stacked.max().item(),
    )
    channel_mean_abs = stacked.abs().mean(dim=(0, 2, 3))
    channel_std = stacked.std(dim=(0, 2, 3))
    print("channel mean_abs:", [round(value, 6) for value in channel_mean_abs.tolist()])
    print("channel std:", [round(value, 6) for value in channel_std.tolist()])

    flat = stacked.flatten(1)
    norms = flat.norm(dim=1)
    print("sample norm range:", norms.min().item(), norms.max().item())
    if len(rows) >= 2:
        cosine01 = torch.nn.functional.cosine_similarity(flat[0:1], flat[1:2]).item()
        print("cosine(sample0,sample1):", cosine01)

    print("first rows:")
    for row in rows[:3]:
        print(
            {
                "question": row["question"][:80],
                "ground_truth": row["ground_truth"],
                "answer": row["answer"],
                "is_correct": row["is_correct"],
                "actmap_shape": tuple(row["actmap"].shape),
            }
        )


def main() -> None:
    args = parse_args()
    for path in args.paths:
        summarize(path)


if __name__ == "__main__":
    main()
