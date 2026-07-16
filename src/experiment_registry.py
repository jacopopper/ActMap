from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "replication" / "actmap_registry.json"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    datasets = registry.get("datasets", [])
    models = registry.get("models", [])
    methods = registry.get("methods", [])

    if not datasets or any(not item.get("id") or not item.get("hf_id") for item in datasets):
        errors.append("every dataset needs id and hf_id")
    if not models or any(not item.get("id") or not item.get("hf_id") for item in models):
        errors.append("every model needs id and hf_id")
    if [item.get("id") for item in methods] != ["actmap_vit2d"]:
        errors.append("the public registry must contain only the ActMap method")

    global_config = registry.get("global", {})
    if global_config.get("splits") != ["train", "validation", "test"]:
        errors.append("splits must be train, validation, and test")
    if global_config.get("detector_seeds") != [42, 123, 456]:
        errors.append("detector seeds must be 42, 123, and 456")
    shape = global_config.get("actmap", {}).get("map_shape")
    if shape != [12, 32, 128]:
        errors.append("ActMap shape must be 12 x 32 x 128")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the public ActMap replication registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args(argv)
    registry = read_json(args.registry)
    errors = validate_registry(registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {len(registry['datasets'])} datasets, {len(registry['models'])} models, ActMap only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
