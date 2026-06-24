from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "experiments" / "experiment_registry.json"
DEFAULT_MATRIX_PATH = REPO_ROOT / "artifacts" / "orchestration" / "experiment_matrix.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "artifacts" / "orchestration" / "sprint_1_registry_report.md"
DEFAULT_SOURCE_PATH = REPO_ROOT / "EXPERIMENTS_NEEDED.md"


DEFAULT_REGISTRY: dict[str, Any] = {
    "schema_version": 1,
    "source_documents": [
        "EXPERIMENTS_NEEDED.md",
        "experiment-sprints-plan.md",
    ],
    "global": {
        "splits": ["train", "validation", "test"],
        "detector_seeds": [42, 123, 456],
        "actmap": {
            "primary_classifier": "compact_vit2d",
            "map_shape": [12, 32, 128],
            "normalization": {
                "mean": 0.0,
                "standard_deviation": 1.0,
            },
        },
        "calibration": {
            "temperature_selection": "best_per_model",
        },
    },
    "datasets": [
        {
            "id": "triviaqa_no_context",
            "display_name": "TriviaQA no-context",
            "legacy_id": "triviaqa",
            "task_type": "short_answer_qa",
            "context": "none",
            "label_target": "answer_correctness",
            "split_strategy": "question_disjoint_train_validation_test",
        },
        {
            "id": "nq_open",
            "display_name": "NQ-Open",
            "legacy_id": "nq_open",
            "task_type": "short_answer_qa",
            "context": "open_domain",
            "label_target": "answer_correctness",
            "split_strategy": "question_disjoint_train_validation_test",
        },
        {
            "id": "web_questions",
            "display_name": "WebQuestions",
            "legacy_id": "web_questions",
            "task_type": "short_answer_qa",
            "context": "open_domain",
            "label_target": "answer_correctness",
            "split_strategy": "question_disjoint_train_validation_test",
        },
        {
            "id": "gsm8k_rationale",
            "display_name": "GSM8K rationale",
            "legacy_id": "gsm8k",
            "task_type": "math_rationale",
            "context": "word_problem",
            "label_target": "final_numeric_answer_correctness",
            "split_strategy": "question_disjoint_train_validation_test",
        },
        {
            "id": "cnn_dailymail_3_0_0",
            "display_name": "CNN/DailyMail 3.0.0",
            "legacy_id": None,
            "task_type": "summarization_factuality",
            "context": "article",
            "label_target": "summary_factuality",
            "split_strategy": "article_disjoint_train_validation_test",
        },
    ],
    "models": [
        {
            "id": "qwen3_8b",
            "display_name": "Qwen3-8B",
            "hf_id": "Qwen/Qwen3-8B",
            "family": "qwen3",
            "size": "8B",
        },
        {
            "id": "qwen3_32b",
            "display_name": "Qwen3-32B",
            "hf_id": "Qwen/Qwen3-32B",
            "family": "qwen3",
            "size": "32B",
        },
        {
            "id": "llama3_1_8b_instruct",
            "display_name": "Llama-3.1-8B-Instruct",
            "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
            "family": "llama3.1",
            "size": "8B",
        },
        {
            "id": "mistral_7b_instruct_v0_3",
            "display_name": "Mistral-7B-Instruct-v0.3",
            "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "family": "mistral",
            "size": "7B",
        },
    ],
    "methods": [
        {
            "id": "actmap_vit2d",
            "display_name": "ActMap",
            "group": "actmap",
            "requires_training": True,
            "uses_sampling": False,
            "applicability": {"datasets": "all", "models": "all"},
            "description": "Generation-time hidden-state activation maps classified with a compact ViT2D.",
        },
        {
            "id": "semantic_entropy",
            "display_name": "Semantic Entropy",
            "group": "black_box",
            "requires_training": False,
            "uses_sampling": True,
            "applicability": {"datasets": "all", "models": "all"},
        },
        {
            "id": "luq",
            "display_name": "LUQ",
            "group": "black_box",
            "requires_training": False,
            "uses_sampling": True,
            "applicability": {
                "datasets": ["cnn_dailymail_3_0_0"],
                "models": "all",
                "reason": "EXPERIMENTS_NEEDED.md lists LUQ for CNN/DailyMail.",
            },
        },
        {
            "id": "perplexity",
            "display_name": "perplexity",
            "group": "grey_box",
            "requires_training": False,
            "uses_sampling": False,
            "applicability": {"datasets": "all", "models": "all"},
        },
        {
            "id": "mte",
            "display_name": "MTE",
            "group": "grey_box",
            "requires_training": False,
            "uses_sampling": False,
            "applicability": {"datasets": "all", "models": "all"},
        },
        {
            "id": "p_true",
            "display_name": "P(True)",
            "group": "grey_box",
            "requires_training": False,
            "uses_sampling": False,
            "applicability": {"datasets": "all", "models": "all"},
        },
        {
            "id": "factoscope",
            "display_name": "Factoscope",
            "group": "white_box",
            "requires_training": True,
            "uses_sampling": False,
            "applicability": {"datasets": "all", "models": "all"},
        },
        {
            "id": "tad",
            "display_name": "TAD",
            "group": "white_box",
            "requires_training": True,
            "uses_sampling": False,
            "applicability": {"datasets": "all", "models": "all"},
        },
        {
            "id": "rauq",
            "display_name": "RAUQ",
            "group": "white_box",
            "requires_training": True,
            "uses_sampling": False,
            "applicability": {"datasets": "all", "models": "all"},
        },
        {
            "id": "halluguard",
            "display_name": "HalluGuard",
            "group": "white_box",
            "requires_training": True,
            "uses_sampling": False,
            "fallback_method_id": "harp",
            "applicability": {"datasets": "all", "models": "all"},
        },
        {
            "id": "harp",
            "display_name": "HARP",
            "group": "white_box",
            "requires_training": True,
            "uses_sampling": False,
            "is_fallback": True,
            "fallback_for": "halluguard",
            "applicability": {"datasets": "all", "models": "all"},
        },
        {
            "id": "eigenscore_inside",
            "display_name": "EigenScore/INSIDE",
            "group": "white_box",
            "requires_training": False,
            "uses_sampling": True,
            "applicability": {"datasets": "all", "models": "all"},
        },
    ],
    "metrics": [
        {
            "id": "auroc",
            "display_name": "AUROC",
            "higher_is_better": True,
        },
        {
            "id": "auprc",
            "display_name": "AUPRC",
            "higher_is_better": True,
            "include_prevalence_baseline": True,
        },
        {
            "id": "ece_10_bin",
            "display_name": "10-bin ECE",
            "higher_is_better": False,
            "bins": 10,
        },
    ],
}


PAIR_RE = re.compile(r"^- (?P<dataset>.+?) x `(?P<model>[^`]+)`", re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def registry_template() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_REGISTRY)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Could not find section {marker!r}")
    rest = text[start + len(marker):]
    next_heading = rest.find("\n## ")
    if next_heading >= 0:
        rest = rest[:next_heading]
    return rest


def clean_method_name(value: str) -> str:
    value = value.strip()
    value = value.strip("`")
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def parse_source_pairs(source_text: str) -> list[tuple[str, str]]:
    matrix = section(source_text, "Dataset x Model Matrix")
    return [
        (match.group("dataset").strip(), match.group("model").strip())
        for match in PAIR_RE.finditer(matrix)
    ]


def parse_source_methods(source_text: str) -> list[str]:
    methods = section(source_text, "Methods")
    names: list[str] = []
    for line in methods.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        content = stripped[2:].strip()
        if content.endswith(":"):
            continue

        links = MARKDOWN_LINK_RE.findall(content)
        if links:
            names.extend(clean_method_name(link) for link in links)
            continue

        content = re.sub(r"[.;].*$", "", content)
        content = re.sub(r"\s+for\s+.*$", "", content, flags=re.IGNORECASE)
        content = clean_method_name(content)
        if content:
            names.append(content)
    return names


def generated_pairs(registry: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    sequence = 0
    for dataset in registry["datasets"]:
        for model in registry["models"]:
            sequence += 1
            pair_id = f"{dataset['id']}__{model['id']}"
            pairs.append(
                {
                    "sequence": sequence,
                    "pair_id": pair_id,
                    "dataset_id": dataset["id"],
                    "dataset_display_name": dataset["display_name"],
                    "model_id": model["id"],
                    "model_hf_id": model["hf_id"],
                    "display_name": f"{dataset['display_name']} x {model['hf_id']}",
                }
            )
    return pairs


def experiment_matrix(registry: dict[str, Any]) -> dict[str, Any]:
    pairs = generated_pairs(registry)
    return {
        "schema_version": 1,
        "registry": "configs/experiments/experiment_registry.json",
        "pair_generation": "cartesian_product(datasets, models)",
        "dataset_count": len(registry["datasets"]),
        "model_count": len(registry["models"]),
        "pair_count": len(pairs),
        "datasets": [dataset["id"] for dataset in registry["datasets"]],
        "models": [model["id"] for model in registry["models"]],
        "pairs": pairs,
    }


def method_names(registry: dict[str, Any]) -> list[str]:
    return [str(method["display_name"]) for method in registry["methods"]]


def dataset_model_display_pairs(registry: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (pair["dataset_display_name"], pair["model_hf_id"])
        for pair in generated_pairs(registry)
    ]


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def validate_registry(registry: dict[str, Any], source_text: str) -> list[str]:
    errors: list[str] = []

    source_pairs = parse_source_pairs(source_text)
    registry_pairs = dataset_model_display_pairs(registry)
    if len(source_pairs) != 20:
        errors.append(f"EXPERIMENTS_NEEDED.md lists {len(source_pairs)} pairs, expected 20")
    if len(registry_pairs) != 20:
        errors.append(f"registry generates {len(registry_pairs)} pairs, expected 20")
    if registry_pairs != source_pairs:
        errors.append("registry dataset/model matrix does not match EXPERIMENTS_NEEDED.md")

    source_methods = parse_source_methods(source_text)
    registry_methods = method_names(registry)
    if _duplicate_values(registry_methods):
        errors.append(f"registry contains duplicate method display names: {_duplicate_values(registry_methods)}")
    if registry_methods != source_methods:
        errors.append("registry method list does not match EXPERIMENTS_NEEDED.md")

    splits = registry.get("global", {}).get("splits")
    if splits != ["train", "validation", "test"]:
        errors.append(f"registry splits are {splits!r}, expected ['train', 'validation', 'test']")

    seeds = registry.get("global", {}).get("detector_seeds")
    if seeds != [42, 123, 456]:
        errors.append(f"registry detector seeds are {seeds!r}, expected [42, 123, 456]")

    normalization = registry.get("global", {}).get("actmap", {}).get("normalization", {})
    if normalization.get("mean") != 0.0 or normalization.get("standard_deviation") != 1.0:
        errors.append("registry ActMap normalization must be mean 0 and standard deviation 1")

    metric_by_id = {metric.get("id"): metric for metric in registry.get("metrics", [])}
    for metric_id in ("auroc", "auprc", "ece_10_bin"):
        if metric_id not in metric_by_id:
            errors.append(f"registry is missing required metric {metric_id!r}")
    if not metric_by_id.get("auprc", {}).get("include_prevalence_baseline"):
        errors.append("AUPRC metric must include a prevalence baseline")
    if metric_by_id.get("ece_10_bin", {}).get("bins") != 10:
        errors.append("ECE metric must use 10 bins")

    dataset_ids = [dataset.get("id", "") for dataset in registry.get("datasets", [])]
    dataset_names = [dataset.get("display_name", "") for dataset in registry.get("datasets", [])]
    pair_ids = [pair["pair_id"] for pair in generated_pairs(registry)]
    direct_markers = dataset_ids + dataset_names + pair_ids
    if any("gsm8k_direct" in value or "GSM8K direct" in value for value in direct_markers):
        errors.append("registry must not include the removed GSM8K direct dataset or pair")

    groups = {method.get("group") for method in registry.get("methods", [])}
    expected_groups = {"actmap", "black_box", "grey_box", "white_box"}
    if groups != expected_groups:
        errors.append(f"registry method groups are {sorted(groups)}, expected {sorted(expected_groups)}")

    return errors


def build_report(registry: dict[str, Any], validation_errors: list[str]) -> str:
    matrix = experiment_matrix(registry)
    datasets = registry["datasets"]
    models = registry["models"]
    methods = registry["methods"]
    status = "failed" if validation_errors else "succeeded"
    lines = [
        "# Sprint 1 Registry Report",
        "",
        f"Status: {status}",
        "",
        "## Outputs",
        "",
        "- `configs/experiments/experiment_registry.json`",
        "- `artifacts/orchestration/experiment_matrix.json`",
        "- `artifacts/orchestration/sprint_1_registry_report.md`",
        "",
        "## Summary",
        "",
        f"- Datasets: {len(datasets)}",
        f"- Models: {len(models)}",
        f"- Dataset/model pairs: {matrix['pair_count']}",
        f"- Methods: {len(methods)}",
        f"- Metrics: {len(registry['metrics'])}",
        "",
        "## Validation Commands",
        "",
        "```bash",
        "python3 -m src.experiment_registry validate",
        "python3 -m src.experiment_registry list-pairs",
        "python3 -m src.experiment_registry list-methods",
        "python3 -m unittest discover",
        "```",
        "",
        "## Validation Result",
        "",
    ]
    if validation_errors:
        lines.extend(f"- {error}" for error in validation_errors)
        lines.extend(["", "## Remaining Blockers", "", "- Fix the validation errors above before starting Sprint 2."])
    else:
        lines.extend(
            [
                "- Registry matches `EXPERIMENTS_NEEDED.md` for dataset/model pairs and methods.",
                "- The generated matrix contains exactly 20 dataset/model pairs.",
                "- The registry contains train, validation, and test splits.",
                "- The removed GSM8K direct dataset/pair is absent; only GSM8K rationale is registered.",
                "",
                "## Remaining Blockers",
                "",
                "- None for Sprint 1.",
            ]
        )
    return "\n".join(lines) + "\n"


def cmd_generate(args: argparse.Namespace) -> int:
    registry = registry_template()
    source_text = read_text(args.source)
    errors = validate_registry(registry, source_text)
    write_json(args.registry, registry)
    write_json(args.matrix, experiment_matrix(registry))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(registry, errors), encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.registry}")
    print(f"wrote {args.matrix}")
    print(f"wrote {args.report}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    registry = read_json(args.registry)
    source_text = read_text(args.source)
    errors = validate_registry(registry, source_text)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: registry matches EXPERIMENTS_NEEDED.md")
    print(f"pairs: {len(generated_pairs(registry))}")
    print(f"methods: {len(method_names(registry))}")
    return 0


def cmd_list_pairs(args: argparse.Namespace) -> int:
    registry = read_json(args.registry)
    for pair in generated_pairs(registry):
        print(pair["display_name"])
    return 0


def cmd_list_methods(args: argparse.Namespace) -> int:
    registry = read_json(args.registry)
    for name in method_names(registry):
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and validate the ActMap experiment registry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Write the Sprint 1 registry artifacts.")
    generate.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    generate.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    generate.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    generate.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    generate.set_defaults(func=cmd_generate)

    validate = subparsers.add_parser("validate", help="Validate the registry against EXPERIMENTS_NEEDED.md.")
    validate.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    validate.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    validate.set_defaults(func=cmd_validate)

    list_pairs = subparsers.add_parser("list-pairs", help="Print the generated dataset/model pairs.")
    list_pairs.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    list_pairs.set_defaults(func=cmd_list_pairs)

    list_methods = subparsers.add_parser("list-methods", help="Print every registered method once.")
    list_methods.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    list_methods.set_defaults(func=cmd_list_methods)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
