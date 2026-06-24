from __future__ import annotations

import unittest

from src import experiment_registry as registry


class ExperimentRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = registry.registry_template()
        self.source_text = registry.read_text(registry.DEFAULT_SOURCE_PATH)

    def test_generated_matrix_has_twenty_pairs(self) -> None:
        pairs = registry.generated_pairs(self.registry)

        self.assertEqual(20, len(pairs))
        self.assertEqual(
            registry.parse_source_pairs(self.source_text),
            [(pair["dataset_display_name"], pair["model_hf_id"]) for pair in pairs],
        )

    def test_methods_match_source_once(self) -> None:
        methods = registry.method_names(self.registry)

        self.assertEqual(registry.parse_source_methods(self.source_text), methods)
        self.assertEqual(len(methods), len(set(methods)))

    def test_registry_validates_against_experiments_needed(self) -> None:
        self.assertEqual([], registry.validate_registry(self.registry, self.source_text))

    def test_removed_gsm8k_direct_pair_does_not_exist(self) -> None:
        datasets = self.registry["datasets"]
        pairs = registry.generated_pairs(self.registry)

        self.assertNotIn("gsm8k_direct", {dataset["id"] for dataset in datasets})
        self.assertTrue(any(dataset["id"] == "gsm8k_rationale" for dataset in datasets))
        self.assertFalse(any("gsm8k_direct" in pair["pair_id"] for pair in pairs))
        self.assertEqual(
            4,
            sum(1 for pair in pairs if pair["dataset_id"] == "gsm8k_rationale"),
        )


if __name__ == "__main__":
    unittest.main()
