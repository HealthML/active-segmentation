"""Test cases for cluster based representativeness sampling"""
import unittest
from parameterized import parameterized

import numpy as np
import torch

from query_strategies.representativeness_sampling_clustering import (
    ClusteringBasedRepresentativenessSamplingStrategy,
)


class DataLoaderMock:
    """Mocks the dataloader class"""

    def __init__(
        self,
        cases_training_set,
        cases_unlabeled_set,
        is_multi_label,
        num_classes,
        batch_size: int = 2,
        image_size: int = 50,
    ):
        self.cases_training_set = cases_training_set
        self.cases_unlabeled_set = cases_unlabeled_set
        self.is_multi_label = is_multi_label
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_size = image_size

    def train_dataloader(self):
        """Mocks the training dataloader"""

        case_id_batches = np.split(np.array(self.cases_training_set), self.batch_size)

        return [
            (
                torch.rand(
                    (
                        self.batch_size,
                        self.image_size,
                        self.image_size,
                    )
                ),
                torch.rand(
                    (
                        self.batch_size,
                        self.num_classes,
                        self.image_size,
                        self.image_size,
                    )
                ),
                None,
                case_id_batch,
            )
            for case_id_batch in case_id_batches
        ]

    def unlabeled_dataloader(self):
        """Mocks the unlabeled dataloader"""

        case_id_batches = np.split(np.array(self.cases_unlabeled_set), self.batch_size)

        return [
            (
                torch.rand(
                    (
                        self.batch_size,
                        self.image_size,
                        self.image_size,
                    )
                ),
                case_id_batch,
            )
            for case_id_batch in case_id_batches
        ]


# pylint: disable=too-few-public-methods
class ModelMock:
    """Mocks the model class"""


class RepresentativenessTestCase(unittest.TestCase):
    """Test case for representativeness sampling"""

    @parameterized.expand(
        [
            (
                # Cases or scan-slice combinations of above predictions
                [
                    "case_1-1",
                    "case_1-3",
                    "case_2-1",
                    "case_2-3",
                ],
                [
                    "case_1-2",
                    "case_1-4",
                    "case_2-2",
                    "case_2-4",
                ],
                # number of cases to be selected
                2,
                # keyword arguments: calculation method
                {"clustering_algorithm": "scans", "feature_dimensionality": 8},
                ["case_1-1", "case_2-1"],
            )
        ]
    )
    def test_select_items_to_label(
        self,
        cases_training_set,
        cases_unlabeled_set,
        num_cases,
        kwargs,
        expected_cases,
    ):
        """Tests the select items method."""
        strategy = ClusteringBasedRepresentativenessSamplingStrategy(**kwargs)
        actual_cases, _ = strategy.select_items_to_label(
            models=ModelMock(),
            data_module=DataLoaderMock(
                cases_training_set=cases_training_set,
                cases_unlabeled_set=cases_unlabeled_set,
                is_multi_label=True,
                num_classes=2,
            ),
            items_to_label=num_cases,
            **kwargs
        )

        for case in expected_cases:
            expected_scan_id = case.split("-")[0]

            actual_scan_ids = [case.split("-")[0] for case in actual_cases]

            self.assertTrue(expected_scan_id in actual_scan_ids)


if __name__ == "__main__":
    unittest.main()
