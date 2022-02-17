"""Test cases for uncertainty sampling"""
import unittest
from parameterized import parameterized

import numpy as np
import torch

from query_strategies.uncertainty_sampling_strategy import UncertaintySamplingStrategy


class DataLoaderMock:
    # pylint: disable=no-self-use
    """Mocks the dataloader class"""

    def __init__(self, cases, is_multi_label, num_classes):
        self.cases = cases
        self.is_multi_label = is_multi_label
        self.num_of_classes = num_classes

    def unlabeled_dataloader(self):
        """Mocks the unlabeled dataloader"""
        return [
            # Dataloader return one batch of slices -> batch size 2, 2 classes (background, tumor), image size 240 x 240
            (torch.Tensor(np.zeros((len(self.cases), 2, 240, 240))), self.cases)
        ]

    def num_classes(self):
        """Mocks the number of classes"""
        return self.num_of_classes

    def multi_label(self):
        """Boolean whether data is multi-label"""
        return self.multi_label


class ModelMock:
    # pylint: disable=no-self-use,unused-argument
    """Mocks the model class"""

    def __init__(self, prediction_tensor):
        """Parameterizes the input tensor"""
        self.prediction_tensor = prediction_tensor

    def predict(self, arg):
        """Mocks the model prediction method"""
        # Prediction for 2 slices
        return self.prediction_tensor

    def to(self, device):
        """Mocks the model device handling"""


class UncertaintyTestCase(unittest.TestCase):
    """Test case for uncertainty sampling"""

    def setUp(self) -> None:
        self.uncertainty_cls = UncertaintySamplingStrategy()

    @parameterized.expand(
        [
            (
                # Edge case: two slices have the exact same distance to the maximum uncertainty values
                # In that case the slices are chosen in the sequence they appear in the provided batch
                torch.cat(
                    [
                        torch.Tensor(np.zeros((1, 2, 240, 240))),
                        torch.Tensor(np.ones((1, 2, 240, 240))),
                    ]
                ),
                # Cases or scan-slice combinations of above predictions
                ["case_1", "case_2"],
                # If its multi-label task
                True,
                # Number of classes in the dataset
                2,
                # number of cases to be selected
                1,
                # keyword arguments: calculation method
                {"calculation_method": "distance"},
                ["case_1"],
            ),
            (
                # Test case for distance calculation with perforation of unique scans
                torch.cat(
                    [
                        torch.tensor(np.full((1, 2, 240, 240), 0.6)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.5)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.8)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.7)),
                    ]
                ),
                ["case_1-32", "case_1-33", "case_2-44", "case_2-22"],
                True,
                2,
                3,
                {"calculation_method": "distance", "prefer_unique_scans": True},
                ["case_1-33", "case_2-22", "case_1-32"],
            ),
            (
                # Test case for entropy calculation method
                torch.cat(
                    [
                        torch.tensor(np.full((1, 2, 240, 240), 0.4)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.5)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.7)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.6)),
                    ]
                ),
                ["case_1", "case_2", "case_3", "case_4"],
                True,
                2,
                3,
                {"calculation_method": "entropy"},
                ["case_2", "case_1", "case_4"],
            ),
            (
                # Test case for single-label task with distance calculation method
                torch.cat(
                    [
                        torch.cat(
                            [torch.ones(1, 1, 240, 240), torch.zeros(1, 2, 240, 240)],
                            dim=1,
                        ),
                        torch.ones((1, 3, 240, 240)) / 3,
                        torch.cat(
                            [
                                torch.ones(1, 1, 240, 240) * 0.5,
                                torch.ones(1, 2, 240, 240) * 0.25,
                            ],
                            dim=1,
                        ),
                    ]
                ),
                ["case_1", "case_2", "case_3"],
                False,
                3,
                1,
                {"calculation_method": "distance"},
                ["case_2"],
            ),
        ]
    )
    def test_select_items_to_label(
        self,
        prediction,
        cases,
        is_multi_label,
        num_classes,
        num_cases,
        kwargs,
        expected_cases,
    ):
        """Tests the select items method"""
        actual_cases = self.uncertainty_cls.select_items_to_label(
            models=ModelMock(prediction_tensor=prediction),
            data_module=DataLoaderMock(
                cases=cases, is_multi_label=is_multi_label, num_classes=num_classes
            ),
            items_to_label=num_cases,
            **kwargs
        )
        self.assertListEqual(expected_cases, actual_cases)


if __name__ == "__main__":
    unittest.main()
