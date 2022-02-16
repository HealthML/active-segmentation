"""Test cases for uncertainty sampling"""
import unittest
from parameterized import parameterized

import numpy as np
import torch

from query_strategies.uncertainty_sampling_strategy import UncertaintySamplingStrategy


class DataLoaderMock:
    # pylint: disable=no-self-use
    """Mocks the dataloader class"""

    def __init__(self, cases):
        self.cases = cases

    def unlabeled_dataloader(self):
        """Mocks the unlabeled dataloader"""
        return [
            # Dataloader return one batch of slices -> batch size 2, 2 classes (background, tumor), image size 240 x 240
            (torch.Tensor(np.zeros((len(self.cases), 2, 240, 240))), self.cases)
        ]

    def num_classes(self):
        """Mocks the number of classes"""
        return 2


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
                torch.cat(
                    [
                        torch.Tensor(np.zeros((1, 2, 240, 240))),
                        torch.Tensor(np.ones((1, 2, 240, 240))),
                    ]
                ),
                ["case_1", "case_2"],
                1,
                "distance",
                ["case_1"],
            ),
            (
                torch.cat(
                    [
                        torch.tensor(np.full((1, 2, 240, 240), 0.6)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.5)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.8)),
                    ]
                ),
                ["case_1", "case_2", "case_3"],
                2,
                "distance",
                ["case_2", "case_1"],
            ),
            (
                torch.cat(
                    [
                        torch.tensor(np.full((1, 2, 240, 240), 0.4)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.5)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.7)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.6)),
                    ]
                ),
                ["case_1", "case_2", "case_3", "case_4"],
                3,
                "entropy",
                ["case_2", "case_1", "case_4"],
            ),
        ]
    )
    def test_select_items_to_label(
        self, prediction, cases, num_cases, calculation_method, expected_cases
    ):
        """Tests the select items method"""
        actual_cases = self.uncertainty_cls.select_items_to_label(
            models=ModelMock(prediction_tensor=prediction),
            data_module=DataLoaderMock(cases=cases),
            items_to_label=num_cases,
            calculation_method=calculation_method,
            exclude_background=True,
        )
        self.assertListEqual(expected_cases, actual_cases)


if __name__ == "__main__":
    unittest.main()
