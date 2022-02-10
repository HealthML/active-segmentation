import unittest
from parameterized import parameterized

import numpy as np
import torch

from query_strategies.uncertainty_sampling_strategy import UncertaintySamplingStrategy


class DataLoaderMock:
    def unlabeled_dataloader(self):
        return [
            # Dataloader return one batch of slices -> batch size 2, 2 classes (background, tumor), image size 240 x 240
            (torch.Tensor(np.zeros((2, 2, 240, 240))), ["case_1", "case_2"])
        ]

    def num_classes(self):
        return 2


class ModelMock:
    def __init__(self, prediction_tensor):
        self.prediction_tensor = prediction_tensor

    def predict(self, arg):
        # Prediction for 2 slices
        return self.prediction_tensor

    def to(self, device):
        pass


class MyTestCase(unittest.TestCase):
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
                ["case_1"],
            ),
            (
                torch.cat(
                    [
                        torch.tensor(np.full((1, 2, 240, 240), 0.6)),
                        torch.tensor(np.full((1, 2, 240, 240), 0.5)),
                    ]
                ),
                ["case_2"],
            ),
        ]
    )
    def test_select_items_to_label(self, prediction, expected_cases):
        actual_cases = self.uncertainty_cls.select_items_to_label(
            models=ModelMock(prediction_tensor=prediction),
            data_module=DataLoaderMock(),
            items_to_label=1,
            exclude_background=True,
        )
        self.assertListEqual(expected_cases, actual_cases)


if __name__ == "__main__":
    unittest.main()
