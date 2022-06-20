"""Tests for the focal loss."""

import unittest
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
from parameterized import parameterized
import torch

from functional import FocalLoss
import tests.utils.test_data_cardinality_metrics as test_data


class TestFocalLoss(unittest.TestCase):
    """
    Test cases for focal loss.
    """

    @staticmethod
    def loss_module(
        ignore_index: Optional[int] = None,
        multi_label: bool = False,
        reduction: Literal["mean", "sum", "none"] = "none",
        epsilon: float = 0,
        gamma: float = 5,
    ) -> torch.nn.Module:
        """
        Args:
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            multi_label (bool, optional): Determines if data is multilabel or not (default = `False`).
            reduction (string, optional): `reduction` parameter of the loss.
            epsilon (float, optional): `epsilon` parameter of the loss.
            gamma (float, optional): `gamma` parameter of the loss.

        Returns:
            torch.nn.Module: The loss module to be tested.
        """

        return FocalLoss(
            multi_label=multi_label,
            ignore_index=ignore_index,
            reduction=reduction,
            epsilon=epsilon,
            gamma=gamma,
        )

    @staticmethod
    def _expected_focal_loss(
        prediction: torch.Tensor, target: torch.Tensor, epsilon: float, gamma: float
    ) -> np.ndarray:
        """
        Computes expected focal loss.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.
            epsilon (float): Smoothing term used to avoid divisions by zero.
            gamma (float): Focal loss gamma parameter.

        Returns:
            numpy.ndarray: Expected loss per pixel / per voxel.
        """

        prediction = prediction.detach().numpy() + epsilon
        target = target.detach().numpy()

        expected_loss = np.zeros(target.shape)

        for class_id in range(prediction.shape[1]):
            expected_loss += (
                -1
                * np.log(prediction[:, class_id])
                * (1 - prediction[:, class_id]) ** gamma
                * (target == class_id).astype(np.int)
            )

        return expected_loss

    # pylint: disable=unused-argument
    @staticmethod
    def _expected_binary_focal_loss(
        prediction: torch.Tensor, target: torch.Tensor, epsilon: float, gamma: float
    ) -> np.ndarray:
        """
        Computes expected binary focal loss.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.
            epsilon (float): Smoothing term used to avoid divisions by zero.
            gamma (float): Focal loss gamma parameter.

        Returns:
            numpy.ndarray: Expected loss per pixel / per voxel.
        """

        prediction = prediction.detach().numpy()
        target = target.detach().numpy()

        return -1 * (
            np.log(prediction) * ((1 - prediction) ** gamma) * target
            + np.log(1 - prediction) * (prediction**gamma) * (1 - target)
        )

    def _test_loss(
        self,
        get_first_slice: Callable[
            [bool],
            Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float],
        ],
        get_second_slice: Callable[
            [bool],
            Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float],
        ],
        multi_label: bool,
        expected_loss: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: Literal["mean", "sum", "none"] = "none",
        epsilon: float = 0,
        gamma: float = 5,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Helper function that calculates the focal loss with the given settings for the given predictions and
        compares it with an expected value.

        Args:
            get_first_slice: Getter function that returns prediction, target and expected metrics for the first slice.
            get_second_slice: Getter function that returns prediction, target and expected metrics for the second slice.
            multi_label (bool): Determines if data is multilabel or not.
            expected_loss (Tensor, optional): Expected loss value.
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            reduction (string, optional): `reduction` parameter of the loss.
            epsilon (float, optional): `epsilon` parameter of the loss.
            gamma (float, optional): `gamma` parameter of the loss.
            weight (Tensor, optional): `weight` parameter of the loss.
        """

        # pylint: disable-msg=too-many-locals

        (
            predictions_first_slice,
            target_first_slice,
            _,
            _,
            _,
        ) = get_first_slice(False)
        (
            predictions_second_slice,
            target_second_slice,
            _,
            _,
            _,
        ) = get_second_slice(False)

        prediction = torch.stack([predictions_first_slice, predictions_second_slice])
        target = torch.stack([target_first_slice, target_second_slice])

        if expected_loss is None:
            if multi_label:
                expected_loss = self._expected_binary_focal_loss(
                    prediction, target, epsilon, gamma
                )
            else:
                expected_loss = self._expected_focal_loss(
                    prediction, target, epsilon, gamma
                )

            expected_loss = torch.from_numpy(expected_loss)

            if weight is not None:
                expanded_weight = weight
                if expected_loss.ndim > 1:
                    for _ in range(1, expected_loss.ndim):
                        expanded_weight = expanded_weight.unsqueeze(axis=-1)
                expected_loss = expected_loss * expanded_weight

            if ignore_index is not None:
                expected_loss = expected_loss * (target != ignore_index)
            if reduction == "mean":
                expected_loss = expected_loss.mean()
            elif reduction == "sum":
                expected_loss = expected_loss.sum()

        prediction = prediction.float()
        target = target.float()

        prediction.requires_grad = True
        target.requires_grad = True

        loss_module = self.loss_module(
            ignore_index=ignore_index,
            reduction=reduction,
            multi_label=multi_label,
            epsilon=epsilon,
            gamma=gamma,
        )

        focal_loss = loss_module(prediction, target, weight)

        self.assertTrue(
            focal_loss.shape == expected_loss.shape,
            f"Returns focal loss tensor with correct shape when reduction is {reduction}.",
        )

        task_type = "multi-label" if multi_label else "single-label"

        test_case_description = f"ignore_index is {ignore_index}, reduction is {reduction}, and epsilon is {epsilon}"

        self.assertNotEqual(
            focal_loss.grad_fn,
            None,
            msg=f"Focal loss is differentiable for {task_type} tasks when {test_case_description}.",
        )

        torch.testing.assert_allclose(
            focal_loss,
            expected_loss,
            msg=f"Correctly computes focal loss for {task_type} tasks when {test_case_description}.",
        )

    # fmt: off
    @parameterized.expand([
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "none", 0, 5),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "mean", 0, 5),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "sum", 0, 5),

        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "none", 1, 5),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "mean", 1, 5),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "sum", 1, 5),

        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "none", 0, 5),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "mean", 0, 5),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "sum", 0, 5),

        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "none", 1, 5),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "mean", 1, 5),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "sum", 1, 5),

        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "none", 0, 0),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "mean", 0, 0),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "sum", 0, 0),

        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "none", 1, 0),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "mean", 1, 0),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "sum", 1, 0),

        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "none", 0, 0),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "mean", 0, 0),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "sum", 0, 0),

        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "none", 1, 0),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "mean", 1, 0),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "sum", 1, 0),
    ])
    # fmt: on
    def test_standard_case(
        self,
        test_slice_1: torch.Tensor,
        test_slice_2: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
        gamma: float,
    ) -> None:
        """
        Tests that the loss is computed correctly tasks when there are both true and false predictions.
        """

        self._test_loss(
            test_slice_1,
            test_slice_2,
            multi_label,
            reduction=reduction,
            epsilon=epsilon,
            gamma=gamma,
        )

    # fmt: off
    @parameterized.expand([
        (test_data.slice_all_true_single_label, False, "none", 0, 5, torch.zeros((2, 3, 3))),
        (test_data.slice_all_true_single_label, False, "mean", 0, 5, torch.as_tensor(0.0)),
        (test_data.slice_all_true_single_label, False, "sum", 0, 5, torch.as_tensor(0.0)),

        (test_data.slice_all_true_single_label, False, "none", 1, 5),
        (test_data.slice_all_true_single_label, False, "mean", 1, 5),
        (test_data.slice_all_true_single_label, False, "sum", 1, 5),

        (test_data.slice_all_true_multi_label, True, "none", 0, 5, torch.zeros((2, 3, 3, 3))),
        (test_data.slice_all_true_multi_label, True, "mean", 0, 5, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 0, 5, torch.as_tensor(0.0)),

        (test_data.slice_all_true_multi_label, True, "none", 1, 5, torch.zeros((2, 3, 3, 3))),
        (test_data.slice_all_true_multi_label, True, "mean", 1, 5, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 1, 5, torch.as_tensor(0.0)),

        (test_data.slice_all_true_single_label, False, "none", 0, 0, torch.zeros((2, 3, 3))),
        (test_data.slice_all_true_single_label, False, "mean", 0, 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_single_label, False, "sum", 0, 0, torch.as_tensor(0.0)),

        (test_data.slice_all_true_single_label, False, "none", 1, 0),
        (test_data.slice_all_true_single_label, False, "mean", 1, 0),
        (test_data.slice_all_true_single_label, False, "sum", 1, 0),

        (test_data.slice_all_true_multi_label, True, "none", 0, 0, torch.zeros((2, 3, 3, 3))),
        (test_data.slice_all_true_multi_label, True, "mean", 0, 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 0, 0, torch.as_tensor(0.0)),

        (test_data.slice_all_true_multi_label, True, "none", 1, 0, torch.zeros((2, 3, 3, 3))),
        (test_data.slice_all_true_multi_label, True, "mean", 1, 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 1, 0, torch.as_tensor(0.0)),
    ])
    # fmt: on
    def test_all_true(
        self,
        test_slice: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
        gamma: float,
        expected_loss: Optional[torch.Tensor] = None,
    ):
        """
        Tests that the loss is computed correctly when all predictions are correct.
        """

        self._test_loss(
            test_slice,
            test_slice,
            multi_label,
            reduction=reduction,
            expected_loss=expected_loss,
            epsilon=epsilon,
            gamma=gamma,
        )

    # fmt: off
    @parameterized.expand([
        (test_data.slice_all_false_single_label, False, "none", 0, 5,  -1 * torch.log(torch.zeros(2, 3, 3))),
        (test_data.slice_all_false_single_label, False, "mean", 0, 5, torch.as_tensor(float("inf"))),
        (test_data.slice_all_false_single_label, False, "sum", 0, 5, torch.as_tensor(float("inf"))),

        (test_data.slice_all_false_single_label, False, "none", 1, 5),
        (test_data.slice_all_false_single_label, False, "mean", 1, 5),
        (test_data.slice_all_false_single_label, False, "sum", 1, 5),

        # PyTorch's BCELoss implementation clamps the loss values to [0, 100]
        (test_data.slice_all_false_multi_label, True, "none", 0, 5, 100 * torch.ones(2, 3, 3, 3)),
        (test_data.slice_all_false_multi_label, True, "mean", 0, 5, torch.as_tensor(100)),
        (test_data.slice_all_false_multi_label, True, "sum", 0, 5,  torch.as_tensor(100 * 2 * 3 * 3 * 3)),

        (test_data.slice_all_false_multi_label, True, "none", 1, 5, 100 * torch.ones(2, 3, 3, 3)),
        (test_data.slice_all_false_multi_label, True, "mean", 1, 5, torch.as_tensor(100)),
        (test_data.slice_all_false_multi_label, True, "sum", 1, 5,  torch.as_tensor(100 * 2 * 3 * 3 * 3)),

        (test_data.slice_all_false_single_label, False, "none", 0, 0,  -1 * torch.log(torch.zeros(2, 3, 3))),
        (test_data.slice_all_false_single_label, False, "mean", 0, 0, torch.as_tensor(float("inf"))),
        (test_data.slice_all_false_single_label, False, "sum", 0, 0, torch.as_tensor(float("inf"))),

        (test_data.slice_all_false_single_label, False, "none", 1, 0),
        (test_data.slice_all_false_single_label, False, "mean", 1, 0),
        (test_data.slice_all_false_single_label, False, "sum", 1, 0),

        # PyTorch's BCELoss implementation clamps the loss values to [0, 100]
        (test_data.slice_all_false_multi_label, True, "none", 0, 0, 100 * torch.ones(2, 3, 3, 3)),
        (test_data.slice_all_false_multi_label, True, "mean", 0, 0, torch.as_tensor(100)),
        (test_data.slice_all_false_multi_label, True, "sum", 0, 0,  torch.as_tensor(100 * 2 * 3 * 3 * 3)),

        (test_data.slice_all_false_multi_label, True, "none", 1, 0, 100 * torch.ones(2, 3, 3, 3)),
        (test_data.slice_all_false_multi_label, True, "mean", 1, 0, torch.as_tensor(100)),
        (test_data.slice_all_false_multi_label, True, "sum", 1, 0,  torch.as_tensor(100 * 2 * 3 * 3 * 3)),
    ])
    # fmt: on
    def test_all_false(
        self,
        test_slice: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
        gamma: float,
        expected_loss: Optional[torch.Tensor] = None,
    ):
        """
        Tests that the loss is computed correctly when all predictions are wrong.
        """

        self._test_loss(
            test_slice,
            test_slice,
            multi_label,
            reduction=reduction,
            expected_loss=expected_loss,
            epsilon=epsilon,
            gamma=gamma,
        )

    # fmt: off
    @parameterized.expand([
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "none", 0, 5),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "mean", 0, 5),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "sum", 0, 5),

        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "none", 1, 5),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "mean", 1, 5),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "sum", 1, 5),

        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "none", 0, 5),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "mean", 0, 5),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "sum", 0, 5),

        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "none", 1, 5),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "mean", 1, 5),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "sum", 1, 5),

        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "none", 0, 0),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "mean", 0, 0),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "sum", 0, 0),

        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "none", 1, 0),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "mean", 1, 0),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "sum", 1, 0),

        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "none", 0, 0),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "mean", 0, 0),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "sum", 0, 0),

        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "none", 1, 0),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "mean", 1, 0),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "sum", 1, 0),
    ])
    # fmt: on
    def test_ignore_index(
        self,
        test_slice_1: torch.Tensor,
        test_slice_2: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
        gamma: float,
    ):
        """
        Tests that the loss is computed correctly when there are are pixels / voxels to be ignored.
        """

        self._test_loss(
            test_slice_1,
            test_slice_2,
            multi_label,
            ignore_index=-1,
            reduction=reduction,
            epsilon=epsilon,
            gamma=gamma,
        )

    # fmt: off
    @parameterized.expand([
        (test_data.slice_all_true_single_label, False, "none", 0, 5, torch.zeros(2, 3, 3)),
        (test_data.slice_all_true_single_label, False, "mean", 0, 5, torch.as_tensor(0.0)),
        (test_data.slice_all_true_single_label, False, "sum", 0, 5, torch.as_tensor(0.0)),

        (test_data.slice_all_true_single_label, False, "none", 1, 5),
        (test_data.slice_all_true_single_label, False, "mean", 1, 5),
        (test_data.slice_all_true_single_label, False, "sum", 1, 5),

        (test_data.slice_all_true_multi_label, True, "none", 0, 5, torch.zeros(2, 3, 3, 3)),
        (test_data.slice_all_true_multi_label, True, "mean", 0, 5, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 0, 5, torch.as_tensor(0.0)),

        (test_data.slice_all_true_multi_label, True, "none", 1, 5, torch.zeros(2, 3, 3, 3)),
        (test_data.slice_all_true_multi_label, True, "mean", 1, 5, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 1, 5, torch.as_tensor(0.0)),

        (test_data.slice_all_true_single_label, False, "none", 0, 0, torch.zeros(2, 3, 3)),
        (test_data.slice_all_true_single_label, False, "mean", 0, 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_single_label, False, "sum", 0, 0, torch.as_tensor(0.0)),

        (test_data.slice_all_true_single_label, False, "none", 1, 0),
        (test_data.slice_all_true_single_label, False, "mean", 1, 0),
        (test_data.slice_all_true_single_label, False, "sum", 1, 0),

        (test_data.slice_all_true_multi_label, True, "none", 0, 0, torch.zeros(2, 3, 3, 3)),
        (test_data.slice_all_true_multi_label, True, "mean", 0, 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 0, 0, torch.as_tensor(0.0)),

        (test_data.slice_all_true_multi_label, True, "none", 1, 0, torch.zeros(2, 3, 3, 3)),
        (test_data.slice_all_true_multi_label, True, "mean", 1, 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 1, 0, torch.as_tensor(0.0)),
    ])
    # fmt: on
    def test_all_true_negative(
        self,
        test_slice: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
        gamma: float,
        expected_loss: Optional[torch.Tensor] = None,
    ):
        """
        Tests that the loss is computed correctly when there are no positives.
        """

        self._test_loss(
            test_slice,
            test_slice,
            multi_label=multi_label,
            reduction=reduction,
            expected_loss=expected_loss,
            epsilon=epsilon,
            gamma=gamma,
        )

    # fmt: off
    @parameterized.expand([
        (test_data.standard_slice_single_label_1, False, "none", 0, 5, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, False, "mean", 0, 5, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, False, "sum", 0, 5, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_single_label_1, False, "none", 0, 5, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_single_label_1, False, "mean", 0, 5, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_single_label_1, False, "sum", 0, 5, torch.Tensor([1.0, 1.0])),

        (test_data.standard_slice_single_label_1, False, "none", 1, 5, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, False, "mean", 1, 5, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, False, "sum", 1, 5, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_single_label_1, False, "none", 1, 5, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_single_label_1, False, "mean", 1, 5, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_single_label_1, False, "sum", 1, 5, torch.Tensor([1.0, 1.0])),

        (test_data.standard_slice_multi_label_1, True, "none", 0, 5, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, True, "mean", 0, 5, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, True, "sum", 0, 5, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_multi_label_1, True, "none", 0, 5, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_multi_label_1, True, "mean", 0, 5, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_multi_label_1, True, "sum", 0, 5, torch.Tensor([1.0, 1.0])),

        (test_data.standard_slice_multi_label_1, True, "none", 1, 5, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, True, "mean", 1, 5, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, True, "sum", 1, 5, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_multi_label_1, True, "none", 1, 5, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_multi_label_1, True, "mean", 1, 5, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_multi_label_1, True, "sum", 1, 5, torch.Tensor([1.0, 1.0])),
    ])
    # fmt: on
    def test_weighted_loss(
        self,
        test_slice: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
        gamma: float,
        weight: torch.Tensor,
    ):
        """
        Tests that the loss is computed correctly when the images of one batch are weighted differently.
        """

        (
            prediction_slice,
            target_slice,
            _,
            _,
            _,
        ) = test_slice(False)

        prediction = torch.stack([prediction_slice, prediction_slice])
        target = torch.stack([target_slice, target_slice])

        if multi_label:
            expected_loss = self._expected_binary_focal_loss(
                prediction, target, epsilon, gamma
            )
        else:
            expected_loss = self._expected_focal_loss(
                prediction, target, epsilon, gamma
            )

        expected_loss = torch.from_numpy(expected_loss)

        for idx, current_weight in enumerate(weight):
            expected_loss[idx] = expected_loss[idx] * current_weight

        if reduction == "mean":
            expected_loss = expected_loss.mean()
        elif reduction == "sum":
            expected_loss = expected_loss.sum()

        self._test_loss(
            test_slice,
            test_slice,
            multi_label=multi_label,
            reduction=reduction,
            weight=weight,
            expected_loss=expected_loss,
            epsilon=epsilon,
            gamma=gamma,
        )
