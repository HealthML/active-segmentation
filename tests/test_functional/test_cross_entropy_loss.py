"""Tests for the cross-entropy loss."""

import unittest
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
from parameterized import parameterized
import torch

from functional import CrossEntropyLoss
import tests.utils.test_data_cardinality_metrics as test_data


class TestCrossEntropyLoss(unittest.TestCase):
    """
    Test cases for cross-entropy loss.
    """

    @staticmethod
    def loss_module(
        ignore_index: Optional[int] = None,
        multi_label: bool = False,
        reduction: Literal["mean", "sum", "none"] = "none",
        epsilon: float = 0,
    ) -> torch.nn.Module:
        """
        Args:
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            multi_label (bool, optional): Determines if data is multilabel or not (default = `False`).
            reduction (string, optional): `reduction` parameter of the loss.
            epsilon (float, optional): `epsilon` parameter of the loss.

        Returns:
            torch.nn.Module: The loss module to be tested.
        """

        return CrossEntropyLoss(
            multi_label=multi_label,
            ignore_index=ignore_index,
            reduction=reduction,
            epsilon=epsilon,
        )

    @staticmethod
    def _expected_cross_entropy_loss(
        prediction: torch.Tensor, target: torch.Tensor, epsilon: float
    ) -> np.ndarray:
        """
        Computes expected cross-entropy loss.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.
            epsilon (float): Smoothing term used to avoid divisions by zero.

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
                * (target == class_id).astype(np.int)
            )

        return expected_loss

    # pylint: disable=unused-argument
    @staticmethod
    def _expected_binary_cross_entropy_loss(
        prediction: torch.Tensor, target: torch.Tensor, epsilon: float
    ) -> np.ndarray:
        """
        Computes expected binary cross-entropy loss.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.
            epsilon (float): Smoothing term used to avoid divisions by zero.

        Returns:
            numpy.ndarray: Expected loss per pixel / per voxel.
        """

        prediction = prediction.detach().numpy()
        target = target.detach().numpy()

        return -1 * (
            np.log(prediction) * target + np.log(1 - prediction) * (1 - target)
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
    ) -> None:
        """
        Helper function that calculates the cross-entropy loss with the given settings for the given predictions and
        compares it with an expected value.

        Args:
            get_first_slice: Getter function that returns prediction, target and expected metrics for the first slice.
            get_second_slice: Getter function that returns prediction, target and expected metrics for the second slice.
            multi_label (bool): Determines if data is multilabel or not.
            expected_loss (Tensor, optional): Expected loss value.
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            reduction (string, optional): `reduction` parameter of the loss.
            epsilon (float, optional): `epsilon` parameter of the loss.
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
                expected_loss = self._expected_binary_cross_entropy_loss(
                    prediction, target, epsilon
                )
            else:
                expected_loss = self._expected_cross_entropy_loss(
                    prediction, target, epsilon
                )

            expected_loss = torch.from_numpy(expected_loss)

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
        )

        cross_entropy_loss = loss_module(prediction, target)

        self.assertTrue(
            cross_entropy_loss.shape == expected_loss.shape,
            f"Returns cross-entropy loss tensor with correct shape when reduction is {reduction}.",
        )

        task_type = "multi-label" if multi_label else "single-label"

        test_case_description = f"ignore_index is {ignore_index}, reduction is {reduction}, and epsilon is {epsilon}"

        self.assertNotEqual(
            cross_entropy_loss.grad_fn,
            None,
            msg=f"Cross-entropy loss is differentiable for {task_type} tasks when {test_case_description}.",
        )

        torch.testing.assert_allclose(
            cross_entropy_loss,
            expected_loss,
            msg=f"Correctly computes cross-entropy loss for {task_type} tasks when {test_case_description}.",
        )

    # fmt: off
    @parameterized.expand([
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "none", 0),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "mean", 0),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "sum", 0),

        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "none", 1),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "mean", 1),
        (test_data.standard_slice_single_label_1, test_data.standard_slice_single_label_2, False, "sum", 1),

        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "none", 0),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "mean", 0),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "sum", 0),

        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "none", 1),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "mean", 1),
        (test_data.standard_slice_multi_label_1, test_data.standard_slice_multi_label_2, True, "sum", 1),
    ])
    # fmt: on
    def test_standard_case(
        self,
        test_slice_1: torch.Tensor,
        test_slice_2: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
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
        )

    # fmt: off
    @parameterized.expand([
        (test_data.slice_all_true_single_label, False, "none", 0, torch.zeros((2, 3, 3))),
        (test_data.slice_all_true_single_label, False, "mean", 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_single_label, False, "sum", 0, torch.as_tensor(0.0)),

        (test_data.slice_all_true_single_label, False, "none", 1),
        (test_data.slice_all_true_single_label, False, "mean", 1),
        (test_data.slice_all_true_single_label, False, "sum", 1),

        (test_data.slice_all_true_multi_label, True, "none", 0, torch.zeros((2, 3, 3, 3))),
        (test_data.slice_all_true_multi_label, True, "mean", 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 0, torch.as_tensor(0.0)),

        (test_data.slice_all_true_multi_label, True, "none", 1, torch.zeros((2, 3, 3, 3))),
        (test_data.slice_all_true_multi_label, True, "mean", 1, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 1, torch.as_tensor(0.0)),
    ])
    # fmt: on
    def test_all_true(
        self,
        test_slice: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
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
        )

    # fmt: off
    @parameterized.expand([
        (test_data.slice_all_false_single_label, False, "none", 0,  -1 * torch.log(torch.zeros(2, 3, 3))),
        (test_data.slice_all_false_single_label, False, "mean", 0, torch.as_tensor(float("inf"))),
        (test_data.slice_all_false_single_label, False, "sum", 0, torch.as_tensor(float("inf"))),

        (test_data.slice_all_false_single_label, False, "none", 1),
        (test_data.slice_all_false_single_label, False, "mean", 1),
        (test_data.slice_all_false_single_label, False, "sum", 1),

        # PyTorch's BCELoss implementation clamps the loss values to [0, 100]
        (test_data.slice_all_false_multi_label, True, "none", 0, 100 * torch.ones(2, 3, 3, 3)),
        (test_data.slice_all_false_multi_label, True, "mean", 0, torch.as_tensor(100)),
        (test_data.slice_all_false_multi_label, True, "sum", 0,  torch.as_tensor(100 * 2 * 3 * 3 * 3)),

        (test_data.slice_all_false_multi_label, True, "none", 1, 100 * torch.ones(2, 3, 3, 3)),
        (test_data.slice_all_false_multi_label, True, "mean", 1, torch.as_tensor(100)),
        (test_data.slice_all_false_multi_label, True, "sum", 1,  torch.as_tensor(100 * 2 * 3 * 3 * 3)),
    ])
    # fmt: on
    def test_all_false(
        self,
        test_slice: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
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
        )

    # fmt: off
    @parameterized.expand([
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "none", 0),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "mean", 0),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "sum", 0),

        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "none", 1),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "mean", 1),
        (test_data.standard_slice_single_label_1, test_data.slice_ignore_index_single_label, False, "sum", 1),

        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "none", 0),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "mean", 0),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "sum", 0),

        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "none", 1),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "mean", 1),
        (test_data.standard_slice_multi_label_1, test_data.slice_ignore_index_multi_label, True, "sum", 1),
    ])
    # fmt: on
    def test_ignore_index(
        self,
        test_slice_1: torch.Tensor,
        test_slice_2: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
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
        )

    # fmt: off
    @parameterized.expand([
        (test_data.slice_all_true_single_label, False, "none", 0, torch.zeros(2, 3, 3)),
        (test_data.slice_all_true_single_label, False, "mean", 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_single_label, False, "sum", 0, torch.as_tensor(0.0)),

        (test_data.slice_all_true_single_label, False, "none", 1),
        (test_data.slice_all_true_single_label, False, "mean", 1),
        (test_data.slice_all_true_single_label, False, "sum", 1),

        (test_data.slice_all_true_multi_label, True, "none", 0, torch.zeros(2, 3, 3, 3)),
        (test_data.slice_all_true_multi_label, True, "mean", 0, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 0, torch.as_tensor(0.0)),

        (test_data.slice_all_true_multi_label, True, "none", 1, torch.zeros(2, 3, 3, 3)),
        (test_data.slice_all_true_multi_label, True, "mean", 1, torch.as_tensor(0.0)),
        (test_data.slice_all_true_multi_label, True, "sum", 1, torch.as_tensor(0.0)),
    ])
    # fmt: on
    def test_all_true_negative(
        self,
        test_slice: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
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
        )
