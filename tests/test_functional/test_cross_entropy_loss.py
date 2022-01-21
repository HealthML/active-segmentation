"""Tests for the cross-entropy loss."""

import unittest
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
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
    ) -> torch.nn.Module:
        """
        Args:
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            multi_label (bool, optional): Determines if data is multilabel or not (default = `False`).
            reduction (string, optional): `reduction` parameter of the loss.

        Returns:
            torch.nn.Module: The loss module to be tested.
        """

        return CrossEntropyLoss(
            multi_label=multi_label,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    @staticmethod
    def _expected_cross_entropy_loss(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> np.ndarray:
        """
        Computes expected cross-entropy loss.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.

        Returns:
            numpy.ndarray: Expected loss per pixel / per voxel.
        """

        prediction = prediction.detach().numpy()
        target = target.detach().numpy()

        expected_loss = np.zeros(target.shape)

        for class_id in range(prediction.shape[1]):
            expected_loss += (
                -1
                * np.log(prediction[:, class_id])
                * (target == class_id).astype(np.int)
            )

        return expected_loss

    @staticmethod
    def _expected_binary_cross_entropy_loss(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> np.ndarray:
        """
        Computes expected binary cross-entropy loss.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.

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
                    prediction, target
                )
            else:
                expected_loss = self._expected_cross_entropy_loss(prediction, target)

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
        )

        cross_entropy_loss = loss_module(prediction, target)

        self.assertTrue(
            cross_entropy_loss.shape == expected_loss.shape,
            f"Returns cross-entropy loss tensor with correct shape when reduction is {reduction}.",
        )

        test_case_description = (
            f"ignore_index is {ignore_index}, and reduction is {reduction}"
        )

        self.assertNotEqual(
            cross_entropy_loss.grad_fn,
            None,
            msg=f"Cross-entropy loss is differentiable when {test_case_description}.",
        )

        torch.testing.assert_allclose(
            cross_entropy_loss,
            expected_loss,
            msg=f"Correctly computes cross-entropy loss value when {test_case_description}.",
        )

    def test_standard_case(self) -> None:
        """
        Tests that the loss is computed correctly tasks when there are both true and false predictions.
        """

        for test_slice_1, test_slice_2, multi_label in [
            (
                test_data.standard_slice_single_label_1,
                test_data.standard_slice_single_label_2,
                False,
            ),
            (
                test_data.standard_slice_multi_label_1,
                test_data.standard_slice_multi_label_2,
                True,
            ),
        ]:
            for reduction in ["none", "mean", "sum"]:
                self._test_loss(
                    test_slice_1,
                    test_slice_2,
                    multi_label,
                    reduction=reduction,
                )

    def test_all_true(self):
        """
        Tests that the loss is computed correctly when all predictions are correct.
        """

        for test_slice, multi_label in [
            (test_data.slice_all_true_single_label, False),
            (test_data.slice_all_true_multi_label, True),
        ]:

            if multi_label:
                expected_loss = torch.zeros((2, 3, 3, 3))
            else:
                expected_loss = torch.zeros((2, 3, 3))

            self._test_loss(
                test_slice,
                test_slice,
                multi_label,
                reduction="none",
                expected_loss=expected_loss,
            )

            for reduction in ["mean", "sum"]:
                self._test_loss(
                    test_slice,
                    test_slice,
                    multi_label,
                    reduction=reduction,
                    expected_loss=torch.as_tensor(0.0),
                )

    def test_all_false(self):
        """
        Tests that the loss is computed correctly when all predictions are wrong.
        """

        for test_slice, multi_label in [
            (test_data.slice_all_false_single_label, False),
            (test_data.slice_all_false_multi_label, True),
        ]:
            if multi_label:
                # PyTorch's BCELoss implementation clamps the loss values to [0, 100]
                expected_loss_no_reduction = 100 * torch.ones(2, 3, 3, 3)
                expected_loss_mean = torch.as_tensor(100)
                expected_loss_sum = torch.as_tensor(100 * 2 * 3 * 3 * 3)
            else:
                expected_loss_no_reduction = -1 * torch.log(torch.zeros(2, 3, 3))
                expected_loss_mean = torch.as_tensor(float("inf"))
                expected_loss_sum = torch.as_tensor(float("inf"))

            self._test_loss(
                test_slice,
                test_slice,
                multi_label,
                reduction="none",
                expected_loss=expected_loss_no_reduction,
            )
            self._test_loss(
                test_slice,
                test_slice,
                multi_label,
                reduction="mean",
                expected_loss=expected_loss_mean,
            )
            self._test_loss(
                test_slice,
                test_slice,
                multi_label,
                reduction="sum",
                expected_loss=expected_loss_sum,
            )

    def test_ignore_index(self):
        """
        Tests that the loss is computed correctly when there are are pixels / voxels to be ignored.
        """

        for test_slice_1, test_slice_2, multi_label in [
            (
                test_data.standard_slice_single_label_1,
                test_data.slice_ignore_index_single_label,
                False,
            ),
            (
                test_data.standard_slice_multi_label_1,
                test_data.slice_ignore_index_multi_label,
                True,
            ),
        ]:
            for reduction in ["none", "mean", "sum"]:
                self._test_loss(
                    test_slice_1,
                    test_slice_2,
                    multi_label,
                    ignore_index=-1,
                    reduction=reduction,
                )

    def test_all_true_negative(self):
        """
        Tests that the loss is computed correctly when there are no positives.
        """

        for test_slice, multi_label in [
            (test_data.slice_all_true_negatives_single_label, False),
            (test_data.slice_all_true_negatives_multi_label, True),
        ]:
            self._test_loss(
                test_slice,
                test_slice,
                multi_label=multi_label,
                reduction="none",
                expected_loss=torch.zeros(2, 3, 3, 3)
                if multi_label
                else torch.zeros(2, 3, 3),
            )

            for reduction in ["mean", "sum"]:
                self._test_loss(
                    test_slice,
                    test_slice,
                    multi_label=multi_label,
                    reduction=reduction,
                    expected_loss=torch.as_tensor(0.0),
                )

    def test_3d(self):
        """
        Tests that the loss is computed correctly when the inputs are 3d images.
        """

        # pylint: disable-msg=too-many-locals

        for test_slice_1, test_slice_2, multi_label in [
            (
                test_data.standard_slice_single_label_1,
                test_data.standard_slice_single_label_2,
                False,
            ),
            (
                test_data.standard_slice_multi_label_1,
                test_data.standard_slice_multi_label_2,
                True,
            ),
        ]:

            (
                prediction_1,
                target_1,
                _,
                probability_positive_1,
                probability_negative_1,
            ) = test_slice_1(False)

            (
                prediction_2,
                target_2,
                _,
                probability_positive_2,
                probability_negative_2,
            ) = test_slice_2(False)

            assert probability_positive_1 == probability_positive_2
            assert probability_negative_1 == probability_negative_2

            prediction = torch.stack([prediction_1, prediction_2])
            target = torch.stack([target_1, target_2])

            # ensure that the class channel is the first dimension
            prediction = prediction.swapaxes(1, 0)
            target = target.swapaxes(1, 0) if multi_label is True else target

            # create batch dimension
            prediction = prediction.unsqueeze(dim=0)
            target = target.unsqueeze(dim=0)

            for reduction in ["none", "mean", "sum"]:
                if multi_label:
                    expected_loss = self._expected_binary_cross_entropy_loss(
                        prediction, target
                    )
                else:
                    expected_loss = self._expected_cross_entropy_loss(
                        prediction, target
                    )
                expected_loss = torch.from_numpy(expected_loss)

                if reduction == "mean":
                    expected_loss = expected_loss.mean()
                elif reduction == "sum":
                    expected_loss = expected_loss.sum()

                prediction = prediction.float()
                target = target.float()

                prediction.requires_grad = True
                target.requires_grad = True

                cross_entropy_loss_module = CrossEntropyLoss(
                    multi_label=multi_label,
                    ignore_index=None,
                    reduction=reduction,
                )

                cross_entropy_loss = cross_entropy_loss_module(prediction, target)

                test_case_description = f"reduction is {reduction}"

                self.assertTrue(
                    cross_entropy_loss.shape == expected_loss.shape,
                    f"Returns cross-entropy loss tensor with correct shape when {test_case_description}.",
                )

                self.assertNotEqual(
                    cross_entropy_loss.grad_fn,
                    None,
                    msg=f"Cross-entropy loss is differentiable when {test_case_description}.",
                )

                torch.testing.assert_allclose(
                    cross_entropy_loss,
                    expected_loss,
                    msg=f"Correctly computes cross-entropy loss value when {test_case_description}.",
                )
