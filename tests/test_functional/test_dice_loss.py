"""Tests for the dice loss."""

import unittest
from typing import Literal, Optional

import torch

from functional import DiceLoss
import tests.utils.test_data_cardinality_metrics as test_data
from .test_loss import LossTestCase


class TestDiceLoss(unittest.TestCase, LossTestCase):
    """
    Test cases for Dice loss.
    """

    def loss_name(self) -> str:
        """
        Returns:
            String: The name of the loss or metric to be tested.
        """

        return "dice_loss"

    def loss_module(
        self,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "none",
        epsilon: float = 0.0001,
        **kwargs
    ) -> torch.nn.Module:
        """
        Args:
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            include_background (bool, optional): `include_background` parameter of the loss.
            reduction (string, optional): `reduction` parameter of the loss.
            epsilon (float, optional): `epsilon` parameter of the loss.

        Returns:
            torch.nn.Module: The loss module to be tested.
        """

        return DiceLoss(
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
            epsilon=epsilon,
        )

    def test_no_true_positives(self):
        """
        Tests that the dice loss is computed correctly when there are no true positives.
        """

        for test_slice in [
            test_data.slice_no_true_positives_single_label,
            test_data.slice_no_true_positives_multi_label,
        ]:
            self._test_loss(
                test_slice,
                test_slice,
                reduction="none",
                epsilon=0,
                expected_loss=torch.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            )
            self._test_loss(
                test_slice,
                test_slice,
                reduction="mean",
                epsilon=0,
                expected_loss=torch.as_tensor(1.0),
            )
            self._test_loss(
                test_slice,
                test_slice,
                reduction="sum",
                epsilon=0,
                expected_loss=torch.as_tensor(6.0),
            )
            self._test_loss(
                test_slice,
                test_slice,
                include_background=False,
                reduction="sum",
                epsilon=0,
                expected_loss=torch.as_tensor(4.0),
            )

            for reduction in ["none", "mean", "sum"]:
                for include_background in [True, False]:
                    self._test_loss(
                        test_slice,
                        test_slice,
                        include_background=include_background,
                        reduction=reduction,
                        epsilon=1,
                    )

    def test_all_true_negative(self):
        """
        Tests that the dice loss is computed correctly when there are no positives.
        """

        for test_slice in [
            test_data.slice_all_true_negatives_single_label,
            test_data.slice_all_true_negatives_multi_label,
        ]:
            predictions_slice, target_slice, _, _, _ = test_slice(False)
            prediction = torch.stack([predictions_slice, predictions_slice])
            target = torch.stack([target_slice, target_slice])

            dice_loss = DiceLoss(epsilon=0, include_background=True, reduction="none")
            loss = dice_loss(prediction, target)

            self.assertTrue(
                torch.isnan(loss).any(),
                "Returns NaN if there are no positives and epsilon is zero.",
            )

            dice_loss = DiceLoss(epsilon=1, include_background=True, reduction="none")
            loss = dice_loss(prediction, target)

            self.assertTrue(
                torch.equal(loss, torch.as_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
                "Returns 0 if there are no positives and epsilon is greater than zero.",
            )

            for reduction in ["mean", "sum"]:
                for include_background in [True, False]:
                    dice_loss = DiceLoss(
                        epsilon=0,
                        include_background=include_background,
                        reduction=reduction,
                    )
                    loss = dice_loss(prediction, target)

                    self.assertTrue(
                        torch.isnan(loss).all(),
                        "Returns NaN if there are no positives and epsilon is " "zero.",
                    )

                    dice_loss = DiceLoss(
                        epsilon=1,
                        include_background=include_background,
                        reduction=reduction,
                    )
                    loss = dice_loss(prediction, target)

                    self.assertTrue(
                        torch.equal(loss, torch.as_tensor(0.0)),
                        "Returns 0 if there are no positives and "
                        "epsilon is greater than zero.",
                    )
