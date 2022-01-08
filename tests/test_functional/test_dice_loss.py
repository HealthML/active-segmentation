"""Tests for the dice loss."""

from typing import Callable, Dict, Literal, Optional, Tuple
import unittest

import numpy as np
import torch

from functional import DiceLoss
import tests.utils


class TestDiceLoss(unittest.TestCase):
    """
    Test cases for dice loss.
    """

    def _test_dice_loss(
        self,
        get_first_slice: Callable[
            [bool],
            Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float],
        ],
        get_second_slice: Callable[
            [bool],
            Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float],
        ],
        expected_loss: Optional[torch.Tensor] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "none",
        epsilon: float = 0.0001,
    ) -> None:
        """
        Helper function that calculates the dice loss with the given settings for the given predictions and compares it
        with an expected value.

        Args:
            get_first_slice: Getter function that returns prediction, target and expected metrics for the first slice.
            get_second_slice: Getter function that returns prediction, target and expected metrics for the second slice.
            reduction (string, optional): Reduction parameter of dice loss.
            epsilon (float, optional): Epsilon parameter of dice loss.
            expected_loss (Tensor, optional): Expected loss value.
        """

        # pylint: disable-msg=too-many-locals

        (
            predictions_first_slice,
            target_first_slice,
            cardinalities_first_slice,
            probability_positive_first_slice,
            probability_negative_first_slice,
        ) = get_first_slice(False)
        (
            predictions_second_slice,
            target_second_slice,
            cardinalities_second_slice,
            probability_positive_second_slice,
            probability_negative_second_slice,
        ) = get_second_slice(False)

        prediction = torch.stack([predictions_first_slice, predictions_second_slice])
        target = torch.stack([target_first_slice, target_second_slice])

        if expected_loss is None:

            expected_dice_scores_first_slice = tests.utils.expected_metrics(
                "dice_score",
                cardinalities_first_slice,
                probability_positive_first_slice,
                probability_negative_first_slice,
                epsilon,
            )

            expected_dice_scores_second_slice = tests.utils.expected_metrics(
                "dice_score",
                cardinalities_second_slice,
                probability_positive_second_slice,
                probability_negative_second_slice,
                epsilon,
            )

            expected_loss_first_slice = 1 - expected_dice_scores_first_slice
            expected_loss_second_slice = 1 - expected_dice_scores_second_slice

            if not include_background:
                expected_loss_first_slice = expected_loss_first_slice[1:]
                expected_loss_second_slice = expected_loss_second_slice[1:]

            expected_losses = np.array(
                [*expected_loss_first_slice, *expected_loss_second_slice]
            )

            if reduction == "mean":
                expected_loss = torch.as_tensor(expected_losses.mean())
            elif reduction == "sum":
                expected_loss = torch.as_tensor(expected_losses.sum())
            else:
                expected_loss = torch.Tensor(
                    [expected_loss_first_slice, expected_loss_second_slice]
                )

        dice_loss = DiceLoss(
            include_background=include_background, reduction=reduction, epsilon=epsilon
        )
        loss = dice_loss(prediction, target)

        self.assertTrue(
            loss.shape == expected_loss.shape, "Returns loss tensor with correct shape."
        )
        torch.testing.assert_allclose(
            loss,
            expected_loss,
            msg=f"Correctly computes loss value when include_background is {include_background}, reduction is "
            f"{reduction} and epsilon is {epsilon}.",
        )

    def test_standard_case_multi_label(self) -> None:
        """
        Tests that the dice loss is computed correctly tasks when there are both true and false predictions.
        """

        for test_slice_1, test_slice_2 in [
            (
                tests.utils.standard_slice_single_label_1,
                tests.utils.standard_slice_single_label_2,
            ),
            (
                tests.utils.standard_slice_multi_label_1,
                tests.utils.standard_slice_multi_label_2,
            ),
        ]:
            for reduction in ["none", "mean", "sum"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        self._test_dice_loss(
                            test_slice_1,
                            test_slice_2,
                            include_background=include_background,
                            reduction=reduction,
                            epsilon=epsilon,
                        )

    def test_all_true(self):
        """
        Tests that the dice loss is computed correctly when all predictions are correct.
        """

        for test_slice in [
            tests.utils.slice_all_true_single_label,
            tests.utils.slice_all_true_multi_label,
        ]:

            for epsilon in [0, 1]:
                self._test_dice_loss(
                    test_slice,
                    test_slice,
                    reduction="none",
                    epsilon=epsilon,
                    expected_loss=torch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                )

                self._test_dice_loss(
                    test_slice,
                    test_slice,
                    reduction="none",
                    include_background=False,
                    epsilon=epsilon,
                    expected_loss=torch.Tensor([[0.0, 0.0], [0.0, 0.0]]),
                )

                for reduction in ["mean", "sum"]:
                    for include_background in [True, False]:
                        self._test_dice_loss(
                            test_slice,
                            test_slice,
                            include_background=include_background,
                            reduction=reduction,
                            epsilon=epsilon,
                            expected_loss=torch.as_tensor(0.0),
                        )

    def test_all_false(self):
        """
        Tests that the dice loss is computed correctly when all predictions are wrong.
        """

        for test_slice in [
            tests.utils.slice_all_false_single_label,
            tests.utils.slice_all_false_multi_label,
        ]:

            self._test_dice_loss(
                test_slice,
                test_slice,
                reduction="none",
                epsilon=0,
                expected_loss=torch.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            )
            self._test_dice_loss(
                test_slice,
                test_slice,
                reduction="none",
                include_background=False,
                epsilon=0,
                expected_loss=torch.Tensor([[1.0, 1.0], [1.0, 1.0]]),
            )
            self._test_dice_loss(
                test_slice,
                test_slice,
                epsilon=0,
                reduction="mean",
                expected_loss=torch.as_tensor(1.0),
            )
            self._test_dice_loss(
                test_slice,
                test_slice,
                epsilon=0,
                reduction="mean",
                include_background=False,
                expected_loss=torch.as_tensor(1.0),
            )
            self._test_dice_loss(
                test_slice,
                test_slice,
                reduction="sum",
                epsilon=0,
                expected_loss=torch.as_tensor(6.0),
            )
            self._test_dice_loss(
                test_slice,
                test_slice,
                reduction="sum",
                epsilon=0,
                include_background=False,
                expected_loss=torch.as_tensor(4.0),
            )

            for reduction in ["none", "mean", "sum"]:
                for include_background in [True, False]:
                    self._test_dice_loss(
                        test_slice,
                        test_slice,
                        include_background=include_background,
                        reduction=reduction,
                        epsilon=1,
                    )

    def test_no_true_positives(self):
        """
        Tests that the dice loss is computed correctly when there are no true
        positives.
        """

        for test_slice in [
            tests.utils.slice_no_true_positives_single_label,
            tests.utils.slice_no_true_positives_multi_label,
        ]:
            self._test_dice_loss(
                test_slice,
                test_slice,
                reduction="none",
                epsilon=0,
                expected_loss=torch.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            )
            self._test_dice_loss(
                test_slice,
                test_slice,
                reduction="mean",
                epsilon=0,
                expected_loss=torch.as_tensor(1.0),
            )
            self._test_dice_loss(
                test_slice,
                test_slice,
                reduction="sum",
                epsilon=0,
                expected_loss=torch.as_tensor(6.0),
            )
            self._test_dice_loss(
                test_slice,
                test_slice,
                include_background=False,
                reduction="sum",
                epsilon=0,
                expected_loss=torch.as_tensor(4.0),
            )

            for reduction in ["none", "mean", "sum"]:
                for include_background in [True, False]:
                    self._test_dice_loss(
                        test_slice,
                        test_slice,
                        include_background=include_background,
                        reduction=reduction,
                        epsilon=1,
                    )

    def test_no_true_negatives(self):
        """
        Tests that the dice loss is computed correctly when there are no true negatives.
        """

        for test_slice in [
            tests.utils.slice_no_true_negatives_single_label,
            tests.utils.slice_no_true_negatives_multi_label,
        ]:
            for reduction in ["none", "mean", "sum"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        self._test_dice_loss(
                            test_slice,
                            test_slice,
                            include_background=include_background,
                            reduction=reduction,
                            epsilon=epsilon,
                        )
