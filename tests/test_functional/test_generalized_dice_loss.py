"""Tests for the generalized dice loss."""

import unittest
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
from parameterized import parameterized
import torch

from functional import GeneralizedDiceLoss
import tests.utils
import tests.utils.test_data_cardinality_metrics as test_data


class TestGeneralizedDiceLoss(unittest.TestCase):
    """
    Test cases for generalized dice loss.
    """

    @staticmethod
    def loss_module(
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "none",
        weight_type: Literal["square", "simple", "uniform"] = "square",
        epsilon: float = 0.0001,
    ) -> torch.nn.Module:
        """
        Args:
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            include_background (bool, optional): `include_background` parameter of the loss.
            reduction (string, optional): `reduction` parameter of the loss.
            weight_type (string, optional): `weight_type` parameter of the loss.
            epsilon (float, optional): `epsilon` parameter of the loss.

        Returns:
            torch.nn.Module: The loss module to be tested.
        """

        return GeneralizedDiceLoss(
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
            weight_type=weight_type,
            epsilon=epsilon,
        )

    # pylint: disable=too-many-arguments
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
        expected_loss: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "none",
        weight: Optional[torch.Tensor] = None,
        weight_type: Literal["square", "simple", "uniform"] = "square",
        epsilon: float = 0.0001,
    ) -> None:
        """
        Helper function that calculates the loss with the given settings for the given predictions and compares it
        with an expected value.

        Args:
            get_first_slice: Getter function that returns prediction, target and expected metrics for the first slice.
            get_second_slice: Getter function that returns prediction, target and expected metrics for the second slice.
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            include_background (bool, optional): `include_background` parameter of the loss.
            weight (Tensor, optional): `weight` parameter of the loss.
            weight_type (string, optional): `weight_type` parameter of the loss.
            reduction (string, optional): `reduction` parameter of the loss.
            epsilon (float, optional): `epsilon` parameter of the loss.
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
            expected_loss_first_slice = tests.utils.expected_generalized_dice_loss(
                cardinalities_first_slice,
                probability_positive_first_slice,
                probability_negative_first_slice,
                epsilon,
                include_background=include_background,
                weight_type=weight_type,
            )

            expected_loss_second_slice = tests.utils.expected_generalized_dice_loss(
                cardinalities_second_slice,
                probability_positive_second_slice,
                probability_negative_second_slice,
                epsilon,
                include_background=include_background,
                weight_type=weight_type,
            )

            expected_losses = np.array(
                [expected_loss_first_slice, expected_loss_second_slice]
            )

            if weight is not None:
                expanded_weight = weight
                if expected_losses.ndim > 1:
                    for _ in range(1, expected_losses.ndim):
                        expanded_weight = expanded_weight.unsqueeze(axis=-1)

                expected_losses = expected_losses * expanded_weight.numpy()

            if reduction == "mean":
                expected_loss = torch.as_tensor(expected_losses.mean())
            elif reduction == "sum":
                expected_loss = torch.as_tensor(expected_losses.sum())
            else:
                expected_loss = torch.Tensor(
                    [expected_loss_first_slice, expected_loss_second_slice]
                )

        # this test currently does not pass
        # see https://github.com/Project-MONAI/MONAI/issues/3618
        # prediction.requires_grad = True
        # target.requires_grad = True

        loss_module = self.loss_module(
            ignore_index=ignore_index,
            include_background=include_background,
            weight_type=weight_type,
            reduction=reduction,
            epsilon=epsilon,
        )
        loss = loss_module(prediction, target, weight=weight)

        self.assertTrue(
            loss.shape == expected_loss.shape,
            f"Returns loss tensor with correct shape if reduction is {reduction}.",
        )

        test_case_description = (
            f"ignore_index is {ignore_index}, include_background is {include_background}, reduction "
            f"is {reduction} and epsilon is {epsilon}"
        )

        # self.assertNotEqual(
        #     loss.grad_fn,
        #     None,
        #     msg=f"Loss is differentiable when {test_case_description}.",
        # )

        torch.testing.assert_allclose(
            loss,
            expected_loss,
            msg=f"Correctly computes loss value when {test_case_description}.",
        )

    def test_standard_case(self) -> None:
        """
        Tests that the loss is computed correctly tasks when there are both true and false predictions.
        """

        for test_slice_1, test_slice_2 in [
            (
                test_data.standard_slice_single_label_1,
                test_data.standard_slice_single_label_2,
            ),
            (
                test_data.standard_slice_multi_label_1,
                test_data.standard_slice_multi_label_2,
            ),
        ]:
            for weight_type in ["square", "simple", "uniform"]:
                for reduction in ["none", "mean", "sum"]:
                    for include_background in [True, False]:
                        for epsilon in [0, 1]:
                            self._test_loss(
                                test_slice_1,
                                test_slice_2,
                                include_background=include_background,
                                weight_type=weight_type,
                                reduction=reduction,
                                epsilon=epsilon,
                            )

    def test_all_true(self):
        """
        Tests that the loss is computed correctly when all predictions are correct.
        """

        for test_slice in [
            test_data.slice_all_true_single_label,
            test_data.slice_all_true_multi_label,
        ]:
            for weight_type in ["square", "simple", "uniform"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        self._test_loss(
                            test_slice,
                            test_slice,
                            weight_type=weight_type,
                            reduction="none",
                            include_background=include_background,
                            epsilon=epsilon,
                            expected_loss=torch.Tensor([0.0, 0.0]),
                        )

                    for reduction in ["mean", "sum"]:
                        self._test_loss(
                            test_slice,
                            test_slice,
                            include_background=include_background,
                            weight_type=weight_type,
                            reduction=reduction,
                            epsilon=epsilon,
                            expected_loss=torch.as_tensor(0.0),
                        )

    def test_all_false(self):
        """
        Tests that the loss is computed correctly when all predictions are wrong.
        """

        for test_slice in [
            test_data.slice_all_false_single_label,
            test_data.slice_all_false_multi_label,
        ]:
            for include_background in [True, False]:
                self._test_loss(
                    test_slice,
                    test_slice,
                    include_background=include_background,
                    weight_type="uniform",
                    reduction="none",
                    epsilon=0,
                    expected_loss=torch.Tensor([1.0, 1.0]),
                )

                self._test_loss(
                    test_slice,
                    test_slice,
                    include_background=include_background,
                    weight_type="uniform",
                    reduction="mean",
                    epsilon=0,
                    expected_loss=torch.as_tensor(1.0),
                )

                self._test_loss(
                    test_slice,
                    test_slice,
                    include_background=include_background,
                    weight_type="uniform",
                    reduction="sum",
                    epsilon=0,
                    expected_loss=torch.as_tensor(2.0),
                )

                for weight_type in ["square", "simple", "uniform"]:
                    for reduction in ["none", "mean", "sum"]:
                        for epsilon in [0, 1]:
                            self._test_loss(
                                test_slice,
                                test_slice,
                                include_background=include_background,
                                weight_type=weight_type,
                                reduction=reduction,
                                epsilon=epsilon,
                            )

    def test_no_true_positives(self):
        """
        Tests that the loss is computed correctly when there are no true positives.
        """

        for test_slice in [
            test_data.slice_no_true_positives_single_label,
            test_data.slice_no_true_positives_multi_label,
        ]:
            for include_background in [True, False]:
                self._test_loss(
                    test_slice,
                    test_slice,
                    include_background=include_background,
                    weight_type="uniform",
                    reduction="none",
                    epsilon=0,
                    expected_loss=torch.Tensor([1.0, 1.0]),
                )

                self._test_loss(
                    test_slice,
                    test_slice,
                    include_background=include_background,
                    weight_type="uniform",
                    reduction="mean",
                    epsilon=0,
                    expected_loss=torch.as_tensor(1.0),
                )

                self._test_loss(
                    test_slice,
                    test_slice,
                    include_background=include_background,
                    weight_type="uniform",
                    reduction="sum",
                    epsilon=0,
                    expected_loss=torch.as_tensor(2.0),
                )

                for weight_type in ["square", "simple", "uniform"]:
                    for reduction in ["none", "mean", "sum"]:
                        for epsilon in [0, 1]:
                            self._test_loss(
                                test_slice,
                                test_slice,
                                include_background=include_background,
                                weight_type=weight_type,
                                reduction=reduction,
                                epsilon=epsilon,
                            )

    def test_all_true_negative(self):
        """
        Tests that the loss is computed correctly when there are no positives.
        """

        for weight_type in ["square", "simple", "uniform"]:
            for include_background in [True, False]:
                (
                    predictions_slice,
                    target_slice,
                    _,
                    _,
                    _,
                ) = test_data.slice_all_true_negatives_multi_label(False)
                prediction = torch.stack([predictions_slice, predictions_slice])
                target = torch.stack([target_slice, target_slice])

                generalized_dice_loss = GeneralizedDiceLoss(
                    epsilon=0,
                    weight_type=weight_type,
                    include_background=include_background,
                    reduction="none",
                )
                loss = generalized_dice_loss(prediction, target)

                self.assertTrue(
                    torch.isnan(loss).any(),
                    "Returns NaN if there are no positives and epsilon is zero.",
                )

                generalized_dice_loss = GeneralizedDiceLoss(
                    epsilon=1,
                    weight_type=weight_type,
                    include_background=include_background,
                    reduction="none",
                )
                loss = generalized_dice_loss(prediction, target)

                self.assertTrue(
                    torch.equal(loss, torch.as_tensor([0.0, 0.0])),
                    "Returns 0 if there are no positives and epsilon is greater than zero.",
                )

                for reduction in ["mean", "sum"]:
                    generalized_dice_loss = GeneralizedDiceLoss(
                        epsilon=0,
                        weight_type=weight_type,
                        include_background=include_background,
                        reduction=reduction,
                    )
                    loss = generalized_dice_loss(prediction, target)

                    self.assertTrue(
                        torch.isnan(loss).all(),
                        "Returns NaN if there are no positives and epsilon is zero.",
                    )

                    dice_loss = GeneralizedDiceLoss(
                        epsilon=1,
                        weight_type=weight_type,
                        include_background=include_background,
                        reduction=reduction,
                    )
                    loss = dice_loss(prediction, target)

                    self.assertTrue(
                        torch.equal(loss, torch.as_tensor(0.0)),
                        "Returns 0 if there are no positives and epsilon is greater than zero.",
                    )

    def test_ignore_index(self):
        """
        Tests that the loss is computed correctly when there are are pixels / voxels to be ignored.
        """

        for test_slice_1, test_slice_2 in [
            (
                test_data.standard_slice_single_label_1,
                test_data.slice_ignore_index_single_label,
            ),
            (
                test_data.standard_slice_multi_label_1,
                test_data.slice_ignore_index_multi_label,
            ),
        ]:
            for weight_type in ["square", "simple", "uniform"]:
                for reduction in ["none", "mean", "sum"]:
                    for include_background in [True, False]:
                        for epsilon in [0, 1]:
                            self._test_loss(
                                test_slice_1,
                                test_slice_2,
                                ignore_index=-1,
                                weight_type=weight_type,
                                include_background=include_background,
                                reduction=reduction,
                                epsilon=epsilon,
                            )

    def test_3d(self):
        """
        Tests that the loss is computed correctly when the inputs are 3d images.
        """

        # pylint: disable-msg=too-many-locals, too-many-nested-blocks

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
                cardinalities_1,
                probability_positive_1,
                probability_negative_1,
            ) = test_slice_1(False)

            (
                prediction_2,
                target_2,
                cardinalities_2,
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

            cardinalities = {}
            for class_id, class_cardinalities in cardinalities_1.items():
                # sum cardinalities for both stacked slices
                cardinalities[class_id] = {
                    key: cardinalities_1[class_id][key] + cardinalities_2[class_id][key]
                    for key in class_cardinalities
                }

            for weight_type in ["square", "simple", "uniform"]:
                for reduction in ["none", "mean", "sum"]:
                    for include_background in [True, False]:
                        for epsilon in [0, 1]:

                            expected_loss = tests.utils.expected_generalized_dice_loss(
                                cardinalities,
                                probability_positive_1,
                                probability_negative_1,
                                epsilon,
                                include_background=include_background,
                                weight_type=weight_type,
                            )

                            if reduction == "mean":
                                expected_loss = torch.as_tensor(expected_loss.mean())
                            elif reduction == "sum":
                                expected_loss = torch.as_tensor(expected_loss.sum())
                            else:
                                expected_loss = torch.Tensor([expected_loss])

                            prediction = prediction.float()
                            target = target.float()

                            loss_module = self.loss_module(
                                include_background=include_background,
                                weight_type=weight_type,
                                reduction=reduction,
                                epsilon=epsilon,
                            )
                            loss = loss_module(prediction, target)

                            self.assertTrue(
                                loss.shape == expected_loss.shape,
                                f"Returns loss tensor with correct shape if reduction is {reduction}.",
                            )

                            test_case_description = (
                                f"include_background is {include_background}, reduction is {reduction} and epsilon is "
                                f"{epsilon}"
                            )

                            torch.testing.assert_allclose(
                                loss,
                                expected_loss,
                                msg=f"Correctly computes loss value when {test_case_description}.",
                            )

    # fmt: off
    @parameterized.expand([
        (test_data.standard_slice_single_label_1, "none", 0, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, "mean", 0, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, "sum", 0, True, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_single_label_1, "none", 0, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, "mean", 0, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, "sum", 0, False, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_single_label_1, "none", 0, True, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_single_label_1, "mean", 0, True, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_single_label_1, "sum", 0, True, torch.Tensor([1.0, 1.0])),

        (test_data.standard_slice_single_label_1, "none", 0, False, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_single_label_1, "mean", 0, False, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_single_label_1, "sum", 0, False, torch.Tensor([1.0, 1.0])),

        (test_data.standard_slice_single_label_1, "none", 1, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, "mean", 1, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, "sum", 1, True, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_single_label_1, "none", 1, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, "mean", 1, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, "sum", 1, False, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_multi_label_1, "none", 0, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, "mean", 0, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, "sum", 0, True, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_multi_label_1, "none", 0, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, "mean", 0, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, "sum", 0, False, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_multi_label_1, "none", 0, True, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_multi_label_1, "mean", 0, True, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_multi_label_1, "sum", 0, True, torch.Tensor([1.0, 1.0])),

        (test_data.standard_slice_multi_label_1, "none", 0, False, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_multi_label_1, "mean", 0, False, torch.Tensor([1.0, 1.0])),
        (test_data.standard_slice_multi_label_1, "sum", 0, False, torch.Tensor([1.0, 1.0])),

        (test_data.standard_slice_multi_label_1, "none", 1, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, "mean", 1, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, "sum", 1, True, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_multi_label_1, "none", 1, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, "mean", 1, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, "sum", 1, False, torch.Tensor([1.0, 0.5])),
    ])
    # fmt: on
    def test_weighted_loss(
        self,
        test_slice: torch.Tensor,
        reduction: str,
        epsilon: float,
        include_background: bool,
        weight: torch.Tensor,
    ):
        """
        Tests that the loss is computed correctly when the images of one batch are weighted differently.
        """

        weight_type = "uniform"

        (
            _,
            _,
            cardinalities,
            probability_positive,
            probability_negative,
        ) = test_slice(False)

        expected_loss = tests.utils.expected_generalized_dice_loss(
            cardinalities,
            probability_positive,
            probability_negative,
            epsilon,
            include_background=include_background,
            weight_type=weight_type,
        )

        expected_loss = torch.Tensor([expected_loss, expected_loss])

        for idx, current_weight in enumerate(weight):
            expected_loss[idx] = expected_loss[idx] * current_weight

        if reduction == "mean":
            expected_loss = expected_loss.mean()
        elif reduction == "sum":
            expected_loss = expected_loss.sum()

        self._test_loss(
            test_slice,
            test_slice,
            include_background=include_background,
            reduction=reduction,
            weight=weight,
            weight_type=weight_type,
            expected_loss=expected_loss,
            epsilon=epsilon,
        )
