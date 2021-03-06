"""Module containing a base class for testing segmentation loss functions."""

import abc
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
from parameterized import parameterized
import torch

import tests.utils
import tests.utils.test_data_cardinality_metrics as test_data


class LossTestCase(abc.ABC):
    """
    Base class for testing segmentation loss functions.
    """

    @abc.abstractmethod
    def loss_name(self) -> str:
        """
        Returns:
            String: The name of the loss or metric to be tested.
        """

    @abc.abstractmethod
    def loss_module(
        self,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "none",
        epsilon: float = 0.0001,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Args:
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            include_background (bool, optional): `include_background` parameter of the loss.
            reduction (string, optional): `reduction` parameter of the loss.
            epsilon (float, optional): `epsilon` parameter of the loss.
            kwargs: Further, loss specific parameters.

        Returns:
            torch.nn.Module: The loss module to be tested.
        """

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
        epsilon: float = 0.0001,
        **kwargs,
    ) -> None:
        """
        Helper function that calculates the loss with the given settings for the given predictions and compares it
        with an expected value.

        Args:
            get_first_slice: Getter function that returns prediction, target and expected metrics for the first slice.
            get_second_slice: Getter function that returns prediction, target and expected metrics for the second slice.
            ignore_index (bool, optional): `ignore_index` parameter of the loss.
            include_background (bool, optional): `include_background` parameter of the loss.
            reduction (string, optional): `reduction` parameter of the loss.
            weight (Tensor, optional): `weight` parameter of the loss.
            epsilon (float, optional): `epsilon` parameter of the loss.
            expected_loss (Tensor, optional): Expected loss value.
            kwargs: Further, loss-specific parameters.
        """

        # pylint: disable-msg=too-many-locals, no-member

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

            expected_loss_first_slice = tests.utils.expected_metrics(
                self.loss_name(),
                cardinalities_first_slice,
                probability_positive_first_slice,
                probability_negative_first_slice,
                epsilon,
            )

            expected_loss_second_slice = tests.utils.expected_metrics(
                self.loss_name(),
                cardinalities_second_slice,
                probability_positive_second_slice,
                probability_negative_second_slice,
                epsilon,
            )

            if not include_background:
                expected_loss_first_slice = expected_loss_first_slice[1:]
                expected_loss_second_slice = expected_loss_second_slice[1:]

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
                expected_loss = torch.Tensor(expected_losses)

        prediction = prediction.float()
        target = target.float()

        prediction.requires_grad = True
        target.requires_grad = True

        loss_module = self.loss_module(
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
            epsilon=epsilon,
            **kwargs,
        )
        loss = loss_module(prediction, target, weight)

        self.assertTrue(
            loss.shape == expected_loss.shape,
            f"Returns loss tensor with correct shape if reduction is {reduction}.",
        )

        test_case_description = (
            f"ignore_index is {ignore_index}, include_background is {include_background}, reduction "
            f"is {reduction} and epsilon is {epsilon}"
        )

        self.assertNotEqual(
            loss.grad_fn,
            None,
            msg=f"Loss is differentiable when {test_case_description}.",
        )

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
            for reduction in ["none", "mean", "sum"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        self._test_loss(
                            test_slice_1,
                            test_slice_2,
                            include_background=include_background,
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

            for epsilon in [0, 1]:
                self._test_loss(
                    test_slice,
                    test_slice,
                    reduction="none",
                    epsilon=epsilon,
                    expected_loss=torch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                )

                self._test_loss(
                    test_slice,
                    test_slice,
                    reduction="none",
                    include_background=False,
                    epsilon=epsilon,
                    expected_loss=torch.Tensor([[0.0, 0.0], [0.0, 0.0]]),
                )

                for reduction in ["mean", "sum"]:
                    for include_background in [True, False]:
                        self._test_loss(
                            test_slice,
                            test_slice,
                            include_background=include_background,
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
                reduction="none",
                include_background=False,
                epsilon=0,
                expected_loss=torch.Tensor([[1.0, 1.0], [1.0, 1.0]]),
            )
            self._test_loss(
                test_slice,
                test_slice,
                epsilon=0,
                reduction="mean",
                expected_loss=torch.as_tensor(1.0),
            )
            self._test_loss(
                test_slice,
                test_slice,
                epsilon=0,
                reduction="mean",
                include_background=False,
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
                reduction="sum",
                epsilon=0,
                include_background=False,
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

    def test_no_true_negatives(self):
        """
        Tests that the loss is computed correctly when there are no true negatives.
        """

        for test_slice in [
            test_data.slice_no_true_negatives_single_label,
            test_data.slice_no_true_negatives_multi_label,
        ]:
            for reduction in ["none", "mean", "sum"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        self._test_loss(
                            test_slice,
                            test_slice,
                            include_background=include_background,
                            reduction=reduction,
                            epsilon=epsilon,
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
            for reduction in ["none", "mean", "sum"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        self._test_loss(
                            test_slice_1,
                            test_slice_2,
                            ignore_index=-1,
                            include_background=include_background,
                            reduction=reduction,
                            epsilon=epsilon,
                        )

    def test_3d(self):
        """
        Tests that the loss is computed correctly when the inputs are 3d images.
        """

        # pylint: disable-msg=too-many-locals, no-member

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

            for reduction in ["none", "mean", "sum"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:

                        expected_loss = tests.utils.expected_metrics(
                            self.loss_name(),
                            cardinalities,
                            probability_positive_1,
                            probability_negative_1,
                            epsilon,
                        )

                        if not include_background:
                            expected_loss = expected_loss[1:]

                        if reduction == "mean":
                            expected_loss = torch.as_tensor(expected_loss.mean())
                        elif reduction == "sum":
                            expected_loss = torch.as_tensor(expected_loss.sum())
                        else:
                            expected_loss = torch.Tensor([expected_loss])

                    prediction = prediction.float()
                    target = target.float()

                    prediction.requires_grad = True
                    target.requires_grad = True

                    loss_module = self.loss_module(
                        include_background=include_background,
                        reduction=reduction,
                        epsilon=epsilon,
                    )
                    loss = loss_module(prediction, target)

                    self.assertTrue(
                        loss.shape == expected_loss.shape,
                        f"Returns loss tensor with correct shape if reduction is {reduction}.",
                    )

                    test_case_description = (
                        f"include_background is {include_background}, reduction is {reduction} "
                        f"and epsilon is {epsilon}"
                    )

                    self.assertNotEqual(
                        loss.grad_fn,
                        None,
                        msg=f"Loss is differentiable when {test_case_description}.",
                    )

                    torch.testing.assert_allclose(
                        loss,
                        expected_loss,
                        msg=f"Correctly computes loss value when {test_case_description}.",
                    )

    # fmt: off
    @parameterized.expand([
        (test_data.standard_slice_single_label_1, False, "none", 0, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, False, "mean", 0, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, False, "sum", 0, True, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_single_label_1, False, "none", 0, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, False, "mean", 0, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_single_label_1, False, "sum", 0, False, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_multi_label_1, True, "none", 0, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, True, "mean", 0, True, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, True, "sum", 0, True, torch.Tensor([1.0, 0.5])),

        (test_data.standard_slice_multi_label_1, True, "none", 0, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, True, "mean", 0, False, torch.Tensor([1.0, 0.5])),
        (test_data.standard_slice_multi_label_1, True, "sum", 0, False, torch.Tensor([1.0, 0.5])),
    ])
    # fmt: on
    def test_weighted_loss(
        self,
        test_slice: torch.Tensor,
        multi_label: bool,
        reduction: str,
        epsilon: float,
        include_background: bool,
        weight: torch.Tensor,
    ):
        """
        Tests that the loss is computed correctly when the images of one batch are weighted differently.
        """

        (
            _,
            _,
            cardinalities,
            probability_positive,
            probability_negative,
        ) = test_slice(False)

        expected_loss = tests.utils.expected_metrics(
            self.loss_name(),
            cardinalities,
            probability_positive,
            probability_negative,
            epsilon,
        )

        if not include_background:
            expected_loss = expected_loss[1:]

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
            multi_label=multi_label,
            include_background=include_background,
            reduction=reduction,
            weight=weight,
            expected_loss=expected_loss,
            epsilon=epsilon,
        )
