"""Tests for the Hausdorff distance metric"""

import math
from typing import Dict, Optional
import unittest

import torch

from functional import HausdorffDistance, hausdorff_distance
import tests.utils.test_data_cardinality_metrics as standard_test_data
import tests.utils.test_data_distance_metrics as test_data


class TestHausdorffDistance(unittest.TestCase):
    """
    Test cases for Hausdorff distance.
    """

    # pylint: disable=too-many-locals,too-many-nested-blocks,too-many-branches,too-many-arguments

    @staticmethod
    def _test_hausdorff_distance(
        prediction: torch.Tensor,
        target: torch.Tensor,
        expected_distances: Dict[int, float],
        maximum_distance: float,
        convert_to_one_hot: bool,
        ignore_index: Optional[int] = None,
        message: str = "",
        percentile: float = 0.95,
        slices_per_image: Optional[int] = None,
    ) -> None:
        """
        Helper function that calculates the Hausdorff distances for the given predictions and compares it with the
        expected values.

        Args:
            prediction (Tensor): Predicted segmentation mask.
            target (Tensor): Target segmentation mask.
            expected_distances (Dict[int, float]): Expected per-class Hausdorff distances.
            maximum_distance (float): Maximum distance to be used for normalization of expected Hausdorff distances.
            convert_to_one_hot (bool): Determines if data is label encoded and needs to be converted to one-hot
                encoding or not.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the
                metric. Defaults to `None`.
            message (string, optional): Description of the test case.
            percentile (float): Percentile for which the Hausdorff distance is to be tested.
            slices_per_image (int, optional): `slices_per_image` parameter of the Hausdorff distance module.
        """

        for include_background in [True, False]:
            for reduction in ["none", "mean", "min", "max"]:
                for normalize in [True, False]:

                    expected_distance = torch.tensor(
                        list(expected_distances.values())
                    ).float()

                    if not include_background:
                        expected_distance = expected_distance[1:]
                    if reduction == "mean":
                        expected_distance = expected_distance.mean()
                    elif reduction == "min":
                        expected_distance = expected_distance.min()
                    elif reduction == "max":
                        expected_distance = expected_distance.max()
                    if normalize:
                        expected_distance = expected_distance / torch.sqrt(
                            torch.as_tensor(maximum_distance)
                        )

                    hausdorff_distance_from_function = hausdorff_distance(
                        prediction,
                        target,
                        3,
                        convert_to_one_hot=convert_to_one_hot,
                        include_background=include_background,
                        ignore_index=ignore_index,
                        normalize=normalize,
                        percentile=percentile,
                        reduction=reduction,
                    )

                    task_type = "single-label" if convert_to_one_hot else "multi-label"

                    test_case_description = (
                        f"include_background is {include_background}, normalize is {normalize}, "
                        f"percentile is {percentile} and reduction is {reduction}"
                    )

                    torch.testing.assert_allclose(
                        hausdorff_distance_from_function,
                        expected_distance,
                        msg=f"Functional implementation correctly computes Hausdorff distance for {task_type} tasks "
                        f"when {message} and {test_case_description}.",
                    )

                    if slices_per_image is None:
                        if prediction.dim() == 2:
                            slices_per_image = 1
                        else:
                            slices_per_image = (
                                prediction.shape[0]
                                if convert_to_one_hot
                                else prediction.shape[1]
                            )

                    hausdorff_distance_module = HausdorffDistance(
                        num_classes=3,
                        slices_per_image=slices_per_image,
                        convert_to_one_hot=convert_to_one_hot,
                        include_background=include_background,
                        ignore_index=ignore_index,
                        normalize=normalize,
                        percentile=percentile,
                        reduction=reduction,
                    )

                    hausdorff_distance_from_module = hausdorff_distance_module(
                        prediction, target
                    )

                    torch.testing.assert_allclose(
                        hausdorff_distance_from_module,
                        expected_distance,
                        msg=f"Module-based implementation correctly computes hausdorff distance when {message}.",
                    )

    def test_standard_case(self):
        """
        Tests that the Hausdorff distance is computed correctly when there are both true and false predictions.
        """

        for percentile in [0.5, 0.95, 1.0]:

            for test_slice, convert_to_one_hot in [
                (test_data.standard_distance_slice_single_label, True),
                (test_data.standard_distance_slice_multi_label, False),
            ]:

                (
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    _,
                    _,
                    maximum_distance,
                ) = test_slice(percentile=percentile)

                self._test_hausdorff_distance(
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    maximum_distance,
                    convert_to_one_hot=convert_to_one_hot,
                    message="there are TP, FP and FN",
                    percentile=percentile,
                )

    def test_all_true(self):
        """
        Tests that the Hausdorff distance is computed correctly when all predictions are correct.
        """

        for percentile in [0.5, 0.95, 1.0]:

            for test_slice, convert_to_one_hot in [
                (standard_test_data.slice_all_true_single_label, True),
                (standard_test_data.slice_all_true_multi_label, False),
            ]:

                (
                    prediction,
                    target,
                    _,
                    _,
                    _,
                ) = test_slice(True)

                self._test_hausdorff_distance(
                    prediction,
                    target,
                    {0: 0, 1: 0, 2: 0},
                    math.sqrt(8),
                    convert_to_one_hot=convert_to_one_hot,
                    message="there are no prediction errors.",
                    percentile=percentile,
                )

    def test_all_false(self):
        """
        Tests that the Hausdorff distance is computed correctly when all predictions are wrong.
        """

        for percentile in [0.5, 0.95, 1.0]:

            for test_slice, convert_to_one_hot in [
                (test_data.distance_slice_all_false_single_label, True),
                (test_data.distance_slice_all_false_multi_label, False),
            ]:

                (
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    _,
                    _,
                    maximum_distance,
                ) = test_slice(percentile=percentile)

                self._test_hausdorff_distance(
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    maximum_distance,
                    convert_to_one_hot=convert_to_one_hot,
                    message="all predictions are wrong",
                    percentile=percentile,
                )

    def test_subset(self):
        """
        Tests that the Hausdorff distance is computed correctly when all the predicted mask is a subset of the target
        mask.
        """

        for percentile in [0.5, 0.95, 1.0]:

            for test_slice, convert_to_one_hot in [
                (test_data.distance_slice_subset_single_label, True),
                (test_data.distance_slice_subset_multi_label, False),
            ]:

                (
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    _,
                    _,
                    maximum_distance,
                ) = test_slice(percentile=percentile)

                self._test_hausdorff_distance(
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    maximum_distance,
                    convert_to_one_hot=convert_to_one_hot,
                    message="the prediction is a subset of the target",
                    percentile=percentile,
                )

    def test_no_positives(self):
        """
        Tests that the Hausdorff distance is computed correctly when there are no positives.
        """

        for percentile in [0.5, 0.95, 1.0]:

            for test_slice, convert_to_one_hot in [
                (standard_test_data.slice_all_true_negatives_single_label, True),
                (standard_test_data.slice_all_true_negatives_multi_label, False),
            ]:

                (
                    prediction,
                    target,
                    _,
                    _,
                    _,
                ) = test_slice(True)

                for include_background in [True, False]:
                    for reduction in ["none", "mean", "min", "max"]:
                        for normalize in [True, False]:
                            hausdorff_distance_from_function = hausdorff_distance(
                                prediction,
                                target,
                                3,
                                convert_to_one_hot=convert_to_one_hot,
                                include_background=include_background,
                                ignore_index=None,
                                normalize=normalize,
                                percentile=percentile,
                                reduction=reduction,
                            )

                            if reduction == "none" and convert_to_one_hot:
                                self.assertTrue(
                                    torch.isnan(hausdorff_distance_from_function[-1]),
                                    "Functional implementation correctly computes Hausdorff distance when there are "
                                    "only TN.",
                                )
                            else:
                                self.assertTrue(
                                    torch.all(
                                        torch.isnan(hausdorff_distance_from_function)
                                    ),
                                    "Functional implementation correctly computes Hausdorff distance when there are "
                                    "only TN.",
                                )

                            hausdorff_distance_module = HausdorffDistance(
                                num_classes=3,
                                slices_per_image=1,
                                convert_to_one_hot=convert_to_one_hot,
                                include_background=include_background,
                                ignore_index=None,
                                normalize=normalize,
                                percentile=percentile,
                                reduction=reduction,
                            )

                            hausdorff_distance_from_module = hausdorff_distance_module(
                                prediction, target
                            )

                            if reduction == "none" and convert_to_one_hot:
                                self.assertTrue(
                                    torch.isnan(hausdorff_distance_from_module[-1]),
                                    "Functional implementation correctly computes Hausdorff distance when there are "
                                    "only TN.",
                                )
                            else:
                                self.assertTrue(
                                    torch.all(
                                        torch.isnan(hausdorff_distance_from_module)
                                    ),
                                    "Functional implementation correctly computes Hausdorff distance when there are "
                                    "only TN.",
                                )

    def test_3d(self):
        """
        Tests that the Hausdorff distance is computed correctly when the predictions are 3-dimensional scans.
        """
        for percentile in [0.5, 0.95, 1.0]:

            for test_slice, convert_to_one_hot in [
                (test_data.distance_slice_3d_single_label, True),
                (test_data.distance_slice_3d_multi_label, False),
            ]:
                (
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    _,
                    _,
                    maximum_distance,
                ) = test_slice(percentile=percentile)

                self._test_hausdorff_distance(
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    maximum_distance,
                    convert_to_one_hot=convert_to_one_hot,
                    message="the input is three-dimensional",
                    percentile=percentile,
                )

    # pylint: disable=no-self-use
    def test_splitted_3d(self):
        """
        Tests that the Hausdorff distance is computed correctly when the predictions are 3-dimensional scans whose
        slices are scattered across multiple batches.
        """

        for percentile in [0.5, 0.95, 1.0]:

            for test_slice, convert_to_one_hot in [
                (test_data.distance_slice_3d_single_label, True),
                (test_data.distance_slice_3d_multi_label, False),
            ]:
                (
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    _,
                    _,
                    maximum_distance,
                ) = test_slice(percentile=percentile)

                for include_background in [True, False]:
                    for reduction in ["none", "mean", "min", "max"]:
                        for normalize in [True, False]:

                            expected_distance = torch.tensor(
                                list(expected_hausdorff_dists.values())
                            ).float()

                            if not include_background:
                                expected_distance = expected_distance[1:]
                            if reduction == "mean":
                                expected_distance = expected_distance.mean()
                            elif reduction == "min":
                                expected_distance = expected_distance.min()
                            elif reduction == "max":
                                expected_distance = expected_distance.max()
                            if normalize:
                                expected_distance = expected_distance / torch.sqrt(
                                    torch.as_tensor(maximum_distance)
                                )

                            hausdorff_distance_module = HausdorffDistance(
                                num_classes=3,
                                slices_per_image=3,
                                convert_to_one_hot=convert_to_one_hot,
                                include_background=include_background,
                                ignore_index=None,
                                normalize=normalize,
                                percentile=percentile,
                                reduction=reduction,
                            )

                            if convert_to_one_hot:
                                for slice_idx in range(prediction.shape[0]):
                                    hausdorff_distance_module.update(
                                        prediction[slice_idx], target[slice_idx]
                                    )
                            else:
                                for slice_idx in range(prediction.shape[1]):
                                    hausdorff_distance_module.update(
                                        prediction[:, slice_idx, :, :],
                                        target[:, slice_idx, :, :],
                                    )

                            hausdorff_distance_from_module = (
                                hausdorff_distance_module.compute()
                            )

                            torch.testing.assert_allclose(
                                hausdorff_distance_from_module,
                                expected_distance,
                                msg="Module-based implementation correctly computes hausdorff distance when the "
                                "predictions are 3-dimensional scans whose slices are scattered across multiple "
                                "batches.",
                            )

    def test_ignore_index(self):
        """
        Tests that the Hausdorff distance is computed correctly when there are pixels / voxels to be ignored.
        """

        for percentile in [0.5, 0.95, 1.0]:

            for test_slice, convert_to_one_hot in [
                (test_data.distance_slice_ignore_index_single_label_2, True),
                (test_data.distance_slice_ignore_index_multi_label_2, False),
            ]:
                (
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    _,
                    _,
                    maximum_distance,
                ) = test_slice(percentile=percentile)

                self._test_hausdorff_distance(
                    prediction,
                    target,
                    expected_hausdorff_dists,
                    maximum_distance,
                    convert_to_one_hot=convert_to_one_hot,
                    ignore_index=-1,
                    message="there are pixels / voxels to be ignored",
                    percentile=percentile,
                    slices_per_image=2,
                )

    def test_different_padding_per_slice(self):
        """
        Tests that the Hausdorff distance is computed correctly when slices from different batches have different
        padding sizes.
        """

        for percentile in [0.5, 0.95, 1.0]:

            for test_slice_1, test_slice_2, convert_to_one_hot, expected_distances in [
                (
                    test_data.distance_slice_ignore_index_single_label_1,
                    test_data.distance_slice_ignore_index_single_label_2,
                    True,
                    test_data.expected_distances_slice_ignore_index_single_label_1_2,
                ),
                (
                    test_data.distance_slice_ignore_index_multi_label_1,
                    test_data.distance_slice_ignore_index_multi_label_2,
                    False,
                    test_data.expected_distances_slice_ignore_index_multi_label_1_2,
                ),
            ]:
                (
                    prediction_1,
                    target_1,
                    _,
                    _,
                    _,
                    _,
                ) = test_slice_1(percentile=percentile)

                (
                    prediction_2,
                    target_2,
                    _,
                    _,
                    _,
                    _,
                ) = test_slice_2(percentile=percentile)

                expected_distances, _, _, maximum_distance = expected_distances(
                    percentile
                )

                for include_background in [True, False]:
                    for reduction in ["none", "mean", "min", "max"]:
                        for normalize in [True, False]:

                            expected_distance = torch.tensor(
                                list(expected_distances.values())
                            ).float()

                            if not include_background:
                                expected_distance = expected_distance[1:]
                            if reduction == "mean":
                                expected_distance = expected_distance.mean()
                            elif reduction == "min":
                                expected_distance = expected_distance.min()
                            elif reduction == "max":
                                expected_distance = expected_distance.max()
                            if normalize:
                                expected_distance = expected_distance / torch.sqrt(
                                    torch.as_tensor(maximum_distance)
                                )

                            hausdorff_distance_module = HausdorffDistance(
                                num_classes=3,
                                slices_per_image=4,
                                convert_to_one_hot=convert_to_one_hot,
                                include_background=include_background,
                                ignore_index=-1,
                                normalize=normalize,
                                percentile=percentile,
                                reduction=reduction,
                            )

                            hausdorff_distance_module.update(prediction_1, target_1)
                            hausdorff_distance_module.update(prediction_2, target_2)

                            hausdorff_distance_from_module = (
                                hausdorff_distance_module.compute()
                            )

                            task_type = (
                                "single-label" if convert_to_one_hot else "multi-label"
                            )

                            test_case_description = (
                                f"include_background is {include_background}, normalize is {normalize}, "
                                f"percentile is {percentile} and reduction is {reduction}"
                            )

                            torch.testing.assert_allclose(
                                hausdorff_distance_from_module,
                                expected_distance,
                                msg=f"Module-based implementation correctly computes Hausdorff distance for {task_type}"
                                f" tasks when the slices of a 3d image are scattered across multiple batches with "
                                f"different padding sizes and {test_case_description}.",
                            )
