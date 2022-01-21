"""Tests for the CombinedPerImageMetric class"""

import unittest

import torch

from metric_tracking import CombinedPerImageMetric
import tests.utils
from tests.utils import test_data_cardinality_metrics as test_data
from tests.utils import test_data_distance_metrics


class TestCombinedPerImageMetric(unittest.TestCase):
    """
    Tests for the CombinedPerImageMetric class.
    """

    # pylint: disable=too-many-branches, too-many-nested-blocks, too-many-locals

    def test_standard_case_cardinality_metrics(self):
        """
        Tests that the CombinedPerImageMetric class correctly computes all cardinality-based metrics for all confidence
        levels.
        """

        id_to_class_mapping = {
            0: "first_test_class",
            1: "second_test_class",
            2: "third_test_class",
        }

        confidence_levels = [0.2, 0.5, 0.8]

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
            for sharp_prediction in [True, False]:
                (
                    prediction_1,
                    target_1,
                    cardinalities_1,
                    _,
                    _,
                ) = test_slice_1(sharp_prediction)

                (
                    prediction_2,
                    target_2,
                    cardinalities_2,
                    _,
                    _,
                ) = test_slice_2(sharp_prediction)

                cardinalities = {}

                for class_id, cardinality_dict in cardinalities_1.items():
                    cardinalities[class_id] = {
                        key: value + cardinalities_2[class_id][key]
                        for key, value in cardinality_dict.items()
                    }

                metrics = ["dice_score", "sensitivity", "specificity"]

                expected_metrics = {}

                for metric_name in metrics:
                    expected_metric_value = tests.utils.expected_metrics(
                        metric_name,
                        cardinalities,
                        probability_positive=1.0,
                        probability_negative=0.0,
                        epsilon=0,
                    )
                    expected_metrics[metric_name] = torch.from_numpy(
                        expected_metric_value
                    ).float()

                metrics_per_image = CombinedPerImageMetric(
                    metrics,
                    id_to_class_mapping,
                    slices_per_image=2,
                    multi_label=multi_label,
                    confidence_levels=confidence_levels,
                )

                metrics_per_image.update(prediction_1, target_1)
                metrics_per_image.update(prediction_2, target_2)

                computed_metrics = metrics_per_image.compute()

                for class_id, class_name in id_to_class_mapping.items():
                    if multi_label:
                        for confidence_level in confidence_levels:
                            for metric_name, expected_value in expected_metrics.items():
                                self.assertTrue(
                                    torch.equal(
                                        computed_metrics[
                                            f"{metric_name}_{class_name}_{confidence_level}"
                                        ],
                                        expected_value[class_id],
                                    ),
                                    f"The {metric_name} is computed correctly for multi-label tasks for confidence "
                                    f"level {confidence_level}.",
                                )

                    else:
                        for metric_name, expected_value in expected_metrics.items():
                            self.assertTrue(
                                torch.equal(
                                    computed_metrics[f"{metric_name}_{class_name}"],
                                    expected_value[class_id],
                                ),
                                f"The {metric_name} is computed correctly for single-label tasks.",
                            )

    # pylint: disable=no-self-use
    def test_standard_case_distance_metrics(self):
        """
        Tests that the CombinedPerImageMetric class correctly computes all distance-based metrics for all confidence
        levels.
        """

        id_to_class_mapping = {
            0: "first_test_class",
            1: "second_test_class",
            2: "third_test_class",
        }

        confidence_levels = [0.2, 0.5, 0.8]

        for test_slice_1, test_slice_2, multi_label, expected_distances in [
            (
                test_data_distance_metrics.distance_slice_ignore_index_single_label_1,
                test_data_distance_metrics.distance_slice_ignore_index_single_label_2,
                False,
                test_data_distance_metrics.expected_distances_slice_ignore_index_single_label_1_2,
            ),
            (
                test_data_distance_metrics.distance_slice_ignore_index_multi_label_1,
                test_data_distance_metrics.distance_slice_ignore_index_multi_label_2,
                True,
                test_data_distance_metrics.expected_distances_slice_ignore_index_multi_label_1_2,
            ),
        ]:
            (
                prediction_1,
                target_1,
                _,
                _,
                _,
                _,
            ) = test_slice_1(percentile=0.95)

            (
                prediction_2,
                target_2,
                _,
                _,
                _,
                _,
            ) = test_slice_2(percentile=0.95)

            expected_hausdorff_distances, _, _, maximum_distance = expected_distances(
                percentile=0.95
            )

            print("expected_hausdorff_distances", expected_hausdorff_distances)
            print("maximum_distance", maximum_distance)

            expected_hausdorff_distances = {
                class_name: torch.as_tensor(expected_distance)
                / torch.sqrt(torch.as_tensor(maximum_distance))
                for class_name, expected_distance in expected_hausdorff_distances.items()
            }

            print("expected_hausdorff_distances", expected_hausdorff_distances)

            metrics_per_image = CombinedPerImageMetric(
                ["hausdorff95"],
                id_to_class_mapping,
                slices_per_image=4,
                multi_label=multi_label,
                confidence_levels=confidence_levels,
            )

            metrics_per_image.update(prediction_1, target_1)
            metrics_per_image.update(prediction_2, target_2)

            computed_metrics = metrics_per_image.compute()

            print("computed_metrics", computed_metrics)
            print("expected_hausdorff_distances", expected_hausdorff_distances)

            for class_id, class_name in id_to_class_mapping.items():
                if multi_label:
                    for confidence_level in confidence_levels:
                        torch.testing.assert_allclose(
                            computed_metrics[
                                f"hausdorff95_{class_name}_{confidence_level}"
                            ],
                            expected_hausdorff_distances[class_id],
                            msg=f"The Hausdorff distance is computed correctly for multi-label tasks for confidence "
                            f"level {confidence_level}.",
                        )
                else:
                    torch.testing.assert_allclose(
                        computed_metrics[f"hausdorff95_{class_name}"],
                        expected_hausdorff_distances[class_id],
                        msg="The Hausdorff distance is computed correctly for single-label tasks.",
                    )

    def test_metric_reset(self):
        """
        Tests that the metrics are reset correctly.
        """

        for test_slice_1, test_slice_2, multi_label in [
            (
                test_data.standard_slice_single_label_1,
                test_data.slice_all_true_single_label,
                False,
            ),
            (
                test_data.standard_slice_multi_label_1,
                test_data.slice_all_true_multi_label,
                True,
            ),
        ]:
            for sharp_prediction in [True, False]:
                (
                    prediction_1,
                    target_1,
                    _,
                    _,
                    _,
                ) = test_slice_1(sharp_prediction)

                (
                    prediction_2,
                    target_2,
                    _,
                    _,
                    _,
                ) = test_slice_2(sharp_prediction)

                metrics = ["dice_score", "sensitivity", "specificity", "hausdorff95"]
                confidence_levels = [0.2, 0.5, 0.8]

                metrics_per_image = CombinedPerImageMetric(
                    metrics,
                    {
                        0: "first_test_class",
                        1: "second_test_class",
                        2: "third_test_class",
                    },
                    slices_per_image=1,
                    multi_label=multi_label,
                    confidence_levels=confidence_levels,
                )

                metrics_per_image.update(prediction_1, target_1)
                metrics_per_image.compute()

                metrics_per_image.reset()

                metrics_per_image.update(prediction_2, target_2)
                computed_metrics = metrics_per_image.compute()

                for metric in metrics:
                    if multi_label:
                        for confidence_level in confidence_levels:
                            metric_value = computed_metrics[
                                f"{metric}_first_test_class_{confidence_level}"
                            ]

                            if metric in ["dice_score", "sensitivity"]:
                                self.assertTrue(
                                    torch.equal(torch.as_tensor(1.0), metric_value),
                                    "The CombinedPerImageMetric correctly resets the Dice score for multi-label "
                                    "classification tasks.",
                                )
                            if metric == "hausdorff95":
                                self.assertTrue(
                                    torch.equal(torch.as_tensor(0.0), metric_value),
                                    "The CombinedPerImageMetric correctly resets the Hausdorff distance for multi-label"
                                    " classification tasks.",
                                )
                            if metric == "specificity":
                                self.assertTrue(
                                    torch.equal(torch.as_tensor(1.0), metric_value),
                                    "The CombinedPerImageMetric correctly resets the specificity for multi-label "
                                    "classification tasks.",
                                )
                            if metric == "sensitivity":
                                self.assertTrue(
                                    torch.equal(torch.as_tensor(1.0), metric_value),
                                    "The CombinedPerImageMetric correctly resets the sensitivity for multi-label "
                                    "classification tasks.",
                                )
                    else:
                        metric_value = computed_metrics[f"{metric}_first_test_class"]

                        if metric in ["dice_score", "sensitivity"]:
                            self.assertTrue(
                                torch.equal(torch.as_tensor(1.0), metric_value),
                                "The CombinedPerImageMetric correctly resets the Dice score for single-label "
                                "classification tasks.",
                            )
                        if metric == "hausdorff95":
                            self.assertTrue(
                                torch.equal(torch.as_tensor(0.0), metric_value),
                                "The CombinedPerImageMetric correctly resets the Hausdorff distance for single-label "
                                "classification tasks.",
                            )
                        if metric == "specificity":
                            self.assertTrue(
                                torch.equal(torch.as_tensor(1.0), metric_value),
                                "The CombinedPerImageMetric correctly resets the specificity for single-label "
                                "classification tasks.",
                            )
                        if metric == "sensitivity":
                            self.assertTrue(
                                torch.equal(torch.as_tensor(1.0), metric_value),
                                "The CombinedPerImageMetric correctly resets the sensitivity for single-label "
                                "classification tasks.",
                            )
