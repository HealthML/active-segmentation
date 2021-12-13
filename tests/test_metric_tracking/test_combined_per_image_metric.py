"""Tests for the CombinedPerImageMetric class"""

from typing import List
import unittest

import numpy as np
import torch

from metric_tracking import CombinedPerImageMetric


class TestCombinedPerImageMetric(unittest.TestCase):
    """
    Tests for the CombinedPerImageMetric class.
    """

    @staticmethod
    def example_data():
        """
        Creates a faked segmentation exmaple that contains both true and false predictions.

        Returns:
            Tuple: Predicted slice, target slice, TP, FP, TN, FN for the confidence levels 0.2, 0.5, and 0.8.
        """

        predicted_probabilities = torch.Tensor(
            [
                [
                    [0.1, 0.2, 0.1],
                    [0.8, 0.1, 0.7],
                    [0.9, 0.6, 0.5],
                ],
                [
                    [0.8, 0.6, 0.8],
                    [0.1, 0.1, 0.3],
                    [0.1, 0.6, 0.2],
                ],
                [
                    [0.1, 0.3, 0.1],
                    [0.2, 0.4, 0.2],
                    [0.3, 0.4, 0.1],
                ],
            ]
        )

        target = torch.Tensor(
            [
                [
                    [1, 1, 1],
                    [1, 1, 0],
                    [1, 1, 0],
                ],
                [
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 0, 0],
                ],
            ]
        )

        tp_0_2, fp_0_2, tn_0_2, fn_0_2 = 7, 7, 6, 7
        tp_0_5, fp_0_5, tn_0_5, fn_0_5 = 5, 3, 10, 9
        tp_0_8, fp_0_8, tn_0_8, fn_0_8 = 1, 0, 13, 13

        return (
            predicted_probabilities,
            target,
            (tp_0_2, fp_0_2, tn_0_2, fn_0_2),
            (tp_0_5, fp_0_5, tn_0_5, fn_0_5),
            (tp_0_8, fp_0_8, tn_0_8, fn_0_8),
        )

    @staticmethod
    def _hausdorff_distance(
        dists_prediction_target: List[float],
        dists_target_prediction: List[float],
        percentile: int = 95,
    ) -> float:
        r"""
        Computes expected Hausdorff distance.

        Args:
            dists_prediction_target: List that contains for each positive prediction pixel the distance to the closest
                positive target pixel.
            dists_target_prediction: List that contains for each positive target pixel the distance to the closest
                positive prediction pixel
            percentile (float, optional): Percentile for which the Hausdorff distance is to be calculated, must be in
                :math:`\[0, 100\]`.

        Returns:
            float: Hausdorff distance.
        """

        distances = np.hstack((dists_prediction_target, dists_target_prediction))
        hausdorff_distance = torch.as_tensor(
            np.percentile(
                distances,
                q=percentile,
            )
        ).float()

        return hausdorff_distance

    # pylint: disable=too-many-locals
    def test_standard_case(self):
        """
        Tests that the CombinedPerImageMetric class correctly computes all metrics for all confidence levels.
        """

        (
            predicted_probabilities,
            target,
            cardinalities_0_2,
            cardinalities_0_5,
            cardinalities_0_8,
        ) = self.example_data()

        tp_0_2, fp_0_2, tn_0_2, fn_0_2 = cardinalities_0_2
        tp_0_5, fp_0_5, tn_0_5, fn_0_5 = cardinalities_0_5
        tp_0_8, fp_0_8, tn_0_8, fn_0_8 = cardinalities_0_8

        expected_dice_score_0_2 = torch.as_tensor(
            2 * tp_0_2 / (2 * tp_0_2 + fp_0_2 + fn_0_2)
        )
        expected_dice_score_0_5 = torch.as_tensor(
            2 * tp_0_5 / (2 * tp_0_5 + fp_0_5 + fn_0_5)
        )
        expected_dice_score_0_8 = torch.as_tensor(
            2 * tp_0_8 / (2 * tp_0_8 + fp_0_8 + fn_0_8)
        )

        expected_sensitivity_0_2 = torch.as_tensor(tp_0_2 / (tp_0_2 + fn_0_2))
        expected_sensitivity_0_5 = torch.as_tensor(tp_0_5 / (tp_0_5 + fn_0_5))
        expected_sensitivity_0_8 = torch.as_tensor(tp_0_8 / (tp_0_8 + fn_0_8))

        expected_specificity_0_2 = torch.as_tensor(tn_0_2 / (tn_0_2 + fp_0_2))
        expected_specificity_0_5 = torch.as_tensor(tn_0_5 / (tn_0_5 + fp_0_5))
        expected_specificity_0_8 = torch.as_tensor(tn_0_8 / (tn_0_8 + fp_0_8))

        # fmt: off
        dists_prediction_target_0_2 = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, np.sqrt(2), 1]
        dist_target_prediction_0_2 = [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]

        dists_prediction_target_0_5 = [0, 1, 0, 0, 0, 0, 1, 1]
        dist_target_prediction_0_5 = [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, np.sqrt(2), np.sqrt(3)]

        dists_prediction_target_0_8 = [0]
        dist_target_prediction_0_8 = [2, np.sqrt(1 + 4), np.sqrt(4 + 4), 1, np.sqrt(2), 0, 1, np.sqrt(1 + 4),
                                      np.sqrt(1 + 1 + 4), np.sqrt(2), np.sqrt(3), np.sqrt(1 + 1 + 4),
                                      np.sqrt(4 + 1 + 1), np.sqrt(4 + 4 + 1)]
        # fmt: on

        maximum_distance = np.sqrt(9 * 3)

        expected_hausdorff_distance_0_2 = self._hausdorff_distance(
            dists_prediction_target_0_2, dist_target_prediction_0_2
        )
        expected_hausdorff_distance_0_2 /= maximum_distance

        expected_hausdorff_distance_0_5 = self._hausdorff_distance(
            dists_prediction_target_0_5, dist_target_prediction_0_5
        )
        expected_hausdorff_distance_0_5 /= maximum_distance

        expected_hausdorff_distance_0_8 = self._hausdorff_distance(
            dists_prediction_target_0_8, dist_target_prediction_0_8
        )
        expected_hausdorff_distance_0_8 /= maximum_distance

        metrics = ["dice_score", "sensitivity", "specificity", "hausdorff95"]
        confidence_levels = [0.2, 0.5, 0.8]

        metrics_per_image = CombinedPerImageMetric(
            metrics=metrics,
            confidence_levels=confidence_levels,
            slices_per_image=3,
        )

        for idx in range(predicted_probabilities.shape[0]):
            metrics_per_image.update(predicted_probabilities[idx], target[idx])

        computed_metrics = metrics_per_image.compute()

        self.assertEqual(
            len(computed_metrics.keys()),
            len(confidence_levels) * len(metrics),
            "The returned metrics objects contains one entry per metric and confidence level",
        )

        self.assertTrue(
            torch.equal(computed_metrics["dice_score_0.2"], expected_dice_score_0_2),
            "The Dice score is computed correctly for confidence level 0.2.",
        )

        self.assertTrue(
            torch.equal(computed_metrics["dice_score_0.5"], expected_dice_score_0_5),
            "The Dice score is computed correctly for confidence level 0.5.",
        )

        self.assertTrue(
            torch.equal(computed_metrics["dice_score_0.8"], expected_dice_score_0_8),
            "The Dice score is computed correctly for confidence level 0.8.",
        )

        self.assertTrue(
            torch.equal(computed_metrics["sensitivity_0.2"], expected_sensitivity_0_2),
            "The sensitivity is computed correctly for confidence level 0.2.",
        )

        self.assertTrue(
            torch.equal(computed_metrics["sensitivity_0.5"], expected_sensitivity_0_5),
            "The sensitivity is computed correctly for confidence level 0.5.",
        )

        self.assertTrue(
            torch.equal(computed_metrics["sensitivity_0.8"], expected_sensitivity_0_8),
            "The sensitivity is computed correctly for confidence level 0.8.",
        )

        self.assertTrue(
            torch.equal(computed_metrics["specificity_0.2"], expected_specificity_0_2),
            "The specificity is computed correctly for confidence level 0.2.",
        )

        self.assertTrue(
            torch.equal(computed_metrics["specificity_0.5"], expected_specificity_0_5),
            "The specificity is computed correctly for confidence level 0.5.",
        )

        self.assertTrue(
            torch.equal(computed_metrics["specificity_0.8"], expected_specificity_0_8),
            "The specificity is computed correctly for confidence level 0.8.",
        )

        torch.testing.assert_allclose(
            computed_metrics["hausdorff95_0.2"],
            expected_hausdorff_distance_0_2,
            msg="The Hausdorff distance is computed correctly for confidence level 0.2.",
        )

        torch.testing.assert_allclose(
            computed_metrics["hausdorff95_0.5"],
            expected_hausdorff_distance_0_5,
            msg="The Hausdorff distance is computed correctly for confidence level 0.5.",
        )

        torch.testing.assert_allclose(
            computed_metrics["hausdorff95_0.8"],
            expected_hausdorff_distance_0_8,
            msg="The Hausdorff distance is computed correctly for confidence level 0.8.",
        )

    def test_metric_reset(self):
        """
        Tests that the metrics are reset correctly.
        """

        predicted_probabilities = torch.Tensor(
            [
                [
                    [0.1, 0.1],
                    [0.9, 0.9],
                ],
                [
                    [0.9, 0.9],
                    [0.9, 0.9],
                ],
            ]
        )

        target = torch.Tensor(
            [
                [
                    [0, 1],
                    [1, 0],
                ],
                [
                    [1, 1],
                    [1, 1],
                ],
            ]
        )

        metrics = ["dice_score", "sensitivity", "specificity", "hausdorff95"]
        confidence_levels = [0.2, 0.5, 0.8]

        metrics_per_image = CombinedPerImageMetric(
            metrics=metrics,
            confidence_levels=confidence_levels,
            slices_per_image=2,
        )

        metrics_per_image.update(predicted_probabilities[0], target[0])
        metrics_per_image.compute()

        metrics_per_image.reset()

        metrics_per_image.update(predicted_probabilities[1], target[1])
        computed_metrics = metrics_per_image.compute()

        for metric in metrics:
            for confidence_level in confidence_levels:
                metric_value = computed_metrics[f"{metric}_{confidence_level}"]

                if metric in ["dice_score", "sensitivity"]:
                    self.assertTrue(
                        torch.equal(torch.as_tensor(1.0), metric_value),
                        "Functional implementation correctly computes dice score when there are only TN.",
                    )
                if metric == "hausdorff95":
                    self.assertTrue(
                        torch.equal(torch.as_tensor(0.0), metric_value),
                        "Functional implementation correctly computes dice score when there are only TN.",
                    )
                if metric == "specificity":
                    self.assertTrue(
                        torch.isnan(metric_value),
                        "Functional implementation correctly computes dice score when there are only TN.",
                    )
