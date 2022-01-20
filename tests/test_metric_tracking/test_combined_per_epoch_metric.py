"""Tests for the CombinedPerImageMetric class"""

import unittest

import torch

from metric_tracking import CombinedPerEpochMetric
import tests.utils
import tests.utils.test_data_cardinality_metrics as test_data


class TestCombinedPerEpochMetric(unittest.TestCase):
    """
    Tests for the CombinedPerImageMetric class.
    """

    # pylint: disable=too-many-locals
    def test_standard_case(self):
        """
        Tests that the CombinedPerEpochMetric class correctly aggregates the per-image metrics.
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
            (
                prediction_1,
                target_1,
                cardinalities_1,
                probability_positive_1,
                probability_negative_1,
            ) = test_slice_1(True)

            (
                prediction_2,
                target_2,
                cardinalities_2,
                probability_positive_2,
                probability_negative_2,
            ) = test_slice_2(True)

            image_ids = ["1", "2"]

            expected_dice_score_1 = tests.utils.expected_metrics(
                "dice_score",
                cardinalities_1,
                probability_positive_1,
                probability_negative_1,
                epsilon=0,
            )
            expected_dice_score_1 = torch.from_numpy(expected_dice_score_1)

            expected_dice_score_2 = tests.utils.expected_metrics(
                "dice_score",
                cardinalities_2,
                probability_positive_2,
                probability_negative_2,
                epsilon=0,
            )
            expected_dice_score_2 = torch.from_numpy(expected_dice_score_2)

            prediction_batch = torch.stack([prediction_1, prediction_2])
            target_batch = torch.stack([target_1, target_2])

            metrics = ["dice_score", "sensitivity", "specificity", "hausdorff95"]
            confidence_levels = [0.2, 0.5, 0.8]

            for include_background_in_reduced_metrics in [True, False]:
                for reduction in ["mean", "min", "max"]:
                    per_epoch_metrics_module = CombinedPerEpochMetric(
                        metrics,
                        {
                            0: "first_test_class",
                            1: "second_test_class",
                            2: "third_test_class",
                        },
                        image_ids,
                        slices_per_image=1,
                        include_background_in_reduced_metrics=include_background_in_reduced_metrics,
                        multi_label=multi_label,
                        confidence_levels=confidence_levels,
                        reduction=reduction,
                    )

                    if include_background_in_reduced_metrics or multi_label:
                        expected_reduced_dice_score = torch.stack(
                            [expected_dice_score_1, expected_dice_score_2]
                        )
                    else:
                        expected_reduced_dice_score = torch.stack(
                            [expected_dice_score_1[1:], expected_dice_score_2[1:]]
                        )

                    expected_reduced_dice_score = expected_reduced_dice_score.mean(
                        dim=0
                    )

                    if reduction == "mean":
                        expected_reduced_dice_score = expected_reduced_dice_score.mean()
                    elif reduction == "min":
                        expected_reduced_dice_score = expected_reduced_dice_score.min()
                    elif reduction == "max":
                        expected_reduced_dice_score = expected_reduced_dice_score.max()

                    per_epoch_metrics_module.update(
                        prediction_batch, target_batch, image_ids
                    )

                    per_epoch_metrics = per_epoch_metrics_module.compute()

                    if multi_label:
                        num_classes = 3
                        self.assertEqual(
                            len(per_epoch_metrics.keys()),
                            len(confidence_levels) * len(metrics) * (num_classes + 1),
                            "The returned metrics object contains one entry per class, metric and confidence level and "
                            "one entry per metric and confidence level for the metrics aggregated over all classes.",
                        )

                        self.assertEqual(
                            per_epoch_metrics["dice_score_first_test_class_0.5"],
                            torch.stack(
                                [expected_dice_score_1[0], expected_dice_score_2[0]]
                            )
                            .float()
                            .mean(),
                            "The returned metrics object contains the mean of the per-image metrics.",
                        )

                        torch.testing.assert_allclose(
                            per_epoch_metrics[f"{reduction}_dice_score_0.5"],
                            expected_reduced_dice_score,
                            msg=f"For multi-label tasks, the returned metrics object contains the {reduction} of the "
                            "per-class metrics.",
                        )
                    else:
                        self.assertEqual(
                            per_epoch_metrics["dice_score_first_test_class"],
                            torch.stack(
                                [expected_dice_score_1[0], expected_dice_score_2[0]]
                            )
                            .float()
                            .mean(),
                            "For single-label tasks, the returned metrics object contains the mean of the per-image"
                            " metrics.",
                        )

                        torch.testing.assert_allclose(
                            per_epoch_metrics[f"{reduction}_dice_score"],
                            expected_reduced_dice_score,
                            msg=f"For single-label tasks, the returned metrics object contains the {reduction} of the "
                            "per-image metrics.",
                        )

    def test_attribute_passing(self):
        """Tests that the relevant attributes are correctly passed to the per-image metrics."""

        for multi_label in [True, False]:
            for include_background in [True, False]:
                confidence_levels = [0.1, 0.2, 0.3, 0.4]

                metrics_per_epoch = CombinedPerEpochMetric(
                    ["dice_score", "sensitivity", "specificity", "hausdorff95"],
                    {
                        0: "first_test_class",
                        1: "second_test_class",
                        2: "third_test_class",
                    },
                    ["1", "2"],
                    1,
                    include_background_in_reduced_metrics=include_background,
                    multi_label=multi_label,
                    confidence_levels=confidence_levels,
                )

                # pylint: disable=protected-access
                for metric_per_image in metrics_per_epoch._metrics_per_image.values():
                    self.assertTrue(metric_per_image.multi_label == multi_label)
                    self.assertTrue(
                        len(metric_per_image.confidence_levels)
                        == len(confidence_levels)
                    )
