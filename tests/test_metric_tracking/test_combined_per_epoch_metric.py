# """Tests for the CombinedPerImageMetric class"""
#
# import unittest
#
# import torch
#
# from metric_tracking import CombinedPerEpochMetric
# import tests.utils
#
#
# class TestCombinedPerEpochMetric(unittest.TestCase):
#     """
#     Tests for the CombinedPerImageMetric class.
#     """
#
#     # pylint: disable=too-many-locals
#     def test_standard_case(self):
#         """
#         Tests that the CombinedPerEpochMetric class correctly aggregates the per-image metrics.
#         """
#
#         prediction_1, target_1, tp_1, fp_1, _, fn_1 = tests.utils.standard_slice_1()
#         prediction_2, target_2, tp_2, fp_2, _, fn_2 = tests.utils.standard_slice_2()
#         image_ids = ["1", "2"]
#
#         prediction_batch = torch.stack([prediction_1, prediction_2])
#         target_batch = torch.stack([target_1, target_2])
#
#         expected_dice_score_0_5_1 = torch.tensor(2 * tp_1 / (2 * tp_1 + fp_1 + fn_1))
#         expected_dice_score_0_5_2 = torch.tensor(2 * tp_2 / (2 * tp_2 + fp_2 + fn_2))
#
#         metrics = ["dice_score", "sensitivity", "specificity", "hausdorff95"]
#         confidence_levels = [0.2, 0.5, 0.8]
#
#         mean_metrics_per_epoch = CombinedPerEpochMetric(
#             metrics=metrics,
#             confidence_levels=confidence_levels,
#             image_ids=image_ids,
#             slices_per_image=3,
#             metrics_to_aggregate=[],
#             reduction="mean",
#         )
#
#         mean_metrics_per_epoch.update(prediction_batch, target_batch, image_ids)
#
#         computed_mean_metrics_per_epoch = mean_metrics_per_epoch.compute()
#
#         self.assertEqual(
#             len(computed_mean_metrics_per_epoch.keys()),
#             len(confidence_levels) * len(metrics),
#             "The returned metrics object contains one entry per metric and confidence level",
#         )
#
#         self.assertEqual(
#             computed_mean_metrics_per_epoch["mean_dice_score_0.5"],
#             torch.stack([expected_dice_score_0_5_1, expected_dice_score_0_5_2]).mean(),
#             "The returned metrics object contains the averages of the per-image metrics.",
#         )
#
#         max_metrics_per_epoch = CombinedPerEpochMetric(
#             metrics=metrics,
#             confidence_levels=confidence_levels,
#             image_ids=image_ids,
#             slices_per_image=3,
#             metrics_to_aggregate=[],
#             reduction="max",
#         )
#
#         max_metrics_per_epoch.update(prediction_batch, target_batch, image_ids)
#
#         computed_max_metrics_per_epoch = max_metrics_per_epoch.compute()
#
#         self.assertEqual(
#             computed_max_metrics_per_epoch["max_dice_score_0.5"],
#             torch.stack([expected_dice_score_0_5_1, expected_dice_score_0_5_2]).max(),
#             "The returned metrics object contains the averages of the per-image metrics.",
#         )
#
#         min_metrics_per_epoch = CombinedPerEpochMetric(
#             metrics=metrics,
#             confidence_levels=confidence_levels,
#             image_ids=image_ids,
#             slices_per_image=3,
#             metrics_to_aggregate=[],
#             reduction="min",
#         )
#
#         min_metrics_per_epoch.update(prediction_batch, target_batch, image_ids)
#
#         computed_min_metrics_per_epoch = min_metrics_per_epoch.compute()
#
#         self.assertEqual(
#             computed_min_metrics_per_epoch["min_dice_score_0.5"],
#             torch.stack([expected_dice_score_0_5_1, expected_dice_score_0_5_2]).min(),
#             "The returned metrics object contains the averages of the per-image metrics.",
#         )
#
#         metrics_per_image = CombinedPerEpochMetric(
#             metrics=metrics,
#             confidence_levels=confidence_levels,
#             image_ids=image_ids,
#             slices_per_image=3,
#             metrics_to_aggregate=[],
#             reduction="none",
#         )
#
#         metrics_per_image.update(prediction_batch, target_batch, image_ids)
#
#         computed_metrics_per_image = metrics_per_image.compute()
#
#         self.assertEqual(
#             len(computed_metrics_per_image.keys()),
#             len(confidence_levels) * len(metrics) * len(image_ids),
#             "The returned metrics object contains one entry per metric, confidence level and image",
#         )
#
#         self.assertEqual(
#             computed_metrics_per_image["dice_score_0.5_1"],
#             expected_dice_score_0_5_1,
#             "The returned metrics contains the per-image metrics.",
#         )
#
#         self.assertEqual(
#             computed_metrics_per_image["dice_score_0.5_2"],
#             expected_dice_score_0_5_2,
#             "The returned metrics contains the per-image metrics.",
#         )
