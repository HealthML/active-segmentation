# """Tests for the dice loss."""
#
# from typing import Callable, Optional, Tuple
# import unittest
# import torch
#
# from functional import DiceLoss
# import tests.utils
#
#
# class TestDiceLoss(unittest.TestCase):
#     """
#     Test cases for dice loss.
#     """
#
#     def _test_dice_loss(
#         self,
#         get_first_slice: Callable[
#             [], Tuple[torch.Tensor, torch.Tensor, float, float, float, float]
#         ],
#         get_second_slice: Callable[
#             [], Tuple[torch.Tensor, torch.Tensor, float, float, float, float]
#         ],
#         reduction: Optional[str] = "none",
#         smoothing: float = 1,
#         expected_loss: Optional[torch.Tensor] = None,
#     ) -> None:
#         """
#         Helper function that calculates the dice loss with the given settings for the given predictions and compares
#         it with an expected value.
#
#         Args:
#             get_first_slice: Getter function that returns prediction, target and cardinalities for the first slice.
#             get_second_slice: Getter function that returns prediction, target and cardinalities for the second slice.
#             reduction (string, optional): Reduction parameter of dice loss.
#             smoothing (float, optional): Smoothing parameter of dice loss.
#             expected_loss (Tensor, optional): Expected loss value.
#         """
#
#         # pylint: disable-msg=too-many-locals
#
#         (
#             predictions_first_slice,
#             target_first_slice,
#             tp_first,
#             fp_first,
#             _,
#             fn_first,
#         ) = get_first_slice()
#         (
#             predictions_second_slice,
#             target_second_slice,
#             tp_second,
#             fp_second,
#             _,
#             fn_second,
#         ) = get_second_slice()
#
#         prediction = torch.stack([predictions_first_slice, predictions_second_slice])
#         target = torch.stack([target_first_slice, target_second_slice])
#
#         if expected_loss is None:
#             loss_first_slice = (
#                 -1
#                 * (2 * tp_first + smoothing)
#                 / (2 * tp_first + fp_first + fn_first + smoothing)
#             )
#             loss_second_slice = (
#                 -1
#                 * (2 * tp_second + smoothing)
#                 / (2 * tp_second + fp_second + fn_second + smoothing)
#             )
#
#             if reduction == "mean":
#                 expected_loss = torch.as_tensor(
#                     (loss_first_slice + loss_second_slice) / 2
#                 )
#             elif reduction == "sum":
#                 expected_loss = torch.as_tensor(loss_first_slice + loss_second_slice)
#             else:
#                 expected_loss = torch.Tensor([[loss_first_slice], [loss_second_slice]])
#
#         dice_loss = DiceLoss(reduction=reduction, smoothing=smoothing)
#         loss = dice_loss(prediction, target)
#
#         self.assertTrue(
#             loss.shape == expected_loss.shape, "Returns loss tensor with correct shape."
#         )
#         torch.testing.assert_allclose(
#             loss, expected_loss, msg="Correctly computes loss value."
#         )
#
#     def test_standard_case(self) -> None:
#         """
#         Tests that the dice loss is computed correctly when there are both true and false predictions.
#         """
#
#         self._test_dice_loss(
#             tests.utils.standard_slice_1, tests.utils.standard_slice_2, smoothing=0
#         )
#         self._test_dice_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=0,
#             reduction="mean",
#         )
#         self._test_dice_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=0,
#             reduction="sum",
#         )
#
#         self._test_dice_loss(
#             tests.utils.standard_slice_1, tests.utils.standard_slice_2, smoothing=1
#         )
#         self._test_dice_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=1,
#             reduction="mean",
#         )
#         self._test_dice_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=1,
#             reduction="sum",
#         )
#
#     def test_all_true(self):
#         """
#         Tests that the dice loss is computed correctly when all predictions are correct.
#         """
#
#         self._test_dice_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=0,
#             expected_loss=torch.Tensor([[-1.0], [-1.0]]),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=0,
#             reduction="mean",
#             expected_loss=torch.as_tensor(-1.0),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=0,
#             reduction="sum",
#             expected_loss=torch.as_tensor(-2.0),
#         )
#
#         self._test_dice_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=1,
#             expected_loss=torch.Tensor([[-1.0], [-1.0]]),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(-1.0),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(-2.0),
#         )
#
#     def test_all_false(self):
#         """
#         Tests that the dice loss is computed correctly when all predictions are wrong.
#         """
#
#         self._test_dice_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=0,
#             expected_loss=torch.Tensor([[0.0], [0.0]]),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=0,
#             reduction="mean",
#             expected_loss=torch.as_tensor(0.0),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=0,
#             reduction="sum",
#             expected_loss=torch.as_tensor(0.0),
#         )
#
#         _, _, _, fp, _, fn = tests.utils.slice_all_false()
#
#         self._test_dice_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=1,
#             expected_loss=torch.Tensor(
#                 [[-1.0 / (1.0 + fp + fn)], [-1.0 / (1.0 + fp + fn)]]
#             ),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(-1.0 / (1.0 + fp + fn)),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(-2.0 / (1.0 + fp + fn)),
#         )
#
#     def test_no_true_positives(self):
#         """
#         Tests that the dice loss is computed correctly when there are no true positives.
#         """
#
#         self._test_dice_loss(
#             tests.utils.slice_no_true_positives,
#             tests.utils.slice_no_true_positives,
#             smoothing=0,
#             expected_loss=torch.Tensor([[0.0], [0.0]]),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_no_true_positives,
#             tests.utils.slice_no_true_positives,
#             smoothing=0,
#             reduction="mean",
#             expected_loss=torch.as_tensor(0.0),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_no_true_positives,
#             tests.utils.slice_no_true_positives,
#             smoothing=0,
#             reduction="sum",
#             expected_loss=torch.as_tensor(0.0),
#         )
#
#         _, _, _, fp, _, fn = tests.utils.slice_no_true_positives()
#
#         self._test_dice_loss(
#             tests.utils.slice_no_true_positives,
#             tests.utils.slice_no_true_positives,
#             smoothing=1,
#             expected_loss=torch.Tensor(
#                 [[-1.0 / (1.0 + fp + fn)], [-1.0 / (1.0 + fp + fn)]]
#             ),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_no_true_positives,
#             tests.utils.slice_no_true_positives,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(-1.0 / (1.0 + fp + fn)),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_no_true_positives,
#             tests.utils.slice_no_true_positives,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(-2.0 / (1.0 + fp + fn)),
#         )
#
#     def test_no_true_negatives(self):
#         """
#         Tests that the dice loss is computed correctly when there are no true negatives.
#         """
#
#         _, _, tp, fp, _, fn = tests.utils.slice_no_true_negatives()
#
#         self._test_dice_loss(
#             tests.utils.slice_no_true_negatives,
#             tests.utils.slice_no_true_negatives,
#             smoothing=0,
#             expected_loss=torch.Tensor(
#                 [[-2.0 * tp / (2.0 * tp + fp + fn)], [-2.0 * tp / (2.0 * tp + fp + fn)]]
#             ),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_no_true_negatives,
#             tests.utils.slice_no_true_negatives,
#             smoothing=0,
#             reduction="mean",
#             expected_loss=torch.as_tensor(-2.0 * tp / (2.0 * tp + fp + fn)),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_no_true_negatives,
#             tests.utils.slice_no_true_negatives,
#             smoothing=0,
#             reduction="sum",
#             expected_loss=torch.as_tensor(-4.0 * tp / (2.0 * tp + fp + fn)),
#         )
#
#         self._test_dice_loss(
#             tests.utils.slice_no_true_negatives,
#             tests.utils.slice_no_true_negatives,
#             smoothing=1,
#             expected_loss=torch.Tensor(
#                 [
#                     [-1.0 * (2.0 * tp + 1.0) / (2 * tp + fp + fn + 1.0)],
#                     [-1.0 * (2.0 * tp + 1.0) / (2 * tp + fp + fn + 1.0)],
#                 ]
#             ),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_no_true_negatives,
#             tests.utils.slice_no_true_negatives,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(
#                 -1.0 * (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
#             ),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_no_true_negatives,
#             tests.utils.slice_no_true_negatives,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(
#                 -2.0 * (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
#             ),
#         )
#
#     def test_all_true_negatives(self):
#         """
#         Tests that the dice loss is computed correctly when there are only true negatives.
#         """
#
#         (
#             predictions_first_slice,
#             target_first_slice,
#             _,
#             _,
#             _,
#             _,
#         ) = tests.utils.slice_all_true_negatives()
#         (
#             predictions_second_slice,
#             target_second_slice,
#             _,
#             _,
#             _,
#             _,
#         ) = tests.utils.slice_all_true_negatives()
#
#         prediction = torch.stack([predictions_first_slice, predictions_second_slice])
#         target = torch.stack([target_first_slice, target_second_slice])
#
#         dice_loss = DiceLoss(smoothing=0, reduction="none")
#         loss = dice_loss(prediction, target)
#         self.assertTrue(torch.isnan(loss).all(), "Correctly computes loss value.")
#
#         dice_loss = DiceLoss(smoothing=0, reduction="mean")
#         loss = dice_loss(prediction, target)
#         self.assertTrue(torch.isnan(loss).all(), "Correctly computes loss value.")
#
#         dice_loss = DiceLoss(smoothing=0, reduction="sum")
#         loss = dice_loss(prediction, target)
#         self.assertTrue(torch.isnan(loss).all(), "Correctly computes loss value.")
#
#         self._test_dice_loss(
#             tests.utils.slice_all_true_negatives,
#             tests.utils.slice_all_true_negatives,
#             smoothing=1,
#             expected_loss=torch.Tensor([[-1.0], [-1.0]]),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_true_negatives,
#             tests.utils.slice_all_true_negatives,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(-1.0),
#         )
#         self._test_dice_loss(
#             tests.utils.slice_all_true_negatives,
#             tests.utils.slice_all_true_negatives,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(-2.0),
#         )
#
#     def test_probabilistic_predictions(self):
#         """
#         Tests that the dice loss is computed correctly when predicted class probabilities are used as input instead of
#         sharp segmentations.
#         """
#
#         expected_intersection = 3.7
#         expected_denominator = 2.97 + 5
#         expected_loss = -2 * expected_intersection / expected_denominator
#
#         self._test_dice_loss(
#             tests.utils.probabilistic_slice,
#             tests.utils.probabilistic_slice,
#             smoothing=0,
#             expected_loss=torch.Tensor([[expected_loss], [expected_loss]]),
#         )
#         self._test_dice_loss(
#             tests.utils.probabilistic_slice,
#             tests.utils.probabilistic_slice,
#             smoothing=0,
#             reduction="mean",
#             expected_loss=torch.as_tensor(expected_loss),
#         )
#         self._test_dice_loss(
#             tests.utils.probabilistic_slice,
#             tests.utils.probabilistic_slice,
#             smoothing=0,
#             reduction="sum",
#             expected_loss=torch.as_tensor(2 * expected_loss),
#         )
#
#         expected_loss = (
#             -1 * (2 * expected_intersection + 1) / (expected_denominator + 1)
#         )
#         self._test_dice_loss(
#             tests.utils.probabilistic_slice,
#             tests.utils.probabilistic_slice,
#             smoothing=1,
#             expected_loss=torch.Tensor([[expected_loss], [expected_loss]]),
#         )
#         self._test_dice_loss(
#             tests.utils.probabilistic_slice,
#             tests.utils.probabilistic_slice,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(expected_loss),
#         )
#         self._test_dice_loss(
#             tests.utils.probabilistic_slice,
#             tests.utils.probabilistic_slice,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(2 * expected_loss),
#         )
#
#     def test_multi_class_prediction(self):
#         """
#         Tests that the dice loss is computed correctly for multi-class, multi-label prediction tasks.
#         """
#         # pylint: disable-msg=too-many-locals, disable-msg=no-self-use
#
#         prediction_1, target_1, tp_1, fp_1, _, fn_1 = tests.utils.standard_slice_1()
#         prediction_2, target_2, tp_2, fp_2, _, fn_2 = tests.utils.standard_slice_2()
#
#         dice_loss = DiceLoss(smoothing=0, reduction="mean")
#
#         loss_1 = -2 * tp_1 / (2 * tp_1 + fp_1 + fn_1)
#         loss_2 = -2 * tp_2 / (2 * tp_2 + fp_2 + fn_2)
#
#         multi_class_prediction_1 = torch.cat([prediction_1, prediction_2])
#         multi_class_prediction_2 = torch.cat([prediction_1, prediction_1])
#
#         multi_class_target_1 = torch.cat([target_1, target_2])
#         multi_class_target_2 = torch.cat([target_1, target_1])
#
#         loss = dice_loss(
#             torch.stack([multi_class_prediction_1, multi_class_prediction_2]),
#             torch.stack([multi_class_target_1, multi_class_target_2]),
#         )
#
#         torch.testing.assert_allclose(
#             loss,
#             torch.as_tensor(((loss_1 + loss_2) / 2 + loss_1) / 2),
#             msg="Correctly computes loss value.",
#         )
