# """Tests for the false positive loss"""
#
# from typing import Callable, Optional, Tuple
# import unittest
# import torch
#
# from functional import FalsePositiveLoss
# import tests.utils
#
#
# class TestFalsePositiveLoss(unittest.TestCase):
#     """
#     Test cases for false positive loss.
#     """
#
#     def _test_fp_loss(
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
#         Helper function that calculates the false positive loss with the given settings for the given predictions and
#         compares it with an expected value.
#
#         Args:
#             get_first_slice: Getter function that returns prediction, target and cardinalities for the first slice.
#             get_second_slice: Getter function that returns prediction, target and cardinalities for the second slice.
#             reduction (string, optional): Reduction parameter of false positive loss.
#             smoothing (float, optional): Smoothing parameter of false positive loss.
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
#             _,
#         ) = get_first_slice()
#         (
#             predictions_second_slice,
#             target_second_slice,
#             tp_second,
#             fp_second,
#             _,
#             _,
#         ) = get_second_slice()
#
#         prediction = torch.stack([predictions_first_slice, predictions_second_slice])
#         target = torch.stack([target_first_slice, target_second_slice])
#
#         if expected_loss is None:
#             loss_first_slice = fp_first / (tp_first + fp_first + smoothing)
#             loss_second_slice = fp_second / (tp_second + fp_second + smoothing)
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
#         fp_loss = FalsePositiveLoss(reduction=reduction, smoothing=smoothing)
#         loss = fp_loss(prediction, target)
#
#         self.assertTrue(
#             loss.shape == expected_loss.shape, "Returns loss tensor with correct shape."
#         )
#         torch.testing.assert_allclose(
#             loss, expected_loss, msg="Correctly computes loss value."
#         )
#
#     def test_standard_case(self):
#         """
#         Tests that the false positive loss is computed correctly when there are both true and false predictions.
#         """
#
#         _, _, tp_1, fp_1, _, _ = tests.utils.standard_slice_1()
#         _, _, tp_2, fp_2, _, _ = tests.utils.standard_slice_2()
#
#         self._test_fp_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=0,
#             expected_loss=torch.Tensor(
#                 [[fp_1 / (tp_1 + fp_1)], [fp_2 / (tp_2 + fp_2)]]
#             ),
#         )
#         self._test_fp_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=0,
#             reduction="mean",
#             expected_loss=torch.as_tensor(
#                 (fp_1 / (tp_1 + fp_1) + fp_2 / (tp_2 + fp_2)) / 2
#             ),
#         )
#         self._test_fp_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=0,
#             reduction="sum",
#             expected_loss=torch.as_tensor(fp_1 / (tp_1 + fp_1) + fp_2 / (tp_2 + fp_2)),
#         )
#
#         self._test_fp_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=1,
#             expected_loss=torch.Tensor(
#                 [[fp_1 / (tp_1 + fp_1 + 1)], [fp_2 / (tp_2 + fp_2 + 1)]]
#             ),
#         )
#         self._test_fp_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(
#                 (fp_1 / (tp_1 + fp_1 + 1) + fp_2 / (tp_2 + fp_2 + 1)) / 2
#             ),
#         )
#         self._test_fp_loss(
#             tests.utils.standard_slice_1,
#             tests.utils.standard_slice_2,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(
#                 fp_1 / (tp_1 + fp_1 + 1) + fp_2 / (tp_2 + fp_2 + 1)
#             ),
#         )
#
#     def test_no_false_positives(self):
#         """
#         Tests that the false positive loss is computed correctly when there are no false positives.
#         """
#
#         _, _, _, _, _, _ = tests.utils.slice_all_true()
#
#         self._test_fp_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=0,
#             expected_loss=torch.Tensor([[0.0], [0.0]]),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=0,
#             reduction="mean",
#             expected_loss=torch.as_tensor(0.0),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=0,
#             reduction="sum",
#             expected_loss=torch.as_tensor(0.0),
#         )
#
#         self._test_fp_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=1,
#             expected_loss=torch.Tensor([[0.0], [0.0]]),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(0.0),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_true,
#             tests.utils.slice_all_true,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(0.0),
#         )
#
#     def test_all_false(self):
#         """
#         Tests that the false positive loss is computed correctly when all predictions are wrong.
#         """
#
#         _, _, _, fp, _, _ = tests.utils.slice_all_false()
#
#         self._test_fp_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=0,
#             expected_loss=torch.Tensor([[1.0], [1.0]]),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=0,
#             reduction="mean",
#             expected_loss=torch.as_tensor(1.0),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=0,
#             reduction="sum",
#             expected_loss=torch.as_tensor(2.0),
#         )
#
#         self._test_fp_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=1,
#             expected_loss=torch.Tensor([[fp / (fp + 1)], [fp / (fp + 1)]]),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(fp / (fp + 1)),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_false,
#             tests.utils.slice_all_false,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(2 * fp / (fp + 1)),
#         )
#
#     def test_no_positives(self):
#         """
#         Tests that the false positive loss is computed correctly when there are no positives.
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
#         fp_loss = FalsePositiveLoss(smoothing=0, reduction="none")
#         loss = fp_loss(prediction, target)
#         self.assertTrue(torch.isnan(loss).all(), "Correctly computes loss value.")
#
#         fp_loss = FalsePositiveLoss(smoothing=0, reduction="mean")
#         loss = fp_loss(prediction, target)
#         self.assertTrue(torch.isnan(loss).all(), "Correctly computes loss value.")
#
#         fp_loss = FalsePositiveLoss(smoothing=0, reduction="sum")
#         loss = fp_loss(prediction, target)
#         self.assertTrue(torch.isnan(loss).all(), "Correctly computes loss value.")
#
#         self._test_fp_loss(
#             tests.utils.slice_all_true_negatives,
#             tests.utils.slice_all_true_negatives,
#             smoothing=1,
#             expected_loss=torch.Tensor([[0.0], [0.0]]),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_true_negatives,
#             tests.utils.slice_all_true_negatives,
#             smoothing=1,
#             reduction="mean",
#             expected_loss=torch.as_tensor(0.0),
#         )
#         self._test_fp_loss(
#             tests.utils.slice_all_true_negatives,
#             tests.utils.slice_all_true_negatives,
#             smoothing=1,
#             reduction="sum",
#             expected_loss=torch.as_tensor(0.0),
#         )
