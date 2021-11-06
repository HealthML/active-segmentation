import torch
from typing import Callable, Optional, Tuple
import unittest

from models import DiceLoss


def standard_slice_1():
    prediction_slice = torch.tensor(
        [[
            [0.1, 0.1, 0.],
            [0.9, 0.9, 0.4],
            [0.7, 0.8, 0.2]
        ]])

    target_slice = torch.tensor(
        [[
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 0]
        ]])

    tp = 4
    fp = 0
    tn = 4
    fn = 1

    return prediction_slice, target_slice, tp, fp, tn, fn


def standard_slice_2():
    prediction_slice = torch.tensor(
        [[
            [0.1, 0.8, 0.2],
            [0.9, 0.9, 0.3],
            [0.7, 0.8, 0.3]
        ]])

    target_slice = torch.tensor(
        [[
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]])

    tp = 4
    fp = 1
    tn = 3
    fn = 1

    return prediction_slice, target_slice, tp, fp, tn, fn


def slice_all_true():
    prediction_slice = torch.tensor(
        [[
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9]
        ]])

    target_slice = torch.tensor(
        [[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]])

    tp = 3
    fp = 0
    tn = 6
    fn = 0

    return prediction_slice, target_slice, tp, fp, tn, fn


def slice_all_false():
    prediction_slice = torch.tensor(
        [[
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.1]
        ]])

    target_slice = torch.tensor(
        [[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]])

    tp = 0
    fp = 6
    tn = 0
    fn = 3

    return prediction_slice, target_slice, tp, fp, tn, fn


def slice_no_true_positives():
    prediction_slice = torch.tensor(
        [[
            [0.1, 0.9, 0.9],
            [0.1, 0.1, 0.9],
            [0.1, 0.1, 0.1]
        ]])

    target_slice = torch.tensor(
        [[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]])

    tp = 0
    fp = 3
    tn = 3
    fn = 3

    return prediction_slice, target_slice, tp, fp, tn, fn


def slice_no_true_negatives():
    prediction_slice = torch.tensor(
        [[
            [0.9, 0.9, 0.9],
            [0.9, 0.9, 0.9],
            [0.9, 0.9, 0.1]
        ]])

    target_slice = torch.tensor(
        [[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]])

    tp = 2
    fp = 6
    tn = 0
    fn = 1

    return prediction_slice, target_slice, tp, fp, tn, fn


def slice_all_true_negatives():
    prediction_slice = torch.tensor(
        [[
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1]
        ]])

    target_slice = torch.tensor(
        [[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]])

    tp = 0
    fp = 0
    tn = 9
    fn = 9

    return prediction_slice, target_slice, tp, fp, tn, fn


class TestDiceLoss(unittest.TestCase):

    def _test_dice_loss(self, get_first_slice: Callable[[], Tuple[torch.Tensor, torch.Tensor, int, int, int, int]],
                        get_second_slice: Callable[[], Tuple[torch.Tensor, torch.Tensor, int, int, int, int]],
                        reduction: Optional[str] = "none",
                        smoothing: float = 1,
                        expected_loss: Optional[torch.Tensor] = None):

        predictions_first_slice, target_first_slice, tp_first, fp_first, _, fn_first = get_first_slice()
        predictions_second_slice, target_second_slice, tp_second, fp_second, _, fn_second = get_second_slice()

        prediction = torch.stack([predictions_first_slice, predictions_second_slice])
        target = torch.stack([target_first_slice, target_second_slice])

        if expected_loss is None:
            loss_first_slice = -1 * (2 * tp_first + smoothing) / (2 * tp_first + fp_first + fn_first + smoothing)
            loss_second_slice = -1 * (2 * tp_second + smoothing) / (2 * tp_second + fp_second + fn_second + smoothing)

            if reduction == "mean":
                expected_loss = torch.tensor((loss_first_slice + loss_second_slice) / 2)
            elif reduction == "sum":
                expected_loss = torch.tensor(loss_first_slice + loss_second_slice)
            else:
                expected_loss = torch.tensor([loss_first_slice, loss_second_slice])

        dice_loss = DiceLoss(reduction=reduction, smoothing=smoothing)
        loss = dice_loss(prediction, target)

        self.assertTrue(loss.shape == expected_loss.shape, "Returns loss tensor with correct shape.")
        self.assertTrue(loss.requires_grad, "Loss tensor requires gradient.")
        self.assertTrue(torch.equal(loss, expected_loss), "Correctly computes loss value.")

    def test_standard_case(self):
        self._test_dice_loss(standard_slice_1, standard_slice_2, smoothing=0)
        self._test_dice_loss(standard_slice_1, standard_slice_2, smoothing=0, reduction="mean")
        self._test_dice_loss(standard_slice_1, standard_slice_2, smoothing=0, reduction="sum")

        self._test_dice_loss(standard_slice_1, standard_slice_2, smoothing=1)
        self._test_dice_loss(standard_slice_1, standard_slice_2, smoothing=1, reduction="mean")
        self._test_dice_loss(standard_slice_1, standard_slice_2, smoothing=1, reduction="sum")

    def test_all_true(self):
        self._test_dice_loss(slice_all_true, slice_all_true, smoothing=0, expected_loss=torch.tensor([-1., -1.]))
        self._test_dice_loss(slice_all_true, slice_all_true, smoothing=0, reduction="mean",
                             expected_loss=torch.tensor(-1.))
        self._test_dice_loss(slice_all_true, slice_all_true, smoothing=0, reduction="sum",
                             expected_loss=torch.tensor(-2.))

        self._test_dice_loss(slice_all_true, slice_all_true, smoothing=1, expected_loss=torch.tensor([-1., -1.]))
        self._test_dice_loss(slice_all_true, slice_all_true, smoothing=1, reduction="mean",
                             expected_loss=torch.tensor(-1.))
        self._test_dice_loss(slice_all_true, slice_all_true, smoothing=1, reduction="sum",
                             expected_loss=torch.tensor(-2.))

    def test_all_false(self):
        self._test_dice_loss(slice_all_false, slice_all_false, smoothing=0,
                             expected_loss=torch.tensor([0., 0.]))
        self._test_dice_loss(slice_all_false, slice_all_false, smoothing=0, reduction="mean",
                             expected_loss=torch.tensor(0.))
        self._test_dice_loss(slice_all_false, slice_all_false, smoothing=0, reduction="sum",
                             expected_loss=torch.tensor(0.))

        _, _, _, fp, _, fn = slice_all_false()

        self._test_dice_loss(slice_all_false, slice_all_false, smoothing=1,
                             expected_loss=torch.tensor([-1. / (1.+fp+fn), - 1. / (1.+fp+fn)]))
        self._test_dice_loss(slice_all_false, slice_all_false, smoothing=1, reduction="mean",
                             expected_loss=torch.tensor(-1. / (1.+fp+fn)))
        self._test_dice_loss(slice_all_false, slice_all_false, smoothing=1, reduction="sum",
                             expected_loss=torch.tensor(-2. / (1.+fp+fn)))

    def test_no_true_positives(self):
        self._test_dice_loss(slice_no_true_positives, slice_no_true_positives, smoothing=0,
                             expected_loss=torch.tensor([0., 0.]))
        self._test_dice_loss(slice_no_true_positives, slice_no_true_positives, smoothing=0, reduction="mean",
                             expected_loss=torch.tensor(0.))
        self._test_dice_loss(slice_no_true_positives, slice_no_true_positives, smoothing=0, reduction="sum",
                             expected_loss=torch.tensor(0.))

        _, _, _, fp, _, fn = slice_no_true_positives()

        self._test_dice_loss(slice_no_true_positives, slice_no_true_positives, smoothing=1,
                             expected_loss=torch.tensor([-1. / (1.+fp+fn), -1. / (1.+fp+fn)]))
        self._test_dice_loss(slice_no_true_positives, slice_no_true_positives, smoothing=1, reduction="mean",
                             expected_loss=torch.tensor(-1. / (1.+fp+fn)))
        self._test_dice_loss(slice_no_true_positives, slice_no_true_positives, smoothing=1, reduction="sum",
                             expected_loss=torch.tensor(-2. / (1.+fp+fn)))

    def test_no_true_negatives(self):
        _, _, tp, fp, _, fn = slice_no_true_negatives()

        self._test_dice_loss(slice_no_true_negatives, slice_no_true_negatives, smoothing=0,
                             expected_loss=torch.tensor([-2.*tp / (2.*tp+fp+fn), -2.*tp / (2.*tp+fp+fn)]))
        self._test_dice_loss(slice_no_true_negatives, slice_no_true_negatives, smoothing=0, reduction="mean",
                             expected_loss=torch.tensor(-2.*tp / (2.*tp+fp+fn)))
        self._test_dice_loss(slice_no_true_negatives, slice_no_true_negatives, smoothing=0, reduction="sum",
                             expected_loss=torch.tensor(-4.*tp / (2.*tp+fp+fn)))

        self._test_dice_loss(slice_no_true_negatives, slice_no_true_negatives, smoothing=1,
                             expected_loss=torch.tensor([-1. * (2.*tp+1.) / (2*tp+fp+fn+1.), -1. * (2.*tp+1.) /
                                                         (2*tp+fp+fn+1.)]))
        self._test_dice_loss(slice_no_true_negatives, slice_no_true_negatives, smoothing=1, reduction="mean",
                             expected_loss=torch.tensor(-1. * (2.*tp+1.) / (2.*tp+fp+fn+1.)))
        self._test_dice_loss(slice_no_true_negatives, slice_no_true_negatives, smoothing=1, reduction="sum",
                             expected_loss=torch.tensor(-2. * (2.*tp+1.) / (2.*tp+fp+fn+1.)))

    def test_all_true_negatives(self):
        predictions_first_slice, target_first_slice, tp_first, fp_first, _, fn_first = slice_all_true_negatives()
        predictions_second_slice, target_second_slice, tp_second, fp_second, _, fn_second = slice_all_true_negatives()

        prediction = torch.stack([predictions_first_slice, predictions_second_slice])
        target = torch.stack([target_first_slice, target_second_slice])

        dice_loss = DiceLoss(smoothing=0, reduction="none")
        loss = dice_loss(prediction, target)
        self.assertTrue(torch.isnan(loss).all(), "Correctly computes loss value.")

        dice_loss = DiceLoss(smoothing=0, reduction="mean")
        loss = dice_loss(prediction, target)
        self.assertTrue(torch.isnan(loss).all(), "Correctly computes loss value.")

        dice_loss = DiceLoss(smoothing=0, reduction="sum")
        loss = dice_loss(prediction, target)
        self.assertTrue(torch.isnan(loss).all(), "Correctly computes loss value.")

        self._test_dice_loss(slice_all_true_negatives, slice_all_true_negatives, smoothing=1,
                             expected_loss=torch.tensor([-1., -1.]))
        self._test_dice_loss(slice_all_true_negatives, slice_all_true_negatives, smoothing=1, reduction="mean",
                             expected_loss=torch.tensor(-1.))
        self._test_dice_loss(slice_all_true_negatives, slice_all_true_negatives, smoothing=1, reduction="sum",
                             expected_loss=torch.tensor(-2.))


if __name__ == '__main__':
    unittest.main()
