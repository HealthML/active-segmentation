from typing import Tuple
import torch


def standard_slice_1() -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    """
    Creates a faked segmentation slice that contains both true and false predictions.

    Returns:
        Tuple: Predicted slice, target slice, true positives, false positives, true negatives, false negatives.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [[
            [0, 0, 0.],
            [1, 1, 0],
            [1, 1, 0]
        ]])

    target_slice = torch.Tensor(
        [[
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 0]
        ]])
    # fmt: on

    tp, fp, tn, fn = (4, 0, 4, 1)

    return prediction_slice, target_slice, tp, fp, tn, fn


def standard_slice_2() -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    """
    Creates another faked segmentation slice that contains both true and false predictions.

    Returns:
        Tuple: Predicted slice, target slice, true positives, false positives, true negatives, false negatives.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [[
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ]])

    target_slice = torch.Tensor(
        [[
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]])
    # fmt: on

    tp, fp, tn, fn = (4, 1, 3, 1)

    return prediction_slice, target_slice, tp, fp, tn, fn


def slice_all_true() -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    """
    Creates a faked segmentation slice that contains no segmentation errors.

    Returns:
        Tuple: Predicted slice, target slice, true positives, false positives, true negatives, false negatives.
    """

    # fmt: off
    target_slice = torch.Tensor(
        [[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]])
    # fmt: on

    tp, fp, tn, fn = (3, 0, 6, 0)

    return target_slice, target_slice, tp, fp, tn, fn


def slice_all_false() -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    """
    Creates a faked segmentation slice that contains only wrong predictions.

    Returns:
        Tuple: Predicted slice, target slice, true positives, false positives, true negatives, false negatives.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [[
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]])

    target_slice = torch.Tensor(
        [[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]])
    # fmt: on

    tp, fp, tn, fn = (0, 6, 0, 3)

    return prediction_slice, target_slice, tp, fp, tn, fn


def slice_no_true_positives() -> Tuple[
    torch.Tensor, torch.Tensor, float, float, float, float
]:
    """
    Creates a faked segmentation slice that contains no true positives.

    Returns:
        Tuple: Predicted slice, target slice, true positives, false positives, true negatives, false negatives.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [[
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ]])

    target_slice = torch.Tensor(
        [[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]])
    # fmt: on

    tp, fp, tn, fn = (0, 3, 3, 3)

    return prediction_slice, target_slice, tp, fp, tn, fn


def slice_no_true_negatives() -> Tuple[
    torch.Tensor, torch.Tensor, float, float, float, float
]:
    """
    Creates a faked segmentation slice that contains no true negatives.

    Returns:
        Tuple: Predicted slice, target slice, true positives, false positives, true negatives, false negatives.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
        ]])

    target_slice = torch.Tensor(
        [[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]])
    # fmt: on

    tp, fp, tn, fn = (2, 6, 0, 1)

    return prediction_slice, target_slice, tp, fp, tn, fn


def slice_all_true_positives() -> Tuple[
    torch.Tensor, torch.Tensor, float, float, float, float
]:
    """
    Creates a faked segmentation slice that only contains true negatives.

    Returns:
        Tuple: Predicted slice, target slice, true positives, false positives, true negatives, false negatives.
    """

    target_slice = torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])

    tp, fp, tn, fn = (9, 0, 0, 0)

    return target_slice, target_slice, tp, fp, tn, fn


def slice_all_true_negatives() -> Tuple[
    torch.Tensor, torch.Tensor, float, float, float, float
]:
    """
    Creates a faked segmentation slice that only contains true negatives.

    Returns:
        Tuple: Predicted slice, target slice, true positives, false positives, true negatives, false negatives.
    """

    target_slice = torch.Tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    tp, fp, tn, fn = (0, 0, 9, 0)

    return target_slice, target_slice, tp, fp, tn, fn


def probabilistic_slice() -> Tuple[
    torch.Tensor, torch.Tensor, float, float, float, float
]:
    """
    Creates a faked segmentation slice that contains class probabilities instead of a sharp segmentation.

    Returns:
        Tuple: Predicted slice, target slice, true positives, false positives, true negatives, false negatives.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [[
            [0.1, 0.1, 0.],
            [0.9, 0.9, 0.4],
            [0.7, 0.8, 0.2]
        ]])

    target_slice = torch.Tensor(
        [[
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 0]
        ]])
    # fmt: on

    tp, fp, tn, fn = (4, 0, 4, 1)

    return prediction_slice, target_slice, tp, fp, tn, fn
