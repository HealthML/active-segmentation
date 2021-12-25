""" Module providing utilities for unit testing. """

from typing import Tuple

import numpy as np
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


def standard_distance_slice(
    percentile: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    Creates a faked segmentation slice that contains both true and false predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, Hausdorff distance between prediction and target, Hausdorff distance
            between target and prediction.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ])

    target_slice = torch.Tensor(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 0],
        ])
    # fmt: on

    dists_prediction_target = [0, 0, 0, 0, 0, 0, 1, np.sqrt(2)]
    dist_target_prediction = [0, 0, 0, 1, 0, 0, np.sqrt(2), 0]

    hausdorff_dist = np.percentile(
        np.hstack((dists_prediction_target, dist_target_prediction)), q=percentile * 100
    )
    hausdorff_dist_prediction_target = np.percentile(
        dists_prediction_target, q=percentile * 100
    )
    hausdorff_dist_target_prediction = np.percentile(
        dist_target_prediction, q=percentile * 100
    )

    return (
        prediction_slice,
        target_slice,
        hausdorff_dist,
        hausdorff_dist_prediction_target,
        hausdorff_dist_target_prediction,
    )


def distance_slice_all_false(
    percentile: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    Creates a faked segmentation slice that contains contains only wrong predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, symmetric Hausdorff distance, Hausdorff distance between prediction and
            target, Hausdorff distance between target and prediction.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ])

    target_slice = torch.Tensor(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
    # fmt: on

    dists_prediction_target = [
        np.sqrt(5),
        np.sqrt(2),
        np.sqrt(8),
        2,
        np.sqrt(10),
        np.sqrt(5),
    ]
    dist_target_prediction = [np.sqrt(2), 2, 2]

    hausdorff_dist = np.percentile(
        np.hstack((dists_prediction_target, dist_target_prediction)), q=percentile * 100
    )
    hausdorff_dist_prediction_target = np.percentile(
        dists_prediction_target, q=percentile * 100
    )
    hausdorff_dist_target_prediction = np.percentile(
        dist_target_prediction, q=percentile * 100
    )

    return (
        prediction_slice,
        target_slice,
        hausdorff_dist,
        hausdorff_dist_prediction_target,
        hausdorff_dist_target_prediction,
    )


def distance_slice_subset(
    percentile: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    Creates a faked segmentation slice where the prediction is a subset of the target.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice,symmetric Hausdorff distance,  Hausdorff distance between prediction and
            target, Hausdorff distance between target and prediction.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])

    target_slice = torch.Tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
        ])
    # fmt: on

    dists_prediction_target = [0, 0, 0, 0]
    dist_target_prediction = [0, 0, 0, 0, 1, 1]

    hausdorff_dist = np.percentile(
        np.hstack((dists_prediction_target, dist_target_prediction)), q=percentile * 100
    )
    hausdorff_dist_prediction_target = np.percentile(
        dists_prediction_target, q=percentile * 100
    )
    hausdorff_dist_target_prediction = np.percentile(
        dist_target_prediction, q=percentile * 100
    )

    return (
        prediction_slice,
        target_slice,
        hausdorff_dist,
        hausdorff_dist_prediction_target,
        hausdorff_dist_target_prediction,
    )


def distance_slices_3d(
    percentile: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    Creates a faked segmentation slice that contains both true and false predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, symmetric Hausdorff distance, Hausdorff distance between prediction and
            target, Hausdorff distance between target and prediction.
    """

    # fmt: off
    prediction_slice = torch.Tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],

            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        ])

    target_slice = torch.Tensor(
        [
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 1, 0],
            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
        ])
    # fmt: on

    dists_prediction_target = [np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
    dist_target_prediction = [
        1,
        np.sqrt(2),
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        np.sqrt(2),
        0,
        1,
        0,
        0,
        0,
        1,
    ]

    hausdorff_dist = np.percentile(
        np.hstack((dists_prediction_target, dist_target_prediction)), q=percentile * 100
    )
    hausdorff_dist_prediction_target = np.percentile(
        dists_prediction_target, q=percentile * 100
    )
    hausdorff_dist_target_prediction = np.percentile(
        dist_target_prediction, q=percentile * 100
    )

    return (
        prediction_slice,
        target_slice,
        hausdorff_dist,
        hausdorff_dist_prediction_target,
        hausdorff_dist_target_prediction,
    )
