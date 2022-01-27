"""Test data for testing distance metrics."""

from typing import Dict, List, Tuple

import numpy as np
import torch

# pylint: disable=too-many-lines


def _expected_hausdorff_distances(
    dists_prediction_target: Dict[int, List[float]],
    dist_target_prediction: Dict[int, List[float]],
    percentile,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Computes expected per-class Hausdorff distances.

    Args:
        dists_prediction_target (Dict[int, List[float]]): List of distances between each prediction pixel / voxel and
            the closest target pixel / voxel.
        dist_target_prediction (Dict[int, List[float]]): List of distances between each target pixel / voxel and the
            closest prediction pixel / voxel.
        percentile (float): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: dictionary with per-class symmetric Hausdorff distances, dictionary with per-class Hausdorff distances
            between prediction and target, dictionary with per-class Hausdorff distances between target and prediction.
    """

    hausdorff_dist = {}
    hausdorff_dist_prediction_target = {}
    hausdorff_dist_target_prediction = {}

    # pylint: disable=consider-using-dict-items
    for class_id in dists_prediction_target:
        hausdorff_dist[class_id] = np.percentile(
            np.hstack(
                (dists_prediction_target[class_id], dist_target_prediction[class_id])
            ),
            q=percentile * 100,
        )
        hausdorff_dist_prediction_target[class_id] = np.percentile(
            dists_prediction_target[class_id], q=percentile * 100
        )
        hausdorff_dist_target_prediction[class_id] = np.percentile(
            dist_target_prediction[class_id], q=percentile * 100
        )

    return (
        hausdorff_dist,
        hausdorff_dist_prediction_target,
        hausdorff_dist_target_prediction,
    )


def standard_distance_slice_single_label(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked single-label segmentation slice that contains both true and false predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [2, 0, 0, 0],
        [2, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
    ])

    target_slice = torch.IntTensor([
        [0, 0, 0, 0],
        [2, 1, 1, 1],
        [2, 2, 0, 1],
        [1, 0, 0, 0],
    ])
    # fmt: on

    dists_prediction_target = {
        0: [0, 0, 0, 1, 1, 0, 0],
        1: [0, 0, 1, 1, 0, 1, np.sqrt(2)],
        2: [1, 0],
    }
    dist_target_prediction = {
        0: [1, 0, 0, 0, 0, np.sqrt(2), 1, 1],
        1: [0, 0, 1, np.sqrt(2), 0],
        2: [1, 0, np.sqrt(2)],
    }

    maximum_distance = 18

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def standard_distance_slice_multi_label(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked multi-label segmentation slice that contains both true and false predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ],
        [
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    ])

    target_slice = torch.IntTensor([
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        [
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
        ]
    ])
    # fmt: on

    dists_prediction_target = {
        0: [0, 0, 0, 0, 1, 0, 1, np.sqrt(2)],
        1: [1, 0, 0, 0, 0, 1],
        2: [np.sqrt(2), 1, 0, 0, 0, 1],
    }
    dist_target_prediction = {
        0: [0, 1, 1, 0, 0, 0, 0],
        1: [0, 0, 0, 0, 1, 1, np.sqrt(2)],
        2: [np.sqrt(2), 0, 0, 1, 1, 1, 2, 1, 0],
    }

    maximum_distance = 18

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_all_false_single_label(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked single-label segmentation slice that contains contains only wrong predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [0, 1, 0, 2],
        [0, 0, 0, 1],
        [1, 2, 2, 2],
        [1, 1, 1, 2],
    ])

    target_slice = torch.IntTensor([
        [1, 2, 2, 0],
        [2, 1, 1, 0],
        [0, 1, 1, 0],
        [2, 0, 0, 0],
    ])
    # fmt: on

    dists_prediction_target = {
        0: [2, 1, 1, np.sqrt(2), 1],
        1: [1, 1, 1, np.sqrt(2), 1, 1],
        2: [1, np.sqrt(2), 2, np.sqrt(5), 3],
    }
    dist_target_prediction = {
        0: [1, 1, 1, np.sqrt(2), 2, 2, np.sqrt(5)],
        1: [1, 1, 1, 1, 1],
        2: [2, 1, np.sqrt(2), np.sqrt(2)],
    }

    maximum_distance = 18

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_all_false_multi_label(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked multi-label segmentation slice that contains contains only wrong predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        [
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
    ])

    target_slice = torch.IntTensor([
        [
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    ])
    # fmt: on

    dists_prediction_target = {
        0: [np.sqrt(18)],
        1: [3, 3, 3, 2, 3],
        2: [np.sqrt(1), 1, 1, np.sqrt(2), 1, 1, 1, 1, np.sqrt(2), 1, 1, np.sqrt(2)],
    }
    dist_target_prediction = {
        0: [np.sqrt(18)],
        1: [np.sqrt(8), np.sqrt(5), 2, np.sqrt(5)],
        2: [1, 1, 1, 1],
    }

    maximum_distance = 18

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_subset_single_label(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked single-label segmentation slice where the prediction for one class is a subset of the target.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [2, 0, 2, 0],
        [0, 1, 0, 0],
        [2, 1, 1, 2],
        [1, 1, 1, 2],
    ])

    target_slice = torch.IntTensor([
        [1, 2, 2, 0],
        [2, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
    ])
    # fmt: on

    dists_prediction_target = {
        0: [2, 0, 1, 1, 0],
        1: [0, 0, 0, 0, 0, 0],
        2: [1, 0, 1, np.sqrt(5), np.sqrt(10)],
    }
    dist_target_prediction = {
        0: [0, 0, 1, 1],
        1: [np.sqrt(2), 0, 1, 0, 0, 0, 0, 0, 1],
        2: [1, 1, 0],
    }

    maximum_distance = 18

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_subset_multi_label(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked multi-label segmentation slice where the prediction is a subset of the target.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    ])

    target_slice = torch.IntTensor([
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
        ],
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    ])
    # fmt: on

    dists_prediction_target = {0: [0, 0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0, 0]}
    dist_target_prediction = {
        0: [0, 0, 0, 0, 1, 1],
        1: [2, 1, 0, 0, 2, 1, 0, 1],
        2: [
            np.sqrt(2),
            1,
            1,
            np.sqrt(2),
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            np.sqrt(2),
            1,
            1,
            np.sqrt(2),
        ],
    }

    maximum_distance = 18

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_3d_single_label(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked three-dimensional single-label segmentation slice.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [0, 1, 1, 2],
            [1, 1, 1, 0],
        ],
        [
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [2, 1, 1, 2],
            [1, 0, 0, 2],
        ],
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 2],
            [2, 2, 2, 2],
        ]
    ])

    target_slice = torch.IntTensor([
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [2, 2, 1, 0],
            [2, 2, 1, 1],
        ],
        [
            [1, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1],
            [2, 1, 1, 1],
            [2, 2, 2, 1],
            [2, 0, 0, 0],
        ]
    ])
    # fmt: on

    dists_prediction_target = {
        0: [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, np.sqrt(3), np.sqrt(2), 1, np.sqrt(2), 1, 1],
        1: [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, np.sqrt(2), 1],
        2: [
            np.sqrt(2),
            np.sqrt(3),
            1,
            np.sqrt(2),
            np.sqrt(2),
            1,
            np.sqrt(2),
            0,
            1,
            1,
            1,
            np.sqrt(2),
            1,
            0,
            1,
            1,
            np.sqrt(2),
        ],
    }
    dist_target_prediction = {
        0: [0, 0, 1, 2, 0, 2, 1, 1, 0, 1, 1, np.sqrt(2), 1, 1, np.sqrt(2)],
        1: [
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            np.sqrt(2),
            0,
            1,
            1,
            np.sqrt(2),
            0,
            1,
            np.sqrt(2),
            np.sqrt(5),
            0,
            1,
            np.sqrt(3),
            np.sqrt(2),
        ],
        2: [1, np.sqrt(2), np.sqrt(2), np.sqrt(3), 1, 0, 1, 1, np.sqrt(2), 1, 1, 1, 0],
    }

    maximum_distance = 22

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_3d_multi_label(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked three-dimensional multi-label segmentation slice.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 0, 0],
            ],
        ],
        [
            [
                [0, 0, 1, 1],
                [0, 0, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
            ],
            [
                [0, 0, 1, 1],
                [1, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            [
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        [
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        ]
    ])

    target_slice = torch.IntTensor([
        [
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 1, 0, 0],
            ],
        ],
        [
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 0, 0],
            ],
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
            ],
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        [
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ]
        ]
    ])
    # fmt: on

    dists_prediction_target = {
        0: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        2: [
            np.sqrt(2),
            np.sqrt(3),
            1,
            np.sqrt(2),
            1,
            1,
            np.sqrt(2),
            0,
            0,
            1,
            1,
            1,
            0,
            1,
        ],
    }
    dist_target_prediction = {
        0: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        1: [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        2: [
            np.sqrt(3),
            np.sqrt(2),
            np.sqrt(6),
            np.sqrt(5),
            1,
            0,
            0,
            np.sqrt(2),
            1,
            1,
            1,
            0,
            np.sqrt(2),
            1,
        ],
    }

    maximum_distance = 22

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_ignore_index_single_label_1(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked single-label segmentation slice that contains true, false and ignored predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        [
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [0, 1, 1, 2],
            [1, 1, 1, 0],
        ],
        [
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [2, 1, 1, 2],
            [1, 0, 0, 2],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    ])

    target_slice = torch.IntTensor([
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        [
            [0,   0,  0, -1],
            [0,   1,  1, -1],
            [2,   2,  1, -1],
            [-1, -1, -1, -1],
        ],
        [
            [1,   1,  1, -1],
            [0,   2,  2, -1],
            [0,   2,  2, -1],
            [-1, -1, -1, -1],
        ],
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ]
    ])
    # fmt: on

    dists_prediction_target = {
        0: [0, 0, 0, 1, 1, 1, 0],
        1: [0, 1, 0, 1, np.sqrt(2), 1],
        2: [np.sqrt(2), 1, 1, 0, 1],
    }
    dist_target_prediction = {
        0: [0, 0, 1, 0, 1, 0],
        1: [0, 1, 0, np.sqrt(2), 1, np.sqrt(2)],
        2: [1, np.sqrt(2), 1, 0, 1, 1],
    }

    maximum_distance = 9

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_ignore_index_single_label_2(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked single-label segmentation slice that contains true, false and ignored predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [2, 2, 1, 0, 0],
            [2, 1, 1, 0, 0],
            [2, 1, 1, 1, 1],
            [0, 0, 1, 0, 1],
            [1, 2, 1, 2, 1],
            [2, 2, 1, 2, 1],
        ],
        [
            [1, 1, 2, 1, 1],
            [2, 2, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 2, 0, 1, 2],
            [1, 2, 0, 1, 2],
            [1, 2, 0, 1, 2],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ])

    target_slice = torch.IntTensor([
        [
            [2,   2,  1, -1, -1],
            [0,   2,  0, -1, -1],
            [1,   2,  1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        [
            [0,   0,  2, -1, -1],
            [0,   1,  2, -1, -1],
            [2,   1,  1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        [
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ]
    ])
    # fmt: on

    dists_prediction_target = {
        0: [1, np.sqrt(2), np.sqrt(2)],
        1: [0, 1, 1, 1, 0, np.sqrt(2), 1, 1],
        2: [0, 0, 1, 1, 0, 1, 1],
    }
    dist_target_prediction = {
        0: [np.sqrt(2), np.sqrt(2), 2, 2, 1],
        1: [0, 1, 0, 1, 1, 1],
        2: [0, 0, 1, 1, 0, 1, 1],
    }

    maximum_distance = 9

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_ignore_index_multi_label_1(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked multi-label segmentation slice that contains true, false and ignored predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            [
                [0, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [0, 1, 0, 0],
            ],
            [
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        ],
        [
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            [
                [0, 0, 1, 1],
                [1, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            [
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        ],
        [
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            [
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        ]
    ])

    target_slice = torch.IntTensor([
        [
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            [
                [0,   0,  0, -1],
                [1,   1,  0, -1],
                [1,   1,  0, -1],
                [-1, -1, -1, -1],
            ],
            [
                [0,   0,  0, -1],
                [0,   1,  1, -1],
                [0,   1,  1, -1],
                [-1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
        ],
        [
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            [
                [1,   1,  1, -1],
                [1,   1,  1, -1],
                [0,   1,  1, -1],
                [-1, -1, -1, -1],
            ],
            [
                [1,   0,  0, -1],
                [1,   1,  0, -1],
                [0,   1,  0, -1],
                [-1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
        ],
        [
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            [
                [1,   0,  0, -1],
                [0,   0,  0, -1],
                [0,   0,  0, -1],
                [-1, -1, -1, -1],
            ],
            [
                [0,   0,  0, -1],
                [0,   0,  0, -1],
                [0,   0,  0, -1],
                [-1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
        ]
    ])
    # fmt: on

    dists_prediction_target = {
        0: [0, 1, 0, 1, np.sqrt(2), 1, 1, 0, 0],
        1: [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        2: [1, 2, np.sqrt(2), np.sqrt(5), np.sqrt(5), np.sqrt(6), np.sqrt(9)],
    }
    dist_target_prediction = {
        0: [0, 1, 0, 1, 0, 1, 0, 1],
        1: [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        2: [1],
    }

    maximum_distance = 9

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def distance_slice_ignore_index_multi_label_2(
    percentile: float = 0.95,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Creates a faked multi-label segmentation slice that contains true, false and ignored predictions.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Predicted slice, target slice, dictionary with per-class symmetric Hausdorff distances, dictionary with
            per-class Hausdorff distances between prediction and target, dictionary with per-class Hausdorff distances
            between target and prediction, and maximum possible Hausdorff distance for the returned slice size.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [
                [1, 1, 0, 0, 1],
                [0, 1, 0, 0, 1],
                [1, 0, 1, 1, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 0, 1],
            ],
            [
                [0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 1],
                [0, 1, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        [
            [
                [0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
            ],
            [
                [0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1],
                [0, 1, 0, 1, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        [
            [
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 1, 0, 1],
                [0, 1, 1, 0, 1],
            ],
            [
                [1, 1, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    ])

    target_slice = torch.IntTensor([
        [
            [
                [0,   1,  1, -1, -1],
                [0,   1,  1, -1, -1],
                [0,   0,  0, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
            [
                [0,   0,  0, -1, -1],
                [1,   1,  1, -1, -1],
                [0,   0,  0, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
        ],
        [
            [
                [1,   1,  0, -1, -1],
                [1,   0,  0, -1, -1],
                [0,   0,  1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
            [
                [1,   0,  0, -1, -1],
                [1,   1,  1, -1, -1],
                [0,   0,  0, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
        ],
        [
            [
                [0,   1,  0, -1, -1],
                [0,   1,  1, -1, -1],
                [0,   1,  1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
            [
                [0,   1,  1, -1, -1],
                [0,   0,  0, -1, -1],
                [0,   0,  0, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
        ]
    ])
    # fmt: on

    dists_prediction_target = {
        0: [1, 0, 0, np.sqrt(2), 1, 1, 0, 0, 1],
        1: [0, 1, 1, 1, 0, 0, 1],
        2: [0, 0, 0, 1, 0],
    }
    dist_target_prediction = {
        0: [0, 1, 0, 1, 0, 0, 1],
        1: [1, 1, 0, 1, 1, 0, 0, 1],
        2: [0, 0, 1, 0, 1, 0, 1],
    }

    maximum_distance = 9

    return (
        prediction_slice,
        target_slice,
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def expected_distances_slice_ignore_index_single_label_1_2(
    percentile: float = 0.95,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], float]:
    """
    Computes expected per-class Hausdorff distances for the case that the slices returned by
    `distance_slice_ignore_index_single_label_1` and `distance_slice_ignore_index_single_label_2` are stacked.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Dictionary with per-class symmetric Hausdorff distances, dictionary with per-class Hausdorff distances
            between prediction and target, dictionary with per-class Hausdorff distances between target and prediction,
            and maximum possible Hausdorff distance for the returned slice size.
    """

    dists_prediction_target = {
        0: [0, 0, 0, 1, 1, 1, 0, 1, np.sqrt(2), np.sqrt(2)],
        1: [0, 1, 0, 1, np.sqrt(2), 1, 0, 1, 1, 1, 0, np.sqrt(2), 1, 1],
        2: [np.sqrt(2), 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
    }
    dist_target_prediction = {
        0: [0, 0, 1, 0, 1, 0, 1, np.sqrt(2), 2, 2, 1],
        1: [0, 1, 0, np.sqrt(2), 1, 1, 0, 1, 0, 1, 1, 1],
        2: [1, np.sqrt(2), 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    }

    maximum_distance = 17

    return (
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )


def expected_distances_slice_ignore_index_multi_label_1_2(
    percentile: float = 0.95,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], float]:
    """
    Computes expected per-class Hausdorff distances for the case that the slices returned by
    `distance_slice_ignore_index_multi_label_1` and `distance_slice_ignore_index_multi_label_2` are stacked.

    Args:
        percentile (float, optional): Percentile for which the expected Hausdorff distances are to be calculated.

    Returns:
        Tuple: Dictionary with per-class symmetric Hausdorff distances, dictionary with per-class Hausdorff distances
            between prediction and target, dictionary with per-class Hausdorff distances between target and prediction,
            and maximum possible Hausdorff distance for the returned slice size.
    """

    dists_prediction_target = {
        0: [0, 1, 0, 1, np.sqrt(2), 1, 1, 0, 0, 1, 0, 0, np.sqrt(2), 1, 1, 0, 0, 1],
        1: [
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
        ],
        2: [1, 2, np.sqrt(2), 2, np.sqrt(2), 1, 1, 0, 0, 0, 1, 0],
    }
    dist_target_prediction = {
        0: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        1: [
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
        ],
        2: [1, 0, 0, 1, 0, 1, 0, 1],
    }

    maximum_distance = 17

    return (
        *_expected_hausdorff_distances(
            dists_prediction_target, dist_target_prediction, percentile
        ),
        maximum_distance,
    )
