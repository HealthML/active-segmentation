""" Module providing utilities for unit testing. """

from typing import Dict, List, Literal, Tuple

import numpy as np
import torch

# pylint: disable=unused-argument, too-many-lines


def expected_dice_score(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    probability_positive: float,
    probability_negative: float,
    epsilon: float,
) -> float:
    """
    Computes the expected Dice similarity coefficient for a single slice and a single class.

    Args:
        tp (int): Number of true positives.
        fp (int): Number of false positives.
        tn (int): Number of true negatives.
        fn (int): Number of false negatives.
        probability_positive (float): Probability used in the fake slices for positive predictions.
        probability_negative (float): Probability used in the fake slices for negative predictions.
        epsilon (float): Smoothing term used to avoid divisions by zero.

    Returns:
        float: Expected Dice similarity coefficient.
    """

    intersection = tp * probability_positive + fn * probability_negative
    denominator = (
        (tp + fp) * probability_positive + (tn + fn) * probability_negative + tp + fn
    )

    return (2.0 * intersection + epsilon) / (denominator + epsilon)


def expected_sensitivity(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    probability_positive: float,
    probability_negative: float,
    epsilon: float,
) -> float:
    """
    Computes the expected sensitivity for a single slice and a single class.

    Args:
        tp (int): Number of true positives.
        fp (int): Number of false positives.
        tn (int): Number of true negatives.
        fn (int): Number of false negatives.
        probability_positive (float): Probability used in the fake slices for positive predictions.
        probability_negative (float): Probability used in the fake slices for negative predictions.
        epsilon (float): Smoothing term used to avoid divisions by zero.

    Returns:
        float: Expected sensitivity.
    """

    return (tp + epsilon) / (tp + fn + epsilon)


def expected_specificity(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    probability_positive: float,
    probability_negative: float,
    epsilon: float,
) -> float:
    """
    Computes the expected specificity for a single slice and a single class.

    Args:
        tp (int): Number of true positives.
        fp (int): Number of false positives.
        tn (int): Number of true negatives.
        fn (int): Number of false negatives.
        probability_positive (float): Probability used in the fake slices for positive predictions.
        probability_negative (float): Probability used in the fake slices for negative predictions.
        epsilon (float): Smoothing term used to avoid divisions by zero.

    Returns:
        float: Expected specificity.
    """

    return (tn + epsilon) / (tn + fp + epsilon)


def expected_dice_loss(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    probability_positive: float,
    probability_negative: float,
    epsilon: float,
) -> float:
    """
    Computes the expected Dice loss for a single slice and a single class.

    Args:
        tp (int): Number of true positives.
        fp (int): Number of false positives.
        tn (int): Number of true negatives.
        fn (int): Number of false negatives.
        probability_positive (float): Probability used in the fake slices for positive predictions.
        probability_negative (float): Probability used in the fake slices for negative predictions.
        epsilon (float): Smoothing term used to avoid divisions by zero.

    Returns:
        float: Expected Dice loss.
    """

    return 1.0 - expected_dice_score(
        tp, fp, tn, fn, probability_positive, probability_negative, epsilon
    )


def expected_generalized_dice_loss(
    cardinalities: Dict[str, int],
    probability_positive: float,
    probability_negative: float,
    epsilon: float,
    include_background: bool = True,
    weight_type: Literal["square", "simple", "uniform"] = "uniform",
) -> float:
    """
    Computes the expected Generalized Dice loss for a single slice.

    Args:
        cardinalities (Dict[int, Dict[str, int]]): A two-level dictionary containing true positives, false positives,
            true negatives, false negatives for all classes (on the first level, the class indices are used as
            dictionary keys, on the second level the keys are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
        probability_positive (float): Probability used in the fake slices for positive predictions.
        probability_negative (float): Probability used in the fake slices for negative predictions.
        epsilon (float): Smoothing term used to avoid divisions by zero.
        include_background (bool, optional): `include_background` parameter of the loss.
        weight_type (string, optional): `weight_type` parameter of the loss.

    Returns:
        float: Expected Generalized Dice loss.
    """

    numerator = 0
    denominator = 0

    class_weights = np.zeros(len(cardinalities.keys()))

    for class_id in cardinalities.keys():
        tp = cardinalities[class_id]["tp"]
        fn = cardinalities[class_id]["fn"]
        if weight_type == "uniform":
            class_weights[class_id] = 1.0
        elif weight_type == "simple":
            class_weights[class_id] = 1.0 / (tp + fn)
        elif weight_type == "square":
            class_weights[class_id] = 1.0 / ((tp + fn) ** 2)

    for class_id in cardinalities.keys():
        if include_background or class_id != 0:
            tp = cardinalities[class_id]["tp"]
            fp = cardinalities[class_id]["fp"]
            tn = cardinalities[class_id]["tn"]
            fn = cardinalities[class_id]["fn"]

            numerator += class_weights[class_id] * (
                tp * probability_positive + fn * probability_negative
            )
            denominator += class_weights[class_id] * (
                (tp + fp) * probability_positive
                + (tn + fn) * probability_negative
                + tp
                + fn
            )

    return 1 - (2 * numerator + epsilon) / (denominator + epsilon)


def expected_false_positive_loss(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    probability_positive: float,
    probability_negative: float,
    epsilon: float,
) -> float:
    """
    Computes the expected false positive loss for a single slice and a single class.

    Args:
        tp (int): Number of true positives.
        fp (int): Number of false positives.
        tn (int): Number of true negatives.
        fn (int): Number of false negatives.
        probability_positive (float): Probability used in the fake slices for positive predictions.
        probability_negative (float): Probability used in the fake slices for negative predictions.
        epsilon (float): Smoothing term used to avoid divisions by zero.

    Returns:
        float: Expected false positive loss.
    """

    false_positives = fp * probability_positive + tn * probability_negative
    positives = false_positives + tp * probability_positive + fn * probability_negative

    return false_positives / (positives + epsilon)


def expected_cross_entropy_loss(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    probability_positive: float,
    probability_negative: float,
    *args,
) -> float:
    """
    Computes the expected cross-entropy loss for a single slice and a single class.

    Args:
        tp (int): Number of true positives.
        fp (int): Number of false positives.
        tn (int): Number of true negatives.
        fn (int): Number of false negatives.
        probability_positive (float): Probability used in the fake slices for positive predictions.
        probability_negative (float): Probability used in the fake slices for negative predictions.

    Returns:
        float: Expected cross-entropy loss.
    """

    return -1 * (
        tp * np.log(probability_positive)
        + fn * np.log(probability_negative)
        + tn * np.log(1 - probability_negative)
        + fp * np.log(1 - probability_positive)
    )


def expected_metrics(
    metric: Literal["dice_score", "dice_loss", "fp_loss", "cross_entropy_loss"],
    cardinalities: Dict[int, Dict[str, int]],
    probability_positive: float,
    probability_negative: float,
    epsilon: float,
) -> np.ndarray:
    """
    Computes expected class-wise metric values for a single slice.

    Args:
        metric (string): Name of the metric to be computed. Must be either `"dice_score"`, `"dice_loss"`, `"fp_loss"`
            or `"cross_entropy_loss"`.
        cardinalities (Dict[int, Dict[str, int]]): A two-level dictionary containing true positives, false positives,
            true negatives, false negatives for all classes (on the first level, the class indices are used as
            dictionary keys, on the second level the keys are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
        probability_positive (float): Probability used in the fake slices for positive predictions.
        probability_negative (float): Probability used in the fake slices for negative predictions.
        epsilon (float): Smoothing term used to avoid divisions by zero.
    Returns:
        numpy.ndarray: List of expected class-wise metric values.
    """

    if metric == "generalized_dice_loss":
        return np.array(
            [
                expected_generalized_dice_loss(
                    cardinalities, probability_positive, probability_negative, epsilon
                )
            ]
        )

    if metric == "dice_score":
        metric_function = expected_dice_score
    elif metric == "dice_loss":
        metric_function = expected_dice_loss
    elif metric == "fp_loss":
        metric_function = expected_false_positive_loss
    elif metric == "cross_entropy_loss":
        metric_function = expected_cross_entropy_loss
    elif metric == "sensitivity":
        metric_function = expected_sensitivity
    elif metric == "specificity":
        metric_function = expected_specificity
    else:
        raise ValueError(f"Invalid metric name: {metric}")

    _expected_metrics = []
    for class_id in cardinalities.keys():
        tp = cardinalities[class_id]["tp"]
        fp = cardinalities[class_id]["fp"]
        tn = cardinalities[class_id]["tn"]
        fn = cardinalities[class_id]["fn"]

        _expected_metrics.append(
            metric_function(
                tp, fp, tn, fn, probability_positive, probability_negative, epsilon
            )
        )
    return np.array(_expected_metrics)


def standard_slice_single_label_1(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains both true and false predictions.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.
            If set to `True`, the prediction slice is label encoded. Otherwise, the prediction slice contains one
            channel per class.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [0, 0, 0],
            [1, 2, 0],
            [1, 2, 2]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [0.8, 0.8, 0.8],
                [0.1, 0.1, 0.8],
                [0.1, 0.1, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.8, 0.8]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 0]
        ])
    # fmt: on

    cardinalities = {
        0: {"tp": 3, "fp": 1, "tn": 4, "fn": 1},
        1: {"tp": 1, "fp": 1, "tn": 5, "fn": 2},
        2: {"tp": 1, "fp": 2, "tn": 5, "fn": 1},
    }

    probability_positive = 1.0 if sharp_predictions else 0.8
    probability_negative = 0.0 if sharp_predictions else 0.1

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        probability_positive,
        probability_negative,
    )


def standard_slice_single_label_2(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains both true and false predictions.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.
            If set to `True`, the prediction slice is label encoded. Otherwise, the prediction slice contains one
            channel per class.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [0, 0, 0],
            [2, 2, 1],
            [1, 2, 2]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [0.8, 0.8, 0.8],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.8, 0.8, 0.1],
                [0.1, 0.8, 0.8]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor(
        [
            [2, 0, 1],
            [2, 1, 1],
            [2, 0, 1]
        ]
    )
    # fmt: on

    cardinalities = {
        0: {"tp": 1, "fp": 2, "tn": 5, "fn": 1},
        1: {"tp": 1, "fp": 1, "tn": 4, "fn": 3},
        2: {"tp": 1, "fp": 3, "tn": 3, "fn": 2},
    }

    probability_positive = 1.0 if sharp_predictions else 0.8
    probability_negative = 0.0 if sharp_predictions else 0.1

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        probability_positive,
        probability_negative,
    )


def slice_ignore_index_single_label(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains true, false and ignored predictions.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.
            If set to `True`, the prediction slice is label encoded. Otherwise, the prediction slice contains one
            channel per class.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [0, 0, 0],
            [1, 2, 0],
            [1, 2, 2]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [0.8, 0.8, 0.8],
                [0.1, 0.1, 0.8],
                [0.1, 0.1, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.8, 0.8]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor(
        [
            [-1, 0, 1],
            [2, 1, -1],
            [2, 0, -1]
        ]
    )
    # fmt: on

    cardinalities = {
        0: {"tp": 1, "fp": 1, "tn": 3, "fn": 1},
        1: {"tp": 0, "fp": 2, "tn": 2, "fn": 2},
        2: {"tp": 0, "fp": 2, "tn": 2, "fn": 2},
    }

    probability_positive = 1.0 if sharp_predictions else 0.8
    probability_negative = 0.0 if sharp_predictions else 0.1

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        probability_positive,
        probability_negative,
    )


def standard_slice_multi_label_1(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains both true and false predictions.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 1, 1],
                [1, 1, 0]
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1]
            ]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [0.9, 0.9, 0.9],
                [0.9, 0.9, 0.1],
                [0.1, 0.1, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.9, 0.9],
                [0.9, 0.9, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1],
                [0.9, 0.9, 0.9]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor([
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ],
        [
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ]
    ])
    # fmt: on

    cardinalities = {
        0: {"tp": 3, "fp": 2, "tn": 3, "fn": 1},
        1: {"tp": 3, "fp": 1, "tn": 3, "fn": 2},
        2: {"tp": 3, "fp": 1, "tn": 3, "fn": 2},
    }

    probability_positive = 1.0 if sharp_predictions else 0.9
    probability_negative = 0.0 if sharp_predictions else 0.1

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        probability_positive,
        probability_negative,
    )


def standard_slice_multi_label_2(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains both true and false predictions.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [
                [1, 1, 0],
                [0, 0, 0],
                [1, 0, 1]
            ],
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 0]
            ],
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 0]
            ]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [0.9, 0.9, 0.1],
                [0.1, 0.1, 0.1],
                [0.9, 0.1, 0.9]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.9, 0.9, 0.9],
                [0.9, 0.9, 0.1]
            ],
            [
                [0.9, 0.1, 0.1],
                [0.9, 0.1, 0.1],
                [0.9, 0.9, 0.1]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor([
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0]
        ],
        [
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0]
        ]
    ])
    # fmt: on

    cardinalities = {
        0: {"tp": 2, "fp": 2, "tn": 3, "fn": 2},
        1: {"tp": 4, "fp": 1, "tn": 3, "fn": 1},
        2: {"tp": 2, "fp": 2, "tn": 2, "fn": 3},
    }

    probability_positive = 1.0 if sharp_predictions else 0.9
    probability_negative = 0.0 if sharp_predictions else 0.1

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        probability_positive,
        probability_negative,
    )


def slice_ignore_index_multi_label(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked multi-label segmentation slice that contains true, false and ignored predictions.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.
            If set to `True`, the prediction slice is label encoded. Otherwise, the prediction slice contains one
            channel per class.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 1, 1],
                [1, 1, 0]
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1]
            ]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [0.9, 0.9, 0.9],
                [0.9, 0.9, 0.1],
                [0.1, 0.1, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.9, 0.9],
                [0.9, 0.9, 0.1]
            ],
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1],
                [0.9, 0.9, 0.9]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor([
        [
            [-1, 1, 1],
            [-1, 0, 1],
            [0, 0, -1]
        ],
        [
            [-1, 0, 0],
            [-1, 1, 0],
            [1, 1, -1]
        ],
        [
            [-1, 1, 0],
            [-1, 1, 0],
            [1, 1, -1]
        ]
    ])
    # fmt: on

    cardinalities = {
        0: {"tp": 2, "fp": 1, "tn": 2, "fn": 1},
        1: {"tp": 3, "fp": 1, "tn": 2, "fn": 0},
        2: {"tp": 3, "fp": 0, "tn": 2, "fn": 1},
    }

    probability_positive = 1.0 if sharp_predictions else 0.9
    probability_negative = 0.0 if sharp_predictions else 0.1

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        probability_positive,
        probability_negative,
    )


def slice_all_true_single_label(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains no segmentation errors.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.
            If set to `True`, the prediction slice is label encoded. Otherwise, the prediction slice contains one
            channel per class.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [0, 0, 1],
            [2, 1, 0],
            [2, 2, 1]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0]
            ],
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor(
        [
            [0, 0, 1],
            [2, 1, 0],
            [2, 2, 1]
        ]
    )
    # fmt: on

    cardinalities = {
        0: {"tp": 3, "fp": 0, "tn": 6, "fn": 0},
        1: {"tp": 3, "fp": 0, "tn": 6, "fn": 0},
        2: {"tp": 3, "fp": 0, "tn": 6, "fn": 0},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_all_true_multi_label(
    *args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked multi-label segmentation slice that contains both true and false predictions.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0]
        ],
        [
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0]
        ]
    ])
    # fmt: on

    # fmt: off
    target_slice = torch.IntTensor([
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0]
        ],
        [
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0]
        ]
    ])
    # fmt: on

    cardinalities = {
        0: {"tp": 4, "fp": 0, "tn": 5, "fn": 0},
        1: {"tp": 5, "fp": 0, "tn": 4, "fn": 0},
        2: {"tp": 5, "fp": 0, "tn": 4, "fn": 0},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_all_false_single_label(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains only wrong predictions.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.
            If set to `True`, the prediction slice is label encoded. Otherwise, the prediction slice contains one
            channel per class.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [0, 1, 1],
            [2, 0, 1],
            [2, 2, 0]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
            [
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0]
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor(
        [
            [1, 2, 2],
            [0, 1, 2],
            [0, 0, 1]
        ]
    )
    # fmt: on

    cardinalities = {
        0: {"tp": 0, "fp": 3, "tn": 3, "fn": 3},
        1: {"tp": 0, "fp": 3, "tn": 3, "fn": 3},
        2: {"tp": 0, "fp": 3, "tn": 3, "fn": 3},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_all_false_multi_label(
    *args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked multi-label segmentation slice that contains only wrong predictions.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 1, 1]
        ],
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1]
        ]
    ])
    # fmt: on

    # fmt: off
    target_slice = torch.IntTensor([
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0]
        ],
        [
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0]
        ]
    ])
    # fmt: on

    cardinalities = {
        0: {"tp": 0, "fp": 5, "tn": 0, "fn": 4},
        1: {"tp": 0, "fp": 4, "tn": 0, "fn": 5},
        2: {"tp": 0, "fp": 4, "tn": 0, "fn": 5},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_no_true_positives_single_label(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains no true positives.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.
            If set to `True`, the prediction slice is label encoded. Otherwise, the prediction slice contains one
            channel per class.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [1, 1, 0],
            [0, 0, 0],
            [1, 2, 2]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0]
            ],
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor(
        [
            [2, 0, 1],
            [2, 1, 1],
            [2, 0, 1]
        ]
    )
    # fmt: on

    cardinalities = {
        0: {"tp": 0, "fp": 4, "tn": 3, "fn": 2},
        1: {"tp": 0, "fp": 3, "tn": 2, "fn": 4},
        2: {"tp": 0, "fp": 2, "tn": 4, "fn": 3},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_no_true_positives_multi_label(
    *args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked multi-label segmentation slice that contains no true positives.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 1]
        ],
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0]
        ]
    ])
    # fmt: on

    # fmt: off
    target_slice = torch.IntTensor([
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0]
        ],
        [
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0]
        ]
    ])
    # fmt: on

    cardinalities = {
        0: {"tp": 0, "fp": 2, "tn": 3, "fn": 4},
        1: {"tp": 0, "fp": 2, "tn": 2, "fn": 5},
        2: {"tp": 0, "fp": 3, "tn": 1, "fn": 5},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_no_true_negatives_single_label(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains no true negatives.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.
            If set to `True`, the prediction slice is label encoded. Otherwise, the prediction slice contains one
            channel per class.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    )
    # fmt: on

    cardinalities = {
        0: {"tp": 0, "fp": 1, "tn": 2, "fn": 6},
        1: {"tp": 2, "fp": 6, "tn": 0, "fn": 1},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_no_true_negatives_multi_label(
    *args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked multi-label segmentation slice that contains no true negatives.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1]
        ],
        [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
        ],
        [
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0]
        ]
    ])
    # fmt: on

    # fmt: off
    target_slice = torch.IntTensor([
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 1, 1]
        ],
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1]
        ]
    ])
    # fmt: on

    cardinalities = {
        0: {"tp": 3, "fp": 4, "tn": 0, "fn": 2},
        1: {"tp": 2, "fp": 5, "tn": 0, "fn": 2},
        2: {"tp": 0, "fp": 5, "tn": 0, "fn": 4},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_all_true_negatives_single_label(
    sharp_predictions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked single-label segmentation slice that contains only true negatives for one class.

    Args:
        sharp_predictions (bool): Whether the prediction slice should contain sharp predictions or class probabilities.
            If set to `True`, the prediction slice is label encoded. Otherwise, the prediction slice contains one
            channel per class.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    if sharp_predictions:
        # fmt: off
        prediction_slice = torch.IntTensor([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        # fmt: on
    else:
        # fmt: off
        prediction_slice = torch.Tensor([
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ],
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ]
        ])
        # fmt: on

    # fmt: off
    target_slice = torch.IntTensor(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    )
    # fmt: on

    cardinalities = {
        0: {"tp": 0, "fp": 0, "tn": 9, "fn": 0},
        1: {"tp": 9, "fp": 0, "tn": 0, "fn": 0},
        2: {"tp": 0, "fp": 0, "tn": 9, "fn": 0},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_all_true_negatives_multi_label(
    *args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked multi-label segmentation slice that contains only true negatives.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
    ])
    # fmt: on

    # fmt: off
    target_slice = torch.IntTensor([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
    ])
    # fmt: on

    cardinalities = {
        0: {"tp": 0, "fp": 0, "tn": 9, "fn": 0},
        1: {"tp": 0, "fp": 0, "tn": 9, "fn": 0},
        2: {"tp": 0, "fp": 0, "tn": 9, "fn": 0},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def slice_all_true_positives_multi_label(
    *args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, int]], float, float]:
    """
    Creates a faked multi-label segmentation slice that contains only true positives.

    Returns:
        Tuple[Tensor, Tensor, Dict[int, Dict[str, int]], float, float]:
            - Predicted slice
            - Target slice
            - A two-level dictionary containing true positives, false positives, true negatives, false negatives for all
             classes (on the first level, the class indices are used as dictionary keys, on the second level the keys
             are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            - Probability used in the fake slices for positive predictions.
            - Probability used in the fake slices for negative predictions.
    """

    # fmt: off
    prediction_slice = torch.IntTensor([
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    ])
    # fmt: on

    # fmt: off
    target_slice = torch.IntTensor([
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    ])
    # fmt: on

    cardinalities = {
        0: {"tp": 9, "fp": 0, "tn": 0, "fn": 0},
        1: {"tp": 9, "fp": 0, "tn": 0, "fn": 0},
        2: {"tp": 9, "fp": 0, "tn": 0, "fn": 0},
    }

    return (
        prediction_slice,
        target_slice,
        cardinalities,
        1.0,
        0.0,
    )


def _expected_hausdorff_distances(
    dists_prediction_target: List[float],
    dist_target_prediction: List[float],
    percentile,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Computes expected per-class Hausdorff distances.

    Args:
        dists_prediction_target (List[float]): List of distances between each prediction pixel / voxel and the closest
            target pixel / voxel.
        dist_target_prediction (List[float]): List of distances between each target pixel / voxel and the closest
            prediction pixel / voxel.
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
