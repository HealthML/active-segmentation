"""Utilities for computing expected metric / loss values"""

from typing import Dict, Literal

import numpy as np

# pylint: disable=unused-argument


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
