"""Test data for testing cardinality-based metrics."""

from typing import Dict, Tuple

import torch

# pylint: disable=unused-argument,too-many-lines


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
