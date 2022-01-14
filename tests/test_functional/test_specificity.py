"""Tests for the specificity metric"""

from typing import Dict, Literal

import unittest

import torch

from functional import (
    Specificity,
    specificity,
)
import tests.utils


class TestSpecificity(unittest.TestCase):
    """
    Test cases for specificity.
    """

    # pylint: disable = too-many-locals, too-many-nested-blocks

    @staticmethod
    def _expected_specificity(
        cardinalities: Dict[int, Dict[str, int]],
        probability_positive: float,
        probability_negative: float,
        epsilon: float,
        include_background: bool,
        reduction: Literal["none", "mean", "min", "max"],
    ) -> torch.Tensor:
        """
        Computes expected specificity for a single slice.

        Args:
            cardinalities (Dict[int, Dict[str, int]]): A two-level dictionary containing true positives, false
                positives, true negatives, false negatives for all classes (on the first level, the class indices are
                used as dictionary keys, on the second level the keys are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            probability_positive (float): Probability used in the fake slices for positive predictions.
            probability_negative (float): Probability used in the fake slices for negative predictions.
            epsilon (float): Smoothing term used to avoid divisions by zero.
            include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from
                the calculation of the specificity (default = `True`).
            reduction (string): A method to reduce specificity values of multiple classes.
                - ``"none"``: no reduction will be applied (default)
                - ``"mean"``: takes the mean
                - ``"min"``: takes the minimum
                - ``"max"``: takes the maximum

        Returns:
            torch.Tensor: Expected specificity values.
        """

        expected_specificity = tests.utils.expected_metrics(
            "specificity",
            cardinalities,
            probability_positive,
            probability_negative,
            epsilon=epsilon,
        )

        if not include_background:
            expected_specificity = expected_specificity[1:]

        if reduction == "min":
            expected_specificity = expected_specificity.min()
        if reduction == "max":
            expected_specificity = expected_specificity.max()
        if reduction == "mean":
            expected_specificity = expected_specificity.mean()

        return torch.as_tensor(expected_specificity).float()

    def test_standard_case(self):
        """
        Tests that the specificity is computed correctly when there are both true and false predictions.
        """

        for test_slice, convert_to_one_hot in [
            (
                tests.utils.standard_slice_single_label_1,
                True,
            ),
            (
                tests.utils.standard_slice_multi_label_1,
                False,
            ),
        ]:

            (
                prediction,
                target,
                cardinalities,
                probability_positive,
                probability_negative,
            ) = test_slice(True)

            for reduction in ["none", "mean", "min", "max"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        expected_specificity = self._expected_specificity(
                            cardinalities,
                            probability_positive,
                            probability_negative,
                            epsilon,
                            include_background,
                            reduction,
                        )

                        task_type = (
                            "single-label"
                            if convert_to_one_hot is True
                            else "multi-label"
                        )
                        test_case_description = (
                            f"reduction is '{reduction}', include_background is "
                            f"{include_background} and epsilon is {epsilon}"
                        )

                        specificity_from_function = specificity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        torch.testing.assert_allclose(
                            specificity_from_function,
                            expected_specificity,
                            msg=f"Functional implementation correctly computes specificity for {task_type} tasks when "
                            f"there are TP, FP and FN and {test_case_description}.",
                        )

                        specificity_module = Specificity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        specificity_from_module = specificity_module(prediction, target)

                        torch.testing.assert_allclose(
                            specificity_from_module,
                            expected_specificity,
                            msg=f"Module-based implementation correctly computes specificity for {task_type} tasks when"
                            f" there are TP, FP and FN and {test_case_description}.",
                        )

                        specificity_module.reset()
                        specificity_module.update(prediction, target)
                        specificity_from_module_compute = specificity_module.compute()

                        torch.testing.assert_allclose(
                            specificity_from_module_compute,
                            expected_specificity,
                            msg="Compute method of module-based implementation returns correct specificity for "
                            "{task_type} tasks when there are TP, FP and FN and {test_case_description}.",
                        )

    def test_all_true(self):
        """
        Tests that the specificity is computed correctly when all predictions are correct.
        """

        for test_slice, convert_to_one_hot in [
            (
                tests.utils.slice_all_true_single_label,
                True,
            ),
            (
                tests.utils.slice_all_true_multi_label,
                False,
            ),
        ]:

            (
                prediction,
                target,
                cardinalities,
                _,
                _,
            ) = test_slice(True)

            for reduction in ["none", "mean", "min", "max"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:

                        task_type = (
                            "single-label"
                            if convert_to_one_hot is True
                            else "multi-label"
                        )
                        test_case_description = (
                            f"reduction is '{reduction}', include_background is "
                            f"{include_background} and epsilon is {epsilon}"
                        )

                        specificity_from_function = specificity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        if reduction == "none":
                            expected_specificity = (
                                torch.ones(3)
                                if include_background is True
                                else torch.ones(2)
                            )
                        else:
                            expected_specificity = torch.tensor(1.0)

                        self.assertTrue(
                            torch.equal(
                                specificity_from_function, expected_specificity
                            ),
                            f"Functional implementation correctly computes specificity for {task_type} tasks when there"
                            f" are no prediction errors and {test_case_description}.",
                        )

                        specificity_module = Specificity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        specificity_from_module = specificity_module(prediction, target)
                        self.assertTrue(
                            torch.equal(specificity_from_module, expected_specificity),
                            f"Module-based implementation correctly computes specificity for {task_type} when there are"
                            f" no prediction errors {test_case_description}.",
                        )

    def test_all_false(self):
        """
        Tests that the specificity is computed correctly when all predictions are wrong.
        """

        for test_slice, convert_to_one_hot in [
            (
                tests.utils.slice_all_false_single_label,
                True,
            ),
            (
                tests.utils.slice_all_false_multi_label,
                False,
            ),
        ]:

            (
                prediction,
                target,
                cardinalities,
                probability_positive,
                probability_negative,
            ) = test_slice(True)

            for reduction in ["none", "mean", "min", "max"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:

                        task_type = (
                            "single-label"
                            if convert_to_one_hot is True
                            else "multi-label"
                        )
                        test_case_description = (
                            f"reduction is '{reduction}', include_background is "
                            f"{include_background} and epsilon is {epsilon}"
                        )

                        specificity_from_function = specificity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        if epsilon == 0 and convert_to_one_hot is False:
                            if reduction == "none":
                                expected_specificity = (
                                    torch.zeros(3)
                                    if include_background is True
                                    else torch.zeros(2)
                                )
                            else:
                                expected_specificity = torch.tensor(0.0)
                        else:
                            expected_specificity = self._expected_specificity(
                                cardinalities,
                                probability_positive,
                                probability_negative,
                                epsilon,
                                include_background,
                                reduction,
                            )

                        self.assertTrue(
                            torch.equal(
                                specificity_from_function, expected_specificity
                            ),
                            f"Functional implementation correctly computes specificity for {task_type} tasks when all "
                            f"predictions are wrong and {test_case_description}.",
                        )

                        specificity_module = Specificity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        specificity_from_module = specificity_module(prediction, target)
                        self.assertTrue(
                            torch.equal(specificity_from_module, expected_specificity),
                            f"Module-based implementation correctly computes specificity for {task_type} when all "
                            f"predictions are wrong and {test_case_description}.",
                        )

    def test_no_true_positives(self):
        """
        Tests that the specificity is computed correctly when there are no true positives.
        """

        for test_slice, convert_to_one_hot in [
            (
                tests.utils.slice_no_true_positives_single_label,
                True,
            ),
            (
                tests.utils.slice_no_true_positives_multi_label,
                False,
            ),
        ]:

            (
                prediction,
                target,
                cardinalities,
                probability_positive,
                probability_negative,
            ) = test_slice(True)

            for reduction in ["none", "mean", "min", "max"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        expected_specificity = self._expected_specificity(
                            cardinalities,
                            probability_positive,
                            probability_negative,
                            epsilon,
                            include_background,
                            reduction,
                        )

                        task_type = (
                            "single-label"
                            if convert_to_one_hot is True
                            else "multi-label"
                        )
                        test_case_description = (
                            f"reduction is '{reduction}', include_background is "
                            f"{include_background} and epsilon is {epsilon}"
                        )

                        specificity_from_function = specificity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        torch.testing.assert_allclose(
                            specificity_from_function,
                            expected_specificity,
                            msg=f"Functional implementation correctly computes specificity for {task_type} tasks when "
                            f"there are no TP and {test_case_description}.",
                        )

                        specificity_module = Specificity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        specificity_from_module = specificity_module(prediction, target)

                        torch.testing.assert_allclose(
                            specificity_from_module,
                            expected_specificity,
                            msg=f"Module-based implementation correctly computes specificity for {task_type} tasks when"
                            f" there are no TP and {test_case_description}.",
                        )

    def test_no_true_negatives(self):
        """
        Tests that the specificity is computed correctly when there are no true negatives.
        """

        for test_slice, convert_to_one_hot in [
            (
                tests.utils.slice_no_true_negatives_single_label,
                True,
            ),
            (
                tests.utils.slice_no_true_negatives_multi_label,
                False,
            ),
        ]:
            (
                prediction,
                target,
                cardinalities,
                probability_positive,
                probability_negative,
            ) = test_slice(True)

            for reduction in ["none", "mean", "min", "max"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:

                        task_type = (
                            "single-label"
                            if convert_to_one_hot is True
                            else "multi-label"
                        )
                        test_case_description = (
                            f"reduction is '{reduction}', include_background is "
                            f"{include_background} and epsilon is {epsilon}"
                        )

                        specificity_from_function = specificity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        if epsilon == 0 and convert_to_one_hot is False:
                            if reduction == "none":
                                expected_specificity = (
                                    torch.zeros(3)
                                    if include_background is True
                                    else torch.zeros(2)
                                )
                            else:
                                expected_specificity = torch.tensor(0.0)
                        else:
                            expected_specificity = self._expected_specificity(
                                cardinalities,
                                probability_positive,
                                probability_negative,
                                epsilon,
                                include_background,
                                reduction,
                            )

                        torch.testing.assert_allclose(
                            specificity_from_function,
                            expected_specificity,
                            msg=f"Functional implementation correctly computes specificity for {task_type} tasks when "
                            f"there are no TN and {test_case_description}.",
                        )

                        specificity_module = Specificity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        specificity_from_module = specificity_module(prediction, target)

                        torch.testing.assert_allclose(
                            specificity_from_module,
                            expected_specificity,
                            msg=f"Module-based implementation correctly computes specificity for {task_type} tasks when"
                            f" there are no TN and {test_case_description}.",
                        )

    def test_all_true_positives(self):
        """
        Tests that the specificity is computed correctly when there are only true positives.
        """

        for test_slice, convert_to_one_hot in [
            (
                # this slice contains only true positives for one class
                tests.utils.slice_all_true_negatives_single_label,
                True,
            ),
            (
                tests.utils.slice_all_true_positives_multi_label,
                False,
            ),
        ]:

            (
                prediction,
                target,
                cardinalities,
                _,
                _,
            ) = test_slice(True)

            for reduction in ["none", "mean", "min", "max"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:

                        task_type = (
                            "single-label"
                            if convert_to_one_hot is True
                            else "multi-label"
                        )
                        test_case_description = (
                            f"reduction is '{reduction}', include_background is "
                            f"{include_background} and epsilon is {epsilon}"
                        )

                        specificity_from_function = specificity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        print("specificity_from_function", specificity_from_function)

                        if epsilon == 0:
                            if convert_to_one_hot is False or reduction != "none":
                                self.assertTrue(
                                    torch.all(torch.isnan(specificity_from_function)),
                                    f"Functional implementation correctly computes specificity for {task_type} tasks "
                                    f"when there are only TP and {test_case_description}.",
                                )
                            else:
                                self.assertTrue(
                                    torch.isnan(specificity_from_function)[
                                        1 if include_background else 0
                                    ],
                                    f"Functional implementation correctly computes specificity for {task_type} tasks "
                                    f"when there are only TP and {test_case_description}.",
                                )
                        else:
                            self.assertTrue(
                                torch.all(specificity_from_function == 1.0),
                                f"Functional implementation correctly computes specificity for {task_type} tasks when "
                                f" there are only TP and {test_case_description}.",
                            )

                        specificity_module = Specificity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        specificity_from_module = specificity_module(prediction, target)

                        if epsilon == 0:
                            if convert_to_one_hot is False or reduction != "none":
                                self.assertTrue(
                                    torch.all(torch.isnan(specificity_from_module)),
                                    f"Module-based implementation correctly computes specificity for {task_type} tasks "
                                    f"when there are only TP and {test_case_description}.",
                                )
                            else:
                                self.assertTrue(
                                    torch.isnan(specificity_from_module)[
                                        1 if include_background else 0
                                    ],
                                    f"Module-based implementation correctly computes specificity when there are only TP"
                                    f" and {test_case_description}.",
                                )
                        else:
                            self.assertTrue(
                                torch.all(specificity_from_module == 1.0),
                                f"Module-based implementation correctly computes smoothed specificity when there are "
                                f"only TP and {test_case_description}.",
                            )

    def test_3d(self):
        """
        Tests that the specificity is computed correctly when the inputs are three-dimensional.
        """

        for test_slice_1, test_slice_2, convert_to_one_hot in [
            (
                tests.utils.standard_slice_single_label_1,
                tests.utils.standard_slice_single_label_2,
                True,
            ),
            (
                tests.utils.standard_slice_multi_label_1,
                tests.utils.standard_slice_multi_label_2,
                False,
            ),
        ]:

            (
                prediction_1,
                target_1,
                cardinalities_1,
                probability_positive_1,
                probability_negative_1,
            ) = test_slice_1(True)

            (
                prediction_2,
                target_2,
                cardinalities_2,
                probability_positive_2,
                probability_negative_2,
            ) = test_slice_2(True)

            assert probability_positive_1 == probability_positive_2
            assert probability_negative_1 == probability_negative_2

            prediction = torch.stack([prediction_1, prediction_2])
            target = torch.stack([target_1, target_2])

            if convert_to_one_hot is False:
                # ensure that the class channel is the first dimension
                prediction = prediction.swapaxes(1, 0)
                target = target.swapaxes(1, 0)

            cardinalities = {}
            for class_id, class_cardinalities in cardinalities_1.items():
                # sum cardinalities for both stacked slices
                cardinalities[class_id] = {
                    key: cardinalities_1[class_id][key] + cardinalities_2[class_id][key]
                    for key in class_cardinalities
                }

            for reduction in ["none", "mean", "min", "max"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        expected_specificity = self._expected_specificity(
                            cardinalities,
                            probability_positive_1,
                            probability_negative_1,
                            epsilon,
                            include_background,
                            reduction,
                        )

                        task_type = (
                            "single-label"
                            if convert_to_one_hot is True
                            else "multi-label"
                        )
                        test_case_description = (
                            f"reduction is '{reduction}', include_background is "
                            f"{include_background} and epsilon is {epsilon}"
                        )

                        specificity_from_function = specificity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        torch.testing.assert_allclose(
                            specificity_from_function,
                            expected_specificity,
                            msg=f"Functional implementation correctly computes specificity for {task_type} tasks when "
                            f"the inputs are three-dimensional and {test_case_description}.",
                        )

                        specificity_module = Specificity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        specificity_from_module = specificity_module(prediction, target)

                        torch.testing.assert_allclose(
                            specificity_from_module,
                            expected_specificity,
                            msg=f"Module-based implementation correctly computes specificity for {task_type} tasks when"
                            f" the inputs are three-dimensional and {test_case_description}.",
                        )

    def test_ignore_index(self):
        """
        Tests that the specificity is computed correctly when there are pixels / voxels to be ignored.
        """

        for test_slice, convert_to_one_hot in [
            (
                tests.utils.slice_ignore_index_single_label,
                True,
            ),
            (
                tests.utils.slice_ignore_index_multi_label,
                False,
            ),
        ]:

            (
                prediction,
                target,
                cardinalities,
                probability_positive,
                probability_negative,
            ) = test_slice(True)

            for reduction in ["none", "mean", "min", "max"]:
                for include_background in [True, False]:
                    for epsilon in [0, 1]:
                        expected_specificity = self._expected_specificity(
                            cardinalities,
                            probability_positive,
                            probability_negative,
                            epsilon,
                            include_background,
                            reduction,
                        )

                        task_type = (
                            "single-label"
                            if convert_to_one_hot is True
                            else "multi-label"
                        )
                        test_case_description = (
                            f"reduction is '{reduction}', include_background is "
                            f"{include_background} and epsilon is {epsilon}"
                        )

                        specificity_from_function = specificity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            ignore_index=-1,
                            include_background=include_background,
                            epsilon=epsilon,
                            reduction=reduction,
                        )

                        torch.testing.assert_allclose(
                            specificity_from_function,
                            expected_specificity,
                            msg=f"Functional implementation correctly computes specificity for {task_type} tasks when "
                            f"there are pixels / voxels to be ignored and {test_case_description}.",
                        )

                        specificity_module = Specificity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            ignore_index=-1,
                            include_background=include_background,
                            epsilon=epsilon,
                            reduction=reduction,
                        )
                        specificity_from_module = specificity_module(prediction, target)

                        torch.testing.assert_allclose(
                            specificity_from_module,
                            expected_specificity,
                            msg=f"Module-based implementation correctly computes specificity for {task_type} tasks when"
                            f" there are pixels / voxels to be ignored and {test_case_description}.",
                        )
