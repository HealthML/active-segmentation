"""Tests for the sensitivity metric"""

from typing import Dict, Literal

import unittest

import torch

from functional import (
    Sensitivity,
    sensitivity,
)
import tests.utils
import tests.utils.test_data_cardinality_metrics as test_data


class TestSensitivity(unittest.TestCase):
    """
    Test cases for sensitivity.
    """

    # pylint: disable = too-many-locals, too-many-nested-blocks

    @staticmethod
    def _expected_sensitivity(
        cardinalities: Dict[int, Dict[str, int]],
        probability_positive: float,
        probability_negative: float,
        epsilon: float,
        include_background: bool,
        reduction: Literal["none", "mean", "min", "max"],
    ) -> torch.Tensor:
        """
        Computes expected sensitivity for a single slice.

        Args:
            cardinalities (Dict[int, Dict[str, int]]): A two-level dictionary containing true positives, false
                positives, true negatives, false negatives for all classes (on the first level, the class indices are
                used as dictionary keys, on the second level the keys are `"tp"`, `"fp"`, `"tn"`, and `"fn"`).
            probability_positive (float): Probability used in the fake slices for positive predictions.
            probability_negative (float): Probability used in the fake slices for negative predictions.
            epsilon (float): Smoothing term used to avoid divisions by zero.
            include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from
                the calculation of the sensitivity (default = `True`).
            reduction (string): A method to reduce sensitivity values of multiple classes.
                - ``"none"``: no reduction will be applied (default)
                - ``"mean"``: takes the mean
                - ``"min"``: takes the minimum
                - ``"max"``: takes the maximum

        Returns:
            torch.Tensor: Expected sensitivity values.
        """

        expected_sensitivity = tests.utils.expected_metrics(
            "sensitivity",
            cardinalities,
            probability_positive,
            probability_negative,
            epsilon=epsilon,
        )

        if not include_background:
            expected_sensitivity = expected_sensitivity[1:]

        if reduction == "min":
            expected_sensitivity = expected_sensitivity.min()
        if reduction == "max":
            expected_sensitivity = expected_sensitivity.max()
        if reduction == "mean":
            expected_sensitivity = expected_sensitivity.mean()

        return torch.as_tensor(expected_sensitivity).float()

    def test_standard_case(self):
        """
        Tests that the sensitivity is computed correctly when there are both true and false predictions.
        """

        for test_slice, convert_to_one_hot in [
            (
                test_data.standard_slice_single_label_1,
                True,
            ),
            (
                test_data.standard_slice_multi_label_1,
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
                        expected_sensitivity = self._expected_sensitivity(
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

                        sensitivity_from_function = sensitivity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        torch.testing.assert_allclose(
                            sensitivity_from_function,
                            expected_sensitivity,
                            msg=f"Functional implementation correctly computes sensitivity for {task_type} tasks when "
                            f"there are TP, FP and FN and {test_case_description}.",
                        )

                        sensitivity_module = Sensitivity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        sensitivity_from_module = sensitivity_module(prediction, target)

                        torch.testing.assert_allclose(
                            sensitivity_from_module,
                            expected_sensitivity,
                            msg=f"Module-based implementation correctly computes sensitivity for {task_type} tasks when"
                            f" there are TP, FP and FN and {test_case_description}.",
                        )

                        sensitivity_module.reset()
                        sensitivity_module.update(prediction, target)
                        sensitivity_from_module_compute = sensitivity_module.compute()

                        torch.testing.assert_allclose(
                            sensitivity_from_module_compute,
                            expected_sensitivity,
                            msg="Compute method of module-based implementation returns correct sensitivity for "
                            "{task_type} tasks when there are TP, FP and FN and {test_case_description}.",
                        )

    def test_all_true(self):
        """
        Tests that the sensitivity is computed correctly when all predictions are correct.
        """

        for test_slice, convert_to_one_hot in [
            (
                test_data.slice_all_true_single_label,
                True,
            ),
            (
                test_data.slice_all_true_multi_label,
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

                        sensitivity_from_function = sensitivity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        if reduction == "none":
                            expected_sensitivity = (
                                torch.ones(3)
                                if include_background is True
                                else torch.ones(2)
                            )
                        else:
                            expected_sensitivity = torch.tensor(1.0)

                        self.assertTrue(
                            torch.equal(
                                sensitivity_from_function, expected_sensitivity
                            ),
                            f"Functional implementation correctly computes sensitivity for {task_type} tasks when there"
                            f" are no prediction errors and {test_case_description}.",
                        )

                        sensitivity_module = Sensitivity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        sensitivity_from_module = sensitivity_module(prediction, target)
                        self.assertTrue(
                            torch.equal(sensitivity_from_module, expected_sensitivity),
                            f"Module-based implementation correctly computes sensitivity for {task_type} when there are"
                            f" no prediction errors {test_case_description}.",
                        )

    def test_all_false(self):
        """
        Tests that the sensitivity is computed correctly when all predictions are wrong.
        """

        for test_slice, convert_to_one_hot in [
            (
                test_data.slice_all_false_single_label,
                True,
            ),
            (
                test_data.slice_all_false_multi_label,
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

                        sensitivity_from_function = sensitivity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        if epsilon == 0:
                            if reduction == "none":
                                expected_sensitivity = (
                                    torch.zeros(3)
                                    if include_background is True
                                    else torch.zeros(2)
                                )
                            else:
                                expected_sensitivity = torch.tensor(0.0)
                        else:

                            expected_sensitivity = self._expected_sensitivity(
                                cardinalities,
                                probability_positive,
                                probability_negative,
                                epsilon,
                                include_background,
                                reduction,
                            )

                        self.assertTrue(
                            torch.equal(
                                sensitivity_from_function, expected_sensitivity
                            ),
                            f"Functional implementation correctly computes sensitivity for {task_type} tasks when all "
                            f"predictions are wrong and {test_case_description}.",
                        )

                        sensitivity_module = Sensitivity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        sensitivity_from_module = sensitivity_module(prediction, target)
                        self.assertTrue(
                            torch.equal(sensitivity_from_module, expected_sensitivity),
                            f"Module-based implementation correctly computes sensitivity for {task_type} when all "
                            f"predictions are wrong and {test_case_description}.",
                        )

    def test_no_true_positives(self):
        """
        Tests that the sensitivity is computed correctly when there are no true positives.
        """

        for test_slice, convert_to_one_hot in [
            (
                test_data.slice_no_true_positives_single_label,
                True,
            ),
            (
                test_data.slice_no_true_positives_multi_label,
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

                        sensitivity_from_function = sensitivity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        if epsilon == 0:
                            if reduction == "none":
                                expected_sensitivity = (
                                    torch.zeros(3)
                                    if include_background is True
                                    else torch.zeros(2)
                                )
                            else:
                                expected_sensitivity = torch.tensor(0.0)
                        else:
                            expected_sensitivity = self._expected_sensitivity(
                                cardinalities,
                                probability_positive,
                                probability_negative,
                                epsilon,
                                include_background,
                                reduction,
                            )

                        torch.testing.assert_allclose(
                            sensitivity_from_function,
                            expected_sensitivity,
                            msg=f"Functional implementation correctly computes sensitivity for {task_type} tasks when "
                            f"there are no TP and {test_case_description}.",
                        )

                        sensitivity_module = Sensitivity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        sensitivity_from_module = sensitivity_module(prediction, target)

                        torch.testing.assert_allclose(
                            sensitivity_from_module,
                            expected_sensitivity,
                            msg=f"Module-based implementation correctly computes sensitivity for {task_type} tasks when"
                            f" there are no TP and {test_case_description}.",
                        )

    def test_no_true_negatives(self):
        """
        Tests that the sensitivity is computed correctly when there are no true negatives.
        """

        for test_slice, convert_to_one_hot in [
            (
                test_data.slice_no_true_negatives_single_label,
                True,
            ),
            (
                test_data.slice_no_true_negatives_multi_label,
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
                        expected_sensitivity = self._expected_sensitivity(
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

                        sensitivity_from_function = sensitivity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        torch.testing.assert_allclose(
                            sensitivity_from_function,
                            expected_sensitivity,
                            msg=f"Functional implementation correctly computes sensitivity for {task_type} tasks when "
                            f"there are no TN and {test_case_description}.",
                        )

                        sensitivity_module = Sensitivity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        sensitivity_from_module = sensitivity_module(prediction, target)

                        torch.testing.assert_allclose(
                            sensitivity_from_module,
                            expected_sensitivity,
                            msg=f"Module-based implementation correctly computes sensitivity for {task_type} tasks when"
                            f" there are no TN and {test_case_description}.",
                        )

    def test_all_true_negatives(self):
        """
        Tests that the sensitivity is computed correctly when there are only true negatives.
        """

        for test_slice, convert_to_one_hot in [
            (
                test_data.slice_all_true_negatives_single_label,
                True,
            ),
            (
                test_data.slice_all_true_negatives_multi_label,
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

                        sensitivity_from_function = sensitivity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        if epsilon == 0:
                            if convert_to_one_hot is False or reduction != "none":
                                self.assertTrue(
                                    torch.all(torch.isnan(sensitivity_from_function)),
                                    f"Functional implementation correctly computes sensitivity for {task_type} tasks "
                                    f"when there are only TN and {test_case_description}.",
                                )
                            else:
                                self.assertTrue(
                                    torch.isnan(sensitivity_from_function)[-1],
                                    f"Functional implementation correctly computes sensitivity for {task_type} tasks "
                                    f"when there are only TN and {test_case_description}.",
                                )
                        else:
                            self.assertTrue(
                                torch.all(sensitivity_from_function == 1.0),
                                f"Functional implementation correctly computes sensitivity for {task_type} tasks when "
                                f" there are only TN and {test_case_description}.",
                            )

                        sensitivity_module = Sensitivity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        sensitivity_from_module = sensitivity_module(prediction, target)

                        if epsilon == 0:
                            if convert_to_one_hot is False or reduction != "none":
                                self.assertTrue(
                                    torch.all(torch.isnan(sensitivity_from_module)),
                                    f"Module-based implementation correctly computes sensitivity for {task_type} tasks "
                                    f"when there are only TN and {test_case_description}.",
                                )
                            else:
                                self.assertTrue(
                                    torch.isnan(sensitivity_from_function)[-1],
                                    f"Module-based implementation correctly computes sensitivity when there are only TN"
                                    f" and {test_case_description}.",
                                )
                        else:
                            self.assertTrue(
                                torch.all(sensitivity_from_module == 1.0),
                                f"Module-based implementation correctly computes smoothed sensitivity when there are "
                                f"only TN and {test_case_description}.",
                            )

    def test_3d(self):
        """
        Tests that the sensitivity is computed correctly when the inputs are three-dimensional.
        """

        for test_slice_1, test_slice_2, convert_to_one_hot in [
            (
                test_data.standard_slice_single_label_1,
                test_data.standard_slice_single_label_2,
                True,
            ),
            (
                test_data.standard_slice_multi_label_1,
                test_data.standard_slice_multi_label_2,
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
                        expected_sensitivity = self._expected_sensitivity(
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

                        sensitivity_from_function = sensitivity(
                            prediction,
                            target,
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )

                        torch.testing.assert_allclose(
                            sensitivity_from_function,
                            expected_sensitivity,
                            msg=f"Functional implementation correctly computes sensitivity for {task_type} tasks when "
                            f"the inputs are three-dimensional and {test_case_description}.",
                        )

                        sensitivity_module = Sensitivity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            epsilon=epsilon,
                            include_background=include_background,
                            reduction=reduction,
                        )
                        sensitivity_from_module = sensitivity_module(prediction, target)

                        torch.testing.assert_allclose(
                            sensitivity_from_module,
                            expected_sensitivity,
                            msg=f"Module-based implementation correctly computes sensitivity for {task_type} tasks when"
                            f" the inputs are three-dimensional and {test_case_description}.",
                        )

    def test_ignore_index(self):
        """
        Tests that the sensitivity is computed correctly when there are pixels / voxels to be ignored.
        """

        for test_slice, convert_to_one_hot in [
            (
                test_data.slice_ignore_index_single_label,
                True,
            ),
            (
                test_data.slice_ignore_index_multi_label,
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
                        expected_sensitivity = self._expected_sensitivity(
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

                        sensitivity_from_function = sensitivity(
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
                            sensitivity_from_function,
                            expected_sensitivity,
                            msg=f"Functional implementation correctly computes sensitivity for {task_type} tasks when "
                            f"there are pixels / voxels to be ignored and {test_case_description}.",
                        )

                        sensitivity_module = Sensitivity(
                            len(cardinalities),
                            convert_to_one_hot=convert_to_one_hot,
                            ignore_index=-1,
                            include_background=include_background,
                            epsilon=epsilon,
                            reduction=reduction,
                        )
                        sensitivity_from_module = sensitivity_module(prediction, target)

                        torch.testing.assert_allclose(
                            sensitivity_from_module,
                            expected_sensitivity,
                            msg=f"Module-based implementation correctly computes sensitivity for {task_type} tasks when"
                            f" there are pixels / voxels to be ignored and {test_case_description}.",
                        )
