"""Tests for the segmentation metrics"""

import unittest
import torch

from functional import DiceScore, dice_score, sensitivity, Sensitivity
import tests.utils


class TestDiceScore(unittest.TestCase):
    """
    Test cases for dice score.
    """

    def test_standard_case(self):
        """
        Tests that the dice score is computed correctly when there are both true and false predictions.
        """

        prediction, target, tp, fp, _, fn = tests.utils.standard_slice_1()

        score_from_function = dice_score(prediction, target)
        self.assertTrue(
            torch.equal(score_from_function, torch.tensor(2 * tp / (2 * tp + fp + fn))),
            "Functional implementation correctly computes dice score when there are TP, FP and FN.",
        )

        smoothed_score_from_function = dice_score(prediction, target, smoothing=1)
        self.assertTrue(
            torch.equal(
                smoothed_score_from_function,
                torch.tensor((2 * tp + 1) / (2 * tp + fp + fn + 1)),
            ),
            "Functional implementation correctly computes smoothed dice score when there are TP, FP and FN.",
        )

        dice_score_module = DiceScore()
        score_from_module = dice_score_module(prediction, target)
        self.assertTrue(
            torch.equal(score_from_module, torch.tensor(2 * tp / (2 * tp + fp + fn))),
            "Module-based implementation correctly computes dice score when there are TP, FP and FN.",
        )

        dice_score_module.update(prediction, target)
        score_from_module_compute = dice_score_module.compute()
        self.assertTrue(
            torch.equal(
                score_from_module_compute, torch.tensor(2 * tp / (2 * tp + fp + fn))
            ),
            "Compute method of module-based implementation returns correct dice score.",
        )

        dice_score_module = DiceScore(smoothing=1)
        smoothed_score_from_module = dice_score_module(prediction, target)
        self.assertTrue(
            torch.equal(
                smoothed_score_from_module,
                torch.tensor((2 * tp + 1) / (2 * tp + fp + fn + 1)),
            ),
            "Module-based implementation correctly computes smoothed dice score when there are TP, FP and FN.",
        )

    def test_all_true(self):
        """
        Tests that the dice score is computed correctly when all predictions are correct.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_true()

        score_from_function = dice_score(prediction, target)
        self.assertTrue(
            torch.equal(score_from_function, torch.tensor(1.0)),
            "Functional implementation correctly computes dice score when there are no prediction errors.",
        )

        dice_score_module = DiceScore()
        score_from_module = dice_score_module(prediction, target)
        self.assertTrue(
            torch.equal(score_from_module, torch.tensor(1.0)),
            "Module-based implementation correctly computes dice score when there are no prediction errors.",
        )

    def test_all_false(self):
        """
        Tests that the dice score is computed correctly when all predictions are wrong.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_false()

        score_from_function = dice_score(prediction, target)
        self.assertTrue(
            torch.equal(score_from_function, torch.tensor(0.0)),
            "Functional implementation correctly computes dice score when all predictions are wrong.",
        )

        dice_score_module = DiceScore()
        score_from_module = dice_score_module(prediction, target)
        self.assertTrue(
            torch.equal(score_from_module, torch.tensor(0.0)),
            "Module-based implementation correctly computes dice score when all predictions are wrong.",
        )

    def test_no_true_positives(self):
        """
        Tests that the dice score is computed correctly when there are no true positives.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_no_true_positives()

        score_from_function = dice_score(prediction, target)
        self.assertTrue(
            torch.equal(score_from_function, torch.tensor(0.0)),
            "Functional implementation correctly computes dice score when there are no TP.",
        )

        dice_score_module = DiceScore()
        score_from_module = dice_score_module(prediction, target)
        self.assertTrue(
            torch.equal(score_from_module, torch.tensor(0.0)),
            "Module-based implementation correctly computes dice score when there are no TP.",
        )

    def test_no_true_negatives(self):
        """
        Tests that the dice score is computed correctly when there are no true negatives.
        """

        prediction, target, tp, fp, _, fn = tests.utils.slice_no_true_negatives()

        score_from_function = dice_score(prediction, target)
        self.assertTrue(
            torch.equal(score_from_function, torch.tensor(2 * tp / (2 * tp + fp + fn))),
            "Functional implementation correctly computes dice score when there are no TN.",
        )

        dice_score_module = DiceScore()
        score_from_module = dice_score_module(prediction, target)
        self.assertTrue(
            torch.equal(score_from_module, torch.tensor(2 * tp / (2 * tp + fp + fn))),
            "Module-based implementation correctly computes dice score when there are no TN.",
        )

    def test_all_true_negatives(self):
        """
        Tests that the dice score is computed correctly when there are only true negatives.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_true_negatives()

        score_from_function = dice_score(prediction, target)
        self.assertTrue(
            torch.isnan(score_from_function),
            "Functional implementation correctly computes dice score when there are only TN.",
        )

        smoothed_score_from_function = dice_score(prediction, target, smoothing=1)
        self.assertTrue(
            torch.equal(smoothed_score_from_function, torch.tensor(1.0)),
            "Functional implementation correctly computes smoothed dice score when there are only TN.",
        )

        dice_score_module = DiceScore()
        score_from_module = dice_score_module(prediction, target)
        self.assertTrue(
            torch.isnan(score_from_module),
            "Module-based implementation correctly computes dice score when there are only TN.",
        )

        dice_score_module = DiceScore(smoothing=1)
        smoothed_score_from_module = dice_score_module(prediction, target)
        self.assertTrue(
            torch.equal(smoothed_score_from_module, torch.tensor(1.00)),
            "Module-based implementation correctly computes smoothed dice score when there are only TN.",
        )


class TestSensitivity(unittest.TestCase):
    """
    Test cases for sensitivity.
    """

    def test_standard_case(self):
        """
        Tests that the sensitivity is computed correctly when there are both true and false predictions.
        """

        prediction, target, tp, _, _, fn = tests.utils.standard_slice_1()

        sensitivity_from_function = sensitivity(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_function, torch.tensor(tp / (tp + fn))),
            "Functional implementation correctly computes sensitivity when there are TP, FP and FN.",
        )

        smoothed_sensitivity_from_function = sensitivity(
            prediction, target, smoothing=1
        )
        self.assertTrue(
            torch.equal(
                smoothed_sensitivity_from_function,
                torch.tensor((tp + 1) / (tp + fn + 1)),
            ),
            "Functional implementation correctly computes smoothed sensitivity when there are TP, FP and FN.",
        )

        sensitivity_module = Sensitivity()
        sensitivity_from_module = sensitivity_module(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_module, torch.tensor(tp / (tp + fn))),
            "Module-based implementation correctly computes sensitivity when there are TP, FP and FN.",
        )

        sensitivity_module.update(prediction, target)
        sensitivity_from_module_compute = sensitivity_module.compute()
        self.assertTrue(
            torch.equal(
                sensitivity_from_module_compute,
                torch.tensor(tp / (tp + fn)),
            ),
            "Compute method of module-based implementation returns correct sensitivity.",
        )

        sensitivity_module = Sensitivity(smoothing=1)
        smoothed_sensitivity_module = sensitivity_module(prediction, target)
        self.assertTrue(
            torch.equal(
                smoothed_sensitivity_module, torch.tensor((tp + 1) / (tp + fn + 1))
            ),
            "Module-based implementation correctly computes smoothed sensitivity when there are TP, FP and FN.",
        )

    def test_all_true(self):
        """
        Tests that the sensitivity is computed correctly when all predictions are correct.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_true()

        sensitivity_from_function = sensitivity(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_function, torch.tensor(1.0)),
            "Functional implementation correctly computes sensitivity when there are no prediction errors.",
        )

        sensitivity_module = Sensitivity()
        sensitivity_from_module = sensitivity_module(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_module, torch.tensor(1.0)),
            "Module-based implementation correctly computes sensitivity when there are no prediction errors.",
        )

    def test_all_false(self):
        """
        Tests that the sensitivity is computed correctly when all predictions are wrong.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_false()

        sensitivity_from_function = sensitivity(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_function, torch.tensor(0.0)),
            "Functional implementation correctly computes sensitivity when all predictions are wrong.",
        )

        sensitivity_module = Sensitivity()
        sensitivity_from_module = sensitivity_module(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_module, torch.tensor(0.0)),
            "Module-based implementation correctly computes sensitivity when all predictions are wrong.",
        )

    def test_no_true_positives(self):
        """
        Tests that the sensitivity is computed correctly when there are no true positives.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_no_true_positives()

        sensitivity_from_function = sensitivity(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_function, torch.tensor(0.0)),
            "Functional implementation correctly computes sensitivity when there are no TP.",
        )

        sensitivity_module = Sensitivity()
        sensitivity_from_module = sensitivity_module(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_module, torch.tensor(0.0)),
            "Module-based implementation correctly computes sensitivity when there are no TP.",
        )

    def test_no_true_negatives(self):
        """
        Tests that the sensitivity is computed correctly when there are no true negatives.
        """

        prediction, target, tp, _, _, fn = tests.utils.slice_no_true_negatives()

        sensitivity_from_function = sensitivity(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_function, torch.tensor(tp / (tp + fn))),
            "Functional implementation correctly computes sensitivity when there are no TN.",
        )

        sensitivity_module = Sensitivity()
        sensitivity_from_module = sensitivity_module(prediction, target)
        self.assertTrue(
            torch.equal(sensitivity_from_module, torch.tensor(tp / (tp + fn))),
            "Module-based implementation correctly computes sensitivity when there are no TN.",
        )

    def test_all_true_negatives(self):
        """
        Tests that the sensitivity is computed correctly when there are only true negatives.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_true_negatives()

        sensitivity_from_function = sensitivity(prediction, target)
        self.assertTrue(
            torch.isnan(sensitivity_from_function),
            "Functional implementation correctly computes sensitivity when there are only TN.",
        )

        smoothed_sensitivity_from_function = sensitivity(
            prediction, target, smoothing=1
        )
        self.assertTrue(
            torch.equal(smoothed_sensitivity_from_function, torch.tensor(1.0)),
            "Functional implementation correctly computes smoothed sensitivity when there are only TN.",
        )

        sensitivity_module = Sensitivity()
        sensitivity_from_module = sensitivity_module(prediction, target)
        self.assertTrue(
            torch.isnan(sensitivity_from_module),
            "Module-based implementation correctly computes sensitivity when there are only TN.",
        )

        sensitivity_module = Sensitivity(smoothing=1)
        smoothed_sensitivity_from_module = sensitivity_module(prediction, target)
        self.assertTrue(
            torch.equal(smoothed_sensitivity_from_module, torch.tensor(1.0)),
            "Module-based implementation correctly computes smoothed sensitivity when there are only TN.",
        )
