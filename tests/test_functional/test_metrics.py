"""Tests for the segmentation metrics"""

import unittest
import torch

from functional import (
    DiceScore,
    dice_score,
    sensitivity,
    specificity,
    Sensitivity,
    Specificity,
    hausdorff_distance,
    HausdorffDistance,
)
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


class TestSpecificity(unittest.TestCase):
    """
    Test cases for specificity.
    """

    def test_standard_case(self):
        """
        Tests that the specificity is computed correctly when there are both true and false predictions.
        """

        prediction, target, _, fp, tn, _ = tests.utils.standard_slice_1()

        specificity_from_function = specificity(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_function, torch.tensor(tn / (tn + fp))),
            "Functional implementation correctly computes specificity when there are TP, FP and FN.",
        )

        smoothed_specificity_from_function = specificity(
            prediction, target, smoothing=1
        )
        self.assertTrue(
            torch.equal(
                smoothed_specificity_from_function,
                torch.tensor((tn + 1) / (tn + fp + 1)),
            ),
            "Functional implementation correctly computes smoothed specificity when there are TP, FP and FN.",
        )

        specificity_module = Specificity()
        specificity_from_module = specificity_module(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_module, torch.tensor(tn / (tn + fp))),
            "Module-based implementation correctly computes specificity when there are TP, FP and FN.",
        )

        specificity_module.update(prediction, target)
        specificity_from_module_compute = specificity_module.compute()
        self.assertTrue(
            torch.equal(
                specificity_from_module_compute,
                torch.tensor(tn / (tn + fp)),
            ),
            "Compute method of module-based implementation returns correct specificity.",
        )

        specificity_module = Specificity(smoothing=1)
        smoothed_specificity_module = specificity_module(prediction, target)
        self.assertTrue(
            torch.equal(
                smoothed_specificity_module, torch.tensor((tn + 1) / (tn + fp + 1))
            ),
            "Module-based implementation correctly computes smoothed specificity when there are TP, FP and FN.",
        )

    def test_all_true(self):
        """
        Tests that the specificity is computed correctly when all predictions are correct.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_true()

        specificity_from_function = specificity(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_function, torch.tensor(1.0)),
            "Functional implementation correctly computes specificity when there are no prediction errors.",
        )

        specificity_module = Specificity()
        specificity_from_module = specificity_module(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_module, torch.tensor(1.0)),
            "Module-based implementation correctly computes specificity when there are no prediction errors.",
        )

    def test_all_false(self):
        """
        Tests that the specificity is computed correctly when all predictions are wrong.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_false()

        specificity_from_function = specificity(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_function, torch.tensor(0.0)),
            "Functional implementation correctly computes specificity when all predictions are wrong.",
        )

        specificity_module = Specificity()
        specificity_from_module = specificity_module(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_module, torch.tensor(0.0)),
            "Module-based implementation correctly computes specificity when all predictions are wrong.",
        )

    def test_no_true_positives(self):
        """
        Tests that the specificity is computed correctly when there are no true positives.
        """

        prediction, target, _, fp, tn, _ = tests.utils.slice_no_true_positives()

        specificity_from_function = specificity(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_function, torch.tensor(tn / (tn + fp))),
            "Functional implementation correctly computes specificity when there are no TP.",
        )

        specificity_module = Specificity()
        specificity_from_module = specificity_module(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_module, torch.tensor(tn / (tn + fp))),
            "Module-based implementation correctly computes specificity when there are no TP.",
        )

    def test_no_true_negatives(self):
        """
        Tests that the specificity is computed correctly when there are no true negatives.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_no_true_negatives()

        specificity_from_function = specificity(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_function, torch.tensor(0.0)),
            "Functional implementation correctly computes specificity when there are no TN.",
        )

        specificity_module = Specificity()
        specificity_from_module = specificity_module(prediction, target)
        self.assertTrue(
            torch.equal(specificity_from_module, torch.tensor(0.0)),
            "Module-based implementation correctly computes specificity when there are no TN.",
        )

    def test_all_true_positives(self):
        """
        Tests that the specificity is computed correctly when there are only true positives.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_true_positives()

        specificity_from_function = specificity(prediction, target)
        self.assertTrue(
            torch.isnan(specificity_from_function),
            "Functional implementation correctly computes specificity when there are only TN.",
        )

        smoothed_specificity_from_function = specificity(
            prediction, target, smoothing=1
        )
        self.assertTrue(
            torch.equal(smoothed_specificity_from_function, torch.tensor(1.0)),
            "Functional implementation correctly computes smoothed specificity when there are only TN.",
        )

        specificity_module = Specificity()
        specificity_from_module = specificity_module(prediction, target)
        self.assertTrue(
            torch.isnan(specificity_from_module),
            "Module-based implementation correctly computes specificity when there are only TN.",
        )

        specificity_module = Specificity(smoothing=1)
        smoothed_specificity_from_module = specificity_module(prediction, target)
        self.assertTrue(
            torch.equal(smoothed_specificity_from_module, torch.tensor(1.0)),
            "Module-based implementation correctly computes smoothed specificity when there are only TN.",
        )


class TestHausdorffDistance(unittest.TestCase):
    """
    Test cases for Hausdorff distance.
    """

    def _test_hausdorff_distance(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        expected_distance: float,
        message: str = "",
        percentile: float = 0.95,
    ) -> None:
        """
        Helper function that calculates the Hausdorff distances for the given predictions and compares it with the
        expected values.

        Args:
            prediction (Tensor): Predicted segmentation mask.
            target (Tensor): Target segmentation mask.
            expected_distance (float): Expected Hausdorff distance.
            message (string, optional): Description of the test case.
            percentile (float): Percentile for which the Hausdorff distance is to be tested.
        """

        hausdorff_distance_from_function = hausdorff_distance(
            prediction, target, percentile=percentile
        )

        self.assertTrue(
            torch.equal(
                hausdorff_distance_from_function,
                torch.tensor(expected_distance).float(),
            ),
            f"Functional implementation correctly computes hausdorff distance when {message}.",
        )

        hausdorff_distance_module = HausdorffDistance(percentile=percentile, dim=prediction.dim(), slices_per_image=1)
        hausdorff_distance_from_module = hausdorff_distance_module(prediction, target)
        self.assertTrue(
            torch.equal(
                hausdorff_distance_from_module,
                torch.tensor(expected_distance).float(),
            ),
            f"Module-based implementation correctly computes hausdorff distance when {message}.",
        )

    def test_standard_case(self):
        """
        Tests that the Hausdorff distance is computed correctly when there are both true and false predictions.
        """

        (
            prediction,
            target,
            expected_hausdorff_dist_50,
            _,
            _,
        ) = tests.utils.standard_distance_slice(percentile=0.5)

        self._test_hausdorff_distance(
            prediction,
            target,
            expected_hausdorff_dist_50,
            "there are TP, FP and FN",
            percentile=0.5,
        )

        (
            prediction,
            target,
            expected_hausdorff_dist_95,
            _,
            _,
        ) = tests.utils.standard_distance_slice()

        self._test_hausdorff_distance(
            prediction,
            target,
            expected_hausdorff_dist_95,
            "there are TP, FP and FN",
            percentile=0.95,
        )

        (
            prediction,
            target,
            expected_hausdorff_dist_100,
            _,
            _,
        ) = tests.utils.standard_distance_slice(percentile=1.0)

        self._test_hausdorff_distance(
            prediction,
            target,
            expected_hausdorff_dist_100,
            "there are TP, FP and FN",
            percentile=1.0,
        )

    def test_all_true(self):
        """
        Tests that the Hausdorff distance is computed correctly when all predictions are correct.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_true()

        self._test_hausdorff_distance(
            prediction, target, 0, "there are no prediction errors"
        )

    def test_all_false(self):
        """
        Tests that the Hausdorff distance is computed correctly when all predictions are wrong.
        """

        (
            prediction,
            target,
            expected_hausdorff_dist,
            _,
            _,
        ) = tests.utils.distance_slice_all_false()

        self._test_hausdorff_distance(
            prediction,
            target,
            expected_hausdorff_dist,
            "all predictions are wrong",
        )

    def test_subset(self):
        """
        Tests that the Hausdorff distance is computed correctly when all the predicted mask is a subset of the target
        mask.
        """
        (
            prediction,
            target,
            expected_hausdorff_dist,
            _,
            _,
        ) = tests.utils.distance_slice_subset()

        self._test_hausdorff_distance(
            prediction,
            target,
            expected_hausdorff_dist,
            "the prediction is a subset of the target",
        )

    def test_no_positives(self):
        """
        Tests that the Hausdorff distance is computed correctly when there are no positives.
        """

        prediction, target, _, _, _, _ = tests.utils.slice_all_true_negatives()

        hausdorff_dist_from_function = hausdorff_distance(prediction, target)
        self.assertTrue(
            torch.isnan(hausdorff_dist_from_function),
            "Functional implementation correctly computes Hausdorff distance when there are only TN.",
        )

        hausdorff_distance_module = HausdorffDistance(slices_per_image=1)
        hausdorff_dist_from_module = hausdorff_distance_module(prediction, target)
        self.assertTrue(
            torch.isnan(hausdorff_dist_from_module),
            "Functional implementation correctly computes Hausdorff distance when there are only TN.",
        )

    def test_3d(self):
        """
        Tests that the Hausdorff distance is computed correctly when the predictions are 3-dimensional scans.
        """
        (
            prediction,
            target,
            expected_hausdorff_dist,
            _,
            _,
        ) = tests.utils.distance_slices_3d()

        self._test_hausdorff_distance(
            prediction,
            target,
            expected_hausdorff_dist,
            "the input is 3-dimensional",
        )
