"""Tests for the combined false positive and dice loss."""

import unittest

from functional import FalsePositiveDiceLoss


class TestFalsePositiveDiceLoss(unittest.TestCase):
    """
    Test cases for combined false positive dice loss.
    """

    def test_attribute_passing(self):
        """Tests that the combined loss module correctly passes its attributes to the submodules."""

        for ignore_index in [None, -1]:
            for include_background in [True, False]:
                for reduction in ["mean", "sum", "none"]:
                    for epsilon in [0, 1]:
                        fp_dice_loss = FalsePositiveDiceLoss(
                            ignore_index=ignore_index,
                            include_background=include_background,
                            reduction=reduction,
                            epsilon=epsilon,
                        )

                        self.assertEqual(
                            ignore_index,
                            fp_dice_loss.dice_loss.ignore_index,
                            "The `ignore_index` attribute is passed to the Dice loss submodule.",
                        )

                        self.assertEqual(
                            ignore_index,
                            fp_dice_loss.fp_loss.ignore_index,
                            "The `ignore_index` attribute is passed to the false positive loss submodule.",
                        )

                        self.assertEqual(
                            include_background,
                            fp_dice_loss.dice_loss.dice_loss.include_background,
                            "The `include_background` attribute is passed to the Dice loss submodule.",
                        )

                        self.assertEqual(
                            include_background,
                            fp_dice_loss.fp_loss.include_background,
                            "The `include_background` attribute is passed to the false positive loss submodule.",
                        )

                        self.assertEqual(
                            reduction,
                            fp_dice_loss.dice_loss.reduction,
                            "The `reduction` attribute is passed to the Dice loss submodule.",
                        )

                        self.assertEqual(
                            reduction,
                            fp_dice_loss.fp_loss.reduction,
                            "The `reduction` attribute is passed to the false positive loss submodule.",
                        )

                        self.assertEqual(
                            epsilon,
                            fp_dice_loss.dice_loss.dice_loss.smooth_nr,
                            "The `epsilon` attribute is passed to the `smooth_nr` parameter of the Dice loss "
                            "submodule.",
                        )

                        self.assertEqual(
                            epsilon,
                            fp_dice_loss.dice_loss.dice_loss.smooth_dr,
                            "The `epsilon` attribute is passed to the `smooth_dr` parameter of the Dice loss "
                            "submodule.",
                        )

                        self.assertEqual(
                            epsilon,
                            fp_dice_loss.fp_loss.epsilon,
                            "The `epsilon` attribute is passed to the false positive loss submodule.",
                        )
