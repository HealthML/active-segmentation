"""Tests for the combined cross-entropy and dice loss."""

import unittest

from functional import CrossEntropyDiceLoss


class TestCrossEntropyDiceLoss(unittest.TestCase):
    """
    Test cases for combined cross-entropy dice loss.
    """

    # pylint: disable=too-many-nested-blocks

    def test_attribute_passing(self):
        """Tests that the combined loss module correctly passes its attributes to the submodules."""

        for multi_label in [True, False]:
            for ignore_index in [None, -1]:
                for include_background in [True, False]:
                    for reduction in ["mean", "sum", "none"]:
                        for epsilon in [0, 1]:
                            cross_entropy_dice_loss = CrossEntropyDiceLoss(
                                multi_label=multi_label,
                                ignore_index=ignore_index,
                                include_background=include_background,
                                reduction=reduction,
                                epsilon=epsilon,
                            )

                            self.assertEqual(
                                multi_label,
                                cross_entropy_dice_loss.cross_entropy_loss.multi_label,
                                "The `multi_label` attribute is passed to the cross-entropy loss submodule.",
                            )

                            self.assertEqual(
                                ignore_index,
                                cross_entropy_dice_loss.dice_loss.ignore_index,
                                "The `ignore_index` attribute is passed to the Dice loss submodule.",
                            )

                            self.assertEqual(
                                ignore_index,
                                cross_entropy_dice_loss.cross_entropy_loss.ignore_index,
                                "The `ignore_index` attribute is passed to the cross-entropy loss submodule.",
                            )

                            if multi_label is False and ignore_index is not None:
                                self.assertEqual(
                                    ignore_index,
                                    cross_entropy_dice_loss.cross_entropy_loss.cross_entropy_loss.ignore_index,
                                    "The `ignore_index` attribute is passed to the cross-entropy loss submodule.",
                                )

                            self.assertEqual(
                                include_background,
                                cross_entropy_dice_loss.dice_loss.dice_loss.include_background,
                                "The `include_background` attribute is passed to the Dice loss submodule.",
                            )

                            self.assertEqual(
                                reduction,
                                cross_entropy_dice_loss.dice_loss.reduction,
                                "The `reduction` attribute is passed to the Dice loss submodule.",
                            )

                            self.assertEqual(
                                reduction,
                                cross_entropy_dice_loss.cross_entropy_loss.reduction,
                                "The `reduction` attribute is passed to the cross-entropy loss submodule.",
                            )

                            self.assertEqual(
                                epsilon,
                                cross_entropy_dice_loss.dice_loss.dice_loss.smooth_nr,
                                "The `epsilon` attribute is passed to the `smooth_nr` parameter of the Dice loss "
                                "submodule.",
                            )

                            self.assertEqual(
                                epsilon,
                                cross_entropy_dice_loss.dice_loss.dice_loss.smooth_dr,
                                "The `epsilon` attribute is passed to the `smooth_dr` parameter of the Dice loss "
                                "submodule.",
                            )
