"""Tests for the binary-cross entropy loss"""

import unittest
import torch

from functional import BCELoss
import tests.utils


class TestBCELoss(unittest.TestCase):
    """
    Test cases for binary cross-entropy loss.
    """

    def test_standard_case(self):
        """
        Tests that the binary cross-entropy loss is computed correctly when there are both true and false predictions.
        """

        # pylint: disable-msg=no-self-use

        (
            prediction,
            target,
            _,
            _,
            _,
            _,
        ) = tests.utils.probabilistic_slice()

        bce_loss = BCELoss(reduction="none")
        loss = bce_loss(
            torch.stack([prediction, prediction]), torch.stack([target, target])
        )
        expected_loss = torch.Tensor([[2.1407], [2.1407]])
        torch.testing.assert_allclose(
            loss, expected_loss, msg="Computes unreduced loss correctly."
        )

        bce_loss = BCELoss(reduction="mean")
        loss = bce_loss(
            torch.stack([prediction, prediction]), torch.stack([target, target])
        )
        expected_loss = torch.as_tensor(2.1407)
        torch.testing.assert_allclose(
            loss, expected_loss, msg="Averages loss correctly."
        )

        bce_loss = BCELoss(reduction="sum")
        loss = bce_loss(
            torch.stack([prediction, prediction]), torch.stack([target, target])
        )
        expected_loss = torch.as_tensor(2 * 2.1407)
        torch.testing.assert_allclose(loss, expected_loss, msg="Sums loss correctly.")
