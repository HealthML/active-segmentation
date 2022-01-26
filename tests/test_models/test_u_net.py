"""Tests for U-net architecture."""

import unittest

import torch

from models import UNet


class TestUNet(unittest.TestCase):
    """Test cases for U-net architecture."""

    def test_output_shape(self):
        """Tests that the U-net architecture produces outputs with the expected shape."""

        for multi_label in [True, False]:
            for dim in [2, 3]:
                in_channels = 1
                num_classes = 5

                u_net = UNet(
                    in_channels=in_channels,
                    out_channels=num_classes,
                    multi_label=multi_label,
                    init_features=32,
                    num_levels=4,
                    dim=dim,
                )

                batch_size = 4
                x = 50
                y = 50
                z = 30
                if dim == 2:
                    test_batch = torch.randn(batch_size, in_channels, y, x)
                else:
                    test_batch = torch.randn(batch_size, in_channels, z, y, x)

                model_output = u_net(test_batch)

                if dim == 2:
                    expected_output_shape = torch.Size((batch_size, num_classes, y, x))
                else:
                    expected_output_shape = torch.Size(
                        (batch_size, num_classes, z, y, x)
                    )

                task_type = "multi-label" if multi_label is True else "single-label"

                self.assertEqual(
                    model_output.shape,
                    expected_output_shape,
                    f"The output of the U-Net architecture has the expected shape for {task_type} tasks if the input is"
                    f" {dim}-dimensional.",
                )

                self.assertTrue(
                    (model_output >= 0.0).all() and (model_output <= 1).all(),
                    "The outputs of the U-Net architecture are probabilities between zero and one.",
                )
                if not multi_label:
                    print("model_output", model_output)
                    summed_probabilities = model_output.sum(dim=1)

                    torch.testing.assert_allclose(
                        summed_probabilities,
                        torch.ones(batch_size, y, x)
                        if dim == 2
                        else torch.ones(batch_size, z, y, x),
                    )
