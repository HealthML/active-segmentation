"""U-Net architecture wrapped as PytorchModel"""

from typing import Iterable

import torch

from .pytorch_model import PytorchModel
from .u_net import UNet


# pylint: disable-msg=too-many-ancestors, abstract-method
class PytorchUNet(PytorchModel):
    """
    U-Net architecture wrapped as PytorchModel.
    Details about the architecture: https://arxiv.org/pdf/1505.04597.pdf
    Args:
        **kwargs: Further, dataset specific parameters.
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.model = UNet(in_channels=1, out_channels=1, init_features=32)

    # wrap model interface
    def eval(self) -> None:
        """
        Sets model to evaluation mode.
        """

        return self.model.eval()

    def train(self, mode: bool = True):
        """
        Sets model to training mode.
        """

        # pylint: disable-msg=unused-argument

        return self.model.train(mode=mode)

    def parameters(self, recurse: bool = True) -> Iterable:
        """

        Returns:
            Iterable: Model parameters.
        """

        # pylint: disable-msg=unused-argument

        return self.model.parameters(recurse=recurse)

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (Tensor): Batch of input images.

        Returns:
            Tensor: Segmentation masks.
        """

        # pylint: disable-msg=arguments-differ

        return self.model.forward(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """
        Trains the model on a given batch of input images.

        Args:
            batch (Tensor): Batch of training images.
            batch_idx: Index of the training batch.

        Returns:
            Loss on the training batch.
        """

        x, y = batch

        probabilities = self(x)
        loss = self.loss(probabilities, y)
        self.log("train/loss", loss)  # log train loss via weights&biases
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """
        Validates the model on a given batch of input images.

        Args:
            batch (Tensor): Batch of validation images.
            batch_idx: Index of the validation batch.
        """

        # pylint: disable-msg=unused-variable
        x, y = batch

        probabilities = self(x)
        loss = self.loss(probabilities, y)
        self.log("validation/loss", loss)  # log validation loss via weights&biases

        # ToDo: this method should return the required performance metrics
