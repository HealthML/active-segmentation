""" Base classes to implement models with pytorch """
import abc
import numpy
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, SGD

from functional import (
    BCELoss,
    BCEDiceLoss,
    DiceLoss,
    FalsePositiveLoss,
    FalsePositiveDiceLoss,
)


class PytorchModel(LightningModule):
    """TBD"""

    # pylint: disable=too-many-ancestors,arguments-differ

    def __init__(self, learning_rate=0.0001, optimizer="adam", loss="dice", **kwargs):
        super().__init__(**kwargs)

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = self.configure_loss(loss)

    @abc.abstractmethod
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """
        Trains the model on a given batch of model inputs.
        # this method should match the requirements of the pytorch lightning framework.
        # see https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        :param batch: A batch of model inputs.
        :param batch_idx: Index of the current batch.
        :return: Training loss.
        """

        return 0

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx) -> None:
        """
        Validates the model on a given batch of model inputs.
        # this method should match the requirements of the pytorch lightning framework.
        # see https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        :param batch: A batch of model inputs.
        :param batch_idx: Index of the current batch.
        """

        # ToDo: this method should return the required performance metrics

        return None

    def configure_optimizers(self):
        """
        this method is called by the PyTorch lightning framework before starting model training
        :return:
        """

        if self.optimizer == "adam":
            return Adam(self.parameters(), lr=self.learning_rate)
        if self.optimizer == "sgd":
            return SGD(self.parameters(), lr=self.learning_rate)
        raise ValueError("Invalid optimizer name.")

    @staticmethod
    def configure_loss(loss: str):
        """
        Configures the loss
        :param loss: name of the loss
        :return:
        """
        if loss == "bce":
            return BCELoss()
        if loss == "bce_dice":
            return BCEDiceLoss()
        if loss == "dice":
            return DiceLoss()
        if loss == "fp":
            return FalsePositiveLoss()
        if loss == "fp_dice":
            return FalsePositiveDiceLoss()
        raise ValueError("Invalid loss name.")

    def predict(self, batch: torch.Tensor) -> numpy.ndarray:
        """
        Computes predictions for a given batch of model inputs.

        :param batch: A batch of model inputs.
        :return: Predictions for the given inputs.
        """

        self.eval()
        with torch.no_grad:
            return self(batch).cpu().numpy()
