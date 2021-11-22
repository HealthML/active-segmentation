""" Base classes to implement models with pytorch """
from abc import ABC, abstractmethod
from typing import Union
import numpy
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, SGD

import functional


class PytorchModel(LightningModule, ABC):
    """
    Base class to implement Pytorch models.
    Args:
        learning_rate: The step size at each iteration while moving towards a minimum of the loss function.
        optimizer: Algorithm used to calculate the loss and update the weights. E.g. 'adam' or 'sgd'.
        lr_scheduler: Algorithm used for dynamically updating the learning rate during training. E.g. 'reduceLROnPlateau'
        loss: The measure of performance. E.g. 'dice', 'bce', 'fp'
        **kwargs:
    """

    # pylint: disable=too-many-ancestors,arguments-differ

    def __init__(
        self,
        learning_rate: float = 0.0001,
        optimizer: str = "adam",
        lr_scheduler: str = None,
        loss: str = "dice",
        **kwargs
    ):

        super().__init__(**kwargs)

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = self.configure_loss(loss)

    @abstractmethod
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """
        Trains the model on a given batch of model inputs.
        # this method should match the requirements of the pytorch lightning framework.
        # see https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        Args:
            batch: A batch of model inputs.
            batch_idx: Index of the current batch.

        Returns:
            Training loss.
        """

        return 0

    @abstractmethod
    def validation_step(self, batch, batch_idx) -> None:
        """
        Validates the model on a given batch of model inputs.
        # this method should match the requirements of the pytorch lightning framework.
        # see https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        Args:
            batch: A batch of model inputs.
            batch_idx: Index of the current batch.

        Returns:
            None.
        """

        # ToDo: this method should return the required performance metrics

        return None

    def configure_optimizers(self) -> Union[Adam, SGD]:
        """
        This method is called by the PyTorch lightning framework before starting model training.

        Returns:
            The optimizer object and optionally a learning rate scheduler object.
        """
        if self.optimizer == "adam":
            opt = Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            opt = SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer name.")

        scheduler = None
        if self.lr_scheduler == "reduceLROnPlateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt),
                "monitor": "validation/loss",
            }

        if scheduler is not None:
            return [opt], [scheduler]
        else:
            return opt

    @staticmethod
    def configure_loss(loss: str) -> functional.losses.SegmentationLoss:
        """
        Configures the loss.
        Args:
            loss: name of the loss

        Returns:
            The loss object.
        """

        if loss == "bce":
            return functional.BCELoss()
        if loss == "bce_dice":
            return functional.BCEDiceLoss()
        if loss == "dice":
            return functional.DiceLoss()
        if loss == "fp":
            return functional.FalsePositiveLoss()
        if loss == "fp_dice":
            return functional.FalsePositiveDiceLoss()
        raise ValueError("Invalid loss name.")

    def predict(self, batch: torch.Tensor) -> numpy.ndarray:
        """
        Computes predictions for a given batch of model inputs.
        Args:
            batch: A batch of model inputs.

        Returns:
            Predictions for the given inputs.
        """

        self.eval()
        with torch.no_grad:
            return self(batch).cpu().numpy()
