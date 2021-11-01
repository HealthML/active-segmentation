import abc
import numpy
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, SGD


class PytorchModel(LightningModule):
    def __init__(self, learning_rate=0.0001, optimizer="adam", **kwargs):
        super().__init__(**kwargs)

        self.learning_rate = learning_rate
        self.optimizer = optimizer

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
        if self.optimizer == "adam":
            return Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer name.")

    def predict(self, batch: torch.Tensor) -> numpy.ndarray:
        """
        Computes predictions for a given batch of model inputs.

        :param batch: A batch of model inputs.
        :return: Predictions for the given inputs.
        """

        self.eval()
        with torch.no_grad:
            return self(batch).cpu().numpy()
