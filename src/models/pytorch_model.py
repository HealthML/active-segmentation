""" Base classes to implement models with pytorch """
from abc import ABC, abstractmethod
from typing import Any, Iterable, Union
import numpy
import torch
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, SGD

import functional
from metric_tracking import CombinedPerEpochMetric


class PytorchModel(LightningModule, ABC):
    """
    Base class to implement Pytorch models.
    Args:
        learning_rate: The step size at each iteration while moving towards a minimum of the loss function.
        optimizer: Algorithm used to calculate the loss and update the weights. E.g. 'adam' or 'sgd'.
        loss: The measure of performance. E.g. 'dice', 'bce', 'fp'
        **kwargs:
    """

    # pylint: disable=too-many-ancestors,arguments-differ

    def __init__(
        self,
        learning_rate: float = 0.0001,
        optimizer: str = "adam",
        loss: str = "dice",
        **kwargs
    ):

        super().__init__(**kwargs)

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = self.configure_loss(loss)

        self.confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        self.train_average_metrics = CombinedPerEpochMetric(
            phase="train",
            metrics=["dice", "sensitivity", "specificity", "hausdorff95"],
            confidence_levels=self.confidence_levels,
            reduction="mean",
            metrics_to_aggregate=["dice", "hausdorff95"],
        )

        self.train_metrics_per_case = CombinedPerEpochMetric(
            phase="train",
            metrics=["dice", "sensitivity", "specificity", "hausdorff95"],
            metrics_to_aggregate=["dice", "hausdorff95"],
            confidence_levels=self.confidence_levels,
            reduction="none",
        )

        self.train_metrics = [self.train_average_metrics, self.train_metrics_per_case]

        self.val_average_metrics = CombinedPerEpochMetric(
            phase="val",
            metrics=["dice", "sensitivity", "specificity", "hausdorff95"],
            confidence_levels=self.confidence_levels,
            reduction="mean",
            metrics_to_aggregate=["dice", "hausdorff95"],
        )

        self.val_metrics_per_case = CombinedPerEpochMetric(
            phase="val",
            metrics=["dice", "sensitivity", "specificity", "hausdorff95"],
            metrics_to_aggregate=["dice", "hausdorff95"],
            confidence_levels=self.confidence_levels,
            reduction="none",
        )

        self.val_metrics = [self.val_average_metrics, self.val_metrics_per_case]

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
            The optimizer object.
        """

        if self.optimizer == "adam":
            return Adam(self.parameters(), lr=self.learning_rate)
        if self.optimizer == "sgd":
            return SGD(self.parameters(), lr=self.learning_rate)
        raise ValueError("Invalid optimizer name.")

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

    def get_train_metrics(self) -> Iterable[torchmetrics.Metric]:
        """
        Returns:
            A list of metrics to be updated in each training step.
        """

        return self.train_metrics

    def get_val_metrics(self) -> Iterable[torchmetrics.Metric]:
        """
        Returns:
            A list of metrics to be updated in each validation step.
        """

        return self.val_metrics

    def training_epoch_end(self, training_step_outputs: Any):
        """
        This method is called by the Pytorch Lightning framework at the end of each training epoch.

        Args:
            training_step_outputs: List of return values of all training steps of the current training epoch. 
        """

        for train_metric in self.train_metrics:
            train_metrics = train_metric.compute()
            self.logger.log_metrics(train_metrics)

    def validation_epoch_end(self, validation_step_outputs: Any):
        """
        This method is called by the Pytorch Lightning framework at the end of each validation epoch.

        Args:
            validation_step_outputs: List of return values of all validation steps of the current validation epoch. 
        """

        for val_metric in self.val_metrics:
            val_metrics = val_metric.compute()
            self.logger.log_metrics(val_metrics)
