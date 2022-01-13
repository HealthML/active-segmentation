""" Base classes to implement models with pytorch """
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy
import torch
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import functional
from metric_tracking import CombinedPerEpochMetric

Optimizer = Union[Adam, SGD]
LRScheduler = Union[ReduceLROnPlateau, CosineAnnealingLR]
LRSchedulerDict = Dict[str, Union[str, LRScheduler]]


# pylint: disable=too-many-instance-attributes
class PytorchModel(LightningModule, ABC):
    """
    Base class to implement Pytorch models.

    Args:
        learning_rate: The step size at each iteration while moving towards a minimum of the loss.
        optimizer: Algorithm used to calculate the loss and update the weights. E.g. 'adam' or 'sgd'.
        lr_scheduler: Algorithm used for dynamically updating the learning rate during training.
            E.g. 'reduceLROnPlateau' or 'cosineAnnealingLR'
        loss: The measure of performance. E.g. 'dice', 'bce', 'fp'
        train_metrics (Iterable[str], optional): A list with the names of the metrics that should be computed and logged
            in each training and validation epoch of the training loop. Defaults to `["dice_score"]`.
        train_metric_confidence_levels (Iterable[float], optional): A list of confidence levels for which the metrics
            specified in the `train_metrics` parameter should be computed in the training loop (`trainer.fit()`).
            Defaults to `[0.25, 0.5, 0.75]`.
        test_metrics (Iterable[str], optional): A list with the names of the metrics that should be computed and logged
            in the model validation or testing loop (`trainer.validate()`, `trainer.test()`). Defaults to
            `["dice_score", "sensitivity", "specificity", "hausdorff95"]`.
        test_metric_confidence_levels (Iterable[float], optional): A list of confidence levels for which the metrics
            specified in the `test_metrics` parameter should be computed in the validation or testing loop. Defaults to
            `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`.
        **kwargs: Further, dataset specific parameters.
    """

    # pylint: disable=too-many-ancestors,arguments-differ,too-many-arguments

    def __init__(
        self,
        learning_rate: float = 0.0001,
        optimizer: str = "adam",
        lr_scheduler: str = None,
        loss: str = "dice",
        train_metrics: Optional[Iterable[str]] = None,
        train_metric_confidence_levels: Optional[Iterable[float]] = None,
        test_metrics: Optional[Iterable[str]] = None,
        test_metric_confidence_levels: Optional[Iterable[float]] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = self.configure_loss(loss)

        if train_metric_confidence_levels is None:
            train_metric_confidence_levels = [0.25, 0.5, 0.75]
        self.train_metric_confidence_levels = train_metric_confidence_levels

        if test_metric_confidence_levels is None:
            test_metric_confidence_levels = [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        self.test_metric_confidence_levels = test_metric_confidence_levels

        if train_metrics is None:
            train_metrics = ["dice_score"]
        self.train_metric_names = train_metrics

        if test_metrics is None:
            test_metrics = ["dice_score", "sensitivity", "specificity", "hausdorff95"]
        self.test_metric_names = test_metrics

        self.train_metrics = torch.nn.ModuleList([])
        self.val_metrics = torch.nn.ModuleList([])
        self.test_metrics = torch.nn.ModuleList([])

        self.stage = None
        self.epochsCounter = 0

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup hook as defined by PyTorch Lightning. Called at the beginning of fit (train + validate), validate, test,
            or predict.

        Args:
            stage(string, optional): Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.stage = stage

        if stage == "fit":
            train_slices_per_image = self.train_dataloader().dataset.slices_per_image()

            train_average_metrics = CombinedPerEpochMetric(
                metrics=self.train_metric_names,
                confidence_levels=self.train_metric_confidence_levels,
                image_ids=self.train_dataloader().dataset.image_ids(),
                reduction="mean",
                metrics_to_aggregate=[],
                slices_per_image=train_slices_per_image,
            )
            self.train_metrics = torch.nn.ModuleList([train_average_metrics])

            val_slices_per_image = self.val_dataloader().dataset.slices_per_image()

            val_average_metrics = CombinedPerEpochMetric(
                metrics=self.train_metric_names,
                confidence_levels=self.train_metric_confidence_levels,
                image_ids=self.val_dataloader().dataset.image_ids(),
                reduction="mean",
                metrics_to_aggregate=[],
                slices_per_image=val_slices_per_image,
            )
            self.val_metrics = torch.nn.ModuleList([val_average_metrics])

        if stage in ["validate"]:
            slices_per_image = self.val_dataloader().dataset.slices_per_image()

            val_average_metrics = CombinedPerEpochMetric(
                metrics=self.test_metric_names,
                confidence_levels=self.test_metric_confidence_levels,
                image_ids=self.val_dataloader().dataset.image_ids(),
                reduction="mean",
                metrics_to_aggregate=[],
                slices_per_image=slices_per_image,
            )
            self.val_metrics = torch.nn.ModuleList([val_average_metrics])
        if stage == "test":
            slices_per_image = self.test_dataloader().dataset.slices_per_image()

            test_average_metrics = CombinedPerEpochMetric(
                metrics=self.test_metric_names,
                confidence_levels=self.test_metric_confidence_levels,
                image_ids=self.test_dataloader().dataset.image_ids(),
                reduction="mean",
                metrics_to_aggregate=[],
                slices_per_image=slices_per_image,
            )
            self.test_metrics = torch.nn.ModuleList([test_average_metrics])

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

    @abstractmethod
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Validates the model on a given batch of model inputs.
        # this method should match the requirements of the pytorch lightning framework.
        # see https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        Args:
            batch: A batch of model inputs.
            batch_idx: Index of the current batch.
        """

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        """
        Compute the model's predictions on a given batch of model inputs.
        # this method should match the requirements of the pytorch lightning framework.
        # see https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html

        Args:
            batch: A batch of model inputs.
            batch_idx: Index of the current batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple val dataloaders used)
        """

        return self.predict(batch)

    @abstractmethod
    def test_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        """
        Compute the model's predictions on a given batch of model inputs from the test set.

        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_id: The index of the dataloader that produced this batch.
                (only if multiple test dataloaders used).
        """

    @abstractmethod
    def input_dimensionality(self) -> int:
        """
        The dimensionality of the input. Usually 2 or 3.
        """

    def configure_optimizers(
        self,
    ) -> Union[List[Optimizer], Tuple[List[Optimizer], List[LRSchedulerDict]]]:
        """
        This method is called by the PyTorch lightning framework before starting model training.

        Returns:
            The optimizer object as a list and optionally a learning rate scheduler object as a list.
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
                "scheduler": ReduceLROnPlateau(opt),
                "monitor": "validation/loss",
            }
        elif self.lr_scheduler == "cosineAnnealingLR":
            scheduler = {
                "scheduler": CosineAnnealingLR(opt, T_max=50),
                "monitor": "validation/loss",
            }

        if scheduler is not None:
            return [opt], [scheduler]

        return [opt]

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
        with torch.no_grad():
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

    def get_test_metrics(self) -> Iterable[torchmetrics.Metric]:
        """
        Returns:
            A list of metrics to be updated in each testing step.
        """

        return self.test_metrics

    def training_epoch_end(self, outputs: Any):
        """
        This method is called by the Pytorch Lightning framework at the end of each training epoch.

        Args:
            outputs: List of return values of all training steps of the current training epoch.
        """
        for train_metric in self.train_metrics:
            train_metrics = train_metric.compute()
            for metric_name, metric_value in train_metrics.items():
                self.log_dict(
                    {
                        f"train/{metric_name}": metric_value,
                        "train/epochsCounter": self.epochsCounter,
                    }
                )
            train_metric.reset()

        self.epochsCounter += 1

    def validation_epoch_end(self, outputs: Any):
        """
        This method is called by the Pytorch Lightning framework at the end of each validation epoch.

        Args:
            outputs: List of return values of all validation steps of the current validation epoch.
        """

        if self.stage == "validate":
            stage_name = "best_model_val"
        else:
            stage_name = "val"

        for val_metric in self.val_metrics:
            val_metrics = val_metric.compute()
            for metric_name, metric_value in val_metrics.items():
                self.log(f"{stage_name}/{metric_name}", metric_value)
            val_metric.reset()

    def test_epoch_end(self, outputs: Any) -> None:
        """
        This method is called by the Pytorch Lightning framework at the end of each testing epoch.

        Args:
            outputs: List of return values of all validation steps of the current testing epoch.
        """

        if self.stage == "test":
            stage_name = "best_model_test"
        else:
            stage_name = "test"

        for test_metric in self.test_metrics:
            test_metrics = test_metric.compute()
            for metric_name, metric_value in test_metrics.items():
                self.log(f"{stage_name}/{metric_name}", metric_value)
            self.logger.log_metrics({stage_name: test_metrics})
            test_metrics.reset()
