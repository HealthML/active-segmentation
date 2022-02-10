""" Base classes to implement models with pytorch """
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import numpy
import torch
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import wandb

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
        learning_rate (float, optional): The step size at each iteration while moving towards a minimum of the loss.
            Defaults to `0.0001`.
        lr_scheduler (string, optional): Algorithm used for dynamically updating the learning rate during training:
            ``"reduceLROnPlateau"`` | ``"cosineAnnealingLR"``. Defaults to using no scheduler.
        loss (string, optional): The optimization criterion: ``"cross_entropy"`` | ``"dice"`` |``"cross_entropy_dice"``
            | ``"general_dice"`` | ``"fp"`` | ``"fp_dice"``. Defaults to ``"cross_entropy"``.
        optimizer (string, optional): Algorithm used to calculate the loss and update the weights: ``"adam"`` |
            ``"sgd"``. Defaults to ``"adam"``.
        train_metrics (Iterable[str], optional): A list with the names of the metrics that should be computed and logged
            in each training and validation epoch of the training loop. Available options: ``"dice_score"`` |
            ``"sensitivity"`` | ``"specificity"`` | ``"hausdorff95"``. Defaults to `["dice_score"]`.
        train_metric_confidence_levels (Iterable[float], optional): A list of confidence levels for which the metrics
            specified in the `train_metrics` parameter should be computed in the training loop (`trainer.fit()`). This
            parameter is used only for mulit-label classification tasks. Defaults to `[0.5]`.
        test_metrics (Iterable[str], optional): A list with the names of the metrics that should be computed and logged
            in the model validation or testing loop (`trainer.validate()`, `trainer.test()`). Available options:
            ``"dice_score"`` | ``"sensitivity"`` | ``"specificity"`` | ``"hausdorff95"`` Defaults to
            `["dice_score", "sensitivity", "specificity", "hausdorff95"]`.
        test_metric_confidence_levels (Iterable[float], optional): A list of confidence levels for which the metrics
            specified in the `test_metrics` parameter should be computed in the validation or testing loop. This
            parameter is used only for mulit-label classification tasks. Defaults to `[0.5]`.
        **kwargs: Further, dataset specific parameters.
    """

    # pylint: disable=too-many-ancestors,arguments-differ,too-many-arguments

    def __init__(
        self,
        learning_rate: float = 0.0001,
        lr_scheduler: Optional[
            Literal["reduceLROnPlateau", "cosineAnnealingLR"]
        ] = None,
        loss: Literal[
            "cross_entropy",
            "dice",
            "cross_entropy_dice",
            "general_dice",
            "fp",
            "fp_dice",
        ] = "cross_entropy",
        optimizer: Literal["adam", "sgd"] = "adam",
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
        self.loss = loss
        self.loss_module = None

        self.train_metric_confidence_levels = train_metric_confidence_levels
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
        self.start_global_step = 0
        self.start_epoch = 0
        self.iteration = 0

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup hook as defined by PyTorch Lightning. Called at the beginning of fit (train + validate), validate, test,
            or predict.

        Args:
            stage(string, optional): Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.stage = stage

        self.configure_metrics()

        if stage in ["fit", "validate", "test"]:
            self.loss_module = self.configure_loss(
                self.loss, self.trainer.datamodule.multi_label()
            )

    @property
    def global_step(self) -> int:
        return self.start_global_step + super().global_step

    @property
    def current_epoch(self) -> int:
        return self.start_epoch + super().current_epoch

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
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """
        Validates the model on a given batch of model inputs.
        # this method should match the requirements of the pytorch lightning framework.
        # see https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        Args:
            batch: A batch of model inputs.
            batch_idx: Index of the current batch.

        Returns:
            Validation loss.
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
                "monitor": "val/loss",
            }
        elif self.lr_scheduler == "cosineAnnealingLR":
            scheduler = {
                "scheduler": CosineAnnealingLR(opt, T_max=50),
                "monitor": "val/loss",
            }

        if scheduler is not None:
            return [opt], [scheduler]

        return [opt]

    @staticmethod
    def configure_loss(
        loss: Literal[
            "cross_entropy",
            "dice",
            "cross_entropy_dice",
            "general_dice",
            "fp",
            "fp_dice",
        ],
        multi_label: bool = False,
    ) -> functional.losses.SegmentationLoss:
        """
        Configures the loss.
        Args:
            loss (string, optional): The optimization criterion: ``"cross_entropy"`` | ``"dice"`` |
                ``"cross_entropy_dice"`` | ``"general_dice"`` | ``"fp"`` | ``"fp_dice"``. Defaults to
                ``"cross_entropy"``.
            multi_label (bool, optional): Determines if data is multilabel or not (default = `False`).

        Returns:
            The loss object.
        """

        loss_kwargs = {
            "ignore_index": -1,
        }

        if loss == "cross_entropy":
            return functional.CrossEntropyLoss(multi_label=multi_label, **loss_kwargs)
        if loss == "cross_entropy_dice":
            return functional.CrossEntropyDiceLoss(
                multi_label=multi_label, include_background=multi_label, **loss_kwargs
            )
        if loss == "dice":
            return functional.DiceLoss(include_background=multi_label, **loss_kwargs)
        if loss == "general_dice":
            return functional.GeneralizedDiceLoss(
                include_background=multi_label, **loss_kwargs
            )
        if loss == "fp":
            return functional.FalsePositiveLoss(
                include_background=multi_label, **loss_kwargs
            )
        if loss == "fp_dice":
            return functional.FalsePositiveDiceLoss(
                include_background=multi_label, **loss_kwargs
            )
        raise ValueError("Invalid loss name.")

    def configure_metrics(self) -> None:
        """
        Configures metrics.
        """

        # define metrics in Weights and Biases
        # see https://docs.wandb.ai/guides/track/log#customize-axes-and-summaries-with-define_metric
        wandb.define_metric("trainer/epoch")
        wandb.define_metric("trainer/iteration")
        wandb.define_metric("train/loss", step_metric="step")
        wandb.define_metric("train/mean_loss", step_metric="trainer/epoch")
        wandb.define_metric("val/mean_loss", step_metric="trainer/epoch")

        metric_kwargs = {
            "id_to_class_names": self.trainer.datamodule.id_to_class_names(),
            "multi_label": self.trainer.datamodule.multi_label(),
            "reduction_across_classes": "mean",
            "reduction_across_images": "mean",
        }

        if self.stage == "fit":
            training_set = self.trainer.datamodule.train_dataloader().dataset

            train_average_metrics = CombinedPerEpochMetric(
                metrics=self.train_metric_names,
                confidence_levels=self.train_metric_confidence_levels,
                image_ids=training_set.image_ids(),
                slices_per_image=training_set.slices_per_image(),
                stage="train",
                ignore_nan_in_reduction=True,
                **metric_kwargs,
            )
            self.train_metrics = torch.nn.ModuleList([train_average_metrics])

            validation_set = self.trainer.datamodule.val_dataloader().dataset

            val_average_metrics = CombinedPerEpochMetric(
                metrics=self.train_metric_names,
                confidence_levels=self.train_metric_confidence_levels,
                image_ids=validation_set.image_ids(),
                slices_per_image=validation_set.slices_per_image(),
                stage="val",
                ignore_nan_in_reduction=True,
                **metric_kwargs,
            )
            self.val_metrics = torch.nn.ModuleList([val_average_metrics])

            for metrics in [*self.train_metrics, *self.val_metrics]:
                for metric_name in metrics.get_metric_names():
                    wandb.define_metric(metric_name, step_metric="trainer/epoch")

        if self.stage == "validate":
            validation_set = self.trainer.datamodule.val_dataloader().dataset

            val_average_metrics = CombinedPerEpochMetric(
                metrics=self.test_metric_names,
                confidence_levels=self.test_metric_confidence_levels,
                image_ids=validation_set.image_ids(),
                slices_per_image=validation_set.slices_per_image(),
                stage="best_model_val",
                **metric_kwargs,
            )
            self.val_metrics = torch.nn.ModuleList([val_average_metrics])

            for metrics in self.val_metrics:
                for metric_name in metrics.get_metric_names():
                    wandb.define_metric(metric_name, step_metric="trainer/iteration")

        if self.stage == "test":
            test_set = self.trainer.datamodule.test_dataloader().dataset

            test_average_metrics = CombinedPerEpochMetric(
                metrics=self.test_metric_names,
                confidence_levels=self.test_metric_confidence_levels,
                image_ids=test_set.image_ids(),
                slices_per_image=test_set.slices_per_image(),
                stage="best_model_test",
                **metric_kwargs,
            )
            self.test_metrics = torch.nn.ModuleList([test_average_metrics])

            for metrics in self.test_metrics:
                for metric_name in metrics.get_metric_names():
                    wandb.define_metric(metric_name, step_metric="trainer/iteration")

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
            return self(batch)

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

    def training_epoch_end(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]]],
    ) -> None:
        """
        This method is called by the Pytorch Lightning framework at the end of each training epoch.

        Args:
            outputs: List of return values of all training steps of the current training epoch.
        """
        train_metrics = {}

        # collect loss values
        if isinstance(outputs, torch.Tensor):
            losses = outputs
        else:
            losses = torch.Tensor(
                [item["loss"] if isinstance(item, dict) else item for item in outputs]
            )

        for train_metric in self.train_metrics:
            train_metrics = {**train_metrics, **train_metric.compute()}
            train_metric.reset()

        self.logger.log_metrics(
            {
                **train_metrics,
                "trainer/epoch": self.current_epoch,
                "trainer/iteration": self.iteration,
                "train/mean_loss": losses.mean(),
            }
        )

    def validation_epoch_end(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]]],
    ):
        """
        This method is called by the Pytorch Lightning framework at the end of each validation epoch.

        Args:
            outputs: List of return values of all validation steps of the current validation epoch.
        """

        val_metrics = {
            "trainer/epoch": self.current_epoch,
            "trainer/iteration": self.iteration,
        }

        for val_metric in self.val_metrics:
            val_metrics = {**val_metrics, **val_metric.compute()}
            val_metric.reset()

        # collect loss values
        if isinstance(outputs, torch.Tensor):
            losses = outputs
        else:
            losses = torch.Tensor(
                [item["loss"] if isinstance(item, dict) else item for item in outputs]
            )

        val_metrics = {**val_metrics, "val/mean_loss": losses.mean()}

        if self.stage == "fit":
            # log to trainer to allow model selection
            self.log_dict(val_metrics, logger=False)

            if not self.trainer.sanity_checking:
                # log to Weights and Biases
                if self.stage == "fit":
                    self.logger.experiment.log(val_metrics, commit=False)
                else:
                    self.logger.log_metrics(val_metrics)
        else:
            self.logger.log_metrics(val_metrics)

    def test_epoch_end(self, outputs: Any) -> None:
        """
        This method is called by the Pytorch Lightning framework at the end of each testing epoch.

        Args:
            outputs: List of return values of all validation steps of the current testing epoch.
        """

        test_metrics = {
            "trainer/epoch": self.current_epoch,
            "trainer/iteration": self.iteration,
        }

        for test_metric in self.test_metrics:
            test_metrics = {**test_metrics, **test_metric.compute()}
            test_metrics.reset()

        self.logger.log_metrics(test_metrics)

    @abstractmethod
    def reset_parameters(self) -> None:
        """
        This method is called when resetting the weights is activated for the active learing loop
        """
