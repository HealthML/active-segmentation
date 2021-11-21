"""U-Net architecture wrapped as PytorchModel"""

from typing import Any, Iterable

import torch

from metric_tracking import MetricPerCaseTracker
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

        self.confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.confidence_level_names = [
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
        ]

        self.train_average_metrics = MetricPerCaseTracker(
            metrics=["dice", "sensitivity", "specificity", "hausdorff95"],
            reduce="mean",
            groups=self.confidence_level_names,
            device=self.device,
        )

        self.train_metrics_per_case = MetricPerCaseTracker(
            metrics=["dice", "sensitivity", "specificity", "hausdorff95"],
            reduce="none",
            groups=self.confidence_level_names,
            device=self.device,
        )

        self.val_average_metrics = MetricPerCaseTracker(
            metrics=["dice", "sensitivity", "specificity", "hausdorff95"],
            reduce="mean",
            groups=self.confidence_level_names,
            device=self.device,
        )

        self.val_metrics_per_case = MetricPerCaseTracker(
            metrics=["dice", "sensitivity", "specificity", "hausdorff95"],
            reduce="none",
            groups=self.confidence_level_names,
            device=self.device,
        )

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

        self.train_average_metrics.to(self.device)
        self.train_metrics_per_case.to(self.device)

        # pylint: disable-msg=unused-variable
        x, y, case_ids = batch

        probabilities = self(x)
        loss = self.loss(probabilities, y)

        # ToDo: log metrics for different confidence levels

        for idx, confidence_level in enumerate(self.confidence_levels):

            predicted_mask = (probabilities > confidence_level).int()

            self.train_average_metrics.update(
                predicted_mask, y, case_ids, group_name=self.confidence_level_names[idx]
            )
            self.train_metrics_per_case.update(
                predicted_mask, y, case_ids, group_name=self.confidence_level_names[idx]
            )

        # ToDo: compute metrics on epoch end and log them to WandB

        self.log("train/loss", loss)  # log train loss via weights&biases
        return loss

    def training_epoch_end(self, training_step_outputs: Any):
        """
        This method is called by the Pytorch Lightning framework at the end of each training epoch.

        Args:
            training_step_outputs: List of return values of all training steps of the current training epoch. 
        """

        for idx, confidence_level in enumerate(self.confidence_levels):
            average_metrics = self.train_average_metrics.compute(
                group_name=self.confidence_level_names[idx]
            )
            metrics_per_case = self.train_metrics_per_case.compute(
                group_name=self.confidence_level_names[idx]
            )

            for metric_name, metric_value in average_metrics.items():
                self.log(f"train/mean_{metric_name}_{confidence_level}", metric_value)

            for metric, metric in metrics_per_case.items():
                for case_id, metric_value in metric.items():
                    self.log(
                        f"train/case_{case_id}_{metric_name}_{confidence_level}",
                        metric_value,
                    )

    def validation_step(self, batch, batch_idx) -> None:
        """
        Validates the model on a given batch of input images.

        Args:
            batch (Tensor): Batch of validation images.
            batch_idx: Index of the validation batch.
        """

        self.val_average_metrics.to(self.device)
        self.val_metrics_per_case.to(self.device)

        x, y, case_ids = batch

        probabilities = self(x)

        loss = self.loss(probabilities, y)
        self.log("validation/loss", loss)  # log validation loss via weights&biases

        for idx, confidence_level in enumerate(self.confidence_levels):

            predicted_mask = (probabilities > confidence_level).int()

            self.val_average_metrics.update(
                predicted_mask, y, case_ids, group_name=self.confidence_level_names[idx]
            )
            self.val_metrics_per_case.update(
                predicted_mask, y, case_ids, group_name=self.confidence_level_names[idx]
            )

    def validation_epoch_end(self, validation_step_outputs: Any):
        """
        This method is called by the Pytorch Lightning framework at the end of each validation epoch.

        Args:
            validation_step_outputs: List of return values of all validation steps of the current validation epoch. 
        """

        for idx, confidence_level in enumerate(self.confidence_levels):
            average_metrics = self.val_average_metrics.compute(
                group_name=self.confidence_level_names[idx]
            )
            metrics_per_case = self.val_metrics_per_case.compute(
                group_name=self.confidence_level_names[idx]
            )

            for metric_name, metric_value in average_metrics.items():
                self.log(f"val/mean_{metric_name}_{confidence_level}", metric_value)

            for metric_name, metric in metrics_per_case.items():
                for case_id, metric_value in metric.items():
                    self.log(
                        f"val/case_{case_id}_{metric_name}_{confidence_level}",
                        metric_value,
                    )
