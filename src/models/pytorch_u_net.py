"""U-Net architecture wrapped as PytorchModel"""

from typing import Iterable, Optional, Tuple

import torch
import numpy as np

from .pytorch_model import PytorchModel
from .u_net import UNet

# pylint: disable-msg=too-many-ancestors, abstract-method
class PytorchUNet(PytorchModel):
    """
    U-Net architecture wrapped as PytorchModel.
    Details about the architecture: https://arxiv.org/pdf/1505.04597.pdf
    Args:
        num_levels (int, optional): Number levels (encoder and decoder blocks) in the U-Net. Defaults to 4.
        input_shape (Tuple[int], optional): The input shape of the U-Net. Defaults to (240, 240).
        **kwargs: Further, model specific parameters.
    """

    def __init__(
        self, num_levels: int = 4, input_shape: Tuple[int] = (240, 240), **kwargs
    ):

        super().__init__(**kwargs)

        self.num_levels = num_levels
        self.input_shape = input_shape

        self.model = None

    def input_dimensionality(self) -> int:
        return len(self.input_shape)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup hook as defined by PyTorch Lightning. Called at the beginning of fit (train + validate), validate, test,
            or predict.

        Args:
            stage(string, optional): Either 'fit', 'validate', 'test', or 'predict'.
        """

        super().setup(stage)

        if stage == "fit":
            dataset = self.train_dataloader().dataset
        elif stage == "validate":
            dataset = self.val_dataloader().dataset
        elif stage == "test":
            dataset = self.test_dataloader().dataset

        multi_label = dataset.multi_label()
        num_classes = len(dataset.id_to_class_names())

        self.model = UNet(
            in_channels=1,
            out_channels=num_classes,
            multi_label=multi_label,
            init_features=32,
            num_levels=self.num_levels,
            input_shape=self.input_shape,
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

        x, y, case_ids = batch

        probabilities = self(x)
        loss = self.loss(probabilities, y)

        for train_metric in self.get_train_metrics():
            train_metric.update(probabilities, y, case_ids)

        self.log("train/loss", loss)  # log train loss via weights&biases
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """
        Validates the model on a given batch of input images.

        Args:
            batch (Tensor): Batch of validation images.
            batch_idx: Index of the validation batch.
        """

        x, y, case_ids = batch

        probabilities = self(x)

        loss = self.loss(probabilities, y)
        if self.stage == "fit":
            self.log("val/loss", loss)  # log validation loss via weights&biases

        for val_metric in self.get_val_metrics():
            val_metric.update(probabilities, y, case_ids)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> np.ndarray:
        """
        Uses the model to predict a given batch of input images.

        Args:
            batch (Tensor): Batch of prediction images.
            batch_idx: Index of the prediction batch.
            dataloader_idx: Index of the dataloader.
        """

        return self.predict(batch)

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Tests the model on a given batch of input images.

        Args:
            batch (Tensor): Batch of prediction images.
            batch_idx: Index of the prediction batch.
            dataloader_idx: Index of the dataloader.
        """

        x, y, case_ids = batch

        probabilities = self(x)

        loss = self.loss(probabilities, y)
        self.log("test/loss", loss)

        for test_metric in self.get_test_metrics():
            test_metric.update(probabilities, y, case_ids)
