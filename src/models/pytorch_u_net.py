"""U-Net architecture wrapped as PytorchModel"""

from typing import Iterable

import torch
import numpy as np

from .pytorch_model import PytorchModel
from .u_net import UNet

# pylint: disable-msg=too-many-ancestors, abstract-method
class PytorchUNet(PytorchModel):
    """
    U-Net architecture wrapped as PytorchModel.
    Details about the architecture can be found in `this paper <https://arxiv.org/pdf/1505.04597.pdf>`_.

    Args:
        num_levels (int, optional): Number levels (encoder and decoder blocks) in the U-Net. Defaults to 4.
        dim (int, optional): The dimensionality of the U-Net. Defaults to 2.
        in_channels (int, optional): Number of input channels. Defaults to 1
        out_channels (int): Number of output channels. Should be equal to the number of classes (for
            multi-label segmentation tasks excluding the background class). Defaults to 2.
        multi_label (bool, optional): Whether the model should produce single-label or multi-label outputs. If set to
            `False`, the model's predictions are computed using a Softmax activation layer. to If set to `True`, sigmoid
            activation layers are used to compute the model's predicitions. Defaults to False.
        p_dropout (float, optional): Probability of applying dropout to the outputs of the encoder layers. Defaults to
            0.
        **kwargs: Further, dataset specific parameters.
    """

    def __init__(
        self,
        num_levels: int = 4,
        dim: int = 2,
        in_channels=1,
        out_channels=2,
        multi_label=False,
        p_dropout: float = 0,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_levels = num_levels
        self.in_channels = in_channels
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.multi_label = multi_label
        self.p_dropout = p_dropout

        self.model = UNet(
            in_channels=self.in_channels,
            out_channels=out_channels,
            multi_label=multi_label,
            init_features=32,
            num_levels=self.num_levels,
            dim=self.dim,
            p_dropout=self.p_dropout,
        )

    def input_dimensionality(self) -> int:
        return self.dim

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

        x, y, is_pseudo_label, case_ids = batch

        weight_pseudo_labels = self.loss_weight_pseudo_labels

        if weight_pseudo_labels is not None:
            weight = torch.ones(len(is_pseudo_label), device=self.device)
            weight[is_pseudo_label] = weight_pseudo_labels
        else:
            weight = None

        probabilities = self(x)
        loss = self.loss_module(probabilities, y, weight=weight)

        for train_metric in self.get_train_metrics():
            train_metric.update(probabilities.detach(), y, case_ids)

        self.logger.log_metrics(
            {
                "train/loss": loss.detach(),
                "trainer/iteration": self.iteration,
                "trainer/epoch": self.current_epoch,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx) -> float:
        """
        Validates the model on a given batch of input images.

        Args:
            batch (Tensor): Batch of validation images.
            batch_idx: Index of the validation batch.

        Returns:
            Loss on the validation batch.
        """

        x, y, _, case_ids = batch

        probabilities = self(x)

        loss = self.loss_module(probabilities, y)

        if self.stage == "fit":
            # log to trainer for model selection
            self.log(
                "val/loss",
                loss.detach(),
                logger=False,
                on_epoch=True,
                on_step=False,
            )

        for val_metric in self.get_val_metrics():
            val_metric.update(probabilities.detach(), y, case_ids)

        return loss

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

        x, y, _, case_ids = batch

        probabilities = self(x)

        loss = self.loss_module(probabilities, y)
        self.logger.log_metrics(
            {
                "test/loss": loss.detach(),
                "trainer/iteration": self.iteration,
                "trainer/epoch": self.current_epoch,
            }
        )

        for test_metric in self.get_test_metrics():
            test_metric.update(probabilities.detach(), y, case_ids)

    def reset_parameters(self) -> None:
        """
        This method is called when resetting the weights is activated for the active learing loop
        """

        self.model = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            multi_label=self.multi_label,
            init_features=32,
            num_levels=self.num_levels,
            dim=self.dim,
            p_dropout=self.p_dropout,
        )
