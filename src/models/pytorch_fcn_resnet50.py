""" Resnet 50 from pytorch """
import torch
from torchvision import models

from .pytorch_model import PytorchModel


class PytorchFCNResnet50(PytorchModel):
    """
    Resnet50 model class for segmentation tasks.
    Documentation: http://pytorch.org/vision/master/generated/torchvision.models.segmentation.fcn_resnet50.html
    Details about the Resnet50 architecture: https://arxiv.org/pdf/1512.03385.pdf
    Args:
        **kwargs: Further, model specific parameters.
    """

    # pylint: disable=unused-argument,unused-variable,too-many-ancestors,arguments-differ
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.model = models.segmentation.fcn_resnet50(
            pretrained=True, progress=True, num_classes=21
        )

    # wrap model interface
    def eval(self):
        """Evaluates the model."""
        return self.model.eval()

    def train(self, **kwargs):
        """Trains the model."""
        return self.model.train()

    def parameters(self, **kwargs):
        """Get model parameters."""
        return self.model.parameters()

    def forward(self, x: torch.Tensor):
        """Execute one forward pass."""
        return self.model.forward(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Execute one training step."""
        x, y = batch

        logits = self(x)["out"]
        loss = self.loss(logits, y)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Execute one validation step."""
        x, y = batch

        logits = self(x)["out"]

        # ToDo: this method should return the required performance metrics
