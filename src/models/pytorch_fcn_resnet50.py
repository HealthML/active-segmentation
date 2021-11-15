import torch
from torch import nn
from torchvision import models

from .pytorch_model import PytorchModel


class PytorchFCNResnet50(PytorchModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = models.segmentation.fcn_resnet50(
            pretrained=True,
            progress=True,
            num_classes=21)

    # wrap model interface
    def eval(self):
        return self.model.eval()

    def train(self, **kwargs):
        return self.model.train()

    def parameters(self, **kwargs):
        return self.model.parameters()

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        x, y = batch

        logits = self(x)["out"]
        loss = self.loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)["out"]

        # ToDo: this method should return the required performance metrics
