import torch

from .pytorch_model import PytorchModel
from .u_net import UNet


class PytorchUNet(PytorchModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = UNet(in_channels=1, out_channels=1, init_features=32)

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

        probabilities = self(x)
        loss = self.loss(probabilities, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)

        # ToDo: this method should return the required performance metrics
