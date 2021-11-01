from pytorch_lightning import Trainer
from tensorflow.keras.utils import Sequence
from torch.utils.data import DataLoader
from typing import Union

from models import PytorchModel, TensorflowModel


class ModelTrainer:
    def fit(self,
            model: Union[PytorchModel, TensorflowModel],
            train_dataloader: Union[DataLoader, Sequence],
            val_dataloader: Union[DataLoader, Sequence],
            epochs: int) -> None:
        if isinstance(model, PytorchModel):
            pytorch_trainer = Trainer(deterministic=True, max_epochs=epochs)
            pytorch_trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        elif isinstance(model, TensorflowModel):
            model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs, initial_epoch=0)
        else:
            raise ValueError("Model must be either an instance of PytorchModel or of TensorflowModel.")
