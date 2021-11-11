from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from query_strategies import QueryStrategy
from datasets import ActiveLearningDataModule
from models import PytorchModel

wandb_logger = WandbLogger(project="active-segmentation", entity="active-segmentation")

class ActiveLearningPipeline:
    def __init__(self,
                 data_module: ActiveLearningDataModule,
                 model: PytorchModel,
                 strategy: QueryStrategy,
                 epochs: int,
                 gpus: int) -> None:
        self.data_module = data_module
        self.model = model
        # check sanity of validation step before running training loop:
        # self.model_trainer = Trainer(num_sanity_val_steps=-1, max_epochs=epochs, logger=wandb_logger, gpus=gpus)
        self.model_trainer = Trainer(profiler="simple", max_epochs=epochs, logger=wandb_logger, gpus=gpus)
        self.strategy = strategy
        self.epochs = epochs
        self.gpus = gpus

    def run(self) -> None:
        self.data_module.setup()

        items_to_label = self.strategy.select_items_to_label(self.model,
                                                             self.data_module.unlabeled_dataloader(),
                                                             self.data_module.unlabeled_set_size())

        self.data_module.label_items(items_to_label)

        self.model_trainer.fit(self.model, self.data_module)
