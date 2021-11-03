from pytorch_lightning import Trainer

from query_strategies import QueryStrategy
from datasets import ActiveLearningDataModule
from models import PytorchModel


class ActiveLearningPipeline:
    def __init__(self,
                 data_module: ActiveLearningDataModule,
                 model: PytorchModel,
                 strategy: QueryStrategy,
                 epochs: int) -> None:
        self.data_module = data_module
        self.model = model
        self.model_trainer = Trainer(deterministic=True, max_epochs=epochs)
        self.strategy = strategy
        self.epochs = epochs

    def run(self) -> None:
        self.data_module.setup()

        items_to_label = self.strategy.select_items_to_label(self.model,
                                                             self.data_module.unlabeled_dataloader(),
                                                             self.data_module.unlabeled_set_size())

        self.data_module.label_items(items_to_label)

        self.model_trainer.fit(self.model, self.data_module)
