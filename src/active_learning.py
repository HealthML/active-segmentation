""" Module containing the active learning pipeline """
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from query_strategies import QueryStrategy
from datasets import ActiveLearningDataModule
from models import PytorchModel

wandb_logger = WandbLogger(project="active-segmentation", entity="active-segmentation")


class ActiveLearningPipeline:
    """
    The pipeline or simulation environment to run active learning experiments.
    Args:
        data_module: A data module object providing data.
        model: A model object with architecture able to be fitted with the data.
        strategy: An active learning strategy to query for new labels.
        epochs: The number of epochs the model should be trained.
    """

    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        data_module: ActiveLearningDataModule,
        model: PytorchModel,
        strategy: QueryStrategy,
        epochs: int,
        gpus: int,
    ) -> None:

        self.data_module = data_module
        self.model = model
        # log gradients, parameter histogram and model topology
        wandb_logger.watch(self.model, log="all")

        callbacks = [
            EarlyStopping("validation/loss"),
            LearningRateMonitor(logging_interval="step"),
        ]
        self.model_trainer = Trainer(
            deterministic=True,
            profiler="simple",
            max_epochs=epochs,
            logger=wandb_logger,
            gpus=gpus,
            benchmark=True,
            callbacks=callbacks,
        )
        self.strategy = strategy
        self.epochs = epochs
        self.gpus = gpus

    def run(self) -> None:
        """Run the pipeline"""
        self.data_module.setup()

        items_to_label = self.strategy.select_items_to_label(
            self.model,
            self.data_module.unlabeled_dataloader(),
            self.data_module.unlabeled_set_size(),
        )

        self.data_module.label_items(items_to_label)

        self.model_trainer.fit(self.model, self.data_module)
