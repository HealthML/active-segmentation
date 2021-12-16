""" Module containing the active learning pipeline """

from typing import Iterable, Union

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from query_strategies import QueryStrategy
from datasets import ActiveLearningDataModule
from models import PytorchModel


class ActiveLearningPipeline:
    """
    The pipeline or simulation environment to run active learning experiments.
    Args:
        data_module (ActiveLearningDataModule): A data module object providing data.
        model (PytorchModel): A model object with architecture able to be fitted with the data.
        strategy (QueryStrategy): An active learning strategy to query for new labels.
        epochs (int): The number of epochs the model should be trained.
        gpus (int): Number of GPUS to use for model training.
        early_stopping (bool, optional): Enable/Disable Early stopping when model
            is not learning anymore (default = False).
        logger: A logger object as defined by Pytorch Lightning.
        lr_scheduler (string, optional): Algorithm used for dynamically updating the
            learning rate during training. E.g. 'reduceLROnPlateau' or 'cosineAnnealingLR'
        active_learning_mode (bool, optional): Enable/Disabled Active Learning Pipeline (default = False).
        number_of_items: Number of items that should be selected for labeling in the active learning run.
            (default = 1).
        iterations: iteration times how often the active learning pipeline should be executed (default = 10).
    """

    # pylint: disable=too-few-public-methods,too-many-arguments
    def __init__(
        self,
        data_module: ActiveLearningDataModule,
        model: PytorchModel,
        strategy: QueryStrategy,
        epochs: int,
        gpus: int,
        active_learning_mode: bool = False,
        number_of_items: int = 1,
        iterations: int = 10,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        early_stopping: bool = False,
        lr_scheduler: str = None,
    ) -> None:

        self.data_module = data_module
        self.model = model
        # log gradients, parameter histogram and model topology
        logger.watch(self.model, log="all")

        callbacks = []
        if lr_scheduler is not None:
            callbacks.append(LearningRateMonitor(logging_interval="step"))
        if early_stopping:
            callbacks.append(EarlyStopping("validation/loss"))

        self.model_trainer = Trainer(
            # deterministic=True,
            profiler="simple",
            max_epochs=epochs,
            logger=logger,
            gpus=gpus,
            benchmark=True,
            callbacks=callbacks,
        )
        self.strategy = strategy
        self.epochs = epochs
        self.gpus = gpus
        self.active_learning_mode = active_learning_mode
        self.number_of_items = number_of_items
        self.iterations = iterations

    def run(self) -> None:
        """Run the pipeline"""
        self.data_module.setup()

        if self.active_learning_mode:
            for i in range(0, self.iterations):
                # query batch selection
                items_to_label = self.strategy.select_items_to_label(
                    self.model, self.data_module, self.number_of_items
                )
                # label batch
                self.data_module.label_items(items_to_label)

                # train model on labeled batch
                self.model_trainer.fit(self.model, self.data_module)
        else:
            self.model_trainer.fit(self.model, self.data_module)
