""" Module containing the active learning pipeline """

from typing import Iterable, Optional, Union

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

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
        checkpoint_dir (str, optional): Directory where the model checkpoints are to be saved.
        early_stopping (bool, optional): Enable/Disable Early stopping when model
            is not learning anymore (default = False).
        logger: A logger object as defined by Pytorch Lightning.
        lr_scheduler (string, optional): Algorithm used for dynamically updating the
            learning rate during training. E.g. 'reduceLROnPlateau' or 'cosineAnnealingLR'
        active_learning_mode (bool, optional): Enable/Disabled Active Learning Pipeline (default = False).
        items_to_label (int, optional): Number of items that should be selected for labeling in the active learning run.
            (default = 1).
        iterations (int, optional): iteration times how often the active learning pipeline should be
        executed (default = 10).
    """

    # pylint: disable=too-few-public-methods,too-many-arguments,too-many-instance-attributes, too-many-locals
    def __init__(
        self,
        data_module: ActiveLearningDataModule,
        model: PytorchModel,
        strategy: QueryStrategy,
        epochs: int,
        gpus: int,
        checkpoint_dir: Optional[str] = None,
        active_learning_mode: bool = False,
        items_to_label: int = 1,
        iterations: int = 10,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        early_stopping: bool = False,
        lr_scheduler: str = None,
        model_selection_criterion="loss",
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

        monitoring_mode = "min" if "loss" in model_selection_criterion else "max"

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model_epoch_{epoch}",
            auto_insert_metric_name=False,
            monitor=f"val/{model_selection_criterion}",
            mode=monitoring_mode,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
        callbacks.append(self.checkpoint_callback)

        self.model_trainer = Trainer(
            deterministic=True,
            profiler="simple",
            max_epochs=epochs,
            logger=logger,
            gpus=gpus,
            benchmark=True,
            callbacks=callbacks,
        )
        self.strategy = strategy
        self.epochs = epochs
        self.logger = logger
        self.gpus = gpus
        self.active_learning_mode = active_learning_mode
        self.items_to_label = items_to_label
        self.iterations = iterations
        self.callbacks = callbacks

    def run(self) -> None:
        """Run the pipeline"""
        self.data_module.setup()

        if self.active_learning_mode:
            # run pipeline
            for iteration in range(0, self.iterations):
                # skip labeling in the first iteration because the model hasn't trained yet
                if iteration != 0:
                    # query batch selection
                    items_to_label = self.strategy.select_items_to_label(
                        self.model, self.data_module, self.items_to_label
                    )
                    # label batch
                    self.data_module.label_items(items_to_label)

                # train model on labeled batch
                self.model_trainer.fit(self.model, self.data_module)

                # don't reset the model trainer in the last iteration
                if iteration != self.iterations - 1:
                    # reset model trainer
                    self.model_trainer = Trainer(
                        deterministic=True,
                        profiler="simple",
                        max_epochs=self.epochs,
                        logger=self.logger,
                        gpus=self.gpus,
                        benchmark=True,
                        callbacks=self.callbacks,
                    )
        else:
            # run regular fit run with all the data if no active learning mode
            self.model_trainer.fit(self.model, self.data_module)

        # compute metrics for the best model on the validation set
        self.model_trainer.validate()
