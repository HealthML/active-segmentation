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
    """

    # pylint: disable=too-few-public-methods,too-many-arguments
    def __init__(
        self,
        data_module: ActiveLearningDataModule,
        model: PytorchModel,
        strategy: QueryStrategy,
        epochs: int,
        gpus: int,
        checkpoint_dir: Optional[str] = None,
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

        # compute metrics for the best model on the validation set
        self.model_trainer.validate()
