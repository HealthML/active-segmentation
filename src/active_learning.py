""" Module containing the active learning pipeline """

import os
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
        initial_epochs (int, optional): Number of epochs the initial model should be trained. Defaults to `epochs`.
        items_to_label (int, optional): Number of items that should be selected for labeling in the active learning run.
            (default = 1).
        iterations (int, optional): iteration times how often the active learning pipeline should be
        executed (default = 10).
        reset_weights (bool, optional): Enable/Disable resetting of weights after every active learning run
        epochs_increase_per_query (int, optional): Increase number of epochs for every query to compensate for
            the increased training dataset size (default = 0).
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
        initial_epochs: Optional[int] = None,
        items_to_label: int = 1,
        iterations: int = 10,
        reset_weights: bool = False,
        epochs_increase_per_query: int = 0,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        early_stopping: bool = False,
        lr_scheduler: str = None,
        model_selection_criterion="loss",
    ) -> None:

        self.data_module = data_module
        self.model = model
        self.model_trainer = None
        # log gradients, parameter histogram and model topology
        logger.watch(self.model, log="all")

        self.strategy = strategy
        self.epochs = epochs
        self.logger = logger
        self.gpus = gpus
        self.active_learning_mode = active_learning_mode
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping = early_stopping
        self.initial_epochs = initial_epochs if initial_epochs is not None else epochs
        self.items_to_label = items_to_label
        self.iterations = iterations
        self.lr_scheduler = lr_scheduler
        self.model_selection_criterion = model_selection_criterion
        self.reset_weights = reset_weights
        self.epochs_increase_per_query = epochs_increase_per_query

    def run(self) -> None:
        """Run the pipeline"""
        self.data_module.setup()

        # pylint: disable=too-many-nested-blocks

        if self.active_learning_mode:
            self.model_trainer = self.setup_trainer(self.initial_epochs, iteration=0)
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

                    self.model_trainer = self.setup_trainer(
                        self.epochs, iteration=iteration
                    )

                # optionally reset weights after fitting on new data
                if self.reset_weights:
                    self.model.reset_parameters()

                self.model.start_epoch = self.model.current_epoch
                self.model.iteration = iteration

                # train model on labeled batch
                self.model_trainer.fit(self.model, self.data_module)

                # compute metrics for the best model on the validation set
                self.model_trainer.validate(
                    ckpt_path="best", dataloaders=self.data_module
                )

        else:
            self.model_trainer = self.setup_trainer(self.epochs, iteration=0)
            # run regular fit run with all the data if no active learning mode
            self.model_trainer.fit(self.model, self.data_module)

            # compute metrics for the best model on the validation set
            self.model_trainer.validate(ckpt_path="best", dataloaders=self.data_module)

    def setup_trainer(self, epochs: int, iteration: Optional[int] = None) -> Trainer:
        """
        Initializes a new Pytorch Lightning trainer object.

        Args:
            epochs (int): Number of training epochs.
            iteration (Optional[int], optional): Current active learning iteration. Defaults to None.

        Returns:
            pytorch_lightning.Trainer: A trainer object.
        """

        callbacks = []
        if self.lr_scheduler is not None:
            callbacks.append(LearningRateMonitor(logging_interval="step"))
        if self.early_stopping:
            callbacks.append(EarlyStopping("validation/loss"))

        monitoring_mode = "min" if "loss" in self.model_selection_criterion else "max"

        if self.checkpoint_dir is not None and iteration is not None:
            checkpoint_dir = os.path.join(self.checkpoint_dir, str(iteration))
        else:
            checkpoint_dir = self.checkpoint_dir

        num_sanity_val_steps = 2 if iteration is None or iteration == 0 else 0

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model_epoch_{epoch}",
            auto_insert_metric_name=False,
            monitor=f"val/{self.model_selection_criterion}",
            mode=monitoring_mode,
            save_last=True,
            every_n_epochs=1,
            every_n_train_steps=0,
            save_on_train_epoch_end=False,
        )

        callbacks.append(checkpoint_callback)

        return Trainer(
            deterministic=False,
            profiler="simple",
            max_epochs=epochs + iteration * self.epochs_increase_per_query
            if iteration is not None
            else epochs,
            logger=self.logger,
            log_every_n_steps=20,
            gpus=self.gpus,
            benchmark=True,
            callbacks=callbacks,
            num_sanity_val_steps=num_sanity_val_steps,
        )
