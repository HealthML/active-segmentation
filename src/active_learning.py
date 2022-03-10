""" Module containing the active learning pipeline """

import math
import os
import shutil
from typing import Iterable, Optional, Union, Tuple, List

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import numpy as np
import wandb

from query_strategies import QueryStrategy
from datasets import ActiveLearningDataModule
from models import PytorchModel
from functional.interpretation import HeatMaps
from src.datasets.doubly_shuffled_nifti_dataset import get_image_slice_ids


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
            is not learning anymore. Defaults to False.
        logger: A logger object as defined by Pytorch Lightning.
        lr_scheduler (string, optional): Algorithm used for dynamically updating the
            learning rate during training. E.g. 'reduceLROnPlateau' or 'cosineAnnealingLR'
        active_learning_mode (bool, optional): Enable/Disabled Active Learning Pipeline. Defaults to False.
        initial_epochs (int, optional): Number of epochs the initial model should be trained. Defaults to `epochs`.
        items_to_label (int, optional): Number of items that should be selected for labeling in the active learning run.
            Defaults to 1.
        iterations (int, optional): iteration times how often the active learning pipeline should be
            executed. If None, the active learning pipeline is run until the whole dataset is labeled. Defaults to None.
        reset_weights (bool, optional): Enable/Disable resetting of weights after every active learning run
        epochs_increase_per_query (int, optional): Increase number of epochs for every query to compensate for
            the increased training dataset size. Defaults to 0.
        heatmaps_per_iteration (int, optional): Number of heatmaps that should be generated per iteration. Defaults to
            0.
        deterministic_mode (bool, optional): Whether only deterministic CUDA operations should be used. Defaults to
            `True`.
        save_model_every_epoch (bool, optional): Whether the model files of all epochs are to be saved or only the
            model file of the best epoch. Defaults to `False`.
        clear_wandb_cache (bool, optional): Whether the whole Weights and Biases cache should be deleted when the run
            is finished. Should only be used when no other runs are running in parallel. Defaults to False.
        **kwargs: Additional, strategy-specific parameters.
    """

    # pylint: disable=too-few-public-methods,too-many-arguments,too-many-instance-attributes,too-many-locals
    # pylint: disable=protected-access
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
        iterations: Optional[int] = None,
        reset_weights: bool = False,
        epochs_increase_per_query: int = 0,
        heatmaps_per_iteration: int = 0,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        early_stopping: bool = False,
        lr_scheduler: str = None,
        model_selection_criterion="loss",
        deterministic_mode: bool = True,
        save_model_every_epoch: bool = False,
        clear_wandb_cache: bool = False,
        **kwargs,
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
        self.heatmaps_per_iteration = heatmaps_per_iteration
        self.lr_scheduler = lr_scheduler
        self.model_selection_criterion = model_selection_criterion
        self.reset_weights = reset_weights
        self.epochs_increase_per_query = epochs_increase_per_query
        self.deterministic_mode = deterministic_mode
        self.save_model_every_epoch = save_model_every_epoch
        self.clear_wandb_cache = clear_wandb_cache
        self.kwargs = kwargs

    def run(self) -> None:
        """Run the pipeline"""
        self.data_module.setup()

        # pylint: disable=too-many-nested-blocks

        if self.active_learning_mode:
            self.model_trainer = self.setup_trainer(self.initial_epochs, iteration=0)

            if self.iterations is None:
                self.iterations = math.ceil(
                    self.data_module.unlabeled_set_size() / self.items_to_label
                )

            # run pipeline
            for iteration in range(0, self.iterations + 1):
                # skip labeling in the first iteration because the model hasn't trained yet
                if iteration != 0:
                    # query batch selection
                    if self.data_module.unlabeled_set_size() > 0:
                        (
                            items_to_label,
                            pseudo_labels,
                        ) = self.strategy.select_items_to_label(
                            self.model,
                            self.data_module,
                            self.items_to_label,
                            **self.kwargs,
                        )
                        # label batch
                        self.data_module.label_items(items_to_label, pseudo_labels)

                    # Log selected items to wandb table
                    self.__log_selected_items(items_to_label, pseudo_labels)

                    if self.heatmaps_per_iteration > 0:
                        # Get latest added items from dataset
                        items_to_inspect = (
                            self.data_module._training_set.get_images_by_id(
                                case_ids=items_to_label[: self.heatmaps_per_iteration],
                            )
                        )
                        # Generate heatmaps using final predictions and heatmaps
                        if len(items_to_inspect) > 0:
                            self.__generate_and_log_heatmaps(
                                items_to_inspect=items_to_inspect, iteration=iteration
                            )

                    self.model_trainer = self.setup_trainer(
                        self.epochs, iteration=iteration
                    )

                # optionally reset weights after fitting on new data
                if self.reset_weights and iteration != 0:
                    self.model.reset_parameters()

                self.model.start_epoch = self.model.current_epoch + 1

                self.model.iteration = iteration
                # train model on labeled batch
                self.model_trainer.fit(self.model, self.data_module)

                # compute metrics for the best model on the validation set
                self.model_trainer.validate(
                    ckpt_path="best", dataloaders=self.data_module
                )
                self.model.step_loss_weight_pseudo_labels_scheduler()

        else:
            self.model_trainer = self.setup_trainer(self.epochs, iteration=0)
            # run regular fit run with all the data if no active learning mode
            self.model_trainer.fit(self.model, self.data_module)

            # compute metrics for the best model on the validation set
            self.model_trainer.validate(ckpt_path="best", dataloaders=self.data_module)

        wandb.run.finish()
        if self.clear_wandb_cache:
            self.remove_wandb_cache()

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

        best_model_checkpoint_callback = ModelCheckpoint(
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

        callbacks.append(best_model_checkpoint_callback)

        if self.save_model_every_epoch:
            all_models_checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(checkpoint_dir, "all_models"),
                filename="epoch_{epoch}",
                auto_insert_metric_name=False,
                save_top_k=-1,
                every_n_epochs=1,
                every_n_train_steps=0,
                save_on_train_epoch_end=False,
            )

            callbacks.append(all_models_checkpoint_callback)

        # Pytorch lightning currently does not support deterministic 3d max pooling
        # therefore this option is only enabled for the 2d case
        # see https://pytorch.org/docs/stable/notes/randomness.html
        deterministic_mode = (
            self.deterministic_mode if self.model.input_dimensionality() == 2 else False
        )

        return Trainer(
            deterministic=deterministic_mode,
            benchmark=not self.deterministic_mode,
            profiler="simple",
            max_epochs=epochs + iteration * self.epochs_increase_per_query
            if iteration is not None
            else epochs,
            logger=self.logger,
            log_every_n_steps=20,
            gpus=self.gpus,
            callbacks=callbacks,
            num_sanity_val_steps=num_sanity_val_steps,
        )

    def __log_selected_items(self, selected_items, pseudo_labels):
        items = self.data_module._training_set.get_items_for_logging(selected_items)
        items = [(*i, False) for i in items]
        table = wandb.Table(columns=["case_id", "image_path", "image_id", "slice_index", "pseudo_label"], data=[[]])

        if pseudo_labels is not None:
            items = self.data_module._training_set.get_items_for_logging(pseudo_labels)
            items = [(*i, True) for i in items]
            table.add(*items)

        wandb.log("Selected Items", table)


    def __generate_and_log_heatmaps(
        self, items_to_inspect: List[Tuple[np.ndarray, str]], iteration: int
    ) -> None:
        """
        Generates heatmaps using gradient based method and the prediction of the last layer of the model.
        Args:
            items_to_inspect (List[Tuple[np.ndarray, str]]): A list with the items to generate heatmaps for.
            iteration (int): The iteration of the active learning loop.

        """

        # Generate heatmaps using final predictions and gradient based method
        gcam_images, logit_images = [], []
        for img, case_id in items_to_inspect:
            gcam_heatmap, logit_heatmap = self.__generate_heatmaps(
                img=img, case_id=case_id
            )
            gcam_images.append(
                wandb.Image(
                    gcam_heatmap,
                    caption=f"AL Iteration: {iteration}, Case ID: {case_id}",
                )
            )
            logit_images.append(
                wandb.Image(
                    logit_heatmap,
                    caption=f"AL Iteration: {iteration}, Case ID: {case_id}",
                )
            )
        wandb.log({"GradCam heatmaps": gcam_images})
        wandb.log({"Logit heatmaps": logit_images})

    def __generate_heatmaps(
        self,
        img: np.ndarray,
        case_id: str,
        target_category: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates two heatmaps: One based on the GradCam method and one based on the predictions of the last layer.
        Args:
            img (np.ndarray): The image as numpy array.
            case_id (str): The id of the current image.
            target_category (int, optional): The label of the target class to analyze.

        Returns:
            A tuple of both heatmaps. GradCamp heatmap, prediction heatmap.
        """

        input_tensor = torch.from_numpy(img)
        heatmap = HeatMaps(model=self.model)
        target_layers = [self.model.model.conv]
        gcam_gray = heatmap.generate_grayscale_cam(
            input_tensor=input_tensor,
            target_category=target_category,
            target_layers=target_layers,
        )
        logits_gray = heatmap.generate_grayscale_logits(
            input_tensor=input_tensor, target_category=target_category
        )
        gcam_img = heatmap.show_grayscale_heatmap_on_image(
            image=img, grayscale_heatmap=gcam_gray
        )
        logits_img = heatmap.show_grayscale_heatmap_on_image(
            image=img, grayscale_heatmap=logits_gray
        )
        print(f"Generated heatmaps for case {case_id}")
        return gcam_img, logits_img

    @staticmethod
    def remove_wandb_cache() -> None:
        """
        Deletes Weights and Biases cache directory. This is necessary since the Weights and Biases client currently does
        not implement proper cache cleanup itself. See https://github.com/wandb/client/issues/1193 for more details.
        """

        wandb_cache_dir = wandb.env.get_cache_dir()

        if wandb_cache_dir is not None:
            shutil.rmtree(wandb_cache_dir)
