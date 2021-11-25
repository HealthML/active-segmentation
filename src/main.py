""" Main module to execute active learning pipeline from CLI """
import json
import os.path
from typing import Iterable, Optional

import fire
from pytorch_lightning.loggers import WandbLogger

from active_learning import ActiveLearningPipeline
from models import PytorchFCNResnet50, PytorchUNet
from datasets import BraTSDataModule, PascalVOCDataModule
from query_strategies import QueryStrategy


# pylint: disable=too-many-arguments,too-many-locals
def run_active_learning_pipeline(
    architecture: str,
    dataset: str,
    strategy: str,
    experiment_name: str,
    batch_size: int = 16,
    data_dir: str = "./data",
    epochs: int = 50,
    experiment_tags: Optional[Iterable[str]] = None,
    gpus: int = 1,
    loss: str = "dice",
    num_workers: int = 4,
    optimizer: str = "adam",
) -> None:
    """
    Main function to execute an active learning pipeline run, or start an active learning simulation.
    Args:
        architecture (string): Name of the desired model architecture. E.g. 'u_net'.
        dataset (string): Name of the dataset. E.g. 'brats'
        strategy (string): Name of the query strategy. E.g. 'base'
        experiment_name (string): Name of the experiment.
        batch_size (int, optional): Size of training examples passed in one training step.
        data_dir (string, optional): Main directory with the dataset. E.g. './data'
        epochs (int, optional): Number of iterations with the full dataset.
        experiment_tags (Iterable[string], optional): Tags with which to label the experiment.
        gpus (int): Number of GPUS to use for model training.
        loss (str, optional): Name of the performance measure to optimize. E.g. 'dice'.
        num_workers (int, optional): Number of workers.
        optimizer (str, optional): Name of the optimization algorithm. E.g. 'adam'.

    Returns:
        None.
    """

    wandb_logger = WandbLogger(
        project="active-segmentation",
        entity="active-segmentation",
        name=experiment_name,
        tags=experiment_tags,
        config=locals().copy(),
    )

    if architecture == "fcn_resnet50":
        model = PytorchFCNResnet50(optimizer=optimizer, loss=loss)
    elif architecture == "u_net":
        model = PytorchUNet(optimizer=optimizer, loss=loss)
    else:
        raise ValueError("Invalid model architecture.")

    if strategy == "base":
        strategy = QueryStrategy()
    else:
        raise ValueError("Invalid query strategy.")

    if dataset == "pascal-voc":
        data_module = PascalVOCDataModule(data_dir, batch_size, num_workers)
    elif dataset == "brats":
        data_module = BraTSDataModule(data_dir, batch_size, num_workers)
    else:
        raise ValueError("Invalid data_module name.")

    pipeline = ActiveLearningPipeline(
        data_module, model, strategy, epochs, gpus, wandb_logger
    )
    pipeline.run()


def run_active_learning_pipeline_from_config(config_file_name: str) -> None:
    """
    Runs the active learning pipeline based on a config file.
    Args:
        config_file_name: Name of or path to the config file.
    """
    if not os.path.isfile(config_file_name):
        print("Config file could not be found.")
        raise FileNotFoundError(f"{config_file_name} is not a valid filename.")

    with open(config_file_name, encoding="utf-8") as config_file:
        config = json.load(config_file)
        run_active_learning_pipeline(
            **config,
        )


if __name__ == "__main__":
    fire.Fire(run_active_learning_pipeline_from_config)
