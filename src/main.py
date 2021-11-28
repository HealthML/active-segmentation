""" Main module to execute active learning pipeline from CLI """
import json
import os.path
from typing import Iterable, List, Literal, Optional

import fire
from pytorch_lightning.loggers import WandbLogger

from active_learning import ActiveLearningPipeline
from inferencing import Inferencer
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
    num_u_net_levels: int = 4,
    u_net_dim: Literal["2d", "3d"] = "2d",
    u_net_input_shape: Optional[List[int]] = None,
    prediction_count: Optional[int] = None,
    prediction_dir: str = "./predictions",
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
        num_u_net_levels: Number levels (encoder and decoder blocks) in the U-Net.
        u_net_dim: Dimension of the U-Net. Either "2d" or "3d".
        u_net_input_shape: Shape of the U-Net input.

    Returns:
        None.
    """

    # Safely assign list default value.
    if u_net_input_shape is None:
        u_net_input_shape = [240, 240]

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
        model = PytorchUNet(
            num_levels=num_u_net_levels,
            optimizer=optimizer,
            loss=loss,
            dim=u_net_dim,
            input_shape=tuple(u_net_input_shape),
        )
    else:
        raise ValueError("Invalid model architecture.")

    if strategy == "base":
        strategy = QueryStrategy()
    else:
        raise ValueError("Invalid query strategy.")

    if dataset == "pascal-voc":
        data_module = PascalVOCDataModule(data_dir, batch_size, num_workers)
    elif dataset == "brats":
        data_module = BraTSDataModule(data_dir, batch_size, num_workers, dim=u_net_dim)
    else:
        raise ValueError("Invalid data_module name.")

    pipeline = ActiveLearningPipeline(
        data_module, model, strategy, epochs, gpus, wandb_logger
    )
    pipeline.run()

    if prediction_count is None:
        return

    inferencer = Inferencer(
        model,
        dataset,
        os.path.join(data_dir, "val"),
        prediction_dir,
        prediction_count,
        u_net_dim,
    )
    inferencer.inference()


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
