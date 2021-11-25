""" Main module to execute active learning pipeline from CLI """
import json
import os.path
from typing import Optional
import fire
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
    data_dir: str = "./data",
    batch_size: int = 16,
    num_workers: int = 4,
    epochs: int = 50,
    gpus: int = 1,
    loss: str = "dice",
    optimizer: str = "adam",
    prediction_count: Optional[int] = None,
    prediction_dir: str = "./predictions",
) -> None:
    """
    Main function to execute an active learning pipeline run, or start an active learning simulation.
    Args:
        architecture: Name of the desired model architecture. E.g. 'u_net'.
        dataset: Name of the dataset. E.g. 'brats'
        strategy: Name of the query strategy. E.g. 'base'
        data_dir: Main directory with the dataset. E.g. './data'
        batch_size: Size of training examples passed in one training step.
        num_workers: Number of workers.
        epochs: Number of iterations with the full dataset.
        loss: Name of the performance measure to optimize. E.g. 'dice'.
        optimizer: Name of the optimization algorithm. E.g. 'adam'.

    Returns:
        None.
    """

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

    pipeline = ActiveLearningPipeline(data_module, model, strategy, epochs, gpus)
    pipeline.run()

    if prediction_count is None:
        return

    inferencer = Inferencer(
        model, dataset, os.path.join(data_dir, "val"), prediction_dir, prediction_count
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
