""" Main module to execute active learning pipeline from CLI """
import json
import os.path
import fire
from active_learning import ActiveLearningPipeline
from models import PytorchFCNResnet50, PytorchUNet
from datasets import BraTSDataModule, PascalVOCDataModule
from query_strategies import QueryStrategy


# pylint: disable=too-many-arguments
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
):
    """
    # TODO (mfr): Add docstring
    :param architecture:
    :param dataset:
    :param strategy:
    :param data_dir:
    :param batch_size:
    :param num_workers:
    :param epochs:
    :param loss:
    :param optimizer:
    :return:
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


def run_active_learning_pipeline_from_config(config_file_name: str) -> None:
    """
    Runs the active learning pipeline based on a config file.
    Args:
        config_file_name: Name of or path to the config file.
    Returns:
        No return.
    """
    if not os.path.isfile(config_file_name):
        print("Config file could not be found.")
        return

    with open(config_file_name, encoding="utf-8") as config_file:
        config = json.load(config_file)
        run_active_learning_pipeline(
            **config,
        )


if __name__ == "__main__":
    fire.Fire(run_active_learning_pipeline_from_config)
