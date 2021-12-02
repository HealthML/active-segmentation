""" Main module to execute active learning pipeline from CLI """
import json
import os.path
from typing import Any, Dict, Iterable, Optional

import fire
from pytorch_lightning.loggers import WandbLogger
import wandb

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
    dataset_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    epochs: int = 50,
    experiment_tags: Optional[Iterable[str]] = None,
    gpus: int = 1,
    num_workers: int = 4,
    prediction_count: Optional[int] = None,
    prediction_dir: str = "./predictions",
    wandb_project_name: str = "active-segmentation",
) -> None:
    """
    Main function to execute an active learning pipeline run, or start an active learning
        simulation.
    Args:
        architecture (string): Name of the desired model architecture. E.g. 'u_net'.
        dataset (string): Name of the dataset. E.g. 'brats'
        strategy (string): Name of the query strategy. E.g. 'base'
        experiment_name (string): Name of the experiment.
        batch_size (int, optional): Size of training examples passed in one training step.
        data_dir (string, optional): Main directory with the dataset. E.g. './data'
        dataset_config (Dict[str, Any], optional): Dictionary with dataset specific parameters.
        model_config (Dict[str, Any], optional): Dictionary with model specific parameters.
        epochs (int, optional): Number of iterations with the full dataset.
        experiment_tags (Iterable[string], optional): Tags with which to label the experiment.
        gpus (int): Number of GPUS to use for model training.
        num_workers (int, optional): Number of workers.
        wandb_project_name (string): Name of the project that the W&B runs are stored in

    Returns:
        None.
    """

    wandb_logger = WandbLogger(
        project=wandb_project_name,
        entity="active-segmentation",
        name=experiment_name,
        tags=experiment_tags,
        config=locals().copy(),
    )

    if architecture == "fcn_resnet50":
        model = PytorchFCNResnet50(**model_config)
    elif architecture == "u_net":
        model = PytorchUNet(**model_config)
    else:
        raise ValueError("Invalid model architecture.")

    if strategy == "base":
        strategy = QueryStrategy()
    else:
        raise ValueError("Invalid query strategy.")

    if dataset_config is None:
        dataset_config = {}

    if dataset == "pascal-voc":
        data_module = PascalVOCDataModule(
            data_dir, batch_size, num_workers, **dataset_config
        )
    elif dataset == "brats":
        data_module = BraTSDataModule(
            data_dir,
            batch_size,
            num_workers,
            dim=model.input_dimensionality(),
            **dataset_config,
        )
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
    )
    inferencer.inference()


def run_active_learning_pipeline_from_config(
    config_file_name: str, hp_optimisation: bool = False
) -> None:
    """
    Runs the active learning pipeline based on a config file.
    Args:
        config_file_name: Name of or path to the config file.
        hp_optimisation: If this flag is set, run the pipeline with different hyperparameters based
            on the configured sweep file
    """
    if not os.path.isfile(config_file_name):
        print("Config file could not be found.")
        raise FileNotFoundError(f"{config_file_name} is not a valid filename.")

    with open(config_file_name, encoding="utf-8") as config_file:
        hyperparameter_defaults = json.load(config_file)
        config = hyperparameter_defaults

        if "dataset_config" in config and "dataset" in config["dataset_config"]:
            config["dataset"] = config["dataset_config"]["dataset"]
            del config["dataset_config"]["dataset"]
        if "dataset_config" in config and "data_dir" in config["dataset_config"]:
            config["data_dir"] = config["dataset_config"]["data_dir"]
            del config["dataset_config"]["data_dir"]

        if "model_config" in config and "architecture" in config["model_config"]:
            config["architecture"] = config["model_config"]["architecture"]
            del config["model_config"]["architecture"]
        if "model_config" in config and "input_shape" in config["model_config"]:
            config["model_config"]["input_shape"] = tuple(
                config["model_config"]["input_shape"]
            )

        if hp_optimisation:
            print("Start Hyperparameter Optimisation using sweep.yaml file")
            wandb.init(
                config=hyperparameter_defaults,
                project=config["wandb_project_name"],
                entity="active-segmentation",
            )
            # Config parameters are automatically set by W&B sweep agent
            config = wandb.config

        run_active_learning_pipeline(
            **config,
        )


if __name__ == "__main__":
    fire.Fire(run_active_learning_pipeline_from_config)
