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
from datasets import BraTSDataModule, PascalVOCDataModule, DecathlonDataModule
from query_strategies import RandomSamplingStrategy, UncertaintySamplingStrategy


def create_data_module(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    random_state: int,
    active_learning_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
):
    """
    Creates the correct data module.

    Args:
        dataset (string): Name of the dataset. E.g. 'brats'
        data_dir (string, optional): Main directory with the dataset. E.g. './data'
        batch_size (int, optional): Size of training examples passed in one training step.
        num_workers (int, optional): Number of workers.
        random_state (int): Random constant for shuffling the data
        active_learning_config (Dict[str, Any): Dictionary with active learning specific parameters.
        dataset_config (Dict[str, Any]): Dictionary with dataset specific parameters.

    Returns:
        The data module.
    """

    if dataset == "pascal-voc":
        data_module = PascalVOCDataModule(
            data_dir, batch_size, num_workers, **dataset_config
        )
    elif dataset == "brats":
        data_module = BraTSDataModule(
            data_dir,
            batch_size,
            num_workers,
            active_learning_mode=active_learning_config.get(
                "active_learning_mode", False
            ),
            initial_training_set_size=active_learning_config.get(
                "initial_training_set_size", 10
            ),
            random_state=random_state,
            **dataset_config,
        )
    elif dataset == "decathlon":
        data_module = DecathlonDataModule(
            data_dir,
            batch_size,
            num_workers,
            active_learning_mode=active_learning_config.get(
                "active_learning_mode", False
            ),
            initial_training_set_size=active_learning_config.get(
                "initial_training_set_size", 10
            ),
            random_state=random_state,
            **dataset_config,
        )
    else:
        raise ValueError("Invalid data_module name.")

    return data_module


# pylint: disable=too-many-arguments,too-many-locals
def run_active_learning_pipeline(
    architecture: str,
    dataset: str,
    strategy: str,
    experiment_name: str,
    batch_size: int = 16,
    checkpoint_dir: Optional[str] = None,
    data_dir: str = "./data",
    dataset_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    model_selection_criterion: Optional[str] = "loss",
    active_learning_config: Optional[Dict[str, Any]] = None,
    epochs: int = 50,
    experiment_tags: Optional[Iterable[str]] = None,
    gpus: int = 1,
    num_workers: int = 4,
    learning_rate: float = 0.0001,
    lr_scheduler: Optional[str] = None,
    num_levels: int = 4,
    prediction_count: Optional[int] = None,
    prediction_dir: str = "./predictions",
    wandb_project_name: str = "active-segmentation",
    early_stopping: bool = False,
    random_state: int = 42,
) -> None:
    """
    Main function to execute an active learning pipeline run, or start an active learning
        simulation.
    Args:
        architecture (string): Name of the desired model architecture. E.g. 'u_net'.
        dataset (string): Name of the dataset. E.g. 'brats'
        strategy (string): Name of the query strategy. E.g. 'random'
        experiment_name (string): Name of the experiment.
        batch_size (int, optional): Size of training examples passed in one training step.
        checkpoint_dir (str, optional): Directory where the model checkpoints are to be saved.
        data_dir (string, optional): Main directory with the dataset. E.g. './data'
        dataset_config (Dict[str, Any], optional): Dictionary with dataset specific parameters.
        model_config (Dict[str, Any], optional): Dictionary with model specific parameters.
        active_learning_config (Dict[str, Any], optional): Dictionary with active learning specific parameters.
        epochs (int, optional): Number of iterations with the full dataset.
        experiment_tags (Iterable[string], optional): Tags with which to label the experiment.
        gpus (int): Number of GPUS to use for model training.
        num_workers (int, optional): Number of workers.
        learning_rate (float): The step size at each iteration while moving towards a minimum of the loss.
        lr_scheduler (string, optional): Algorithm used for dynamically updating the learning rate during training.
            E.g. 'reduceLROnPlateau' or 'cosineAnnealingLR'
        num_levels (int, optional): Number levels (encoder and decoder blocks) in the U-Net. Defaults to 4.
        early_stopping (bool, optional): Enable/Disable Early stopping when model
            is not learning anymore (default = False).
        random_state (int): Random constant for shuffling the data
        wandb_project_name (string, optional): Name of the project that the W&B runs are stored in.

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

    if dataset_config is None:
        dataset_config = {}

    if active_learning_config is None:
        active_learning_config = {}

    data_module = create_data_module(
        dataset,
        data_dir,
        batch_size,
        num_workers,
        random_state,
        active_learning_config,
        dataset_config,
    )

    if architecture == "fcn_resnet50":
        if data_module.data_channels() != 1:
            raise ValueError(
                f"{architecture} does not support multiple input channels."
            )

        model = PytorchFCNResnet50(
            learning_rate=learning_rate, lr_scheduler=lr_scheduler, **model_config
        )
    elif architecture == "u_net":
        model = PytorchUNet(
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            num_levels=num_levels,
            in_channels=data_module.data_channels(),
            **model_config,
        )
    else:
        raise ValueError("Invalid model architecture.")

    strategy = create_query_strategy(strategy)

    if checkpoint_dir is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, f"{wandb_logger.experiment.id}")

    prediction_dir = os.path.join(prediction_dir, f"{wandb_logger.experiment.id}")

    pipeline = ActiveLearningPipeline(
        data_module,
        model,
        strategy,
        epochs,
        gpus,
        checkpoint_dir,
        active_learning_mode=active_learning_config.get("active_learning_mode", False),
        items_to_label=active_learning_config.get("items_to_label", 1),
        iterations=active_learning_config.get("iterations", 10),
        logger=wandb_logger,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        model_selection_criterion=model_selection_criterion,
    )
    pipeline.run()

    if prediction_count is None:
        return

    inferencer = Inferencer(
        model,
        dataset,
        data_dir,
        prediction_dir,
        prediction_count,
        dataset_config=dataset_config,
    )
    inferencer.inference()


def create_query_strategy(strategy: str):
    """
    Initialises the chosen query strategy
    Args:
        strategy (str): Name of the query strategy. E.g. 'random'
    """
    if strategy == "random":
        return RandomSamplingStrategy()
    if strategy == "uncertainty":
        return UncertaintySamplingStrategy()
    raise ValueError("Invalid query strategy.")


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
        if (
            "dataset_config" in config
            and "mask_filter_values" in config["dataset_config"]
        ):
            config["dataset_config"]["mask_filter_values"] = tuple(
                config["dataset_config"]["mask_filter_values"]
            )

        if (
            "model_config" in config
            and "dim" in config["model_config"]
            and "dataset_config" in config
        ):
            config["dataset_config"]["dim"] = config["model_config"]["dim"]
        if "model_config" in config and "architecture" in config["model_config"]:
            config["architecture"] = config["model_config"]["architecture"]
            del config["model_config"]["architecture"]
        if "model_config" in config and "learning_rate" in config["model_config"]:
            config["learning_rate"] = config["model_config"]["learning_rate"]
            del config["model_config"]["learning_rate"]
        if "model_config" in config and "lr_scheduler" in config["model_config"]:
            config["lr_scheduler"] = config["model_config"]["lr_scheduler"]
            del config["model_config"]["lr_scheduler"]
        if "model_config" in config and "num_levels" in config["model_config"]:
            config["num_levels"] = config["model_config"]["num_levels"]
            del config["model_config"]["num_levels"]
        if (
            "model_config" in config
            and "model_selection_criterion" in config["model_config"]
        ):
            config["model_selection_criterion"] = config["model_config"][
                "model_selection_criterion"
            ]
            del config["model_config"]["model_selection_criterion"]

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
