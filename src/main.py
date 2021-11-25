""" Main module to execute active learning pipeline from CLI """
import json
import os.path
import fire
from active_learning import ActiveLearningPipeline
from models import PytorchFCNResnet50, PytorchUNet
from datasets import BraTSDataModule, PascalVOCDataModule
from query_strategies import QueryStrategy
import wandb


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
    learning_rate: float = 0.0001,
    lr_scheduler: str = None,
    num_u_net_levels: int = 4,
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
        learning_rate: The step size at each iteration while moving towards a minimum of the loss function.
        lr_scheduler: Name of the learning rate scheduler algorithm. E.g. 'reduceLROnPlateau'.
        num_u_net_levels: Number levels (encoder and decoder blocks) in the U-Net.
    Returns:
        None.
    """

    if architecture == "fcn_resnet50":
        model = PytorchFCNResnet50(
            optimizer=optimizer,
            loss=loss,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
        )
    elif architecture == "u_net":
        model = PytorchUNet(
            num_levels=num_u_net_levels,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            lr_scheduler=lr_scheduler,
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
        data_module = BraTSDataModule(data_dir, batch_size, num_workers)
    else:
        raise ValueError("Invalid data_module name.")

    pipeline = ActiveLearningPipeline(data_module, model, strategy, epochs, gpus)
    pipeline.run()


def run_active_learning_pipeline_from_config(
    config_file_name: str, hp_optimisation: bool = False
) -> None:
    """
    Runs the active learning pipeline based on a config file.
    Args:
        config_file_name: Name of or path to the config file.
        hp_optimisation: If this flag is set, run the pipeline with different hyperparameters based on the configured sweep file
    """
    if not os.path.isfile(config_file_name):
        print("Config file could not be found.")
        raise FileNotFoundError(f"{config_file_name} is not a valid filename.")

    with open(config_file_name, encoding="utf-8") as config_file:
        hyperparameter_defaults = json.load(config_file)
        config = hyperparameter_defaults
        if hp_optimisation:
            print("Start Hyperparameter Optimisation using sweep.yaml file")
            wandb.init(config=hyperparameter_defaults)
            # Config parameters are automatically set by W&B sweep agent
            config = wandb.config

        run_active_learning_pipeline(
            **config,
        )


if __name__ == "__main__":
    fire.Fire(run_active_learning_pipeline_from_config)
