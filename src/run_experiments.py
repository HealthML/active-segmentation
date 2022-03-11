"""Helper script that allows to create multiple config files and sbatch scripts."""

import copy
import json
import os
import shutil
import subprocess
import stat
from typing import Any, Dict, Literal

import fire


def _expand_config(
    experiment_config: Dict[str, Any], config_to_add: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Updates experiment tags and experiment name of a config using a specific subobject of the config, e.g. the loss
    config or the strategy config.

    Args:
        experiment_config (Dict[str, Any]): Full config object.
        config_to_add (Dict[str, Any]): Subconfig object, e.g. loss config.

    Returns:
        Dict[str, Any]: Updated config object.
    """

    experiment_config = copy.deepcopy(experiment_config)

    name = (
        config_to_add["description"]
        if "description" in config_to_add
        else config_to_add["type"]
    )

    if "experiment_name" in experiment_config:
        experiment_config[
            "experiment_name"
        ] = f"{experiment_config['experiment_name']}-{name}"
    else:
        experiment_config["experiment_name"] = name

    if "experiment_tags" in config_to_add:
        if "experiment_tags" in experiment_config:
            if isinstance(config_to_add["experiment_tags"], str):
                experiment_config["experiment_tags"].append(
                    config_to_add["experiment_tags"]
                )
            else:
                experiment_config["experiment_tags"].extend(
                    config_to_add["experiment_tags"]
                )
        else:
            experiment_config["experiment_tags"] = config_to_add["experiment_tags"]
        config_to_add.pop("experiment_tags", None)

    return experiment_config


def create_config_files(config_file_path: str, output_dir: str) -> None:
    """
    This function takes the path of a config file as input. The input config file can contain lists for the
    `dataset_config`, the `strategy_config` and the `random_state` config options. For each combination of the specified
    datasets, strategies and random states, this function will create a separate config file in the specified
    `output_dir`.

    Args:
        config_file_path (string): Path of a config file that may contain lists of configs / values for the
            `dataset_config`, the `strategy_config` and the `random_state` options.
        output_dir (string): Path of the directory in which the created config files are to be saved.
    """

    # pylint: disable=too-many-branches, too-many-statements

    os.makedirs(output_dir, exist_ok=True)
    with open(config_file_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

        dataset_configs = []

        if "dataset_config" in config and isinstance(config["dataset_config"], list):
            for dataset_config in config["dataset_config"]:
                current_config = copy.deepcopy(config)

                dataset_name = dataset_config["dataset"]

                if "wandb_project_name" in current_config:
                    current_config[
                        "wandb_project_name"
                    ] = f"{current_config['wandb_project_name']}-{dataset_name}"

                if "experiment_tags" in dataset_config:
                    if "experiment_tags" in current_config:
                        current_config["experiment_tags"].extend(
                            dataset_config["experiment_tags"]
                        )
                    else:
                        current_config["experiment_tags"] = dataset_config[
                            "experiment_tags"
                        ]
                    del dataset_config["experiment_tags"]

                current_config["dataset_config"] = dataset_config

                dataset_configs.append(current_config)
        else:
            dataset_configs = [config]

        strategy_configs = []

        for config in dataset_configs:
            if "strategy_config" in config and isinstance(
                config["strategy_config"], list
            ):
                for strategy_config in config["strategy_config"]:
                    current_config = _expand_config(config, strategy_config)
                    current_config["strategy_config"] = strategy_config
                    strategy_configs.append(current_config)
            else:

                strategy_configs.append(config)

        loss_configs = []

        for config in strategy_configs:
            if (
                "model_config" in config
                and "loss_config" in config["model_config"]
                and isinstance(config["model_config"]["loss_config"], list)
            ):
                for loss_config in config["model_config"]["loss_config"]:
                    current_config = _expand_config(config, loss_config)
                    current_config["model_config"]["loss_config"] = loss_config
                    loss_configs.append(current_config)
            else:
                loss_configs.append(config)

        random_state_configs = []

        for config in loss_configs:
            if "random_state" in config and isinstance(config["random_state"], list):
                for random_state in config["random_state"]:
                    current_config = copy.deepcopy(config)
                    current_config["random_state"] = random_state
                    if "strategy_config" in current_config:
                        current_config["strategy_config"]["random_state"] = random_state

                    random_state_configs.append(current_config)
            else:
                random_state_configs.append(config)

        for config in random_state_configs:
            file_name = f"{config['experiment_name']}-{config['random_state']}.json"
            current_config_file_path = os.path.join(output_dir, file_name)
            with open(current_config_file_path, "w", encoding="utf-8") as config_file:
                json.dump(
                    config,
                    config_file,
                    indent=2,
                    separators=(",", ":"),
                )


def create_sbatch_jobs_from_config_files(
    config_dir: str,
    sbatch_dir: str,
    run_scripts: bool = True,
    partition: Literal["gpu", "gpupro", "gpua100"] = "gpu",
    memory: int = 50,
) -> None:
    """
    This function takes folder containing multiple config files, creates an sbatch script for each config file and
    optionally starts the sbatch runs.

    Args:
        config_dir (string): Path of a directory containing multiple config files for which sbatch runs are to be
            started.
        sbatch_dir (string): Directory in which the sbatch scripts are to be saved.
        run_scripts (bool, optional): Whether the sbatch runs should be started by this function. Defaults to `True`.
        partition (string, optional): Partition on which the slurm job is to be run. Defaults to `"gpu"`.
        memory (int, optional): Memory to be requested for the slurm job in giga bytes. Defaults to 50.
    """

    os.makedirs(sbatch_dir, exist_ok=True)

    config_dir = config_dir.rstrip("/")

    for config_file in os.listdir(config_dir):
        sbatch_script = (
            "#!/bin/sh \n\n"
            + f"#SBATCH --mem={memory}gb"
            + f"\n#SBATCH --partition={partition}"
            + "\n#SBATCH --gpus=1 \n\n"
            + f"python3 src/main.py {config_dir}/{config_file}"
        )

        sbatch_script_path = os.path.join(
            sbatch_dir, config_file.replace(".json", ".sh")
        )

        with open(sbatch_script_path, "w", encoding="utf-8") as text_file:
            text_file.write(sbatch_script)

        file_stats = os.stat(sbatch_script_path)
        os.chmod(sbatch_script_path, file_stats.st_mode | stat.S_IEXEC)

        if run_scripts:
            stdout = subprocess.check_output(
                ["sbatch", sbatch_script_path], stderr=subprocess.PIPE
            )

            print(stdout)

            os.remove(sbatch_script_path)


def start_sbatch_runs(sbatch_run_dir: str, sbatch_finished_dir: str) -> None:
    """
    This function starts multiple sbatch runs.

    Args:
        sbatch_run_dir (string): Path of a directory containing multiple sbatch scripts to be started.
        sbatch_finished_dir (string): Path of a directory to which the sbatch scripts are to be moved when the
            corresponding sbatch runs finished.
    """

    os.makedirs(sbatch_finished_dir, exist_ok=True)

    for sbatch_script in os.listdir(sbatch_run_dir):
        sbatch_script_path = os.path.join(sbatch_run_dir, sbatch_script)
        stdout = subprocess.check_output(
            ["sbatch", sbatch_script_path], stderr=subprocess.PIPE
        )

        print(stdout)

        sbatch_script_finished_path = os.path.join(sbatch_finished_dir, sbatch_script)

        shutil.move(sbatch_script_path, sbatch_script_finished_path)


if __name__ == "__main__":
    fire.Fire(
        {
            "create_config_files": create_config_files,
            "create_sbatch_jobs_from_config_files": create_sbatch_jobs_from_config_files,
            "start_sbatch_runs": start_sbatch_runs,
        }
    )
