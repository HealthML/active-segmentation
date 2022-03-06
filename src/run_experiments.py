"""Helper script that allows to create multiple config files and sbatch scripts."""

import copy
import json
import os
import shutil
import subprocess
import stat

import fire


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
                    current_config = copy.deepcopy(config)

                    strategy_name = strategy_config["type"]
                    if "description" in strategy_config:
                        strategy_name = (
                            f"{strategy_name}-{strategy_config['description']}"
                        )

                    if "experiment_tags" in strategy_config:
                        if "experiment_tags" in current_config:
                            current_config["experiment_tags"].extend(
                                strategy_config["experiment_tags"]
                            )
                        else:
                            current_config["experiment_tags"] = strategy_config[
                                "experiment_tags"
                            ]
                        del strategy_config["experiment_tags"]

                    current_config["strategy_config"] = strategy_config
                    if "experiment_name" in current_config:
                        current_config[
                            "experiment_name"
                        ] = f"{current_config['experiment_name']}-{strategy_name}"
                    else:
                        current_config["experiment_name"] = strategy_name
                    strategy_configs.append(current_config)
            else:
                strategy_configs.append(config)

        random_state_configs = []

        for config in strategy_configs:
            if "random_state" in config and isinstance(config["random_state"], list):
                for random_state in config["random_state"]:
                    current_config = copy.deepcopy(config)
                    current_config["random_state"] = random_state
                    if "strategy_config" in current_config:
                        current_config["strategy_config"]["random_state"] = random_state

                    random_state_configs.append(current_config)

        for idx, config in enumerate(random_state_configs):
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
    config_dir: str, sbatch_dir: str, run_scripts: bool = True
) -> None:
    """
    This function takes folder containing multiple config files, creates an sbatch script for each config file and
    optionally starts the sbatch runs.

    Args:
        config_dir (string): Path of a directory containing multiple config files for which sbatch runs are to be
            started.
        sbatch_dir (string): Directory in which the sbatch scripts are to be saved.
        run_scripts (bool, optional): Whether the sbatch runs should be started by this function. Defaults to `True`.
    """

    os.makedirs(sbatch_dir, exist_ok=True)

    config_dir = config_dir.rstrip("/")

    for config_file in os.listdir(config_dir):
        sbatch_script = (
            "#!/bin/sh \n\n"
            + "#SBATCH --mem=50gb"
            + "\n#SBATCH --partition=gpu"
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
