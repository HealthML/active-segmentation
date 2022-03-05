import json
import os
import shutil
import subprocess
import stat

import fire


def create_config_files(config_file_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(config_file_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

        dataset_configs = []

        if "dataset_config" in config and isinstance(config["dataset_config"], list):
            for dataset_config in config["dataset_config"]:
                current_config = config.copy()

                if "name" in dataset_config:
                    dataset_name = dataset_config["name"]
                    del dataset_config["name"]

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
                for idx, strategy_config in enumerate(config["strategy_config"]):
                    current_config = config.copy()

                    if "name" in strategy_config:
                        strategy_name = strategy_config["name"]
                        del strategy_config["name"]
                    else:
                        strategy_name = f"strategy-{idx}"

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
                    strategy_configs.append(current_config)
            else:
                strategy_configs.append(config)

        random_state_configs = []

        for config in strategy_configs:
            if "random_state" in config and isinstance(config["random_state"], list):
                for random_state in config["random_state"]:
                    current_config = config.copy()
                    current_config["random_state"] = random_state
                    if "strategy_config" in current_config:
                        current_config["strategy_config"]["random_state"] = random_state

                    random_state_configs.append(current_config)

        for idx, config in enumerate(random_state_configs):
            file_name = f"{config['experiment_name']}.json"
            current_config_file_path = os.path.join(output_dir, file_name)
            with open(current_config_file_path, "w", encoding="utf-8") as config_file:
                json.dump(
                    config, config_file, indent=2, separators=(",", ":"),
                )


def create_sbatch_jobs_from_config_files(
    config_dir: str, sbatch_dir: str, run_scripts: bool = True
) -> None:
    os.makedirs(sbatch_dir, exist_ok=True)

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

        with open(sbatch_script_path, "w") as text_file:
            text_file.write(sbatch_script)

        file_stats = os.stat(sbatch_script_path)
        os.chmod(sbatch_script_path, file_stats.st_mode | stat.S_IEXEC)

        if run_scripts:
            stdout = subprocess.check_output(
                ["sbatch", sbatch_script_path], stderr=subprocess.PIPE
            )

            print(stdout)

            os.remove(sbatch_script_path)


def start_sbatch_runs(sbatch_run_dir, sbatch_finished_dir):
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
        }
    )

