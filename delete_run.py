import os.path
import shutil
import wandb

from fire import Fire


def delete_run(run_name: str, model_dir: str, prediction_dir: str) -> None:
    """
    Deletes an experiment run from Weights and Biases and removes model artifacts and predictions saved locally.

    Args:
        run_name (string): Full name of the Weights and Biases run. Must have the form "<entity>/<project>/<run_id>".
            E.g. `active-segmentation/active-segmentation-tests/35trq9b8`.
        model_dir (string): Path of the directory in which the model files are saved locally.
        prediction_dir (string): Path of the directory in which the predictions are saved locally.
    """

    run_id = os.path.basename(run_name)

    api = wandb.Api()
    run = api.run(run_name)
    run.delete()

    run_model_dir = os.path.join(model_dir, run_id)
    run_prediction_dir = os.path.join(prediction_dir, run_id)

    if os.path.exists(run_model_dir):
        shutil.rmtree(run_model_dir)

    if os.path.exists(run_prediction_dir):
        shutil.rmtree(run_prediction_dir)

    print("Deleted run", run_name)


if __name__ == "__main__":
    Fire(delete_run)
