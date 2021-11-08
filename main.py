import fire
from active_learning import ActiveLearningPipeline
from models import PytorchFCNResnet50, PytorchUNet
from datasets import BraTSDataModule, PascalVOCDataModule
from query_strategies import QueryStrategy


def run_active_learning_pipeline(
        architecture: str,
        dataset: str,
        strategy: str,
        data_dir: str = "./data",
        batch_size: int = 16,
        epochs: int = 50,
        gpus: int = 1,
        loss: str = "dice",
        optimizer: str = "adam",
):

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
        data_module = PascalVOCDataModule(data_dir, batch_size)
    elif dataset == "brats":
        data_module = BraTSDataModule(data_dir, batch_size)
    else:
        raise ValueError("Invalid data_module name.")

    pipeline = ActiveLearningPipeline(data_module, model, strategy, epochs, gpus)
    pipeline.run()


if __name__ == '__main__':
    fire.Fire(run_active_learning_pipeline)
