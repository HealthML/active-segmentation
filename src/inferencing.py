""" Module containing inferencing logic """
import os
import uuid
import torch
import numpy as np
import nibabel as nib
from models import PytorchModel
from datasets import BraTSDataModule, BraTSDataset


class Inferencer:
    """
    The inferencer to use a given model for inferencing.
    Args:
        model: A model object to be used for inferencing.
        dataset: Name of the dataset. E.g. 'brats'
        data_dir: Main directory with the dataset. E.g. './data'
        prediction_dir: Main directory with the predictions. E.g. './predictions'
        prediction_count: The amount of predictions to be generated.
    """

    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        model: PytorchModel,
        dataset: str,
        data_dir: str,
        prediction_dir: str,
        prediction_count: int,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.data_dir = data_dir
        self.prediction_dir = prediction_dir
        self.prediction_count = prediction_count

    def inference(self) -> None:
        """Run the inferencing."""
        if not self.dataset == "brats":
            print(f"Inferencing is not implemented for the {self.dataset} dataset.")
            return

        output_folder_name = f"model-{str(uuid.uuid4())}"
        output_dir = os.path.join(self.prediction_dir, output_folder_name)
        os.mkdir(output_dir)

        image_paths, annotation_paths = BraTSDataModule.discover_paths(
            self.data_dir,
            random_samples=self.prediction_count,
        )
        data = BraTSDataset(
            image_paths=image_paths,
            annotation_paths=annotation_paths,
            dimensionality="3d",
        )

        for i in range(self.prediction_count):
            x = data.__getitem__(i)[0]

            x = torch.swapaxes(x, 0, 1)
            pred = self.model(x)
            pred = torch.swapaxes(pred, 0, 1)

            seg = pred.detach().cpu().numpy()[0]
            seg = (seg >= 0.5) * 255
            seg = np.moveaxis(seg, 0, 2)
            seg = np.rot90(seg, 2, (0, 1))
            seg = seg.astype("float64")

            img = nib.Nifti1Image(seg, np.eye(4))
            file_name = os.path.basename(annotation_paths[i]).replace("seg", "pred")
            path = os.path.join(output_dir, file_name)
            nib.save(img, path)
            print(path)
