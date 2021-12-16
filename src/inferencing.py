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
        self.model_dim = model.input_dimensionality()

    def inference(self) -> None:
        """Run the inferencing."""
        if not self.dataset == "brats":
            print(f"Inferencing is not implemented for the {self.dataset} dataset.")
            return

        os.makedirs(self.prediction_dir, exist_ok=True)

        image_paths, annotation_paths = BraTSDataModule.discover_paths(
            dir_path=self.data_dir,
            random_samples=self.prediction_count,
        )
        data = BraTSDataset(
            image_paths=image_paths,
            annotation_paths=annotation_paths,
            dim=3,
        )

        for i, (x, _, _) in enumerate(data):
            # For 2d case:
            # Switching axes to predict for single slices.
            # Swap from (1, z, x, y) to (z, 1, x, y) and after predicting swap back.
            # Basically represents the 3d images as a batch of z 2d slices.
            # For 3d case:
            # Adding one more dimension to represent the image as a batch of one single image.
            x = (
                torch.swapaxes(x, 0, 1)
                if self.model_dim == 2
                else torch.unsqueeze(x, 0)
            )
            pred = self.model.predict(x)
            seg = np.squeeze(np.swapaxes(pred, 0, 1) if self.model_dim == 2 else pred)

            seg = (seg >= 0.5) * 255
            seg = np.moveaxis(seg, 0, 2)
            seg = np.rot90(seg, 2, (0, 1))
            seg = seg.astype("float64")

            img = nib.Nifti1Image(seg, np.eye(4))
            file_name = os.path.basename(annotation_paths[i]).replace("seg", "pred")
            path = os.path.join(self.prediction_dir, file_name)
            nib.save(img, path)
            print(f"Predictions stored in path {path}")
