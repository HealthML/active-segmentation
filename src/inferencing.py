""" Module containing inferencing logic """
import os
from typing import Any, Dict, Optional
import torch
import numpy as np
import nibabel as nib
from models import PytorchModel
from datasets import (
    BraTSDataModule,
    BraTSDataset,
    DecathlonDataModule,
    DecathlonDataset,
)


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
        dataset_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.data_dir = data_dir
        self.prediction_dir = prediction_dir
        self.prediction_count = prediction_count
        self.model_dim = model.input_dimensionality()
        self.dataset_config = dataset_config

    def inference(self) -> None:
        """Run the inferencing."""

        os.makedirs(self.prediction_dir, exist_ok=True)

        if self.dataset == "brats":
            self.data_dir = os.path.join(self.data_dir, "val")
            image_paths, annotation_paths = BraTSDataModule.discover_paths(
                dir_path=self.data_dir, random_samples=self.prediction_count
            )
            data = BraTSDataset(
                image_paths=image_paths, annotation_paths=annotation_paths, dim=3
            )
        elif self.dataset == "decathlon":
            image_paths, annotation_paths = DecathlonDataModule.discover_paths(
                dir_path=os.path.join(self.data_dir, self.dataset_config["task"]),
                subset="val",
                random_samples=self.prediction_count,
            )
            self.dataset_config["dim"] = 3
            if "cache_size" in self.dataset_config:
                del self.dataset_config["cache_size"]
            if "pin_memory" in self.dataset_config:
                del self.dataset_config["pin_memory"]
            if "task" in self.dataset_config:
                del self.dataset_config["task"]
            data = DecathlonDataset(
                image_paths=image_paths,
                annotation_paths=annotation_paths,
                **self.dataset_config,
            )
        else:
            print(f"Inferencing is not implemented for the {self.dataset} dataset.")
            return

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
            seg = seg.astype("float64")

            input_img = nib.load(image_paths[i])
            img = nib.Nifti1Image(seg, input_img.affine)
            file_name = os.path.basename(annotation_paths[i]).replace(
                ".nii", "_pred.nii"
            )
            path = os.path.join(self.prediction_dir, file_name)
            nib.save(img, path)
            print(f"Predictions stored in path {path}")
