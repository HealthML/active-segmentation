""" Module containing inferencing logic """
import os
import logging
from typing import Any, Dict, Optional
from itertools import islice
import torch
import numpy as np
import nibabel as nib
from PIL import Image
from models import PytorchModel
from datasets import (
    BraTSDataModule,
    DecathlonDataModule,
    DoublyShuffledNIfTIDataset,
    BCSSDataModule,
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
        """Wraps the inference execution for image and scan datasets."""

        os.makedirs(self.prediction_dir, exist_ok=True)

        if self.dataset in ["brats", "decathlon"]:
            self.inference_scan()
        elif self.dataset in ["bcss"]:
            self.inference_image()
        else:
            logging.error(
                "Dataset %s has no implemented inference method", self.dataset
            )
            raise NotImplementedError(
                f"Dataset {self.dataset} has no implemented inference method"
            )

    def inference_scan(self) -> None:
        """Run the inferencing."""

        if self.dataset == "brats":
            self.data_dir = os.path.join(self.data_dir, "val")
            image_paths, annotation_paths = BraTSDataModule.discover_paths(
                dir_path=self.data_dir, random_samples=self.prediction_count
            )

        elif self.dataset == "decathlon":
            image_paths, annotation_paths = DecathlonDataModule.discover_paths(
                dir_path=os.path.join(self.data_dir, self.dataset_config["task"]),
                subset="val",
                random_samples=self.prediction_count,
            )

        else:
            print(f"Inferencing is not implemented for the {self.dataset} dataset.")
            return

        self.dataset_config["dim"] = 3
        if "cache_size" in self.dataset_config:
            del self.dataset_config["cache_size"]
        if "pin_memory" in self.dataset_config:
            del self.dataset_config["pin_memory"]
        if "task" in self.dataset_config:
            del self.dataset_config["task"]
        data = DoublyShuffledNIfTIDataset(
            image_paths=image_paths,
            annotation_paths=annotation_paths,
            **self.dataset_config,
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

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
            pred = self.model.predict(x.to(device)).cpu().numpy()
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

    def inference_image(self):
        """Runs inference for images."""
        # pylint: disable=protected-access
        if self.dataset == "bcss":
            data_module = BCSSDataModule(
                data_dir=self.data_dir,
                batch_size=1,
                num_workers=1,
                image_shape=self.dataset_config["image_shape"],
            )
            data = data_module._create_test_set()

        for i, (x, _, case_id) in enumerate(islice(data, self.prediction_count)):
            # Add an additional dimension to emulate one batch (1, x, y) -> (1, 1, x, y)
            x = torch.unsqueeze(x, 0)

            pred = self.model.predict(x)
            # Remove batch dimension
            seg = np.squeeze(pred, axis=0)

            seg = (seg >= 0.5) * 255
            seg = seg.astype("float64")

            original_image = Image.open(data.image_paths[i])

            # Save only the segmentation in original size
            seg_img = Image.fromarray(np.squeeze(seg, axis=0).astype(np.uint8)).resize(
                original_image.size
            )
            seg_img.save(os.path.join(self.prediction_dir, f"{case_id}_SEG_ONLY.png"))

            # Repeat dimensions to emulate RGB channels (1, x, y) -> (3, x, y)
            seg = np.repeat(seg, repeats=3, axis=0)
            # Move channel axis to the front for masking (x, y, 3) -> (3, x, y)
            original_image = np.moveaxis(
                np.asarray(original_image.resize(data.image_shape)), 2, 0
            )
            image_with_seg = np.ma.masked_array(original_image, seg)

            # Move channel axis to the end for storing on disk (3, x, y) -> (x, y, 3)
            image = Image.fromarray(np.moveaxis(image_with_seg, 0, 2))
            output_path = os.path.join(self.prediction_dir, f"{case_id}_PRED.png")
            image.save(output_path)
            print(f"Prediction for case {case_id} stored in {output_path}.")
