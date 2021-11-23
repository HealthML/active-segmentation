""" Module to load and batch brats dataset """
from typing import Any, Callable, List, Literal, Optional, Tuple
import math
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# pylint: disable=too-many-instance-attributes
class BraTSDataset(Dataset):
    """
    The BraTS dataset is published in the course of the annual MultimodalBrainTumorSegmentation Challenge (BraTS)
    held since 2012. It is composed of 3T multimodal MRI scans from patients affected by glioblastoma or lower grade
    glioma, as well as corresponding ground truth labels provided by expert board-certified neuroradiologists.
    Further information: https://www.med.upenn.edu/cbica/brats2020/data.html
    Args:
        image_paths: List with the paths to the images.
        annotation_paths: List with the paths to the annotations.
        clip_mask: Flag to clip the annotation labels, if True only label 1 is kept.
        transform: Function to transform the images.
        target_transform: Function to transform the annotations.
    """

    IMAGE_DIMENSIONS = (155, 240, 240)

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """
        Normalizes an image by
            1. Dividing by the maximum value
            2. Subtracting the mean, zeros will be ignored while calculating the mean
            3. Dividing by the negative minimum value
        Args:
            img: The input image that should be normalized.

        Returns:
            Normalized image with background values normalized to -1
        """
        tmp = img / np.max(img)
        # ignore zero values for mean calculation because background dominates
        tmp = tmp - np.mean(tmp[tmp > 0])
        # make normalize original zero values to -1
        return tmp / (-np.min(tmp))

    @staticmethod
    def __read_image_as_array(
        filepath: str, norm: bool, clip: bool = False
    ) -> np.ndarray:
        """
        Reads image or annotation as numpy array.
        Args:
            filepath: Path of the image file.
            norm: Whether the image should be normalized.
            clip: Whether the image should be clipped.

        Returns:
            The array representation of an image.
        """
        img = nib.load(filepath).get_fdata()

        if clip:
            img = np.clip(img, 0, 1)

        if norm:
            img = BraTSDataset.normalize(img)
        return np.moveaxis(img, 2, 0)

    def __init__(
        self,
        image_paths: List[str],
        annotation_paths: List[str],
        clip_mask: bool = True,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        target_transform: Optional[Callable[[Any], torch.Tensor]] = None,
        dimensionality: Literal["2d", "3d"] = "2d",
    ):

        self.image_paths = image_paths
        self.images = [
            self.__read_image_as_array(filepath=image_path, norm=True)
            for image_path in self.image_paths
        ]
        self.annotation_paths = annotation_paths
        self.clip_mask = clip_mask
        self.masks = [
            self.__read_image_as_array(
                filepath=annotation_path, norm=False, clip=self.clip_mask
            )
            for annotation_path in self.annotation_paths
        ]
        self.num_images = len(image_paths)
        self.num_annotations = len(annotation_paths)
        assert self.num_images == self.num_annotations
        self._current_image = None
        self._current_image_index = None
        self._current_mask = None

        self.transform = transform
        self.target_transform = target_transform

        self.dimensionality = dimensionality

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dimensionality == "2d":
            image_index = math.floor(index / BraTSDataset.IMAGE_DIMENSIONS[0])
            slice_index = index - image_index * BraTSDataset.IMAGE_DIMENSIONS[0]
            if image_index != self._current_image_index:
                self._current_image_index = image_index
                self._current_image = self.images[self._current_image_index]
                self._current_mask = self.masks[self._current_image_index]

            x = torch.from_numpy(self._current_image[slice_index, :, :])
            y = torch.from_numpy(self._current_mask[slice_index, :, :])
        else:
            x = torch.from_numpy(self.images[index])
            y = torch.from_numpy(self.masks[index])

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)

    def __len__(self) -> int:
        return self.num_images * BraTSDataset.IMAGE_DIMENSIONS[0]

    def add_image(self, image_path: str, annotation_path: str) -> None:
        """
        Adds an image to this dataset.
        Args:
            image_path: Path of the image to be added.
            annotation_path: Path of the annotation of the image to be added.

        Returns:
            None. Raises ValueError if image already exists.
        """

        if (image_path not in self.image_paths) and (
            annotation_path not in self.annotation_paths
        ):
            self.image_paths.append(image_path)
            self.annotation_paths.append(annotation_path)
            self.num_images += 1
        else:
            raise ValueError("Image already belongs to this dataset.")

    def remove_image(self, image_path: str, annotation_path: str) -> None:
        """
        Removes an image from this dataset.
        Args:
            image_path: Path of the image to be removed.
            annotation_path: Path of the annotation of the image to be removed.

        Returns:
            None. Raises ValueError if image already exists.
        """

        if image_path in self.image_paths and annotation_path in self.annotation_paths:
            self.image_paths.remove(image_path)
            self.annotation_paths.remove(annotation_path)
            self.num_images -= 1
        else:
            raise ValueError("Image does not belong to this dataset.")
