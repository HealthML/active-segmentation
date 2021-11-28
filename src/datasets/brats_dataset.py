""" Module to load and batch brats dataset """
from typing import Any, Callable, List, Literal, Optional, Tuple
import math
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import IterableDataset


# pylint: disable=too-many-instance-attributes
class BraTSDataset(IterableDataset):
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
        dimensionality: "2d" or "3d" literal to define if the datset should return 2d slices of whole 3d images.
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

    @staticmethod
    def __get_case_id(filepath: str) -> str:
        """
        Retrieves case ID from the file path of an image.

        Args:
            filepath (str): Path of the image file whose case ID is to be retrieved.

        Returns:
            str: Case ID.
        """

        # retrieve folder name from path
        return os.path.split(os.path.split(filepath)[0])[1]

    def __init__(
        self,
        image_paths: List[str],
        annotation_paths: List[str],
        clip_mask: bool = True,
        is_unlabeled: bool = False,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        target_transform: Optional[Callable[[Any], torch.Tensor]] = None,
        dimensionality: Literal["2d", "3d"] = "2d",
    ):

        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.clip_mask = clip_mask

        self.num_images = len(image_paths)
        self.num_annotations = len(annotation_paths)
        assert self.num_images == self.num_annotations

        self.is_unlabeled = is_unlabeled

        self._current_image = None
        self._current_mask = None
        self._current_image_index = None

        self.transform = transform
        self.target_transform = target_transform

        self.dimensionality = dimensionality

        self.start_index = 0
        self.end_index = self.num_images * BraTSDataset.IMAGE_DIMENSIONS[0]
        self.current_index = 0

    def __iter__(self):
        """
        Returns:
            Iterator: Iterator that yields the whole dataset if a single process is used for data loading
                or a subset of the dataset if the dataloading is split across multiple worker processes.
        """
        
        worker_info = torch.utils.data.get_worker_info()

        # check whether data loading is split across multiple workers
        if worker_info is not None:
            # code adapted from https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
            per_worker = int(math.ceil((self.end_index - self.start_index) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.start_index = self.start_index + worker_id * per_worker
            self.current_index = self.start_index
            self.end_index = min(self.start_index + per_worker, self.end_index)
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dimensionality == "2d":
            image_index = math.floor(self.current_index / BraTSDataset.IMAGE_DIMENSIONS[0])
            slice_index = self.current_index - image_index * BraTSDataset.IMAGE_DIMENSIONS[0]
            if image_index != self._current_image_index:
                self._current_image_index = image_index
                self._current_image = self.__read_image_as_array(self.image_paths[self._current_image_index], norm=True)
                self._current_mask = self.__read_image_as_array(
                self.annotation_paths[self._current_image_index], norm=False, clip=self.clip_mask
            )
            case_id = self.__get_case_id(
                filepath=self.image_paths[self._current_image_index]
            )

            x = torch.from_numpy(self._current_image[slice_index, :, :])
            y = torch.from_numpy(self._current_mask[slice_index, :, :])
        else:
            case_id = self.__get_case_id(filepath=self.image_paths[index])

            x = torch.from_numpy(self.images[index])
            y = torch.from_numpy(self.masks[index])

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        if self.is_unlabeled:
            return (
                torch.unsqueeze(x, 0),
                f"{case_id}-{slice_index}" if self.dimensionality == "2d" else case_id,
            )

        self.current_index += 1

        return torch.unsqueeze(x, 0), torch.unsqueeze(y, 0), case_id

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
