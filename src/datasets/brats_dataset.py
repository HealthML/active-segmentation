""" Module to load and batch brats dataset """
from typing import Any, Callable, List, Optional, Tuple
import math
from multiprocessing import Manager
import os
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import IterableDataset


# pylint: disable=too-many-instance-attributes,abstract-method
class BraTSDataset(IterableDataset):
    """
    The BraTS dataset is published in the course of the annual MultimodalBrainTumorSegmentation Challenge (BraTS)
    held since 2012. It is composed of 3T multimodal MRI scans from patients affected by glioblastoma or lower grade
    glioma, as well as corresponding ground truth labels provided by expert board-certified neuroradiologists.
    Further information: https://www.med.upenn.edu/cbica/brats2020/data.html
    Args:
        image_paths (List[str]): List with the paths to the images.
        annotation_paths (List[str]): List with the paths to the annotations.
        cache_size (int, optional): Number of images to keep in memory to speed-up data loading in subsequent epochs.
            Defaults to zero.
        clip_mask (bool, optinal): Flag to clip the annotation labels, if True only label 1 is kept. Defaults to true.
        shuffle (bool, optional): Whether the data should be shuffled.
        transform (Callable[[Any], Tensor], optional): Function to transform the images.
        target_transform (Callable[[Any], Tensor], optional): Function to transform the annotations.
        dimensionality (int, optional): 2 or 3 to define if the datset should return 2d slices of whole 3d images.
            Defaults to 2.
        slice_indices (List[np.array], optional): Array of indices per image which should be part of the dataset.
            Uses all slices if None. Defaults to None.
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

    @staticmethod
    def __arange_image_slice_indices(
        filepaths: List[str],
        dim: int = 2,
        shuffle: bool = False,
        seed: Optional[int] = None,
        slice_indices: Optional[List[np.array]] = None,
    ) -> List[Tuple[int]]:
        """
        Reads the slice indices for the images at the provided slice paths and pairs them with their image index.

        Implements efficient shuffling for 2D image datasets like the DecathlonDataset whose elements represent the
        slices of multiple 3D images. To allow for efficient image pre-fetching, first the order of all 3D images is
        shuffled and then the order of slices within each 3D image is shuffled. This way the 3D images can still be
        loaded as a whole.

        Args:
            filepaths (List[str]): The paths of the images.
            dim (int, optional): The dimensionality of the dataset. Defaults to 2.
            shuffle (boolean, optional): Flag indicating wether to shuffle the slices. Defaults to False.
            seed (int, optional): Random seed for shuffling.
            slice_indices (List[np.array], optional): Array of indices per image which should be part of the dataset.
                Uses all slices if None. Defaults to None.

        Returns:
            A list of (image_index, slice_index) tuples.
        """
        if slice_indices is None:
            slice_indices = [
                np.arange(BraTSDataset.IMAGE_DIMENSIONS[0] if dim == 2 else 1)
                for _ in filepaths
            ]

        if shuffle:
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)

            # Shuffle the slices within the images
            for slices in slice_indices:
                np.random.shuffle(slices)

            # Shuffle the images
            enumerated_slice_indices = list(enumerate(slice_indices))
            random.shuffle(enumerated_slice_indices)
        else:
            enumerated_slice_indices = enumerate(slice_indices)

        # Pair up the slices indices with their image index and concatenate for all images
        # (e.g. [5,1,9,0,...] for image index 3 becomes [(3,5),(3,1),(3,9),(3,0),...])
        image_slice_indices = [
            (image_index, slice_index)
            for image_index, slices in enumerated_slice_indices
            for slice_index in slices
        ]

        # Concatenate the [image_index, slice_index] pairs for all images
        return image_slice_indices

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        image_paths: List[str],
        annotation_paths: List[str],
        cache_size: int = 0,
        clip_mask: bool = True,
        is_unlabeled: bool = False,
        shuffle: bool = False,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        target_transform: Optional[Callable[[Any], torch.Tensor]] = None,
        dim: int = 2,
        slice_indices: Optional[List[np.array]] = None,
    ):

        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.clip_mask = clip_mask

        self.is_unlabeled = is_unlabeled

        self._current_image = None
        self._current_mask = None
        self._current_image_index = None
        self.cache_size = cache_size

        manager = Manager()

        # since the PyTorch dataloader uses multiple processes for data loading (if num_workers > 0),
        # a shared dict is used to share the cache between all processes have to use
        # see https://github.com/ptrblck/pytorch_misc/blob/master/shared_dict.py and
        # https://discuss.pytorch.org/t/reuse-of-dataloader-worker-process-and-caching-in-dataloader/30620/14
        # for more information
        self.image_cache = manager.dict()
        self.mask_cache = manager.dict()

        self.shuffle = shuffle

        self.transform = transform
        self.target_transform = target_transform

        self.dim = dim

        self.image_slice_indices = BraTSDataset.__arange_image_slice_indices(
            filepaths=self.image_paths,
            dim=self.dim,
            shuffle=self.shuffle,
            seed=42,
            slice_indices=slice_indices,
        )

        self.start_index = 0
        self.end_index = self.__len__()
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
            per_worker = int(
                math.ceil(
                    (self.end_index - self.start_index) / float(worker_info.num_workers)
                )
            )
            worker_id = worker_info.id
            self.start_index = self.start_index + worker_id * per_worker
            self.current_index = self.start_index
            self.end_index = min(self.start_index + per_worker, self.end_index)
        return self

    def __load_image_and_mask(self, image_index: int) -> None:
        """
        Loads image with the given index either from cache or from disk.

        Args:
            image_index (int): Index of the image to load.
        """

        self._current_image_index = image_index

        # check if image and mask are in cache
        if image_index in self.image_cache and image_index in self.mask_cache:
            self._current_image = self.image_cache[image_index]
            self._current_mask = self.mask_cache[image_index]
        # read image and mask from disk otherwise
        else:
            self._current_image = self.__read_image_as_array(
                self.image_paths[image_index], norm=True
            )
            self._current_mask = self.__read_image_as_array(
                self.annotation_paths[image_index], norm=False, clip=self.clip_mask
            )

        # cache image and mask if there is still space in cache
        if len(self.image_cache.keys()) < self.cache_size:
            self.image_cache[image_index] = self._current_image
            self.mask_cache[image_index] = self._current_mask

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current_index >= self.end_index:
            raise StopIteration

        image_slice_index = self.image_slice_indices[self.current_index]
        image_index = image_slice_index[0]

        if image_index != self._current_image_index:
            self.__load_image_and_mask(image_index)

        case_id = self.__get_case_id(
            filepath=self.image_paths[self._current_image_index]
        )

        if self.dim == 2:
            slice_index = image_slice_index[1]

            x = torch.from_numpy(self._current_image[slice_index, :, :])
            y = torch.from_numpy(self._current_mask[slice_index, :, :])
        else:
            x = torch.from_numpy(self._current_image)
            y = torch.from_numpy(self._current_mask)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        if self.is_unlabeled:
            return (
                torch.unsqueeze(x, 0),
                f"{case_id}-{slice_index}" if self.dim == 2 else case_id,
            )

        self.current_index += 1

        return torch.unsqueeze(x, 0), torch.unsqueeze(y, 0), case_id

    def __len__(self) -> int:
        return len(self.image_slice_indices)

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

            image_index = self.image_paths.index(image_path)

            new_image_slice_indices = [
                (image_index, slice_index)
                for slice_index in range(
                    BraTSDataset.IMAGE_DIMENSIONS[0] if self.dim == 2 else 1
                )
            ]

            if self.shuffle:
                random.shuffle(new_image_slice_indices)

            # add new image slice indices to existing ones
            self.image_slice_indices = (
                self.image_slice_indices + new_image_slice_indices
            )
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
            image_index = self.image_paths.index(image_path)

            self.image_slice_indices = [
                (index, slice_index)
                for (index, slice_index) in self.image_slice_indices
                if image_index != index
            ]
        else:
            raise ValueError("Image does not belong to this dataset.")
