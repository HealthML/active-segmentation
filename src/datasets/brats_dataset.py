""" Module to load and batch brats dataset """
from typing import Any, Callable, List, Literal, Optional, Tuple
import math
from multiprocessing import Manager
import os

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
        image_paths: List with the paths to the images.
        annotation_paths: List with the paths to the annotations.
        cache_size (int, optional): Number of images to keep in memory to speed-up data loading in subsequent epochs.
            Defaults to zero.
        clip_mask: Flag to clip the annotation labels, if True only label 1 is kept.
        shuffle (bool, optional): Whether the data should be shuffled.
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

    @staticmethod
    def __shuffled_indices(
        dataset_size: int, slices_per_image: int, seed: Optional[int] = None
    ) -> List[int]:
        r"""
        Implements efficient shuffling for 2D image datasets like the BraTSDataset whose elements represent the slices
        of multiple 3D images. It is assumed that `dataset_size` is equal to :math:`N \cdot S` where :math:`N` is the
        number of 3D images and :math:`S` the number of 2D slices per 3D image. It is further assumed that all 2D slices
        of one 3D image have contiguous indices in the dataset. To allow for efficient image pre-fetching, first
        the order of all 3D images is shuffled and then the order of slices within each 3D image is shuffled. This way
        the 3D images can still be loaded as a whole.

        Args:
            dataset_size (int): Number of 2D images in the dataset.
            slices_per_image (int): Number of slices per 3D image.
            seed (int, optional): Random seed for shuffling.
        Returns:
            List[int]: List of shuffled indices.
        """

        if seed is not None:
            np.random.seed(seed)

        number_2d_slices = dataset_size
        number_3d_images = math.ceil(number_2d_slices) / slices_per_image
        assert number_3d_images * slices_per_image == number_2d_slices

        indices = np.arange(number_2d_slices)
        indices = np.array(np.split(indices, number_3d_images))

        # shuffle order of 3D images
        np.random.shuffle(indices)

        # shuffle 2D slice indices within each 3D image
        np.apply_along_axis(np.random.shuffle, 1, indices)

        return list(indices.flatten())

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
        self.cache_size = cache_size

        manager = Manager()

        # since the PyTorch dataloader uses multiple processes for data loading (if num_workers > 0),
        # a shared dict is used to share the cache between all processes have to use
        # see https://github.com/ptrblck/pytorch_misc/blob/master/shared_dict.py and
        # https://discuss.pytorch.org/t/reuse-of-dataloader-worker-process-and-caching-in-dataloader/30620/14
        # for more information
        self.image_cache = manager.dict()
        self.mask_cache = manager.dict()

        self.transform = transform
        self.target_transform = target_transform

        self.dimensionality = dimensionality

        self.start_index = 0
        self.end_index = self.__len__()
        self.current_index = 0

        if shuffle:
            self.shuffled_indices = BraTSDataset.__shuffled_indices(
                self.__len__(), BraTSDataset.IMAGE_DIMENSIONS[0]
            )
        else:
            self.shuffled_indices = np.arange(self.__len__())

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

        shuffled_index = self.shuffled_indices[self.current_index]
        image_index = math.floor(shuffled_index / BraTSDataset.IMAGE_DIMENSIONS[0])
        slice_index = shuffled_index - image_index * BraTSDataset.IMAGE_DIMENSIONS[0]

        if image_index != self._current_image_index:
            self.__load_image_and_mask(image_index)

        case_id = self.__get_case_id(
            filepath=self.image_paths[self._current_image_index]
        )

        if self.dimensionality == "2d":
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
