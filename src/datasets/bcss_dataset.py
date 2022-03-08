""" Module to load and batch bcss dataset """
from typing import List, Optional, Tuple, Union
from pathlib import Path
import math

from multiprocessing import Manager

from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import IterableDataset


class BCSSDataset(IterableDataset):
    """
    The BCSS dataset contains over 20,000 segmentation annotations of tissue region from breast cancer images from TCGA.
    Detailed description can be found either here: https://github.com/PathologyDataScience/BCSS
    or here: https://bcsegmentation.grand-challenge.org
        Args:
            image_paths (List[Path]): List with all images to load, can be obtained by BCSSDataModule.discover_paths.
            annotation_paths (List[Path]): List with all annotations to load,
                can be obtained by BCSSDataModule.discover_paths.
            target_label (int, optional): The label to use for learning. Following labels are in the annotations:
                outside_roi	0
                tumor	1
                stroma	2
                lymphocytic_infiltrate	3
                necrosis_or_debris	4
                glandular_secretions	5
                blood	6
                exclude	7
                metaplasia_NOS	8
                fat	9
                plasma_cells	10
                other_immune_infiltrate	11
                mucoid_material	12
                normal_acinus_or_duct	13
                lymphatics	14
                undetermined	15
                nerve	16
                skin_adnexa	17
                blood_vessel	18
                angioinvasion	19
                dcis	20
                other	21
            is_unlabeled (bool, optional): Whether the dataset is used as "unlabeled" for the active learning loop.
            shuffle (bool, optional): Whether the data should be shuffled.
            channels (int, optional): Number of channels of the images. 3 means RGB, 2 means greyscale.
            image_shape (tuple, optional): Shape of the image.
            random_state (int, optional): Controls the data shuffling. Pass an int for reproducible output across
                multiple runs.
    """

    # pylint: disable=too-many-instance-attributes,abstract-method

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """
        Normalizes an image by
            1. Dividing by the mean value
            2. Subtracting the std
        Args:
            img: The input image that should be normalized.

        Returns:
            Normalized image with background values normalized to -1
        """

        return (img - np.mean(img)) / np.std(img)

    @staticmethod
    def __align_axis(img: np.ndarray) -> np.ndarray:
        """Align the axes of the image based on the dimension"""
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        if img.shape[2] == 3:
            img = np.moveaxis(img, 2, 0)
        return img

    @staticmethod
    def get_case_id(filepath: Union[str, Path]) -> str:
        """Gets the case ID for a given filepath."""
        return Path(filepath).name.split("_")[0]

    @staticmethod
    def get_institute_name(filepath: Union[str, Path]) -> str:
        """Gets the name of the institute which donated the image."""
        return Path(filepath).name.split("-")[1]

    def __init__(
        self,
        image_paths: List[Path],
        annotation_paths: List[Path],
        cache_size: int = 0,
        target_label: int = 1,
        is_unlabeled: bool = False,
        shuffle: bool = True,
        channels: int = 3,
        image_shape: tuple = (300, 300),
        random_state: Optional[int] = None,
    ) -> None:

        super().__init__()

        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.target_label = target_label
        self.channels = channels
        self.image_shape = tuple(image_shape)
        self.cache_size = cache_size

        manager = Manager()

        # since the PyTorch dataloader uses multiple processes for data loading (if num_workers > 0),
        # a shared dict is used to share the cache between all processes have to use
        # see https://github.com/ptrblck/pytorch_misc/blob/master/shared_dict.py and
        # https://discuss.pytorch.org/t/reuse-of-dataloader-worker-process-and-caching-in-dataloader/30620/14
        # for more information
        self.image_cache = manager.dict()
        self.mask_cache = manager.dict()

        self.num_images = len(self.image_paths)
        self.num_masks = len(self.annotation_paths)
        assert self.num_images == self.num_masks

        self.is_unlabeled = is_unlabeled

        self._current_image = None
        self._current_mask = None
        self._current_image_index = None

        self.indices = list(np.arange(self.num_images))
        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(self.indices)

        self.start_index = 0
        self.end_index = self.__len__()
        self.current_index = 0

    def __load_image_and_mask(self, index: int) -> None:
        """Loads images and annotations into _current_image and _current_mask variables."""

        self._current_image_index = index
        # check if image and mask are in cache
        if index in self.image_cache and index in self.mask_cache:
            self._current_image = self.image_cache[index]
            self._current_mask = self.mask_cache[index]
        # read image and mask from disk otherwise
        else:
            self._current_image = self.__load_image_as_array(
                filepath=self.image_paths[self._current_image_index].as_posix(),
                norm=True,
                is_mask=False,
            )
            self._current_mask = self.__load_image_as_array(
                filepath=self.annotation_paths[self._current_image_index].as_posix(),
                norm=False,
                is_mask=True,
            )
        # cache image and mask if there is still space in cache
        if len(self.image_cache.keys()) < self.cache_size:
            self.image_cache[index] = self._current_image
            self.mask_cache[index] = self._current_mask

    def __load_image_as_array(
        self, filepath: str, norm: bool = True, is_mask: bool = False
    ) -> np.ndarray:
        """Loads one image in memory."""
        img = Image.open(filepath).resize((self.image_shape[0], self.image_shape[1]))
        if self.channels == 2:
            img = ImageOps.grayscale(img)
        img = np.asarray(img)
        if norm:
            img = BCSSDataset.normalize(img=img)
        if is_mask:
            img = self.__restrict_on_target_class(img=img)
        img = self.__align_axis(img)
        return img

    def __restrict_on_target_class(self, img: np.ndarray) -> np.ndarray:
        """Only keeps set target class and sets rest of the image to background with value 0."""
        img[np.where(img != self.target_label)] = 0
        return img

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

    def __next__(
        self,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, str], Tuple[torch.Tensor, str]]:
        """One iteration yields a tuple of image, annotation, case id"""
        if self.current_index >= self.__len__():
            raise StopIteration

        index = self.indices[self.current_index]
        case_id = self.get_case_id(filepath=self.image_paths[index].as_posix())

        self.__load_image_and_mask(index=index)
        x = torch.from_numpy(self._current_image)
        y = torch.from_numpy(self._current_mask)
        self.current_index += 1

        if self.is_unlabeled:
            return x, case_id

        return x, y, False, case_id

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.indices)

    def add_image(self, image_path: Path, annotation_path: Path) -> None:
        """
        Adds an image to this dataset.
        Args:
            image_path: Path of the image to be added.
            annotation_path: Path of the annotation of the image to be added.

        Returns:
            None. Raises ValueError if image already exists.
        """

        if image_path not in self.image_paths:
            self.image_paths.append(image_path)
        if annotation_path not in self.annotation_paths:
            self.annotation_paths.append(annotation_path)

        image_index = self.image_paths.index(image_path)

        if image_index not in self.indices:
            # add new image index to existing ones
            self.indices.append(image_index)
        else:
            raise ValueError("The image already belongs to this dataset.")

    def remove_image(self, image_path: Path, annotation_path: Path) -> None:
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
            self.indices = list(np.arange(self.num_images))
        else:
            raise ValueError("Image does not belong to this dataset.")

    def slices_per_image(self) -> List[int]:
        """For each image returns the number of slices"""
        return [1] * len(self.indices)

    def image_ids(self) -> List[str]:
        """For each image returns the case ID's"""
        return [self.get_case_id(filepath=path) for path in self.image_paths]

    def size(self) -> int:
        """
        Returns:
            int: Size of the dataset.
        """

        return self.__len__()
