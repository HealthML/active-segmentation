""" Module to load and batch bcss dataset """
from typing import List, Tuple, Union
from pathlib import Path
import math

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
            target_label (int, optional): The label to use for learning. Following labels are in the annonations:
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
            is_unlabeled (bool, optional): Wether the dataset is used as "unlabeled" for the active learning loop.
            shuffle (bool, optional): Whether the data should be shuffled.
            dim (int, optional): Dimension of the dataset.
            image_shape (tuple, optional): The shape to size the image.
    """

    # pylint: disable=too-many-instance-attributes,abstract-method

    @staticmethod
    def __ensure_channel_dim(img: torch.Tensor, dim: int) -> torch.Tensor:
        """Ensures the correct dimensionality of the image"""
        return img if len(img.shape) == dim + 1 else torch.unsqueeze(img, 0)

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
        target_label: int = 1,
        is_unlabeled: bool = False,
        shuffle: bool = True,
        dim: int = 2,
        image_shape: tuple = (240, 240),
    ) -> None:

        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.target_label = target_label
        self.dim = dim
        self.image_shape = image_shape

        self.num_images = len(self.image_paths)
        self.num_masks = len(self.annotation_paths)
        assert self.num_images == self.num_masks

        self.is_unlabeled = is_unlabeled

        self._current_image = None
        self._current_mask = None
        self._current_image_index = None

        self.start_index = 0
        self.end_index = self.num_images
        self.current_index = 0

        self.indices = np.arange(self.num_images)
        if shuffle:
            np.random.shuffle(self.indices)

    def __load_image_and_mask(self, index: int) -> None:
        """Loads images and annotations into _current_image and _current_mask variables."""
        self._current_image_index = index
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

    def __load_image_as_array(
        self, filepath: str, norm: bool = True, is_mask: bool = False
    ) -> np.ndarray:
        """Loads one image in memory."""
        img = Image.open(filepath).resize((self.image_shape[0], self.image_shape[1]))
        if self.dim == 2:
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

    def __align_axis(self, img: torch.Tensor):
        """Align the axes of the image based on the dimension"""
        if self.dim == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.moveaxis(img, 2, 0)
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
        if self.current_index >= self.end_index:
            raise StopIteration

        index = self.indices[self.current_index]
        case_id = self.get_case_id(filepath=self.image_paths[index].as_posix())

        self.__load_image_and_mask(index=index)
        x = torch.from_numpy(self._current_image)
        y = torch.from_numpy(self._current_mask)

        if self.is_unlabeled:
            return BCSSDataset.__ensure_channel_dim(x, self.dim), case_id

        self.current_index += 1
        return (
            BCSSDataset.__ensure_channel_dim(x, self.dim),
            BCSSDataset.__ensure_channel_dim(y, self.dim),
            case_id,
        )

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.num_images

    def slices_per_image(self) -> List[int]:
        """For each image returns the number of slices"""
        return [1 if self.dim == 2 else 3] * len(self.indices)

    def image_ids(self) -> List[str]:
        """For each image returns the case ID's"""
        return [self.get_case_id(filepath=path) for path in self.image_paths]
