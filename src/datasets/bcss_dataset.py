""" Module to load and batch bcss dataset """
from typing import List, Tuple, Union
from pathlib import Path
import math

from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import IterableDataset

from datasets import BraTSDataset


class BCSSDataset(IterableDataset):
    """
    The BCSS dataset contains over 20,000 segmentation annotations of tissue region from breast cancer images from TCGA.
    Detailed description can be found either here: https://github.com/PathologyDataScience/BCSS
    or here: https://bcsegmentation.grand-challenge.org
        Args:
            image_paths(List[Path]): List with all images to load, can be obtained by BCSSDataModule.discover_paths.
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
            shuffle (bool, optional): Whether the data should be shuffled.
            image_shape (tuple, optional): The shape to size the image and set the dimension.
    """
    # pylint: disable=too-many-instance-attributes,abstract-method

    def __init__(
        self,
        image_paths: List[Path],
        annotation_paths: List[Path],
        target_label: int = 1,
        shuffle: bool = True,
        image_shape: tuple = (240, 240, 3),
    ) -> None:

        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.target_label = target_label
        self.image_shape = image_shape

        self.num_images = len(self.image_paths)
        self.num_masks = len(self.annotation_paths)
        assert self.num_images == self.num_masks

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
        if self.image_shape[2] == 1:
            img = ImageOps.grayscale(img)
        img = np.asarray(img)
        if norm:
            img = BraTSDataset.normalize(img=img)
        if is_mask:
            img = self.__restrict_on_target_class(img=img)
        return img

    def __restrict_on_target_class(self, img: np.ndarray) -> np.ndarray:
        """Only keeps set target class and sets rest of the image to background with value 0."""
        img[np.where(img != self.target_label)] = 0
        return img

    @staticmethod
    def get_case_id(filepath: Union[str, Path]) -> str:
        """Gets the case ID for a given filepath."""
        return Path(filepath).name.split("_")[0]

    def __iter__(self):
        """
        # TODO(mfr): Is this necessary?
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

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """One iteration yields a tuple of image, annotation, case id"""
        if self.current_index >= self.end_index:
            raise StopIteration

        index = self.indices[self.current_index]
        case_id = self.get_case_id(filepath=self.image_paths[index].as_posix())

        self.__load_image_and_mask(index=index)
        x = torch.from_numpy(self._current_image)
        y = torch.from_numpy(self._current_mask)

        self.current_index += 1
        return torch.unsqueeze(x, 0), torch.unsqueeze(y, 0), case_id

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.num_images
