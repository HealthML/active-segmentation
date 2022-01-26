""" Module to load pascal voc data """
import os
from typing import Any, Dict, List, Optional
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import torch

from datasets.data_module import ActiveLearningDataModule
from .pascal_voc_dataset import PascalVOCDataset


class PILMaskToTensor:
    """TBD"""

    # pylint: disable=too-few-public-methods
    def __call__(self, target):
        target = np.array(target)
        target = np.where(target == 255, 0, target)
        return torch.as_tensor(target, dtype=torch.int64)


class PascalVOCDataModule(ActiveLearningDataModule):
    """
    The PASCAL Visual Object Classes (VOC) 2012 dataset contains 20 object categories including vehicles, household,
    animals, and other: aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant,
    sofa, TV/monitor, bird, cat, cow, dog, horse, sheep, and person. Each image in this dataset has pixel-level
    segmentation annotations, bounding box annotations, and object class annotations.
    Further information: http://host.robots.ox.ac.uk/pascal/VOC/
    Args:
        data_dir: Path of the directory that contains the data.
        batch_size: Batch size.
        um_workers: Number of workers for DataLoader.
        shuffle: Flag if the data should be shuffled.
        **kwargs: Further, dataset specific parameters.
    """

    # pylint: disable=unused-argument,no-self-use,too-few-public-methods
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool = True,
        **kwargs
    ):

        super().__init__(data_dir, batch_size, num_workers, shuffle, **kwargs)

        self.data_folder = os.path.join(data_dir, "voc-segmentation")
        self.__download_dataset = not os.path.exists(self.data_folder)

        self.__image_transformation = transforms.Compose(
            [transforms.Resize((255, 255)), transforms.ToTensor()]
        )
        self.__annotation_transformation = transforms.Compose(
            [transforms.Resize((255, 255)), PILMaskToTensor()]
        )
        self.__training_set_size = 4
        self.__validation_set_size = 4

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """TBD"""
        # ToDo: implement labeling logic
        return None

    def _create_training_set(self) -> Optional[Dataset]:
        """Creates a training dataset."""
        training_set = PascalVOCDataset(
            self.data_folder,
            year="2012",
            image_set="train",
            download=self.__download_dataset,
            transform=self.__image_transformation,
            target_transform=self.__annotation_transformation,
        )
        return random_split(
            training_set,
            [self.__training_set_size, len(training_set) - self.__training_set_size],
        )[0]

    def _create_validation_set(self) -> Optional[Dataset]:
        """Creates a validation dataset."""
        validation_set = PascalVOCDataset(
            self.data_folder,
            year="2012",
            image_set="val",
            download=self.__download_dataset,
            transform=self.__image_transformation,
            target_transform=self.__annotation_transformation,
        )
        return random_split(
            validation_set,
            [
                self.__validation_set_size,
                len(validation_set) - self.__validation_set_size,
            ],
        )[0]

    def _create_test_set(self) -> Optional[Dataset]:
        # faked test set
        # ToDo: implement test set
        return self._create_validation_set()

    def _create_unlabeled_set(self) -> Optional[Dataset]:
        # faked unlabeled set
        # ToDo: implement unlabeled set
        return self._create_training_set()

    def id_to_class_names(self) -> Dict[int, str]:
        return {}

    def multi_label(self) -> bool:
        return False
