import numpy as np
import os
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
import torch
from typing import Any, List, Optional, Union

from .data_module import ActiveLearningDataModule


class PILMaskToTensor:
    def __call__(self, target):
        target = np.array(target)
        target = np.where(target == 255, 0, target)
        return torch.as_tensor(target, dtype=torch.int64)


class PascalVOCDataModule(ActiveLearningDataModule):
    def __init__(self, data_dir: str, batch_size, **kwargs):
        """
        :param data_dir: Path of the directory that contains the data.
        :param batch_size: Batch size.
        :param kwargs: Further, dataset specific parameters.
        """

        super().__init__(data_dir, batch_size, **kwargs)

        self.data_folder = os.path.join(data_dir, "voc-segmentation")
        self.__download_dataset = not os.path.exists(self.data_folder)

        self.__image_transformation = transforms.Compose([transforms.Resize((255, 255)), transforms.ToTensor()])
        self.__annotation_transformation = transforms.Compose([transforms.Resize((255, 255)), PILMaskToTensor()])
        self.__training_set_size = 4
        self.__validation_set_size = 4

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        # ToDo: implement labeling logic
        return None

    def _create_training_set(self) -> Union[Dataset, None]:
        training_set = datasets.VOCSegmentation(self.data_folder,
                                                year="2012",
                                                image_set="train",
                                                download=self.__download_dataset,
                                                transform=self.__image_transformation,
                                                target_transform=self.__annotation_transformation)
        return random_split(training_set, [self.__training_set_size, len(training_set) - self.__training_set_size])[0]

    def _create_validation_set(self) -> Union[Dataset, None]:
        validation_set = datasets.VOCSegmentation(self.data_folder,
                                                  year="2012",
                                                  image_set="val",
                                                  download=self.__download_dataset,
                                                  transform=self.__image_transformation,
                                                  target_transform=self.__annotation_transformation)
        return random_split(validation_set, [self.__validation_set_size, len(validation_set) - self.__validation_set_size])[0]

    def _create_test_set(self) -> Union[Dataset, None]:
        # faked test set
        # ToDo: implement test set
        return self._create_validation_set()

    def _create_unlabeled_set(self) -> Union[Dataset, None]:
        # faked unlabeled set
        # ToDo: implement unlabeled set
        return self._create_training_set()
