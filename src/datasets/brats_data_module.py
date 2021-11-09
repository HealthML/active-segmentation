""" Module containing the data module for brats data """
import os
from typing import Any, List, Optional, Union
from torch.utils.data import Dataset

from datasets.data_module import ActiveLearningDataModule
from datasets.brats_dataset import BraTSDataset


class BraTSDataModule(ActiveLearningDataModule):
    """Brats data module"""

    @staticmethod
    def discover_paths(dir_path: str, modality="flair"):
        """
        Discover the .nii.gz file paths with a given modality
        :param dir_path: directory to discover paths in
        :param modality: modality of scan
        :return: list of files as tuple of image paths, annotation paths
        """
        cases = sorted(os.listdir(dir_path))
        cases = [case for case in cases if not case.startswith(".")]

        image_paths = [
            os.path.join(
                dir_path, case, f"{os.path.basename(case)}_{modality}.nii.gz"
            )
            for case in cases
        ]
        annotation_paths = [
            os.path.join(dir_path, case, f"{os.path.basename(case)}_seg.nii.gz")
            for case in cases
        ]

        return image_paths, annotation_paths

    def __init__(self, data_dir: str, batch_size, shuffle=True, **kwargs):
        """
        :param data_dir: Path of the directory that contains the data.
        :param batch_size: Batch size.
        :param kwargs: Further, dataset specific parameters.
        """

        super().__init__(data_dir, batch_size, shuffle, **kwargs)

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        # ToDo: implement labeling logic
        return None

    def _create_training_set(self) -> Union[Dataset, None]:
        train_image_paths, train_annotation_paths = BraTSDataModule.discover_paths(
            os.path.join(self.data_dir, "train")
        )
        return BraTSDataset(
            image_paths=train_image_paths, annotation_paths=train_annotation_paths
        )

    def _create_validation_set(self) -> Union[Dataset, None]:
        val_image_paths, val_annotation_paths = BraTSDataModule.discover_paths(
            os.path.join(self.data_dir, "val")
        )
        return BraTSDataset(
            image_paths=val_image_paths, annotation_paths=val_annotation_paths
        )

    def _create_test_set(self) -> Union[Dataset, None]:
        # faked test set
        # ToDo: implement test set
        return self._create_validation_set()

    def _create_unlabeled_set(self) -> Union[Dataset, None]:
        # faked unlabeled set
        # ToDo: implement unlabeled set
        return self._create_training_set()
