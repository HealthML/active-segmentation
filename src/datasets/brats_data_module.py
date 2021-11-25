""" Module containing the data module for brats data """
import os
import random
from typing import Any, List, Optional, Tuple
from torch.utils.data import Dataset

from datasets.data_module import ActiveLearningDataModule
from datasets.brats_dataset import BraTSDataset


class BraTSDataModule(ActiveLearningDataModule):
    """
    Initializes the BraTS data module.
    Args:
        data_dir: Path of the directory that contains the data.
        batch_size: Batch size.
        num_workers: Number of workers for DataLoader.
        shuffle: Flag if the data should be shuffled.
        **kwargs: Further, dataset specific parameters.
    """

    # pylint: disable=unused-argument,no-self-use
    @staticmethod
    def discover_paths(
        dir_path: str,
        modality: str = "flair",
        random_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Discover the .nii.gz file paths with a given modality

        Args:
            dir_path: directory to discover paths in
            modality: modality of scan
            random_samples: the amount of random samples from the data sets

        Returns:
            list of files as tuple of image paths, annotation paths
        """
        cases = sorted(os.listdir(dir_path))
        cases = [
            case
            for case in cases
            if not case.startswith(".") and os.path.isdir(os.path.join(dir_path, case))
        ]

        if random_samples is not None and random_samples < len(cases):
            random.seed(42)
            cases = random.sample(cases, random_samples)

        image_paths = [
            os.path.join(dir_path, case, f"{os.path.basename(case)}_{modality}.nii.gz")
            for case in cases
        ]
        annotation_paths = [
            os.path.join(dir_path, case, f"{os.path.basename(case)}_seg.nii.gz")
            for case in cases
        ]

        return image_paths, annotation_paths

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool = True,
        **kwargs,
    ):

        super().__init__(data_dir, batch_size, num_workers, shuffle, **kwargs)
        self.data_folder = self.data_dir

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """TBD"""
        # ToDo: implement labeling logic
        return None

    def _create_training_set(self) -> Optional[Dataset]:
        """Creates a training dataset."""
        train_image_paths, train_annotation_paths = BraTSDataModule.discover_paths(
            os.path.join(self.data_folder, "train")
        )
        return BraTSDataset(
            image_paths=train_image_paths, annotation_paths=train_annotation_paths
        )

    def _create_validation_set(self) -> Optional[Dataset]:
        """Creates a validation dataset."""
        val_image_paths, val_annotation_paths = BraTSDataModule.discover_paths(
            os.path.join(self.data_folder, "val")
        )
        return BraTSDataset(
            image_paths=val_image_paths, annotation_paths=val_annotation_paths
        )

    def _create_test_set(self) -> Optional[Dataset]:
        # faked test set
        # ToDo: implement test set
        return self._create_validation_set()

    def _create_unlabeled_set(self) -> Optional[Dataset]:
        # faked unlabeled set
        # ToDo: implement unlabeled set
        unlabeled_set = self._create_training_set()
        unlabeled_set.is_unlabeled = True
        return unlabeled_set
