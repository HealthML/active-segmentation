""" Module containing the data module for brats data """
import os
import random
from typing import Any, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset

from .data_module import ActiveLearningDataModule
from .doubly_shuffled_nifti_dataset import DoublyShuffledNIfTIDataset


class BraTSDataModule(ActiveLearningDataModule):
    """
    Initializes the BraTS data module.

    Args:
        data_dir: Path of the directory that contains the data.
        batch_size: Batch size.
        num_workers: Number of workers for DataLoader.
        cache_size (int, optional): Number of images to keep in memory between epochs to speed-up data loading
            (default = 0).
        pin_memory (bool, optional): `pin_memory` parameter as defined by the PyTorch `DataLoader` class.
        shuffle: Flag if the data should be shuffled.
        dim: 2 or 3 to define if the datsets should return 2d slices of whole 3d images.
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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        cache_size: int = 0,
        pin_memory: bool = True,
        shuffle: bool = True,
        dim: int = 2,
        mask_join_non_zero: bool = True,
        mask_filter_values: Optional[Tuple[int]] = None,
        **kwargs,
    ):

        super().__init__(
            data_dir,
            batch_size,
            num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            **kwargs,
        )
        self.data_folder = self.data_dir
        self.dim = dim
        self.cache_size = cache_size
        self.mask_join_non_zero = mask_join_non_zero
        self.mask_filter_values = mask_filter_values

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """
        TBD
        """

        # ToDo: implement labeling logic
        return None

    def _create_training_set(self) -> Optional[Dataset]:
        """
        Creates a training dataset.
        """

        train_image_paths, train_annotation_paths = BraTSDataModule.discover_paths(
            os.path.join(self.data_folder, "train")
        )
        return DoublyShuffledNIfTIDataset(
            image_paths=train_image_paths,
            annotation_paths=train_annotation_paths,
            dim=self.dim,
            cache_size=self.cache_size,
            shuffle=self.shuffle,
            mask_join_non_zero=self.mask_join_non_zero,
            mask_filter_values=self.mask_filter_values,
        )

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Returns:
            Pytorch dataloader or Keras sequence representing the training set.
        """

        # disable shuffling in the dataloader since the dataset is a subclass of
        # IterableDataset and implements it's own shuffling
        if self._training_set:
            return DataLoader(
                self._training_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._get_collate_fn(),
            )
        return None

    def _create_validation_set(self) -> Optional[Dataset]:
        """Creates a validation dataset."""
        val_image_paths, val_annotation_paths = BraTSDataModule.discover_paths(
            os.path.join(self.data_folder, "val")
        )
        return DoublyShuffledNIfTIDataset(
            image_paths=val_image_paths,
            annotation_paths=val_annotation_paths,
            dim=self.dim,
            cache_size=self.cache_size,
            mask_join_non_zero=self.mask_join_non_zero,
            mask_filter_values=self.mask_filter_values,
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
