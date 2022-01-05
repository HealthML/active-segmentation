""" Module containing the data module for decathlon data """
from io import TextIOWrapper
import json
import os
import random
from typing import Any, Callable, List, Literal, Optional, Tuple
from torch.utils.data import DataLoader, Dataset

from .collate import batch_padding_collate_fn
from .data_module import ActiveLearningDataModule
from .doubly_shuffled_nifti_dataset import DoublyShuffledNIfTIDataset


class DecathlonDataModule(ActiveLearningDataModule):
    """
    Initializes the Decathlon data module.
    Args:
        data_dir: Path of the directory that contains the data.
        batch_size: Batch size.
        num_workers: Number of workers for DataLoader.
        task (str, optional): The task from the medical segmentation decathlon.
        cache_size (int, optional): Number of images to keep in memory between epochs to speed-up data loading
            (default = 0).
        pin_memory (bool, optional): `pin_memory` parameter as defined by the PyTorch `DataLoader` class.
        shuffle: Flag if the data should be shuffled.
        dim: 2 or 3 to define if the datsets should return 2d slices of whole 3d images.
        mask_join_non_zero (bool, optional): Flag if the non zero values of the annotations should be merged.
            (default = True)
        mask_filter_values (Tuple[int], optional): Values from the annotations which should be used. Defaults to using
            all values.
        **kwargs: Further, dataset specific parameters.
    """

    @staticmethod
    def __open_dataset_file(dir_path: str) -> TextIOWrapper:
        """
        Open the dataset JSON file.

        Args:
            dir_path (str): Directory the dataset is inside.

        Returns:
            The openend file.
        """
        dataset_file_name = os.path.join(dir_path, "dataset.json")

        if not os.path.isfile(dataset_file_name):
            print("Dataset file could not be found.")
            raise FileNotFoundError(f"{dataset_file_name} is not a valid filename.")

        return open(dataset_file_name, encoding="utf-8")

    @staticmethod
    def __read_data_channels(dir_path: str) -> int:
        """
        Read the amount of data channels from the dataset JSON file.

        Args:
            dir_path (str): Directory the dataset is inside.

        Returns:
            The amount of dataset channels.
        """
        with DecathlonDataModule.__open_dataset_file(dir_path) as dataset_file:
            dataset_info = json.load(dataset_file)
            return len(dataset_info["modality"])

    @staticmethod
    def discover_paths(
        dir_path: str,
        subset: Literal["train", "val", "test"],
        random_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Discover the .nii file paths from the corresponding JSON file.

        Args:
            dir_path (str): Directory the dataset is inside.
            subset (Literal["train", "val", "test"]): The subset of paths of the whole dataset.
            random_samples (int, optional): The amount of random samples from the data set.

        Returns:
            list of files as tuple of image paths, annotation paths
        """

        with DecathlonDataModule.__open_dataset_file(dir_path) as dataset_file:
            dataset_info = json.load(dataset_file)

            if subset in ("train", "val"):
                all_cases = dataset_info["training"]

                cases = [
                    case
                    for index, case in enumerate(all_cases)
                    if (subset == "train" and index % 5 != 0)
                    or (subset == "val" and index % 5 == 0)
                ]

                if random_samples is not None and random_samples < len(cases):
                    random.seed(42)
                    cases = random.sample(cases, random_samples)

                image_paths = [
                    os.path.join(dir_path, case["image"][2:]) for case in cases
                ]
                annotation_paths = [
                    os.path.join(dir_path, case["label"][2:]) for case in cases
                ]

                return image_paths, annotation_paths

            # ToDo: Decide how to implement test dataset
            raise ValueError("Test dataset is not implemented for decathlon data yet.")

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        task: str = "Task06_Lung",
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
        self.data_folder = os.path.join(self.data_dir, task)
        self.dim = dim
        self.cache_size = cache_size
        self.mask_join_non_zero = mask_join_non_zero
        self.mask_filter_values = mask_filter_values
        self._data_channels = DecathlonDataModule.__read_data_channels(self.data_folder)

    def data_channels(self) -> int:
        """Returns the amount of data channels."""

        return self._data_channels

    def _get_collate_fn(self) -> Optional[Callable[[List[Any]], Any]]:
        """Returns the batchwise padding collate function."""

        return batch_padding_collate_fn

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """TBD"""
        # ToDo: implement labeling logic
        return None

    def _create_training_set(self) -> Optional[Dataset]:
        """Creates a training dataset."""
        train_image_paths, train_annotation_paths = DecathlonDataModule.discover_paths(
            self.data_folder, "train"
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

        # disable shuffling in the dataloader since the Decathlon dataset is a subclass of
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
        val_image_paths, val_annotation_paths = DecathlonDataModule.discover_paths(
            self.data_folder, "val"
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
