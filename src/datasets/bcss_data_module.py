"""Module containing the data module for the BCSS dataset"""
from typing import Tuple, List, Optional, Any
from pathlib import Path
import os

from torch.utils.data import Dataset, DataLoader

from datasets.data_module import ActiveLearningDataModule
from datasets.bcss_dataset import BCSSDataset


class BCSSDataModule(ActiveLearningDataModule):
    """
    Initializes the BCSS data module.
    Args:
        data_dir: Path of the directory that contains the data.
        batch_size: Batch size.
        num_workers: Number of workers for DataLoader.
        cache_size (int, optional): Number of images to keep in memory between epochs to speed-up data loading
            (default = 0).
        pin_memory (bool, optional): `pin_memory` parameter as defined by the PyTorch `DataLoader` class.
        shuffle: Flag if the data should be shuffled.
        image_shape (tuple, optional): The shape to size the image and set the dimension.
        **kwargs: Further, dataset specific parameters.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        shuffle: bool = True,
        image_shape: tuple = (240, 240, 3),
        **kwargs,
    ):

        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            **kwargs,
        )
        self.data_dir = self.data_dir
        self.image_shape = image_shape

    @staticmethod
    def discover_paths(image_dir: str, mask_dir: str) -> Tuple[List[Path], List[Path]]:
        """
        Discover the '.png' files in a given directory.
        Args:
            image_dir: The directory to the images.
            mask_dir: The directory to the annotations.

        Returns:
            list of file paths as tuple of image paths, annotation paths
        """
        image_paths = list(Path(image_dir).glob("*.png"))
        image_ids = [BCSSDataset.get_case_id(path) for path in image_paths]
        annotation_paths = [
            path
            for path in Path(mask_dir).glob("*.png")
            if BCSSDataset.get_case_id(path) in image_ids
        ]
        return image_paths, annotation_paths

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """TBD"""
        raise NotImplementedError

    def _create_training_set(self) -> Optional[Dataset]:
        """Creates a training dataset."""
        train_image_paths, train_annotation_paths = BCSSDataModule.discover_paths(
            image_dir=os.path.join(self.data_dir, "train", "images"),
            mask_dir=os.path.join(self.data_dir, "train", "masks"),
        )
        return BCSSDataset(
            image_paths=train_image_paths,
            annotation_paths=train_annotation_paths,
            shuffle=self.shuffle,
            image_shape=self.image_shape,
        )

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Returns:
            Pytorch dataloader or Keras sequence representing the training set.
        """

        # disable shuffling in the dataloader since the BCSS dataset is a subclass of
        # IterableDataset and implements it's own shuffling
        if self._training_set:
            return DataLoader(
                self._training_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return None

    def _create_validation_set(self) -> Optional[Dataset]:
        """Creates a validation dataset."""
        val_image_paths, val_annotation_paths = BCSSDataModule.discover_paths(
            image_dir=os.path.join(self.data_dir, "val", "images"),
            mask_dir=os.path.join(self.data_dir, "val", "masks"),
        )
        return BCSSDataset(
            image_paths=val_image_paths,
            annotation_paths=val_annotation_paths,
            shuffle=self.shuffle,
            image_shape=self.image_shape,
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
