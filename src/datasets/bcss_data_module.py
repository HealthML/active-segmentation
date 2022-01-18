"""Module containing the data module for the BCSS dataset"""
import shutil
from typing import Tuple, List, Optional, Any
from pathlib import Path
import os

from fire import Fire
from sklearn.model_selection import train_test_split
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
        shuffle (bool, optional): Flag if the data should be shuffled.
        channels (int, optional): Number of channels of the images. 3 means RGB, 2 means greyscale.
        image_shape (tuple, optional): Shape of the image.
        target_label (int, optional): The label to use for learning. Details are in BCSSDataset.
        val_set_size (float, optional): The size of the validation set, e.g. 0.3.
        stratify (bool, optional): The option to stratify the train val split by the institutes.
        **kwargs: Further, dataset specific parameters.
    """

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

    @staticmethod
    def build_stratification_labels(image_paths: List[Path]) -> List[str]:
        """Build a list with class labels used for a stratified split"""
        institute_names = [BCSSDataset.get_institute_name(path) for path in image_paths]
        stratify = [
            name if institute_names.count(name) > 1 else "OTHER"
            for name in institute_names
        ]
        return stratify

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        shuffle: bool = True,
        cache_size: int = 0,
        channels: int = 3,
        image_shape: tuple = (300, 300),
        target_label: int = 1,
        val_set_size: float = 0.3,
        stratify: bool = True,
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
        self.channels = channels
        self.image_shape = tuple(image_shape)
        self.target_label = target_label
        self.cache_size = cache_size
        self.val_set_size = val_set_size
        self.stratify = stratify
        self.split = {}

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """TBD"""
        raise NotImplementedError

    def _get_train_and_val_paths(self) -> None:
        """Discovers the directory and splits into a train and a test dataset"""
        image_paths, annotation_paths = BCSSDataModule.discover_paths(
            image_dir=os.path.join(self.data_dir, "train_val", "images"),
            mask_dir=os.path.join(self.data_dir, "train_val", "masks"),
        )
        stratify = (
            BCSSDataModule.build_stratification_labels(image_paths=image_paths)
            if self.stratify
            else None
        )
        self._split_train_val(
            image_paths=image_paths,
            annotation_paths=annotation_paths,
            stratify=stratify,
        )

    def _split_train_val(
        self,
        image_paths: List[Path],
        annotation_paths: List[Path],
        stratify: Optional[List[Any]] = None,
    ) -> None:
        """Splits the images and annotations into a train and a test set."""
        (
            self.split["train_image_paths"],
            self.split["val_image_paths"],
            self.split["train_annotation_paths"],
            self.split["val_annotation_paths"],
        ) = train_test_split(
            image_paths,
            annotation_paths,
            test_size=self.val_set_size,
            stratify=stratify,
            random_state=42,
        )

    def _create_training_set(self) -> Optional[Dataset]:
        """Creates a training dataset."""
        if not self.split:
            self._get_train_and_val_paths()
        return BCSSDataset(
            image_paths=self.split["train_image_paths"],
            annotation_paths=self.split["train_annotation_paths"],
            shuffle=self.shuffle,
            channels=self.channels,
            image_shape=self.image_shape,
            target_label=self.target_label,
            cache_size=self.cache_size,
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
        if not self.split:
            self._get_train_and_val_paths()
        return BCSSDataset(
            image_paths=self.split["val_image_paths"],
            annotation_paths=self.split["val_annotation_paths"],
            shuffle=self.shuffle,
            channels=self.channels,
            image_shape=self.image_shape,
            target_label=self.target_label,
            cache_size=self.cache_size,
        )

    def _create_test_set(self) -> Optional[Dataset]:
        """Creates a test dataset."""
        test_image_paths, test_annotation_paths = BCSSDataModule.discover_paths(
            image_dir=os.path.join(self.data_dir, "test", "images"),
            mask_dir=os.path.join(self.data_dir, "test", "masks"),
        )
        return BCSSDataset(
            image_paths=test_image_paths,
            annotation_paths=test_annotation_paths,
            shuffle=False,
            channels=self.channels,
            image_shape=self.image_shape,
            target_label=self.target_label,
            cache_size=self.cache_size,
        )

    def _create_unlabeled_set(self) -> Optional[Dataset]:
        # faked unlabeled set
        # ToDo: implement unlabeled set
        unlabeled_set = self._create_training_set()
        unlabeled_set.is_unlabeled = True
        return unlabeled_set

    def data_channels(self) -> int:
        """Returns the number of channels"""
        return self.channels


def copy_test_set_to_separate_folder(source_dir: str, target_dir: str) -> None:
    """
    Reproduces the test set used in the baseline implementation of the challenge, by copying the scans of the
    respective institution into a separate folder.
    Args:
        source_dir (str): Directory where all the downloaded images and masks are stored.
        target_dir (str): Directory where to store the test data.
    """

    test_set_institutes = ["OL", "LL", "E2", "EW", "GM", "S3"]
    image_paths = list(Path(source_dir, "images").glob("*.png"))
    annotation_paths = list(Path(source_dir, "masks").glob("*.png"))
    test_image_paths = [
        path for path in image_paths if path.name.split("-")[1] in test_set_institutes
    ]
    test_annotation_paths = [
        path
        for path in annotation_paths
        if path.name.split("-")[1] in test_set_institutes
    ]
    for image_path, mask_path in zip(test_image_paths, test_annotation_paths):
        shutil.move(
            image_path.as_posix(), os.path.join(target_dir, "images", image_path.name)
        )
        print(
            f"Moved {image_path.as_posix()} -> {os.path.join(target_dir, 'images', image_path.name)}"
        )
        shutil.move(
            mask_path.as_posix(), os.path.join(target_dir, "masks", mask_path.name)
        )
        print(
            f"Moved {mask_path.as_posix()} -> {os.path.join(target_dir, 'masks', mask_path.name)}"
        )
    print(
        f"Validation: Number of images in {target_dir}: {len(list(Path(target_dir, 'images').glob('*.png')))}"
    )
    print(
        f"Validation: Number of masks in {target_dir}: {len(list(Path(target_dir, 'masks').glob('*.png')))}"
    )


if __name__ == "__main__":
    Fire(copy_test_set_to_separate_folder)
