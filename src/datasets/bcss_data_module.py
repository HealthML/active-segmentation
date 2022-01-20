"""Module containing the data module for the BCSS dataset"""
import shutil
from typing import Tuple, List, Optional, Any, Union
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
        active_learning_mode (bool, optional): Whether the datamodule should be configured for active learning or for
            conventional model training (default = False).
        initial_training_set_size (int, optional): Initial size of the training set if the active learning mode is
            activated.
        pin_memory (bool, optional): `pin_memory` parameter as defined by the PyTorch `DataLoader` class.
        shuffle (bool, optional): Flag if the data should be shuffled.
        channels (int, optional): Number of channels of the images. 3 means RGB, 2 means greyscale.
        image_shape (tuple, optional): Shape of the image.
        target_label (int, optional): The label to use for learning. Details are in BCSSDataset.
        val_set_size (float, optional): The size of the validation set (default = 0.3).
        stratify (bool, optional): The option to stratify the train val split by the institutes.
        random_state (int): Random constant for shuffling the data
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

    @staticmethod
    def _case_id_to_filepaths(case_id: str, directory: str) -> Tuple[Path, Path]:
        """Generates the correct filepath to the image of the given case_id"""
        potential_image_filenames = list(
            Path(directory, "images").glob(f"{case_id}*.png")
        )
        potential_mask_filenames = list(
            Path(directory, "masks").glob(f"{case_id}*.png")
        )
        if len(potential_image_filenames) != 1 or len(potential_mask_filenames) != 1:
            raise ValueError(
                f"Error in generating image path for case id {case_id}."
                f"Found {len(potential_image_filenames)} potential image filenames"
                f"and {len(potential_mask_filenames)} potential mask filenames."
            )
        return potential_image_filenames[0], potential_mask_filenames[0]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        shuffle: bool = True,
        cache_size: int = 0,
        active_learning_mode: bool = False,
        initial_training_set_size: int = 1,
        channels: int = 3,
        image_shape: tuple = (300, 300),
        target_label: int = 1,
        val_set_size: float = 0.3,
        stratify: bool = True,
        random_state: int = 42,
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
        self.active_learning_mode = active_learning_mode
        self.initial_training_set_size = initial_training_set_size
        self.val_set_size = val_set_size
        self.stratify = stratify
        self.random_state = random_state
        self.split = {}

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """Moves the given samples from the unlabeled dataset to the labeled dataset."""

        if self._training_set is not None and self._unlabeled_set is not None:
            labeled_image_and_annotation_paths = [
                self._case_id_to_filepaths(
                    case_id=case_id[0],
                    directory=os.path.join(self.data_dir, "train_val"),
                )
                for case_id in ids
            ]
            for _, (labeled_image_path, labeled_image_annotation_path) in enumerate(
                labeled_image_and_annotation_paths
            ):
                self._training_set.add_image(
                    labeled_image_path, labeled_image_annotation_path
                )
                self._unlabeled_set.remove_image(
                    labeled_image_path, labeled_image_annotation_path
                )

    def _get_train_and_val_paths(self) -> None:
        """Discovers the directory and splits into a train and a test dataset"""
        image_paths, annotation_paths = BCSSDataModule.discover_paths(
            image_dir=os.path.join(self.data_dir, "train_val", "images"),
            mask_dir=os.path.join(self.data_dir, "train_val", "masks"),
        )
        (
            self.split["train_image_paths"],
            self.split["val_image_paths"],
            self.split["train_annotation_paths"],
            self.split["val_annotation_paths"],
        ) = self._split_train_val(
            image_paths=image_paths,
            annotation_paths=annotation_paths,
            stratify=self._stratify_split(image_paths=image_paths),
            test_size=self.val_set_size,
        )

    def _split_train_val(
        self,
        image_paths: List[Path],
        annotation_paths: List[Path],
        stratify: Optional[List[Any]] = None,
        **kwargs,
    ) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
        """Splits the images and annotations into a train and a test set."""
        (
            train_image_paths,
            val_image_paths,
            train_annotation_paths,
            val_annotation_paths,
        ) = train_test_split(
            image_paths,
            annotation_paths,
            stratify=stratify,
            random_state=self.random_state,
            **kwargs,
        )
        return (
            train_image_paths,
            val_image_paths,
            train_annotation_paths,
            val_annotation_paths,
        )

    def _stratify_split(
        self, image_paths: List[Path] = None
    ) -> Union[List[Path], None]:
        """Builds a list for stratification of a split"""
        stratify = (
            BCSSDataModule.build_stratification_labels(image_paths=image_paths)
            if self.stratify
            else None
        )
        return stratify

    def _create_training_set(self) -> Optional[Dataset]:
        """Creates a training dataset."""
        self._get_train_and_val_paths()
        if self.active_learning_mode:
            (
                self.split["train_image_paths"],
                _,
                self.split["train_annotation_paths"],
                _,
            ) = self._split_train_val(
                image_paths=self.split["train_image_paths"],
                annotation_paths=self.split["train_annotation_paths"],
                stratify=self._stratify_split(
                    image_paths=self.split["train_image_paths"]
                ),
                train_size=self.initial_training_set_size,
            )
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
        """Creates an unlabeled dataset."""
        if self.active_learning_mode:
            self._get_train_and_val_paths()
            (
                _,
                initial_unlabeled_image_paths,
                _,
                initial_unlabeled_annotation_paths,
            ) = self._split_train_val(
                image_paths=self.split["train_image_paths"],
                annotation_paths=self.split["train_annotation_paths"],
                stratify=self._stratify_split(
                    image_paths=self.split["train_image_paths"]
                ),
                train_size=self.initial_training_set_size,
            )
            return BCSSDataset(
                image_paths=initial_unlabeled_image_paths,
                annotation_paths=initial_unlabeled_annotation_paths,
                shuffle=False,
                channels=self.channels,
                image_shape=self.image_shape,
                target_label=self.target_label,
                cache_size=self.cache_size,
                is_unlabeled=True,
            )
        # Unlabeled set is empty
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
