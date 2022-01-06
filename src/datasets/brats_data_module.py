""" Module containing the data module for brats data """
import os
import random
from typing import Any, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .data_module import ActiveLearningDataModule
from .brats_dataset import BraTSDataset


class BraTSDataModule(ActiveLearningDataModule):
    """
    Initializes the BraTS data module.

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
        shuffle: Flag if the data should be shuffled.
        dim: 2 or 3 to define if the datsets should return 2d slices of whole 3d images.
        random_state (int): Random constant for shuffling the data
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
            modality (string, optional): modality of scan
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

    @staticmethod
    def __case_id_to_filepaths(
        case_id: str, dir_path: str, modality: str = "flair"
    ) -> Tuple[str, str]:
        """
        Returns the image and annotation file path for a given case ID.

        Args:
            case_id: Case ID for which the file paths are to be determined.
            dir_path: directory to where the images are located
            modality (string, optional): modality of scan.

        Returns:
            Tuple[str]: Image and annotation path.
        """
        image_path = os.path.join(dir_path, case_id, f"{case_id}_{modality}.nii.gz")
        annotation_path = os.path.join(dir_path, case_id, f"{case_id}_seg.nii.gz")
        return image_path, annotation_path

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        cache_size: int = 0,
        active_learning_mode: bool = False,
        initial_training_set_size: int = 1,
        pin_memory: bool = True,
        shuffle: bool = True,
        dim: int = 2,
        random_state: int = 42,
        **kwargs,
    ):

        super().__init__(
            data_dir,
            batch_size,
            num_workers,
            active_learning_mode,
            initial_training_set_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            **kwargs,
        )
        self.data_folder = self.data_dir
        self.dim = dim
        self.cache_size = cache_size
        self.random_state = random_state

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """Moves the given samples from the unlabeled dataset to the labeled dataset."""

        if self.dim == 2:
            # create list of files as tuple of image id and slice index
            image_slice_ids = [
                (case_id.split("-")[0], case_id.split("-")[1]) for case_id in ids
            ]
            ids = [image_id for image_id, slice_index in image_slice_ids]

        if self._training_set is not None and self._unlabeled_set is not None:
            labeled_image_and_annotation_paths = [
                self.__case_id_to_filepaths(
                    case_id, os.path.join(self.data_folder, "train")
                )
                for case_id in ids
            ]
            for index, (
                labeled_image_path,
                labeled_image_annotation_path,
            ) in enumerate(labeled_image_and_annotation_paths):
                if self.dim == 2:
                    # additionally pass slice index for dimension 2
                    slice_index = int(image_slice_ids[index][1])
                else:
                    # 3D images only have one slice index of 0
                    slice_index = 0

                self._training_set.add_image(
                    labeled_image_path, labeled_image_annotation_path, slice_index
                )
                self._unlabeled_set.remove_image(
                    labeled_image_path, labeled_image_annotation_path, slice_index
                )

    def _create_training_set(self) -> Optional[Dataset]:
        """
        Creates a training dataset.
        """

        train_image_paths, train_annotation_paths = BraTSDataModule.discover_paths(
            os.path.join(self.data_folder, "train")
        )

        if self.active_learning_mode:
            # initialize the training set with randomly selected samples
            (train_image_paths, _, train_annotation_paths, _) = train_test_split(
                train_image_paths,
                train_annotation_paths,
                train_size=self.initial_training_set_size,
                random_state=self.random_state,
            )

        return BraTSDataset(
            image_paths=train_image_paths,
            annotation_paths=train_annotation_paths,
            dim=self.dim,
            cache_size=self.cache_size,
            shuffle=self.shuffle,
        )

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Returns:
            Pytorch dataloader or Keras sequence representing the training set.
        """

        # disable shuffling in the dataloader since the BraTS dataset is a subclass of
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
        val_image_paths, val_annotation_paths = BraTSDataModule.discover_paths(
            os.path.join(self.data_folder, "val")
        )
        return BraTSDataset(
            image_paths=val_image_paths,
            annotation_paths=val_annotation_paths,
            dim=self.dim,
            cache_size=self.cache_size,
        )

    def _create_test_set(self) -> Optional[Dataset]:
        """Creates a test dataset."""

        test_image_paths, test_annotation_paths = BraTSDataModule.discover_paths(
            os.path.join(self.data_folder, "test")
        )
        return BraTSDataset(
            image_paths=test_image_paths,
            annotation_paths=test_annotation_paths,
            dim=self.dim,
            cache_size=self.cache_size,
        )

    def _create_unlabeled_set(self) -> Optional[Dataset]:
        """Creates an unlabeled dataset."""
        if self.active_learning_mode:
            train_image_paths, train_annotation_paths = BraTSDataModule.discover_paths(
                os.path.join(self.data_folder, "train")
            )

            # use all images that are not in initial training set
            (
                _,
                initial_unlabeled_image_paths,
                _,
                initial_unlabeled_annotation_paths,
            ) = train_test_split(
                train_image_paths,
                train_annotation_paths,
                train_size=self.initial_training_set_size,
                random_state=self.random_state,
            )

            return BraTSDataset(
                image_paths=initial_unlabeled_image_paths,
                annotation_paths=initial_unlabeled_annotation_paths,
                dim=self.dim,
                cache_size=self.cache_size,
                is_unlabeled=True,
                shuffle=self.shuffle,
            )

        # unlabeled set is empty
        unlabeled_set = self._create_training_set()
        unlabeled_set.is_unlabeled = True
        return unlabeled_set
