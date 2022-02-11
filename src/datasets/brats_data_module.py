""" Module containing the data module for brats data """
import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from torch.utils.data import DataLoader, Dataset

from .data_module import ActiveLearningDataModule
from .doubly_shuffled_nifti_dataset import DoublyShuffledNIfTIDataset


class BraTSDataModule(ActiveLearningDataModule):
    """
    Initializes the BraTS data module.

    Args:
        data_dir (string): Path of the directory that contains the data.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        active_learning_mode (bool, optional): Whether the datamodule should be configured for active learning or for
            conventional model training (default = False).
        batch_size_unlabeled_set (int, optional): Batch size for the unlabeled set. Defaults ot :attr:`batch_size`.
        cache_size (int, optional): Number of images to keep in memory between epochs to speed-up data loading
            (default = 0).
        initial_training_set_size (int, optional): Initial size of the training set if the active learning mode is
            activated.
        pin_memory (bool, optional): `pin_memory` parameter as defined by the PyTorch `DataLoader` class.
        shuffle (boolean): Flag if the data should be shuffled.
        dim (int): 2 or 3 to define if the datsets should return 2d slices of whole 3d images.
        mask_join_non_zero (bool, optional): Flag if the non zero values of the annotations should be merged.
            (default = True)
        mask_filter_values (Tuple[int], optional): Values from the annotations which should be used. Defaults to using
            all values.
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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        active_learning_mode: bool = False,
        batch_size_unlabeled_set: Optional[int] = None,
        cache_size: int = 0,
        initial_training_set_size: int = 1,
        pin_memory: bool = True,
        shuffle: bool = True,
        dim: int = 2,
        mask_join_non_zero: bool = True,
        mask_filter_values: Optional[Tuple[int]] = None,
        random_state: int = 42,
        **kwargs,
    ):

        super().__init__(
            data_dir,
            batch_size,
            num_workers,
            active_learning_mode=active_learning_mode,
            batch_size_unlabeled_set=batch_size_unlabeled_set,
            initial_training_set_size=initial_training_set_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            **kwargs,
        )
        self.data_folder = self.data_dir
        self.dim = dim
        self.cache_size = cache_size
        self.mask_join_non_zero = mask_join_non_zero
        self.mask_filter_values = mask_filter_values

        if self.active_learning_mode:
            (
                self.initial_training_samples,
                self.initial_unlabeled_samples,
            ) = DoublyShuffledNIfTIDataset.generate_active_learning_split(
                BraTSDataModule.discover_paths(os.path.join(self.data_folder, "train"))[
                    0
                ],
                dim,
                initial_training_set_size,
                random_state,
            )
        else:
            self.initial_training_samples = None
            self.initial_unlabeled_samples = None

    def multi_label(self) -> bool:
        """
        Returns:
            bool: Whether the dataset is a multi-label or a single-label dataset.
        """

        return False

    def id_to_class_names(self) -> Dict[int, str]:
        """
        Returns:
            Dict[int, str]: A mapping of class indices to descriptive class names.
        """

        if self.mask_join_non_zero:
            return {0: "background", 1: "tumor"}

        labels = {
            0: "background",
            1: "non-enhancing tumor core",
            2: "peritumoral edema",
            4: "GD-enhancing tumor",
        }

        if self.mask_filter_values is not None:
            return {
                class_id: class_name
                for class_id, class_name in labels.items()
                if class_id in self.mask_filter_values or class_id == 0
            }
        return labels

    def label_items(
        self, ids: List[str], pseudo_labels: Optional[Dict[str, np.array]] = None
    ) -> None:
        """Moves the given samples from the unlabeled dataset to the labeled dataset."""

        # create list of files as tuple of image id and slice index
        image_slice_ids = [case_id.split("-") for case_id in ids]
        image_slice_ids = [
            (split_id[0], int(split_id[1]) if len(split_id) > 1 else None)
            for split_id in image_slice_ids
        ]

        if self.training_set is not None and self._unlabeled_set is not None:

            for case_id, (image_id, slice_id) in zip(ids, image_slice_ids):
                if self.dim == 3 and slice_id is None:
                    slice_id = 0

                if pseudo_labels is not None and case_id in pseudo_labels:
                    self.training_set.add_image(
                        image_id, slice_id, pseudo_labels[case_id]
                    )
                else:
                    self.training_set.add_image(image_id, slice_id)
                    self._unlabeled_set.remove_image(image_id, slice_id)

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
            slice_indices=self.initial_training_samples,
        )

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Returns:
            Pytorch dataloader or Keras sequence representing the training set.
        """

        # disable shuffling in the dataloader since the dataset is a subclass of
        # IterableDataset and implements it's own shuffling
        if self.training_set:
            return DataLoader(
                self.training_set,
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
            case_id_prefix="val",
        )

    def _create_test_set(self) -> Optional[Dataset]:
        # faked test set
        # ToDo: implement test set
        return self._create_validation_set()

    def _create_unlabeled_set(self) -> Optional[Dataset]:
        """Creates an unlabeled dataset."""
        if self.active_learning_mode:
            train_image_paths, train_annotation_paths = BraTSDataModule.discover_paths(
                os.path.join(self.data_folder, "train")
            )

            return DoublyShuffledNIfTIDataset(
                image_paths=train_image_paths,
                annotation_paths=train_annotation_paths,
                dim=self.dim,
                cache_size=self.cache_size,
                is_unlabeled=True,
                shuffle=self.shuffle,
                mask_join_non_zero=self.mask_join_non_zero,
                mask_filter_values=self.mask_filter_values,
                slice_indices=self.initial_unlabeled_samples,
            )

        # unlabeled set is empty
        unlabeled_set = self._create_training_set()
        unlabeled_set.is_unlabeled = True
        return unlabeled_set
