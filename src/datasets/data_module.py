""" Module containing abstract classes for the data modules"""
from abc import ABC, abstractmethod
import warnings
from typing import Any, Callable, List, Optional
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings(
    "ignore",
    ".*DataModule.setup has already been called, so it will not be called again.*",
)


class ActiveLearningDataModule(LightningDataModule, ABC):
    """
    Abstract base class to structure the dataset creation for active learning
    Args:
        data_dir: Path of the directory that contains the data.
        batch_size: Batch size.
        num_workers: Number of workers for DataLoader.
        active_learning_mode (bool, optional): Whether the datamodule should be configured for active learning or for
            conventional model training (default = False).
        initial_training_set_size (int, optional): Initial size of the training set if the active learning mode is
            activated.
        pin_memory (bool, optional): `pin_memory` parameter as defined by the PyTorch `DataLoader` class.
        shuffle: Flag if the data should be shuffled.
        **kwargs: Further, dataset specific parameters.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        active_learning_mode: bool = False,
        initial_training_set_size: int = 1,
        pin_memory: bool = True,
        shuffle: bool = True,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.active_learning_mode = active_learning_mode
        self.initial_training_set_size = initial_training_set_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        self._training_set = None
        self._validation_set = None
        self._test_set = None
        self._unlabeled_set = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates the datasets managed by this data module.
        Args:
            stage: Current training stage.
        """

        self._training_set = self._create_training_set()
        self._validation_set = self._create_validation_set()
        self._test_set = self._create_test_set()
        self._unlabeled_set = self._create_unlabeled_set()

    @staticmethod
    def data_channels() -> int:
        """
        Can be overwritten by subclasses if the data has multiple channels.

        Returns:
            The amount of data channels. Defaults to 1.
        """

        return 1

    @staticmethod
    def _get_collate_fn() -> Optional[Callable[[List[Any]], Any]]:
        """
        Can be overwritten by subclasses to pass a custom collate function to the dataloaders.

        Returns:
            Callable[[List[torch.Tensor]], Any] that combines batches. Defaults to None.
        """

        return None

    @abstractmethod
    def _create_training_set(self) -> Optional[Dataset]:
        """
        Returns:
            Pytorch data_module or Keras sequence representing the training set.
        """

    @abstractmethod
    def _create_validation_set(self) -> Optional[Dataset]:
        """
        Returns:
            Pytorch data_module or Keras sequence representing the validation set.
        """

    @abstractmethod
    def _create_test_set(self) -> Optional[Dataset]:
        """
        Returns:
            Pytorch data_module or Keras sequence representing the test set.
        """

    @abstractmethod
    def _create_unlabeled_set(self) -> Optional[Dataset]:
        """
        Returns:
            Pytorch data_module or Keras sequence representing the unlabeled set.
        """

    @abstractmethod
    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """
        Moves data items from the unlabeled set to one of the labeled sets (training, validation or test set).
        Args:
            ids: IDs of the items to be labeled.
            labels: Labels for the selected data items.

        Returns:
            None.
        """

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Returns:
            Pytorch dataloader or Keras sequence representing the training set.
        """

        if self._training_set:
            return DataLoader(
                self._training_set,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._get_collate_fn(),
            )
        return None

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Returns:
            Pytorch dataloader or Keras sequence representing the validation set.
        """

        if self._validation_set:
            return DataLoader(
                self._validation_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._get_collate_fn(),
            )
        return None

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Returns:
            Pytorch dataloader or Keras sequence representing the test set.
        """

        if self._test_set:
            return DataLoader(
                self._test_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self._get_collate_fn(),
            )
        return None

    def unlabeled_dataloader(self) -> Optional[DataLoader]:
        """
        Returns:
            Pytorch dataloader or Keras sequence representing the unlabeled set.
        """

        if self._unlabeled_set:
            return DataLoader(
                self._unlabeled_set,
                batch_size=1,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._get_collate_fn(),
            )
        return None

    def training_set_size(self) -> int:
        """
        Returns:
            Size of training set.
        """

        if self._training_set:
            return len(self._training_set)
        return 0

    def validation_set_size(self) -> int:
        """
        Returns:
            Size of validation set.
        """

        if self._validation_set:
            return len(self._validation_set)
        return 0

    def test_set_size(self) -> int:
        """
        Returns:
            Size of test set.
        """

        if self._test_set:
            return len(self._test_set)
        return 0

    def unlabeled_set_size(self) -> int:
        """
        Returns:
            Number of unlabeled items.
        """

        if self._unlabeled_set:
            return len(self._unlabeled_set)
        return 0
