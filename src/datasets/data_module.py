from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from typing import Any, List, Union
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings(
    "ignore",
    ".*DataModule.setup has already been called, so it will not be called again.*",
)


class ActiveLearningDataModule(LightningDataModule):
    _training_set = None
    _validation_set = None
    _test_set = None
    _unlabeled_set = None

    def __init__(self, data_dir: str, batch_size, shuffle=True, **kwargs):
        """
        :param data_dir: Path of the directory that contains the data.
        :param batch_size: Batch size.
        :param kwargs: Further, dataset specific parameters.
        """

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates the datasets managed by this data module.

        :param stage: Current training stage.
        """

        self._training_set = self._create_training_set()
        self._validation_set = self._create_validation_set()
        self._test_set = self._create_test_set()
        self._unlabeled_set = self._create_unlabeled_set()

    def _create_training_set(self) -> Optional[Dataset]:
        """
        :return: Pytorch data_module or Keras sequence representing the training set.
        """

        # this method should be overwritten in derived classes to create the training set
        return None

    def _create_validation_set(self) -> Optional[Dataset]:
        """
        :return: Pytorch data_module or Keras sequence representing the validation set.
        """

        # this method should be overwritten in derived classes to create the validation set
        return None

    def _create_test_set(self) -> Optional[Dataset]:
        """
        :return: Pytorch data_module or Keras sequence representing the test set.
        """

        # this method should be overwritten in derived classes to create the test set
        return None

    def _create_unlabeled_set(self) -> Optional[Dataset]:
        """
        :return: Pytorch data_module or Keras sequence representing the unlabeled set.
        """

        # this method should be overwritten in derived classes to create the unlabeled set
        return None

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        """
        Moves data items from the unlabeled set to one of the labeled sets (training, validation or test set).

        :param ids: IDs of the items to be labeled.
        :param labels: Labels for the selected data items.
        """
        # this method should be overwritten in derived classes to implement the labeling logic
        return None

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        :return: Pytorch dataloader or Keras sequence representing the training set.
        """

        if self._training_set:
            return DataLoader(
                self._training_set,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=8,
            )
        return None

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        :return: Pytorch dataloader or Keras sequence representing the validation set.
        """

        if self._validation_set:
            return DataLoader(
                self._validation_set, batch_size=self.batch_size, num_workers=8
            )
        return None

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        :return: Pytorch dataloader or Keras sequence representing the test set.
        """

        if self._test_set:
            return DataLoader(self._test_set, batch_size=self.batch_size, num_workers=8)
        return None

    def unlabeled_dataloader(self) -> Optional[DataLoader]:
        """
        :return: Pytorch dataloader or Keras sequence representing the unlabeled set.
        """

        if self._unlabeled_set:
            return DataLoader(
                self._unlabeled_set, batch_size=self.batch_size, num_workers=8
            )
        return None

    def training_set_size(self) -> int:
        """
        :return: Size of training set.
        """

        if self._training_set:
            return len(self._training_set)
        return 0

    def validation_set_size(self) -> int:
        """
        :return: Size of validation set.
        """

        if self._validation_set:
            return len(self._validation_set)
        return 0

    def test_set_size(self) -> int:
        """
        :return: Size of test set.
        """

        if self._test_set:
            return len(self._test_set)
        return 0

    def unlabeled_set_size(self) -> int:
        """
        :return: Number of unlabeled items.
        """

        if self._unlabeled_set:
            return len(self._unlabeled_set)
        return 0
