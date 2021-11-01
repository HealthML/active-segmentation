from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from typing import Any, List, Union
from tensorflow.keras.utils import Sequence


class ActiveLearningDataModule(LightningDataModule):
    _training_set = None
    _validation_set = None
    _test_set = None
    _unlabeled_set = None

    def __init__(self, data_dir: str, batch_size, **kwargs):
        """
        :param data_dir: Path of the directory that contains the data.
        :param batch_size: Batch size.
        :param kwargs: Further, dataset specific parameters.
        """

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates the datasets managed by this data module.

        :param stage: Current training stage.
        """

        self._training_set = self._create_training_set()
        self._validation_set = self._create_validation_set()
        self._test_set = self._create_test_set()
        self._unlabeled_set = self._create_unlabeled_set()

    def _create_training_set(self) -> Union[Dataset, Sequence, None]:
        """
        :return: Pytorch data_module or Keras sequence representing the training set.
        """

        # this method should be overwritten in derived classes to create the training set
        return None

    def _create_validation_set(self) -> Union[Dataset, Sequence, None]:
        """
        :return: Pytorch data_module or Keras sequence representing the validation set.
        """

        # this method should be overwritten in derived classes to create the validation set
        return None

    def _create_test_set(self) -> Union[Dataset, Sequence, None]:
        """
        :return: Pytorch data_module or Keras sequence representing the test set.
        """

        # this method should be overwritten in derived classes to create the test set
        return None

    def _create_unlabeled_set(self) -> Union[Dataset, Sequence, None]:
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

    def train_dataloader(self) -> Union[DataLoader, Sequence, None]:
        """
        :return: Pytorch dataloader or Keras sequence representing the training set.
        """

        if self._training_set:
            if isinstance(self._training_set, Sequence):
                return self._training_set
            else:
                return DataLoader(self._training_set, batch_size=self.batch_size)
        return None

    def val_dataloader(self) -> Union[DataLoader, Sequence, None]:
        """
        :return: Pytorch dataloader or Keras sequence representing the validation set.
        """

        if self._validation_set:
            if isinstance(self._validation_set, Sequence):
                return self._validation_set
            else:
                return DataLoader(self._validation_set, batch_size=self.batch_size)
        return None

    def test_dataloader(self) -> Union[DataLoader, Sequence, None]:
        """
        :return: Pytorch dataloader or Keras sequence representing the test set.
        """

        if self._test_set:
            if isinstance(self._test_set, Sequence):
                return self._test_set
            else:
                return DataLoader(self._test_set, batch_size=self.batch_size)
        return None

    def unlabeled_dataloader(self) -> Union[DataLoader, Sequence, None]:
        """
        :return: Pytorch dataloader or Keras sequence representing the unlabeled set.
        """

        if self._unlabeled_set:
            if isinstance(self._unlabeled_set, Sequence):
                return self._unlabeled_set
            else:
                for image in DataLoader(self._unlabeled_set, batch_size=self.batch_size):
                    print("image dl", image)
                return DataLoader(self._unlabeled_set, batch_size=self.batch_size)
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
