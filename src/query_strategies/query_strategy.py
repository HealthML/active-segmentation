# pylint: disable=all
from torch.utils.data import DataLoader
from typing import List, Union
from abc import ABC, abstractmethod

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel


class QueryStrategy(ABC):
    @abstractmethod
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs
    ) -> List[str]:
        """
        Selects subset of the unlabeled data that should be labeled next.
        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            dataloader: Pytorch dataloader representing the unlabeled dataset.
            items_to_label: Number of items that should be selected for labeling.
            **kwargs: Additional, strategy-specific parameters.

        Returns:
            IDs of the data items to be labeled.
        """

        raise NotImplementedError()
