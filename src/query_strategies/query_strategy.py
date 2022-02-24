"""Module containing abstract superclass for query strategies."""
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel


# pylint: disable=too-few-public-methods
class QueryStrategy(ABC):
    """Abstract superclass for query strategies."""

    @abstractmethod
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs
    ) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """
        Selects subset of the unlabeled data that should be labeled next.
        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            dataloader: Pytorch dataloader representing the unlabeled dataset.
            items_to_label: Number of items that should be selected for labeling.
            **kwargs: Additional, strategy-specific parameters.

        Returns:
            Tuple[List[str], Optional[Dict[str, np.array]]]: List of IDs of the data items to be labeled
            and an optional dictonary of pseudo labels with the corresponding IDs as keys.
        """

        raise NotImplementedError()
