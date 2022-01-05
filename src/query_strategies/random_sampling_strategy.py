""" Module for random sampling strategy """
from typing import List, Union

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods
class RandomSamplingStrategy(QueryStrategy):
    """
    Class for selecting items via a random sampling strategy
    """

    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        number_of_items: int,
        **kwargs
    ) -> List[str]:
        """
        Selects random subset of the unlabeled data that should be labeled next.
        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            number_of_items: Number of items that should be selected for labeling.
            **kwargs: Additional, strategy-specific parameters.

        Returns:
            IDs of the data items to be labeled.
        """
        # randomly select ids to query

        selected_ids = []
        selected_items = 0

        for _, image_id in data_module.unlabeled_dataloader():
            if selected_items == number_of_items:
                break
            selected_ids.append(image_id[0])
            selected_items += 1

        return selected_ids
