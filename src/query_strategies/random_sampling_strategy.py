""" Module for random sampling strategy """
from typing import List, Union, Optional

import numpy as np

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods
class RandomSamplingStrategy(QueryStrategy):
    """
    Class for selecting items via a random sampling strategy

    Args:
        random_state (int, optional): Random state for selecting items to label. Pass an int for reproducible outputs
            across multiple runs.
    """

    # pylint: disable=unused-argument
    def __init__(self, random_state: Optional[int] = None, **kwargs):
        self.random_state = random_state

    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs,
    ) -> List[str]:
        """
        Selects random subset of the unlabeled data that should be labeled next. We are using
        the shuffling of the dataset for randomisation.
        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            items_to_label (int): Number of items that should be selected for labeling.
            **kwargs: Additional, strategy-specific parameters.

        Returns:
            IDs of the data items to be labeled.
        """
        # randomly select ids to query

        unlabeled_image_ids = []

        for _, image_ids in data_module.unlabeled_dataloader():
            unlabeled_image_ids.extend(image_ids)

        items_to_label = min(items_to_label, data_module.unlabeled_set_size())

        rng = np.random.default_rng(self.random_state)

        return list(rng.choice(unlabeled_image_ids, size=items_to_label, replace=False))
