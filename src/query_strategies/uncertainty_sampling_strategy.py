""" Module for uncertainty sampling strategy """
import torch
import numpy as np
from typing import List, Union

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods
class UncertaintySamplingStrategy(QueryStrategy):
    """
    Class for selecting items via a random sampling strategy
    """

    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs
    ) -> List[str]:
        """
        Selects subset of the unlabeled data with the highest uncertainty that should be labeled next.
        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            items_to_label (int): Number of items that should be selected for labeling.
            **kwargs: Additional, strategy-specific parameters.

        Returns:
            IDs of the data items to be labeled.
        """
        # randomly select ids to query

        if type(models) is List[PytorchModel]:
            raise ValueError(
                "Uncertainty sampling is only implemented for one model. You passed a list."
            )

        selected_ids = []
        selected_items = 0

        # Sum of the distance of all predictions compared to 0.5
        uncertainties = [
            (
                torch.sum(
                    torch.abs(
                        0.5
                        - models.predict(
                            torch.swapaxes(x, 0, 1)
                            if models.dim == 2
                            else torch.unsqueeze(x, 0)
                        ).flatten()
                    )
                ),
                ids[0],
            )
            for x, ids in data_module.unlabeled_dataloader()
        ]

        print("uncertainties: ", uncertainties)
        print("first uncertainty: ", uncertainties[0])
        print("first uncertainty shape: ", uncertainties[0][0].shape)

        uncertainties.sort(key=lambda y: y[0], reverse=True)

        print("highest uncertainty: ", uncertainties[0:items_to_label])

        selected_ids = [id for (_, id) in uncertainties[0:items_to_label]]
        print("selected_ids: ", selected_ids)

        return selected_ids