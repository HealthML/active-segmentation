""" Module for uncertainty sampling strategy """
from typing import List, Union

import torch

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods
class UncertaintySamplingStrategy(QueryStrategy):
    """
    Class for selecting items to label by highest uncertainty
    """

    # pylint: disable=too-many-locals
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
                Keyword Args:
                    exclude_background (bool): Whether to exclude the background dimension in calculating the
                        uncertainty value.

        Returns:
            IDs of the data items to be labeled.
        """

        if isinstance(models, List):
            raise ValueError(
                "Uncertainty sampling is only implemented for one model. You passed a list."
            )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        models.to(device)

        uncertainties = []

        for images, case_ids in data_module.unlabeled_dataloader():
            predictions = models.predict(images.to(device))
            if kwargs.get("exclude_background", False):
                predictions = predictions[:, 1:, :, :]
            max_uncertainty_value = 1 / data_module.num_classes()
            uncertainty = (
                torch.sum(torch.abs(max_uncertainty_value - predictions), (1, 2, 3))
                .cpu()
                .numpy()
            )

            for idx, case_id in enumerate(case_ids):
                uncertainties.append((uncertainty[idx], case_id))

        uncertainties.sort(key=lambda y: y[0])

        selected_ids = [id for (_, id) in uncertainties[0:items_to_label]]
        return selected_ids
