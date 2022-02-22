""" Module for uncertainty sampling strategy """
from typing import List, Union, Tuple

import torch

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods
class UncertaintySamplingStrategy(QueryStrategy):
    """
    Class for selecting items to label by highest uncertainty
    """

    @staticmethod
    def compute_uncertainties(
        model: PytorchModel, data_module: ActiveLearningDataModule
    ) -> Tuple[List[float], List[str]]:
        """

        Args:
            model (PytorchModel): Current model that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.

        Returns:
            Tuple[List[float], List[str]]: Model uncertainties and case IDs for all items in the unlabeled set.
        """

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        uncertainties = []
        case_ids = []

        for images, current_case_ids in data_module.unlabeled_dataloader():
            predictions = model.predict(images.to(device))
            uncertainty = (
                torch.sum(torch.abs(0.5 - predictions), (1, 2, 3)).cpu().numpy()
            )

            uncertainties.extend(list(uncertainty))
            case_ids.extend(current_case_ids)

        return uncertainties, case_ids

    # pylint: disable=too-many-locals
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs
    ) -> Tuple[List[str], List[float]]:
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

        if isinstance(models, List):
            raise ValueError(
                "Uncertainty sampling is only implemented for one model. You passed a list."
            )

        uncertainties, case_ids = self.compute_uncertainties(models, data_module)
        uncertainties = list(zip(uncertainties, case_ids))

        uncertainties.sort(key=lambda y: y[0])

        selected_ids = [id for (_, id) in uncertainties[0:items_to_label]]
        selected_uncertainties = [
            uncertainty for (uncertainty, _) in uncertainties[0:items_to_label]
        ]

        return selected_ids, selected_uncertainties
