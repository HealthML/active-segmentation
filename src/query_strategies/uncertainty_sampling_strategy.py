""" Module for uncertainty sampling strategy """
from typing import List, Union, Tuple

import torch

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .functions import clean_duplicate_scans, select_uncertainty_calculation
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods
class UncertaintySamplingStrategy(QueryStrategy):
    """
    Class for selecting items to label by highest uncertainty
        Args:
        **kwargs: Optional keyword arguments:
            calculation_method (str): Specification of the method used to calculate the uncertainty
                values: `"distance"` |  `"entropy"`.
            exclude_background (bool): Whether to exclude the background dimension in calculating the
                uncertainty value.
            prefer_unique_scans (bool): Whether to prefer among the uncertain scan-slice combinations unique
                scans, if possible.
                E.g. with items_to_label set to 2:
                ['slice_1-32', 'slice_1-33', 'slice_2-50'] -> ['slice_1-32', 'slice_2-50']
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    # pylint: disable=too-many-locals
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs,
    ) -> Tuple[List[str], None]:
        """
        Selects subset of the unlabeled data with the highest uncertainty that should be labeled next.
        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            items_to_label (int): Number of items that should be selected for labeling.
            calculation_method (str, optional): Specification of the method used to calculate the uncertainty values.
                (default = 'distance')
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

        # For the max_uncertainty_value we have two cases in out datasets:
        # 1. multi-class, single-label (multiple foreground classes that are mutually exclusive) -> max_uncertainty is
        #   1 / num_classes. This is because the softmax layer sums up all to 1 across labels.
        # 2. multi-class, multi-label (multiple foreground classes that can overlap) -> max_uncertainty is 0.5
        max_uncertainty_value = (
            0.5 if data_module.multi_label() else 1 / data_module.num_classes()
        )

        for images, case_ids in data_module.unlabeled_dataloader():
            predictions = models.predict(images.to(device))
            uncertainty_calculation = select_uncertainty_calculation(
                calculation_method=self.kwargs.get("calculation_method", None)
            )
            uncertainty = uncertainty_calculation(
                predictions=predictions,
                max_uncertainty_value=max_uncertainty_value,
                **self.kwargs,
            )

            for idx, case_id in enumerate(case_ids):
                uncertainties.append((uncertainty[idx], case_id))

        uncertainties.sort(key=lambda y: y[0])
        if self.kwargs.get("prefer_unique_scans", False):
            uncertainties = clean_duplicate_scans(
                uncertainties=uncertainties, items_to_label=items_to_label
            )

        selected_ids = [id for (_, id) in uncertainties[0:items_to_label]]
        return selected_ids, None
