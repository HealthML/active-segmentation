""" Module for uncertainty sampling strategy """
from typing import List, Tuple, Union, Callable

import numpy as np
import torch

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods
class UncertaintySamplingStrategy(QueryStrategy):
    """
    Class for selecting items to label by highest uncertainty
    """

    # pylint: disable=too-many-locals, arguments-differ
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        calculation_method: str = "distance",
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

        for images, case_ids in data_module.unlabeled_dataloader():
            predictions = models.predict(images.to(device))
            uncertainty_calculation = self.__select_uncertainty_calculation(
                calculation_method=calculation_method
            )
            uncertainty = uncertainty_calculation(predictions=predictions, **kwargs)

            for idx, case_id in enumerate(case_ids):
                uncertainties.append((uncertainty[idx], case_id))

        uncertainties.sort(key=lambda y: y[0])

        selected_ids = [id for (_, id) in uncertainties[0:items_to_label]]
        return selected_ids, None

    @staticmethod
    def __select_uncertainty_calculation(calculation_method: str) -> Callable:
        """
        Selects the calculation function based on the provided name.
        Args:
            calculation_method (str): Name of the calculation method.

        Returns:
            A callable function to calculate uncertainty based on predictions.
        """
        if calculation_method == "distance":
            return UncertaintySamplingStrategy.__distance_to_max_uncertainty
        if calculation_method == "entropy":
            return UncertaintySamplingStrategy.__entropy
        print(
            "No valid calculation method provided, choosing default method: distance to max uncertainty."
        )
        return UncertaintySamplingStrategy.__distance_to_max_uncertainty

    @staticmethod
    def __distance_to_max_uncertainty(
        predictions: torch.Tensor, max_uncertainty_value: float = 0.5, **kwargs
    ) -> np.ndarray:
        r"""
        Calculates the uncertainties based on the distance to a maximum uncertainty value:
            .. math::
                \sum | max\_uncertainty\_value - predictions |
        Args:
            predictions (torch.Tensor): The predictions of the model.
            max_uncertainty_value (float, optional): The maximum value of uncertainty in the predictions.
                (default = 0.5)
            **kwargs: Keyword arguments specific for this calculation.

        Returns:
            Uncertainty value for each image in the batch of predictions.
        """
        if kwargs.get("exclude_background", False):
            predictions = predictions[:, 1:, :, :]
        uncertainty = (
            torch.sum(torch.abs(max_uncertainty_value - predictions), (1, 2, 3))
            .cpu()
            .numpy()
        )
        return uncertainty

    @staticmethod
    def __entropy(
        predictions: torch.Tensor, max_uncertainty_value: float = 0.5, **kwargs
    ) -> np.ndarray:
        r"""
        Calculates the uncertainties based on the entropy of the distance to a maximum uncertainty value:
            .. math::
                - \sum | max\_uncertainty\_value - predictions | \cdot | \log({max\_uncertainty\_value - predictions}) |
        Args:
            predictions (torch.Tensor): The predictions of the model.
            max_uncertainty_value (float, optional): The maximum value of uncertainty in the predictions.
             (default = 0.5)
            **kwargs: Keyword arguments specific for this calculation.

        Returns:
            Uncertainty value for each image in the batch of predictions.
        """
        # pylint: disable=unused-argument
        # Smoothing to avoid taking log of zero
        predictions[predictions == max_uncertainty_value] = (
            max_uncertainty_value + 1e-10
        )
        uncertainty = (
            -torch.sum(
                torch.multiply(
                    torch.abs(max_uncertainty_value - predictions),
                    torch.log(torch.abs(max_uncertainty_value - predictions)),
                ),
                (1, 2, 3),
            )
            .cpu()
            .numpy()
        )
        return uncertainty
