""" Module for uncertainty sampling strategy """
import math
from typing import List, Union

import torch

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods
class UncertaintySamplingStrategy(QueryStrategy):
    """
    Class for selecting items via a random sampling strategy
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

        Returns:
            IDs of the data items to be labeled.
        """
        # randomly select ids to query

        if isinstance(models, List):
            raise ValueError(
                "Uncertainty sampling is only implemented for one model. You passed a list."
            )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        models.to(device)

        batch_size = 128
        data_items = len(data_module.unlabeled_dataloader())
        batches = math.ceil(data_items / batch_size)

        uncertainties = []
        data_index = -1
        data = iter(data_module.unlabeled_dataloader())
        with torch.no_grad():
            for _ in range(batches):
                batch = []
                for _ in range(batch_size):
                    data_index += 1
                    if data_index < data_items:
                        batch.append(next(data))

                x = torch.cat([x for [x, _] in batch], dim=1)

                batch_stack = (
                    torch.swapaxes(x, 0, 1)
                    if models.dim == 2
                    else torch.unsqueeze(x, 0)
                )
                batch_pred = models.predict(batch_stack.to(device))
                uncert = torch.sum(torch.abs(0.5 - batch_pred), (1, 2, 3)).cpu().numpy()

                for i, (_, case_id) in enumerate(batch):
                    uncertainties.append((uncert[i], case_id[0]))

        uncertainties.sort(key=lambda y: y[0], reverse=True)

        selected_ids = [id for (_, id) in uncertainties[0:items_to_label]]
        return selected_ids
