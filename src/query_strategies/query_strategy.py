# pylint: disable=all
from torch.utils.data import DataLoader
from typing import List, Union

from models.pytorch_model import PytorchModel


class QueryStrategy:
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        dataloader: DataLoader,
        number_of_items: int,
        **kwargs
    ) -> List[str]:
        """
        Selects subset of the unlabeled data that should be labeled next.

        :param models: Current models that should be improved by selecting additional data for labeling.
        :param dataloader: Pytorch dataloader representing the unlabeled dataset.
        :param number_of_items: Number of items that should be selected for labeling.
        :param kwargs: Additional, strategy-specific parameters.
        :return: IDs of the data items to be labeled.
        """

        # default strategy: select first n data items
        # this method should be overwritten in derived classes to implement other strategies

        selected_ids = []
        selected_items = 0

        for image, image_id in dataloader:
            if selected_items == number_of_items:
                break
            selected_ids.append(image_id)
            selected_items += 1

        return selected_ids
