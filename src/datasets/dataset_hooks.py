"""Module defining hooks that each dataset class should implement"""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Union


class DatasetHooks(ABC):
    """
    Class that defines hooks that should be implemented by each dataset class.
    """

    @abstractmethod
    def image_ids(self) -> Iterable[str]:
        """
        Returns:
            List of all image IDs included in the dataset.
        """

    @abstractmethod
    def slices_per_image(self, **kwargs) -> Union[int, List[int]]:
        """
        Args:
            kwargs: Dataset specific parameters.
        Returns:
            Union[int, List[int]]: Number of slices that each image of the dataset contains. If a single integer
                value is provided, it is assumed that all images of the dataset have the same number of slices.
        """

    @abstractmethod
    def multi_label(self) -> bool:
        """
        Returns:
            bool: Whether the dataset is a multi-label or a single-label dataset.
        """

    @abstractmethod
    def id_to_class_names(self) -> Dict[int, str]:
        """
        Returns:
            Dict[int, str]: A mapping of class indices to descriptive class names.
        """
