"""Module defining hooks that each dataset class should implement"""

from abc import ABC, abstractmethod
from typing import Iterable

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
    def slices_per_image(self, **kwargs):
        """
        Args:
            kwargs: Dataset specific parameters.
        Returns:
            int: Number of slices that each image of the dataset contains.
        """
