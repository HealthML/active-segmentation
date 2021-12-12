"""Module to load and batch Pascal VOC segmentation dataset"""

from typing import Any, Iterable, Tuple

from torch.utils.data import Dataset
from torchvision import datasets

from .dataset_hooks import DatasetHooks


class PascalVOCDataset(Dataset, DatasetHooks):
    """
    Wrapper class for the VOCSegmentation dataset class from the torchvision package.

    Args:
        root (str): Root directory of the VOC Dataset.
        kwargs: Additional keyword arguments as defined in the VOCSegmentation class from the torchvision package.
    """

    def __init__(self, root: str, **kwargs):

        self.pascal_voc_datset = datasets.VOCSegmentation(root, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.pascal_voc_datset.__getitem__(index)

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset.
        """

        return self.pascal_voc_datset.__len__()

    def image_ids(self) -> Iterable[str]:
        """
        Returns:
            List of all image IDs included in the dataset.
        """

        return range(self.__len__)

    def slices_per_image(self, **kwargs) -> Tuple[int]:
        """
        Returns:
            int: Number of slices that each image of the dataset contains.
        """

        return 1
