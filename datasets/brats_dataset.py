import math
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple


class BraTSDataset(Dataset):
    IMAGE_DIMENSIONS = (155, 240, 240)

    @staticmethod
    def normalize(img):
        tmp = img / np.max(img)
        # ignore zero values for mean calculation because background dominates
        tmp = tmp - np.mean(tmp[tmp > 0])
        # make normalize original zero values to -1
        return tmp / (-np.min(tmp))

    @staticmethod
    def __read_image_as_array(
        filepath: str, norm: bool, clip: bool = False
    ) -> np.ndarray:
        """
        Reads image or annotation as numpy array.

        :param filepath: Path of the image file.
        :param norm: Whether the image should be normalized.
        :param clip: Whether the image should be clipped.
        :return:
        """

        img = nib.load(filepath).get_fdata()

        if clip:
            img = np.clip(img, 0, 1)

        if norm:
            img = BraTSDataset.normalize(img)
        return np.moveaxis(img, 2, 0)

    def __init__(
        self,
        image_paths: List[str],
        annotation_paths: List[str],
        clip_mask: bool = True,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        target_transform: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.num_images = len(image_paths)
        self.num_annotations = len(annotation_paths)
        assert self.num_images == self.num_annotations
        self.clip_mask = clip_mask
        self._current_image = None
        self._current_image_index = None
        self._current_mask = None

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_index = math.ceil(index / BraTSDataset.IMAGE_DIMENSIONS[0])
        slice_index = index - image_index * BraTSDataset.IMAGE_DIMENSIONS[0]
        if image_index != self._current_image_index:
            self._current_image_index = image_index
            self._current_image = self.__read_image_as_array(
                filepath=self.image_paths[self._current_image_index], norm=True
            )
            self._current_mask = self.__read_image_as_array(
                filepath=self.annotation_paths[self._current_image_index],
                norm=False,
                clip=self.clip_mask,
            )

        x = torch.from_numpy(self._current_image[slice_index, :, :])
        y = torch.from_numpy(self._current_mask[slice_index, :, :])

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self) -> int:
        return self.num_images * (BraTSDataset.IMAGE_DIMENSIONS[0] - 1)

    def add_image(self, image_path: str, annotation_path: str) -> None:
        """
        Adds an image to this dataset.

        :param image_path: Path of the image to be added.
        :param annotation_path: Path of the annotation of the image to be added.
        """

        if (image_path not in self.image_paths) and (
            annotation_path not in self.annotation_paths
        ):
            self.image_paths.append(image_path)
            self.annotation_paths.append(annotation_path)
            self.num_images += 1
        else:
            raise ValueError("Image already belongs to this dataset.")

    def remove_image(self, image_path: str, annotation_path: str) -> None:
        """
        Removes an image from this dataset.

        :param image_path: Path of the image to be removed.
        :param annotation_path: Path of the annotation of the image to be removed.
        """

        if image_path in self.image_paths and annotation_path in self.annotation_paths:
            self.image_paths.remove(image_path)
            self.annotation_paths.remove(annotation_path)
            self.num_images -= 1
        else:
            raise ValueError("Image does not belong to this dataset.")
