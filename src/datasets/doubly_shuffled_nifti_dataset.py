""" Module to load and batch nifti datasets """
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from multiprocessing import Manager
from sklearn.model_selection import train_test_split

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import IterableDataset

from datasets.dataset_hooks import DatasetHooks


# pylint: disable=too-many-instance-attributes,abstract-method
class DoublyShuffledNIfTIDataset(IterableDataset, DatasetHooks):
    """
    This dataset can be used with NIfTI images. It is iterable and can return both 2D and 3D images.

    Args:
        image_paths (List[str]): List with the paths to the images. Has to contain paths of all images which can ever
            become part of the dataset.
        annotation_paths (List[str]): List with the paths to the annotations. Has to contain paths of all images which
            can ever become part of the dataset.
        cache_size (int, optional): Number of images to keep in memory to speed-up data loading in subsequent epochs.
            Defaults to zero.
        combine_foreground_classes (bool, optional): Flag if the non-zero values of the annotations should be merged.
            Defaults to False.
        mask_filter_values (Tuple[int], optional): Values from the annotations which should be used. Defaults to using
            all values.
        shuffle (bool, optional): Whether the data should be shuffled.
        transform (Callable[[Any], Tensor], optional): Function to transform the images.
        target_transform (Callable[[Any], Tensor], optional): Function to transform the annotations.
        dim (int, optional): 2 or 3 to define if the dataset should return 2d slices of whole 3d images.
            Defaults to 2.
        slice_indices (List[np.array], optional): Array of indices per image which should be part of the dataset.
            Uses all slices if None. Defaults to None.
        random_state (int, optional): Controls the data shuffling. Pass an int for reproducible output across multiple
            runs.
    """

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """
        Normalizes an image by
            #. Dividing by the maximum value
            #. Subtracting the mean, zeros will be ignored while calculating the mean
            #. Dividing by the negative minimum value

        Args:
            img: The input image that should be normalized.

        Returns:
            Normalized image with background values normalized to -1
        """

        tmp = img

        tmp /= np.max(tmp)

        # ignore zero values for mean calculation because background dominates
        tmp -= np.mean(tmp[tmp > 0])

        tmp /= -np.min(tmp)

        return tmp

    @staticmethod
    def __read_image(filepath: str) -> Any:
        """
        Reads image or annotation.

        Args:
            filepath (str): Path of the image file.

        Returns:
            The image. See https://nipy.org/nibabel/reference/nibabel.spatialimages.html#module-nibabel.spatialimages
        """
        return nib.load(filepath)

    @staticmethod
    def __read_slice_count(filepath: str, dim: int = 2) -> int:
        """
        Reads image or annotation.

        Args:
            filepath (str): Path of the image file.
            dim (int, optional): The dimensionality of the dataset. Defaults to 2.

        Returns:
            The slice count of the image at the filepath or 1 if dim is not 2.
        """
        return (
            DoublyShuffledNIfTIDataset.__read_image(filepath).shape[2]
            if dim == 2
            else 1
        )

    @staticmethod
    def generate_active_learning_split(
        filepaths: List[str],
        dim: int,
        initial_training_set_size: int,
        random_state: Optional[int] = None,
    ) -> Tuple[List[np.array]]:
        """
        Generates a split between initial training set and initially unlabeled set for active learning.

        Args:
            filepaths (List[str]): The file paths to the Nifti files.
            dim (int): The dimensionality of the dataset. (2 or 3.)
            initial_training_set_size (int): The number of samples in the initial training set.
            random_state (int, optional): The random state used to generate the split. Pass an int for reproducibility
                across runs.

        Returns:
            A tuple of two lists of np.arrays. The lists contain one array per filepath which contains the
            slice indices of the slices which should be part of the training and unlabeled sets respectively.
            The lists can be passed as `slice_indices` for initialization of a DoublyShuffledNIfTIDataset.
        """

        if dim == 3:
            image_indices = range(len(filepaths))
            (initial_training_samples, initial_unlabeled_samples,) = train_test_split(
                image_indices,
                train_size=initial_training_set_size,
                random_state=random_state,
            )
            return (
                [
                    np.array([0] if image_index in initial_training_samples else [])
                    for image_index in range(len(filepaths))
                ],
                [
                    np.array([0] if image_index in initial_unlabeled_samples else [])
                    for image_index in range(len(filepaths))
                ],
            )

        all_samples = [
            [image_index, slice_index]
            for image_index, filepath in enumerate(filepaths)
            for slice_index in range(
                DoublyShuffledNIfTIDataset.__read_slice_count(filepath, dim=dim)
            )
        ]

        (initial_training_samples, initial_unlabeled_samples) = train_test_split(
            all_samples,
            train_size=initial_training_set_size,
            random_state=random_state,
        )

        return (
            [
                np.array(
                    [
                        slice_index
                        for image_index, slice_index in initial_training_samples
                        if image_index == image_id
                    ]
                )
                for image_id in range(len(filepaths))
            ],
            [
                np.array(
                    [
                        slice_index
                        for image_index, slice_index in initial_unlabeled_samples
                        if image_index == image_id
                    ]
                )
                for image_id in range(len(filepaths))
            ],
        )

    @staticmethod
    def __align_axes(img: np.ndarray) -> np.ndarray:
        """
        Aligns the axes to (slice, x, y) or (channel, slice, x, y), depending on if there is a channel dimension.

        Args:
            img (np.ndarray): The image

        Returns:
            The images with realigned axes.
        """
        img = np.moveaxis(img, 2, 0)  # slice dimension to front

        if len(img.shape) == 4:
            img = np.moveaxis(img, 3, 0)  # channel dimension to front

        return img

    @staticmethod
    def __read_image_as_array(
        filepath: str,
        norm: bool,
        join_non_zero: bool = False,
        filter_values: Optional[Tuple[int]] = None,
    ) -> np.ndarray:
        """
        Reads image or annotation as numpy array.

        Args:
            filepath: Path of the image file.
            norm: Whether the image should be normalized.
            join_non_zero: Whether the non-zero values of the image should be joined. Will set all non-zero values to 1.
            filter_values: Values to be filtered from the images. All other values will be set to zero.
                Can be used together with join_non_zero. Filtering will be applied before joining.

        Returns:
            The array representation of an image.
        """
        img = DoublyShuffledNIfTIDataset.__read_image(filepath).get_fdata()

        if filter_values is not None:
            map_to_filtered_value = np.vectorize(
                lambda value: value if value in filter_values else 0
            )

            img = map_to_filtered_value(img)

        if join_non_zero:
            img = np.clip(img, 0, 1)

        if norm:
            img = DoublyShuffledNIfTIDataset.normalize(img)
        return DoublyShuffledNIfTIDataset.__align_axes(img)

    @staticmethod
    def __ensure_channel_dim(img: torch.Tensor, dim: int) -> torch.Tensor:
        return img if len(img.shape) == dim + 1 else torch.unsqueeze(img, 0)

    def __arange_image_slice_indices(
        self,
        filepaths: List[str],
        dim: int = 2,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        slice_indices: Optional[List[np.array]] = None,
    ) -> Dict[int, Dict[int, Optional[np.array]]]:
        """
        Reads the slice indices for the images at the provided slice paths and pairs them with their image index.

        Implements efficient shuffling for 2D image datasets like the DoublyShuffledNIfTIDataset whose elements
        represent the slices of multiple 3D images. To allow for efficient image pre-fetching, first the order of all 3D
        images is shuffled and then the order of slices within each 3D image is shuffled. This way the 3D images can
        still be loaded as a whole.

        Args:
            filepaths (List[str]): The paths of the images.
            dim (int, optional): The dimensionality of the dataset. Defaults to 2.
            shuffle (boolean, optional): Flag indicating whether to shuffle the slices. Defaults to False.
            random_state (int, optional): Random seed for shuffling.
            slice_indices (List[np.array], optional): Array of indices per image which should be part of the dataset.
                Uses all slices if None. Defaults to None.

        Returns:
            A dictionary of per-image dictionaries which contain slice indices as keys. If a slice index is not part of
            the per-image dictionary, it is not part of the dataset. If its value is None, it does not have a pseudo
            label. If its value is a np.array, this is the pseudo label for that slice.
        """
        if slice_indices is None:
            slice_indices = [
                np.arange(
                    DoublyShuffledNIfTIDataset.__read_slice_count(filepath, dim=dim)
                )
                for filepath in filepaths
            ]

        if shuffle:
            rng = np.random.default_rng(random_state)

            # Shuffle the slices within the images
            for slices in slice_indices:
                rng.shuffle(slices)

            # Shuffle the images
            enumerated_slice_indices = list(enumerate(slice_indices))
            rng.shuffle(enumerated_slice_indices)
        else:
            enumerated_slice_indices = enumerate(slice_indices)

        # Pair up the slices indices with their image index and concatenate for all images
        # (e.g. [5,1,9,0,...] for image index 3 becomes [(3,5),(3,1),(3,9),(3,0),...])
        image_slice_indices = self.manager.dict()

        for image_index, slices in enumerated_slice_indices:
            if len(slices) > 0:
                image_slice_indices[image_index] = self.manager.dict()
                for slice_index in slices:
                    image_slice_indices[image_index][slice_index] = None

        # Concatenate the [image_index, slice_index] pairs for all images
        return image_slice_indices

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        image_paths: List[str],
        annotation_paths: List[str],
        cache_size: int = 0,
        combine_foreground_classes: bool = False,
        mask_filter_values: Optional[Tuple[int]] = None,
        is_unlabeled: bool = False,
        shuffle: bool = False,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        target_transform: Optional[Callable[[Any], torch.Tensor]] = None,
        dim: int = 2,
        slice_indices: Optional[List[np.array]] = None,
        case_id_prefix: str = "train",
        random_state: Optional[int] = None,
    ):
        self.manager = Manager()

        self.image_paths = self.manager.list(image_paths)
        self.annotation_paths = self.manager.list(annotation_paths)
        self.combine_foreground_classes = combine_foreground_classes
        self.mask_filter_values = mask_filter_values

        assert len(image_paths) == len(annotation_paths)

        self.is_unlabeled = is_unlabeled

        self._current_image = None
        self._current_mask = None
        self._currently_loaded_image_index = None
        self.cache_size = cache_size

        # since the PyTorch dataloader uses multiple processes for data loading (if num_workers > 0),
        # a shared dict is used to share the cache between all processes have to use
        # see https://github.com/ptrblck/pytorch_misc/blob/master/shared_dict.py and
        # https://discuss.pytorch.org/t/reuse-of-dataloader-worker-process-and-caching-in-dataloader/30620/14
        # for more information
        self.image_cache = self.manager.dict()
        self.mask_cache = self.manager.dict()

        self.shuffle = shuffle

        self.transform = transform
        self.target_transform = target_transform

        self.dim = dim

        self.case_id_prefix = case_id_prefix

        self.image_slice_indices = self.__arange_image_slice_indices(
            filepaths=self.image_paths,
            dim=self.dim,
            shuffle=self.shuffle,
            random_state=random_state,
            slice_indices=slice_indices,
        )

        self.num_workers = 1
        self.current_image_key_index = 0
        self.current_slice_key_index = 0

    def __get_case_id(self, image_index: int):
        return f"{self.case_id_prefix}_{image_index}"

    def __get_image_index(self, case_id: str):
        return int(case_id.replace(f"{self.case_id_prefix}_", ""))

    def __iter__(self):
        """
        Returns:
            Iterator: Iterator that yields the whole dataset if a single process is used for data loading
                or a subset of the dataset if the dataloading is split across multiple worker processes.
        """

        worker_info = torch.utils.data.get_worker_info()

        # check whether data loading is split across multiple workers
        if worker_info is not None:
            self.num_workers = worker_info.num_workers
            self.current_image_key_index = worker_info.id
            self.current_slice_key_index = 0
        else:
            self.num_workers = 1
            self.current_image_key_index = 0
            self.current_slice_key_index = 0
        return self

    def read_mask_for_image(self, image_index: int) -> np.array:
        """
        Reads the mask for the image from file. Uses correct mask specific parameters.

        Args:
            image_index (int): Index of the image to load.
        """
        return self.__read_image_as_array(
            self.annotation_paths[image_index],
            norm=False,
            join_non_zero=self.combine_foreground_classes,
            filter_values=self.mask_filter_values,
        )

    def __load_image_and_mask(self, image_index: int) -> None:
        """
        Loads image with the given index either from cache or from disk.

        Args:
            image_index (int): Index of the image to load.
        """

        self._currently_loaded_image_index = image_index

        # check if image and mask are in cache
        if image_index in self.image_cache and image_index in self.mask_cache:
            self._current_image = self.image_cache[image_index]
            self._current_mask = self.mask_cache[image_index]
        # read image and mask from disk otherwise
        else:
            self._current_image = self.__read_image_as_array(
                self.image_paths[image_index], norm=True
            )
            self._current_mask = self.read_mask_for_image(image_index)

        # cache image and mask if there is still space in cache
        if len(self.image_cache.keys()) < self.cache_size:
            self.image_cache[image_index] = self._current_image
            self.mask_cache[image_index] = self._current_mask

    # pylint: disable=too-many-branches
    def __next__(
        self,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, str], Tuple[torch.Tensor, torch.Tensor]
    ]:
        if self.current_image_key_index >= len(self.image_slice_indices):
            raise StopIteration

        image_index = list(self.image_slice_indices.keys())[
            self.current_image_key_index
        ]

        if image_index != self._currently_loaded_image_index:
            self.__load_image_and_mask(image_index)

        case_id = self.__get_case_id(image_index)

        if self.dim == 2:
            slice_index = list(self.image_slice_indices[image_index].keys())[
                self.current_slice_key_index
            ]
            case_id = f"{case_id}-{slice_index}"

            if len(self._current_image.shape) == 4:
                x = torch.from_numpy(self._current_image[:, slice_index, :, :])
            else:
                x = torch.from_numpy(self._current_image[slice_index, :, :])
            pseudo_label = self.image_slice_indices[image_index][slice_index]
            is_pseudo_label = pseudo_label is not None
            y = (
                torch.from_numpy(pseudo_label).int()
                if is_pseudo_label
                else torch.from_numpy(self._current_mask[slice_index, :, :]).int()
            )

            self.current_slice_key_index += 1

            if self.current_slice_key_index >= len(
                self.image_slice_indices[image_index]
            ):
                self.current_image_key_index += self.num_workers
                self.current_slice_key_index = 0
        else:
            x = torch.from_numpy(self._current_image)
            y = torch.from_numpy(self._current_mask).int()

            for slice_id in range(len(y)):
                if slice_id not in self.image_slice_indices[image_index]:
                    y[slice_id, :, :] = -1
                elif self.image_slice_indices[image_index][slice_id] is not None:
                    y[slice_id, :, :] = torch.from_numpy(
                        self.image_slice_indices[image_index][slice_id]
                    ).int()

            self.current_image_key_index += self.num_workers

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        x = DoublyShuffledNIfTIDataset.__ensure_channel_dim(x, self.dim)

        if self.is_unlabeled:
            return (x, case_id)

        return (x, y, is_pseudo_label, case_id)

    def add_image(
        self,
        image_id: str,
        slice_index: int = 0,
        pseudo_label: Optional[np.array] = None,
    ) -> None:
        """
        Adds an image to this dataset.

        Args:
            image_id (str): The id of the image.
            slice_index (int): Index of the slice to be added.
            pseudo_label (np.array, optional): An optional pseudo label for the slice. If no pseudo label is provided,
                the actual label from the corresponding file is used.
        """

        image_index = self.__get_image_index(image_id)

        if (
            image_index in self.image_slice_indices
            and slice_index in self.image_slice_indices[image_index]
            and self.image_slice_indices[image_index][slice_index] is None
        ):
            if pseudo_label is not None:
                # If a pseudo label is added even though the real label already exists it should be ignored
                return
            raise ValueError("Slice of image already belongs to this dataset.")

        if image_index not in self.image_slice_indices:
            self.image_slice_indices[image_index] = self.manager.dict()

        if (
            slice_index not in self.image_slice_indices[image_index]
            or self.image_slice_indices[image_index][slice_index] is not None
        ):
            self.image_slice_indices[image_index][slice_index] = pseudo_label

    def remove_image(self, image_id: str, slice_index: int = 0) -> None:
        """
        Removes an image from this dataset.

        Args:
            image_id (str): The id of the image.
            slice_index (int): Index of the slice to be removed.
        """

        image_index = self.__get_image_index(image_id)
        if (
            image_index in self.image_slice_indices
            and slice_index in self.image_slice_indices[image_index]
        ):
            del self.image_slice_indices[image_index][slice_index]
            if len(self.image_slice_indices[image_index]) == 0:
                self.image_slice_indices.pop(image_index)

    def get_images_by_id(
        self,
        case_ids: List[str],
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Retrieves the last n images and corresponding case ids from the images that were last added to the dataset.

        Args:
            case_ids (List[str]): List with case_ids to get.

        Returns:
            A list of all the images with provided case ids.
        """

        # create list of files as tuple of image id and slice index
        image_slice_ids = [case_id.split("-") for case_id in case_ids]
        image_slice_ids = [
            (split_id[0], int(split_id[1]) if len(split_id) > 1 else None)
            for split_id in image_slice_ids
        ]

        all_images = []
        for case_id, (image_id, slice_index) in zip(case_ids, image_slice_ids):
            image_index = self.__get_image_index(image_id)
            # check if image and mask are in cache
            if image_index in self.image_cache:
                current_image = self.image_cache[image_index]
            # read image and mask from disk otherwise
            else:
                current_image = self.__read_image_as_array(
                    self.image_paths[image_index], norm=True
                )
            all_images.append((current_image[slice_index, :, :], case_id))
        return all_images

    def get_items_for_logging(
        self, case_ids: List[str]
    ) -> List[Tuple[str, str, Optional[int], str]]:
        """
        Creates a list of files as tuple of image id and slice index.

        Args:
            case_ids (List[str]): List with case_ids to get.
        """
        image_slice_ids = [case_id.split("-") for case_id in case_ids]
        image_slice_ids = [
            (split_id[0], int(split_id[1]) if len(split_id) > 1 else None)
            for split_id in image_slice_ids
        ]

        items = []
        for case_id, (image_id, slice_index) in zip(case_ids, image_slice_ids):
            image_index = self.__get_image_index(image_id)
            image_path = self.image_paths[image_index]
            items.append((case_id, image_path, image_id, slice_index))

        return items

    def __image_indices(self) -> Iterable[str]:
        return self.image_slice_indices.keys()

    def image_ids(self) -> Iterable[str]:
        return [self.__get_case_id(image_idx) for image_idx in self.__image_indices()]

    def slices_per_image(self, **kwargs) -> List[int]:
        return [
            DoublyShuffledNIfTIDataset.__read_slice_count(self.image_paths[image_idx])
            for image_idx in self.__image_indices()
        ]

    def size(self) -> int:
        """
        Returns:
            int: Size of the dataset.
        """

        if self.dim == 2:
            size = 0

            for inner_dict in self.image_slice_indices.values():
                size += len(inner_dict)

            return size

        return len(self.image_ids())
