import os
from torch.utils.data import Dataset
from typing import Any, List, Optional, Union

from .data_module import ActiveLearningDataModule
from .brats_dataset import BraTSDataset


class BraTSDataModule(ActiveLearningDataModule):
    @staticmethod
    def discover_paths(dir_path: str, modality='flair'):
        cases = sorted(os.listdir(dir_path))
        cases = [case for case in cases if not case.startswith('.') and os.path.isdir(os.path.join(dir_path, case))]

        image_paths = [os.path.join(dir_path, case, "{}_{}.nii.gz".format(
            os.path.basename(case), modality)) for case in cases]
        annotation_paths = [os.path.join(dir_path, case, "{}_seg.nii.gz".format(
            os.path.basename(case))) for case in cases]

        return image_paths, annotation_paths

    def __init__(self, data_dir: str, batch_size, shuffle=True, **kwargs):
        """
        :param data_dir: Path of the directory that contains the data.
        :param batch_size: Batch size.
        :param kwargs: Further, dataset specific parameters.
        """

        super().__init__(data_dir, batch_size, shuffle, **kwargs)

    def label_items(self, ids: List[str], labels: Optional[Any] = None) -> None:
        # ToDo: implement labeling logic
        return None

    def _create_training_set(self) -> Union[Dataset, None]:
        train_image_paths, train_annotation_paths = BraTSDataModule.discover_paths(os.path.join(self.data_dir, "train"))
        return BraTSDataset(
            image_paths=train_image_paths,
            annotation_paths=train_annotation_paths
        )

    def _create_validation_set(self) -> Union[Dataset, None]:
        val_image_paths, val_annotation_paths = BraTSDataModule.discover_paths(os.path.join(self.data_dir, "val"))
        return BraTSDataset(
            image_paths=val_image_paths,
            annotation_paths=val_annotation_paths
        )

    def _create_test_set(self) -> Union[Dataset, None]:
        # faked test set
        # ToDo: implement test set
        return self._create_validation_set()

    def _create_unlabeled_set(self) -> Union[Dataset, None]:
        # faked unlabeled set
        # ToDo: implement unlabeled set
        return self._create_training_set()
