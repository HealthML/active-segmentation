""" Module for interpolation sampling strategy """
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods
class InterpolationSamplingStrategy(QueryStrategy):
    """
    Class for selecting blocks to label by highest uncertainty and then interpolating within those
    blocks to generate additonal pseudo labels.
    """

    # pylint: disable=too-many-locals
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, np.array]]:
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

        if isinstance(models, List):
            raise ValueError(
                "Uncertainty sampling is only implemented for one model. You passed a list."
            )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        models.to(device)

        slice_uncertainties = {}

        for images, case_ids in data_module.unlabeled_dataloader():
            predictions = models.predict(images.to(device))
            uncertainty = (
                torch.sum(torch.abs(0.5 - predictions), (1, 2, 3)).cpu().numpy()
            )

            for idx, case_id in enumerate(case_ids):
                slice_uncertainties[case_id] = uncertainty[idx]

        block_thickness = kwargs.get("block_thickness", 5)

        block_uncertainties = []
        for case_id in slice_uncertainties:
            prefix, image_slice_ids = case_id.split("_")
            image_id, slice_id = map(int, image_slice_ids.split("-"))

            if slice_id < block_thickness - 1:
                # We only want blokcs which have the full thickness
                continue

            uncertainties = [
                slice_uncertainties.get(f"{prefix}_{image_id}-{slice_id - i}")
                for i in range(block_thickness)
            ]

            if None in uncertainties:
                # We only want blocks which are fully unlabeled
                continue

            block_uncertainties.append(
                (
                    sum(uncertainties),
                    (prefix, image_id, slice_id),
                )
            )

        block_uncertainties.sort(key=lambda y: y[0])

        block_ids = []
        for _, (
            block_prefix,
            block_image_id,
            block_top_slice_id,
        ) in block_uncertainties:

            overlaps = [
                prefix == block_prefix
                and image_id == block_image_id
                and abs(top_slice_id - block_top_slice_id) < block_thickness
                for prefix, image_id, top_slice_id in block_ids
            ]

            if not any(overlaps):
                block_ids.append((block_prefix, block_image_id, block_top_slice_id))

            if len(block_ids) >= items_to_label / 2:
                break

        pseudo_labels = {}

        class_ids = [id for id in data_module.id_to_class_names().keys() if id != 0]

        for prefix, image_id, slice_id in block_ids:
            label = data_module.training_set.read_mask_for_image(image_id)
            top = label[slice_id, :, :]
            bottom = label[slice_id - block_thickness + 1, :, :]
            interpolation = self._interpolate_slices(
                top, bottom, class_ids, block_thickness
            )
            # TODO: Use save returned pseudo labels in pseudo_labels

        # TODO: Construct selected_ids
        selected_ids = []

        return selected_ids, pseudo_labels

    @staticmethod
    def _interpolate_slices(
        top: np.array,
        bottom: np.array,
        class_ids: Iterable[int],
        block_thickness: int,
    ) -> Optional[np.array]:
        """
        Interpolates between top and bottom slices if possible.

        Args:
            top (np.array): The top slice of the block.
            bottom (np.array): The bottom slice of the block.
            class_ids (Iterable[int]): The class ids.
            block_thickness (int): THe thickness of the block.
        """

        def signed_dist(mask):
            inverse_mask = np.ones(mask.shape) - mask
            return (
                distance_transform_edt(mask)
                - distance_transform_edt(inverse_mask)
                + 0.5
            )

        def interpolate(start, end, dist):
            dist_start = signed_dist(start)
            dist_end = signed_dist(end)
            interp = (dist_start * (1 - dist)) + (dist_end * dist)
            interp = interp >= 0
            return interp

        step = 1 / (block_thickness - 1)
        interpolation_steps = [i * step for i in range(1, block_thickness - 1)]
        single_class_interpolations = {}
        for class_id in class_ids:
            class_top = top == class_id
            class_bottom = bottom == class_id

            if not np.any(class_top) and not np.any(class_bottom):
                single_class_interpolations[class_id] = np.zeros(
                    (len(interpolation_steps), *top.shape), dtype=bool
                )
            elif not np.any(np.logical_and(class_top, class_bottom)):
                return None
            else:
                slices = [
                    interpolate(class_top, class_bottom, step)
                    for step in interpolation_steps
                ]
                single_class_interpolations[class_id] = np.array(slices)

        result = np.zeros((len(interpolation_steps), *top.shape))

        for class_id, interpolation in single_class_interpolations.items():
            result[interpolation] = class_id

        return result
