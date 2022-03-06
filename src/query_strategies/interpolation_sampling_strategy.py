""" Module for interpolation sampling strategy """
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union, Literal
import math

import torch
import numpy as np
import itk
from scipy.ndimage import distance_transform_edt
import wandb

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from functional.metrics import DiceScore
from .utils import select_uncertainty_calculation
from .query_strategy import QueryStrategy


# pylint: disable=too-few-public-methods,too-many-branches
class InterpolationSamplingStrategy(QueryStrategy):
    """
    Class for selecting blocks to label by highest uncertainty and then interpolating within those
    blocks to generate additonal pseudo labels.
    Args:
        **kwargs: Optional keyword arguments:
            block_selection (str): The selection strategy for the blocks to interpolate: `"uncertainty"` | `"random"`.
            block_thickness (int): The thickness of the interpolation blocks. Defaults to 5.
            calculation_method (str): Specification of the method used to calculate the uncertainty
                values: `"distance"` |  `"entropy"`.
            exclude_background (bool): Whether to exclude the background dimension in calculating the
                uncertainty value.
            epsilon (float): Small numerical value used for smoothing when using "entropy" as the uncertainty
                metric.
            block_thickness (int): The thickness of the interpolation blocks. Defaults to 5.
            interpolation_type (str): The interpolation algorithm to use.
                values: `"signed-distance"` | `"morph-contour"`.
            interpolation_quality_metric (str): The metric used for evaluating the performance of the interpolation
                e.g. "dice"
            random_state (int, optional): Random state for selecting items to label. Pass an int for reproducible
                outputs across multiple runs.

    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.log_id = 0

    def _randomly_ranked_blocks(
        self, data_module: ActiveLearningDataModule, block_thickness: int
    ) -> Iterator[Tuple[str, int, int]]:
        """
        Randomly selects blocks of the unlabeled data that should be labeled next.
        Args:
            data_module (ActiveLearningDataModule): A data module object providing data.
            block_thickness (int): The thickness of the blocks.

        Returns:
            A sorted iterator of blocks which should be labeled.
        """

        unlabeled_case_ids = []
        for _, case_ids in data_module.unlabeled_dataloader():
            for case_id in case_ids:
                prefix, image_slice_id = case_id.split("_")
                image_id, slice_id = map(int, image_slice_id.split("-"))
                unlabeled_case_ids.append((prefix, image_id, slice_id))

        available_blocks = []
        for prefix, image_id, top_slice_id in unlabeled_case_ids:
            if top_slice_id < block_thickness - 1:
                # We only want blocks which have the full thickness
                continue

            fully_unlabeled = True
            for i in range(block_thickness):
                if (prefix, image_id, top_slice_id - i) not in unlabeled_case_ids:
                    # We only want blocks which are fully unlabeled
                    fully_unlabeled = False

            if not fully_unlabeled:
                continue

            available_blocks.append((prefix, image_id, top_slice_id))

        rng = np.random.default_rng(self.kwargs.get("random_state", None))
        rng.shuffle(available_blocks)

        return available_blocks

    def _uncertainty_ranked_blocks(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        block_thickness: int,
    ) -> Iterator[Tuple[str, int, int]]:
        """
        Selects blocks of the unlabeled data with the highest uncertainty that should be labeled next.
        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            block_thickness (int): The thickness of the blocks.

        Returns:
            A sorted iterator of blocks which should be labeled.
        """

        if isinstance(models, List):
            raise ValueError(
                "Uncertainty sampling is only implemented for one model. You passed a list."
            )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        models.to(device)

        slice_uncertainties = {}

        # For the max_uncertainty_value we have two cases in out datasets:
        # 1. multi-class, single-label (multiple foreground classes that are mutually exclusive) -> max_uncertainty is
        #   1 / num_classes. This is because the softmax layer sums up all to 1 across labels.
        # 2. multi-class, multi-label (multiple foreground classes that can overlap) -> max_uncertainty is 0.5
        max_uncertainty_value = (
            0.5 if data_module.multi_label() else 1 / data_module.num_classes()
        )

        for images, case_ids in data_module.unlabeled_dataloader():
            predictions = models.predict(images.to(device))
            uncertainty_calculation = select_uncertainty_calculation(
                calculation_method=self.kwargs.get("calculation_method", None)
            )
            uncertainty = uncertainty_calculation(
                predictions=predictions,
                max_uncertainty_value=max_uncertainty_value,
                **self.kwargs,
            )

            for idx, case_id in enumerate(case_ids):
                slice_uncertainties[case_id] = uncertainty[idx]

        block_uncertainties = []
        for case_id in slice_uncertainties:
            prefix, image_slice_id = case_id.split("_")
            image_id, top_slice_id = map(int, image_slice_id.split("-"))

            if top_slice_id < block_thickness - 1:
                # We only want blocks which have the full thickness
                continue

            uncertainties = [
                slice_uncertainties.get(f"{prefix}_{image_id}-{top_slice_id - i}")
                for i in range(block_thickness)
            ]

            if None in uncertainties:
                # We only want blocks which are fully unlabeled
                continue

            block_uncertainties.append(
                (
                    sum(uncertainties),
                    (prefix, image_id, top_slice_id),
                )
            )

        block_uncertainties.sort(key=lambda y: y[0])

        return [id for _, id in block_uncertainties]

    # pylint: disable=too-many-locals
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, np.array]]:
        """
        Uses a sampling strategy to select blocks for labeling and generates pseudo labels by interpolation
        between the bottom and the top slice of a block.

        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            items_to_label (int): Number of items that should be selected for labeling.
            **kwargs: Additional, strategy-specific parameters.

        Returns:
            Tuple[List[str], Dict[str, np.array]]: List of IDs of the data items to be labeled and a
            dictionary of pseudo labels with the corresponding IDs as keys.
        """

        block_thickness = self.kwargs.get("block_thickness", 5)
        block_selection = self.kwargs.get("block_selection", "uncertainty")

        block_ids = []

        for thickness in reversed(range(3, block_thickness + 1)):
            ranked_ids = (
                self._uncertainty_ranked_blocks(models, data_module, thickness)
                if block_selection == "uncertainty"
                else self._randomly_ranked_blocks(data_module, thickness)
            )

            for (
                block_prefix,
                block_image_id,
                block_top_slice_id,
            ) in ranked_ids:

                overlaps = [
                    prefix == block_prefix
                    and image_id == block_image_id
                    and (
                        (
                            block_top_slice_id
                            >= top_slice_id
                            > block_top_slice_id - thickness
                        )
                        or (top_slice_id >= block_top_slice_id > top_slice_id - thick)
                    )
                    for (prefix, image_id, top_slice_id), thick in block_ids
                ]

                if not any(overlaps):
                    block_ids.append(
                        ((block_prefix, block_image_id, block_top_slice_id), thickness)
                    )

                if len(block_ids) >= items_to_label / 2:
                    break

            # only continue if the inner loop didn't break
            else:
                continue
            break

        class_ids = [id for id in data_module.id_to_class_names().keys() if id != 0]

        selected_ids = []
        pseudo_labels = {}

        for (prefix, image_id, top_slice_id), thickness in block_ids:
            bottom_slice_id = top_slice_id - thickness + 1

            selected_ids.append(f"{prefix}_{image_id}-{top_slice_id}")
            selected_ids.append(f"{prefix}_{image_id}-{bottom_slice_id}")

            label = data_module.training_set.read_mask_for_image(image_id)
            top = label[top_slice_id, :, :]
            bottom = label[bottom_slice_id, :, :]

            interpolation = self._interpolate_slices(
                top,
                bottom,
                class_ids,
                thickness,
                self.kwargs.get("interpolation_type", None),
            )
            if interpolation is not None:
                for i, pseudo_label in enumerate(interpolation):
                    case_id = f"{prefix}_{image_id}-{top_slice_id - 1 - i}"
                    selected_ids.append(case_id)
                    pseudo_labels[case_id] = pseudo_label

                if self.kwargs.get("interpolation_quality_metric", None) in ["dice"]:
                    _ = self._calculate_and_log_interpolation_quality_score(
                        interpolation=interpolation,
                        ground_truth=label[bottom_slice_id : top_slice_id - 1, :, :],
                        num_classes=data_module.num_classes(),
                        metric=self.kwargs.get("interpolation_quality_metric"),
                    )
        return selected_ids, pseudo_labels

    @staticmethod
    def _interpolate_slices(
        top: np.array,
        bottom: np.array,
        class_ids: Iterable[int],
        block_thickness: int,
        interpolation_type: Literal["signed-distance", "morph-contour"],
    ) -> Optional[np.array]:
        """
        Interpolates between top and bottom slices if possible. Uses a signed distance function to interpolate.

        Args:
            top (np.array): The top slice of the block.
            bottom (np.array): The bottom slice of the block.
            class_ids (Iterable[int]): The class ids.
            block_thickness (int): The thickness of the block.
            interpolation_type (Literal): The type of interpolation to use. One of ["signed-distance", "morph-contour"]
        Returns:
            Optional[np.array]: The interpolated slices between top and bottom.
        """

        interpolation_thickness = block_thickness - 2

        single_class_interpolations = {}
        for class_id in class_ids:
            class_top = top == class_id
            class_bottom = bottom == class_id

            if not np.any(class_top) and not np.any(class_bottom):
                single_class_interpolations[class_id] = np.zeros(
                    (interpolation_thickness, *top.shape), dtype=bool
                )
            elif not np.any(np.logical_and(class_top, class_bottom)):
                return None
            else:
                if interpolation_type == "signed-distance":
                    interpolation_fn = signed_distance_interpolation

                elif interpolation_type == "morph-contour":
                    interpolation_fn = morphological_contour_interpolation

                else:
                    raise ValueError(
                        f"Invalid interpolation type {interpolation_type}."
                    )

                slices = interpolation_fn(class_top, class_bottom, block_thickness)
                single_class_interpolations[class_id] = slices

        result = np.zeros((interpolation_thickness, *top.shape))

        for class_id, interpolation in single_class_interpolations.items():
            result[interpolation] = class_id

        return result

    def _calculate_and_log_interpolation_quality_score(
        self,
        interpolation: np.ndarray,
        ground_truth: np.ndarray,
        num_classes: int,
        metric: Literal["dice"] = "dice",
    ) -> float:
        """
        Calculates a quality score for the interpolations and logs them on wandb.
        Args:
            interpolation (np.ndarray): The interpolated slices.
            ground_truth (np.ndarray): The ground truth slices.
            num_classes (int): Number of classes in the corresponding dataset.
            metric (str): The metric to calculate the score.
                values: `"dice"`.

        Returns:
            The calculated value.
        """
        if metric == "dice":
            dice = DiceScore(
                num_classes=num_classes,
                reduction="mean",
            )
            dice.update(
                prediction=torch.from_numpy(interpolation).int(),
                target=torch.from_numpy(ground_truth).int(),
            )
            mean_dice_score = dice.compute().item()
            wandb.log(
                {
                    "val/interpolation_id": self.log_id,
                    "val/mean_dice_score_interpolation": mean_dice_score
                    if not math.isnan(mean_dice_score)
                    # NaN means that neither the interpolation nor the ground truth include foreground pixels
                    else 1,
                    "val/interpolation_thickness": interpolation.shape[0] + 2,
                }
            )
            self.log_id += 1
            return mean_dice_score
        raise ValueError(f"Chosen metric {metric} not supported. Choose from: 'dice' .")


def signed_distance_interpolation(
    top: np.array,
    bottom: np.array,
    block_thickness: int,
) -> np.array:
    """
    Interpolates between top and bottom slices if possible. Uses a signed distance function to interpolate.

    Args:
        top (np.array): The top slice of the block.
        bottom (np.array): The bottom slice of the block.
        block_thickness (int): The thickness of the block.
    Returns:
        np.array: The interpolated slices between top and bottom.
    """

    def signed_dist(mask):
        inverse_mask = np.ones(mask.shape) - mask
        return distance_transform_edt(mask) - distance_transform_edt(inverse_mask) + 0.5

    def interpolation(start, end, dist):
        dist_start = signed_dist(start)
        dist_end = signed_dist(end)
        interp = (dist_start * (1 - dist)) + (dist_end * dist)
        interp = interp >= 0
        return interp

    step = 1 / (block_thickness - 1)
    interpolation_steps = [i * step for i in range(1, block_thickness - 1)]

    return np.array([interpolation(top, bottom, step) for step in interpolation_steps])


def morphological_contour_interpolation(
    top: np.array,
    bottom: np.array,
    block_thickness: int,
) -> np.array:
    """
    Interpolates between top and bottom slices using the morphological_contour_interpolator from itk.
    https://www.researchgate.net/publication/307942551_ND_morphological_contour_interpolation

    Args:
        top (np.array): The top slice of the block.
        bottom (np.array): The bottom slice of the block.
        block_thickness (int): The thickness of the block.
    Returns:
        np.array: The interpolated slices between top and bottom.
    """

    block = np.zeros((block_thickness, *top.shape))
    block[0, :, :] = bottom
    block[-1, :, :] = top
    image_type = itk.Image[itk.UC, 3]
    itk_img = itk.image_from_array(block.astype(np.uint8), ttype=(image_type,))
    image = itk.morphological_contour_interpolator(itk_img)
    interpolated_block = itk.GetArrayFromImage(image)
    interpolated_slices = interpolated_block[1:-1, :, :]

    return interpolated_slices.astype(bool)
