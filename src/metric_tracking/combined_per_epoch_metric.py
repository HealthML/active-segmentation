"""
Module containing a metrics class for tracking and aggregating several metrics related to multiple 3D MRT scans
whose slices may be scattered across different batches.
"""

from typing import Dict, Iterable, List, Optional, Union

import torch
import torchmetrics

from .combined_per_image_metric import CombinedPerImageMetric, MetricName


class CombinedPerEpochMetric(torchmetrics.Metric):
    """
    A metrics class that tracks the metrics of multiple 3D images whose slices may be scattered across different
    batches. Different metrics can be tracked per image, e.g. dice score and Hausdorff distance, and in case of
    multi-label segmentation tasks, the metrics can also be tracked for different confidence levels. If `reduction` is
    not `"none"`, the per-scan metrics are  aggregated into global per-epoch metric values.

    Args:
        metrics (Iterable[str]): A list of metric names to be tracked. Available options: ``"dice_score"`` |
            ``"sensitivity"``| ``"specificity"``, and ``"hausdorff95"``.
        id_to_class_names (Dict[int, str]): A mapping of class indices to descriptive class names.
        image_ids (Iterable[str]): List of the ids of all images for which the metrics are to be tracked.
        slices_per_image (Union[int, List[int]]): Number of slices per 3d image. If a single integer value is
            provided, it is assumed that all images of the dataset have the same number of slices.
        include_background_in_aggregated_metrics (bool, optional): if `False`, class channel index 0 (background class)
            is excluded from the calculation of aggregated metrics. This parameter is used only if `multi_label` is set
            to `False`. Defaults to `True`.
        multi_label (bool, optional): Determines whether the data is multi-label or not (default = `False`).
        confidence_levels (Iterable[float], optional): A list of confidence levels for which the metrics are to be
            tracked separately. This parameter is used only if `multi_label` is set to `True`. Defaults to `[0.5]`.
       reduction (string, optional):  Reduction function that is to be used to aggregate the metric values of all
            images, must be either "mean", "max", "min" or "none". Defaults to `"mean"`.
    Note:
        If `multi_label` is `False`, the `prediction` tensor is expected to be either the output of the final softmax
        layer of a segmentation model or a label-encoded, sharp prediction. In the first case, the prediction tensor
        must be of floating point type and have the shape :math:`(N, C, X, Y)` or :math:`(N, C, X, Y, Z)` where
        `N = batch size` and `C = number of classes`. In the second case, the prediction tensor must be of integer type
        and have the shape :math:`(N, X, Y)` or :math:`(N, X, Y, Z)`. The `target` tensor is expected to be
        label-encoded in both cases. Thus, it needs to have the shape :math:`(N, X, Y)` or :math:`(N, X, Y, Z)` and be
        of integer type.

        If `multi_label` is `True`, the `prediction` tensor is expected to be either the output of the final sigmoid
        layer of a segmentation model or a sharp prediction. In the first case, the prediction tensors needs to be of
        floating point type and in the second type of integer type. the In both cases the prediction tensor needs to
        have the shape :math:`(N, C, X, Y)` or :math:`(N, C, X, Y, Z)`. The target tensor is expected to contain sharp
        predictions and to have the shape :math:`(N, C, X, Y)` or :math:`(N, C, X, Y, Z)`.

    Shape:
        - Prediction: :math:`(N, C, X, Y, ...)`, where `N = batch size` and `C = number of classes` (see Notes above).
        - Target: :math:`(N, X, Y, ...)`, or :math:`(N, C, X, Y, ...)` (see Notes above).
        - Image_ids: :math:`(N)`, where `N = batch size`.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        metrics: Iterable[MetricName],
        id_to_class_names: Dict[int, str],
        image_ids: Iterable[str],
        slices_per_image: Union[int, List[int]],
        include_background_in_aggregated_metrics: bool = False,
        multi_label: bool = False,
        confidence_levels: Optional[Iterable[float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.metrics = metrics
        self.include_background_in_aggregated_metrics = include_background_in_aggregated_metrics
        self.multi_label = multi_label
        self.confidence_levels = (
            confidence_levels if confidence_levels is not None else [0.5]
        )
        self.slices_per_image = slices_per_image
        self.id_to_class_names = id_to_class_names

        self._metrics_per_image = torch.nn.ModuleDict(
            {
                image_id: CombinedPerImageMetric(
                    self.metrics,
                    id_to_class_names,
                    multi_label=multi_label,
                    slices_per_image=self.slices_per_image[idx]
                    if isinstance(self.slices_per_image, list)
                    else self.slices_per_image,
                    confidence_levels=self.confidence_levels,
                )
                for idx, image_id in enumerate(image_ids)
            }
        )
        self.metrics_to_compute = set()

        if reduction not in ["mean", "max", "min", "none"]:
            raise ValueError("Invalid reduction method.")

        self.reduction = reduction

    def reset(self) -> None:
        """
        Resets internal state such that metric ready for new data.
        """

        for metric in self._metrics_per_image.values():
            metric.reset()
        super().reset()

    # pylint: disable=arguments-differ
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        image_ids: Iterable[str],
    ) -> None:
        """
        Takes the prediction and target of a given batch and updates the metrics accordingly.

        Args:
            prediction (Tensor): A batch of predictions.
            target (Tensor): A batch of targets.
            image_ids (Iterable[string]): Image IDs of each slice in the prediction and target batches.
        """

        for idx, image_id in enumerate(image_ids):
            self._metrics_per_image[image_id].update(prediction[idx], target[idx])
            self.metrics_to_compute.add(image_id)

    def _reduce_metric(self, metric: torch.Tensor) -> torch.Tensor:
        """
        Aggregates metric values.

        Args:
            metric (Tensor): Metric to be aggregated.

        Returns:
            Tensor: Aggregated metric.
        """

        if self.reduction == "mean":
            return metric.mean()
        if self.reduction == "max":
            return metric.max()
        if self.reduction == "min":
            return metric.min()
        return metric

    # pylint: disable=too-many-branches
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes the metrics for each scan and aggregates them if `reduction` is not `"none"`.

        Returns:
            Dict[string, Tensor]: Mapping of metric names to metric values.
                If `reduction` is `"none"`, the keys have the form
                `<metric name>_<confidence_level>_<image ID>`.
                Otherwise the keys have the form `<reduction>_<metric name>_<confidence_level>`
                If `"metrics_to_aggregate"` is provided and `reduction` is not `"none"`, the dictionary additionally
                contains the keys `<reduction>_aggregated_<confidence_level>`.
        """
        per_image_metrics = {}

        for image_id in self.metrics_to_compute:
            metrics = self._metrics_per_image[image_id].compute()
            for metric_name, metric_value in metrics.items():
                if self.reduction == "none":
                    per_image_metrics[f"{metric_name}_{image_id}"] = metric_value
                else:
                    if metric_name not in per_image_metrics:
                        per_image_metrics[metric_name] = []
                    per_image_metrics[metric_name].append(metric_value)

        if self.reduction == "none":
            return per_image_metrics

        aggregated_metrics = {}

        for metric_name in per_image_metrics.keys():
            aggregated_metrics[metric_name] = self._reduce_metric(
                torch.tensor(per_image_metrics[metric_name])
            )

        for metric_name in self.metrics:
            if self.multi_label:
                for confidence_level in self.confidence_levels:
                    average_metric = []
                    for class_id, class_name in self.id_to_class_names.items():
                        if class_id != 0 or self.multi_label or self.include_background_in_aggregated_metrics:
                            per_class_metric = aggregated_metrics[
                                f"{metric_name}_{class_name}_{confidence_level}"
                            ]
                            average_metric.append(per_class_metric)
                    aggregated_metrics[
                        f"{self.reduction}_{metric_name}_{confidence_level}"
                    ] = self._reduce_metric(torch.Tensor(average_metric))
            else:
                average_metric = []
                for class_id, class_name in self.id_to_class_names.items():
                    if class_id != 0 or self.multi_label or self.include_background_in_aggregated_metrics:
                        per_class_metric = aggregated_metrics[f"{metric_name}_{class_name}"]
                        average_metric.append(per_class_metric)
                aggregated_metrics[
                    f"{self.reduction}_{metric_name}"
                ] = self._reduce_metric(torch.Tensor(average_metric))

        return aggregated_metrics
