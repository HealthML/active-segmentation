"""
Module containing a metrics class for tracking and aggregating several metrics related to multiple 3D MRT scans
whose slices may be scattered across different batches.
"""

from typing import Dict, Iterable, Optional

import torch
import torchmetrics

from .combined_per_image_metric import CombinedPerImageMetric


class CombinedPerEpochMetric(torchmetrics.Metric):
    """
    A metrics class that tracks the metrics of multiple 3D images whose slices may be scattered across
    different batches. Different metrics can be tracked per scan, e.g. dice score and Hausdorff distance, and the
    metrics can also be tracked for different confidence levels. If `reduction` is not `"none"`, the per-scan metrics
    are  aggregated into global per-epoch metric values.

    Args:
        stage (string): Descriptive name of the current stage, e.g. "train", "val" or "test".
        metrics (Iterable[str]): A list of metric names to be tracked. Available options: "dice", "sensitivity",
            "specificity", and "hausdorff95".
        confidence_levels (Iterable[float]): A list of confidence levels for which the metrics are to be tracked
            separately.
        image_ids (Iterable[str]): List of the ids of all images for which the metrics are to be tracked.
        slices_per_image (int): Number of slices per 3d image.
        metrics_to_aggregate (Iterable[str], optional): A list of metric names from which to calculate an aggregated
            metric. Must be a subset of the metric names passed to the `metrics` parameter. If, for example,
            `metrics_to_aggregate=["dice", "hausdorff95"]` and `reduction="mean"`, the mean of the dice score and the
            Hausdorff distance is calculated and returned as the `aggregated` metric. If `reduction` is `"none"`, this
                parameter will be ignored.
        reduction (string, optional):  Reduction function that is to be used to aggregate the metric values of all 3d
            images, must be either "mean", "sum" or "none". Default: `"mean"`.
    Note:
        In this method, the `prediction` tensor is expected to be the output of the final sigmoid layer of a
            single-class segmentation task.

    Shape:
        - Prediction: :math:`(N, height, width)`, where `N = batch size`.
        - Target: Must have the same dimensions as the prediction.
        - Image_ids: :math:`(N)`, where `N = batch size`.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        stage: str,
        metrics: Iterable[str],
        confidence_levels: Iterable[float],
        image_ids: Iterable[str],
        slices_per_image: int,
        metrics_to_aggregate: Optional[Iterable[str]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.stage = stage
        self.metrics = metrics
        self.confidence_levels = confidence_levels
        self.slices_per_image = slices_per_image

        if metrics_to_aggregate is not None and not set(metrics_to_aggregate).issubset(
            set(metrics)
        ):
            raise ValueError(
                "'metrics_to_aggregate must be a subset' of the metric names passed to the 'metrics' parameter."
            )

        self.metrics_to_aggregate = metrics_to_aggregate
        self._metrics_per_image = torch.nn.ModuleDict(
            {
                image_id: CombinedPerImageMetric(
                    self.stage,
                    self.metrics,
                    self.confidence_levels,
                    slices_per_image=self.slices_per_image,
                )
                for image_id in image_ids
            }
        )
        self.metrics_to_compute = set()

        if reduction not in ["mean", "sum", "none"]:
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
        self, prediction: torch.Tensor, target: torch.Tensor, image_ids: Iterable[str],
    ) -> None:
        """
        Takes the prediction and target of a given batch and updates the metrics accordingly.

        Args:
            prediction (Tensor): A batch of predictions.
            target (Tensor): A batch of targets.
            image_ids (Iterable[string]): Image IDs of each slice in the prediction and target batches.
        """

        for idx, image_id in enumerate(image_ids):
            self._metrics_per_image[image_id].update(
                prediction[idx].squeeze(dim=0), target[idx].squeeze(dim=0)
            )
            self.metrics_to_compute.add(image_id)

    # pylint: disable=too-many-branches
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes the metrics for each scan and aggregates them if `reduction` is not `"none"`.

        Returns:
            Dict[string, Tensor]: Mapping of metric names to metric values.
                If `reduction` is `"none"`, the keys have the form
                `<stage>/<metric name>_<confidence_level>_<image ID>`.
                Otherwise the keys have the form `<stage>/<reduction>_<metric name>_<confidence_level>`
                If `"metrics_to_aggregate"` is provided and `reduction` is not `"none"`, the dictionary additionally
                contains the keys `<stage>/<reduction>_aggregated_<confidence_level>`.
        """
        per_image_metrics = {}

        for image_id in self.metrics_to_compute:
            metrics = self._metrics_per_image[image_id].compute()
            for metric_name, metric_value in metrics.items():
                if self.reduction == "none":
                    per_image_metrics[
                        f"{self.stage}/{metric_name.lstrip(f'{self.stage}/')}_{image_id}"
                    ] = metric_value
                else:
                    if metric_name not in per_image_metrics:
                        per_image_metrics[metric_name] = []
                    per_image_metrics[metric_name].append(metric_value)

        if self.reduction == "none":
            return per_image_metrics

        aggregated_metrics = {}

        for metric_name, metric_value in per_image_metrics.items():
            aggregated_metric_name = (
                f"{self.stage}/{self.reduction}_{metric_name.lstrip(f'{self.stage}/')}"
            )
            if self.reduction == "mean":
                aggregated_metrics[aggregated_metric_name] = torch.tensor(
                    per_image_metrics[metric_name]
                ).mean()
            elif self.reduction == "sum":
                aggregated_metrics[aggregated_metric_name] = torch.tensor(
                    per_image_metrics[metric_name]
                ).sum()

        if self.metrics_to_aggregate is not None:
            for confidence_level in self.confidence_levels:
                metric_values = []
                for metric_name in self.metrics_to_aggregate:
                    metric_value = aggregated_metrics[
                        f"{self.stage}/{self.reduction}_{metric_name}_{str(confidence_level).strip('0')}"
                    ]
                    if "hausdorff" in metric_name:
                        # invert Hausdorff distances for a meaningful aggregation with the other metrics where 1.0 is
                        # the best value and 0.0 the worst
                        metric_value = torch.as_tensor(1.0) - metric_value

                    metric_values.append(metric_value)

                aggregated_metric_name = f"{self.stage}/{self.reduction}_aggregated_{str(confidence_level).strip('0')}"
                if self.reduction == "mean":
                    aggregated_metrics[aggregated_metric_name] = torch.tensor(
                        metric_values
                    ).mean()
                elif self.reduction == "sum":
                    aggregated_metrics[aggregated_metric_name] = torch.tensor(
                        metric_values
                    ).sum()

        return aggregated_metrics
