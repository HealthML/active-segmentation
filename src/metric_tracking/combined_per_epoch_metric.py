""" Module containing a metrics class for tracking and aggregating several metrics related to multiple 3D MRT scans whose slices may be scattered across different batches. """

from typing import Dict, Iterable, Optional

import torch
import torchmetrics

from .combined_per_scan_metric import CombinedPerScanMetric


class CombinedPerEpochMetric(torchmetrics.Metric):
    """
    A metrics class that tracks the metrics of multiple 3D MRT scans (cases) whose slices may be scattered across different batches.
    Different metrics can be tracked per scan, e.g. dice score and Hausdorff distance, and the metrics can also be tracked
    for different confidence levels. If `reduction` is not `"none"`, the per-scan metrics are  aggregated into global 
    per-epoch metric values.

    Args:
        phase (string): Descriptive name of the current pipeline phase, e.g. "train", "val" or "test".
        metrics (Iterable[str]): A list of metric names to be tracked. Available options: "dice", "sensitivity",
            "specificity", and "hausdorff95".
        confidence_levels (Iterable[float]): A list of confidence levels for which the metrics are to be tracked separately.
        metrics_to_aggregate (Iterable[str], optional): A list of metric names from which to calculate an aggregated 
            metric. Must be a subset of the metric names passed to the `metrics` parameter. If, for example,
            `metrics_to_aggregate=["dice", "hausdorff95"]` and `reduction="mean"`, the mean of the dice score and the 
            Hausdorff distance is calculated and returned as the `aggregated` metric. If `reduction` is `"none"`, this parameter
            will be ignored.
        reduction (string, optional):  Reduction function that is to be used to aggregate the metric values of all scans / cases, must be
            either "mean", "sum" or "none". Default: `"mean"`.

    Note:
        In this method, the `prediction` tensor is expected to be the output of the final sigmoid layer of a single-class segmentation task.

    Shape:
        - Prediction: :math:`(N, height, width)`, where `N = batch size`.
        - Target: Must have the same dimensions as the prediction.
        - Case_ids: :math:`(N)`, where `N = batch size`.
    """

    def __init__(
        self,
        phase: str,
        metrics: Iterable[str],
        confidence_levels: Iterable[float],
        metrics_to_aggregate: Optional[Iterable[str]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.phase = phase
        self.metrics = metrics
        self.confidence_levels = confidence_levels

        if metrics_to_aggregate is not None and not set(metrics_to_aggregate).issubset(
            set(metrics)
        ):
            raise ValueError(
                "'metrics_to_aggregate must be a subset' of the metric names passed to the 'metrics' parameter."
            )

        self.metrics_to_aggregate = metrics_to_aggregate
        self._metrics_per_case = {}

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method.")

        self.reduction = reduction

    def reset(self) -> None:
        """
        Resets internal state such that metric ready for new data.
        """
        
        for metric in self._metrics_per_case.values():
            metric.reset()
        super().reset()

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, case_ids: Iterable[str],
    ) -> None:
        """
        Takes the prediction and target of a given batch and updates the metrics accordingly.

        Args:
            prediction (Tensor): A batch of predictions.
            target (Tensor): A batch of targets.
            case_ids (Iterable[string]): Case IDs of each slice in the prediction and target batches.
        """

        for idx, case_id in enumerate(case_ids):
            if case_id not in self._metrics_per_case:
                self._metrics_per_case[case_id] = CombinedPerScanMetric(
                    self.phase, self.metrics, self.confidence_levels
                )

            self._metrics_per_case[case_id].update(prediction[idx], target[idx])

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes the metrics for each scan and aggregates them if `reduction` is not `"none"`.

        Returns:
            Dict[string, Tensor]: Mapping of metric names to metric values.
                If `reduction` is `"none"`, the keys have the form `<phase>/<metric name>_<confidence_level>_<case ID>`.
                Otherwise the keys have the form `<phase>/<reduction>_<metric name>_<confidence_level>`
                If `"metrics_to_aggregate"` is provided and `reduction` is not `"none"`, the dictionary additionally contains 
                the keys `<phase>/<reduction>_aggregated_<confidence_level>`.
        """
        per_case_metrics = {}

        for case_id, case_metric in self._metrics_per_case.items():
            case_metrics = case_metric.compute()
            for metric_name, metric_value in case_metrics.items():
                if self.reduction == "none":
                    per_case_metrics[
                        f"{self.phase}/{metric_name.lstrip(f'{self.phase}/')}_{case_id}"
                    ] = metric_value
                else:
                    if metric_name not in per_case_metrics:
                        per_case_metrics[metric_name] = []
                    per_case_metrics[metric_name].append(metric_value)

        if self.reduction == "none":
            return per_case_metrics

        aggregated_metrics = {}

        for metric_name, metric_value in per_case_metrics.items():
            aggregated_metric_name = (
                f"{self.phase}/{self.reduction}_{metric_name.lstrip(f'{self.phase}/')}"
            )
            if self.reduction == "mean":
                aggregated_metrics[aggregated_metric_name] = torch.tensor(
                    per_case_metrics[metric_name]
                ).mean()
            elif self.reduction == "sum":
                aggregated_metrics[aggregated_metric_name] = torch.tensor(
                    per_case_metrics[metric_name]
                ).sum()

        if self.metrics_to_aggregate is not None:
            for confidence_level in self.confidence_levels:
                metric_values = []
                for metric_name in self.metrics_to_aggregate:
                    metric_values.append(
                        aggregated_metrics[
                            f"{self.phase}/{self.reduction}_{metric_name}_{confidence_level:.1f}"
                        ]
                    )

                aggregated_metric_name = (
                    f"{self.phase}/{self.reduction}_aggregated_{confidence_level:.1f}"
                )
                if self.reduction == "mean":
                    aggregated_metrics[aggregated_metric_name] = torch.tensor(
                        metric_values
                    ).mean()
                elif self.reduction == "sum":
                    aggregated_metrics[aggregated_metric_name] = torch.tensor(
                        metric_values
                    ).sum()

        return aggregated_metrics
