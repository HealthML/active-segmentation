""" Module containing a helper class for tracking and aggregating several metrics related to one 3D MRT scan. """

from typing import Dict, Iterable

import torch

from .metric_tracker import MetricTracker


class MetricPerCaseTracker:
    """
    A helper class for tracking and aggregating several metrics related to one 3D MRT scan.
    Provides utilities to collect the predictions and targets of scans whose slices are scattered over multiple batches.

    Args:
        metrics (Iterable[str]): A list of metric names to be tracked. Available options: "dice", "sensitivity",
            "specificity", and "hausdorff95".
        reduce (string):  Reduction function that is to be used to aggregate the metric values of all cases, must be
            either "mean", "sum" or "none".
        device: The target device as defined in PyTorch.
    """

    def __init__(
        self,
        metrics: Iterable[str],
        reduce: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.metrics = metrics
        self._metrics_per_case = {}

        if reduce not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method.")

        self.reduce = reduce

    def to(self, device: torch.device):
        """
        Moves metric tracker to the given device.

        Args:
            device: The target device as defined in PyTorch.
        """

        self.device = device
        for metric_tracker in self._metrics_per_case.values():
            metric_tracker.to(device)

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, case_ids: Iterable[str]
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
                self._metrics_per_case[case_id] = MetricTracker(
                    self.metrics, self.device
                )

            self._metrics_per_case[case_id].update(prediction[idx], target[idx])

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes per-scan metrics and aggregates them if `self.reduce` is not `None`.

        Returns:
            Dict[string, Tensor]: Mapping of metric names to metric values.
        """

        if self.reduce == "none":
            aggregated_metrics = {metric: {} for metric in self.metrics}
        else:
            aggregated_metrics = {metric: [] for metric in self.metrics}

        for case_id, case_metric_tracker in self._metrics_per_case.items():
            case_metrics = case_metric_tracker.compute()

            for metric in self.metrics:
                if self.reduce == "none":
                    aggregated_metrics[metric][case_id] = case_metrics[metric]
                else:
                    aggregated_metrics[metric].append(case_metrics[metric])

        if self.reduce == "mean":
            for metric in self.metrics:
                aggregated_metrics[metric] = torch.tensor(
                    aggregated_metrics[metric], device=self.device
                ).mean()
        if self.reduce == "sum":
            for metric in self.metrics:
                aggregated_metrics[metric] = torch.tensor(
                    aggregated_metrics[metric], device=self.device
                ).sum()

        return aggregated_metrics
