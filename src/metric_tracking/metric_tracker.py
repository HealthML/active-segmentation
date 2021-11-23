""" Module containing a helper class for tracking several metrics related to one scan, patient, or task """

from typing import Dict, Iterable
import torch

from functional import DiceScore, Sensitivity, Specificity, HausdorffDistance


class MetricTracker:
    """
    A helper class for tracking several metrics related to one scan, patient, or task.
    Provides utilities to collect the predictions and targets of samples that are scattered over multiple batches.

    Args:
        metrics (Iterable[str]): A list of metric names to be tracked. Available options: "dice", "sensitivity",
            "specificity", and "hausdorff95".
        device: The target device as defined in PyTorch.
    """

    supported_metrics = ["dice", "sensitivity", "specificity", "hausdorff95"]

    def __init__(
        self, metrics: Iterable[str], device: torch.device = torch.device("cpu")
    ):
        self._metrics = {}

        for metric in set(metrics):
            if metric not in MetricTracker.supported_metrics:
                raise ValueError(f"Invalid metric name: {metric}")
            if metric == "dice":
                self._metrics[metric] = DiceScore(smoothing=0, device=device)
            if metric == "sensitivity":
                self._metrics[metric] = Sensitivity(smoothing=0, device=device)
            if metric == "specificity":
                self._metrics[metric] = Specificity(smoothing=0, device=device)
            if metric == "hausdorff95":
                self._metrics[metric] = HausdorffDistance(
                    percentile=0.95, device=device
                )

    def to(self, device: torch.device):
        """
        Moves metric tracker to the given device.

        Args:
            device: The target device as defined in PyTorch.
        """

        for metric in self._metrics.values():
            metric.to(device)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Takes the prediction and target of a given batch and updates the metrics accordingly.

        Args:
            prediction (Tensor): A batch of predictions.
            target (Tensor): A batch of targets.
        """

        for metric in self._metrics.values():
            metric.update(prediction, target)

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes metrics for each scan, patient or task.

        Returns:
            Dict[string, Tensor]: Mapping of metric names to metric values.
        """
        metric_results = {}
        for metric_name, metric in self._metrics.items():
            metric_results[metric_name] = metric.compute()

        return metric_results
