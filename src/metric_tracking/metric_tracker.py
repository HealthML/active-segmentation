""" Module containing a helper class for tracking several metrics related to one scan, patient, or task """

from typing import Dict, Iterable, Optional
import torch

from functional import DiceScore, Sensitivity, Specificity, HausdorffDistance


class MetricTracker:
    """
    A helper class for tracking several metrics related to one scan, patient, or task.
    Provides utilities to collect the predictions and targets of samples that are scattered over multiple batches.

    Args:
        metrics (Iterable[str]): A list of metric names to be tracked. Available options: "dice", "sensitivity",
            "specificity", and "hausdorff95".
        groups (Iterable[str], optional): A list of group names for which the metrics are to be tracked separately.
        device: The target device as defined in PyTorch.
    """

    supported_metrics = ["dice", "sensitivity", "specificity", "hausdorff95"]

    def __init__(
        self,
        metrics: Iterable[str],
        groups: Optional[Iterable[str]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        if groups is None:
            self.groups = ["default"]

        self._metrics = {group_name: {} for group_name in groups}

        for group_name in groups:
            for metric in set(metrics):
                if metric not in MetricTracker.supported_metrics:
                    raise ValueError(f"Invalid metric name: {metric}")
                if metric == "dice":
                    self._metrics[group_name][metric] = DiceScore(
                        smoothing=0, device=device
                    )
                if metric == "sensitivity":
                    self._metrics[group_name][metric] = Sensitivity(
                        smoothing=0, device=device
                    )
                if metric == "specificity":
                    self._metrics[group_name][metric] = Specificity(
                        smoothing=0, device=device
                    )
                if metric == "hausdorff95":
                    self._metrics[group_name][metric] = HausdorffDistance(
                        percentile=0.95, device=device
                    )

    def to(self, device: torch.device):
        """
        Moves metric tracker to the given device.

        Args:
            device: The target device as defined in PyTorch.
        """

        self.device = device
        for group in self._metrics.values():
            for metric in group.values():
                metric.to(device)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        group_name: str = "default",
    ) -> None:
        """
        Takes the prediction and target of a given batch and updates the metrics accordingly.

        Args:
            prediction (Tensor): A batch of predictions.
            target (Tensor): A batch of targets.
            group_name (str, optional): Name of the group to update.
        """

        if group_name not in self._metrics:
            raise ValueError(f"Metric tracking group {group_name} does not exist.")

        for metric in self._metrics[group_name].values():
            metric.update(prediction, target)

    def compute(self, group_name: str = "default") -> Dict[str, torch.Tensor]:
        """
        Computes metrics for each scan, patient or task.

        Args:
            group_name (str, optional): Name of the group for which the metrics are to be computed.

        Returns:
            Dict[string, Tensor]: Mapping of metric names to metric values.
        """
        metric_results = {}
        for metric_name, metric in self._metrics[group_name].items():
            metric_results[metric_name] = metric.compute()

        return metric_results
