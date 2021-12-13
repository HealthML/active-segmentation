""" Module containing a metrics class for tracking several metrics related to one 3d image """

from typing import Dict, Iterable, Optional
import torch
import torchmetrics

from functional import DiceScore, Sensitivity, Specificity, HausdorffDistance


class CombinedPerImageMetric(torchmetrics.Metric):
    """
    A metrics class that tracks several metrics related to one 3d image whose slices may be scattered across
    different batches. The metrics can be tracked for different confidence levels.

    Args:
        phase (string): Descriptive name of the current pipeline phase, e.g. "train", "val" or "test".
        metrics (Iterable[str]): A list of metric names to be tracked. Available options: "dice", "sensitivity",
            "specificity", and "hausdorff95".
        confidence_levels (Iterable[float]): A list of confidence levels for which the metrics are to be tracked
            separately.
        slices_per_image (int): Number of slices per 3d image.

    Note:
        In this method, the `prediction` tensor is expected to be one output slice of the final sigmoid layer of a
            single-class segmentation task.

    Shape:
        - Prediction: :math:`(height, width)`.
        - Target: Must have the same dimensions as the prediction.
    """

    def __init__(
        self,
        phase: str,
        metrics: Iterable[str],
        confidence_levels: Iterable[float],
        slices_per_image: int,
    ):
        super().__init__()
        self.phase = phase
        # PyTorch does not allow "." in module names, therefore we first replace them by "," and later replace them
        # again by "."
        self.confidence_levels = [
            (confidence_level, f"{str(confidence_level).strip('0')}".replace(".", ","))
            for confidence_level in confidence_levels
        ]

        self._metrics = {
            confidence_level_name: {}
            for _, confidence_level_name in self.confidence_levels
        }

        for _, confidence_level_name in self.confidence_levels:
            for metric in set(metrics):
                if metric == "dice_score":
                    self._metrics[confidence_level_name][metric] = DiceScore(
                        smoothing=0
                    )
                elif metric == "sensitivity":
                    self._metrics[confidence_level_name][metric] = Sensitivity(
                        smoothing=0
                    )
                elif metric == "specificity":
                    self._metrics[confidence_level_name][metric] = Specificity(
                        smoothing=0
                    )
                elif metric == "hausdorff95":
                    self._metrics[confidence_level_name][metric] = HausdorffDistance(
                        slices_per_image=slices_per_image,
                        percentile=0.95,
                        normalize=True,
                    )
                else:
                    raise ValueError(f"Invalid metric name: {metric}")

            # the ModuleDict is required by PyTorch Lightning in order to place the metrics on the correct device
            self._metrics[confidence_level_name] = torch.nn.ModuleDict(
                self._metrics[confidence_level_name]
            )
        self._metrics = torch.nn.ModuleDict(self._metrics)

    def reset(self) -> None:
        """
        Resets internal state such that metric ready for new data.
        """

        for _, confidence_level_name in self.confidence_levels:
            for metric in self._metrics[confidence_level_name].values():
                metric.reset()

        super().reset()

    # pylint: disable=arguments-differ
    def update(self, prediction: torch.Tensor, target: torch.Tensor,) -> None:
        """
        Takes a prediction and a target slice of the 3d image and updates the metrics accordingly.

        Args:
            prediction (Tensor): A prediction slice or a whole 3d image.
            target (Tensor): A target slice or a whole 3d image.
        """

        for confidence_level, confidence_level_name in self.confidence_levels:
            for metric in self._metrics[confidence_level_name].values():
                sharp_prediction = (prediction > confidence_level).int()

                metric.update(sharp_prediction, target)

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes the metrics for the 3d image.

        Returns:
            Dict[string, Tensor]: Mapping of metric names to metric values.
                The keys have the form `<phase>/<metric name>_<confidence_level>`.
        """
        metric_results = {}
        for _, confidence_level_name in self.confidence_levels:
            for metric_name, metric in self._metrics[confidence_level_name].items():
                metric_results[
                    f"{self.phase}/{metric_name}_{confidence_level_name.replace(',','.')}"
                ] = metric.compute()

        return metric_results
