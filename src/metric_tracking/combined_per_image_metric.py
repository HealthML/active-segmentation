""" Module containing a metrics class for tracking several metrics related to one 3d image """

from typing import Dict, Iterable, Literal, Optional
import torch
import torchmetrics

from functional import DiceScore, Sensitivity, Specificity, HausdorffDistance


MetricName = Literal["dice_score", "sensitivity", "specificity", "hausdorff95"]


class CombinedPerImageMetric(torchmetrics.Metric):
    """
    A metrics class that tracks several metrics related to one 3d image whose slices may be scattered across
    different batches. In case of multi-label segmentation tasks, the metrics can be tracked for different confidence
    levels.

    Args:
        metrics (Iterable[str]): A list of metric names to be tracked. Available options: ``"dice_score"`` |
            ``"sensitivity"``| ``"specificity"``, and ``"hausdorff95"``.
        id_to_class_names (Dict[int, str]): A mapping of class indices to descriptive class names.
        confidence_levels (Iterable[float], optional): A list of confidence levels for which the metrics are to be
            tracked separately. This parameter is used only if `multi_label` is set to `True`. Defaults to `[0.5]`.
        multi_label (bool, optional): Determines whether the task is a multi-label segmentation task or not
            (default = `False`).
        slices_per_image (int): Number of slices per 3d image.

    Note:
        If `multi_label` is `False`, the `prediction` tensor is expected to be either the output of the final softmax
        layer of a segmentation model or a label-encoded, sharp prediction. In the first case, the prediction tensor
        must be of floating point type and have the shape :math:`(C, X, Y)` or :math:`(C, X, Y, Z)` where
        `C = number of classes`. In the second case, the prediction tensor must be of integer type and have the shape
        :math:`(X, Y)` or :math:`(X, Y, Z)`. The `target` tensor is expected to be label-encoded in both cases. Thus,
        it needs to have the shape :math:`(X, Y)` or :math:`(X, Y, Z)` and be of integer type.

        If `multi_label` is `True`, the `prediction` tensor is expected to be either the output of the final sigmoid
        layer of a segmentation model or a sharp prediction. In the first case, the prediction tensors needs to be of
        floating point type and in the second type of integer type. the In both cases the prediction tensor needs to
        have the shape :math:`(C, X, Y)` or :math:`(C, X, Y, Z)`. The target tensor is expected to contain sharp
        predictions and to have the shape :math:`(C, X, Y)` or :math:`(C, X, Y, Z)`.

    Shape:
        - Prediction: :math:`(C, X, Y, ...)`, where `C = number of classes` (see Notes above).
        - Target: :math:`(X, Y, ...)`, or :math:`(C, X, Y, ...)` (see Notes above).
    """

    def __init__(
        self,
        metrics: Iterable[MetricName],
        id_to_class_names: Dict[int, str],
        slices_per_image: int,
        multi_label: bool = False,
        confidence_levels: Optional[Iterable[float]] = None,
    ):
        super().__init__()

        self.id_to_class_names = id_to_class_names
        self.num_classes = len(id_to_class_names)
        self.multi_label = multi_label

        confidence_levels = (
            confidence_levels if confidence_levels is not None else [0.5]
        )
        # PyTorch does not allow "." in module names, therefore we first replace them by "," and later replace them
        # again by "."
        self.confidence_levels = [
            (confidence_level, f"{str(confidence_level).rstrip('0')}".replace(".", ","))
            for confidence_level in confidence_levels
        ]

        if self.multi_label:
            self._metrics = {
                confidence_level_name: {}
                for _, confidence_level_name in self.confidence_levels
            }

            for _, confidence_level_name in self.confidence_levels:
                for metric in set(metrics):
                    self._metrics[confidence_level_name][metric] = self._create_metric(
                        metric, slices_per_image
                    )

                # the ModuleDict is required by PyTorch Lightning in order to place the metrics on the correct device
                self._metrics[confidence_level_name] = torch.nn.ModuleDict(
                    self._metrics[confidence_level_name]
                )
        else:
            self._metrics = {
                metric: self._create_metric(metric, slices_per_image)
                for metric in set(metrics)
            }

        self._metrics = torch.nn.ModuleDict(self._metrics)

    def _create_metric(
        self,
        metric: MetricName,
        slices_per_image: int,
    ) -> torchmetrics.Metric:
        """
        Creates a metric object of the specified metric type.

        Args:
            metric (str): Name of the metric type for which a metric object is to be created. Available options:
                ``"dice_score"`` | ``"sensitivity"``| ``"specificity"``, and ``"hausdorff95"``.
            slices_per_image (int): Number of slices per 3d image.

        Returns:
            torchmetrics.Metric: A metric object.
        """

        if metric == "dice_score":
            return DiceScore(
                self.num_classes,
                convert_to_one_hot=not self.multi_label,
                epsilon=0,
                include_background=self.multi_label,
            )
        if metric == "sensitivity":
            return Sensitivity(
                self.num_classes,
                convert_to_one_hot=not self.multi_label,
                epsilon=0,
                include_background=self.multi_label,
            )
        if metric == "specificity":
            return Specificity(
                self.num_classes,
                convert_to_one_hot=not self.multi_label,
                epsilon=0,
                include_background=self.multi_label,
            )
        if metric == "hausdorff95":
            return HausdorffDistance(
                self.num_classes,
                slices_per_image,
                convert_to_one_hot=not self.multi_label,
                include_background=self.multi_label,
                normalize=True,
                percentile=0.95,
            )
        raise ValueError(f"Invalid metric name: {metric}.")

    def reset(self) -> None:
        """
        Resets internal state such that metric ready for new data.
        """

        if self.multi_label:
            for _, confidence_level_name in self.confidence_levels:
                for metric in self._metrics[confidence_level_name].values():
                    metric.reset()
        else:
            for metric in self._metrics.values():
                metric.reset()

        super().reset()

    # pylint: disable=arguments-differ
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Takes a prediction and a target slice of the 3d image and updates the metrics accordingly.

        Args:
            prediction (Tensor): A prediction slice or a whole 3d image.
            target (Tensor): A target slice or a whole 3d image.
        """

        if self.multi_label:
            for confidence_level, confidence_level_name in self.confidence_levels:
                for metric in self._metrics[confidence_level_name].values():

                    sharp_prediction = (prediction > confidence_level).int()

                    metric.update(sharp_prediction, target)
        else:
            for metric in self._metrics.values():
                if prediction.dtype == torch.int:
                    sharp_prediction = prediction
                else:
                    sharp_prediction = torch.argmax(prediction, dim=0).int()
                metric.update(sharp_prediction, target)

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes the metrics for the image.

        Returns:
            Dict[string, Tensor]: Mapping of metric names to metric values.
                The keys have the form `<metric name>_<class_name>_<confidence_level>` for multi-label segmentation
                tasks and `<metric name>_<class_name>` for single-label segmentation tasks.
        """
        metric_results = {}

        if self.multi_label:
            for _, confidence_level_name in self.confidence_levels:
                for metric_name, metric in self._metrics[confidence_level_name].items():
                    per_class_metrics = metric.compute()

                    for class_id, class_name in self.id_to_class_names.items():
                        metric_results[
                            f"{metric_name}_{class_name}_{confidence_level_name.replace(',','.')}"
                        ] = per_class_metrics[class_id]

        else:
            for metric_name, metric in self._metrics.items():
                per_class_metrics = metric.compute()

                for class_id, class_name in self.id_to_class_names.items():
                    if class_id != 0:
                        metric_results[
                            f"{metric_name}_{class_name}"
                        ] = per_class_metrics[class_id]

        return metric_results
