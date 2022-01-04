""" Module containing segmentation losses """
import abc
from typing import Literal, Tuple

from monai.losses.dice import (
    DiceLoss as MonaiDiceLoss,
    GeneralizedDiceLoss as MonaiGeneralizedDiceLoss,
)
import torch


class SegmentationLoss(torch.nn.Module, abc.ABC):
    r"""
    Base class for implementation of segmentation losses.

    Args:
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `False`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        include_background: bool = False,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):

        super().__init__()
        self.include_background = include_background
        if reduction and reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method.")
        self.reduction = reduction
        self.epsilon = epsilon

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        r"""
        Aggregates the loss values of the different classes of an image as well as the different images of a batch.

        Args:
            loss (Tensor): Loss to be aggregated.

        Returns:
            Aggregated loss value.

        Shape:
            - Loss: :math:`(N, C)`, where `N = batch size`, and `C = number of classes`, or `(N)` for binary
                segmentation tasks.
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(N, C)`. Otherwise, scalar.
        """

        assert loss.dim() == 1 or loss.dim() == 2

        # aggregate loss values for all class channels and the entire batch
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def _one_hot_encode(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
        r"""
        Comverts a label encoded tensor to a one-hot encoded tensor.

        Args:
            tensor (Tensor): Label encoded tensor that is to be converted to one-hot encoding.
            num_classes (int): Number of classes.

        Returns:
            Tensor: One-hot encoded tensor.

        Shape:
            - Tensor: :math:`(N, X, Y, ...)` where each element represents a class index of integer type and `N = batch
                size`.
            - Output: :math:`(N, C, X, Y, ...)` where each element represent a binary class label.
        """

        tensor_one_hot = torch.nn.functional.one_hot(tensor.long(), num_classes)

        # one_hot outputs a tensor of shape (N, X, Y, ..., C)
        # this tensor is converted to a tensor of shape (N, C, X, Y, ...)
        return tensor_one_hot.permute(
            (0, tensor_one_hot.ndim - 1, *range(1, tensor_one_hot.ndim - 1))
        )

    @staticmethod
    def _flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Flattens a tensor except for its first two dimensions (batch dimension and class dimension).

        Args:
            tensor (Tensor): The tensor to be flattened.
        Returns:
            Tensor: Flattened view of the input tensor.

        Shape:
            - Tensor: :math:`(N, C, X, Y, ...)` where `N = batch size`, and `C = number of classes`.
            - Output: :math:`(N, C, X*Y*...)`
        """

        return tensor.view(*tensor.shape[0:2], -1).float()

    def _flatten_tensors(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        r"""
        Reshapes and flattens prediction and target tensors except for the first two dimensions (batch dimension and
        class dimension).

        Args:
            prediction (Tensor): A prediction tensor to be flattened.
            target (Tensor): A target tensor to be flattened.
        Returns:
            Tuple[torch.Tensor]: Flattened prediction and target tensor.

        Shape:
            - Prediction: :math:`(N, C, X, Y, ...)` where `N = batch size`, and `C = number of classes`.
            - Target: :math:`(N, C, X, Y, ...)` in case of one-hot or multi-hot encoding or :math:`(N, X, Y, ...)` in
                case of label encoding.
            - Output: :math:`(N, C, X*Y*...)` for both tensors.
        """

        if prediction.ndim != target.ndim:
            # in this case the target tensor is label encoded and needs to be converted to a one-hot encoding
            target = self._one_hot_encode(target, prediction.shape[1])

        return self._flatten_tensor(prediction), self._flatten_tensor(target)


class DiceLoss(SegmentationLoss):
    r"""
    Implementation of the Dice loss for segmentation tasks. The Dice loss for binary segmentation tasks originally was
    formulated in:

        `Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for
        volumetric medical image segmentation, 2016 <https://arxiv.org/pdf/1606.04797.pdf>`_.

    In this implementation an adapted version of the Dice loss is used that a includes an epsilon term :math:`epsilon`to
    avoid divisions by zero and does not square the terms in the denominator. Additionally, the loss formulation is
    generalized to multi-class classification tasks by averaging dice losses over the class and batch dimension:

        :math:`DL = 1 - \frac{1}{N \cdot L} \cdot \sum_{n=1}^N \sum_{l=1}^L 2 \cdot \frac{\sum_{i} r_{nli} p_{nli} +
            \epsilon}{\sum_{n=1}^N \sum_{l=1}^L \sum_{i} (r_{nli} + p_{nli}) + \epsilon}` where :math:`N` is the batch
            size, :math:`L` is the number of classes, :math:`r_{nli}` are the ground-truth labels for class :math:`l` in
            the :math:`i`-th voxel of the :math:`n`-th image. Analogously, :math:`p` is the predicted probability for
            class :math:`l` in the :math:`i`-th voxel of the :math:`n`-th image.

    This implementation is a wrapper of the Dice loss implementation from the `MONAI package
    <https://docs.monai.io/en/stable/losses.html#diceloss>`_ that adapts the reduction behaviour. It supports both
    single-label and multi-label segmentation tasks. For a discussion on different dice loss implementations, see
    https://github.com/pytorch/pytorch/issues/1249.

    Args:
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `False`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        include_background: bool = False,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(epsilon=epsilon, reduction=reduction)
        self.dice_loss = MonaiDiceLoss(
            include_background=include_background,
            reduction=reduction,
            smooth_nr=epsilon,
            smooth_dr=epsilon,
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            prediction (Tensor): Predicted segmentation mask which is either the output of a softmax or a sigmoid layer.
            target (Tensor): Target segmentation mask which is either label encoded, one-hot encoded or multi-hot
                encoded.
        Returns:
            Tensor: Dice loss.

        Shape:
            - Prediction: :math:`(N, C, X, Y, ...)`, where `N = batch size`, `C = number of classes` and each value is
                in :math:`[0, 1]`
            - Target: :math:`(N, X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
                or :math:`(N, C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot
                encoding.
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(N, C)`. Otherwise, scalar.
        """

        assert prediction.shape == target.shape or prediction.dim() == target.dim() + 1

        if prediction.dim() != target.dim():
            target = self._one_hot_encode(target, prediction.shape[1])

        dice_loss = self.dice_loss(prediction, target)

        if self.reduction == "none":
            # the MONAI Dice loss implementation returns a loss tensor of shape `(N, C, X, Y, ...)` when reduction is
            # set to "none"
            # since the spatial dimensions only contain a single element, they are squeezed here
            dice_loss = dice_loss.reshape((dice_loss.shape[0], dice_loss.shape[1]))
        return dice_loss


class GeneralizedDiceLoss(SegmentationLoss):
    r"""
    Implementation of Generalized Dice loss for segmentation tasks. The Generalized Dice loss was formulated in:

        `Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M. Jorge Cardoso: Generalised Dice overlap
        as a deep learning loss function for highly unbalanced segmentations, 2017.
        <https://arxiv.org/pdf/1707.03237.pdf>`_

    It is formulated as:

        :math:`GDL = 1 - \frac{1}{N \cdot L} \cdot \sum_{n=1}^N 2 \cdot \sum_{l=1}^L w_l \cdot \frac{\sum_{i} r_{nli}
            p_{nli} + \epsilon}{\sum_{n=1}^N \sum_{l=1}^L w_l \cdot \sum_{i} (r_{nli} + p_{nli}) + \epsilon}` where
            :math:`N` is the batch size, :math:`L` is the number of classes, :math:`w_l` is a class weight,
            :math:`r_{nli}` are the ground-truth labels for class :math:`l` in the :math:`i`-th voxel of the
            :math:`n`-th image. Analogously, :math:`p` is the predicted probability for class :math:`l` in the
            :math:`i`-th voxel of the :math:`n`-th image.

    This implementation is a wrapper of the Generalized Dice loss implementation from the `MONAI package
    <https://docs.monai.io/en/stable/losses.html#generalizeddiceloss>`_ that adapts the reduction behaviour. It supports
    both single-label and multi-label segmentation tasks.

    Args:
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `False`).
        weight_type (string, optional): Type of function to transform ground truth volume to a weight factor:
            `"square"` | `"simple"` | `"uniform"`. Defaults to `"square"`.
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        include_background: bool = False,
        weight_type: Literal["square", "simple", "uniform"] = "square",
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(epsilon=epsilon, reduction=reduction)
        self.generalized_dice_loss = MonaiGeneralizedDiceLoss(
            include_background=include_background,
            w_type=weight_type,
            reduction=reduction,
            smooth_nr=epsilon,
            smooth_dr=epsilon,
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            prediction (Tensor): Predicted segmentation mask which is either the output of a softmax or a sigmoid layer.
            target (Tensor): Target segmentation mask which is either label encoded, one-hot encoded or multi-hot
            encoded.
        Returns:
            Tensor: Generalized Dice loss.

        Shape:
            - Prediction: :math:`(N, C, X, Y, ...)`, where `N = batch size`, `C = number of classes` and each value is
                in :math:`[0, 1]`.
            - Target: :math:`(N, X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
                and :math:`(N, C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot
                encoding.
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(N, C)`. Otherwise, scalar.
        """

        # ToDo: refactor duplicated code

        assert prediction.shape == target.shape or prediction.dim() == target.dim() + 1

        if prediction.dim() != target.dim():
            target = self._one_hot_encode(target, prediction.shape[1])

        dice_loss = self.generalized_dice_loss(prediction, target)

        if self.reduction == "none":
            # the MONAI Dice loss implementation returns a loss tensor of shape `(N, C, X, Y, ...)` when reduction is
            # set to "none"
            # since the spatial dimensions only contain a single element, they are squeezed here
            dice_loss = dice_loss.reshape((dice_loss.shape[0], dice_loss.shape[1]))
        return dice_loss


class FalsePositiveLoss(SegmentationLoss):
    """
    Implementation of false positive loss.

    Args:
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `False`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        include_background: bool = False,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(
            include_background=include_background, reduction=reduction, epsilon=epsilon
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            prediction (Tensor): Predicted segmentation mask which is either the output of a softmax or a sigmoid layer.
            target (Tensor): Target segmentation mask which is either label encoded, one-hot encoded or multi-hot
            encoded.
        Returns:
            Tensor: False positive loss.

        Shape:
            - Prediction: :math:`(N, C, X, Y, ...)`, where `N = batch size`, `C = number of classes` and each value is
                in :math:`[0, 1]`
            - Target: :math:`(N, X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
                and :math:`(N, C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot
                encoding.
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(N, C)`. Otherwise, scalar.
        """

        assert prediction.shape == target.shape or prediction.dim() == target.dim() + 1

        if not self.include_background:
            prediction = prediction[:, 1:]
            target = target[:, 1:]

        # flatten spatial dimensions
        flattened_prediction, flattened_target = self._flatten_tensors(
            prediction, target
        )

        false_positives = ((1 - flattened_target) * flattened_prediction).sum(-1)
        positives = flattened_prediction.sum(-1)

        # in contrast to the Dice loss, the epsilon term is only added to the denominator
        # if there are no positives at all, this will yield an optimal loss value of zero
        fp_loss = false_positives / (self.epsilon + positives)

        return self._reduce_loss(fp_loss)


class FalsePositiveDiceLoss(SegmentationLoss):
    """
    Implements a loss function that combines the Dice loss with the false positive loss.

    Args:
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `False`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        include_background: bool = False,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(reduction=reduction, epsilon=epsilon)
        self.fp_loss = FalsePositiveLoss(
            include_background=include_background, reduction=reduction, epsilon=epsilon
        )
        self.dice_loss = DiceLoss(
            include_background=include_background, reduction=reduction, epsilon=epsilon
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            prediction (Tensor): Predicted segmentation mask which is either the output of a softmax or a sigmoid layer.
            target (Tensor): Target segmentation mask which is either label encoded, one-hot encoded or multi-hot
                encoded.
        Returns:
            Tensor: Combined loss.

        Shape:
            - Prediction: :math:`(N, C, X, Y, ...)`, where `N = batch size`, `C = number of classes` and each value is
                in :math:`[0, 1]`
            - Target: :math:`(N, X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
                and :math:`(N, C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot
                encoding.
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(N, C)`. Otherwise, scalar.
        """

        return self.fp_loss(prediction, target) + self.dice_loss(prediction, target)


class CrossEntropyLoss(SegmentationLoss):
    """
    Wrapper for the PyTorch implementation of BCE loss / NLLLoss to ensure uniform reduction behaviour for all losses.

    Args:
        multi_label (bool, optional): Determines if data is multilabel or not (default = `False`).
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `False`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
    """

    def __init__(
        self,
        multi_label: bool = False,
        include_background: bool = False,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__(include_background=include_background, reduction=reduction)
        self.multi_label = multi_label
        if self.multi_label:
            self.cross_entropy_loss = torch.nn.BCELoss(reduction="none")
        else:
            self.cross_entropy_loss = torch.nn.NLLLoss(reduction="none")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            prediction (Tensor): Predicted segmentation mask which is either the output of a softmax or a sigmoid layer.
            target (Tensor): Target segmentation mask which is either label encoded, one-hot encoded or multi-hot
            encoded.
        Returns:
            Tensor: Cross-entropy loss.

        Shape:
            - Prediction: :math:`(N, C, X, Y, ...)`, where `N = batch size`, `C = number of classes` and each value is
                in :math:`[0, 1]`
            - Target: :math:`(N, X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
                and :math:`(N, C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot
                encoding.
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(N, C)`. Otherwise, scalar.
        """

        assert prediction.shape == target.shape or prediction.dim() == target.dim() + 1

        if not self.include_background:
            prediction = prediction[:, 1:]
            target = target[:, 1:]

        loss = self.cross_entropy_loss(prediction.float(), target.float())

        if self.reduction == "mean":
            # the images in one batch can have different sizes and thus the padding size can differ
            # in order to weight the loss term of each image equally regardless of its size, the loss tensor is averaged
            # over the spatial dimension (and the class dimension in case of multi-label segmentation tasks) before
            # passing it to the reduction function
            axis_to_reduce = range(1, loss.dim())
            loss = loss.mean(dim=axis_to_reduce)

        return self._reduce_loss(loss)


class BCEDiceLoss(SegmentationLoss):
    """
    Implements a loss function that combines the Dice loss with the binary cross-entropy (negative log-likelihood) loss.

    Args:
        epsilon (float, optional): Laplacian epsilon factor.
        reduction (str, optional): Reduction function that is to be used to aggregate the loss values of the images of
            one batch, must be either "mean", "sum" or "none".

    Args:
        multi_label (bool, optional): Determines if data is multilabel or not (default = `False`).
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `False`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        multi_label: bool = False,
        include_background: bool = False,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(
            include_background=include_background, reduction=reduction, epsilon=epsilon
        )
        self.bce_loss = CrossEntropyLoss(
            multi_label=multi_label,
            include_background=include_background,
            reduction=reduction,
        )
        self.dice_loss = DiceLoss(
            include_background=include_background, reduction=reduction, epsilon=epsilon
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            prediction (Tensor): Predicted segmentation mask which is either the output of a softmax or a sigmoid layer.
            target (Tensor): Target segmentation mask which is either label encoded, one-hot encoded or multi-hot
                encoded.
        Returns:
            Tensor: Combined loss.

        Shape:
            - Prediction: :math:`(N, C, X, Y, ...)`, where `N = batch size`, `C = number of classes` and each value is
                in :math:`[0, 1]`
            - Target: :math:`(N, X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
                and :math:`(N, C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot
                encoding.
            - Output: If :attr:`reduction` is `"none"`, shape :math:`(N, C)`. Otherwise, scalar.
        """

        return self.bce_loss(prediction, target) + self.dice_loss(prediction, target)
