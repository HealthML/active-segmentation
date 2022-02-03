""" Module containing segmentation losses """
import abc
from typing import Literal, Optional, Tuple

from monai.losses.dice import (
    DiceLoss as MonaiDiceLoss,
    GeneralizedDiceLoss as MonaiGeneralizedDiceLoss,
)
import torch

from .utils import mask_tensor, one_hot_encode


class SegmentationLoss(torch.nn.Module, abc.ABC):
    r"""
    Base class for implementation of segmentation losses.

    Args:
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input
            gradient. Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):

        super().__init__()
        self.ignore_index = ignore_index
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

        # aggregate loss values for all class channels and the entire batch
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

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

        return tensor.contiguous().view(*tensor.shape[0:2], -1).float()

    def _preprocess_inputs(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        This method implements preprocessing steps that are needed for most segmentation losses:

        1. Conversion from label encoding to one-hot encoding if necessary
        2. Mapping of pixels/voxels labeled with the :attr:`ignore_index` to true negatives

        Args:
            prediction (Tensor): The prediction tensor to be preprocessed.
            target (Tensor): The target tensor to be preprocessed.

        Returns:
            Tuple[Tensor, Tensor]: The preprocessed prediction and target tensors.

        Shape:
            - Prediction: :math:`(N, C, X, Y, ...)`, where `N = batch size`, `C = number of classes` and each value is
                in :math:`[0, 1]`
            - Target: :math:`(N, X, Y, ...)` where each value is in :math:`\{0, ..., C - 1\}` in case of label encoding
                or :math:`(N, C, X, Y, ...)`, where each value is in :math:`\{0, 1\}` in case of one-hot or multi-hot
                encoding.
            - Output: :math:`(N, C, X, Y, ...)` for both prediction and target.
        """

        # during one-hot encoding the ignore index is removed, therefore the original target including the ignore index
        # is copied
        target_including_ignore_index = target
        target = target.clone()

        if prediction.dim() != target.dim():
            assert prediction.dim() == target.dim() + 1
            target = one_hot_encode(
                target,
                target.dim() - 1,
                prediction.shape[1],
                ignore_index=self.ignore_index,
            )
            target_including_ignore_index = target_including_ignore_index.unsqueeze(
                dim=1
            )

        prediction = mask_tensor(
            prediction, target_including_ignore_index, self.ignore_index
        )
        target = mask_tensor(target, target_including_ignore_index, self.ignore_index)

        return prediction, target


class AbstractDiceLoss(SegmentationLoss, abc.ABC):
    r"""
    Base class for implementation of Dice loss and Generalized Dice loss.

    Args:
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input
            gradient. Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(
            epsilon=epsilon,
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
        )

    @abc.abstractmethod
    def get_dice_loss_module(self) -> torch.nn.Module:
        """
        Returns:
            Module: Dice loss module.
        """

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

        prediction, target = self._preprocess_inputs(prediction, target)

        dice_loss_module = self.get_dice_loss_module()
        dice_loss = dice_loss_module(prediction, target)

        return dice_loss


class DiceLoss(AbstractDiceLoss):
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
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input
            gradient. Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(
            epsilon=epsilon, ignore_index=ignore_index, reduction=reduction
        )
        self.dice_loss = MonaiDiceLoss(
            include_background=include_background,
            reduction=reduction,
            smooth_nr=epsilon,
            smooth_dr=epsilon,
        )

    def get_dice_loss_module(self) -> torch.nn.Module:
        """
        Returns:
            Module: Dice loss module.
        """

        return self.dice_loss

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = super().forward(prediction, target)
        if self.reduction == "none":
            # the MONAI Dice loss implementation returns a loss tensor of shape `(N, C, X, Y, ...)` when reduction is
            # set to "none"
            # since the spatial dimensions only contain a single element, they are squeezed here
            dice_loss = dice_loss.reshape((dice_loss.shape[0], dice_loss.shape[1]))

        return dice_loss


class GeneralizedDiceLoss(AbstractDiceLoss):
    r"""
    Implementation of Generalized Dice loss for segmentation tasks. The Generalized Dice loss was formulated in:

        `Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M. Jorge Cardoso: Generalised Dice overlap
        as a deep learning loss function for highly unbalanced segmentations, 2017.
        <https://arxiv.org/pdf/1707.03237.pdf>`_

    It is formulated as:

        :math:`GDL = \frac{1}{N} \cdot \sum_{n=1}^N (1 - 2 \frac{\sum_{l=1}^L w_l \cdot \sum_{i} r_{nli} p_{nli} +
            \epsilon}{\sum_{l=1}^L w_l \cdot \sum_{i} (r_{nli} + p_{nli}) + \epsilon})` where
            :math:`N` is the batch size, :math:`L` is the number of classes, :math:`w_l` is a class weight,
            :math:`r_{nli}` are the ground-truth labels for class :math:`l` in the :math:`i`-th voxel of the
            :math:`n`-th image. Analogously, :math:`p` is the predicted probability for class :math:`l` in the
            :math:`i`-th voxel of the :math:`n`-th image.

    This implementation is a wrapper of the Generalized Dice loss implementation from the `MONAI package
    <https://docs.monai.io/en/stable/losses.html#generalizeddiceloss>`_ that adapts the reduction behaviour. It supports
    both single-label and multi-label segmentation tasks.

    Args:
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input
            gradient. Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        weight_type (string, optional): Type of function to transform ground truth volume to a weight factor:
            `"square"` | `"simple"` | `"uniform"`. Defaults to `"square"`.
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        weight_type: Literal["square", "simple", "uniform"] = "square",
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(
            epsilon=epsilon, ignore_index=ignore_index, reduction=reduction
        )
        self.generalized_dice_loss = MonaiGeneralizedDiceLoss(
            include_background=include_background,
            w_type=weight_type,
            reduction=reduction,
            smooth_nr=epsilon,
            smooth_dr=epsilon,
        )

    def get_dice_loss_module(self) -> torch.nn.Module:
        """
        Returns:
            Module: Dice loss module.
        """

        return self.generalized_dice_loss

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = super().forward(prediction, target)
        if self.reduction == "none":
            # the MONAI Dice loss implementation returns a loss tensor of shape `(N, C, X, Y, ...)` when reduction is
            # set to "none"
            # since the class dimension and the spatial dimensions only contain a single element, they are squeezed here
            dice_loss = dice_loss.reshape(dice_loss.shape[0])

        return dice_loss


class FalsePositiveLoss(SegmentationLoss):
    """
    Implementation of false positive loss.

    Args:
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input
            gradient. Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
            epsilon=epsilon,
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
        assert prediction.isfinite().all() and target.isfinite().all()

        prediction, target = self._preprocess_inputs(prediction, target)

        if not self.include_background:
            prediction = prediction[:, 1:]
            target = target[:, 1:]

        # flatten spatial dimensions
        flattened_prediction = self._flatten_tensor(prediction)
        flattened_target = self._flatten_tensor(target)

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
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input
            gradient. Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            calculation (default = `True`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(reduction=reduction, epsilon=epsilon)
        self.fp_loss = FalsePositiveLoss(
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
            epsilon=epsilon,
        )
        self.dice_loss = DiceLoss(
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
            epsilon=epsilon,
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
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input
            gradient. Defaults to `None`.
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        multi_label: bool = False,
        ignore_index: Optional[int] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(
            ignore_index=ignore_index,
            include_background=True,
            reduction=reduction,
            epsilon=epsilon,
        )
        self.multi_label = multi_label
        if self.multi_label:
            self.cross_entropy_loss = torch.nn.BCELoss(reduction="none")
        else:
            self.cross_entropy_loss = torch.nn.NLLLoss(
                ignore_index=ignore_index if ignore_index is not None else -100,
                reduction="none",
            )

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
        assert prediction.isfinite().all() and target.isfinite().all()

        if self.multi_label:
            target = target.float()
        else:
            # the Pytorch NLLLoss expect the inputs to be the output of a LogSoftmax layer
            # since this loss expects the output of a Softmax layer as input, the log is taken here

            prediction = torch.log(prediction + self.epsilon)
            target = target.long()

        loss = self.cross_entropy_loss(prediction, target)

        if self.multi_label and self.ignore_index is not None:
            # the BCELoss from Pytorch does not provide an `ignore_index` argument
            # therefore the loss values for the voxels to be ignored have to be set to zero here
            loss = mask_tensor(loss, target, self.ignore_index)

        if self.reduction == "mean":
            # the images in one batch can have different sizes and thus the padding size can differ
            # in order to weight the loss term of each image equally regardless of its size, the loss tensor is averaged
            # over the spatial dimension (and the class dimension in case of multi-label segmentation tasks) before
            # passing it to the reduction function
            axis_to_reduce = tuple(range(1, loss.dim()))
            loss = loss.mean(dim=axis_to_reduce)

        return self._reduce_loss(loss)


class CrossEntropyDiceLoss(SegmentationLoss):
    """
    Implements a loss function that combines the Dice loss with the binary cross-entropy (negative log-likelihood) loss.

    Args:
        multi_label (bool, optional): Determines if data is multilabel or not (default = `False`).
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input
            gradient. Defaults to `None`.
        include_background (bool, optional): if `False`, class channel index 0 (background class) is excluded from the
            Dice loss calculation, but not from the Cross-entropy loss calculation (default = `True`).
        reduction (str, optional): Specifies the reduction to aggregate the loss values over the images of a batch and
            multiple classes: `"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean
            of the output is taken, `"sum"`: the output will be summed (default = `"mean"`).
        epsilon (float, optional): Laplacian smoothing term to avoid divisions by zero (default = `1e-5`).
    """

    def __init__(
        self,
        multi_label: bool = False,
        ignore_index: Optional[int] = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-5,
    ):
        super().__init__(
            include_background=include_background, reduction=reduction, epsilon=epsilon
        )
        self.cross_entropy_loss = CrossEntropyLoss(
            multi_label=multi_label,
            ignore_index=ignore_index,
            reduction=reduction,
            epsilon=epsilon,
        )
        self.dice_loss = DiceLoss(
            ignore_index=ignore_index,
            include_background=include_background,
            reduction=reduction,
            epsilon=epsilon,
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

        return self.cross_entropy_loss(prediction, target) + self.dice_loss(
            prediction, target
        )
