"""Module to generate heat maps for trained model"""
import logging
from typing import List

import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.pytorch_model import PytorchModel


class SemanticSegmentationTarget:
    """
    Wraps the target for generating heat maps.
    All pixels belonging to the given prediction of the category will be summed up.
    More details can be found here: https://arxiv.org/abs/2002.11434
    Args:
        category (int): The category or class of the predicted mask.
        mask (np.ndarray): The predicted mask as numpy array.
    """

    # pylint:disable=too-few-public-methods
    def __init__(self, category: int, mask: np.ndarray):

        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output: np.ndarray) -> float:

        return (model_output[self.category, :, :] * self.mask).sum()


class GradCamHeatMap:
    """
    Generates heat maps for a trained model with respect to given layers.
    SegGradCam (Gradient Based Activation Methods) are used to calculate relevance values of individual pixels.
    Details can be found here: https://arxiv.org/abs/2002.11434
    Args:
        model (PytorchModel): A trained segmentation model.
        target_layers (List[torch.nn.Module]): A list of layers of the given model architecture.
            Used for calculating the gradient of the GradCam method.
    """

    @staticmethod
    def show_grayscale_cam_on_image(image: np.ndarray, grayscale_cam: np.ndarray):
        """
        Maps the calculated grayscale relevance values to a given image.
        Args:
            image (np.ndarray): The image to map the pixel relevance values against.
            grayscale_cam: The calculated relevance values in grayscale format.

        Returns:
            The input image in RGB format with mapped heatmap.
        """
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=2)
            logging.info("Expanded image to %s", image.shape)
        cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        return cam_image

    def __init__(
        self,
        model: PytorchModel,
        target_layers: List[torch.nn.Module],
    ):
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = torch.cuda.is_available()

    def generate_grayscale_cam(
        self,
        input_tensor: torch.Tensor,
        target_category: int,
    ) -> np.ndarray:
        """
        Calculates the grayscale representation of the relevance of individual pixel values using GradCam.
        Args:
            input_tensor (torch.Tensor): The tensor to calculate the desired values for.
            target_category (int): The category or class of the predicted mask.

        Returns:
            An array with the grayscale pixel importance values.
        """
        targets = self.__initialize_targets_for_gradcam(
            input_tensor=input_tensor, category=target_category
        )
        with GradCAM(
            model=self.model, target_layers=self.target_layers, use_cuda=self.use_cuda
        ) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        return grayscale_cam

    def __initialize_targets_for_gradcam(
        self,
        input_tensor: torch.Tensor,
        category: int,
    ) -> List[SemanticSegmentationTarget]:
        """
        Builds a list of targets wrapped up to be used when calculating pixel relevance values.
        Args:
            input_tensor (torch.Tensor): The tensor to calculate the desired values for.
            category (int): The category or class of the predicted mask.

        Returns:
            A list usable as targets for the GradCam calculation.
        """
        prediction = self.model.predict(input_tensor)
        segmentation = np.squeeze(prediction)
        segmentation = (segmentation >= 0.5) * 255
        targets = [
            SemanticSegmentationTarget(category=category, mask=np.float32(segmentation))
        ]
        return targets
