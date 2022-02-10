"""Module to generate heat maps for trained model"""
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


class HeatMaps:
    """
    Generates heat maps for a trained model with respect to given layers.
    SegGradCam (Gradient Based Activation Methods) are used to calculate relevance values of individual pixels.
    Details can be found here: https://arxiv.org/abs/2002.11434
    Args:
        model (PytorchModel): A trained segmentation model.
    """

    @staticmethod
    def show_grayscale_heatmap_on_image(
        image: np.ndarray, grayscale_heatmap: np.ndarray
    ):
        """
        Maps the calculated grayscale relevance values to a given image.
        Args:
            image (np.ndarray): The image to map the pixel relevance values against.
            grayscale_heatmap (np.ndarray): The calculated relevance values in grayscale format.

        Returns:
            The input image in RGB format with mapped heatmap.
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        # Clip negative values if image was padded
        image = np.clip(image, 0, 1)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        elif image.shape[0] == 1:
            image = np.moveaxis(image, 0, 2)
        cam_image = show_cam_on_image(image, grayscale_heatmap, use_rgb=True)
        return cam_image

    def __init__(self, model: PytorchModel):
        self.model = model
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model = self.model.cuda()

    def generate_grayscale_cam(
        self,
        input_tensor: torch.Tensor,
        target_category: int,
        target_layers: List[torch.nn.Module],
    ) -> np.ndarray:
        """
        Calculates the grayscale representation of the relevance of individual pixel values using GradCam.
        Args:
            input_tensor (torch.Tensor): The tensor to calculate the desired values for.
            target_category (int): The category or class of the predicted mask.
            target_layers (List[torch.nn.Module]): A list of layers of the given model architecture.
                Used for calculating the gradient of the GradCam method.

        Returns:
            An array with the grayscale pixel importance values.
        """
        input_tensor = self.__prepare_tensor_for_model(input_tensor=input_tensor)
        targets = self.__initialize_targets_for_gradcam(
            input_tensor=input_tensor, category=target_category
        )
        with GradCAM(
            model=self.model, target_layers=target_layers, use_cuda=self.use_cuda
        ) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        return grayscale_cam

    def generate_grayscale_logits(
        self,
        input_tensor: torch.Tensor,
        target_category: int,
    ) -> np.ndarray:
        """
        Calculates the grayscale representation of the relevance of individual pixel values using the model prediction.
        Args:
            input_tensor (torch.Tensor): The tensor to calculate the desired values for.
            target_category (int): The category or class of the predicted mask.

        Returns:
            An array with the grayscale pixel importance values.
        """
        input_tensor = self.__prepare_tensor_for_model(input_tensor=input_tensor)
        prediction = self.model.predict(input_tensor)
        if self.use_cuda:
            prediction = prediction.cpu()
        segmentation = np.squeeze(prediction.numpy())[target_category]
        return segmentation

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
        if self.use_cuda:
            prediction = prediction.cpu()
        segmentation = np.squeeze(prediction.numpy())
        segmentation = (segmentation >= 0.5) * 255
        segmentation = np.float32(segmentation)
        targets = [SemanticSegmentationTarget(category=category, mask=segmentation)]
        return targets

    def __prepare_tensor_for_model(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Brings the tensor in correct shape to be used by the model
        Args:
            input_tensor (torch.Tensor): The tensor to use for the model later on.

        Returns:
            The input tensor in correct shape for the model.
        """
        if len(input_tensor.size()) == 2:
            input_tensor = input_tensor.unsqueeze(0)
        if len(input_tensor.size()) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        return input_tensor
