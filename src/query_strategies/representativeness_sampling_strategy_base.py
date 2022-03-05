""" Base class for implementing representativeness sampling strategies """
from abc import ABC, abstractmethod
import math
from typing import List, Literal, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
import torch

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from models import PytorchUNet
from .query_strategy import QueryStrategy

# pylint: disable=too-few-public-methods
class RepresentativenessSamplingStrategyBase(QueryStrategy, ABC):
    """
    Base class for implementing representativeness sampling strategies

    Args:
        feature_dimensionality (int, optional): Number of dimensions the reduced feature vector should have.
            Defaults to 10.
        feature_type (string, optional): Type of feature vectors to be used: `"model_features"` | `"image_features"`:
            - `"model_features"`: Feature vectors retrieved from the inner layers of the model are used.
            - `"image_features"`: The input images are used as feature vectors.
            Defaults to `model_features`.
        feature_dimensionality (int, optional): Number of dimensions the reduced feature vector should have.
            Defaults to 10.
    """

    def __init__(
        self,
        feature_type: Literal["model_features", "image_features"] = "model_features",
        feature_dimensionality: int = 10,
        **kwargs,
    ):
        if feature_type not in ["model_features", "image_features"]:
            raise ValueError(f"Invalid feature type: {feature_type}.")

        self.feature_type = feature_type
        self.feature_dimensionality = feature_dimensionality

        if self.feature_type == "model_features":
            # variable for storing feature vectors in the forward hook
            self.feature_vector = torch.empty(1)

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.random_state = kwargs.get("random_state", None)

    # pylint: disable=unused-argument
    def __interception_hook(
        self, module: torch.nn.Module, inputs: torch.Tensor, output: torch.Tensor
    ) -> None:
        """
        Method to be registered as forward hook in the module from which the feature vectors are to be retrieved.

        Args:
            module (torch.nn.Module): Module in which the forward hook was registered.
            inputs (torch.Tensor): Inputs of the forward method in which the hook was registered.
            output (torch.Tensor): Outputs of the forward method of the module in which the hook was registered.
        """

        self.feature_vector = output.detach()
        self.feature_vector = self.feature_vector.view(
            self.feature_vector.shape[0], self.feature_vector.shape[1], -1
        )
        max_values, _ = self.feature_vector.max(dim=-1)
        self.feature_vector = max_values

    def _retrieve_model_feature_vectors(
        self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[List[np.array], List[str]]:
        """
        Retrieves a feature vector from the model's intermediate layers for each data item in the provided dataloader.
        The feature vectors are retrieved using the method `__interception_hook` that needs to be registered as forward
        hook before calling this method.

        Args:
            model (torch.nn.Module): A model.
            dataloader (torch.utils.data.DataLoader): A dataloader.

        Returns:
            Tuple[List[numpy.array], List[str]]: List of feature vectors and list of corresponding case IDs.
        """

        feature_vectors = []
        case_ids = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    images, ids = batch
                else:
                    images, _, ids = batch

                model.predict(images.to(self.device))

                case_ids.extend(ids)
                feature_vectors.extend(list(self.feature_vector.split(1)))

        feature_vectors = [
            feature_vector.flatten().cpu().numpy() for feature_vector in feature_vectors
        ]

        return feature_vectors, case_ids

    @staticmethod
    def _retrieve_image_feature_vectors(
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[List[np.array], List[str]]:
        """
        Retrieves images from dataloader and uses the images as feature vectors.

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader.

        Returns:
            Tuple[List[numpy.array], List[str]]: List of feature vectors and list of corresponding case IDs.
        """

        feature_vectors = []
        case_ids = []

        for batch in dataloader:
            if len(batch) == 2:
                images, ids = batch
            else:
                images, _, ids = batch

            case_ids.extend(ids)
            feature_vectors.extend(list(images.split(1)))

        feature_vectors = [
            feature_vector.flatten().cpu().numpy() for feature_vector in feature_vectors
        ]

        return feature_vectors, case_ids

    def reduce_features(
        self,
        feature_vectors: np.array,
        epsilon: float = 1e-10,
    ) -> np.array:
        """
        Reduces the dimensionality of feature vectors using a principle component analysis.

        Args:
            feature_vectors (numpy.array): Feature vectors to be reduced.
            epsilon (float, optional): Smoothing operator.

        Returns:
            numpy.array: Reduced feature vectors.
        """

        min_values = feature_vectors.min(axis=0, keepdims=True)
        max_values = feature_vectors.max(axis=0, keepdims=True)

        normalized_feature_vectors = (feature_vectors - min_values) / (
            max_values - min_values + epsilon
        )

        pca = PCA(n_components=self.feature_dimensionality).fit(
            normalized_feature_vectors
        )

        return pca.transform(feature_vectors)

    # pylint: disable=too-many-locals
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs,
    ) -> Tuple[List[str], None]:
        """
        Selects a subset of the unlabeled data that increases the representativeness of the training set.

        Args:
            models (PytorchModel): Current models that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            items_to_label (int): Number of items that should be selected for labeling.
            **kwargs: Additional, strategy-specific parameters.

        Returns:
            Tuple[List[str], None]: List of IDs of the data items to be labeled and None because no pseudo labels are
                generated.
        """

        if isinstance(models, List):
            raise ValueError(
                "Uncertainty sampling is only implemented for one model. You passed a list."
            )

        if self.feature_type == "model_features":

            if isinstance(models, PytorchUNet):
                interception_module = models.model.bottleneck
            else:
                raise ValueError(
                    "Representativeness sampling is not implemented for the provided model architecture."
                )

            models.to(self.device)
            interception_hook = interception_module.register_forward_hook(
                self.__interception_hook
            )

            (
                feature_vectors_training_set,
                case_ids_training_set,
            ) = self._retrieve_model_feature_vectors(
                models, data_module.train_dataloader()
            )
            (
                feature_vectors_unlabeled_set,
                case_ids_unlabeled_set,
            ) = self._retrieve_model_feature_vectors(
                models, data_module.unlabeled_dataloader()
            )
        else:
            (
                feature_vectors_training_set,
                case_ids_training_set,
            ) = self._retrieve_image_feature_vectors(data_module.train_dataloader())
            (
                feature_vectors_unlabeled_set,
                case_ids_unlabeled_set,
            ) = self._retrieve_image_feature_vectors(data_module.unlabeled_dataloader())

        all_feature_vectors = np.concatenate(
            [feature_vectors_training_set, feature_vectors_unlabeled_set]
        )

        max_size = np.max(
            [len(feature_vector) for feature_vector in all_feature_vectors]
        )

        all_feature_vectors_padded = -1 * np.ones((len(all_feature_vectors), max_size))

        for idx, feature_vector in enumerate(all_feature_vectors):
            pad_width = max(max_size - len(feature_vector), 0)
            pad_width_front = math.floor(pad_width / 2)
            pad_width_back = math.ceil(pad_width / 2)

            all_feature_vectors_padded[idx, :] = np.pad(
                feature_vector,
                pad_width=(pad_width_front, pad_width_back),
                constant_values=(-1, -1),
            )

        reduced_feature_vectors = self.reduce_features(all_feature_vectors_padded)
        feature_vectors_training_set = reduced_feature_vectors[
            : len(feature_vectors_training_set)
        ]
        feature_vectors_unlabeled_set = reduced_feature_vectors[
            len(feature_vectors_training_set) :
        ]

        selected_ids = []

        self.prepare_representativeness_computation(
            feature_vectors_training_set,
            case_ids_training_set,
            feature_vectors_unlabeled_set,
            case_ids_unlabeled_set,
        )

        for _ in range(min(items_to_label, len(feature_vectors_unlabeled_set))):
            representativeness_scores = self.compute_representativeness_scores(
                models,
                data_module,
                feature_vectors_training_set,
                feature_vectors_unlabeled_set,
                case_ids_unlabeled_set,
            )

            # sort unlabeled items by their representativeness score
            unlabeled_indices = np.arange(len(case_ids_unlabeled_set))
            representativeness_scores = list(
                zip(representativeness_scores, unlabeled_indices)
            )

            # shuffle list before sorting so that among items with the same score one is randomly selected
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(representativeness_scores)

            representativeness_scores.sort(key=lambda y: y[0], reverse=True)
            # select the sample with the highest representativeness score
            index_of_most_representative_sample = representativeness_scores[0][1]
            case_id_of_most_representative_sample = case_ids_unlabeled_set[
                index_of_most_representative_sample
            ]
            selected_ids.append(case_id_of_most_representative_sample)

            self.on_select_item(case_id_of_most_representative_sample)

            np.insert(
                feature_vectors_training_set,
                0,
                feature_vectors_unlabeled_set[index_of_most_representative_sample],
            )
            np.delete(
                feature_vectors_unlabeled_set,
                index_of_most_representative_sample,
                axis=0,
            )
            del case_ids_unlabeled_set[index_of_most_representative_sample]

        if self.feature_type == "model_features":
            interception_hook.remove()

        return selected_ids, None

    def on_select_item(self, case_id: str) -> None:
        """
        Callback that is called when an item is selected for labeling.

        Args:
            case_id (string): Case ID of the selected item.
        """

    def prepare_representativeness_computation(
        self,
        feature_vectors_training_set: np.ndarray,
        case_ids_training_set: List[str],
        feature_vectors_unlabeled_set: np.ndarray,
        case_ids_unlabeled_set: List[str],
    ) -> None:
        """
        Can be overridden in subclasses to perform global computations on all feature vectors before item selection
        starts.


        Args:
            feature_vectors_training_set (numpy.ndarray): Feature vectors of the items in the training set.
            case_ids_training_set (List[str]): Case IDs of the items in the training set.
            feature_vectors_unlabeled_set (numpy.ndarray): Feature vectors of the items in the unlabeled set.
            case_ids_unlabeled_set (List[str]): Case IDs of the items in the unlabeled set.
        """

    @abstractmethod
    def compute_representativeness_scores(
        self,
        model: PytorchModel,
        data_module: ActiveLearningDataModule,
        feature_vectors_training_set: np.ndarray,
        feature_vectors_unlabeled_set: np.ndarray,
        case_ids_unlabeled_set: List[str],
    ) -> List[float]:
        """
        Must be overridden in subclasses to compute the representativeness scores for the items in the unlabeled set.

        Args:
            model (PytorchModel): Current model that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            feature_vectors_training_set (np.ndarray): Feature vectors of the items in the training set.
            feature_vectors_unlabeled_set (np.ndarray): Feature vectors of the items in the unlabeled set.
            case_ids_unlabeled_set (List[str]): Case IDs of the items in the unlabeled set.

        Returns:
            List[float]: Representativeness score for each item in the unlabeled set. Items that are underrepresented in
                the training set should receive higher scores.
        """
