""" Module for representativeness sampling strategy """
import math
from typing import List, Literal, Tuple, Union

import numpy as np
import psutil
import scipy.spatial
import torch

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from models import PytorchUNet
from .query_strategy import QueryStrategy

# pylint: disable=too-few-public-methods
class RepresentativenessSamplingStrategy(QueryStrategy):
    """
    Class for selecting items via a representativeness sampling strategy

    Args:
        distance_metric (string, optional): Metric to be used for calculation the distance between feature vectors.
    """

    def __init__(
        self,
        distance_metric: Literal["euclidean", "cosine", "russellrao"] = "euclidean",
    ):
        if distance_metric not in ["euclidean", "cosine", "russellrao"]:
            raise ValueError(f"Invalid distance metric: {distance_metric}.")

        self.distance_metric = distance_metric

        # variable for storing feature vectors in the forward hook
        self.feature_vector = torch.empty(1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def _retrieve_feature_vectors(
        self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.array, List[str]]:
        """
        Retrieves a feature vector from the model's intermediate layers for each data item in the provided dataloader.
        The feature vectors are retrieved using the method `__interception_hook` that needs to be registered as forward
        hook before calling this method.

        Args:
            model (torch.nn.Module): A model.
            dataloader (torch.utils.data.DataLoader): A dataloader.

        Returns:
            Tuple[numpy.array, List[str]]: List of feature vectors and list of corresponding case IDs.
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

        feature_vectors = np.array(
            [
                feature_vector.flatten().cpu().numpy()
                for feature_vector in feature_vectors
            ]
        )

        return feature_vectors, case_ids

    def _feature_distances(
        self,
        feature_vectors_training_set: np.array,
        feature_vectors_unlabeled_set: np.array,
    ) -> np.array:
        """
        Computes average distances between the feature vectors from the unlabeled set and the feature vectors from the
        training set.

        Args:
            feature_vectors_training_set (numpy.array): Feature vectors from the training set.
            feature_vectors_unlabeled_set (numpy.array): Feature vectors from the unlabeled set.

        Returns:
            np.array: For each feature vector from the unlabeled set, average distance to the feature vectors from the
                training set
        """

        # as the feature vectors possibly might be large, the feature vectors from the unlabeled set are split into
        # chunks so that one chunk of feature vectors from the unlabeled set and all feature vectors from the training
        # set fit into memory

        free_memory = psutil.virtual_memory().available

        memory_consumption_feature_vectors_training_set = np.zeros(
            len(feature_vectors_training_set)
        ).nbytes

        split_size = math.floor(
            max(
                math.floor(
                    free_memory / memory_consumption_feature_vectors_training_set
                )
                - 1,
                1,
            )
        )

        n_splits = math.ceil(len(feature_vectors_unlabeled_set) / split_size)

        feature_vectors_unlabeled_set_split = np.array_split(
            feature_vectors_unlabeled_set, n_splits
        )

        average_feature_distances = np.zeros(len(feature_vectors_unlabeled_set))

        for idx, current_chunk_feature_vectors_unlabeled_set in enumerate(
            feature_vectors_unlabeled_set_split
        ):
            feature_distances = scipy.spatial.distance.cdist(
                current_chunk_feature_vectors_unlabeled_set,
                feature_vectors_training_set,
                self.distance_metric,
            )

            average_distances_for_current_chunk = feature_distances.mean(axis=1)
            split_size = len(current_chunk_feature_vectors_unlabeled_set)
            start_index = idx * split_size
            end_index = (idx + 1) * split_size
            average_feature_distances[
                start_index:end_index
            ] = average_distances_for_current_chunk

        return average_feature_distances

    # pylint: disable=too-many-locals
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        data_module: ActiveLearningDataModule,
        items_to_label: int,
        **kwargs,
    ) -> List[str]:
        """
        Selects a subset of the unlabeled data that increases the representativeness of the training set.

        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            items_to_label (int): Number of items that should be selected for labeling.
            **kwargs: Additional, strategy-specific parameters.

        Returns:
            IDs of the data items to be labeled.
        """

        if isinstance(models, List):
            raise ValueError(
                "Representativeness sampling is only implemented for one model. You passed multiple."
            )

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

        feature_vectors_training_set, _ = self._retrieve_feature_vectors(
            models, data_module.train_dataloader()
        )
        feature_vectors_unlabeled_set, case_ids = self._retrieve_feature_vectors(
            models, data_module.unlabeled_dataloader()
        )

        selected_ids = []

        for _ in range(items_to_label):
            if len(feature_vectors_unlabeled_set) > 0:
                average_feature_distances = self._feature_distances(
                    feature_vectors_training_set, feature_vectors_unlabeled_set
                )

                # sort unlabeled items by their average feature distance to the labeled items in the training set
                unlabeled_indices = np.arange(len(case_ids))
                average_feature_distances = list(
                    zip(average_feature_distances, unlabeled_indices)
                )
                average_feature_distances.sort(key=lambda y: y[0], reverse=True)
                # select the sample with the highest average distance to the training set
                _, index_of_most_distant_sample = average_feature_distances[0]
                selected_ids.append(case_ids[index_of_most_distant_sample])

                np.insert(
                    feature_vectors_training_set,
                    0,
                    feature_vectors_unlabeled_set[index_of_most_distant_sample],
                )
                np.delete(
                    feature_vectors_unlabeled_set, index_of_most_distant_sample, axis=0
                )
                del case_ids[index_of_most_distant_sample]

        # free memory
        del feature_vectors_unlabeled_set
        del feature_vectors_training_set
        del average_feature_distances

        interception_hook.remove()

        return selected_ids