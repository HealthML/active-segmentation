""" Module for representativeness sampling strategy """
import logging
import math
from typing import List, Literal, Tuple, Union

import numpy as np
import psutil
import scipy.spatial
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
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
        algorithm (string, optional): The algorithm to be used to select the most representative samples:
            `"most_distant_sample"` | `"cluster_coverage"`. Defaults to `"cluster_coverage"`.
                - `"most_distant_sample"`: The unlabeled item that has the highest feature distance to the labeled set
                    is selected for labeling.
                - `"cluster_coverage"`: The features of the unlabeled and labeled items are clustered and an item from
                    the most underrepresented cluster is selected for labeling.
    """

    def __init__(
        self,
        distance_metric: Literal["euclidean", "cosine", "russellrao"] = "euclidean",
        algorithm: Literal[
            "most_distant_sample", "cluster_coverage"
        ] = "cluster_coverage",
    ):
        if distance_metric not in ["euclidean", "cosine", "russellrao"]:
            raise ValueError(f"Invalid distance metric: {distance_metric}.")

        self.distance_metric = distance_metric

        if algorithm not in ["most_distant_sample", "cluster_coverage"]:
            raise ValueError(f"Invalid algorithm: {algorithm}.")

        self.algorithm = algorithm

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
            Tuple[List[str], None]: List of IDs of the data items to be labeled and None because no pseudo labels are
                generated.
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

        if self.algorithm == "most_distant_sample":
            selected_ids = self.select_most_distant_samples(
                feature_vectors_training_set,
                feature_vectors_unlabeled_set,
                case_ids,
                items_to_label,
            )
        else:
            selected_ids = self.select_items_with_best_cluster_coverage(
                feature_vectors_training_set,
                feature_vectors_unlabeled_set,
                case_ids,
                items_to_label,
            )

        # free memory
        del feature_vectors_unlabeled_set
        del feature_vectors_training_set

        interception_hook.remove()

        return selected_ids

    # pylint: disable=too-many-locals
    def select_most_distant_samples(
        self,
        feature_vectors_training_set: np.ndarray,
        feature_vectors_unlabeled_set: np.ndarray,
        case_ids: List[str],
        items_to_label: int,
    ) -> List[str]:
        """
        Selects a subset of the unlabeled data that increases the representativeness of the training set.

        Args:
            feature_vectors_training_set (numpy.ndarray): Feature vectors of the items in the training set.
            feature_vectors_unlabeled_set (numpy.ndarray): Feature vectors of the items in the unlabeled set.
            case_ids (List[str]): Case Ids of the unlabeled items.
            items_to_label (int): Number of items that should be selected for labeling.

        Returns:
            IDs of the data items to be labeled.
        """

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
        del average_feature_distances

        return selected_ids

    @staticmethod
    def reduce_features(
        feature_vectors: np.array, target_dimensionality: int = 10
    ) -> np.array:
        """
        Reduces the dimensionality of feature vectors using a principle component analysis.

        Args:
            feature_vectors (numpy.array): Feature vectors to be reduced.
            target_dimensionality (int, optional): Number of dimensions the reduced feature vector should have.
                Defaults to 10.

        Returns:
            numpy.array: Reduced feature vectors.
        """

        min_values = feature_vectors.min(axis=0, keepdims=True)
        max_values = feature_vectors.max(axis=0, keepdims=True)

        normalized_feature_vectors = (feature_vectors - min_values) / (
            max_values - min_values
        )

        pca = PCA(n_components=target_dimensionality).fit(normalized_feature_vectors)

        return pca.transform(feature_vectors)

    def select_items_with_best_cluster_coverage(
        self,
        feature_vectors_training_set: np.ndarray,
        feature_vectors_unlabeled_set: np.ndarray,
        case_ids: List[str],
        items_to_label: int,
    ) -> List[str]:
        """
        Selects a subset of the unlabeled data that increases the representativeness of the training set.

        Args:
            feature_vectors_training_set (numpy.ndarray): Feature vectors of the items in the training set.
            feature_vectors_unlabeled_set (numpy.ndarray): Feature vectors of the items in the unlabeled set.
            case_ids (List[str]): Case Ids of the unlabeled items.
            items_to_label (int): Number of items that should be selected for labeling.

        Returns:
            IDs of the data items to be labeled.
        """

        case_ids = np.array(case_ids)

        selected_ids = []

        training_set_size = len(feature_vectors_training_set)
        unlabeled_set_size = len(feature_vectors_unlabeled_set)

        feature_vectors = np.concatenate(
            (feature_vectors_training_set, feature_vectors_unlabeled_set)
        )

        reduced_feature_vectors = self.reduce_features(feature_vectors)

        clustering = MeanShift(bandwidth=5).fit(reduced_feature_vectors)

        cluster_labels_training_set = clustering.labels_[:training_set_size]
        cluster_labels_unlabeled_set = clustering.labels_[training_set_size:]

        cluster_ids, cluster_sizes = np.unique(clustering.labels_, return_counts=True)
        cluster_sizes_total = dict(zip(cluster_ids, cluster_sizes))

        logging.info("Sizes of current feature clusters: %s", cluster_sizes_total)

        cluster_ids_training_set, cluster_sizes_training_set = np.unique(
            cluster_labels_training_set, return_counts=True
        )
        cluster_sizes_training_set = dict(
            zip(cluster_ids_training_set, cluster_sizes_training_set)
        )

        cluster_ids_unlabeled_set, cluster_sizes_unlabeled_set = np.unique(
            cluster_labels_unlabeled_set, return_counts=True
        )
        cluster_sizes_unlabeled_set = dict(
            zip(cluster_ids_unlabeled_set, cluster_sizes_unlabeled_set)
        )

        is_selected = np.zeros_like(cluster_labels_unlabeled_set, dtype=np.bool)

        for _ in range(min(items_to_label, unlabeled_set_size)):

            relative_cluster_sizes_training_set = {}

            for cluster_id, total_cluster_size in cluster_sizes_total.items():
                if cluster_id in cluster_sizes_training_set:
                    relative_cluster_sizes_training_set[cluster_id] = (
                        cluster_sizes_training_set[cluster_id] / total_cluster_size
                    )
                else:
                    relative_cluster_sizes_training_set[cluster_id] = 0

            relative_cluster_sizes = list(relative_cluster_sizes_training_set.items())

            relative_cluster_sizes.sort(key=lambda y: y[1])

            selected_cluster_id = relative_cluster_sizes[0][0]

            # pylint: disable=singleton-comparison
            case_ids_belonging_to_selected_cluster = case_ids[
                (cluster_labels_unlabeled_set == selected_cluster_id)
                & (is_selected == False)
            ]

            assert len(case_ids_belonging_to_selected_cluster) > 0

            selected_case_id = np.random.choice(case_ids_belonging_to_selected_cluster)

            is_selected[case_ids == selected_case_id] = True
            if selected_cluster_id not in cluster_sizes_training_set:
                cluster_sizes_training_set[selected_cluster_id] = 1
            else:
                cluster_sizes_training_set[selected_cluster_id] += 1
            cluster_sizes_unlabeled_set[selected_cluster_id] -= 1

            selected_ids.append(selected_case_id)

        return selected_ids, None
