"""Clustering-based representativeness sampling strategy"""

from typing import Dict, List, Literal, Tuple

import numpy as np
from sklearn.cluster import MeanShift, KMeans

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .representativeness_sampling_strategy_base import (
    RepresentativenessSamplingStrategyBase,
)


class ClusteringBasedRepresentativenessSamplingStrategy(
    RepresentativenessSamplingStrategyBase
):
    """
    Representativeness sampling strategy that clusters the feature vectors and randomly selects items from the clusters
    least represented in the training set.

    Args:
        clustering_algorithm (string, optional): Clustering algorithm to be used: `"mean_shift"` | `"k_means"` |
            `"scans"`:
            - `"mean_shift"`: the mean shift clustering algorithm is used, allowing a variable number of clusters.
            - `"k_means"`: the k-means clustering algorithm is used, with a fixed number of clusters.
            -  `"scans"`: all slices from one scan are considered to represent one cluster.
            Defaults to `"mean_shift"`.
        feature_type (string, optional): Type of feature vectors to be used: `"model_features"` | `"image_features"`:
            - `"model_features"`: Feature vectors retrieved from the inner layers of the model are used.
            - `"image_features"`: The input images are used as feature vectors.
            Defaults to `model_features`.
        feature_dimensionality (int, optional): Number of dimensions the reduced feature vector should have.
            Defaults to 10.

        **kwargs: Optional keyword arguments:
            bandwidth (float, optional): Kernel bandwidth of the mean shift clustering algorithm. Defaults to 5. Only
                used if `clustering_algorithm = "mean_shift"`.
            cluster_all (bool, optional): Whether all data items including outliers should be assigned to a cluster.
                Defaults to `False`. Only used if `clustering_algorithm = "mean_shift"`.
            n_clusters (int, optional): Number of clusters. Defaults to 10.  Only used if
                `clustering_algorithm = "k_means"`.
            random_state (int, optional): Random state for centroid initialization of k-means algorithm. Defaults to
                `None`. Only used if `clustering_algorithm = "k_means"`.
    """

    def __init__(
        self,
        clustering_algorithm: Literal["mean_shift", "k_means", "scans"] = "mean_shift",
        feature_type: Literal["model_features", "image_features"] = "model_features",
        feature_dimensionality: int = 10,
        **kwargs,
    ):
        super().__init__(
            feature_type=feature_type
            if clustering_algorithm != "scans"
            else "image_features",
            feature_dimensionality=feature_dimensionality,
            **kwargs,
        )

        if clustering_algorithm not in ["mean_shift", "k_means", "scans"]:
            raise ValueError(f"Invalid clustering algorithm: {clustering_algorithm}.")

        self.clustering_algorithm = clustering_algorithm
        self.feature_dimensionality = feature_dimensionality

        if clustering_algorithm == "mean_shift":
            self.bandwidth = kwargs.get("bandwidth", 5)
            self.cluster_all = kwargs.get("cluster_all", False)
        elif clustering_algorithm == "k_means":
            self.n_clusters = kwargs.get("n_clusters", 10)
            self.random_state = kwargs.get("n_clusters", None)

        self.is_labeled = None
        self.cluster_ids = None
        self.cluster_labels = None

    def prepare_representativeness_computation(
        self,
        feature_vectors_training_set: np.ndarray,
        case_ids_training_set: List[str],
        feature_vectors_unlabeled_set: np.ndarray,
        case_ids_unlabeled_set: List[str],
    ) -> None:
        """
        Clusters the feature vectors.

        Args:
            feature_vectors_training_set (numpy.ndarray): Feature vectors of the items in the training set.
            case_ids_training_set (List[str]): Case IDs of the items in the training set.
            feature_vectors_unlabeled_set (numpy.ndarray): Feature vectors of the items in the unlabeled set.
            case_ids_unlabeled_set (List[str]): Case IDs of the items in the unlabeled set.
        """

        case_ids_training_set = np.array(case_ids_training_set)
        case_ids_unlabeled_set = np.array(case_ids_unlabeled_set)
        case_ids = np.concatenate([case_ids_training_set, case_ids_unlabeled_set])

        feature_vectors = np.concatenate(
            (feature_vectors_training_set, feature_vectors_unlabeled_set)
        )

        if feature_vectors.shape[1] > self.feature_dimensionality:
            feature_vectors = self.reduce_features(feature_vectors)

        if self.clustering_algorithm == "k_means":
            clustering = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state
            ).fit(feature_vectors)
            cluster_labels = clustering.labels_
        elif self.clustering_algorithm == "scans":
            scan_ids = ["-".join(case_id.split("-")[:-1]) for case_id in case_ids]

            unique_case_ids = np.unique(scan_ids)
            unique_case_ids.sort()

            scan_id_to_cluster_id = {
                scan_id: idx for idx, scan_id in enumerate(unique_case_ids)
            }

            cluster_labels = [
                scan_id_to_cluster_id[scan_id] for idx, scan_id in enumerate(scan_ids)
            ]
        else:
            clustering = MeanShift(
                bandwidth=self.bandwidth, cluster_all=self.cluster_all
            ).fit(feature_vectors)
            cluster_labels = clustering.labels_

        self.is_labeled = {
            case_id: case_id in case_ids_training_set for case_id in case_ids
        }
        self.cluster_ids = np.unique(cluster_labels)
        self.cluster_labels = {
            case_id: cluster_labels[idx] for idx, case_id in enumerate(case_ids)
        }

    def _compute_cluster_sizes(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Recomputes cluster sizes for current training set.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Cluster sizes for the training set, cluster sizes for the total
                dataset.
        """
        total_cluster_sizes = {}
        cluster_sizes_training_set = {}

        for cluster_id in self.cluster_ids:
            cluster_sizes_training_set[cluster_id] = 0
            total_cluster_sizes[cluster_id] = 0

        for case_id, cluster_label in self.cluster_labels.items():
            total_cluster_sizes[cluster_label] += 1
            if self.is_labeled[case_id]:
                cluster_sizes_training_set[cluster_label] += 1

        return (
            cluster_sizes_training_set,
            total_cluster_sizes,
        )

    def on_select_item(self, case_id: str) -> None:
        """
        Callback that is called when an item is selected for labeling.

        Args:
            case_id (string): Case ID of the selected item.
        """

        self.is_labeled[case_id] = True

    # pylint: disable = unused-argument
    def compute_representativeness_scores(
        self,
        model: PytorchModel,
        data_module: ActiveLearningDataModule,
        feature_vectors_training_set,
        feature_vectors_unlabeled_set,
        case_ids_unlabeled_set,
    ) -> List[float]:
        """
        Computes representativeness scores for all unlabeled items.

        Args:
            model (PytorchModel): Current model that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            feature_vectors_training_set (np.ndarray): Feature vectors of the items in the training set.
            feature_vectors_unlabeled_set (np.ndarray): Feature vectors of the items in the unlabeled set.
            case_ids_unlabeled_set (List[str]): Case IDs of the items in the unlabeled set.

        Returns:
            List[float]: Representativeness score for each item in the unlabeled set. Items that are underrepresented in
                the training receive higher scores.
        """

        relative_cluster_sizes_training_set = {}

        (
            cluster_sizes_training_set,
            total_cluster_sizes,
        ) = self._compute_cluster_sizes()

        for cluster_id, total_cluster_size in total_cluster_sizes.items():
            if cluster_id == -1 and not self.cluster_all:
                # set relative cluster size of outliers to 1 so that they are selected last
                relative_cluster_sizes_training_set[cluster_id] = 1
            else:
                relative_cluster_sizes_training_set[cluster_id] = (
                    cluster_sizes_training_set[cluster_id] / total_cluster_size
                )

        # pylint: disable=singleton-comparison
        representativeness_scores = [
            1 - relative_cluster_sizes_training_set[self.cluster_labels[case_id]]
            for case_id in case_ids_unlabeled_set
        ]

        return representativeness_scores
