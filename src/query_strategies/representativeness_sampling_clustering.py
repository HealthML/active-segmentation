"""Clustering-based representativeness sampling strategy"""

import logging
from typing import List, Literal

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
        clustering_algorithm (string, optional): Clustering algorithm to be used: `"mean_shift"` | `"k_means"`. Defaults
            to `"mean_shift"`.
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
        clustering_algorithm: Literal["mean_shift", "k_means"] = "mean_shift",
        feature_type: Literal["model_features", "image_features"] = "model_features",
        feature_dimensionality: int = 10,
        **kwargs,
    ):
        super().__init__(
            feature_type=feature_type, feature_dimensionality=feature_dimensionality
        )

        if clustering_algorithm not in ["mean_shift", "k_means"]:
            raise ValueError(f"Invalid clustering algorithm: {clustering_algorithm}.")

        self.clustering_algorithm = clustering_algorithm
        self.feature_dimensionality = feature_dimensionality

        if clustering_algorithm == "mean_shift":
            self.bandwidth = kwargs.get("bandwidth", 5)
            self.cluster_all = kwargs.get("cluster_all", False)
        elif clustering_algorithm == "k_means":
            self.n_clusters = kwargs.get("n_clusters", 10)
            self.random_state = kwargs.get("n_clusters", None)

        self.is_selected = None
        self.cluster_sizes_total = {}
        self.cluster_sizes_training_set = {}
        self.cluster_sizes_unlabeled_set = {}
        self.cluster_labels_training_set = None
        self.cluster_labels_unlabeled_set = None

    def prepare_representativeness_computation(
        self, feature_vectors_training_set, feature_vectors_unlabeled_set
    ) -> None:
        """
        Clusters the feature vectors using mean shift algorithm.

        Args:
            feature_vectors_training_set (np.array): Feature vectors of the items in the training set.
            feature_vectors_unlabeled_set (np.array): Feature vectors of the items in the unlabeled set.
        """

        training_set_size = len(feature_vectors_training_set)

        feature_vectors = np.concatenate(
            (feature_vectors_training_set, feature_vectors_unlabeled_set)
        )

        if feature_vectors.shape[1] > self.feature_dimensionality:
            feature_vectors = self.reduce_features(feature_vectors)

        if self.clustering_algorithm == "k_means":
            clustering = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state
            ).fit(feature_vectors)

        else:
            clustering = MeanShift(
                bandwidth=self.bandwidth, cluster_all=self.cluster_all
            ).fit(feature_vectors)

        self.cluster_labels_training_set = clustering.labels_[:training_set_size]
        self.cluster_labels_unlabeled_set = clustering.labels_[training_set_size:]

        cluster_ids, cluster_sizes = np.unique(clustering.labels_, return_counts=True)
        self.cluster_sizes_total = dict(zip(cluster_ids, cluster_sizes))

        logging.info("Sizes of current feature clusters: %s", self.cluster_sizes_total)

        cluster_ids_training_set, cluster_sizes_training_set = np.unique(
            self.cluster_labels_training_set, return_counts=True
        )
        self.cluster_sizes_training_set = dict(
            zip(cluster_ids_training_set, cluster_sizes_training_set)
        )

        cluster_ids_unlabeled_set, cluster_sizes_unlabeled_set = np.unique(
            self.cluster_labels_unlabeled_set, return_counts=True
        )
        self.cluster_sizes_unlabeled_set = dict(
            zip(cluster_ids_unlabeled_set, cluster_sizes_unlabeled_set)
        )

        self.is_selected = np.zeros_like(
            self.cluster_labels_unlabeled_set, dtype=np.bool
        )

    # pylint: disable = unused-argument
    def compute_representativeness_scores(
        self,
        model: PytorchModel,
        data_module: ActiveLearningDataModule,
        feature_vectors_training_set,
        feature_vectors_unlabeled_set,
    ) -> List[float]:
        """
        Computes representativeness scores for all unlabeled items.

        Args:
            model (PytorchModel): Current model that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            feature_vectors_training_set (np.ndarray): Feature vectors of the items in the training set.
            feature_vectors_unlabeled_set (np.ndarray): Feature vectors of the items in the unlabeled set.

        Returns:
            List[float]: Representativeness score for each item in the unlabeled set. Items that are underrepresented in
                the training receive higher scores.
        """

        relative_cluster_sizes_training_set = {}

        for cluster_id, total_cluster_size in self.cluster_sizes_total.items():
            if cluster_id == -1 and not self.cluster_all:
                # set relative cluster size of outliers to 1 so that they are selected last
                relative_cluster_sizes_training_set[cluster_id] = 1
            else:
                if cluster_id in self.cluster_sizes_training_set:
                    relative_cluster_sizes_training_set[cluster_id] = (
                        self.cluster_sizes_training_set[cluster_id] / total_cluster_size
                    )
                else:
                    relative_cluster_sizes_training_set[cluster_id] = 0

        # pylint: disable=singleton-comparison
        representativeness_scores = [
            1 - relative_cluster_sizes_training_set[cluster_id]
            for cluster_id in self.cluster_labels_unlabeled_set[
                self.is_selected == False
            ]
        ]

        return representativeness_scores
