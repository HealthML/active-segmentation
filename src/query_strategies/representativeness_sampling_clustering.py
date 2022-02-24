"""Clustering-based representativeness sampling strategy"""

import logging
from typing import List

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

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
        bandwidth (float, optional): Kernel bandwidth of the mean shift clustering algorithm. Defaults to 5.
        feature_dimensionality (int, optional): Number of dimensions the reduced feature vector should have.
            Defaults to 10.
    """

    def __init__(self, bandwidth: float = 5, feature_dimensionality: int = 10):
        super().__init__()

        self.bandwidth = bandwidth
        self.feature_dimensionality = feature_dimensionality

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

        reduced_feature_vectors = self.reduce_features(feature_vectors)

        clustering = MeanShift(bandwidth=self.bandwidth).fit(reduced_feature_vectors)

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
