"""Distance-based representativeness sampling strategy"""

import math
from typing import List, Literal

import psutil
import numpy as np
import scipy

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from .representativeness_sampling_strategy_base import (
    RepresentativenessSamplingStrategyBase,
)


class DistanceBasedRepresentativenessSamplingStrategy(
    RepresentativenessSamplingStrategyBase
):
    # pylint: disable=too-few-public-methods

    """
    Representativeness sampling strategy that selects the items with the highest average feature distance to the items
    in the training set.

    Args:
        distance_metric (string, optional):  Metric to be used for calculation the distance between feature vectors:
            `"euclidean"` | `"cosine"` | `"russellrao"`.
    """

    def __init__(
        self,
        distance_metric: Literal["euclidean", "cosine", "russellrao"] = "euclidean",
    ):
        super().__init__()

        if distance_metric not in ["euclidean", "cosine", "russellrao"]:
            raise ValueError(f"Invalid distance metric: {distance_metric}.")

        self.distance_metric = distance_metric

    def _average_feature_distances(
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

        feature_vectors_unlabeled_set_splitted = np.array_split(
            feature_vectors_unlabeled_set, n_splits
        )

        average_feature_distances = np.zeros(len(feature_vectors_unlabeled_set))

        for idx, current_chunk_feature_vectors_unlabeled_set in enumerate(
            feature_vectors_unlabeled_set_splitted
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

        return self._average_feature_distances(
            feature_vectors_training_set, feature_vectors_unlabeled_set
        )
