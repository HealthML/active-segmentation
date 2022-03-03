""" Combined representativeness and uncertainty sampling strategy """

from typing import List, Literal

import numpy as np

from .representativeness_sampling_strategy_base import (
    RepresentativenessSamplingStrategyBase,
)
from .representativeness_sampling_distances import (
    DistanceBasedRepresentativenessSamplingStrategy,
)
from .representativeness_sampling_clustering import (
    ClusteringBasedRepresentativenessSamplingStrategy,
)
from .uncertainty_sampling_strategy import UncertaintySamplingStrategy


class UncertaintyRepresentativenessSamplingStrategy(
    RepresentativenessSamplingStrategyBase
):
    """
    Sampling strategy that combines representativeness and uncertainty sampling.

    Args:
        representativeness_algorithm (string, optional): The algorithm to be used to select the most representative
            samples: `"most_distant_sample"` | `"cluster_coverage"`. Defaults to `"cluster_coverage"`.
                - `"most_distant_sample"`: The unlabeled item that has the highest feature distance to the labeled set
                    is selected for labeling.
                - `"cluster_coverage"`: The features of the unlabeled and labeled items are clustered and an item from
                    the most underrepresented cluster is selected for labeling.
        calculation_method (string, optional): The algorithm to be used for computing the uncertainty: `"distance"` |
            "`entropy`".
    """

    def __init__(
        self,
        representativeness_algorithm: Literal[
            "most_distant_sample", "cluster_coverage"
        ] = "cluster_coverage",
        calculation_method: Literal["distance", "entropy"] = "entropy",
    ):
        super().__init__()

        if representativeness_algorithm == "most_distant_sample":
            self.representativeness_sampling_strategy = (
                DistanceBasedRepresentativenessSamplingStrategy()
            )
        elif representativeness_algorithm == "cluster_coverage":
            self.representativeness_sampling_strategy = (
                ClusteringBasedRepresentativenessSamplingStrategy()
            )
        else:
            raise ValueError(
                f"Invalid representativeness sampling algorithm: {representativeness_algorithm}."
            )
        self.uncertainty_sampling_strategy = UncertaintySamplingStrategy(
            calculation_method=calculation_method
        )

    def prepare_representativeness_computation(
        self, feature_vectors_training_set, feature_vectors_unlabeled_set
    ) -> None:
        """
        Prepares computation of representativeness scores.

        Args:
            feature_vectors_training_set (np.array): Feature vectors of the items in the training set.
            feature_vectors_unlabeled_set (np.array): Feature vectors of the items in the unlabeled set.
        """

        self.representativeness_sampling_strategy.prepare_representativeness_computation(
            feature_vectors_training_set, feature_vectors_unlabeled_set
        )

    def compute_representativeness_scores(
        self,
        model,
        data_module,
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

        representativeness_scores = (
            self.representativeness_sampling_strategy.compute_representativeness_scores(
                model,
                data_module,
                feature_vectors_training_set,
                feature_vectors_unlabeled_set,
            )
        )
        representativeness_scores = self._normalize_scores(
            np.array(representativeness_scores)
        )

        (
            uncertainty_scores,
            _,
        ) = self.uncertainty_sampling_strategy.compute_uncertainties(model, data_module)
        uncertainty_scores = self._normalize_scores(np.array(uncertainty_scores))

        return representativeness_scores + uncertainty_scores

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Normalizes vector of representativeness scores.

        Args:
            scores (np.ndarray): Vector to be normalized.

        Returns:
            np.ndarray: Normalized vector.
        """

        return (scores - scores.min(keepdims=True)) / (
            scores.max(keepdims=True) - scores.min(keepdims=True)
        )
