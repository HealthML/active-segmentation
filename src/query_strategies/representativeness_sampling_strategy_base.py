""" Base class for implementing representativeness sampling strategies """
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import torch

from datasets import ActiveLearningDataModule
from models.pytorch_model import PytorchModel
from models import PytorchUNet
from .query_strategy import QueryStrategy

# pylint: disable=too-few-public-methods
class RepresentativenessSamplingStrategyBase(QueryStrategy, ABC):
    """
    Base class for implementing representativeness sampling strategies
    """

    def __init__(self):
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

        self.prepare_representativeness_computation(
            feature_vectors_training_set, feature_vectors_unlabeled_set
        )

        for _ in range(min(items_to_label, len(feature_vectors_unlabeled_set))):
            representativeness_scores = self.compute_representativeness_scores(
                models,
                data_module,
                feature_vectors_training_set,
                feature_vectors_unlabeled_set,
            )

            # sort unlabeled items by their representativeness score
            unlabeled_indices = np.arange(len(case_ids))
            representativeness_scores = list(
                zip(representativeness_scores, unlabeled_indices)
            )
            representativeness_scores.sort(key=lambda y: y[0], reverse=True)
            # select the sample with the highest representativeness score
            index_of_most_representative_sample = representativeness_scores[0][1]
            selected_ids.append(case_ids[index_of_most_representative_sample])

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
            del case_ids[index_of_most_representative_sample]

        interception_hook.remove()

        return selected_ids, None

    def prepare_representativeness_computation(
        self, feature_vectors_training_set, feature_vectors_unlabeled_set
    ) -> None:
        """
        Can be overridden in subclasses to perform global computations on all feature vectors before item selection
        starts.

        Args:
            feature_vectors_training_set (np.array): Feature vectors of the items in the training set.
            feature_vectors_unlabeled_set (np.array): Feature vectors of the items in the unlabeled set.
        """

    @abstractmethod
    def compute_representativeness_scores(
        self,
        model: PytorchModel,
        data_module: ActiveLearningDataModule,
        feature_vectors_training_set: np.ndarray,
        feature_vectors_unlabeled_set: np.ndarray,
    ) -> List[float]:
        """
        Must be overridden in subclasses to compute the representativeness scores for the items in the unlabeled set.

        Args:
            model (PytorchModel): Current model that should be improved by selecting additional data for labeling.
            data_module (ActiveLearningDataModule): A data module object providing data.
            feature_vectors_training_set (np.ndarray): Feature vectors of the items in the training set.
            feature_vectors_unlabeled_set (np.ndarray): Feature vectors of the items in the unlabeled set.

        Returns:
            List[float]: Representativeness score for each item in the unlabeled set. Items that are underrepresented in
                the training set should receive higher scores.
        """
