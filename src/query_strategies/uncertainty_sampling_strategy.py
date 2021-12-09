from .query_strategy import QueryStrategy


class UncertaintySamplingStrategy(QueryStrategy):
    def select_items_to_label(
        self,
        models: Union[PytorchModel, List[PytorchModel]],
        dataloader: DataLoader,
        number_of_items: int,
        **kwargs
    ) -> List[str]:
        """
        Selects subset of the unlabeled data that should be labeled next.
        Args:
            models: Current models that should be improved by selecting additional data for labeling.
            dataloader: Pytorch dataloader representing the unlabeled dataset.
            number_of_items: Number of items that should be selected for labeling.
            **kwargs: Additional, strategy-specific parameters.

        Returns:
            IDs of the data items to be labeled.
        """
        # Uncertainty Sampling: run items trough model and select the ones with highest uncertainty

        return []
