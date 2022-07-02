Extending the Active Segmentation Framework
=============================================

Integration of Custom Model Architectures
------------------------------------------

Custom model architectures can be implemented by subclassing :py:meth:`models.pytorch_model.PytorchModel`. This class is
a subclass of the :py:meth:`LightningModule` class from `PyTorchLightning <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule>`__.
Accordingly, training and inference logic can be implemented by overwriting the :py:meth:`training_step(self, batch: torch.Tensor, batch_idx: int) -> float:`, :py:meth:`validation_step(self, batch: torch.Tensor, batch_idx: int) -> float:`, and :py:meth:`predict_step(self, batch: torch.Tensor, batch_idx: int) -> Any` methods of :py:meth:`models.pytorch_model.PytorchModel`.
Additionally, the method `input_dimensionality(self) -> int` must be overwritten and return the input dimensionality expected by the custom model.

Integration of Custom Datasets
------------------------------------------

Custom datasets can be implemented by subclassing :py:meth:`datasets.data_module.ActiveLearningDataModule`, which is a subclass of the :py:meth:`LightningModule` class from `PyTorchLightning <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.LightningDataModule.html?highlight=LightningDataModule>`__.
Subclasses must implement the methods :py:meth:`_create_training_set(self) -> torch.utils.data.Dataset`, :py:meth:`_create_validation_set(self) -> torch.utils.data.Dataset`, and :py:meth:` _create_test_set(self) -> torch.utils.data.Dataset`. In these methods custom datasets for training, validation, and testing can be created and must be returned as instances of `torch.utils.data.Dataset`.
The active learning mode requires subclasses to implement two additional methods: :py:meth:`_create_unlabeled_set(self) -> torch.utils.data.Dataset` and `label_items(self, ids: List[str], pseudo_labels: Optional[Dict[str, Any]]) -> None`. The first method must return an instance of `torch.utils.data.Dataset`, which represents the pool of unlabeled data from which a subset is to be selected for labeling.
The second method will be called in the active learning loop to simulate the annotation procedure and should move the data items whose ID is in `ids` from the unlabeled dataset to the training dataset.

Additionally, subclasses of :py:meth:`datasets.data_module.ActiveLearningDataModule` must implement the method :py:meth:`id_to_class_names(self) -> Dict[int, str]`, which returns a dictionary mapping class indices to class names, and the method py:meth:`multi_label(self) -> bool`, which returns whether the dataset defines a single-label or a multi-label classification task.

Integration of Custom Query Strategies
------------------------------------------

Custom query strategies can be implemented by subclassing :py:meth:`query_strategies.query_strategy.QueryStrategy` and implementing the :py:meth:`select_items_to_label` method of this class.
This method will be called in each active learning iteration of the active learning loop. It must implement the logic for selecting the most informative subset of the unlabeled pool that should be labeled next.
For this, both the current model and the current datasets are passed as parameters.
