Configuration
==============================

The configuration file is in the format of a `.json` file.
Most settings must be set explicitly in the configuration file.

.. _config-section-label:

The configuration file is split in several sections. There are no requirements for ordering within
these sections, or for ordering of the sections themselves.

.. _general_config-label:

General
--------------

.. _wandb_project_name-label:

.. index:: ! wandb_project_name

* *wandb_project_name*
    The name of the `weights and biases <https://wandb.ai/>`_ project.
    Weights and biases is setup in the project for experiment tracking and logging.

.. _experiment_name-label:

.. index:: ! experiment_name

* *experiment_name*
    The name of the `weights and biases <https://wandb.ai/>`_ experiment.

.. _experiment_tags-label:

.. index:: ! experiment_tags

* *experiment_tags*
    List of tags to place on the run in `weights and biases <https://wandb.ai/>`_.

.. _checkpoint_dir-label:

.. index:: ! checkpoint_dir

* *checkpoint_dir*
    The local directory where the model checkpoints are stored.

.. _batch_size-label:

.. index:: ! batch_size

* *batch_size*
    Number of training examples passed in one training step.

.. _epochs-label:

.. index:: ! epochs

* *epochs*
    The number of full dataset forward iterations the model should be trained with.

.. _num_workers-label:

.. index:: ! num_workers

* *num_workers*
    Number of workers processes for the data loading. If ``0``, multi-process data loading is disabled.

.. _gpus-label:

.. index:: ! gpus

* *gpus*
    The number of gpus to be used. If ``0``, the training runs on CPU.

.. _prediction_count-label:

.. index:: ! prediction_count

* *prediction_count*
    The number of sample predictions to be generated for the validation set. This option is intended for use cases where you want to assess a model's quality using sample predictions. 

.. _prediction_dir-label:

.. index:: ! prediction_dir

* *prediction_dir*
    The local directory to store the sample predictions.

.. _random_state-label:

.. index:: ! random_state

* *random_state*
    Constant to ensure reproducibility of random operations.

.. _model_config-label:

[model_config] section
-------------------------

The ``model_config`` section specifies parameters to setup the segmentation model architecture and losses.

.. _architecture-label:

.. index:: ! architecture

* *architecture*
    The name of the architecture to use.  Allowable values are: ``"u_net"``.

.. note::
    If the model architecture of your choice is not yet included in the framework,
    it can be added by subclassing :py:meth:`models.pytorch_model.PytorchModel`.

.. _optimizer-label:

.. index:: ! optimizer

* *optimizer*
    The name of the algorithm used to calculate the loss and update the weights. Allowable values are: ``"adam"`` and ``"sgd"`` (gradient descent).

.. _loss_config-label:

.. index:: ! loss_config

* *loss_config*
    Dictionary with loss parameters. Mandatory is the key ``"type"`` with one of the allowable values:
    ``"cross_entropy"``, ``"dice"``, ``"cross_entropy_dice"``, ``"general_dice"``, ``"fp"``, ``"fp_dice"``,
    ``"focal"``.
    More detailed documentation and configuration options of the losses can be looked up in
    :py:mod:`functional.losses<functional.losses>`.

.. _learning_rate-label:

.. index:: ! learning_rate

* *learning_rate*
    The step size at each iteration while moving towards a minimum of the loss. Defaults to ``0.0001``.

.. _num_levels-label:

.. index:: ! num_levels

* *num_levels*
    Number of levels (encoder and decoder blocks) in the U-Net. Defaults to ``4``.

.. _dim-label:

.. index:: ! dim

* *dim*
    The dimensionality of the U-Net. Allowable values are: ``2`` and ``3``. Defaults to ``2``.

.. _model_selection_criterion-label:

.. index:: ! model_selection_criterion

* *model_selection_criterion*
    The criterion for selecting the best model for checkpointing. Defaults to ``"loss"``.

.. _train_metrics-label:

.. index:: ! train_metrics

* *train_metrics*
    A list with the names of the metrics that should be computed and logged in each training and
    validation epoch of the training loop. Available options: ``"dice_score"``, ``"sensitivity"``,
    ``"specificity"``, ``"hausdorff95"``. Defaults to ``["dice_score"]``.

.. _train_metric_confidence_levels-label:

.. index:: ! train_metric_confidence_levels

* *train_metric_confidence_levels*
    A list of confidence levels for which the metrics specified in the :ref:`train_metrics<train_metrics-label>` parameter
    should be computed in the training loop (``trainer.fit()``). This parameter is used only for
    multi-label classification tasks. Defaults to ``[0.5]``.

.. _test_metrics-label:

.. index:: ! test_metrics

* *test_metrics*
    A list with the names of the metrics that should be computed and logged in the model validation
    or testing loop (``trainer.validate()``, ``trainer.test()``). Available options:
    ``"dice_score"``, ``"sensitivity"``, ``"specificity"``, ``"hausdorff95"``.
    Defaults to ``["dice_score", "sensitivity", "specificity", "hausdorff95"]``.

.. _test_metric_confidence_levels-label:

.. index:: ! test_metric_confidence_levels

* *test_metric_confidence_levels*
    A list of confidence levels for which the metrics specified in the :ref:`test_metrics<test_metrics-label>` parameter
    should be computed in the validation or testing loop. This parameter is used only for
    multi-label classification tasks. Defaults to ``[0.5]``.

.. _dataset_config-label:

[dataset_config] section
-------------------------

The ``dataset_config`` section specifies parameters to setup the dataset and data loading.

.. _dataset-label:

.. index:: ! dataset

* *dataset*
    The name of the dataset to use. Allowable values are: ``"brats"``, ``"decathlon"`` and ``"bcss"``.

.. _data_dir-label:

.. index:: ! data_dir

* *data_dir*
    The directory where the data of the selected :ref:`dataset<dataset-label>` resides.

.. _cache_size-label:

.. index:: ! cache_size

* *cache_size*
    Number of images to keep in memory between epochs to speed-up data loading. Defaults to ``0``.

.. note::
    Further mandatory or optional fields can be found in the documentation of the respective data module.
    Available data modules as of now are :py:meth:`datasets.decathlon_data_module.DecathlonDataModule`,
    :py:meth:`datasets.brats_data_module.BraTSDataModule` and :py:meth:`datasets.bcss_data_module.BCSSDataModule`.


.. _active_learning_config-label:

[active_learning_config] section
---------------------------------

The ``active_learning_config`` section specifies parameters to run the active learning loop.

.. _active_learning_mode-label:

.. index:: ! active_learning_mode

* *active_learning_mode*
    Enable/Disabled Active Learning Pipeline. Defaults to ``False``. If ``False``, the model is trained on the full training dataset.

.. _reset_weights-label:

.. index:: ! reset_weights

* *reset_weights*
    Enable/Disable resetting of weights after every active learning iteration. Defaults to ``False``.

.. _initial_training_set_size-label:

.. index:: ! initial_training_set_size

* *initial_training_set_size*
    Initial size of the training set if the active learning mode is activated. Defaults to ``1``.

.. _iterations-label:

.. index:: ! iterations

* *iterations*
    Iteration times how often the active learning pipeline should be executed.
    If ``None``, the active learning pipeline is run until the whole dataset is labeled. Defaults to ``None``.

.. _items_to_label-label:

.. index:: ! items_to_label

* *items_to_label*
    Number of items that should be selected for labeling in each active learning iteration. Defaults to ``1``.

.. _batch_size_unlabeled_set-label:

.. index:: ! batch_size_unlabeled_set

* *batch_size_unlabeled_set*
    Batch size for the unlabeled set. Defaults to :ref:`batch_size<batch_size-label>`.

.. _heatmaps_per_iteration-label:

.. index:: ! heatmaps_per_iteration

* *heatmaps_per_iteration*
    Number of heatmaps to be generated per active learning iteration. This option is intended for uses cases where you want to assess the quality of a sampling strategy using heatmaps of the model's predictions. Defaults to ``0``.


.. _strategy_config-label:

[strategy_config] section
--------------------------

The ``strategy_config`` section specifies parameters to setup the strategy to query for new examples to be labeled.

.. _type-label:

.. index:: ! type

* *type*
    Name of the sampling strategy to use. Allowable values are: ``"random"``, ``"interpolation"``,
    ``"uncertainty"``, ``"representativeness_distance"``, ``"representativeness_clustering"`` and
    ``"representativeness_uncertainty"``.

.. _description-label:

.. index:: ! description

* *description*
    Detailed description about the configuration of the strategy.
    The information is logged to make experiments clearer.

.. note::
    Further mandatory or optional fields can be found in the documentation of the respective strategy.
    Available strategies and their documentations can be found in the :doc:`query_strategies package<query\_strategies>`.
