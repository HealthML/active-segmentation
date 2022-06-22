.. Active Segmentation documentation master file, created by
   sphinx-quickstart on Sat Nov  6 14:42:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Active Segmentation
=======================================

*An active learning benchmarking framework for scientists seeking comparability and reproducibility.*

``active segmentation`` provides a flexible and expressive API for evaluating active learning strategies for
medical image segmentation.

With ``active segmentation``, you can:

#. Define own segmentation models by implementing :py:meth:`models.pytorch_model.PytorchModel`, or use pre-configured :ref:`models<models>` like the U-Net.
#. Add own datasets by implementing :py:meth:`datasets.data_module.ActiveLearningDataModule`, or use already added :ref:`datasets<datasets>` like the
   `medical segmentation decathlon <http://medicaldecathlon.com>`_.
#. Evaluate active learning :ref:`query strategies<query_strategies`, or define and test new strategies by implementing :py:meth:`query_strategies.query_strategy.QueryStrategy`.
#. Train two-dimensional or three-dimensional segmentation models.
#. Run fully reproducible experiments, with seeded random processes and only deterministic operations.
#. Track various :ref:`metrics<metric_tracking>` with `Weights and Biases <https://wandb.ai/>`_.

Issues
------

Submit issues, feature requests or bugfixes on
`github <https://github.com/HealthML/active-segmentation/issues>`__.


How to Cite
-----------

If you use ``active segmentation`` in the context of academic or industry research, please
consider citing the paper.

`Paper <...>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    @...{
    }

License
-------------------

``active segmentation`` is licensed under the `AGPL-3.0 license <https://github.com/HealthML/active-segmentation/blob/main/LICENSE>`_.


.. toctree::
   :maxdepth: 2
   :caption: Introduction
   :hidden:

   self

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   configuration
   active_segmentation

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
