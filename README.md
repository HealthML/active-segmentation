# ActiveSegmentation: A Simulation Framework for Benchmarking Active Learning Strategies for 3D Medical Image Segmentation


## Project Documentation

The documentation of our framework is available [here](https://healthml.github.io/active-segmentation/).
## Project Setup

The recommended way of installing the project's dependencies is to use a virtual environment. To create a virtual environment, run the venv module inside the repository:

```
python3 -m venv venv
```

Once you have created a virtual environment, you may activate it. On Windows, run:

```
venv\Scripts\activate.bat
```

On Unix or MacOS, run:

```
source ./venv/bin/activate
```

To install the dependencies, run:

```
python3 -m pip install -r requirements.txt
```

To execute the code, you have to add the `src`directory to your `PYTHONPATH`:
On Unix or MacOS, run:

```
export PYTHONPATH=$PWD:$PWD/src/
```

On Windows, run:

```
set PYTHONPATH=%cd%/src
```

To be able to import the modules from the repository, run:

```
python3 -m pip install -e .
```

## Setting up using conda
`conda env create -f environment.yaml`
`conda activate al`
## Additional Setup Steps

Install and log into Weights and Biases:

```
pip install wandb
wandb login
```

## Running the Active Learning Pipeline

To execute the active learning pipeline, run:

```
python3 src/main.py pascal_voc_example_config.json
```

Example command to train a U-net with the BraTS dataset on a GPU on the DHC Server:

```
srun -p gpupro --gpus=1 -c 18 --mem 150000 python3 src/main.py brats_example_config.json
```

For an overview of the available configuration options, see the `brats_example_config.json` file in this repository.
## Running Weights & Biases Sweeps

To execute the hyperparameter optimisation using W&B sweeps, run:

```
wandb sweep sweep.yaml
```

This will output a new `Sweep_ID` that you can use to run the agent via the provided shell script:

```
sbatch batch_sweeps.slurm <sweep_ID>
```

Configuring which hyperparameters should be optimized is done in the `sweep.yaml` file.

## Building the Documentation

The documentation is based on [Sphinx](https://www.sphinx-doc.org/en/master/). For detailed instructions on how to setup Sphinx documentations in general, see this [Read the Docs tutorial](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/install.html).

To generate the documentation files using the docstrings in the code, run:

```
sphinx-apidoc -o ./docs/source ./src
```

To build the HTML documentation, run the following command inside the `./docs` directory:

```
make clean && make html
```

To view the documentation, open the `./docs/build/html/index.html` file in your browser.

## Formatting
To see what black wants to format run:
```
black ./src --check
black ./tests --check
```

To actually apply the formatting:
```
black ./src
black ./tests
```
## Linting
```
pylint src --rcfile=.rcfile
```

## Running the Tests

To execute all unit tests, run:

```
python3 -m unittest discover
```

To execute only a specific test module run:

```
python3 -m unittest test.<name of test module>
```

To execute only a specific test case, run:

```
python3 -m unittest test.<name of test module>.<name of test case>
```
