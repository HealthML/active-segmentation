# Master Project "Medical Image Segmentation"

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

## Running the Active Learning Pipeline

To execute the active learning pipeline, run:

```
python3 src/main.py \
    --architecture "fcn_resnet50" \
    --dataset "pascal-voc" \
    --strategy "base" \
    --batch_size 16 \
    --epochs 1 \
    --num_workers 2 \
    --optimizer "adam" \
    --loss "bce" \
    --gpus 1
```

Example command to train a U-net with the BraTS dataset on a GPU on the DHC Server:

```
srun -p gpupro --gpus=1 -c 18 --mem 150000 python -m memory_profiler main.py \
    --architecture "u_net" \
    --dataset "brats" \
    --strategy "base" \
    --batch_size 16 \
    --epochs 3 \
    --num_workers 8 \
    --optimizer "adam" \
    --loss "dice" \
    --data_dir "/dhc/groups/mpws2021cl1/Data" \
    --gpus 1
```

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