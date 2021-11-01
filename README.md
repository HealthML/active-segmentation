# Master Project "Medical Image Segmentation"

## Project Setup

The recommended way of installing the project's dependencies is to use a virtual environment. To create a virtual environment, run the venv module inside the repository:

```
python3 -m venv env
```

Once you have created a virtual environment, you may activate it. On Windows, run:

```
env\Scripts\activate.bat
```

On Unix or MacOS, run:

```
source ./env/bin/activate
```

To install the dependencies, run:

```
python3 -m pip install -r requirements.txt
```

To be able to import the modules from the repository, run:

```
python3 -m pip install -e .
```

## Running the Active Learning Pipeline

To execute the active learning pipeline, run:

```
python3 main.py \
    --architecture "fcn_resnet50" \
    --dataset "pascal-voc" \
    --strategy "base" \
    --batch_size 16 \
    --epochs 1 \
    --optimizer "adam"
```