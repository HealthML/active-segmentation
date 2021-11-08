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
    --optimizer "adam" \
    --loss "bce" \
    --gpus 1
```

Example command to train a U-net with the BraTS dataset on a GPU on the DHC Server:
```
srun -p gpupro --gpus=1 python main.py \
    --architecture "u_net" \
    --dataset "brats" \
    --strategy "base" \
    --batch_size 16 \
    --epochs 1 \
    --optimizer "adam" \
    --loss "dice" \
    --data_dir "/dhc/groups/mpws2021cl1/Data"
    --gpus 1
```