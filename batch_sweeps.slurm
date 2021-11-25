#!/bin/bash -eux

#SBATCH --job-name=as_experiment

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=<USER_EMAIL>

#SBATCH --partition=gpupro

#SBATCH --mem=200000

#SBATCH --gpus=1

#SBATCH --output=job_%j.log
 
wandb agent $1