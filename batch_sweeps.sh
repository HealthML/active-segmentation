#!/bin/bash -eux

#SBATCH --job-name=as_experiment

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=johannes.hagemann@student.hpi.de

#SBATCH --partition=gpua100

#SBATCH --mem=200000

#SBATCH --gpus=1

#SBATCH --output=job_%j.log
 
wandb agent $1