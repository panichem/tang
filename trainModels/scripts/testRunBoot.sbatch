#!/bin/bash
#SBATCH -p tirin
#SBATCH -t 360
#SBATCH -o ../logs/slurm-%j.out

module load py-pytorch/2.2.1_py312

python3 trainModelsBootWrapper.py

