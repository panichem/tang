#!/bin/bash
#SBATCH -t 30
#SBATCH -o ../logs/slurm-%j.out
#SBATCH -p tirin,normal

module load py-pytorch/2.2.1_py312

python3 -u trainModelsOmniPCA.py --epochs $1 --hiddenLayers $2 --learnRate $3 --l2 $4 $5 