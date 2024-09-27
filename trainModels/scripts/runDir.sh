#!/bin/bash

for lr in -3 -2 -1
do
    for l2 in -9 -3 -2 -1
    do

        sbatch trainModels.sbatch 300 32 $lr $l2
        sbatch trainModels.sbatch 300 32 $lr $l2 --isLin

        sbatch trainModels.sbatch 300 32,32 $lr $l2
        sbatch trainModels.sbatch 300 32,32 $lr $l2 --isLin

        sbatch trainModelsBoot.sbatch 300 32 $lr $l2
        sbatch trainModelsBoot.sbatch 300 32 $lr $l2 --isLin

        sbatch trainModelsBoot.sbatch 300 32,32 $lr $l2
        sbatch trainModelsBoot.sbatch 300 32,32 $lr $l2 --isLin

        #sbatch trainModels.sbatch 300 0 $lr $l2 --isLin

        #sbatch trainModels.sbatch 300 2 $lr $l2
        #sbatch trainModels.sbatch 300 2 $lr $l2 --isLin

        #sbatch trainModels.sbatch 300 4 $lr $l2
        #sbatch trainModels.sbatch 300 4 $lr $l2 --isLin

        #sbatch trainModels.sbatch 300 8 $lr $l2
        #sbatch trainModels.sbatch 300 8 $lr $l2 --isLin

        #sbatch trainModels.sbatch 300 16 $lr $l2
        #sbatch trainModels.sbatch 300 16 $lr $l2 --isLin

        #sbatch trainModels.sbatch 300 2,2 $lr $l2
        #sbatch trainModels.sbatch 300 2,2 $lr $l2 --isLin

        #sbatch trainModels.sbatch 300 4,4 $lr $l2
        #sbatch trainModels.sbatch 300 4,4 $lr $l2 --isLin

        #sbatch trainModels.sbatch 300 8,8 $lr $l2
        #sbatch trainModels.sbatch 300 8,8 $lr $l2 --isLin

        #sbatch trainModels.sbatch 300 16,16 $lr $l2
        #sbatch trainModels.sbatch 300 16,16 $lr $l2 --isLin


        #sbatch trainModelsOmniPCA.sbatch 600 0 $lr $l2 --isLin

        #sbatch trainModelsOmniPCA.sbatch 600 2 $lr $l2
        #sbatch trainModelsOmniPCA.sbatch 600 2 $lr $l2 --isLin

        #sbatch trainModelsOmniPCA.sbatch 600 4 $lr $l2
        #sbatch trainModelsOmniPCA.sbatch 600 4 $lr $l2 --isLin

        #sbatch trainModelsOmniPCA.sbatch 600 8 $lr $l2
        #sbatch trainModelsOmniPCA.sbatch 600 8 $lr $l2 --isLin

        #sbatch trainModelsOmni.sbatch 600 256 $lr $l2
        #sbatch trainModelsOmni.sbatch 600 256 $lr $l2 --isLin

        #sbatch trainModelsOmni.sbatch 600 16 $lr $l2
        #sbatch trainModelsOmni.sbatch 600 16 $lr $l2 --isLin

        #sbatch trainModelsOmni.sbatch 600 16,16 $lr $l2
        #sbatch trainModelsOmni.sbatch 600 16,16 $lr $l2 --isLin

        #sbatch trainModelsOmni.sbatch 600 16,16,16 $lr $l2
        #sbatch trainModelsOmni.sbatch 600 16,16,16 $lr $l2 --isLin

        #sbatch trainModelsOmni.sbatch 600 32,16,8,4 $lr $l2
        #sbatch trainModelsOmni.sbatch 600 32,16,8,4 $lr $l2 --isLin

        #sbatch trainModelsOmni.sbatch 600 32,16,8,4,2 $lr $l2
        #sbatch trainModelsOmni.sbatch 600 32,16,8,4,2 $lr $l2 --isLin
    done
done
