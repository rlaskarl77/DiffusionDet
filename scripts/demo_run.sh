#!/bin/bash

#SBATCH --job-name=example
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-00:15:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8

source /home/${USER}/.bashrc
conda activate dt2

srun python demo.py --config-file configs/diffdet.coco.res50.yaml \
    --input image.jpg --opts MODEL.WEIGHTS diffdet_coco_res50.pth