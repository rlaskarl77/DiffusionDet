#!/bin/bash

#SBATCH --partition=mlvu
#SBATCH --job-name=lvis_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=400000MB
#SBATCH --cpus-per-task=64
#SBATCH --output=lvis_train_resnet101.out
#SBATCH --error=test.err

source /home/n0/${USER}/.bashrc
source /home/n0/${USER}/anaconda3/etc/profile.d/conda.sh
conda activate dt2
cd /home/n0/mlvu008/DiffusionDet

srun python train_net.py --num-gpus 4 \
        --config-file configs/diffdet.lvis.res101.yaml \
        OUTPUT_DIR /home/n0/mlvu008/DiffusionDet/train/lvis_resnet101 > logs/train_lvis_resnet101.log 2>&1
