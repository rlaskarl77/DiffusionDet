#!/bin/bash

#SBATCH --partition=mlvu
#SBATCH --job-name=lvis_inference
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-01:20:00
#SBATCH --mem=32000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=lvis_swinbase.out
#SBATCH --error=test.err

source /home/n0/${USER}/.bashrc
source /home/n0/${USER}/anaconda3/etc/profile.d/conda.sh
conda activate dt2
cd /home/n0/mlvu008/DiffusionDet

srun python train_net.py --num-gpus 4 \
        --config-file configs/diffdet.lvis.swinbase.yaml \
        --eval-only \
        MODEL.WEIGHTS /home/n0/mlvu008/DiffusionDet/baselines/diffdet_lvis_swinbase.pth \
        OUTPUT_DIR /home/n0/mlvu008/DiffusionDet/inference/lvis_swinbase > logs/lvis_swinbase.log 2>&1
