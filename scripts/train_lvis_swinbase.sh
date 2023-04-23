#!/bin/bash

#SBATCH --partition=mlvu
#SBATCH --job-name=lvis_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=400000MB
#SBATCH --cpus-per-task=64
#SBATCH --output=lvis_trains_swinbase.out
#SBATCH --error=test.err

source /home/n0/${USER}/.bashrc
source /home/n0/${USER}/anaconda3/etc/profile.d/conda.sh
conda activate dt2
cd /home/n0/mlvu008/DiffusionDet

srun python train_net.py --num-gpus 4 \
        --config-file configs/diffdet.lvis.swinbase.yaml \
        MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 128 \
        MODEL.RPN.BATCH_SIZE_PER_IMAGE 64 \
        SOLVER.IMS_PER_BATCH 4 \
        OUTPUT_DIR /home/n0/mlvu008/DiffusionDet/train/lvis_swinbase2 > logs/train_lvis_swinbase2.log 2>&1
