#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p volta-gpu
#SBATCH --mem=32g
#SBATCH -t 12:00:00
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=/work/users/r/e/regarry/BrainRegistrationPipeline/Pytorch-UNet/logs/train-%j.out
#SBATCH --error=/work/users/r/e/regarry/BrainRegistrationPipeline/Pytorch-UNet/logs/train-%j.out

# Record the start time
start_time=$(date +%s)
echo "Start time: $(date)"

module purge
module load anaconda
conda deactivate
conda activate unet
echo "Active conda environment:"
conda info --envs | grep '*' | awk '{print $1}'

module load cuda/11.8
hostname
lscpu
#free -h
#df -h
top -b | head -n 20
nvidia-smi
nvcc --version

#export CUDA_VISIBLE_DEVICES=0
conda run -n unet python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.current_device())"
cd /work/users/r/e/regarry/BrainRegistrationPipeline/Pytorch-UNet

conda run -n unet python train.py --amp --validation 20 --batch_size 50 --epochs 100 --scale .5

# Record the finish time
finish_time=$(date +%s)
echo "Finish time: $(date)"

# Calculate and print the elapsed time
elapsed_time=$((finish_time - start_time))
echo "Time elapsed: $(date -ud "@$elapsed_time" +'%H:%M:%S')"