#!/bin/bash
#SBATCH -J gpt2-train					  # name of job
#SBATCH -p gpu      					  # name of partition or queue (if not specified default partition is used)
#SBATCH --cpus-per-task=8                 # number of cores/threads per task (default 1)
#SBATCH --gpus=1                          # number of GPUs to request (default 0)
#SBATCH --gres=gpu:1          # reserve 1 GPU card
#SBATCH --mem=32G                         # request 16 gigabytes memory (per node, default depends on node)
#SBATCH --time 3-00:00:00 # time (D-HH:MM:SS)
#SBATCH -o train.%A_%a.%N.out				  # name of output file for this submission script
#SBATCH -e train.%A_%a.%N.err

hostname
date

source ${HOME}/gpt-env/bin/activate

module avail
module load cuda/11.2

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICE=0

cd ${HOME}/main/hw3/

ARG=( "$@" )

srun python3 test.py

date
