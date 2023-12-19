#!/bin/bash
#SBATCH --job-name=amazontime_GPU_opt_p_w_4
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --output=%x.out
cd /scratch/ds7000/ssh2/
module load python/intel/3.8.6
python amazontime_GPU_opt_parallel.py
