#!/bin/bash
#SBATCH --job-name=amazontime_GPU
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x.out
cd /scratch/ds7000/ssh2/
module load python/intel/3.8.6
python amazontime_GPU.py
