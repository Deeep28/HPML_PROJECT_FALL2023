#!/bin/bash
#SBATCH --job-name=amazontime_opt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=%x.out

cd /scratch/ds7000/ssh2/

module load python/intel/3.8.6

python amazontime_opt.py
