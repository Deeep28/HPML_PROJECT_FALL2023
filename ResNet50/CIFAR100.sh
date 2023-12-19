#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output="/scratch/hn2276/Project/c100.txt"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:4

cd /scratch/hn2276/Project

module load intel/19.1.2
module load anaconda3/2020.07
module load python/intel/3.8.6
module load cuda/11.6.2

# Run your Python script
python c100.py