#!/bin/bash -l
#SBATCH -o std_out
#SBATCH -e std_err 
#SBATCH -p Quick 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB
#SBATCH --gpus-per-task=8

python run_idp.py
