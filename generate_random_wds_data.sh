#!/bin/bash
#SBATCH --job-name=advmpnn
#SBATCH --array=1-10
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --mem=10G
#SBATCH --time=10:00:00
#SBATCH --output=logs_datagen/gendata_%A_%a.log

# Number of parallel jobs
NUM_JOBS=10

INDEX=$((SLURM_ARRAY_TASK_ID - 1))

mkdir -p logs_datagen

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wquality

echo $START_FRAC 
echo $FRAC

python -u run_random_wds_data_generator.py $INDEX $NUM_JOBS

wait