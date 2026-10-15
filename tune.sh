#!/bin/bash
#SBATCH --job-name=advmpnn
#SBATCH --array=1-3                  # Run 4 jobs in the array
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH --output=logs2/tune_params_%A_%a.log  # JobID_ArrayIndex.log

# Number of parallel jobs
NUM_JOBS=3

mkdir -p logs2

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wquality

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    # python -u tune_hyperparameters.py MeGA-MP-learned &
    # python -u tune_hyperparameters.py GPSConv &
    # python -u tune_hyperparameters.py NNConv
    python -u tune_hyperparameters.py PDE-GNN &
    python -u tune_hyperparameters.py AntiSymmetricConv &
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    # python -u tune_hyperparameters.py GPSConv &
    # python -u tune_hyperparameters.py NNConv &
    python -u tune_hyperparameters.py PDE-GNN &
    python -u tune_hyperparameters.py AntiSymmetricConv &
    # python -u tune_hyperparameters.py MeGA-MP-learned &
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    # python -u tune_hyperparameters.py NNConv &
    python -u tune_hyperparameters.py PDE-GNN &
    python -u tune_hyperparameters.py AntiSymmetricConv &
    # python -u tune_hyperparameters.py MeGA-MP-learned &
    # python -u tune_hyperparameters.py GPSConv &
elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    python -u tune_hyperparameters.py PDE-GNN &
    python -u tune_hyperparameters.py AntiSymmetricConv &
    # python -u tune_hyperparameters.py MeGA-MP-learned
    # python -u tune_hyperparameters.py GPSConv &
    # python -u tune_hyperparameters.py NNConv &
else
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi



# python -u tune_hyperparameters.py MeGA-MP-learned &
# python -u tune_hyperparameters.py GPSConv &
# python -u tune_hyperparameters.py NNConv &
# python -u tune_hyperparameters.py PDE-GNN &
# python -u tune_hyperparameters.py AntiSymmetricConv &

wait