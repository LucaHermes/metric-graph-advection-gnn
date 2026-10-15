#!/bin/bash
#SBATCH --job-name=advmpnn
#SBATCH --array=1-6                  # Run 6 jobs in the array
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs_train/training_rnd_%A_%a.log  # JobID_ArrayIndex.log

# Number of parallel jobs
NUM_JOBS=6

INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wquality

# if [ $INDEX -eq 0 ]; then
#     python -u run_experiments_pde_gnn.py --train_models "PDE-GNN" --eval_models "PDE-GNN" --exp_ids 1 2
# elif [ $INDEX -eq 1 ]; then
#     python -u run_experiments_pde_gnn.py --train_models "MeGA-MP-learned" --eval_models "MeGA-MP-learned" --exp_ids 1 2
# elif [ $INDEX -eq 2 ]; then
#     python -u run_experiments_pde_gnn.py --train_models "NNConv" --eval_models "NNConv" --exp_ids 1 2
# elif [ $INDEX -eq 3 ]; then
#     python -u run_experiments_pde_gnn.py --train_models "AntiSymmetricConv" --eval_models "AntiSymmetricConv" --exp_ids 1 2
#     python -u run_experiments_pde_gnn.py --eval_models "AntiSymmetricConv-Dia" --exp_ids 1 2
# elif [ $INDEX -eq 4 ]; then
#     python -u run_experiments_pde_gnn.py --train_models "GPSConv" --eval_models "GPSConv" --exp_ids 1 2
# elif [ $INDEX -eq 5 ]; then
#     python -u run_experiments_pde_gnn.py --train_models "MeGA-MP+MLP" --eval_models "MeGA-MP+MLP" --exp_ids 1 2
#     python -u run_experiments_pde_gnn.py --eval_models "MeGA-MP" --exp_ids 1 2
# fi

# if [ $INDEX -eq 0 ]; then
#     python -u run_experiments_pde_gnn.py --eval_models "PDE-GNN" --exp_ids 1 2
# elif [ $INDEX -eq 1 ]; then
#     python -u run_experiments_pde_gnn.py --eval_models "MeGA-MP-learned" --exp_ids 1 2
# elif [ $INDEX -eq 2 ]; then
#     python -u run_experiments_pde_gnn.py --eval_models "NNConv" --exp_ids 1 2
# elif [ $INDEX -eq 3 ]; then
#     python -u run_experiments_pde_gnn.py --eval_models "AntiSymmetricConv" --exp_ids 1 2
#     python -u run_experiments_pde_gnn.py --eval_models "AntiSymmetricConv-Dia" --exp_ids 1 2
# elif [ $INDEX -eq 4 ]; then
#     python -u run_experiments_pde_gnn.py --eval_models "GPSConv" --exp_ids 1 2
# elif [ $INDEX -eq 5 ]; then
#     python -u run_experiments_pde_gnn.py --eval_models "MeGA-MP" --exp_ids 1 2
# fi


if [ $INDEX -eq 0 ]; then
    python -u run_experiments_random_wds.py --eval_models "PDE-GNN"
elif [ $INDEX -eq 1 ]; then
    python -u run_experiments_random_wds.py --eval_models "MeGA-MP-learned"
elif [ $INDEX -eq 2 ]; then
    python -u run_experiments_random_wds.py --eval_models "NNConv"
elif [ $INDEX -eq 3 ]; then
    python -u run_experiments_random_wds.py --eval_models "AntiSymmetricConv"
elif [ $INDEX -eq 4 ]; then
    # python -u run_experiments_random_wds.py --eval_models "GPSConv"
    python -u run_experiments_random_wds.py --eval_models "AntiSymmetricConv-Dia"
elif [ $INDEX -eq 5 ]; then
    python -u run_experiments_random_wds.py --eval_models "MeGA-MP+MLP"
    python -u run_experiments_random_wds.py --eval_models "MeGA-MP"
fi