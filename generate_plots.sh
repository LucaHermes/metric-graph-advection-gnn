# Experiment 1.1 and 1.2: Results Figures WDS Experiments
python -u run_experiments_pde_gnn.py --results_plots_models MeGA-MP NNConv PDE-GNN GPSConv AntiSymmetricConv --exp_ids 1 &
python -u run_experiments_pde_gnn.py --results_plots_models MeGA-MP+MLP NNConv PDE-GNN GPSConv AntiSymmetricConv --exp_ids 2 &
# Ablation Study: Versions of MeGA-MP
python -u run_experiments_pde_gnn.py --results_plots_models MeGA-MP MeGA-MP+MLP MeGA-MP-learned --exp_ids 1 --out_prefix "ablation_" &
python -u run_experiments_pde_gnn.py --results_plots_models MeGA-MP MeGA-MP+MLP MeGA-MP-learned --exp_ids 2 --out_prefix "ablation_" &

wait