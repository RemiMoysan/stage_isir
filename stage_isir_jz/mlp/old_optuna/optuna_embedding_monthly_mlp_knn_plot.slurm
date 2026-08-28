#!/bin/bash
#SBATCH --job-name=optuna_plots      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/embedding_2_embedding/optuna/log/optunaplots_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/embedding_2_embedding/optuna/log/optunaplots_%j.out
#SBATCH -A uxg@cpu                     # Allocation du projet (uxg) sur les heures CPU
#SBATCH --partition=prepost
#SBATCH --time=2:00:00                  # Temps d'exécution 
#SBATCH --nodes=1                        # Un seul nœud de calcul
#SBATCH --ntasks=1                       # Une seule tâche
#SBATCH --cpus-per-task=10               # 10 cœurs pour accélérer Numpy/Scikit-learn
#SBATCH --hint=nomultithread             # Hyperthreading désactivé

# 1. Nettoyage de l'environnement
module purge

# 2. Chargement du module Python/PyTorch
# (On charge le même module que d'habitude pour avoir tes paquets, il marchera très bien sur CPU)
module load arch/h100
module load pytorch-gpu/py3/2.5.1

# 3. Activation de ton environnement local (pour cftime, netCDF4, eofs, cartopy)
export PYTHONUSERBASE=$WORK/.local_vit_v100

# 4. Déplacement dans ton dossier de travail
# (À adapter avec le nom exact du dossier où se trouve ton script python)
cd $WORK/stage_isir_jz/embedding_2_embedding/optuna

# 5. Exécution du code
srun python optuna_embedding_monthly_mlp_knn_plot.py \
    --study-name cnn_monthly_pc1_correlation_max_n_trials500_months2_nb_epochs20_intra_evals15_patience5_nb_members_val5_sampler_pruner35_15_2_3 \
    --storage-name stage_isir_jz/embedding_2_embedding/optuna/Optuna_ResNet_r2_m2_ep20_val2_rollTrue_lag1False_FIXED_lossmse_slpLagsempty_sstLags23456_optuna_s100_p25/optuna.db