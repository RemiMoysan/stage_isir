#!/bin/bash
#SBATCH --job-name=optuna_plots      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/log/optunaplots_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/log/optunaplots_%j.out
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
cd $WORK/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna

# 5. Exécution du code
srun python optuna_cnn_monthly_r2R2_plot.py \
    --study-name cnn_monthly_pc1_correlation_max_n_trials500_months2_nb_epochs20_intra_evals15_patience5_nb_members_val5_sampler_pruner35_15_2_3 \
    --storage-name stage_isir_jz/cnn/cnn_with_slp_embedding/optuna/monthly_pca_correlation_max_months2_nb_epochs20_intra_evals15_patience5_nb_members_val5_sampler_pruner100_25_2_3_fixed_seedNone_include_lag1False/cnn_optuna.db