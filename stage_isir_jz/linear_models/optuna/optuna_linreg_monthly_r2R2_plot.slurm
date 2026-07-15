#!/bin/bash
#SBATCH --job-name=optuna_plots      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/optuna/log/optunaplots_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/optuna/log/optunaplots_%j.out
#SBATCH -A uxg@cpu                     # Allocation du projet (uxg) sur les heures CPU
#SBATCH --partition=prepost
#SBATCH --time=1:00:00                  # Temps d'exécution 
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
cd $WORK/stage_isir_jz/linear_models/optuna

# 5. Exécution du code
srun python optuna_linreg_monthly_r2R2_plot.py \
    --storage-name stage_isir_jz/linear_models/optuna/Optuna_LinReg_r2_months2/optuna.db
