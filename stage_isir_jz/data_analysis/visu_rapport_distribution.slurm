#!/bin/bash
#SBATCH --job-name=visu_rapport      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/visu_rapport_%j.out          # Fichier de sortie standard (%j = numéro du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/visu_rapport_%j.out           # Fichier d'erreur (fusionné avec la sortie)
#SBATCH -A uxg@cpu                     # Allocation du projet (uxg) sur les heures CPU
#SBATCH --partition=prepost
#SBATCH --time=10:00:00                  # Temps d'exécution 
#SBATCH --nodes=1                        # Un seul nœud de calcul
#SBATCH --ntasks=1                       # Une seule tâche
#SBATCH --cpus-per-task=10               # 10 cœurs pour accélérer Numpy/Scikit-learn
#SBATCH --hint=nomultithread             # Hyperthreading désactivé

# 1. Nettoyage de l'environnement
module purge

# 2. Chargement du module Python/PyTorch
# (On charge le même module que d'habitude pour avoir tes paquets, il marchera très bien sur CPU)
module load pytorch-gpu/py3/2.0.0

# 3. Activation de ton environnement local (pour cftime, netCDF4, eofs, cartopy)
export PYTHONUSERBASE=$WORK/.local_vit_v100

# 4. Déplacement dans ton dossier de travail
# (À adapter avec le nom exact du dossier où se trouve ton script python)
cd $WORK/stage_isir_jz/data_analysis/

#     --monthly_reduction \
#     --normalize \
#  --lat_weight

# 5. Exécution du code
srun python visu_rapport_distribution.py \
    --machine jean-zay-work \
    --number_of_members 10 \
    --max_component 10 \
    --pca_monthly /lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/monthly_reduction/lat_weightTrue/IPCA_latent128_winter_months_11_12_1_2_87members_normalizeFalse_monthly_reduction_wgtTrue_slp_std467.02/best_pca_model.joblib \
    --pca_daily /lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/0d/lat_weightTrue/IPCA_latent128_winter_months_11_12_1_2_87members_normalizeFalse_duree_lissage0_wgtTrue_slp_std979.45/best_pca_model.joblib

