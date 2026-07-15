#!/bin/bash
#SBATCH --job-name=pc1_composites      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/pc1_composites_%j.out          # Fichier de sortie standard (%j = numéro du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/pc1_composites_%j.out           # Fichier d'erreur (fusionné avec la sortie)
#SBATCH -A uxg@cpu                       # Allocation du projet (uxg) sur les heures CPU
#SBATCH --time=20:00:00                  # Temps d'exécution 
#SBATCH --partition=prepost
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

# 5. Exécution du code

#    --lat_weight \

srun python composites_pc1.py \
    --machine jean-zay-work \
    --n_quantiles 5 \
    --model_path /lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/monthly_reduction/IPCA_latent1_winter_months_11_87members_normalizeTrue_monthly_reduction_wgtFalse_slp_std436.44/best_pca_model.joblib \
    --normalize \
    --duree_lissage 10 \
    --number_of_members 89 \
    --winter_months 2 \
    --monthly_reduction \
