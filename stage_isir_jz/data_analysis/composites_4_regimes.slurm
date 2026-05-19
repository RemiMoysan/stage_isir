#!/bin/bash
#SBATCH --job-name=regimes_composites      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/regimes_composites_%j.out          # Fichier de sortie standard (%j = numéro du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/regimes_composites_%j.out           # Fichier d'erreur (fusionné avec la sortie)
#SBATCH -A uxg@cpu                       # Allocation du projet (uxg) sur les heures CPU
#SBATCH --time=10:00:00                  # Temps d'exécution 
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
# --normalize \ à ajouter pour le nom du fichier si on utilise des PC normalisés
srun python composites_4_regimes.py \
    --machine jean-zay-work \
    --embedding_method pca \
    --model_path /lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/IPCA_latent128_NDJF_87members_normalizeFalse_duree_lissage30/best_pca_model.joblib \
    --duree_lissage 30 \
    --latent_dim 128 \
    --number_of_members 89