#!/bin/bash
#SBATCH --job-name=latentexpl      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/latentvae_%j.out          # Fichier de sortie standard (%j = numéro du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/latentvae_error_%j.out           # Fichier d'erreur (fusionné avec la sortie)
#SBATCH -A uxg@cpu                       # Allocation du projet (uxg) sur les heures CPU
#SBATCH --time=01:00:00                  # Temps d'exécution 
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
cd $WORK/stage_isir_jz/data_analysis/vae_slp/

# 5. Exécution du code
srun python latent_space_exploration.py \
    --machine jean-zay-work \
    --number_of_members 50 \
    --z_idx 0 \
    --model_name "ConvVAE_bs128_lr0.0001_NDJF_beta500.0_80members_latent1_duree10d" \
    --duree_lissage 10 \
    --latent_dim 1 \
    --sampling_type gaussian