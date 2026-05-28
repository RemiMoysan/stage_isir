#!/bin/bash
#SBATCH --job-name=Lightgbm      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/trees/log/lightgbm_output_%j.out   # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/trees/log/lightgbm_error_%j.err     # Fichier d'erreur
#SBATCH -A uxg@cpu                       # Allocation du projet (uxg) sur les heures CPU
#SBATCH --time=10:00:00                  # Temps d'exécution 
#SBATCH --partition=prepost              # Partition CPU
#SBATCH --nodes=1                        # Un seul nœud de calcul
#SBATCH --ntasks=1                       # Une seule tâche
#SBATCH --cpus-per-task=10               # 10 cœurs pour accélérer Numpy/Scikit-learn/LightGBM
#SBATCH --hint=nomultithread             # Hyperthreading désactivé

# 1. Nettoyage de l'environnement
module purge

# 2. Chargement de l'environnement PyTorch officiel de Jean Zay
# (Même si c'est la version GPU, elle tournera sur CPU sans problème pour ton DataLoader)
module load pytorch-gpu/py3/2.0.0
export PYTHONUSERBASE=$WORK/.local_vit_v100 

# 3. Déplacement dans ton répertoire de travail
cd $WORK/stage_isir_jz/trees/

# --- DÉBUT DU MONITORING (CPU UNIQUEMENT) ---

# Création du dossier 'log' s'il n'existe pas encore
mkdir -p log

# Chemin du fichier (relatif au dossier actuel)
LOG_CPU="log/cpu_usage_${SLURM_JOB_ID}.log"

echo "Début du monitoring. Fichier : $LOG_CPU"

# Monitoring CPU/RAM toutes les 300 secondes (5 min) en tâche de fond (&)
(while true; do 
    echo "$(date) | CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')% | RAM: $(free -h | grep Mem | awk '{print $3}') used" >> $LOG_CPU; 
    sleep 300; 
done) &
PID_CPU=$!

# --- FIN DU MONITORING ---


# 4. Lancement de ton script Python
# Attention : suppression de --update, --dr et --nb_epochs qui n'existent pas pour LightGBM
srun python train_lightgbm.py \
    --machine jean-zay-work \
    --seed 42 \
    --nb_members_train 80 \
    --nb_members_val 7 \
    --sst_lags_days 15 35 65 \
    --slp_lags_days \
    --winter_months 1 2 \
    --bs 128 \
    --lr 0.05 \
    --duree_lissage 30 \
    --metric mse \
    --smooth_sigma 1.5 \
    --master_ref_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/composites_4_regimes/master_ref_generator_89members_normalizeFalse_duree_lissage30_embedding_method_pca_latent_dim_128/master_reference_global.npz"


# 5. NETTOYAGE INDISPENSABLE
# On tue la boucle de monitoring pour que le job SLURM puisse se terminer proprement
kill $PID_CPU
echo "Entraînement et monitoring terminés."