#!/bin/bash
#SBATCH --job-name=ViTPcaEmb      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/log/vit_output_%j.out   # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/log/vit_error_%j.err     # Fichier d'erreur
#SBATCH --partition=gpu_p6           # <-- CORRIGÉ : Partition H100
#SBATCH -C h100                      # Force la carte H100
#SBATCH --time=01:00:00              # Temps alloué (test rapide)
#SBATCH --nodes=1                    # 1 seul noeud
#SBATCH --ntasks=1                   # 1 seule tâche PyTorch
#SBATCH --gres=gpu:1                 # 1 seul GPU H100
#SBATCH --cpus-per-task=24           # Max CPU autorisé pour 1 H100 (96/4)
#SBATCH --hint=nomultithread         # Désactive l'hyperthreading
#SBATCH --account=uxg@h100           # Ton projet sur H100
#SBATCH --qos=qos_gpu_h100-dev

# 1. Nettoyage de l'environnement
module purge

# 2. Chargement de l'environnement PyTorch pour H100
module load arch/h100
module load pytorch-gpu/py3/2.5.1    # <-- CORRIGÉ : Version supportant Hopper
export PYTHONUSERBASE=$WORK/.local_vit_h100 

# 3. Déplacement dans ton répertoire de travail
cd $WORK/stage_isir_jz/vision_transformer/vit_with_slp_embedding/

# --- DÉBUT DES AJOUTS POUR LE MONITORING ---

# Création du dossier 'log' s'il n'existe pas encore
mkdir -p log

# Chemins des fichiers (relatifs au dossier actuel vit_with_full_slp)
LOG_GPU="log/gpu_usage_${SLURM_JOB_ID}.log"
LOG_CPU="log/cpu_usage_${SLURM_JOB_ID}.log"

echo "Début du monitoring. Fichiers : $LOG_GPU et $LOG_CPU"

# Monitoring GPU toutes les 300 secondes (5 min) en tâche de fond (&)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 300 > $LOG_GPU &
PID_GPU=$!

# Monitoring CPU/RAM toutes les 300 secondes (5 min) en tâche de fond (&)
(while true; do 
    echo "$(date) | CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')% | RAM: $(free -h | grep Mem | awk '{print $3}') used" >> $LOG_CPU; 
    sleep 300; 
done) &
PID_CPU=$!

# --- FIN DES AJOUTS POUR LE MONITORING ---

#     --use_lags_attention \

# 4. Lancement de ton script Python
srun python main.py \
    --update 0 \
    --machine jean-zay-work \
    --embed_method pca \
    --embed_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/monthly_reduction/IPCA_latent1_winter_months_2_87members_normalizeTrue_monthly_reduction_wgtFalse_slp_std575.01/best_pca_model.joblib" \
    --seed 1 \
    --nb_members_train 80 \
    --nb_members_val 4 \
    --nb_members_test 3 \
    --latent_dim 1 \
    --sst_lags_months 1 2 3 4 \
    --slp_lags_months \
    --winter_months 2 \
    --bs 32 \
    --lr 1e-5 \
    --dr 0.15 \
    --nb_epochs 5 \
    --duree_lissage 10 \
    --normalize \
    --loss_type mse \
    --roll_sst \
    --monthly_reduction \


# 5. NETTOYAGE INDISPENSABLE
# On tue les boucles de monitoring pour que le job SLURM puisse se terminer proprement
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."