#!/bin/bash
#SBATCH --job-name=eval_cnn      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/log/eval_cnn_%j.out   # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/log/eval_cnn_%j.out    # Fichier d'erreur
#SBATCH --partition=gpu_p2l           # <-- NOUVEAU : Partition avec les V100 32 Go
#SBATCH --time=02:00:00              # Temps alloué
#SBATCH --nodes=1                    # 1 seul noeud
#SBATCH --ntasks=1                   # 1 seule tâche PyTorch
#SBATCH --gres=gpu:1                 # 1 seul GPU V100 (32 Go)
#SBATCH --cpus-per-task=3            # <-- CRUCIAL : Le maximum autorisé par GPU sur gpu_p2 !
#SBATCH --hint=nomultithread         # Désactive l'hyperthreading
#SBATCH --account=uxg@v100           # Ton projet
#SBATCH --qos=qos_gpu-dev            # Qualité de service pour les GPUs

# 1. Nettoyage de l'environnement
module purge

# 2. Chargement de l'environnement PyTorch officiel de Jean Zay
module load pytorch-gpu/py3/2.0.0
export PYTHONUSERBASE=$WORK/.local_vit_v100 

# 3. Déplacement dans ton répertoire de travail
cd $WORK/stage_isir_jz/cnn/cnn_with_slp_embedding/

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

# mettre --roll_sst si utilisé pendant l'entrainement
#     --monthly_mean \

# 4. Lancement de ton script Python
srun python eval_cnn.py \
    --machine jean-zay-work \
    --embed_method vae \
    --cnn_dir /lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/CNN_Latent_vae_beta_500.0_embedding_1_lags_15_30_45_sst__slp_bs128_lr1e-05_dr0.4_months_1_2_train50_val_3_members_seed_1_30d \
    --embed_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/vae_slp/ConvVAE_bs128_lr0.0001_NDJF_beta500.0_80members_latent1_duree30d/best_vae.pth" \
    --nb_members_train 50 \
    --nb_members_val 3 \
    --seed 1 \
    --latent_dim 1 \
    --sst_lags_days 15 30 45 \
    --slp_lags_days \
    --winter_months 1 2 \
    --duree_lissage 30 \
    --bs 128 \
    --n_bootstraps 20000 \
    --model_type final 

    


# 5. NETTOYAGE INDISPENSABLE
# On tue les boucles de monitoring pour que le job SLURM puisse se terminer proprement
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."