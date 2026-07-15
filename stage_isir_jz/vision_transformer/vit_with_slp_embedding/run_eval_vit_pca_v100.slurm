#!/bin/bash
#SBATCH --job-name=eval_vit      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/log/vit_output_%j.out   # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/log/vit_error_%j.err     # Fichier d'erreur
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

# mettre --roll_sst si utilisé pendant l'entrainement
#     --monthly_mean \
#     --use_lags_attention \
#     --roll_sst \
#     --loss_type quantile \
# seul cas où l'argument loss type est utile, les trois autres cas sont équivalents (l1,mse,corr)
#     --quantiles 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \

# 4. Lancement de ton script Python
srun python eval_vit.py \
    --machine jean-zay-work \
    --embed_method pca \
    --vit_dir /lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/ViT_Latent_lags_attention_True_pca_normalize_True_loss_correlation_embedding_1_lags_35_65_95_sst__slp_bs32_lr1e-05_dr0.5_months_1_2_train50_val_1_members_seed_1_10d_roll_sst_True \
    --embed_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/IPCA_latent1_NDJF_87members_normalizeTrue_duree_lissage10/best_pca_model.joblib" \
    --nb_members_train 10 \
    --nb_members_val 1 \
    --seed 1 \
    --latent_dim 1 \
    --sst_lags_days 35 65 95 \
    --slp_lags_days \
    --winter_months 1 2 \
    --duree_lissage 10 \
    --bs 32 \
    --n_bootstraps 10000 \
    --model_type best \
    --use_lags_attention \
    --roll_sst \
    --loss_type correlation \
    --monthly_mean \



    


# 5. NETTOYAGE INDISPENSABLE
# On tue les boucles de monitoring pour que le job SLURM puisse se terminer proprement
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."