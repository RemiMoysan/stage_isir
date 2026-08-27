#!/bin/bash
#SBATCH --job-name=optuna_monthly      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_embedding/log/optuna_embedding_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_embedding/log/optuna_embedding_%j.out
#SBATCH --partition=gpu_p6           # <-- CORRIGÉ : Partition H100
#SBATCH -C h100                      # Force la carte H100
#SBATCH --time=20:00:00              # Temps alloué (test rapide)
#SBATCH --nodes=1                    # 1 seul noeud
#SBATCH --ntasks=1                   # 1 seule tâche PyTorch
#SBATCH --gres=gpu:1                 # 1 seul GPU H100
#SBATCH --cpus-per-task=24           # Max CPU autorisé pour 1 H100 (96/4)
#SBATCH --hint=nomultithread         # Désactive l'hyperthreading
#SBATCH --account=knn@h100           # Ton projet sur H100
   


# 1. Nettoyage de l'environnement
module purge

# 2. Chargement de l'environnement PyTorch pour H100
module load arch/h100
module load pytorch-gpu/py3/2.5.1    # <-- CORRIGÉ : Version supportant Hopper
export PYTHONUSERBASE=$WORK/.local_vit_h100 

# 3. Déplacement dans ton répertoire de travail
cd $WORK/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_embedding/

# --- DÉBUT DES AJOUTS POUR LE MONITORING ---
mkdir -p log

LOG_GPU="log/gpu_usage_${SLURM_JOB_ID}.log"
LOG_CPU="log/cpu_usage_${SLURM_JOB_ID}.log"

echo "Début du monitoring. Fichiers : $LOG_GPU et $LOG_CPU"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 300 > $LOG_GPU &
PID_GPU=$!

(while true; do 
    echo "$(date) | CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')% | RAM: $(free -h | grep Mem | awk '{print $3}') used" >> $LOG_CPU; 
    sleep 300; 
done) &
PID_CPU=$!
# --- FIN DES AJOUTS POUR LE MONITORING ---

#     --include_lag1 \
#     --fixed_seed 1 \

# 4. Lancement de ton script Python
srun python optuna_hyperparameters.py \
    --optimize_metric correlation \
    --n_trials 700 \
    --nb_epochs 20 \
    --patience 5 \
    --nb_members_val 5 \
    --nb_intra_evals 15 \
    --n_startup_trials_tpe 150 \
    --n_startup_trials_pruner 50 \
    --n_warmup_steps 2 \
    --interval_steps 3 \
    --winter_months 12 \
    --embed_method pca \
    --embed_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/monthly_reduction/lat_weightTrue/IPCA_latent128_winter_months_11_12_1_2_87members_normalizeFalse_monthly_reduction_wgtTrue_slp_std467.2/best_pca_model.joblib" \
    --latent_dim 1 \
    --roll_sst \
    --lat_weight \
    --fixed_seed 1 \
    --activation relu \
    --use_gap false \
    --slp_lags_months \
    --sequential_lags \
    --pool_type max \
    --sst_pool_x 2 \
    --sst_pool_y 3 \
    --sst_kx 3 \
    --sst_ky 5 \
    --pool_strategy progressive \
    --loss_type correlation
    
# 5. NETTOYAGE INDISPENSABLE
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."