#!/bin/bash
#SBATCH --job-name=optunaNAO      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/optuna_embedding/log/loocv_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/optuna_embedding/log/loocv_%j.out
#SBATCH --partition=gpu_p6           # <-- CORRIGÉ : Partition H100
#SBATCH -C h100                      # Force la carte H100
#SBATCH --time=20:00:00              # Temps alloué (test rapide)
#SBATCH --nodes=1                    # 1 seul noeud
#SBATCH --ntasks=1                   # 1 seule tâche PyTorch
#SBATCH --gres=gpu:1                 # 1 seul GPU H100
#SBATCH --cpus-per-task=24           # Max CPU autorisé pour 1 H100 (96/4)
#SBATCH --hint=nomultithread         # Désactive l'hyperthreading
#SBATCH --account=uxg@h100           # Ton projet sur H100
   


# 1. Nettoyage de l'environnement
module purge

# 2. Chargement de l'environnement PyTorch pour H100
module load arch/h100
module load pytorch-gpu/py3/2.5.1    # <-- CORRIGÉ : Version supportant Hopper
export PYTHONUSERBASE=$WORK/.local_vit_h100 

# 3. Déplacement dans ton répertoire de travail
cd $WORK/stage_isir_jz/linear_models/optuna_embedding/

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
srun python optuna_cross_loocv.py \
    --optimize_metric R2 \
    --lat_weight \
    --nb_epochs 20 \
    --patience 5 \
    --nb_intra_evals 15 \
    --winter_months 12 \
    --embed_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/monthly_reduction/lat_weightTrue/IPCA_latent128_winter_months_11_12_1_2_87members_normalizeFalse_monthly_reduction_wgtTrue_slp_std467.2/best_pca_model.joblib" \
    --embed_path_sst "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_sst/IPCA_latent128_winter_months_8_9_10_11_12_1_2_87members_normalizeFalse_monthly_reduction_wgtTrue_sst_std0.55_roll_sstTrue/best_pca_model.joblib" \
    --latent_dim 1 \
    --machine jean-zay-work \
    --roll_sst \
    --bs 128 \
    --lr 0.0003004906363513125 \
    --weight_decay 2.1090895344113965e-06 \
    --noise_std 0.0002065430887463024 \
    --gradient_clip 1.9294772159363898 \
    --alpha_l1 1.984961837417368e-09 \
    --alpha_tik 1.4703429021712197e-09 \
    --alpha_lap 3.898028549037523e-09 \
    --loss_type mse \
    --input_format raw \
    --sst_lags_months 1 2 \
    --slp_lags_months
    
# 5. NETTOYAGE INDISPENSABLE
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."