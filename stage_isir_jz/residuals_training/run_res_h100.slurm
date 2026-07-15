#!/bin/bash
#SBATCH --job-name=Res_h100      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/residuals_training/res_h100_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/residuals_training/res_h100_%j.out
#SBATCH --partition=gpu_p6           # <-- CORRIGÉ : Partition H100
#SBATCH -C h100                      # Force la carte H100
#SBATCH --time=10:00:00              # Temps alloué (test rapide)
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
cd $WORK/stage_isir_jz/residuals_training/

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

# --roll_sst

# 4. Lancement de ton script Python
srun python train_residual_model.py \
    --machine jean-zay-work \
    --embed_method pca \
    --embed_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/IPCA_latent1_NDJF_87members_normalizeTrue_duree_lissage30/best_pca_model.joblib" \
    --latent_dim 1 \
    --winter_months 1 2 \
    --duree_lissage 30 \
    --roll_sst \
    --seed 42 \
    --base_model_type "linreg" \
    --base_model_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/LinReg_l2_0.0_pca_emb_1_sst_-7_slp__bs128_lr5e-05_seed_42_30d_train80_val5/best_model_LinReg.pth" \
    --base_sst_lags "-7" \
    --base_slp_lags \
    --res_model_type cnn \
    --res_sst_lags 10 20 30 \
    --res_slp_lags \
    --nb_epochs 15 \
    --bs 128 \
    --lr 5e-5 \
    --dr 0.5 \
    --nb_members_train 80 \
    --nb_members_val 5


# 5. NETTOYAGE INDISPENSABLE
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."