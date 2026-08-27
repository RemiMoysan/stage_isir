#!/bin/bash
#SBATCH --job-name=linear_models     # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/log/linear_models_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/log/linear_models_%j.out
#SBATCH --time=1:00:00              # Temps alloué
#SBATCH --nodes=1                    # 1 seul noeud
#SBATCH --ntasks=1                   # 1 seule tâche PyTorch
#SBATCH --gres=gpu:1                 # 1 seul GPU A100 : 80 Go
#SBATCH --cpus-per-task=8            # <-- CRUCIAL : Le maximum autorisé par GPU sur A100
#SBATCH --hint=nomultithread         # Désactive l'hyperthreading
#SBATCH --account=uxg@a100           # Ton projet
#SBATCH --partition=gpu_p5           # <-- AJOUT : On tape dans la partition A100
#SBATCH -C a100                      # <-- AJOUT : On force la carte A100
#SBATCH --qos=qos_gpu_a100-dev

# 1. Nettoyage de l'environnement
module purge

# 2. Chargement de l'environnement PyTorch officiel de Jean Zay
module load arch/a100
module load pytorch-gpu/py3/2.0.0
export PYTHONUSERBASE=$WORK/.local_vit_a100 

# 3. Déplacement dans ton répertoire de travail
cd $WORK/stage_isir_jz/linear_models/

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

#     --optimize_metric R2 \
#       --patience 5 \
#     --include_lag1 \

# 4. Lancement de ton script Python
srun python main_with_shap.py \
    --lat_weight \
    --nb_epochs 20 \
    --nb_intra_evals 15 \
    --winter_months 2 \
    --embed_method pca \
    --embed_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/monthly_reduction/lat_weightTrue/IPCA_latent128_winter_months_11_12_1_2_87members_normalizeFalse_monthly_reduction_wgtTrue_slp_std467.2/best_pca_model.joblib" \
    --embed_path_sst "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_sst/IPCA_latent128_winter_months_8_9_10_11_12_1_2_87members_normalizeFalse_monthly_reduction_wgtTrue_sst_std0.55_roll_sstTrue/best_pca_model.joblib" \
    --latent_dim 1 \
    --machine jean-zay-work \
    --roll_sst \
    --bs 64 \
    --lr 0.0007795063382258341 \
    --weight_decay 6.3349405566944445e-06 \
    --noise_std 0.00028503184479410185 \
    --gradient_clip 62.90763323258471 \
    --alpha_l1 3.075643109417942e-09 \
    --alpha_tik 0.043550033817629646 \
    --alpha_lap 0.00029884025558256106 \
    --loss_type mse \
    --input_format raw \
    --slp_lags_months \
    --sst_lags_months 1 \
    --nb_members_train 87 \
    --force_val_members 1251.002 \
    --force_test_members 1081.005 \
    --nb_members_test 1 \
    --seed 1 \
    --monthly_reduction \
    --update 0


# 5. NETTOYAGE INDISPENSABLE
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."