#!/bin/bash
#SBATCH --job-name=cnn     # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/log/cnn_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/log/cnn_%j.out
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
cd $WORK/stage_isir_jz/cnn/cnn_with_slp_embedding/

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
srun python main_with_shap.py \
    --lat_weight \
    --nb_epochs 20 \
    --nb_intra_evals 15 \
    --winter_months 2 \
    --embed_method pca \
    --embed_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/monthly_reduction/lat_weightTrue/IPCA_latent128_winter_months_11_12_1_2_87members_normalizeFalse_monthly_reduction_wgtTrue_slp_std467.2/best_pca_model.joblib" \
    --latent_dim 34 \
    --roll_sst \
    --bs 128 \
    --lr 0.0004940742776420815 \
    --dr_conv 0.07627009102145332 \
    --dr_fc 0.5179574728374685 \
    --fc_dim 63 \
    --depth 2 \
    --n_feat 8 \
    --filter_mult 1.25 \
    --pool_type max \
    --sst_pool_x 2 \
    --sst_pool_y 3 \
    --sst_kx 3 \
    --sst_ky 5 \
    --activation relu \
    --pool_strategy progressive \
    --loss_type mse \
    --weight_decay 3.4791557369513082e-06 \
    --noise_std 0.004434448276742697 \
    --gradient_clip 28.104529655060375 \
    --early_fusion_sst \
    --slp_lags_months \
    --sst_lags_months 1 2 3 \
    --nb_members_train 87 \
    --force_val_members 1251.002 \
    --force_test_members 1081.005 \
    --nb_members_test 5 \
    --seed 1 \
    --monthly_reduction \
    --update 0


    


    
    



# 5. NETTOYAGE INDISPENSABLE
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."