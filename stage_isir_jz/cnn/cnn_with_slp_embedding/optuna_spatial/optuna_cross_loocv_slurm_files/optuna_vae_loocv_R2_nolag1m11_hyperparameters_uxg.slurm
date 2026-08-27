#!/bin/bash
#SBATCH --job-name=optuna_vae_spatial     # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_spatial/log/loocv_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_spatial/log/loocv_%j.out
#SBATCH --time=10:00:00              # Temps alloué
#SBATCH --nodes=1                    # 1 seul noeud
#SBATCH --ntasks=1                   # 1 seule tâche PyTorch
#SBATCH --gres=gpu:1                 # 1 seul GPU A100 : 80 Go
#SBATCH --cpus-per-task=8            # <-- CRUCIAL : Le maximum autorisé par GPU sur A100
#SBATCH --hint=nomultithread         # Désactive l'hyperthreading
#SBATCH --account=uxg@a100           # Ton projet
#SBATCH --partition=gpu_p5           # <-- AJOUT : On tape dans la partition A100
#SBATCH -C a100                      # <-- AJOUT : On force la carte A100
   
   


# 1. Nettoyage de l'environnement
module purge

# 2. Chargement de l'environnement PyTorch officiel de Jean Zay
module load arch/a100
module load pytorch-gpu/py3/2.0.0
export PYTHONUSERBASE=$WORK/.local_vit_a100 

# 3. Déplacement dans ton répertoire de travail
cd $WORK/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_spatial/

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
    --winter_months 11 \
    --embed_method vae \
    --embed_path "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/vae_slp/monthly_reduction/ConvVAE_11_12_1_2_wgtTrue/beta0.01latent25.pth" \
    --latent_dim 25 \
    --roll_sst \
    --bs 64 \
    --lr 4.98122703669286e-05 \
    --dr_conv 0.09740983333804633 \
    --dr_fc 0.3605324274785175 \
    --fc_dim 44 \
    --depth 2 \
    --n_feat 6 \
    --filter_mult 1.0 \
    --pool_type max \
    --sst_pool_x 2 \
    --sst_pool_y 3 \
    --sst_kx 3 \
    --sst_ky 5 \
    --activation relu \
    --pool_strategy progressive \
    --loss_type mse \
    --weight_decay 3.062322309485233e-06 \
    --noise_std 0.0015069966245523708 \
    --gradient_clip 1.558539352930367 \
    --early_fusion_sst \
    --slp_lags_months \
    --sst_lags_months 2 3

    


    
    



# 5. NETTOYAGE INDISPENSABLE
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."