#!/bin/bash
#SBATCH --job-name=train_vae      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/vae_%j.out   # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/vae_%j.out    # Fichier d'erreur
#SBATCH --time=0:40:00              # Temps alloué
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
cd $WORK/stage_isir_jz/data_analysis/

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
# --lat_weight
# --monthly_reduction

# 4. Lancement de ton script Python
srun python vae_on_slp.py \
    --machine jean-zay-work \
    --bs 128 \
    --lr 1e-4 \
    --beta_kld 50.0 \
    --nb_members_train 80 \
    --nb_epochs 50 \
    --latent_dim 5 \
    --duree_lissage 0 \
    --winter_months 11 12 1 2 \




# 5. NETTOYAGE INDISPENSABLE
# On tue les boucles de monitoring pour que le job SLURM puisse se terminer proprement
kill $PID_GPU $PID_CPU
echo "Entraînement et monitoring terminés."