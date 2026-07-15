#!/bin/bash
#SBATCH --job-name=optuna_plots      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/optuna/log/optunaplots_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/optuna/log/optunaplots_%j.out
#SBATCH -A uxg@cpu                     # Allocation du projet (uxg) sur les heures CPU
#SBATCH --partition=prepost
#SBATCH --time=1:00:00                  # Temps d'exécution 
#SBATCH --nodes=1                        # Un seul nœud de calcul
#SBATCH --ntasks=1                       # Une seule tâche
#SBATCH --cpus-per-task=10               # 10 cœurs pour accélérer Numpy/Scikit-learn
#SBATCH --hint=nomultithread             # Hyperthreading désactivé

# 1. Nettoyage de l'environnement
module purge

# 2. Chargement du module Python/PyTorch
# (On charge le même module que d'habitude pour avoir tes paquets, il marchera très bien sur CPU)
module load arch/h100
module load pytorch-gpu/py3/2.5.1

# 3. Activation de ton environnement local (pour cftime, netCDF4, eofs, cartopy)
export PYTHONUSERBASE=$WORK/.local_vit_v100

# 4. Déplacement dans ton dossier de travail
# (À adapter avec le nom exact du dossier où se trouve ton script python)
cd $WORK/stage_isir_jz/vision_transformer/vit_with_slp_embedding/optuna

# 5. Exécution du code
srun python optuna_loocv_vit_plot.py \
    --storage-name stage_isir_jz/vision_transformer/vit_with_slp_embedding/optuna/loocv/LOOCV_metric_R2_pca1_months2_epochs20_sst23456slp_pat5intra15bs64dr0.4attFalselr5e-05rollTruetest1001.001lossmsedepth5head216mlp6.0actrelunormfirstFalsepoolclsheadactrelubottleneckFalseweightdecay0.0_patchsst17x15/vit_optuna.db