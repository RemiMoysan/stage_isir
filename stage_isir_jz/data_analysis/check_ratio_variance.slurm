#!/bin/bash
#SBATCH --job-name=visu_rapport      # Nom du job
#SBATCH --output=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/var_ratio_%j.out          # Fichier de sortie standard (%j = numéro du job)
#SBATCH --error=/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/log/var_ratio_%j.out           # Fichier d'erreur (fusionné avec la sortie)
#SBATCH -A uxg@cpu                     # Allocation du projet (uxg) sur les heures CPU
#SBATCH --partition=prepost
#SBATCH --time=00:10:00                  # Temps d'exécution 
#SBATCH --nodes=1                        # Un seul nœud de calcul
#SBATCH --ntasks=1                       # Une seule tâche
#SBATCH --cpus-per-task=10               # 10 cœurs pour accélérer Numpy/Scikit-learn
#SBATCH --hint=nomultithread             # Hyperthreading désactivé

# 1. Nettoyage de l'environnement
module purge

# 2. Chargement du module Python/PyTorch
# (On charge le même module que d'habitude pour avoir tes paquets, il marchera très bien sur CPU)
module load pytorch-gpu/py3/2.0.0

# 3. Activation de ton environnement local (pour cftime, netCDF4, eofs, cartopy)
export PYTHONUSERBASE=$WORK/.local_vit_v100

# 4. Déplacement dans ton dossier de travail
# (À adapter avec le nom exact du dossier où se trouve ton script python)
cd $WORK/stage_isir_jz/data_analysis/


# 5. Exécution du code
srun python check_rapport_var_nan_ou_0_sst.py --file_sst /lustre/fswork/projects/rech/uxg/uca57ub/data/SST/SST_anom_LE2-1001.001_T_regrid_1mo.nc
    