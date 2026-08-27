import optuna
import numpy as np

# Charge UNE SEULE base de données (ex: ton dernier run)
db_path = "sqlite:///stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_embedding/loocv_embedding/weird_Comparison_pca1_R2_m2_bs32_dp2_dr10.064_dr20.255_fusTrue_fc14_mult1.000_grad484.121_lossmse_lr7.4e-04_feat17_noise0.002_stratprogressive_poolmax_kx3_ky5_sstlags12_x2_y3_wd1.1e-06/Comparison_pca1_R2_m2_bs32_dp2_dr10.064_dr20.255_fusTrue_fc14_mult1.000_grad484.121_lossmse_lr7.4e-04_feat17_noise0.002_stratprogressive_poolmax_kx3_ky5_sstlags12_x2_y3_wd1.1e-06.db"
# 1. Extraction dynamique du nom de l'étude
study_summaries = optuna.get_all_study_summaries(storage=db_path)
if not study_summaries:
    raise ValueError("Aucune étude n'a été trouvée dans cette base de données !")

# On prend le nom de la première (et unique) étude
nom_dynamique = study_summaries[0].study_name
print(f"✅ Étude trouvée et chargée automatiquement : {nom_dynamique}")

# Charge l'étude
study = optuna.load_study(study_name=nom_dynamique, storage=db_path)
# Met la variance globale de ton CSV pour Février
var_global = 250.1139 ** 2 # (Remplace par la valeur exacte de ton CSV)

membres, r2_locaux, r2_globaux = [], [], []

for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        mem = trial.params.get("test_member")
        
        # On récupère les erreurs (MSE) de ce run spécifique
        mse_list = trial.user_attrs.get("mse_per_sample")
        if mse_list is None: continue
        
        mse_array = np.array(mse_list)
        mse_moyen = np.mean(mse_array)
        
        # 1. Calcul du VRAI R2 local (sur ces prédictions)
        # On a besoin de l'erreur absolue pour remonter à la variance
        r2_local_optuna = trial.user_attrs.get("best_test_R2")
        
        # 2. Calcul du R2 global (sur ces MEMES prédictions)
        r2_global_retro = 1 - (mse_moyen / var_global)
        
        membres.append(mem)
        r2_locaux.append(r2_local_optuna)
        r2_globaux.append(r2_global_retro)

# Calcul de la corrélation
ratios_effectifs = [(1 - g) / (1 - l) for g, l in zip(r2_globaux, r2_locaux)]
print("Ratios effectifs (1 - R2_global) / (1 - R2_local):", ratios_effectifs)
# Tu peux imprimer ou plotter "ratios_effectifs" et comparer avec ton CSV !