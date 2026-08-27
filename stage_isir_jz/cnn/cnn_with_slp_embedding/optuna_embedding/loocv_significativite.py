import os
import optuna
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
from pathlib import Path

# ============================================================
# 1. CONFIGURATION ET CHARGEMENT OPTUNA
# ============================================================
db_path = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_embedding/loocv_embedding/Comparison_pca1_R2_m2_bs32_dp2_dr10.064_dr20.255_fusTrue_fc14_mult1.000_grad484.121_lossmse_lr7.4e-04_feat17_noise0.002_stratprogressive_poolmax_kx3_ky5_sstlags12_x2_y3_wd1.1e-06/Comparison_pca1_R2_m2_bs32_dp2_dr10.064_dr20.255_fusTrue_fc14_mult1.000_grad484.121_lossmse_lr7.4e-04_feat17_noise0.002_stratprogressive_poolmax_kx3_ky5_sstlags12_x2_y3_wd1.1e-06.db"
study_name = os.path.basename(db_path).replace('.db', '')

ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

print(f"Chargement de l'étude Optuna : {study_name}...")
study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")

# Extraction sécurisée via dictionnaire
member_to_mse = {}
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        test_member = trial.params.get("test_member")
        mse_list = trial.user_attrs.get("mse_per_sample")
        if mse_list is not None:
            member_to_mse[test_member] = mse_list

# Reconstruction de la liste plate dans le bon ordre
all_mse_ordered = []
valid_members = []
for m in ALL_MEMBERS:
    if m in member_to_mse:
        all_mse_ordered.extend(member_to_mse[m])
        valid_members.append(m)
    else:
        print(f"⚠️ Membre {m} introuvable ou non complété dans Optuna.")

n_members = len(valid_members)
n_samples_per_member = len(member_to_mse[valid_members[0]])
total_samples = len(all_mse_ordered)
all_mse_ordered = np.array(all_mse_ordered)

print(f"Extraction réussie : {n_members} membres, {n_samples_per_member} échantillons par membre ({total_samples} au total).")

project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

grand_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if grand_parent_dir not in sys.path:
    sys.path.append(grand_parent_dir)

import torch
import joblib
import xarray as xr
from torch.utils.data import DataLoader
from shared_tools.datasets import Dataset_mensuel
from shared_tools.optuna_loop_helpers import encode_to_latent_gpu

# ============================================================
# 2. CALCUL DE LA VARIANCE GLOBALE (NAO)
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paramètres extraits de ton chemin Optuna
winter_months = [2] 
latent_dim = 1
dynamic_slp_std = 467.2 
pca_path = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/monthly_reduction/lat_weightTrue/IPCA_latent128_winter_months_11_12_1_2_87members_normalizeFalse_monthly_reduction_wgtTrue_slp_std467.2/best_pca_model.joblib"

print("Chargement du modèle IPCA et des poids de latitude...")
# Chargement PCA
pca_model = joblib.load(pca_path)
pca_mean_gpu = torch.tensor(pca_model.mean_, dtype=torch.float32, device=device)
pca_components_gpu = torch.tensor(pca_model.components_[:latent_dim], dtype=torch.float32, device=device)

# Chargement Poids Latitude (sur un membre arbitraire)
sample_path = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-{ALL_MEMBERS[0]}_1mo.nc"
ds_sample = xr.open_dataset(sample_path)
lats = ds_sample['lat'].values
coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
h, w = len(lats), len(ds_sample['lon'].values)
wgts = np.sqrt(coslat).reshape(h, 1)
wgts_gpu = torch.tensor(np.broadcast_to(wgts, (h, w)).flatten(), dtype=torch.float32, device=device)
ds_sample.close()

print("Extraction des targets sur l'ensemble du dataset (tous membres)...")
# On ne met pas de lags SST/SLP car on veut juste récupérer y_target rapidement
full_dataset = Dataset_mensuel(
    members=ALL_MEMBERS, 
    selected_months=winter_months, 
    machine='jean-zay-work', 
    target_type='map', 
    sst_lags_months=[], 
    slp_lags_months=[], 
    roll_sst=False, 
    slp_std=dynamic_slp_std, 
    augment=False
)

# On peut utiliser un gros batch_size vu qu'on ne charge que la target
full_loader = DataLoader(full_dataset, batch_size=256, shuffle=False, num_workers=4)

all_nao_values = []
with torch.no_grad():
    # Attention au tuple retourné par ton Dataset (selon ta définition exacte)
    for batch in full_loader:
        # Normalement batch[2] est y_target dans ta boucle d'entraînement
        y_target = batch[2].to(device, non_blocking=True)
        
        # Application exacte de ta fonction de projection
        target_embed = encode_to_latent_gpu(
            y_target, 'pca', latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, None
        )
        all_nao_values.append(target_embed.cpu().numpy())

all_nao_values = np.concatenate(all_nao_values, axis=0)

# Puisque latent_dim=1, all_nao_values est de shape (N, 1)
var_tot = np.var(all_nao_values[:, 0])
print(f"Variance globale calculée pour la NAO : {var_tot:.4f}")

# Calcul des scores R² individuels
r2_samples = 1 - (all_mse_ordered / var_tot)
# ============================================================

# ============================================================
# 3. THÉORIE DU TCL ET SCORES RÉELS
# ============================================================
# Paramètres théoriques TCL
mu_th = np.mean(r2_samples)
sigma_th_sample = np.std(r2_samples, ddof=1)
sigma_th_mean = sigma_th_sample / np.sqrt(n_samples_per_member)

# Agrégation réelle (par vrais membres)
real_member_r2 = []
for i in range(n_members):
    member_scores = r2_samples[i * n_samples_per_member : (i + 1) * n_samples_per_member]
    real_member_r2.append(np.mean(member_scores))

real_std = np.std(real_member_r2, ddof=1)

# ============================================================
# 4. TEST DE PERMUTATION STRATIFIÉ (À DATE FIXÉE)
# ============================================================
n_permutations = 5000
permuted_stds_stratified = []
all_fake_member_r2 = [] # NOUVEAU : on va stocker TOUS les scores de toutes les permutations

r2_matrix = r2_samples.reshape(n_members, n_samples_per_member)

print(f"Lancement de {n_permutations} permutations à date fixée...")
for p in range(n_permutations):
    rand_indices = np.random.rand(n_members, n_samples_per_member).argsort(axis=0)
    shuffled_matrix = np.take_along_axis(r2_matrix, rand_indices, axis=0)
    
    fake_member_r2 = np.mean(shuffled_matrix, axis=1)
    
    permuted_stds_stratified.append(np.std(fake_member_r2, ddof=1))
    
    # On accumule tous les scores pour avoir un histogramme orange parfaitement lisse
    all_fake_member_r2.extend(fake_member_r2)

p_value_stratified = np.sum(np.array(permuted_stds_stratified) >= real_std) / n_permutations

# ============================================================
# 5. AFFICHAGE ET SAUVEGARDE DES RÉSULTATS
# ============================================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# --- GRAPHE 1 : Distributions des R² ---
ax1 = plt.subplot(1, 2, 1)
# On trace la réalité (89 valeurs, donc peu de bins)
ax1.hist(real_member_r2, bins=15, alpha=0.6, color='dodgerblue', density=True, label='Vrais Membres (LOOCV)')
# On trace TOUTES les permutations (445 000 valeurs, donc on peut mettre beaucoup de bins pour que ce soit lisse)
ax1.hist(all_fake_member_r2, bins=80, alpha=0.5, color='darkorange', density=True, label=f'Toutes les Permutations (n = {n_permutations})')

# Courbe TCL
x_axis = np.linspace(mu_th - 4*sigma_th_mean, mu_th + 4*sigma_th_mean, 1000)
ax1.plot(x_axis, norm.pdf(x_axis, mu_th, sigma_th_mean), color='black', linewidth=2, linestyle='--', 
         label=f'TCL Théorique $\\mathcal{{N}}(\\mu, \\sigma^2/164)$ \n $\\mu$ = {mu_th:.3f}, $\\sigma$ = {sigma_th_sample:.3f}')
ax1.set_title("Distribution du score R² sur l'échantillon réelle des 89 membres et pour un membre théorique")
ax1.set_xlabel("Score R² Moyen")
ax1.set_ylabel("Densité")

# On augmente la hauteur pour la légende
y_min, y_max = ax1.get_ylim()
ax1.set_ylim(y_min, y_max * 1.3)
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# --- GRAPHE 2 : Analyse de la Dispersion Inter-membres ---
ax2 = plt.subplot(1, 2, 2)
counts, bins, patches = ax2.hist(permuted_stds_stratified, bins=40, alpha=0.7, color='grey', label=f'Dispersion des Permutations (n={n_permutations})')

ax2.axvline(sigma_th_mean, color='black', linestyle='--', linewidth=2, label=f'Écart-type TCL ({sigma_th_mean:.4f})')
ax2.axvline(real_std, color='crimson', linestyle='-', linewidth=2.5, label=f'Écart-type Réel (p-value = {p_value_stratified:.4f})')

ax2.set_title("La variabilité entre les 89 membres est-elle physique ?")
ax2.set_xlabel("Écart-type des scores R²")
ax2.set_ylabel("Fréquence")

# On augmente drastiquement la hauteur (plafond) de l'axe Y pour faire de la place à la légende
y_min, y_max = ax2.get_ylim()
ax2.set_ylim(y_min, y_max * 1.35)
ax2.legend(loc='upper right')
ax2.grid(alpha=0.3)

plt.tight_layout()

# --- SAUVEGARDE DE L'IMAGE ---
plot_path_png = db_path.replace('.db', '_analysis_TCL_permutations2.png')
plot_path_pdf = db_path.replace('.db', '_analysis_TCL_permutations2.pdf')

plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
plt.savefig(plot_path_pdf, bbox_inches='tight')

print(f"\n✅ Graphiques sauvegardés avec succès dans :")
print(f" -> {plot_path_png}")
print(f" -> {plot_path_pdf}")

plt.close()

print("\n--- RÉSULTATS FINAUX ---")
print(f"Écart-type attendu selon le TCL : {sigma_th_mean:.5f}")
print(f"Écart-type observé (vrais membres): {real_std:.5f}")
print(f"P-value (probabilité que l'écart-type réel soit un artefact) : {p_value_stratified:.5f}")