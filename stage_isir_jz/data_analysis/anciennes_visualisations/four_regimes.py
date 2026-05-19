import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from datetime import timedelta
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import argparse

# ============================================================
# ARGUMENTS & PATHS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
args = parser.parse_args()

# parameters
number_of_members = 89
sst_lags = [35, 65, 95, 140, 175, 210, 245, 280, 315, 350] 
slp_lags = [15, 30, 45, 60]         

if args.machine == 'hacienda':
    base_home = f"/home/moysan/stage_isir_jz/data_analysis/four_regimes_result_{number_of_members}_members_lags_{'_'.join(map(str, sst_lags))}_sst_{'_'.join(map(str, slp_lags))}_slp/"
    data_dir = "/data/moysan/data/"
elif args.machine == 'jean-zay-work': 
    base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/four_regimes_result_{number_of_members}_members_lags_{'_'.join(map(str, sst_lags))}_sst_{'_'.join(map(str, slp_lags))}_slp/" 
    data_dir = "/lustre/fswork/projects/rech/uxg/uca57ub/data/"
elif args.machine == 'jean-zay-scratch':
    base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/four_regimes_result_{number_of_members}_members_lags_{'_'.join(map(str, sst_lags))}_sst_{'_'.join(map(str, slp_lags))}_slp/" 
    data_dir = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/"

os.makedirs(base_home, exist_ok=True)
path_sst = os.path.join(data_dir, "SST/")
path_slp = os.path.join(data_dir, "SLP/")

# ==========================================
# 1. DÉFINITION DU MODÈLE VAE
# ==========================================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.flatten_size = 64 * 7 * 15 
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 64, 7, 15)
        x = self.decoder(x)
        return F.interpolate(x, size=(53, 113), mode='bilinear', align_corners=False)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z), mu, logvar

# ==========================================
# 2. FONCTION PRINCIPALE DE CLUSTERING
# ==========================================
def extract_weather_regimes_and_precursors(member_ids, sst_lags=[], slp_lags=[], n_clusters=4, 
                                           embedding_method='pca', vae_model_path=None,
                                           path_slp="/data/moysan/data/SLP/",
                                           path_sst="/data/moysan/data/SST/", n_samples=10):

    slp_std, sst_std = 596.0, 0.707
    vmax_plot = 2.

    if isinstance(member_ids, str):
        member_ids = [member_ids]
        
    print(f"--- Recherche de {n_clusters} Régimes sur le membre {member_ids[0]} ---")
    
    all_slp_data = []
    all_dates = []
    
    # Pré-chargement des datasets complets (pour pouvoir chercher les lags dans le passé)
    sst_datasets = {}
    slp_full_datasets = {}
    
    for mem in member_ids:
        # Chargement SLP et filtrage NDJF pour le clustering
        file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}_10d.nc')
        ds_slp = xr.open_dataset(file_slp)
        ds_slp_winter = ds_slp.sel(time=ds_slp['time'].dt.month.isin([11, 12, 1, 2]))
        
        all_dates.extend([(mem, t) for t in ds_slp_winter.time.values])
        
        slp_array = ds_slp_winter['PSL'].values / slp_std
        all_slp_data.append(np.nan_to_num(slp_array, nan=0.0))
        
        # On garde les datasets complets ouverts pour les lags hors NDJF
        slp_full_datasets[mem] = ds_slp
        
        ds_sst_raw = xr.open_dataset(os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid.nc'))
        ds_sst_raw = ds_sst_raw.assign_coords(lon=(((ds_sst_raw.lon + 180) % 360) - 180)).sortby('lon')
        ds_sst_raw = ds_sst_raw.sel(lat = slice(-15,70))
        sst_datasets[mem] = ds_sst_raw

    slp_data_full = np.concatenate(all_slp_data, axis=0)
    N, H, W = slp_data_full.shape
    slp_flat = slp_data_full.reshape(N, H * W)
    
    # ==========================================
    # EMBEDDING & K-MEANS
    # ==========================================
    if embedding_method == 'raw':
        X_encoded = slp_flat
    elif embedding_method == 'pca':
        pca = PCA(n_components=10, random_state=42)
        X_encoded = pca.fit_transform(slp_flat)
    elif embedding_method == 'vae':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ConvVAE(latent_dim=128).to(device)
        if vae_model_path and os.path.exists(vae_model_path):
            model.load_state_dict(torch.load(vae_model_path, map_location=device))
        model.eval()
        slp_tensor = torch.tensor(slp_data_full).unsqueeze(1).float()
        encoded_list = []
        with torch.no_grad():
            for i in range(0, N, 512):
                batch_slp = slp_tensor[i:i+512].to(device)
                mu, _ = model.encode(batch_slp)
                encoded_list.append(mu.cpu().numpy())
        X_encoded = np.concatenate(encoded_list, axis=0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_encoded)
    
    # ==========================================
    # TRI DES CLUSTERS PAR FRÉQUENCE (NAO+, AR, SBL, NAO-)
    # ==========================================
    cluster_counts_raw = [(k, np.sum(labels == k)) for k in range(n_clusters)]
    cluster_counts_raw.sort(key=lambda x: x[1], reverse=True) # Tri décroissant
    
    sorted_clusters = [x[0] for x in cluster_counts_raw]
    sorted_counts = [x[1] for x in cluster_counts_raw]
    
    # Noms théoriques supposés d'après la fréquence
    regime_names = ["NAO+", "Atlantic Ridge (AR)", "Scandinavian Blocking (S Bl)", "NAO-"]

    # ==========================================
    # EXTRACTION DES COMPOSITES (Dynamique selon les lags)
    # ==========================================
    composites = {k: {} for k in sorted_clusters}
    global_means = {}

    # 1. Base SLP (t=0)
    for k in sorted_clusters:
        idx_cluster = np.where(labels == k)[0]
        composites[k]['slp_0'] = np.mean(slp_data_full[idx_cluster], axis=0)
    global_means['slp_0'] = np.mean(slp_data_full, axis=0)

    # 2. Fonction utilitaire pour extraire les lags
    def extract_lagged_composite(lags_list, var_name, datasets_dict, std_factor):
        for lag in lags_list:
            key = f'{var_name}_lag_{lag}'
            global_maps = []
            for k in sorted_clusters:
                idx_cluster = np.where(labels == k)[0]
                cluster_maps = []
                for idx in idx_cluster:
                    mem, target_date = all_dates[idx]
                    lagged_date = target_date - timedelta(days=lag)
                    try:
                        var_key = 'SST' if var_name == 'sst' else 'PSL'
                        val_map = datasets_dict[mem][var_key].sel(time=lagged_date, method='nearest').values / std_factor
                        cluster_maps.append(val_map)
                        global_maps.append(val_map)
                    except KeyError:
                        pass
                composites[k][key] = np.nanmean(cluster_maps, axis=0) if cluster_maps else np.zeros((85, 360) if var_name=='sst' else (53, 113))
            global_means[key] = np.nanmean(global_maps, axis=0) if global_maps else np.zeros((85, 360) if var_name=='sst' else (53, 113))

    # Extraction
    extract_lagged_composite(slp_lags, 'slp', slp_full_datasets, slp_std)
    extract_lagged_composite(sst_lags, 'sst', sst_datasets, sst_std)

    # ==========================================
    # SAUVEGARDE DES MATRICES (Pour moyenne multi-membres)
    # ==========================================
    save_dict = {}
    for i, k in enumerate(sorted_clusters):
        prefix = f"regime_{i+1}_{regime_names[i].split()[0]}"
        save_dict[f"{prefix}_count"] = sorted_counts[i]
        save_dict[f"{prefix}_slp_0"] = composites[k]['slp_0']
        for lag in slp_lags:
            save_dict[f"{prefix}_slp_lag_{lag}"] = composites[k][f'slp_lag_{lag}']
        for lag in sst_lags:
            save_dict[f"{prefix}_sst_lag_{lag}"] = composites[k][f'sst_lag_{lag}']
    
    for key, val in global_means.items():
        save_dict[f"global_mean_{key}"] = val

    save_path = f"{base_home}/Composites_Mem{member_ids[0]}_{embedding_method}.npz"
    np.savez(save_path, **save_dict)
    print(f"Matrices de composites sauvegardées dans : {save_path}")

    # ==========================================
    # VISUALISATION DES RÉSULTATS (Grille dynamique)
    # ==========================================
    n_cols = 1 + len(slp_lags) + len(sst_lags)
    fig, axes = plt.subplots(n_clusters + 1, n_cols, figsize=(5.5 * n_cols, 3.5 * (n_clusters + 1)), 
                             subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False)
    
    # CRÉATION DE LA NORME NON-LINÉAIRE
    # linthresh=0.5 étire les couleurs : 0.5 aura déjà une couleur assez prononcée !
    my_norm = mcolors.SymLogNorm(linthresh=0.5, vmin=-vmax_plot, vmax=vmax_plot, base=10)
    # On définit de jolies graduations pour la barre de couleur
    cbar_ticks = [-2, -1, -0.4, 0, 0.4, 1, 2]
    
    for i, k in enumerate(sorted_clusters):
        count = sorted_counts[i]
        pct = (count / N) * 100
        col_idx = 0
        
        # 1. SLP (t=0)
        ax = axes[i, col_idx]
        im = ax.imshow(composites[k]['slp_0'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, 0, 90], norm=my_norm)
        ax.set_title(f"{regime_names[i]}\n(N={count}, {pct:.1f}%)", fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks,format="%g")
        col_idx += 1
        
        # 2. SLP Lags
        for lag in slp_lags:
            ax = axes[i, col_idx]
            im = ax.imshow(composites[k][f'slp_lag_{lag}'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, 0, 90], norm=my_norm)
            ax.set_title(f"SLP (Lag -{lag}j)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks,format="%g")
            col_idx += 1
            
        # 3. SST Lags
        for lag in sst_lags:
            ax = axes[i, col_idx]
            im = ax.imshow(composites[k][f'sst_lag_{lag}'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, -90, 90], norm=my_norm)
            ax.set_title(f"SST (Lag -{lag}j)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks,format="%g")
            col_idx += 1

    # LIGNE DES MOYENNES GLOBALES
    col_idx = 0
    
    ax = axes[n_clusters, col_idx]
    im = ax.imshow(global_means['slp_0'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, 0, 90], norm=my_norm)
    ax.set_title("MOYENNE GLOBALE (SLP)", fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks,format="%g")
    col_idx += 1
    
    for lag in slp_lags:
        ax = axes[n_clusters, col_idx]
        im = ax.imshow(global_means[f'slp_lag_{lag}'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, 0, 90], norm=my_norm)
        ax.set_title(f"Moy. Globale SLP (-{lag}j)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks,format="%g")
        col_idx += 1
        
    for lag in sst_lags:
        ax = axes[n_clusters, col_idx]
        im = ax.imshow(global_means[f'sst_lag_{lag}'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, -90, 90], norm=my_norm)
        ax.set_title(f"Moy. Globale SST (-{lag}j)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks,format="%g")
        col_idx += 1

    plt.suptitle(f"Weather Regimes sur le membre {member_ids[0]}", fontsize=18, y=1.02)
    plt.tight_layout()
    fig_name = f"{base_home}/Weather_Regimes_Mem{member_ids[0]}_{embedding_method}.png"
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure principale sauvegardée : {fig_name}")

    # Fermeture propre
    for ds in sst_datasets.values(): ds.close()
    for ds in slp_full_datasets.values(): ds.close()
        
    print("Terminé !")

## APPEL DE LA FONCTION
# Il y a bien les tous les 89 membres ci dessous.
all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
members_used = all_members[:number_of_members]
start_time = time.time()
rang = 0
for test_member in members_used:
    rang += 1
    print(f"\n=== TRAITEMENT DU MEMBRE {test_member} (Range {rang}/{number_of_members}) ===")
    extract_weather_regimes_and_precursors(
        member_ids=test_member, 
        sst_lags=sst_lags, # <-- AJOUTE AUTANT DE LAGS QUE TU VEUX
        slp_lags=slp_lags,         # <-- AJOUTE DES LAGS SLP ICI
        n_clusters=4, 
        path_slp=path_slp,
        path_sst=path_sst,
        embedding_method='pca'
    )
    current_time = time.time()
    elapsed_time = current_time - start_time
    print(f"Temps écoulé pour le {rang}-ème membre : {elapsed_time:.2f} secondes")