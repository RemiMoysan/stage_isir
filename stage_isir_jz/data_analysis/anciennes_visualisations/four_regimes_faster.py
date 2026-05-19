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
import joblib # Pour charger le PCA
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
parser.add_argument('--embedding_method', type=str, default='pca', choices=['pca', 'vae'])
parser.add_argument('--model_path', type=str, required=True, help="Chemin vers le modèle PCA (.pkl) ou VAE (.pth) pré-entraîné")
args = parser.parse_args()

# parameters
number_of_members = 89
sst_lags = [35, 65, 95, 140, 175, 210, 245, 280, 315, 350] 
slp_lags = [15, 30, 45, 60]      
duree_lissage = 10    

if args.machine == 'hacienda':
    base_home = f"/home/moysan/stage_isir_jz/data_analysis/master_ref_generator_{number_of_members}_members_{duree_lissage}d_embedding_method_{args.embedding_method}/"
    data_dir = "/data/moysan/data/"
elif args.machine == 'jean-zay-work': 
    base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/master_ref_generator_{number_of_members}_members_{duree_lissage}d_embedding_method_{args.embedding_method}/"
    data_dir = "/lustre/fswork/projects/rech/uxg/uca57ub/data/"
elif args.machine == 'jean-zay-scratch':
    base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/master_ref_generator_{number_of_members}_members_{duree_lissage}d_embedding_method_{args.embedding_method}/"
    data_dir = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/"

os.makedirs(base_home, exist_ok=True)
path_sst = os.path.join(data_dir, "SST/")
path_slp = os.path.join(data_dir, "SLP/")

# ==========================================
# 1. DÉFINITION DU MODÈLE VAE (Si utilisé)
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

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

def load_projector(method, path):
    print(f"Chargement du projecteur pré-entraîné ({method})...")
    if method == 'pca':
        return joblib.load(path)
    elif method == 'vae':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ConvVAE(latent_dim=128).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

# ==========================================
# 2. GÉNÉRATION DE LA MASTER REFERENCE
# ==========================================
def generate_master_reference(member_ids, sst_lags, slp_lags, n_clusters=4):
    slp_std, sst_std = 596.0, 0.707
    vmax_plot = 2.

    projector = load_projector(args.embedding_method, args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n=== PASSE 1 : Extraction des Embeddings (Faible RAM) ===")
    all_latents = []
    all_dates = [] # Stocke (member_id, date)
    
    start_time = time.time()
    for mem in member_ids:
        file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}_{duree_lissage}d.nc')
        try:
            with xr.open_dataset(file_slp) as ds_slp:
                ds_winter = ds_slp.sel(time=ds_slp['time'].dt.month.isin([11, 12, 1, 2]))
                dates = ds_winter.time.values
                all_dates.extend([(mem, t) for t in dates])
                
                slp_array = ds_winter['PSL'].values / slp_std
                slp_array = np.nan_to_num(slp_array, nan=0.0)
                
                # Projection
                if args.embedding_method == 'pca':
                    slp_flat = slp_array.reshape(len(dates), -1)
                    latent = projector.transform(slp_flat)
                elif args.embedding_method == 'vae':
                    slp_tensor = torch.tensor(slp_array).unsqueeze(1).float()
                    encoded_list = []
                    with torch.no_grad():
                        for i in range(0, len(dates), 512):
                            batch = slp_tensor[i:i+512].to(device)
                            mu, _ = projector.encode(batch)
                            encoded_list.append(mu.cpu().numpy())
                    latent = np.concatenate(encoded_list, axis=0)
                
                all_latents.append(latent)
                print(f"Membre {mem} projeté.", end='\r')
        except FileNotFoundError:
            print(f"Fichier manquant pour {mem}, ignoré.")
            
    print(f"\nPasse 1 terminée en {time.time() - start_time:.2f}s")
    
    # K-Means Global
    global_latents = np.vstack(all_latents)
    print(f"Lancement du K-Means sur {global_latents.shape[0]} jours (dimension {global_latents.shape[1]})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    global_labels = kmeans.fit_predict(global_latents)
    
    # Tri des clusters par fréquence
    counts = [(k, np.sum(global_labels == k)) for k in range(n_clusters)]
    counts.sort(key=lambda x: x[1], reverse=True)
    sorted_clusters = [x[0] for x in counts]
    sorted_counts = [x[1] for x in counts]
    
    # EXTRACTION DES 4 EMBEDDINGS DE RÉFÉRENCE (La priorité !)
    ref_centroids = kmeans.cluster_centers_[sorted_clusters]
    
    print("\n=== PASSE 2 : Accumulation Spatiale (Vectorisée & Lazy Loading) ===")
    sums = {k: {'slp_0': 0} for k in sorted_clusters}
    valid_counts = {k: {'slp_0': 0} for k in sorted_clusters}
    
    for k in sorted_clusters:
        for lag in slp_lags:
            sums[k][f'slp_lag_{lag}'] = 0; valid_counts[k][f'slp_lag_{lag}'] = 0
        for lag in sst_lags:
            sums[k][f'sst_lag_{lag}'] = 0; valid_counts[k][f'sst_lag_{lag}'] = 0
            
    date_to_label = {d: l for d, l in zip(all_dates, global_labels)}
    
    start_time = time.time()
    for mem in member_ids:
        file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}_{duree_lissage}d.nc')
        file_sst = os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid.nc')
        
        if not os.path.exists(file_slp) or not os.path.exists(file_sst):
            continue
            
        with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
            ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15,70))
            
            # On récupère toutes les dates d'intérêt pour ce membre
            mem_dates = [t for m, t in all_dates if m == mem]
            
            # --- LA MAGIE DE LA VECTORISATION ---
            for cluster_id in sorted_clusters:
                # 1. On regroupe toutes les dates qui appartiennent à ce cluster pour ce membre
                dates_for_cluster = [d for d in mem_dates if date_to_label[(mem, d)] == cluster_id]
                
                if not dates_for_cluster:
                    continue
                
                # 2. SLP t=0 (On extrait TOUTES les cartes du cluster d'un coup et on les somme)
                # On utilise intersect1d pour éviter les erreurs si une date déborde
                valid_dates = np.intersect1d(dates_for_cluster, ds_slp.time.values)
                if len(valid_dates) > 0:
                    slp_0_sum = ds_slp['PSL'].sel(time=valid_dates).sum(dim='time').values / slp_std
                    sums[cluster_id]['slp_0'] += np.nan_to_num(slp_0_sum, nan=0.0)
                    valid_counts[cluster_id]['slp_0'] += len(valid_dates)
                
                # 3. Lags SLP vectorisés
                for lag in slp_lags:
                    lagged_dates = [d - timedelta(days=lag) for d in dates_for_cluster]
                    valid_lagged = np.intersect1d(lagged_dates, ds_slp.time.values)
                    if len(valid_lagged) > 0:
                        slp_lag_sum = ds_slp['PSL'].sel(time=valid_lagged).sum(dim='time').values / slp_std
                        sums[cluster_id][f'slp_lag_{lag}'] += np.nan_to_num(slp_lag_sum, nan=0.0)
                        valid_counts[cluster_id][f'slp_lag_{lag}'] += len(valid_lagged)
                
                # 4. Lags SST vectorisés
                for lag in sst_lags:
                    lagged_dates = [d - timedelta(days=lag) for d in dates_for_cluster]
                    valid_lagged = np.intersect1d(lagged_dates, ds_sst.time.values)
                    if len(valid_lagged) > 0:
                        sst_lag_sum = ds_sst['SST'].sel(time=valid_lagged).sum(dim='time').values / sst_std
                        sums[cluster_id][f'sst_lag_{lag}'] += np.nan_to_num(sst_lag_sum, nan=0.0)
                        valid_counts[cluster_id][f'sst_lag_{lag}'] += len(valid_lagged)

        print(f"Cartes accumulées pour le membre {mem}.", end='\r')

    print(f"\nPasse 2 terminée en {time.time() - start_time:.2f}s")
    
    # ==========================================
    # SAUVEGARDE DU MASTER DICTIONARY
    # ==========================================
    print("\nCalcul des moyennes finales et sauvegarde...")
    master_dict = {}
    regime_names = ["NAO+", "Atlantic_Ridge", "Scand_Blocking", "NAO-"]
    
    # 1. Sauvegarde des Embeddings Latents
    master_dict["ref_centroids_latent"] = ref_centroids
    master_dict["embedding_method"] = args.embedding_method
    
    # 2. Sauvegarde des cartes composites
    composites_final = {k: {} for k in sorted_clusters}
    
    for i, k in enumerate(sorted_clusters):
        prefix = f"regime_{i+1}_{regime_names[i]}"
        master_dict[f"{prefix}_count"] = sorted_counts[i]
        
        # SLP 0
        mean_slp_0 = sums[k]['slp_0'] / max(1, valid_counts[k]['slp_0'])
        master_dict[f"{prefix}_slp_0"] = mean_slp_0
        composites_final[k]['slp_0'] = mean_slp_0
        
        # Lags
        for lag in slp_lags:
            mean_val = sums[k][f'slp_lag_{lag}'] / max(1, valid_counts[k][f'slp_lag_{lag}'])
            master_dict[f"{prefix}_slp_lag_{lag}"] = mean_val
            composites_final[k][f'slp_lag_{lag}'] = mean_val
            
        for lag in sst_lags:
            mean_val = sums[k][f'sst_lag_{lag}'] / max(1, valid_counts[k][f'sst_lag_{lag}'])
            master_dict[f"{prefix}_sst_lag_{lag}"] = mean_val
            composites_final[k][f'sst_lag_{lag}'] = mean_val

    save_path = f"{base_home}/master_reference_global.npz"
    np.savez(save_path, **master_dict)
    print(f"Master Reference sauvegardée avec succès : {save_path}")
    
    # ==========================================
    # VISUALISATION DES RÉSULTATS (Corrigée et Adaptée)
    # ==========================================
    print("\nGénération de la figure récapitulative...")
    
    # 1. Recalcul de N (Total des jours)
    N = sum(sorted_counts)
    
    # 2. Recalcul rapide des moyennes globales à partir des sommes (Zéro RAM supplémentaire)
    global_means = {}
    for var_lag in ['slp_0'] + [f'slp_lag_{l}' for l in slp_lags] + [f'sst_lag_{l}' for l in sst_lags]:
        total_sum = sum(sums[k][var_lag] for k in sorted_clusters)
        total_count = sum(valid_counts[k][var_lag] for k in sorted_clusters)
        global_means[var_lag] = total_sum / max(1, total_count)

    # 3. Tracé de la figure
    n_cols = 1 + len(slp_lags) + len(sst_lags)
    fig, axes = plt.subplots(n_clusters + 1, n_cols, figsize=(5.5 * n_cols, 3.5 * (n_clusters + 1)), 
                             subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False)
    
    my_norm = mcolors.SymLogNorm(linthresh=0.5, vmin=-vmax_plot, vmax=vmax_plot, base=10)
    cbar_ticks = [-2, -1, -0.4, 0, 0.4, 1, 2]
    
    # Lignes des clusters
    for i, k in enumerate(sorted_clusters):
        count = sorted_counts[i]
        pct = (count / N) * 100
        col_idx = 0
        
        # SLP (t=0)
        ax = axes[i, col_idx]
        im = ax.imshow(composites_final[k]['slp_0'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, 0, 90], norm=my_norm)
        ax.set_title(f"{regime_names[i]}\n(N={count}, {pct:.1f}%)", fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks, format="%g")
        col_idx += 1
        
        # SLP Lags
        for lag in slp_lags:
            ax = axes[i, col_idx]
            im = ax.imshow(composites_final[k][f'slp_lag_{lag}'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, 0, 90], norm=my_norm)
            ax.set_title(f"SLP (Lag -{lag}j)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks, format="%g")
            col_idx += 1
            
        # SST Lags
        for lag in sst_lags:
            ax = axes[i, col_idx]
            im = ax.imshow(composites_final[k][f'sst_lag_{lag}'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, -90, 90], norm=my_norm)
            ax.set_title(f"SST (Lag -{lag}j)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks, format="%g")
            col_idx += 1

    # LIGNE DES MOYENNES GLOBALES
    col_idx = 0
    
    ax = axes[n_clusters, col_idx]
    im = ax.imshow(global_means['slp_0'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, 0, 90], norm=my_norm)
    ax.set_title("MOYENNE GLOBALE (SLP)", fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks, format="%g")
    col_idx += 1
    
    for lag in slp_lags:
        ax = axes[n_clusters, col_idx]
        im = ax.imshow(global_means[f'slp_lag_{lag}'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, 0, 90], norm=my_norm)
        ax.set_title(f"Moy. Globale SLP (-{lag}j)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks, format="%g")
        col_idx += 1
        
    for lag in sst_lags:
        ax = axes[n_clusters, col_idx]
        im = ax.imshow(global_means[f'sst_lag_{lag}'], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=[-180, 180, -90, 90], norm=my_norm)
        ax.set_title(f"Moy. Globale SST (-{lag}j)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks, format="%g")
        col_idx += 1

    plt.suptitle(f"Weather Regimes sur {len(member_ids)} membres", fontsize=18, y=1.02)
    plt.tight_layout()
    fig_name = f"{base_home}/Weather_Regimes_Global_{args.embedding_method}.png"
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure principale sauvegardée : {fig_name}")

if __name__ == "__main__":
    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    members_used = all_members[:number_of_members]
    
    generate_master_reference(members_used, sst_lags, slp_lags)