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

import matplotlib
matplotlib.use('Agg')
import argparse


# Attention à la RAM ce code il vaut mieux de toute façon faire membre par membre pour comparer je pense. 
# Il manque le slice de la SST (-15,70) par rapport à ce que donne Dataset dans ce code (c'est genant si on veut utiliser ces cartes de ref comme composite, mais de toute façon ce code sert juste à la visualisation)

parser = argparse.ArgumentParser()
parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'], help='Machine sur laquelle le code tourne (adapte les chemins automatiquement)')
args = parser.parse_args()

# Routage dynamique du dossier de sortie et du dossier de données en fonction de la machine
if args.machine == 'hacienda':
    base_home = "/home/moysan/stage_isir_jz/data_analysis/four_regimes_result/"
    data_dir = "/data/moysan/data/"
elif args.machine == 'jean-zay-work': 
    # WORK_uxg=/lustre/fswork/projects/rech/uxg/uca57ub
    base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/four_regimes_result" 
    data_dir = "/lustre/fswork/projects/rech/uxg/uca57ub/data/"
elif args.machine == 'jean-zay-scratch':
    # SCRATCH_uxg=/lustre/fsn1/projects/rech/uxg/uca57ub
    base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/four_regimes_result" 
    data_dir = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/"
else:
    raise ValueError("Machine argument must be 'hacienda', 'jean-zay-work' or 'jean-zay-scratch'.")

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
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # La taille 6720 dépend de la taille d'entrée SLP (53x113)
        self.flatten_size = 64 * 7 * 15 
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 64, 7, 15)
        x = self.decoder(x)
        x = F.interpolate(x, size=(53, 113), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z), mu, logvar

# ==========================================
# 2. FONCTION PRINCIPALE DE CLUSTERING
# ==========================================
def extract_weather_regimes_and_precursors(member_ids, lag=35, n_clusters=4, 
                                           embedding_method='pca', vae_model_path=None,
                                           path_slp="/data/moysan/data/SLP/",
                                           path_sst="/data/moysan/data/SST/", n_samples = 10):
    """
    Extrait les 4 régimes de SLP d'hiver (NDJF) sur un ou plusieurs membres 
    et trace les anomalies SST associées à -lag jours.
    """

    slp_std, sst_std = 596.0, 0.707
    vmax_plot = 2.

    # Si on donne un seul membre (string), on le transforme en liste
    if isinstance(member_ids, str):
        member_ids = [member_ids]
        
    print(f"--- Recherche de {n_clusters} Régimes sur {len(member_ids)} membre(s) ---")
    
    # Listes pour accumuler les données de tous les membres
    all_slp_data = []
    all_dates = []
    
    # 1. CHARGEMENT ET FILTRAGE DES DONNÉES (NDJF) POUR TOUS LES MEMBRES
    for mem in member_ids:
        file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}_10d.nc')
        ds_slp = xr.open_dataset(file_slp)
        
        # Filtrer la SLP pour Novembre, Décembre, Janvier, Février
        ds_slp_winter = ds_slp.sel(time=ds_slp['time'].dt.month.isin([11, 12, 1, 2]))
        
        # On stocke les dates avec une info sur le membre (pour la SST plus tard)
        # On va créer un tuple (member_id, date)
        dates_with_mem = [(mem, t) for t in ds_slp_winter.time.values]
        all_dates.extend(dates_with_mem)
        
        # Stockage des matrices
        slp_array = ds_slp_winter['PSL'].values / slp_std
        slp_array = np.nan_to_num(slp_array, nan=0.0)
        all_slp_data.append(slp_array)
        ds_slp.close()

    # Concaténation de tout le volume de données (N_total, H, W)
    slp_data_full = np.concatenate(all_slp_data, axis=0)
    N, H, W = slp_data_full.shape
    slp_flat = slp_data_full.reshape(N, H * W)
    
    print(f"Volume total analysé : {N} jours (Hiver NDJF).")

    # --- CALCUL DE LA RÉFÉRENCE GLOBALE (MOYENNE DU MEMBRE) ---
    global_mean_slp = np.mean(slp_data_full, axis=0)

    # ==========================================
    # 2. EMBEDDING (Réduction de dimension)
    # ==========================================
    print(f"Méthode d'embedding : {embedding_method.upper()}")
    
    if embedding_method == 'raw':
        X_encoded = slp_flat
        
    elif embedding_method == 'pca':
        pca = PCA(n_components=10, random_state=42)
        X_encoded = pca.fit_transform(slp_flat)
        print(f"Variance expliquée (10 PCs) : {pca.explained_variance_ratio_.sum()*100:.1f}%")
        
    elif embedding_method == 'vae':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ConvVAE(latent_dim=128).to(device)
        
        if vae_model_path and os.path.exists(vae_model_path):
            model.load_state_dict(torch.load(vae_model_path, map_location=device))
            print(f"Poids du VAE chargés depuis : {vae_model_path}")
        else:
            print("⚠️ Attention : Aucun poids VAE trouvé, utilisation d'un VAE non entraîné (aléatoire).")
            
        model.eval()
        
        # Le VAE a besoin de la forme (Batch, Channels, H, W)
        slp_tensor = torch.tensor(slp_data_full).unsqueeze(1).float()
        
        # Traitement par batchs pour éviter un OOM si on a beaucoup de membres
        batch_size = 512
        encoded_list = []
        
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch_slp = slp_tensor[i:i+batch_size].to(device)
                # On utilise 'mu' (la moyenne de la distribution latente) comme embedding
                mu, _ = model.encode(batch_slp)
                encoded_list.append(mu.cpu().numpy())
                
        X_encoded = np.concatenate(encoded_list, axis=0)
        print(f"VAE Embedding généré. Shape: {X_encoded.shape}")

    else:
        raise ValueError("embedding_method doit être 'raw', 'pca', ou 'vae'")

    # ==========================================
    # 3. CLUSTERING (K-Means)
    # ==========================================
    print("Calcul du K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_encoded)
    
    # ==========================================
    # 4. CRÉATION DES COMPOSITES (Moyennes)
    # ==========================================
    # Pour calculer les moyennes, on va regrouper les données brutes par cluster
    composites_slp = []
    composites_sst = []
    cluster_counts = []
    
    # Pré-chargement SST avec réordonnancement de la longitude
    sst_datasets = {}
    for mem in member_ids:
        ds_sst_raw = xr.open_dataset(os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid.nc'))
        # Centrage sur l'Atlantique
        ds_sst_raw = ds_sst_raw.assign_coords(lon=(((ds_sst_raw.lon + 180) % 360) - 180)).sortby('lon')
        sst_datasets[mem] = ds_sst_raw
    
    all_sst_maps = []

    for k in range(n_clusters):
        idx_cluster = np.where(labels == k)[0]
        cluster_counts.append(len(idx_cluster))
        
        # --- Composite SLP ---
        # Moyenne numpy (très rapide) des cartes brutes SLP qui appartiennent à ce cluster
        comp_slp = np.mean(slp_data_full[idx_cluster], axis=0)
        composites_slp.append(comp_slp)
        
        # --- Composite SST (Précurseurs) ---
        sst_cluster_maps = []
        for idx in idx_cluster:
            mem, target_date = all_dates[idx]
            # On décale la date pour chercher le précurseur
            lagged_date = target_date - timedelta(days=lag)
            
            # Extraction de la carte SST correspondante (on gère les cas où la date décalée sort du dataset)
            try:
                sst_map = sst_datasets[mem]['SST'].sel(time=lagged_date, method='nearest').values / sst_std
                sst_cluster_maps.append(sst_map)
                all_sst_maps.append(sst_map)
            except KeyError:
                continue
                
        if len(sst_cluster_maps) > 0:
            comp_sst = np.nanmean(sst_cluster_maps, axis=0)
        else:
            comp_sst = np.zeros((85, 360)) # Fallback si toutes les dates sortent (peu probable)
            
        composites_sst.append(comp_sst)

    global_mean_sst = np.nanmean(all_sst_maps, axis=0) if all_sst_maps else np.zeros((85,360))

    # ==========================================
    # 5. VISUALISATION DES RÉSULTATS
    # ==========================================
    print("Génération de la figure...")
    fig, axes = plt.subplots(n_clusters+1, 2, figsize=(12, 3.5 * (n_clusters+1)), 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    regime_names = [f"Regime {k+1}" for k in range(n_clusters)]
    
    for k in range(n_clusters):

        count = cluster_counts[k]
        percentage = (count / N) * 100
        # --- Plot SLP ---
        ax_slp = axes[k, 0]
        comp_slp = composites_slp[k]
        # Création d'un DataArray temporaire pour utiliser la fonction plot de xarray avec cartopy
        da_slp = xr.DataArray(comp_slp, dims=['lat', 'lon'])
        
        im1 = ax_slp.imshow(comp_slp, transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                            origin='lower', vmin=-vmax_plot, vmax=vmax_plot, extent=[-180, 180, 0, 90]) # Extent à ajuster selon ta vraie grille SLP
        #ax_slp.coastlines()
        ax_slp.set_title(f"{regime_names[k]} : SLP (N={cluster_counts[k]}, {percentage:.1f}%)")
        fig.colorbar(im1, ax=ax_slp, fraction=0.046, pad=0.04)

        # --- Plot SST ---
        ax_sst = axes[k, 1]
        comp_sst = composites_sst[k]
        
        im2 = ax_sst.imshow(comp_sst, transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                            origin='lower', vmin=-vmax_plot, vmax=vmax_plot, extent=[-180, 180, -90, 90]) # Extent à ajuster selon ta grille SST
        #ax_sst.coastlines()
        ax_sst.set_title(f"SST Précurseur (Lag: -{lag}j)")
        fig.colorbar(im2, ax=ax_sst, fraction=0.046, pad=0.04)

    # --- DERNIÈRE LIGNE : RÉFÉRENCE GLOBALE ---
    ax_ref_slp = axes[n_clusters, 0]
    vmax_ref_slp = np.abs(global_mean_slp).max()
    im_ref1 = ax_ref_slp.imshow(global_mean_slp, transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                                origin='lower', vmin=-vmax_plot, vmax=vmax_plot, extent=[-180, 180, 0, 90])
    ax_ref_slp.set_title("MOYENNE GLOBALE DU MEMBRE (SLP)", fontweight='bold')
    fig.colorbar(im_ref1, ax=ax_ref_slp, fraction=0.046, pad=0.04)

    ax_ref_sst = axes[n_clusters, 1]
    vmax_ref_sst = np.nanmax(np.abs(global_mean_sst))
    im_ref2 = ax_ref_sst.imshow(global_mean_sst, transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                                origin='lower', vmin=-vmax_plot, vmax=vmax_plot, extent=[-180, 180, -90, 90])
    ax_ref_sst.set_title(f"MOYENNE GLOBALE SST (Lag -{lag}j)", fontweight='bold')
    fig.colorbar(im_ref2, ax=ax_ref_sst, fraction=0.046, pad=0.04)

    plt.suptitle(f"Weather Regimes et Forçages Océaniques ({len(member_ids)} Membres)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{base_home}/Weather_Regimes_Lag{lag}_{embedding_method}_N{len(member_ids)}_member0_{member_ids[0]}.png", dpi=200, bbox_inches='tight')
    

# ==========================================
    # 6. VISUALISATION DES ÉCHANTILLONS INDIVIDUELS
    # ==========================================
    print(f"Génération de la figure des {n_samples} échantillons par cluster...")

    # n_clusters lignes. Et pour les colonnes : n_samples pour SLP + n_samples pour SST
    fig2, axes2 = plt.subplots(2 * n_samples,n_clusters, figsize=(3.5 * n_clusters,2 * 2 * n_samples), 
                               subplot_kw={'projection': ccrs.PlateCarree()})

    for k in range(n_clusters):
        idx_cluster = np.where(labels == k)[0]
        
        # Sélection de n_samples au hasard dans ce cluster
        if len(idx_cluster) >= n_samples:
            sample_indices = np.random.choice(idx_cluster, n_samples, replace=False)
        else:
            sample_indices = idx_cluster # Sécurité s'il y a très peu de jours
            
        for i, idx in enumerate(sample_indices):
            mem, target_date = all_dates[idx]
            lagged_date = target_date - pd.Timedelta(days=lag)
            
            # --- Extraction des données ---
            slp_sample = slp_data_full[idx]
            try:
                sst_sample = sst_datasets[mem]['SST'].sel(time=lagged_date, method='nearest').values
            except KeyError:
                sst_sample = np.zeros((85, 360)) * np.nan
                
            # --- Colonnes de GAUCHE : SLP ---
            ax_slp = axes2[i, k]
            ax_slp.imshow(slp_sample, transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                          origin='lower', vmin=-vmax_plot, vmax=vmax_plot, extent=[-180, 180, 0, 90])
            #ax_slp.coastlines()
            
            # Titres et labels
            if k == 0: ax_slp.set_title(f"SLP Sample {i+1}")
            if i == 0: ax_slp.text(-0.1, 0.5, f"{regime_names[k]}", va='center', ha='center', 
                                   rotation='vertical', transform=ax_slp.transAxes, fontsize=12, fontweight='bold')
            ax_slp.text(0.02, 0.05, f"{target_date.strftime('%Y-%m-%d')}\nMem:{mem}", 
                        transform=ax_slp.transAxes, fontsize=8, backgroundcolor='white')

            # --- Colonnes de DROITE : SST ---
            ax_sst = axes2[n_samples + i,k]
            ax_sst.imshow(sst_sample, transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                          origin='lower', vmin=-vmax_plot, vmax=vmax_plot, extent=[-180, 180, -90, 90])
            #ax_sst.coastlines()
            
            if k == 0: ax_sst.set_title(f"SST Sample {i+1} (-{lag}j)")

    plt.suptitle(f"Échantillons individuels par Régime (SST à -{lag} jours)", fontsize=16, y=1.02)
    plt.tight_layout()
    # Nom adapté à base_home
    plt.savefig(f"{base_home}/Weather_Regimes_Samples_Lag{lag}_{embedding_method}_N{len(member_ids)}_member0_{member_ids[0]}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Figure des échantillons individuels sauvegardée.")

    # Fermeture propre des fichiers NetCDF
    for ds in sst_datasets.values():
        ds.close()
        
    print("Terminé ! Figure sauvegardée.")


## APPEL 

all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
members_used = all_members[:20]
for test_member in members_used:
    extract_weather_regimes_and_precursors(
        member_ids=test_member, 
        lag=35, 
        n_clusters=4, 
        path_slp=path_slp,
        path_sst=path_sst,
        embedding_method='pca', 
        vae_model_path=None # Mets le chemin de ton .pth ici quand tu l'auras !
    )