import xarray as xr
import pandas as pd
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import joblib
import cartopy.crs as ccrs

# Avantage de IncrementalPCA par rapport à juste accumuler les E(XY), E(X^2), E(Y^2) : on peut garder l'écosystème sklearn .transform(), .inverse_transform(). 

# Si on choisit de pondérer par la latitude, naturellement l'erreur de reconstruction sera aussi pondérée par la latitude. 
# ça marche bien pour mse car comportmeent carré comme PCA mais pour norme L1 il faudrait multiplier par cos sans la racine par exemple 
# Donc dans le cas général pour le calcul d'une loss, il vaut mieux pondéré à la toute fin 

# Si pondération : on multiplie compo par compo par la racine du cos de la latitude, de sorte que multiplie les carré par cos(phi) pour PCA et MSE qui est la bonne pondération
# On divise à la fin le résultat par la somme des cos 


# ============================================================
# 1. FONCTIONS ET CLASSES (Préparation, Visu)
# ============================================================

def compute_global_std(members, file_path_SST, selected_months, duree_lissage=10, monthly_reduction=False, lat_weight=False):
    """
    Calcule l'écart-type global de manière incrémentale (sans saturer la RAM).
    """
    print("Calcul de sst_std rigoureux (Pass 1/2)...")
    total_sum_sq = 0.0
    total_weights = 0.0
    
    # NOUVEAU : On va stocker la somme des cosinus ici une seule fois
    map_weight_sum = None 
    
    for member in members:
        X_chunk, _, _, wgts_flat = load_member_data(
            member, file_path_SST, selected_months, 
            sst_std=1.0, 
            duree_lissage=duree_lissage, 
            monthly_reduction=monthly_reduction,
            lat_weight=lat_weight,
            roll_sst=args.roll_sst
        )
        
        # Calcul unique de l'aire de la carte à la première itération
        if lat_weight and map_weight_sum is None and wgts_flat is not None:
            map_weight_sum = np.sum(wgts_flat**2)
            
        total_sum_sq += np.sum(X_chunk**2)
        n_samples = X_chunk.shape[0]
        
        # Utilisation de la constante
        if lat_weight:
            total_weights += n_samples * map_weight_sum
        else:
            total_weights += X_chunk.size
            
    global_variance = total_sum_sq / total_weights
    sst_std_rigoureux = np.sqrt(global_variance)
    
    sst_std_rigoureux = float(np.round(sst_std_rigoureux, 2))
    print(f"--> sst_std calculé et arrondi : {sst_std_rigoureux}")
    return sst_std_rigoureux

def load_member_data(member, file_path_SST, selected_months, sst_std=1.0, duree_lissage = 10, monthly_reduction=False,lat_weight=False, roll_sst=False):
    """
    Charge un seul membre, le filtre sur les mois d'hiver, 
    gère les NaNs et aplatit les cartes pour la PCA.
    """
    
    if not monthly_reduction:
        if duree_lissage != 0:
            ds = xr.open_dataset(os.path.join(file_path_SST, f'SST_anom_LE2-{member}_T_regrid_{duree_lissage}d.nc'))
        else:
            ds = xr.open_dataset(os.path.join(file_path_SST, f'SST_anom_LE2-{member}_T_regrid.nc'))
    else:
        ds = xr.open_dataset(os.path.join(file_path_SST, f'SST_anom_LE2-{member}_T_regrid_1mo.nc'))
    
    da = ds["SST"]
        
    da = da.sel(time=da['time'].dt.month.isin(selected_months))
    da = da.sel(lat=slice(-15, 70))
    dates = da.time.values
    
    # Récupération des données brutes
    data = da.values
    
    # Dimensions (Time, Lat, Lon) -> (Time, 85, 360)
    n_samples, h, w = data.shape
    
    # Remplacement des NaNs par 0 et normalisation (comme dans le VAE)
    data = np.nan_to_num(data, nan=0.0) / sst_std

    # NOUVEAU : roll optionnel en longitude (équivalent numpy de torch.roll(..., shifts=180, dims=-1))
    if roll_sst:
        data = np.roll(data, shift=180, axis=-1)

    # GESTION DE LA PONDÉRATION PAR LATITUDE
    wgts_flat = None
    if lat_weight:
        lats = da['lat'].values
        coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
        # Shape (1, 85, 1) pour le broadcast sur (Time, Lat, Lon)
        wgts = np.sqrt(coslat).reshape(1, h, 1) 
        data = data * wgts  # Application du poids
        
        # On crée un vecteur 1D des poids pour décoder les sorties de la PCA plus tard
        wgts_flat = np.broadcast_to(wgts.reshape(h, 1), (h, w)).flatten()
    
    data_flat = data.reshape(n_samples, h * w)
    return data_flat, dates, (h, w), wgts_flat

def plot_pca_reconstructions(y_true_flat, y_recon_flat, time_list, shape_2d, outdir, latent_dim, fixed_indices=[100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000], monthly_reduction=False,wgts_flat = None, roll_sst=False):
    """Trace les vraies SST vs les reconstructions de la PCA"""
    
    extent_sst = [-180, 180, -15, 70] if roll_sst else [0, 359.9, -15, 70]
    if monthly_reduction:
        fixed_indices = [i//30 for i in fixed_indices]  # Ajustement des indices pour les données mensuelles

    N = y_true_flat.shape[0]
    valid_indices = [idx for idx in fixed_indices if idx < N]
    num_samples = len(valid_indices)
    
    if num_samples == 0: return

    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4 * num_samples), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f"PCA Reconstructions (Latent Dim = {latent_dim})", fontsize=16)

    h, w = shape_2d

    for i, idx in enumerate(valid_indices):
        true_vec = y_true_flat[idx]
        recon_vec = y_recon_flat[idx]
        
        # RETOUR À L'ESPACE PHYSIQUE POUR L'AFFICHAGE
        if wgts_flat is not None:
            safe_wgts = np.maximum(wgts_flat, 1e-5) # Sécurité division par zéro
            true_vec = true_vec / safe_wgts
            recon_vec = recon_vec / safe_wgts

        true_map = true_vec.reshape(h, w)
        recon_map = recon_vec.reshape(h, w)
        
        # Formatage de la date (gère les cftime climatiques)
        date_item = time_list[idx]
        if hasattr(date_item, 'strftime'):
            date_str = date_item.strftime('%Y-%m-%d')
        else:
            date_str = str(date_item)[:10] # Fallback (AAAA-MM-JJ)

        vmin, vmax = -2, 2
        ax_row = axes[i] if num_samples > 1 else axes

        # Vraie SST
        im1 = ax_row[0].imshow(true_map, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_sst)
        ax_row[0].set_title(f"Vraie SST - {date_str}")
        fig.colorbar(im1, ax=ax_row[0], fraction=0.046, pad=0.04)
        ax_row[0].coastlines()
        ax_row[0].set_extent(extent_sst, crs=ccrs.PlateCarree())

        # SST Reconstruite
        im2 = ax_row[1].imshow(recon_map, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_sst)
        ax_row[1].set_title(f"Reconstruction PCA (Latent Dim = {latent_dim})")
        fig.colorbar(im2, ax=ax_row[1], fraction=0.046, pad=0.04)
        ax_row[1].coastlines()
        ax_row[1].set_extent(extent_sst, crs=ccrs.PlateCarree())

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_reconstructions.png"), dpi=150)
    plt.close()

def plot_explained_variance(ipca, outdir, val_curves=None):
    """
    Plots the cumulative explained variance of the PCA for Train and Validation sets.
    Optimized to handle perfectly overlapping curves smoothly.
    """
    plt.style.use('seaborn-v0_8-whitegrid') 
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    # Train cumulative variance
    cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
    components = np.arange(1, len(cumulative_variance) + 1)
    
    # Courbe d'entraînement : Épaisse et semi-transparente pour servir de fond
    ax.plot(components, cumulative_variance, color='#1f77b4', 
            linestyle='-', linewidth=5, alpha=0.4, label='Train (87 Members)')
    
    # Remplissage très léger sous la courbe
    ax.fill_between(components, cumulative_variance, color='#1f77b4', alpha=0.05)
    
    # Courbes de validation : Lignes plus fines, styles distincts, SANS marqueurs
    if val_curves:
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        linestyles = ['--', '-.'] # Alternance tirets et point-tirets
        
        for i, (member, cum_var) in enumerate(val_curves.items()):
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            ax.plot(components, cum_var, color=color, 
                    linestyle=linestyle, linewidth=2, 
                    label=f'Validation (Member {member})')

    # Titres et labels
    ax.set_xlabel('Number of Principal Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax.set_title('PCA Cumulative Explained Variance', fontsize=14, fontweight='bold', pad=15)
    
    # Ajustements des axes
    # On ajuste le Y minimum dynamiquement pour couper l'espace blanc
    min_y = min(cumulative_variance[0], 0.3)
    ax.set_ylim(bottom=min_y, top=1.02)
    ax.set_xlim(0, len(cumulative_variance) + 1)
    
    # Grille plus discrète
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Légende épurée
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True, borderpad=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'Fig_explained_variance_with_val.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Graphique de variance expliquée épuré sauvegardé.")


def plot_eof_patterns(ipca, shape_2d, outdir, n_eofs=10,wgts_flat = None,sst_std=1.0,roll_sst=False):
    """
    Affiche les n_eofs premières structures spatiales (EOFs).
    """
    extent_sst = [-180, 180, -15, 70] if roll_sst else [0, 359.9, -15, 70]
    h, w = shape_2d
    # Les EOFs sont dans ipca.components_ (shape: n_components, n_features)

    # On s'assure de ne pas demander plus d'EOFs qu'il n'y a de composantes calculées
    n_eofs = min(n_eofs, ipca.components_.shape[0])

    eofs_ortho = ipca.components_[:n_eofs]
    # Variance expliquée pour les titres
    var_exp = ipca.explained_variance_ratio_[:n_eofs] * 100
    pc_stds = np.sqrt(ipca.explained_variance_[:n_eofs])
    eofs_phys = eofs_ortho * pc_stds[:, np.newaxis]

    # RETOUR À L'ESPACE PHYSIQUE POUR L'AFFICHAGE
    if wgts_flat is not None:
        safe_wgts = np.maximum(wgts_flat, 1e-5)
        eofs_phys = eofs_phys / safe_wgts
    eofs = eofs_phys * sst_std  # Remise à l'échelle pour l'affichage

    rows = (n_eofs + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f"Top {n_eofs} EOF Patterns (Spatial Structures)", fontsize=20)

    axes = axes.flatten()

    for i in range(n_eofs):
        eof_map = eofs[i].reshape(h, w)
        
        # On centre l'échelle de couleur sur 0
        vlim = np.max(np.abs(eof_map)) 
        
        im = axes[i].imshow(eof_map, cmap='RdBu_r', origin='lower', vmin=-vlim, vmax=vlim, transform=ccrs.PlateCarree(), extent=extent_sst)
        axes[i].set_title(f"EOF {i+1} ({var_exp[i]:.1f}% Expl. Var.)", fontsize=12, fontweight='bold', pad=10)
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, shrink=0.5)
        cbar.set_label('SST Anomaly (K)')

        axes[i].coastlines()
        axes[i].set_extent(extent_sst, crs=ccrs.PlateCarree())

    # Si n_eofs est impair, on cache le dernier axe vide
    if n_eofs % 2 != 0:
        axes[-1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(outdir, f"top_{n_eofs}_eofs.png"), dpi=150)
    plt.close()
    print(f"--> EOFs sauvegardées dans {outdir}")

# ============================================================
# 2. CONFIGURATION GÉNÉRALE
# ============================================================

if __name__ == "__main__": 
    


    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch',"mac_local"])
    parser.add_argument('--normalize', action='store_true', help='Si on veut normaliser les PC, à omettre si on ne veut pas les normaliser')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours (10 ou 30)')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension de l\'espace latent (nombre de composantes principales)')
    parser.add_argument('--monthly_reduction', action='store_true', help='Si on veut réduire les données à une moyenne mensuelle avant la PCA')
    parser.add_argument('--lat_weight', action='store_true', help='Applique la pondération sqrt(cos(lat))')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[11, 12, 1, 2], help='Mois d\'hiver à considérer (ex: 11 12 1 2 pour NDJF)')
    parser.add_argument('--roll_sst', action='store_true', help='Si on veut centrer l\'océan atlantique')
    args = parser.parse_args()

    # Hyperparamètre principal (équivalent de la taille du bottleneck du VAE)
    latent_dim = args.latent_dim
    normalize = args.normalize
    duree_lissage = args.duree_lissage
    monthly_reduction = args.monthly_reduction
    lat_weight = args.lat_weight
    train_members_87 = ['1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']


    nb_members_train = 87
    train_members = train_members_87[:nb_members_train]
    val_members = ['1001.001','1041.003']

    winter_months = args.winter_months # NDJF
    months_label = "_".join(map(str, winter_months))

    if args.machine == 'hacienda':
        path_sst = "/data/moysan/data/SST/"
        base_home = "/home/moysan/stage_isir_jz/data_analysis/pca_sst/"
    elif args.machine == 'jean-zay-work':
        path_sst = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_sst/"
    elif args.machine == 'jean-zay-scratch':
        path_sst = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/SST/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_sst/"
    else: # mac_local
        path_sst = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SST/"
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/pca_sst/"


    # ============================================================
    # 3. INITIALISATION DU MODÈLE PCA
    # ============================================================

    # CALCUL DYNAMIQUE DU SST_STD
    dynamic_sst_std = compute_global_std(
        train_members, path_sst, winter_months, 
        duree_lissage=duree_lissage, 
        monthly_reduction=monthly_reduction, 
        lat_weight=lat_weight
    )

    if not monthly_reduction:
        outdir_name = f'IPCA_latent{latent_dim}_winter_months_{months_label}_{nb_members_train}members_normalize{normalize}_duree_lissage{duree_lissage}_wgt{lat_weight}_sst_std{dynamic_sst_std}_roll_sst{args.roll_sst}'
    else:
        outdir_name = f'IPCA_latent{latent_dim}_winter_months_{months_label}_{nb_members_train}members_normalize{normalize}_monthly_reduction_wgt{lat_weight}_sst_std{dynamic_sst_std}_roll_sst{args.roll_sst}'
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    # IncrementalPCA permet de fitter le modèle chunk par chunk (membre par membre)
    # sans saturer la RAM.
    ipca = IncrementalPCA(n_components=latent_dim,whiten=normalize)
    
    print(f"Initialisation de l'Incremental PCA avec latent_dim = {latent_dim}")

    # ============================================================
    # 4. "TRAINING" LOOP (Ajustement de la PCA)
    # ============================================================
    start_time = time.time() 
    shape_2d = None
    global_wgts_flat = None # On le sauvegarde pour les plots d'EOF

    print("\n--- Début du fit (Entraînement) ---")
    for i, member in enumerate(train_members):
        print(f"Chargement et fit du membre {member} ({i+1}/{nb_members_train})...")
        
        # On charge les données d'un seul membre
        X_train_chunk, _, shape_2d, global_wgts_flat = load_member_data(member, path_sst, winter_months, duree_lissage=duree_lissage, monthly_reduction=monthly_reduction,lat_weight=lat_weight,sst_std=dynamic_sst_std, roll_sst=args.roll_sst)
        
        # On met à jour la PCA avec ce membre
        ipca.partial_fit(X_train_chunk)

    print(f"Fit terminé. Variance totale expliquée par {latent_dim} composantes : {np.sum(ipca.explained_variance_ratio_)*100:.2f}%")
    plot_explained_variance(ipca, outdir)

    # NOUVEAU : Plot des EOFs
    plot_eof_patterns(ipca, shape_2d, outdir, n_eofs=10,wgts_flat=global_wgts_flat,sst_std=dynamic_sst_std, roll_sst=args.roll_sst)

    # SAUVEGARDE DU MODÈLE LÉGER
    model_path = os.path.join(outdir, 'best_pca_model.joblib')
    joblib.dump(ipca, model_path)
    print(f"--> Modèle PCA sauvegardé sous : {model_path}")
    print("    (Tu pourras le recharger avec : pca = joblib.load('best_pca_model.joblib'))")

    # ============================================================
    # 5. VALIDATION & RECONSTRUCTION
    # ============================================================
    print("\n--- Début de la Validation (Reconstruction) ---")
    
    val_true_list = []
    val_recon_list = []
    val_dates_list = []
    
    # Dictionnaire pour stocker les courbes de variance de chaque membre
    val_cumulative_variances = {}

    val_map_weight_sum = None
    if lat_weight and global_wgts_flat is not None:
        val_map_weight_sum = np.sum(global_wgts_flat**2)

    for member in val_members:
        print(f"Évaluation sur le membre de validation : {member}")
        X_val, dates_val, _, global_wgts_flat = load_member_data(member, path_sst, winter_months, duree_lissage=duree_lissage, monthly_reduction=monthly_reduction,lat_weight=lat_weight,sst_std=dynamic_sst_std, roll_sst=args.roll_sst)
        
        # 1. Évaluation de la Loss totale (ton code actuel)
        latent_vectors = ipca.transform(X_val)
        X_val_recon = ipca.inverse_transform(latent_vectors)
        squared_diff = (X_val - X_val_recon)**2
        
        if lat_weight and val_map_weight_sum is not None:
            n_samples = X_val.shape[0]
            total_weight = n_samples * val_map_weight_sum
            mse = np.sum(squared_diff) / total_weight
        else:
            mse = np.mean(squared_diff)
        print(f"MSE de reconstruction sur la validation : {mse:.4f}")

        # 2. CALCUL DE LA VARIANCE EXPLIQUÉE POUR CHAQUE COMPOSANTE SUR LA VALIDATION
        # Formule du R2 score : 1 - (SSE / SST)
        print(f"Calcul de la variance expliquée cumulative pour {member}...")
        
        # Variance totale (SST) du membre autour de la moyenne globale du modèle
        X_val_centered = X_val - ipca.mean_
        sst = np.sum(X_val_centered**2) 
        
        cum_var_member = []
        for k in range(1, ipca.n_components + 1):
            # On isole les k premières composantes latentes (le reste est mis à zéro)
            Z_k = np.zeros_like(latent_vectors)
            Z_k[:, :k] = latent_vectors[:, :k]
            
            # On reconstruit à partir de ces k composantes
            X_recon_k = ipca.inverse_transform(Z_k)
            
            # Somme des erreurs au carré (SSE) pour k composantes
            sse = np.sum((X_val - X_recon_k)**2)
            
            # Variance expliquée (équivalent au score R2 multivarié)
            explained_var = 1.0 - (sse / sst)
            cum_var_member.append(explained_var)
            
        val_cumulative_variances[member] = cum_var_member

        # 3. Stockage pour les cartes
        val_true_list.append(X_val)
        val_recon_list.append(X_val_recon)
        val_dates_list.extend(dates_val)

    # Concaténation pour le plot spatial
    y_true_all = np.concatenate(val_true_list, axis=0)
    y_recon_all = np.concatenate(val_recon_list, axis=0)

    # Plot des reconstructions spatiales
    plot_pca_reconstructions(y_true_all, y_recon_all, val_dates_list, shape_2d, outdir, latent_dim=latent_dim, monthly_reduction=monthly_reduction, wgts_flat=global_wgts_flat, roll_sst=args.roll_sst)

    # GÉNÉRATION DU NOUVEAU PLOT COMBINÉ
    plot_explained_variance(ipca, outdir, val_curves=val_cumulative_variances)

    elapsed_time = (time.time() - start_time) / 60
    print(f"\nExécution complète en {elapsed_time:.2f} minutes.")