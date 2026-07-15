import xarray as xr
import pandas as pd
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
from scipy.linalg import svd
from datetime import timedelta
from dateutil.relativedelta import relativedelta # <-- Indispensable pour les mois
import joblib
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# éventuellement rechecker les normalisations successives

# ============================================================
# 1. FONCTIONS DE PRÉPARATION DE DONNÉES
# ============================================================

def compute_mca_stds(members, path_SST, path_SLP, selected_months, sst_lags, duree_lissage=10, roll_sst=False, monthly_reduction=False, lat_weight=False):
    """
    Calcule l'écart-type global de la SST et de la SLP de manière incrémentale.
    """
    print("Calcul dynamique de sst_std et slp_std rigoureux (Pass 1/2)...")
    total_sum_sq_sst = 0.0
    total_sum_sq_slp = 0.0
    total_weights_sst = 0.0
    total_weights_slp = 0.0
    
    map_weight_sum_sst = None
    map_weight_sum_slp = None
    
    for member in members:
        # On charge avec std=1.0 pour ne pas altérer les données brutes
        X_sst, Y_slp, _, _, _, w_sst, w_slp = load_member_mca_data(
            member, path_SST, path_SLP, selected_months, 
            sst_lags=sst_lags, duree_lissage=duree_lissage, 
            sst_std=1.0, slp_std=1.0, # <-- CRUCIAL
            roll_sst=roll_sst, monthly_reduction=monthly_reduction, lat_weight=lat_weight
        )
        
        # Initialisation de l'aire des cartes au premier passage
        if lat_weight and map_weight_sum_sst is None and w_sst is not None:
            map_weight_sum_sst = np.sum(w_sst**2)
            map_weight_sum_slp = np.sum(w_slp**2)
            
        total_sum_sq_sst += np.sum(X_sst**2)
        total_sum_sq_slp += np.sum(Y_slp**2)
        
        n_samples = X_sst.shape[0]
        
        if lat_weight:
            total_weights_sst += n_samples * map_weight_sum_sst
            total_weights_slp += n_samples * map_weight_sum_slp
        else:
            total_weights_sst += X_sst.size
            total_weights_slp += Y_slp.size
            
    sst_std_rigoureux = np.sqrt(total_sum_sq_sst / total_weights_sst)
    slp_std_rigoureux = np.sqrt(total_sum_sq_slp / total_weights_slp)
    
    print(f"--> sst_std calculé : {sst_std_rigoureux:.4f}")
    print(f"--> slp_std calculé : {slp_std_rigoureux:.4f}")
    
    return sst_std_rigoureux, slp_std_rigoureux

def load_member_mca_data(member, path_SST, path_SLP, selected_months, sst_lags=[35], duree_lissage=10, sst_std=0.707, slp_std=596.0,roll_sst=False,monthly_reduction=False,lat_weight=False):
    """
    Charge les paires (SST retardée, SLP actuelle) pour un membre.
    Les différents lags de SST sont concaténés comme des "features" supplémentaires.
    """
    # Chargement des datasets
    if not monthly_reduction:
        if duree_lissage != 0:
            ds_slp = xr.open_dataset(os.path.join(path_SLP, f'PSL_anom_LE2-{member}_{duree_lissage}d.nc'))["PSL"]
        else:
            ds_slp = xr.open_dataset(os.path.join(path_SLP, f'PSL_anom_LE2-{member}.nc'))["PSL"]
        ds_sst = xr.open_dataset(os.path.join(path_SST, f'SST_anom_LE2-{member}_T_regrid.nc'))["SST"]
    else:
        ds_slp = xr.open_dataset(os.path.join(path_SLP, f'PSL_anom_LE2-{member}_1mo.nc'))["PSL"]
        ds_sst = xr.open_dataset(os.path.join(path_SST, f'SST_anom_LE2-{member}_T_regrid_1mo.nc'))["SST"]
    
    if roll_sst: #centre l'océan atlantique
        ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon')

    # 1. Identifier les dates cibles valides pour la SLP (mois d'hiver)
    slp_winter = ds_slp.sel(time=ds_slp['time'].dt.month.isin(selected_months))
    years = slp_winter['time'].dt.year
    
    # Éviter les bornes extrêmes où les lags passés / futurs pourraient ne pas exister
    valid_mask = (years > years.min()) & (years < years.max() - 1)
    target_dates = slp_winter.sel(time=valid_mask).time.values
    
    # Récupération des dimensions spatiales 
    ds_sst_cropped = ds_sst.sel(lat=slice(-15, 70))
    shape_sst = ds_sst_cropped.shape[1:] # (lat, lon)
    shape_slp = ds_slp.shape[1:]         # (lat, lon)
    
    # 3. GESTION DE LA PONDÉRATION SPATIALE (Sur les 2 variables)
    wgts_sst_flat = None
    wgts_slp_flat = None
    wgts_sst_3d = None
    wgts_slp_2d = None
    
    if lat_weight:
        # Poids SLP
        lats_slp = ds_slp['lat'].values
        coslat_slp = np.cos(np.deg2rad(lats_slp)).clip(0., 1.)
        wgts_slp_2d = np.sqrt(coslat_slp).reshape(shape_slp[0], 1)
        wgts_slp_flat = np.broadcast_to(wgts_slp_2d, shape_slp).flatten()
        
        # Poids SST (Attention : dimension 3D avec les lags)
        lats_sst = ds_sst_cropped['lat'].values
        coslat_sst = np.cos(np.deg2rad(lats_sst)).clip(0., 1.)
        wgts_sst_1d = np.sqrt(coslat_sst).reshape(shape_sst[0], 1)
        # On prépare le tenseur pour le broadcasting (1, lat, 1) pour multiplier (lags, lat, lon)
        wgts_sst_3d = wgts_sst_1d.reshape(1, shape_sst[0], 1)
        
        # Le tenseur flat final servira à dé-pondérer les EOFs : shape (lags * lat * lon)
        wgts_sst_full = np.broadcast_to(wgts_sst_3d, (len(sst_lags), shape_sst[0], shape_sst[1]))
        wgts_sst_flat = wgts_sst_full.flatten()

    # 2. Extraction vectorisée / itérative
    X_sst_list = []
    Y_slp_list = []
    valid_target_dates = []

    # L'extraction date par date avec xarray peut être lente. On boucle sur les dates.
    # Pour des perfs optimales, on pourrait utiliser des indexateurs numpy, mais xarray `.sel` garantit l'alignement.
    for t_target in target_dates:
        # Reconstitution de la date (gestion des types cftime)
        if isinstance(t_target, np.datetime64):
            t_obj = pd.Timestamp(t_target)
        else:
            t_obj = t_target # cftime object
            
        if not monthly_reduction:
            dates_sst = [t_obj - timedelta(days=d) for d in sst_lags]
        else:
            # Pas viable avec No Leap dates_sst = [t_obj - relativedelta(months=d) for d in sst_lags]
            dates_sst = []
            for m in sst_lags:
                y_shift = (t_obj.month - m - 1) // 12
                new_month = (t_obj.month - m - 1) % 12 + 1
                dates_sst.append(t_obj.replace(year=t_obj.year + y_shift, month=new_month))
        
        try:
            # Extraction SLP
            slp_t = ds_slp.sel(time=t_obj)
            
            # Extraction SST (on se restreint à la même zone que dans le Dataset PyTorch)
            sst_lags_ds = ds_sst.sel(time=dates_sst, lat=slice(-15, 70))
            
            # Traitement NaNs et Normalisation
            slp_np = np.nan_to_num(slp_t.values, nan=0.0) / slp_std
            sst_np = np.nan_to_num(sst_lags_ds.values, nan=0.0) / sst_std

            if lat_weight:
                slp_np *= wgts_slp_2d
                sst_np *= wgts_sst_3d
            
            # Aplatissement. Si plusieurs lags, on les met à plat avec l'espace (features = lags * lat * lon)
            Y_slp_list.append(slp_np.flatten())
            X_sst_list.append(sst_np.flatten())
            valid_target_dates.append(t_target)
            
        except KeyError:
            # Si un lag déborde du calendrier disponible, on passe
            continue
            
    X_mat = np.array(X_sst_list) # Shape: (N, lags * lat_sst * lon_sst)
    Y_mat = np.array(Y_slp_list) # Shape: (N, lat_slp * lon_slp)
        
    return X_mat, Y_mat, valid_target_dates, shape_sst, shape_slp, wgts_sst_flat, wgts_slp_flat


# ============================================================
# 2. FONCTIONS DE VISUALISATION (MCA)
# ============================================================

def plot_mca_modes(U, V, s, shape_sst, shape_slp, sst_lags, outdir,Cxx, Cyy, sst_std=1.0, slp_std=1.0, n_modes=5,roll_sst=False,wgts_sst_flat=None,wgts_slp_flat=None):
    """
    Affiche les premiers modes couplés de la MCA.
    S'adapte dynamiquement pour afficher tous les lags fournis pour la SST.
    """
    # === RETOUR À LA PHYSIQUE (Dé-pondération) ===
    U_phys = np.copy(U)
    V_phys = np.copy(V)
    # U contient les modes SST (features_X, modes)
    if wgts_sst_flat is not None:
        safe_wgts_sst = np.maximum(wgts_sst_flat, 1e-5)
        U_phys = U_phys / safe_wgts_sst[:, None] # Division ligne par ligne

    # V contient les modes SLP (modes, features_Y)
    if wgts_slp_flat is not None:
        safe_wgts_slp = np.maximum(wgts_slp_flat, 1e-5)
        V_phys = V_phys / safe_wgts_slp[None, :] # Division colonne par colonne

    extent_slp = [-100, 40, 20, 70] 
    if roll_sst:
        extent_sst = [-180, 180, -15, 70]
    else:
        extent_sst = [0, 359.9, -15, 70]

    scf = (s**2) / np.sum(s**2) * 100 # Squared Covariance Fraction
    
    h_sst, w_sst = shape_sst
    h_slp, w_slp = shape_slp
    num_lags = len(sst_lags)
    
    # Création d'une grille adaptative : n_modes lignes, (num_lags + 1) colonnes
    fig, axes = plt.subplots(n_modes, num_lags + 1, figsize=(6 * (num_lags + 1), 4 * n_modes),subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f"Maximum Covariance Analysis (MCA) - Top {n_modes} Modes", fontsize=18)
    
    # Sécurité si un seul mode est demandé (axes devient 1D au lieu de 2D)
    if n_modes == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(n_modes):
        # 1. CALCUL DES ÉCARTS-TYPES ASSOCIES AUX MODES TEMPORELS
        sigma_Ai = np.sqrt(np.maximum(U[:, i].T @ Cxx @ U[:, i], 1e-10))
        sigma_Bi = np.sqrt(np.maximum(V[i, :] @ Cyy @ V[i, :].T, 1e-10))
        # Reconstruction 3D (lags, lat, lon) pour la SST
        # mode_sst_full = U[:, i].reshape((num_lags, h_sst, w_sst))
        # mode_slp = V[i, :].reshape((h_slp, w_slp))
        # 2. APPLICATION DU DOUBLE SCALING (Ecart-type du mode X Ecart-type global)
        mode_sst_full = (U_phys[:, i] * sigma_Ai * sst_std).reshape((num_lags, h_sst, w_sst))
        mode_slp = (V_phys[i, :] * sigma_Bi * slp_std).reshape((h_slp, w_slp))


        # On calcule une limite de couleur globale pour TOUS les lags de la SST 
        # afin que l'échelle d'intensité soit comparable entre les différentes temporalités
        vlim_sst = np.max(np.abs(mode_sst_full))
        vlim_slp = np.max(np.abs(mode_slp))
        
        # 1. Plot de tous les Lags de SST
        for l in range(num_lags):
            ax_sst = axes[i, l]
            im_sst = ax_sst.imshow(mode_sst_full[l], cmap='RdBu_r', origin='lower', vmin=-vlim_sst, vmax=vlim_sst,transform=ccrs.PlateCarree(), extent=extent_sst)
            unit = "mois" if "monthly" in outdir else "jours"
            ax_sst.set_title(f"Mode {i+1} SST (-{sst_lags[l]} {unit})")
            ax_sst.coastlines()
            
            # On n'ajoute la barre de couleur qu'au dernier lag SST pour ne pas surcharger la figure
            if l == num_lags - 1:
                cbar_sst = fig.colorbar(im_sst, ax=ax_sst, fraction=0.046, pad=0.04)
                cbar_sst.set_label('SST Anom (K)') # Échelle Océan
        
        # 2. Plot de la SLP (dernière colonne)
        ax_slp = axes[i, num_lags]
        im_slp = ax_slp.imshow(mode_slp, cmap='RdBu_r', origin='lower', vmin=-vlim_slp, vmax=vlim_slp,transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_slp.set_title(f"Mode {i+1} SLP (SCF = {scf[i]:.2f}%)")
        ax_slp.coastlines()
        cbar_slp = fig.colorbar(im_slp, ax=ax_slp, fraction=0.046, pad=0.04)
        cbar_slp.set_label('SLP Anom (hPa)') # Échelle Atmosphère
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(outdir, "mca_coupled_modes.png"), dpi=150)
    plt.close()

def plot_scf(s, outdir, n_modes_plot=50):
    """Plot de la fraction de covariance au carré."""
    scf = (s**2) / np.sum(s**2)
    scf_cum = np.cumsum(scf)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_modes_plot + 1), scf_cum[:n_modes_plot], marker='o', color='red', label='SCF cumulée')
    plt.bar(range(1, n_modes_plot + 1), scf[:n_modes_plot], alpha=0.5, label='SCF par mode')
    
    plt.xlabel('Numéro du mode')
    plt.ylabel('Fraction de covariance expliquée (SCF)')
    plt.title('Maximum Covariance Analysis : Spectre des valeurs singulières')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'mca_scf_variance.png'), dpi=150)
    plt.close()

def plot_mca_metrics_dashboard(RV_coeff, R_value, VF, HVF_X, HVF_Y, A1, B1, outdir):
    """
    Génère un dashboard synthétique pour les métriques globales et les projections temporelles.
    """
    fig = plt.figure(figsize=(15, 7)) # Légèrement agrandi pour les 5 barres
    fig.suptitle("Synthèse des Métriques de Couplage (SST vs SLP)", fontsize=16, fontweight='bold')
    
    # --- Sous-graphe 1 : Jauges / Barres des métriques ---
    ax1 = plt.subplot(1, 2, 1)
    metrics = [
        'Coefficient RV\n(Couplage spatial global)', 
        '|R| Temporel\n(Corrélation Mode 1)', 
        'Var. interne SST\nexpliquée par son Mode 1 (HVF)',
        'Var. interne SLP\nexpliquée par son Mode 1 (HVF)',
        'Var. SLP expliquée\npar la SST (VF croisée)'
    ]
    
    # On met tout en proportion (0-1) pour harmoniser l'échelle visuelle
    values = [RV_coeff, np.abs(R_value), HVF_X / 100, HVF_Y / 100, VF / 100] 
    actual_texts = [f"{RV_coeff:.4f}", f"{R_value:.3f}", f"{HVF_X:.2f} %", f"{HVF_Y:.2f} %", f"{VF:.2f} %"]
    
    y_pos = np.arange(len(metrics))
    # Nouvelles couleurs pour bien distinguer (Global, Temporel, Interne, Interne, Croisé)
    colors = ['#2ca02c', '#1f77b4', '#9467bd', '#e377c2', '#ff7f0e']
    
    bars = ax1.barh(y_pos, values, color=colors, alpha=0.8)
    ax1.set_xlim(0, 1.05) 
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics, fontsize=11)
    ax1.invert_yaxis() 
    ax1.set_title("Indicateurs d'intensité et de représentativité", fontsize=14)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Ajouter le texte exact sur/à côté des barres
    for i, bar in enumerate(bars):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                 actual_texts[i], va='center', fontsize=12, fontweight='bold', color='black')

    # Boîte de texte explicative mise à jour
    explications = (
        "• RV : Force de la superposition globale des deux grandeurs\n"
        "• HVF : Poids physique du mode au sein de son propre fluide\n"
        "• VF croisée : Capacité de l'océan (SST) à forcer/expliquer l'atmosphère (SLP)"
    )
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8)
    ax1.text(0.02, -0.25, explications, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=props, style='italic')

    # --- Sous-graphe 2 : Scatter plot A1 vs B1 ---
    ax2 = plt.subplot(1, 2, 2)
    
    A1_std = (A1 - np.mean(A1)) / np.std(A1)
    B1_std = (B1 - np.mean(B1)) / np.std(B1)
    
    alpha_val = max(0.05, min(0.4, 500 / len(A1_std))) 
    ax2.scatter(A1_std, B1_std, alpha=alpha_val, edgecolors='none', color='#1f77b4')
    
    m, b = np.polyfit(A1_std, B1_std, 1)
    x_line = np.array([np.min(A1_std), np.max(A1_std)])
    ax2.plot(x_line, m*x_line + b, color='#d62728', linewidth=2.5, 
             label=f'Régression (R = {R_value:.3f})')
    
    ax2.set_title("Diagramme de dispersion (Mode 1)", fontsize=14)
    ax2.set_xlabel("Amplitude normalisée SST ($A_1$)")
    ax2.set_ylabel("Amplitude normalisée SLP ($B_1$)")
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    plt.savefig(os.path.join(outdir, "mca_metrics_dashboard.png"), dpi=200, bbox_inches='tight')
    plt.close()
# ============================================================
# 3. SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda')
    parser.add_argument('--duree_lissage', type=int, default=10)
    # On peut rajouter des arguments pour gérer plusieurs lags
    parser.add_argument('--sst_lags', type=int, nargs='+', default=[35], help='Lags en jours pour la SST (ex: 35 65 95)')
    parser.add_argument('--roll_sst', action='store_true', help='Appliquer un lissage glissant à la SST avant la MCA')
    # NOUVEAUX ARGUMENTS
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données mensuelles (_1mo.nc)')
    parser.add_argument('--lat_weight', action='store_true', help='Appliquer la pondération spatiale sqrt(cos(lat))')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[11, 12, 1, 2], help='Mois d\'hiver à considérer (ex: 11 12 1 2 pour NDJF)')
    args = parser.parse_args()

    train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    
    nb_members_train = len(train_members_87)
    train_members = train_members_87
    winter_months = args.winter_months
    

    if args.machine == 'hacienda':
        path_SLP = "/data/moysan/data/SLP/"
        path_SST = "/data/moysan/data/SST/"
        base_home = "/home/moysan/stage_isir_jz/data_analysis/mca_slp_sst/"
    elif args.machine == 'jean-zay-work':
        path_SLP = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/"
        path_SST = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/mca_slp_sst/"
    elif args.machine == 'jean-zay-scratch':
        path_SLP = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/SLP/"
        path_SST = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/SST/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/mca_slp_sst/"
    else: # mac_local
        path_SLP = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SLP/"
        path_SST = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SST/"
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/mca_slp_sst/"

    # ============================================================
    # 3.5 CALCUL DES ÉCARTS-TYPES GLOBAUX
    # ============================================================
    dynamic_sst_std, dynamic_slp_std = compute_mca_stds(
        train_members, path_SST, path_SLP, winter_months, 
        sst_lags=args.sst_lags, duree_lissage=args.duree_lissage, 
        roll_sst=args.roll_sst, monthly_reduction=args.monthly_reduction, lat_weight=args.lat_weight
    )

    if not args.monthly_reduction:
        outdir_name = f'MCA_months_{"_".join(map(str, winter_months))}_{nb_members_train}membres_lags_{"_".join(map(str, args.sst_lags))}_lissage_{args.duree_lissage}d_roll_sst_{args.roll_sst}_lat_weight_{args.lat_weight}_sst_std_{dynamic_sst_std:.4f}_slp_std_{dynamic_slp_std:.4f}'
    else:
        outdir_name = f'MCA_months_{"_".join(map(str, winter_months))}_{nb_members_train}membres_lags_{"_".join(map(str, args.sst_lags))}_monthly_reduction_roll_sst_{args.roll_sst}_lat_weight_{args.lat_weight}_sst_std_{dynamic_sst_std:.4f}_slp_std_{dynamic_slp_std:.4f}'
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    print(f"Lancement de la MCA avec les lags de SST : {args.sst_lags} jours")

    # ============================================================
    # CALCUL INCRÉMENTAL DES MATRICES DE COVARIANCE
    # ============================================================
    start_time = time.time()
    
    C = None # Matrice de covariance croisée (X^T Y)
    Cxx = None # Auto-covariance de X
    Cyy = None # Auto-covariance de Y
    N_total = 0 # Nombre total d'échantillons

    global_wgts_sst = None
    global_wgts_slp = None
    
    for i, member in enumerate(train_members):
        print(f"Traitement du membre {member} ({i+1}/{nb_members_train})...")
        X, Y, dates, shape_sst, shape_slp, w_sst, w_slp = load_member_mca_data(
            member, path_SST, path_SLP, winter_months, 
            sst_lags=args.sst_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, monthly_reduction=args.monthly_reduction, lat_weight=args.lat_weight,sst_std=dynamic_sst_std, slp_std=dynamic_slp_std
        )
        
        # Sauvegarde de la grille des poids à la première itération
        if w_sst is not None and global_wgts_sst is None:
            global_wgts_sst = w_sst
            global_wgts_slp = w_slp

        N_total += X.shape[0]
        
        if C is None:
            C = X.T @ Y
            Cxx = X.T @ X
            Cyy = Y.T @ Y
        else:
            C += X.T @ Y
            Cxx += X.T @ X
            Cyy += Y.T @ Y

    # Normalisation pour obtenir les vraies matrices de covariance
    C = C / (N_total - 1)
    Cxx = Cxx / (N_total - 1)
    Cyy = Cyy / (N_total - 1)
    
    print(f"Matrices de covariance calculées. Dimension croisée : {C.shape}. Calcul de la SVD...")

    # ============================================================
    # SVD ET EXTRACTION DES MODES
    # ============================================================
    
    # full_matrices=False permet de gagner drastiquement en temps et en mémoire
    U, s, Vt = svd(C, full_matrices=False)
    
    print("SVD terminée. Génération des plots...")
    
    # Tracer le spectre des valeurs singulières (SCF)
    plot_scf(s, outdir, n_modes_plot=50)
    
    # Tracer les 5 premiers modes couplés
    plot_mca_modes(U, Vt, s, shape_sst, shape_slp, args.sst_lags, outdir, Cxx=Cxx, Cyy=Cyy, sst_std=dynamic_sst_std, slp_std=dynamic_slp_std, n_modes=5, roll_sst=args.roll_sst, wgts_sst_flat=global_wgts_sst, wgts_slp_flat=global_wgts_slp)
    
    # Sauvegarde du modèle (les matrices) au cas où l'on veut calculer des projections (expansion coefficients) plus tard
    model_path = os.path.join(outdir, 'mca_model.joblib')
    joblib.dump({'U': U, 'V': Vt.T, 's': s, 'shape_sst': shape_sst, 'shape_slp': shape_slp}, model_path)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Exécution complète en {elapsed:.2f} minutes.")
    print(f"Résultats et figures sauvegardés dans : {outdir}")

    # ============================================================
    # 4. CALCUL ET SAUVEGARDE DES MÉTRIQUES GLOBALES ET PAR MODE
    # ============================================================
    print("\n--- Calcul des métriques de corrélation ---")
    
    # 1. Coefficient RV
    num_RV = np.sum(s**2) 
    den_RV = np.sqrt(np.sum(Cxx**2) * np.sum(Cyy**2))
    RV_coeff = num_RV / den_RV
    print(f"Coefficient RV global : {RV_coeff:.4f}")

    # 2. Homogeneous Variance Fraction (HVF) pour le Mode 1
    # Calcul matriciel direct de la variance expliquée (Traces)
    trace_Cxx = np.trace(Cxx)
    trace_Cyy = np.trace(Cyy)
    
    # U[:, 0] et Vt[0, :] sont des vecteurs 1D. La forme u.T @ M @ u donne la variance projetée.
    var_expl_X = U[:, 0] @ Cxx @ U[:, 0] 
    HVF_X = (var_expl_X / trace_Cxx) * 100
    
    var_expl_Y = Vt[0, :] @ Cyy @ Vt[0, :]
    HVF_Y = (var_expl_Y / trace_Cyy) * 100
    
    print(f"Variance interne SST expliquée par son Mode 1 (HVF) : {HVF_X:.2f}%")
    print(f"Variance interne SLP expliquée par son Mode 1 (HVF) : {HVF_Y:.2f}%")

    # 3. Séries temporelles et Fraction de Variance Croisée (VF)
    mode_idx = 0
    A1_list, B1_list, Y_all_list = [], [], []
    
    for member in train_members:
        X, Y, _, _, _,_,_ = load_member_mca_data(
            member, path_SST, path_SLP, winter_months, 
            sst_lags=args.sst_lags, duree_lissage=args.duree_lissage,roll_sst=args.roll_sst, monthly_reduction=args.monthly_reduction, lat_weight=args.lat_weight,sst_std=dynamic_sst_std, slp_std=dynamic_slp_std
        )
        A1_list.append(X @ U[:, mode_idx])
        B1_list.append(Y @ Vt[mode_idx, :].T)
        Y_all_list.append(Y)
        
    A1 = np.concatenate(A1_list)
    B1 = np.concatenate(B1_list)
    Y_mat_all = np.concatenate(Y_all_list)

    # Corrélation Temporelle (R)
    R_value = np.corrcoef(A1, B1)[0, 1]
    print(f"Corrélation temporelle R (Mode 1) : {R_value:.3f}")

    # Fraction de Variance (VF croisée)
    A1_norm = (A1 - np.mean(A1)) / np.std(A1)
    Y_predicted_by_X = np.outer(A1_norm, (A1_norm.T @ Y_mat_all) / len(A1_norm))
    var_explained = np.var(Y_predicted_by_X)
    var_total = np.var(Y_mat_all)
    VF = (var_explained / var_total) * 100
    print(f"Fraction de Variance croisée (VF SLP expliquée par SST) : {VF:.2f}%")

    # === GÉNÉRATION DU PLOT DE SYNTHÈSE ===
    print("Génération du dashboard des métriques...")
    plot_mca_metrics_dashboard(RV_coeff, R_value, VF, HVF_X, HVF_Y, A1, B1, outdir)
    print(f"Dashboard sauvegardé dans : {os.path.join(outdir, 'mca_metrics_dashboard.png')}")