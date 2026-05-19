import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import joblib # Pour charger le PCA
from datetime import timedelta
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import argparse



# ==========================================
# 1. GÉNÉRATION DE LA MASTER REFERENCE
# ==========================================
def generate_master_reference(member_ids, sst_lags, slp_lags, n_quantiles=4):
    slp_std, sst_std = 596.0, 0.707
    vmax_plot_mean = 2.0
    vmax_plot_std = 2.0 # Ajuster pour bien voir les contrastes de variance
    vmax_plot_std_diff = 0.3 # Echelle pour l'anomalie d'écart-type

    print("Chargement du modèle PCA...")
    projector = joblib.load(args.model_path)

    print("\n=== PASSE 1 : Extraction de la PC1 ===")
    all_latents = []
    all_dates = [] 
    
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
                
                # Projection directe avec PCA
                slp_flat = slp_array.reshape(len(dates), -1)
                latent = projector.transform(slp_flat)
                all_latents.append(latent)
                print(f"Membre {mem} projeté.", end='\r')
        except FileNotFoundError:
            print(f"Fichier manquant pour {mem}, ignoré.")
            
    print(f"\nPasse 1 terminée en {time.time() - start_time:.2f}s")
    
    global_latents = np.vstack(all_latents)
    pc1 = global_latents[:, 0] # On extrait uniquement la PC1
    
    # --------------------------------------------------------
    # DISTRIBUTION DE LA PC1
    # --------------------------------------------------------
    print(f"Calcul des {n_quantiles} quantiles et tracé de la distribution de la PC1...")
    global_labels, bins = pd.qcut(pc1, q=n_quantiles, labels=False, retbins=True)
    
    plt.figure(figsize=(8, 5))
    plt.hist(pc1, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    for b in bins:
        plt.axvline(b, color='red', linestyle='--', linewidth=1.5)
    plt.title(f"Distribution de la PC1 - {n_quantiles} quantiles", fontsize=14, fontweight='bold')
    plt.xlabel("Amplitude de la PC1")
    plt.ylabel("Nombre de jours")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    dist_fig_path = f"{base_home}/PC1_Distribution_{n_quantiles}_quantiles.png"
    plt.savefig(dist_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure de distribution sauvegardée : {dist_fig_path}")

    sorted_groups = list(range(n_quantiles))
    sorted_counts = [np.sum(global_labels == k) for k in sorted_groups]
    group_names = [f"Q{k+1}" for k in sorted_groups]
    
    # --------------------------------------------------------
    # PASSE 2 : ACCUMULATION (Somme et Somme des Carrés)
    # --------------------------------------------------------
    print("\n=== PASSE 2 : Accumulation Spatiale ===")
    sums = {k: {'slp_0': 0} for k in sorted_groups}
    sums_sq = {k: {'slp_0': 0} for k in sorted_groups} 
    valid_counts = {k: {'slp_0': 0} for k in sorted_groups}
    
    for k in sorted_groups:
        for lag in slp_lags:
            sums[k][f'slp_lag_{lag}'] = 0; sums_sq[k][f'slp_lag_{lag}'] = 0; valid_counts[k][f'slp_lag_{lag}'] = 0
        for lag in sst_lags:
            sums[k][f'sst_lag_{lag}'] = 0; sums_sq[k][f'sst_lag_{lag}'] = 0; valid_counts[k][f'sst_lag_{lag}'] = 0
            
    date_to_label = {d: l for d, l in zip(all_dates, global_labels)}
    
    start_time = time.time()
    for mem in member_ids:
        file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}_{duree_lissage}d.nc')
        file_sst = os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid.nc')
        
        if not os.path.exists(file_slp) or not os.path.exists(file_sst):
            continue
            
        with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
            ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15,70))
            mem_dates = [t for m, t in all_dates if m == mem]
            
            for group_id in sorted_groups:
                dates_for_group = [d for d in mem_dates if date_to_label[(mem, d)] == group_id]
                
                if not dates_for_group:
                    continue
                
                # Fonction utilitaire pour éviter la répétition
                def process_var(ds, var_name, dates, lag, norm_std, key):
                    target_dates = [d - timedelta(days=lag) for d in dates] if lag > 0 else dates
                    valid = np.intersect1d(target_dates, ds.time.values)
                    if len(valid) > 0:
                        vals = ds[var_name].sel(time=valid).values / norm_std
                        vals = np.nan_to_num(vals, nan=0.0)
                        sums[group_id][key] += vals.sum(axis=0)
                        sums_sq[group_id][key] += (vals**2).sum(axis=0)
                        valid_counts[group_id][key] += len(valid)

                process_var(ds_slp, 'PSL', dates_for_group, 0, slp_std, 'slp_0')
                for lag in slp_lags:
                    process_var(ds_slp, 'PSL', dates_for_group, lag, slp_std, f'slp_lag_{lag}')
                for lag in sst_lags:
                    process_var(ds_sst, 'SST', dates_for_group, lag, sst_std, f'sst_lag_{lag}')

        print(f"Cartes accumulées pour le membre {mem}.", end='\r')

    print(f"\nPasse 2 terminée en {time.time() - start_time:.2f}s")
    
    # --------------------------------------------------------
    # CALCUL DES MOYENNES, VARIANCES, ET GESTION DE LA LIGNE "GLOBALE"
    # --------------------------------------------------------
    print("\nCalcul des statistiques et sauvegarde...")
    master_dict = {}
    
    composites_mean = {k: {} for k in sorted_groups}
    composites_var  = {k: {} for k in sorted_groups}
    composites_std  = {k: {} for k in sorted_groups}
    
    # Statistiques Globales (indépendantes des quantiles)
    global_mean = {}
    global_var = {}
    global_std = {}
    
    all_keys = ['slp_0'] + [f'slp_lag_{l}' for l in slp_lags] + [f'sst_lag_{l}' for l in sst_lags]
    
    # Calcul Global direct via sommes des quantiles
    for key in all_keys:
        tot_sum = sum(sums[k][key] for k in sorted_groups)
        tot_sum_sq = sum(sums_sq[k][key] for k in sorted_groups)
        tot_count = sum(valid_counts[k][key] for k in sorted_groups)
        
        g_mean = tot_sum / max(1, tot_count)
        g_var = (tot_sum_sq / max(1, tot_count)) - (g_mean**2)
        
        global_mean[key] = g_mean
        global_var[key] = g_var
        global_std[key] = np.sqrt(np.maximum(0, g_var))
        
        master_dict[f"GLOBAL_{key}_mean"] = g_mean
        master_dict[f"GLOBAL_{key}_var"] = g_var
        master_dict[f"GLOBAL_{key}_std"] = np.sqrt(np.maximum(0, g_var))
    # Calcul par Quantile
    for i, k in enumerate(sorted_groups):
        prefix = f"quantile_{group_names[i]}"
        master_dict[f"{prefix}_count"] = sorted_counts[i]
        
        for key in all_keys:
            N_val = max(1, valid_counts[k][key])
            mean_val = sums[k][key] / N_val
            var_val  = (sums_sq[k][key] / N_val) - (mean_val**2)
            std_val = np.sqrt(np.maximum(0, var_val))

            master_dict[f"{prefix}_{key}_mean"] = mean_val
            master_dict[f"{prefix}_{key}_var"] = var_val
            master_dict[f"{prefix}_{key}_std"] = std_val
            composites_mean[k][key] = mean_val
            composites_var[k][key] = var_val
            composites_std[k][key] = std_val


    master_dict['pc1_bins'] = bins
    save_path = f"{base_home}/master_reference_quantiles_PCA.npz"
    np.savez(save_path, **master_dict)
    
    # --------------------------------------------------------
    # VISUALISATION DES RÉSULTATS (Avec Ligne Globale)
    # --------------------------------------------------------
    print("\nGénération des figures récapitulatives...")
    
    n_cols = 1 + len(slp_lags) + len(sst_lags)
    # n_quantiles + 1 (pour ajouter la ligne globale)
    n_rows = n_quantiles + 1 
    
    my_norm_mean = mcolors.SymLogNorm(linthresh=0.5, vmin=-vmax_plot_mean, vmax=vmax_plot_mean, base=10)
    cbar_ticks_mean = [-2, -1, -0.4, 0, 0.4, 1, 2]
    
    fig_m, axes_m = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), 
                                 subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False)
    
    fig_v, axes_v = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), 
                                 subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False)

    # FIGURE ANOMALIE ÉCARTS-TYPES
    fig_d, axes_d = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), 
                                 subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False)


    N_total = sum(sorted_counts)

    # Lignes pour chaque Quantile
    for i, k in enumerate(sorted_groups):
        count = sorted_counts[i]
        pct = (count / N_total) * 100
        col_idx = 0
        
        for key in all_keys:
            # Gestion des extents et des colormaps selon SLP/SST
            is_slp = 'slp' in key
            cmap_mean = 'RdBu_r'
            extent = [-100, 40, 20, 70] if is_slp else [-180, 180, -15, 70]
            title_prefix = "SLP" if is_slp else "SST"
            title_suffix = "t=0" if key == 'slp_0' else f"-{key.split('_')[-1]}j"
            
            # --- MOYENNE ---
            ax_m = axes_m[i, col_idx]
            im_m = ax_m.imshow(composites_mean[k][key], transform=ccrs.PlateCarree(), cmap=cmap_mean, origin='lower', extent=extent, norm=my_norm_mean)
            ax_m.set_extent(extent, crs=ccrs.PlateCarree()) # Force le zoom
            ax_m.coastlines(color='black', linewidth=0.8, alpha=0.7) # Ajoute les côtes
            if col_idx == 0:
                ax_m.set_title(f"{group_names[i]} Mean\n(N={count}, {pct:.1f}%)", fontweight='bold')
            else:
                ax_m.set_title(f"Mean {title_prefix} ({title_suffix})")
            fig_m.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04, ticks=cbar_ticks_mean, format="%g")
            
            # --- ÉCART-TYPE ---
            ax_v = axes_v[i, col_idx]
            im_v = ax_v.imshow(composites_std[k][key], transform=ccrs.PlateCarree(), cmap='Reds', origin='lower', extent=extent, vmax=vmax_plot_std)
            ax_v.set_extent(extent, crs=ccrs.PlateCarree()) # Force le zoom
            ax_v.coastlines(color='black', linewidth=0.8, alpha=0.7) # Ajoute les côtes
            if col_idx == 0:
                ax_v.set_title(f"{group_names[i]} Écart-type\n(N={count}, {pct:.1f}%)", fontweight='bold')
            else:
                ax_v.set_title(f"Écart-type {title_prefix} ({title_suffix})")
            fig_v.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)

            # 3. ANOMALIE D'ÉCART-TYPE (Différence)
            diff_std = composites_std[k][key] - global_std[key]
            ax_d = axes_d[i, col_idx]
            im_d = ax_d.imshow(diff_std, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, vmin=-vmax_plot_std_diff, vmax=vmax_plot_std_diff)
            ax_d.set_extent(extent, crs=ccrs.PlateCarree()) # Force le zoom
            ax_d.coastlines(color='black', linewidth=0.8, alpha=0.7)
            ax_d.set_title(f"{group_names[i]} Anom Std\n(N={count}, {pct:.1f}%)" if col_idx == 0 else f"Anom Std {title_prefix} ({title_suffix})", fontweight='bold' if col_idx == 0 else 'normal')
            fig_d.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)    
            
            col_idx += 1

    # ================= LIGNE FINALE : GLOBALE =================
    col_idx = 0
    for key in all_keys:
        is_slp = 'slp' in key
        extent = [-100, 40, 20, 70] if is_slp else [-180, 180, -15, 70]
        title_prefix = "SLP" if is_slp else "SST"
        title_suffix = "t=0" if key == 'slp_0' else f"-{key.split('_')[-1]}j"
        
        # Globale Moyenne
        ax_m = axes_m[n_quantiles, col_idx]
        im_m = ax_m.imshow(global_mean[key], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, norm=my_norm_mean)
        ax_m.set_extent(extent, crs=ccrs.PlateCarree()) # Force le zoom
        ax_m.coastlines(color='black', linewidth=0.8, alpha=0.7) # Ajoute les côtes
        if col_idx == 0:
            ax_m.set_title("GLOBALE (Moyenne)", fontweight='bold')
        else:
            ax_m.set_title(f"Glob. Moy {title_prefix} ({title_suffix})")
        fig_m.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04, ticks=cbar_ticks_mean, format="%g")
        
        # Globale Écart-type
        ax_v = axes_v[n_quantiles, col_idx]
        im_v = ax_v.imshow(global_std[key], transform=ccrs.PlateCarree(), cmap='Reds', origin='lower', extent=extent, vmax=vmax_plot_std)
        ax_v.set_extent(extent, crs=ccrs.PlateCarree()) # Force le zoom
        ax_v.coastlines(color='black', linewidth=0.8, alpha=0.7) # Ajoute les côtes

        if col_idx == 0:
            ax_v.set_title("GLOBALE (Écart-type)", fontweight='bold')
        else:
            ax_v.set_title(f"Glob. Écart-type {title_prefix} ({title_suffix})")
        fig_v.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)

        # Sur la figure des anomalies, on met l'écart-type global pur en bas pour avoir la référence visuelle
        ax_d = axes_d[n_quantiles, col_idx]
        im_d_ref = ax_d.imshow(global_std[key], transform=ccrs.PlateCarree(), cmap='Reds', origin='lower', extent=extent, vmax=vmax_plot_std)
        ax_d.set_extent(extent, crs=ccrs.PlateCarree()) # Force le zoom
        ax_d.coastlines(color='black', linewidth=0.8, alpha=0.7)
        ax_d.set_title("GLOBALE (Référence Absolue)" if col_idx == 0 else f"Glob. Std {title_prefix} ({title_suffix})", fontweight='bold' if col_idx == 0 else 'normal')
        fig_d.colorbar(im_d_ref, ax=ax_d, fraction=0.046, pad=0.04)

        
        col_idx += 1

    # Finalisation des figures
    fig_m.suptitle(f"Composites Moyens (Quantiles PC1) sur {len(member_ids)} membres", fontsize=18, y=1.02)
    fig_m.tight_layout()
    fig_name_mean = f"{base_home}/Quantiles_Means_PCA.png"
    fig_m.savefig(fig_name_mean, dpi=200, bbox_inches='tight')
    plt.close(fig_m)
    
    fig_v.suptitle(f"Cartes de Écarts-types (Quantiles PC1) sur {len(member_ids)} membres", fontsize=18, y=1.02)
    fig_v.tight_layout()
    fig_name_var = f"{base_home}/Quantiles_Ecarts-types_PCA.png"
    fig_v.savefig(fig_name_var, dpi=200, bbox_inches='tight')
    plt.close(fig_v)

    fig_d.suptitle(f"Anomalies d'Écarts-types (Quantiles PC1) sur {len(member_ids)} membres", fontsize=18, y=1.02)
    fig_d.tight_layout()
    fig_name_diff = f"{base_home}/Quantiles_Anomalies_Ecarts-types_PCA.png"
    fig_d.savefig(fig_name_diff, dpi=200, bbox_inches='tight')
    plt.close(fig_d)
    
    print(f"Figures sauvegardées :\n - {fig_name_mean}\n - {fig_name_var}")

if __name__ == "__main__":
    # ============================================================
    # ARGUMENTS & PATHS
    # ============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
    parser.add_argument('--model_path', type=str, required=True, help="Chemin vers le modèle PCA (.pkl) pré-entraîné")
    parser.add_argument('--n_quantiles', type=int, default=4, help="Nombre de quantiles pour la PC1")
    parser.add_argument('--duree_lissage', type=int, help="Durée du lissage en jours (10 ou 30)")
    parser.add_argument('--normalize', action='store_true', help="PC normalisée (True) ou non (False) (pour le nom du dossier, doit matcher le nom du modèle PCA utilisé)")
    parser.add_argument('--number_of_members', type=int, default=89, help="Nombre de membres à utiliser (doit être <= 89)")
    args = parser.parse_args()

    # parameters
    number_of_members = args.number_of_members
    sst_lags = [35, 65, 95, 140, 175, 210, 245, 280, 315, 350] 
    slp_lags = [15, 30, 45, 60]      
    duree_lissage = args.duree_lissage

    # Dossiers adaptés en enlevant l'argument 'embedding_method' devenu obsolète
    if args.machine == 'hacienda':
        base_home = f"/home/moysan/stage_isir_jz/data_analysis/composites_pc1_quantiles/pc1_{args.n_quantiles}_quantiles_master_ref_generator_{number_of_members}members_normalize{args.normalize}_duree_lissage{duree_lissage}/"
        data_dir = "/data/moysan/data/"
    elif args.machine == 'jean-zay-work': 
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/composites_pc1_quantiles/pc1_{args.n_quantiles}_quantiles_master_ref_generator_{number_of_members}members_normalize{args.normalize}_duree_lissage{duree_lissage}/"
        data_dir = "/lustre/fswork/projects/rech/uxg/uca57ub/data/"
    elif args.machine == 'jean-zay-scratch':
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/composites_pc1_quantiles/pc1_{args.n_quantiles}_quantiles_master_ref_generator_{number_of_members}members_normalize{args.normalize}_duree_lissage{duree_lissage}/"
        data_dir = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/"

    os.makedirs(base_home, exist_ok=True)
    path_sst = os.path.join(data_dir, "SST/")
    path_slp = os.path.join(data_dir, "SLP/")

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    members_used = all_members[:number_of_members]
    
    generate_master_reference(members_used, sst_lags, slp_lags, n_quantiles=args.n_quantiles)