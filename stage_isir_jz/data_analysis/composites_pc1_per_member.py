import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import joblib # Pour charger le PCA
from datetime import timedelta
import time
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import argparse

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

import calendar
from shared_tools.evaluation_functions import compute_two_bootstraps, plot_two_bootstrap_histograms, stats, plot_time_series

# ==========================================
# 1.5 FONCTION UTILITAIRE POUR LES CHEMINS
# ==========================================
def get_file_paths(path_slp, path_sst, mem, duree_lissage, monthly_reduction):
    if monthly_reduction:
        file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}_1mo.nc')
        file_sst = os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid_1mo.nc')
    else:
        if duree_lissage != 0:
            file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}_{duree_lissage}d.nc')
        else:
            file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}.nc')
        file_sst = os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid.nc')
    return file_slp, file_sst

# ==========================================
# 1. GÉNÉRATION DE LA MASTER REFERENCE
# ==========================================
def generate_master_reference(member_ids, sst_lags, slp_lags, winter_months,duree_lissage,monthly_reduction,lat_weight,base_home,n_quantiles=4,slp_std=596.0):
    
    print("Chargement du modèle PCA...")
    projector = joblib.load(args.model_path)

    print("\n=== PASSE 1 : Extraction de la PC1 ===")
    all_latents = []
    all_dates = [] 
    
    start_time = time.time()
    for mem in member_ids:
        file_slp, _ = get_file_paths(path_slp, path_sst, mem, duree_lissage, monthly_reduction)
        try:
            with xr.open_dataset(file_slp) as ds_slp:
                # APRÈS : On coupe strictement tout ce qui dépasse le 31 décembre 2014
                ds_winter = ds_slp.sel(
                    time=slice(None, "2014-12-31"),
                    drop=False,  # Sécurité
                )
                ds_winter = ds_slp.sel(time=ds_slp['time'].dt.month.isin(winter_months))
                dates = ds_winter.time.values
                all_dates.extend([(mem, t) for t in dates])
                
                slp_array = ds_winter['PSL'].values / slp_std
                slp_array = np.nan_to_num(slp_array, nan=0.0)

                if lat_weight:
                    lats = ds_winter['lat'].values
                    coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
                    wgts = np.sqrt(coslat).reshape(1, len(lats), 1)
                    slp_array = slp_array * wgts
                
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
    plt.title(f"PC1 Distribution - {n_quantiles} quantiles", fontsize=14, fontweight='bold')
    plt.xlabel("PC1 Magnitude")
    plt.ylabel("Number of days")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    dist_fig_path = f"{base_home}/PC1_Distribution_{n_quantiles}_quantiles.png"
    plt.savefig(dist_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure de distribution sauvegardée : {dist_fig_path}")

    sorted_groups = list(range(n_quantiles))
    sorted_counts = [np.sum(global_labels == k) for k in sorted_groups]
    group_names = [f"Q{k+1}" for k in sorted_groups]
    pc1_means_per_quantile = [np.mean(pc1[global_labels == k]) for k in sorted_groups]
    
    # --------------------------------------------------------
    # PASSE 2 : ACCUMULATION (Somme et Somme des Carrés)
    # --------------------------------------------------------
    print("\n=== PASSE 2 : Accumulation Spatiale ===")
    
    all_keys = ['slp_0'] + [f'slp_lag_{lag}' for lag in slp_lags] + [f'sst_lag_{lag}' for lag in sst_lags]
    
    sums = {k: {key: 0 for key in all_keys} for k in sorted_groups}
    sums_sq = {k: {key: 0 for key in all_keys} for k in sorted_groups} 
    valid_counts = {k: {key: 0 for key in all_keys} for k in sorted_groups}
    
    # NOUVEAU : Accumulateurs par membre pour générer les heatmaps détaillées
    mem_sums = {mem: {k: {key: 0 for key in all_keys} for k in sorted_groups} for mem in member_ids}
    mem_valid_counts = {mem: {k: {key: 0 for key in all_keys} for k in sorted_groups} for mem in member_ids}

    date_to_label = {d: l for d, l in zip(all_dates, global_labels)}
    
    start_time = time.time()
    for mem in member_ids:
        file_slp_mem, file_sst_mem = get_file_paths(path_slp, path_sst, mem, duree_lissage, monthly_reduction)
        
        if not os.path.exists(file_slp_mem) or not os.path.exists(file_sst_mem):
            continue
            
        with xr.open_dataset(file_slp_mem) as ds_slp, xr.open_dataset(file_sst_mem) as ds_sst:
            ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15,70))
            mem_dates = [t for m, t in all_dates if m == mem]
            
            for group_id in sorted_groups:
                dates_for_group = [d for d in mem_dates if date_to_label[(mem, d)] == group_id]
                
                if not dates_for_group:
                    continue
                
                # Fonction utilitaire pour éviter la répétition
                def process_var(ds, var_name, dates, lag, key):
                    if not monthly_reduction:
                        target_dates = [d - timedelta(days=lag) for d in dates]
                    else:
                        target_dates = []
                        for d in dates:
                            y_shift = (d.month - lag - 1) // 12
                            new_month = (d.month - lag - 1) % 12 + 1
                            target_dates.append(d.replace(year=d.year + y_shift, month=new_month))
                    valid = np.intersect1d(target_dates, ds.time.values)
                    if len(valid) > 0:
                        vals = ds[var_name].sel(time=valid).values
                        vals = np.nan_to_num(vals, nan=0.0)
                        
                        s_vals = vals.sum(axis=0)
                        s_sq_vals = (vals**2).sum(axis=0)
                        count_val = len(valid)
                        
                        # Accumulation Globale
                        sums[group_id][key] += s_vals
                        sums_sq[group_id][key] += s_sq_vals
                        valid_counts[group_id][key] += count_val
                        
                        # Accumulation par Membre
                        mem_sums[mem][group_id][key] += s_vals
                        mem_valid_counts[mem][group_id][key] += count_val

                process_var(ds_slp, 'PSL', dates_for_group, 0, 'slp_0')
                for lag in slp_lags:
                    process_var(ds_slp, 'PSL', dates_for_group, lag, f'slp_lag_{lag}')
                for lag in sst_lags:
                    process_var(ds_sst, 'SST', dates_for_group, lag, f'sst_lag_{lag}')

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
    
    global_mean = {}
    global_var = {}
    global_std = {}
    
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
    col_widths = [1.0 if "slp" in key else 1.5 for key in all_keys]
    
    # n_quantiles + 1 (pour ajouter la ligne globale)
    n_rows = n_quantiles + 1 
    
    cbar_ticks_mean_normalized = [-2, -1, -0.4, 0, 0.4, 1, 2]
    magnitude_slp = 500
    magnitude_sst = 0.5
    cbar_ticks_mean_sst = magnitude_sst * np.array(cbar_ticks_mean_normalized)
    cbar_ticks_mean_slp = magnitude_slp * np.array(cbar_ticks_mean_normalized)
    
    fig_m, axes_m = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), 
                                 subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={"width_ratios": col_widths}, squeeze=False)
    
    fig_v, axes_v = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), 
                                 subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={"width_ratios": col_widths}, squeeze=False)

    fig_d, axes_d = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), 
                                 subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={"width_ratios": col_widths}, squeeze=False)

    fig_snr, axes_snr = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={"width_ratios": col_widths}, squeeze=False)

    N_total = sum(sorted_counts)
    unit_time = "month(s)" if monthly_reduction else "day(s)"

    # Lignes pour chaque Quantile
    for i, k in enumerate(sorted_groups):
        count = sorted_counts[i]
        pct = (count / N_total) * 100
        col_idx = 0
        
        for key in all_keys:
            is_slp = 'slp' in key
            cmap_mean = 'RdBu_r'
            extent = [-100, 40, 20, 70] if is_slp else [-180, 180, -15, 70]
            title_prefix = "SLP" if is_slp else "SST"
            
            lag_val_str = key.split('_')[-1]
            lag_val = int(lag_val_str)
            
            if key == 'slp_0':
                title_suffix = "t=0"
            elif lag_val < 0:
                title_suffix = f"t+{abs(lag_val)} [{unit_time}]"
            elif lag_val ==0:
                title_suffix = f"current t [{unit_time}]"
            else:
                title_suffix = f"t-{lag_val} [{unit_time}]"

            unit_label = "Pa" if is_slp else "K"
            vmax_std_slp = 2*magnitude_slp
            vmax_std_sst = 2*magnitude_sst
            vmax_diff_sst = 0.3*magnitude_sst
            vmax_diff_slp = 0.3*magnitude_slp
            cbar_shrink = 0.7 if is_slp else 0.6

            if is_slp:
                vmax_mean = 2*magnitude_slp
                linthresh_val = 0.4*magnitude_slp  
                cbar_ticks_mean = cbar_ticks_mean_slp
                vmax_std = 2*magnitude_slp
                vmax_diff = 0.3*magnitude_slp
                my_norm_mean = mcolors.Normalize(vmin=-vmax_mean, vmax=vmax_mean)
            else:
                vmax_mean = 2*magnitude_sst
                linthresh_val = 0.4 * magnitude_sst 
                cbar_ticks_mean = cbar_ticks_mean_sst
                vmax_std = 2*magnitude_sst
                vmax_diff = 0.3*magnitude_sst
                my_norm_mean = mcolors.SymLogNorm(linthresh=linthresh_val, vmin=-vmax_mean, vmax=vmax_mean, base=10)
            
            # --- MOYENNE ---
            ax_m = axes_m[i, col_idx]
            im_m = ax_m.imshow(composites_mean[k][key], transform=ccrs.PlateCarree(), cmap=cmap_mean, origin='lower', extent=extent, norm=my_norm_mean)
            ax_m.set_extent(extent, crs=ccrs.PlateCarree()) 
            ax_m.coastlines(color='black', linewidth=0.8, alpha=0.7) 
            if col_idx == 0:
                ax_m.set_title(f"{group_names[i]} Mean SLP current t [{unit_time}] \n(N={count}, {pct:.1f}%)", fontweight='bold')
            else:
                ax_m.set_title(f"Mean {title_prefix} {title_suffix}")
            cbar_m = fig_m.colorbar(im_m, ax=ax_m, fraction=0.035, shrink=cbar_shrink, pad=0.04,format="%g")
            cbar_m.set_label(f'Anom ({unit_label})')
            cbar_m.set_ticks(cbar_ticks_mean)  
            
            # --- ÉCART-TYPE ---
            ax_v = axes_v[i, col_idx]
            im_v = ax_v.imshow(composites_std[k][key], transform=ccrs.PlateCarree(), cmap='Reds', origin='lower', extent=extent, vmax=vmax_std)
            ax_v.set_extent(extent, crs=ccrs.PlateCarree()) 
            ax_v.coastlines(color='black', linewidth=0.8, alpha=0.7) 
            if col_idx == 0:
                ax_v.set_title(f"{group_names[i]} Standard deviation SLP current t [{unit_time}] \n(N={count}, {pct:.1f}%)", fontweight='bold')
            else:
                ax_v.set_title(f"Standard deviation {title_prefix} {title_suffix}")
            cbar_v = fig_v.colorbar(im_v, ax=ax_v, fraction=0.035, shrink=cbar_shrink, pad=0.04)
            cbar_v.set_label(f'Std ({unit_label})')

            # --- ANOMALIE D'ÉCART-TYPE (Différence) ---
            diff_std = composites_std[k][key] - global_std[key]
            ax_d = axes_d[i, col_idx]
            im_d = ax_d.imshow(diff_std, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, vmin=-vmax_diff, vmax=vmax_diff)
            ax_d.set_extent(extent, crs=ccrs.PlateCarree()) 
            ax_d.coastlines(color='black', linewidth=0.8, alpha=0.7)
            ax_d.set_title(f"{group_names[i]} Anom Std\n(N={count}, {pct:.1f}%)" if col_idx == 0 else f"Anom Std {title_prefix} {title_suffix}", fontweight='bold' if col_idx == 0 else 'normal')
            cbar_d = fig_d.colorbar(im_d, ax=ax_d, fraction=0.035, shrink=cbar_shrink, pad=0.04)
            cbar_d.set_label(f'ΔStd ({unit_label})')

            # --- CARTE DE SNR (Signal-to-Noise Ratio) ---
            mean_val = composites_mean[k][key]
            std_val = composites_std[k][key]
            snr = mean_val / np.maximum(std_val, 1e-5)
            
            ax_snr = axes_snr[i, col_idx]
            im_snr = ax_snr.imshow(snr, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, vmin=-1.5, vmax=1.5)
            ax_snr.set_extent(extent, crs=ccrs.PlateCarree())
            ax_snr.coastlines(color='black', linewidth=0.8, alpha=0.7)
            if col_idx == 0:
                ax_snr.set_title(f"{group_names[i]} SNR (Mean/Std) SLP current t [{unit_time}] \n(N={count})", fontweight='bold')
            else:
                ax_snr.set_title(f"SNR {title_prefix} {title_suffix}")
            cbar_snr = fig_snr.colorbar(im_snr, ax=ax_snr, fraction=0.035, shrink=cbar_shrink, pad=0.04)
            cbar_snr.set_label('SNR (unitless)')

            col_idx += 1

    # ================= LIGNE FINALE : GLOBALE =================
    col_idx = 0
    for key in all_keys:
        is_slp = 'slp' in key
        extent = [-100, 40, 20, 70] if is_slp else [-180, 180, -15, 70]
        title_prefix = "SLP" if is_slp else "SST"
        lag_val_str = key.split('_')[-1]
        lag_val = int(lag_val_str)
        
        if key == 'slp_0': title_suffix = "t=0"
        elif lag_val < 0:  title_suffix = f"t+{abs(lag_val)}{unit_time}"
        elif lag_val ==0:  title_suffix = f"current t {unit_time}"
        else:              title_suffix = f"t-{lag_val}{unit_time}"
        
        unit_label = "Pa" if is_slp else "K"
        cbar_ticks_mean = cbar_ticks_mean_slp if is_slp else cbar_ticks_mean_sst
        vmax_std = vmax_std_slp if is_slp else vmax_std_sst
        cbar_shrink = 0.85 if is_slp else 0.65

        if is_slp:
            vmax_mean = 2 * magnitude_slp
            my_norm_mean = mcolors.Normalize(vmin=-vmax_mean, vmax=vmax_mean)
        else:
            vmax_mean = 2 * magnitude_sst
            linthresh_val = 0.4 * magnitude_sst
            my_norm_mean = mcolors.SymLogNorm(linthresh=linthresh_val, vmin=-vmax_mean, vmax=vmax_mean, base=10)
        
        ax_m = axes_m[n_quantiles, col_idx]
        im_m = ax_m.imshow(global_mean[key], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, norm=my_norm_mean)
        ax_m.set_extent(extent, crs=ccrs.PlateCarree()) 
        ax_m.coastlines(color='black', linewidth=0.8, alpha=0.7) 
        if col_idx == 0:
            ax_m.set_title("GLOBALE (Moyenne)", fontweight='bold')
        else:
            ax_m.set_title(f"Glob. Moy {title_prefix} ({title_suffix})")
        cbar_gm = fig_m.colorbar(im_m, ax=ax_m, fraction=0.035, shrink=cbar_shrink, pad=0.04, ticks=cbar_ticks_mean, format="%g")
        cbar_gm.set_label(f'Anom ({unit_label})')
        cbar_gm.set_ticks(cbar_ticks_mean)  
        
        ax_v = axes_v[n_quantiles, col_idx]
        im_v = ax_v.imshow(global_std[key], transform=ccrs.PlateCarree(), cmap='Reds', origin='lower', extent=extent, vmax=vmax_std)
        ax_v.set_extent(extent, crs=ccrs.PlateCarree()) 
        ax_v.coastlines(color='black', linewidth=0.8, alpha=0.7) 

        if col_idx == 0:
            ax_v.set_title("GLOBALE (Écart-type)", fontweight='bold')
        else:
            ax_v.set_title(f"Glob. Écart-type {title_prefix} ({title_suffix})")
        cbar_gv = fig_v.colorbar(im_v, ax=ax_v, fraction=0.035, shrink=cbar_shrink, pad=0.04)
        cbar_gv.set_label(f'Std ({unit_label})')

        ax_d = axes_d[n_quantiles, col_idx]
        im_d_ref = ax_d.imshow(global_std[key], transform=ccrs.PlateCarree(), cmap='Reds', origin='lower', extent=extent, vmax=vmax_std)
        ax_d.set_extent(extent, crs=ccrs.PlateCarree()) 
        ax_d.coastlines(color='black', linewidth=0.8, alpha=0.7)
        ax_d.set_title("GLOBALE (Référence Absolue)" if col_idx == 0 else f"Glob. Std {title_prefix} ({title_suffix})", fontweight='bold' if col_idx == 0 else 'normal')
        cbar_gd = fig_d.colorbar(im_d_ref, ax=ax_d, fraction=0.035, shrink=cbar_shrink, pad=0.04)
        cbar_gd.set_label(f'Std ({unit_label})')

        g_mean_val = global_mean[key]
        g_std_val = global_std[key]
        
        g_snr = g_mean_val / np.maximum(g_std_val, 1e-5)
        ax_snr = axes_snr[n_quantiles, col_idx]
        im_snr_g = ax_snr.imshow(g_snr, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, vmin=-1.5, vmax=1.5)
        ax_snr.set_extent(extent, crs=ccrs.PlateCarree())
        ax_snr.coastlines(color='black', linewidth=0.8, alpha=0.7)
        ax_snr.set_title("GLOBALE (SNR)" if col_idx == 0 else f"Glob. SNR {title_prefix} ({title_suffix})", fontweight='bold' if col_idx == 0 else 'normal')
        fig_snr.colorbar(im_snr_g, ax=ax_snr, fraction=0.035, shrink=cbar_shrink, pad=0.04).set_label('SNR')
        col_idx += 1

    fig_m.suptitle(f"Mean  (Quantiles PC1)", fontsize=18, y=1.02)
    fig_m.tight_layout()
    fig_m.savefig(f"{base_home}/Quantiles_Means_PCA.png", dpi=200, bbox_inches='tight')
    plt.close(fig_m)
    
    fig_v.suptitle(f"Cartes de Écarts-types (Quantiles PC1) sur {len(member_ids)} membres", fontsize=18, y=1.02)
    fig_v.tight_layout()
    fig_v.savefig(f"{base_home}/Quantiles_Ecarts-types_PCA.png", dpi=200, bbox_inches='tight')
    plt.close(fig_v)

    fig_d.suptitle(f"Anomalies d'Écarts-types (Quantiles PC1) sur {len(member_ids)} membres", fontsize=18, y=1.02)
    fig_d.tight_layout()
    fig_d.savefig(f"{base_home}/Quantiles_Anomalies_Ecarts-types_PCA.png", dpi=200, bbox_inches='tight')
    plt.close(fig_d)

    fig_snr.suptitle(f"Signal-to-Noise Ratio (Mean/Std) sur {len(member_ids)} membres", fontsize=18, y=1.02)
    fig_snr.tight_layout()
    fig_snr.savefig(f"{base_home}/Quantiles_SNR_PCA.png", dpi=200, bbox_inches='tight')
    plt.close(fig_snr)

    # =========================================================================
    # EXPORT INDIVIDUEL DES LIGNES (QUANTILES) POUR ANIMATION GIF
    # =========================================================================
    print("\nGénération des images par quantile (lignes individuelles pour GIFs)...")
    
    dir_gif_mean = f"{base_home}/GIF_lignes_Mean/"
    dir_gif_snr  = f"{base_home}/GIF_lignes_SNR/"
    os.makedirs(dir_gif_mean, exist_ok=True)
    os.makedirs(dir_gif_snr, exist_ok=True)

    unit_label_time = "months" if monthly_reduction else "days"

    for i, k in enumerate(sorted_groups):
        count = sorted_counts[i]
        pct = (count / N_total) * 100
        
        fig_row_m, axes_row_m = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 3.8), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={"width_ratios": col_widths}, squeeze=False)
        fig_row_snr, axes_row_snr = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 3.8), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={"width_ratios": col_widths}, squeeze=False)

        col_idx = 0
        for key in all_keys:
            is_slp = 'slp' in key
            extent = [-100, 40, 20, 70] if is_slp else [-180, 180, -15, 70]
            title_prefix = "SLP" if is_slp else "SST"
            cbar_shrink = 0.85 if is_slp else 0.65
            lag_val = int(key.split('_')[-1])
            
            if lag_val < 0:    title_suffix = f"Lag +{abs(lag_val)} [{unit_label_time}]"
            elif lag_val == 0: title_suffix = f"Lag 0 [{unit_label_time}]"
            else:              title_suffix = f"Lag -{lag_val} [{unit_label_time}]"

            unit_label = "Pa" if is_slp else "K"
            
            if is_slp:
                vmax_mean = 2 * magnitude_slp
                my_norm_mean = mcolors.Normalize(vmin=-vmax_mean, vmax=vmax_mean)
                cbar_ticks = cbar_ticks_mean_slp
            else:
                vmax_mean = 2 * magnitude_sst
                linthresh_val = 0.4 * magnitude_sst
                my_norm_mean = mcolors.SymLogNorm(linthresh=linthresh_val, vmin=-vmax_mean, vmax=vmax_mean, base=10)
                cbar_ticks = cbar_ticks_mean_sst

            ax_m = axes_row_m[0, col_idx]
            im_m = ax_m.imshow(composites_mean[k][key], transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, norm=my_norm_mean)
            ax_m.set_extent(extent, crs=ccrs.PlateCarree())
            ax_m.coastlines(color='black', linewidth=0.8, alpha=0.7)
            
            if col_idx == 0:
                ax_m.set_title(f"{group_names[i]} Mean SLP Lag 0 [{unit_label_time}]\n(N={count} {unit_label_time}, {pct:.1f}%)", fontweight='bold')
            else:
                ax_m.set_title(f"Mean {title_prefix} {title_suffix}")
                
            cbar_m = fig_row_m.colorbar(im_m, ax=ax_m, fraction=0.035, shrink=cbar_shrink, pad=0.04, format="%g", ticks=cbar_ticks)
            cbar_m.set_label(f'Anomaly ({unit_label})')

            ax_snr = axes_row_snr[0, col_idx]
            snr_val = composites_mean[k][key] / np.maximum(global_std[key], 1e-5)
            im_snr = ax_snr.imshow(snr_val, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, vmin=-1.5, vmax=1.5)
            ax_snr.set_extent(extent, crs=ccrs.PlateCarree())
            ax_snr.coastlines(color='black', linewidth=0.8, alpha=0.7)
            
            if col_idx == 0:
                ax_snr.set_title(f"{group_names[i]} SNR SLP Lag 0 [{unit_label_time}]\n(N={count} {unit_label_time}, {pct:.1f}%)", fontweight='bold')
            else:
                ax_snr.set_title(f"SNR {title_prefix} {title_suffix}")
                
            fig_row_snr.colorbar(im_snr, ax=ax_snr, fraction=0.035, shrink=cbar_shrink, pad=0.04).set_label('SNR (unitless)')

            col_idx += 1

        months_studied = []
        if 11 in winter_months: months_studied.append("November")
        if 12 in winter_months: months_studied.append("December")
        if 1 in winter_months:  months_studied.append("January")
        if 2 in winter_months:  months_studied.append("February")

        fig_row_m.suptitle(f"Mean PC1 Quantiles Composite {group_names[i]} ({args.number_of_members} members, {', '.join(months_studied)})", fontsize=16, y=0.98)
        fig_row_m.tight_layout(rect=[0, 0, 1, 0.96])
        fig_row_m.savefig(f"{dir_gif_mean}/Row_{i:02d}_{group_names[i]}_Mean.png", dpi=150, bbox_inches='tight')
        plt.close(fig_row_m)

        fig_row_snr.suptitle(f"SNR PC1 Quantiles Composite {group_names[i]} ({args.number_of_members} members, {', '.join(months_studied)})", fontsize=16, y=0.98)
        fig_row_snr.tight_layout(rect=[0, 0, 1, 0.96])
        fig_row_snr.savefig(f"{dir_gif_snr}/Row_{i:02d}_{group_names[i]}_SNR.png", dpi=150, bbox_inches='tight')
        plt.close(fig_row_snr)


    # =========================================================================
    # HEATMAPS 2D PAR MEMBRE ET PAR LAG (Barplot Agrégé + Heatmap Détaillée)
    # =========================================================================
    # =========================================================================
    # HEATMAPS 2D PAR MEMBRE ET PAR LAG (Énergie Totale & Projection Spatiale)
    # =========================================================================
    def generate_member_heatmap_per_lag(keys_subset, domain_name, wgts_domain):
        print(f"\nGénération des heatmaps Membres vs Quantiles pour {domain_name}...")
        
        unit_label_time = "months" if monthly_reduction else "days"
        
        for key in keys_subset:
            lag_val = int(key.split('_')[-1])
            if lag_val < 0:    label_lag = f"Lag +{abs(lag_val)} [{unit_label_time}]"
            elif lag_val == 0: label_lag = f"Lag 0 [{unit_label_time}]"
            else:              label_lag = f"Lag -{lag_val} [{unit_label_time}]"
            
            matrix_energy = np.zeros((n_quantiles, len(member_ids)))
            matrix_beta = np.zeros((n_quantiles, len(member_ids)))
            
            # --- Carte Globale de référence pour ce lag (Tous membres confondus) ---
            g_snr_ref = global_mean[key] / np.maximum(global_std[key], 1e-5)
            w_map = wgts_domain[0, :, :] if (lat_weight and wgts_domain is not None) else 1.0
            den_beta = np.sum(w_map * (g_snr_ref**2))
            
            for i, k in enumerate(sorted_groups):
                for j, mem in enumerate(member_ids):
                    N_mem = mem_valid_counts[mem][k][key]
                    if N_mem > 0:
                        mem_mean = mem_sums[mem][k][key] / N_mem
                        snr_map = mem_mean / np.maximum(global_std[key], 1e-5)
                        
                        # 1. ÉNERGIE (Norme L2)
                        total_snr = np.sqrt(np.sum((snr_map**2) * w_map))
                        matrix_energy[i, j] = total_snr
                        
                        # 2. SIMILARITÉ SPATIALE (Régression Linéaire : Pattern Amplitude)
                        num_beta = np.sum(w_map * snr_map * g_snr_ref)
                        beta = num_beta / max(den_beta, 1e-10)
                        matrix_beta[i, j] = beta
                    else:
                        matrix_energy[i, j] = np.nan
                        matrix_beta[i, j] = np.nan
                        
            # =================================================================
            # FONCTION INTERNE DE TRACÉ (Pour éviter de dupliquer le code)
            # =================================================================
            def plot_matrix(matrix_data, metric_name, cmap_name, is_diverging=False):
                clean_metric_name = metric_name.replace('_', ' ')
                member_mean = np.nanmean(matrix_data, axis=0)
                mean_overall = np.nanmean(member_mean)
                
                # --- SAUVEGARDE CSV ---
                df_csv = pd.DataFrame(matrix_data, index=group_names, columns=member_ids)
                df_csv.loc['Mean_Score'] = member_mean
                csv_path = os.path.join(base_home, f"Member_Heatmap_{metric_name}_{domain_name}_{key}.csv")
                df_csv.to_csv(csv_path)
                
                fig = plt.figure(figsize=(15, 7.5))
                gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 2], width_ratios=[50, 1], wspace=0.02, hspace=0.4)
                
                ax_bar = fig.add_subplot(gs[0, 0])
                ax_hm = fig.add_subplot(gs[1, 0], sharex=ax_bar) 
                cbar_ax = fig.add_subplot(gs[:, 1]) 
                
                M = len(member_ids)
                x_positions = np.arange(M)
                
                # --- SYNCHRONISATION EXACTE DES ÉCHELLES ---
                if is_diverging:
                    # Pour la régression (β), on centre le blanc sur 0, avec +/- 2.0 en limites
                    vmin_shared, vmax_shared = -2.0, 2.0
                    norm_shared = mcolors.TwoSlopeNorm(vmin=vmin_shared, vcenter=0.0, vmax=vmax_shared)
                else:
                    # Pour l'énergie, on prend le min/max de la matrice complète
                    vmin_shared = np.nanmin(matrix_data)
                    vmax_shared = np.nanmax(matrix_data)
                    norm_shared = mcolors.Normalize(vmin=vmin_shared, vmax=vmax_shared)
                
                cmap_obj = plt.get_cmap(cmap_name)
                cmap_obj.set_bad(color='grey', alpha=0.5)
                
                # --- BARPLOT ---
                bar_colors = [cmap_obj(norm_shared(val)) if not np.isnan(val) else (0,0,0,0) for val in member_mean]
                ax_bar.bar(x_positions, member_mean, color=bar_colors, edgecolor='black', linewidth=0.3, width=0.9)
                ax_bar.set_xlim(-0.5, M - 0.5)
                
                ax_bar.set_xticks(x_positions)
                ax_bar.set_xticklabels(member_ids, rotation=90, fontsize=6)
                
                ax_bar.set_ylabel(f"Mean {clean_metric_name}", fontsize=10, labelpad=10)
                ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
                
                # --- Correction du nom de la métrique pour l'affichage ---
                clean_metric_name = metric_name.replace('_', ' ')
                
                title_text = f"Member-wise {clean_metric_name} ({', '.join(months_studied)}) {domain_name}) — {label_lag}\nMean Overall Score: {mean_overall:.2f}"
                ax_bar.set_title(title_text, fontsize=14, fontweight='bold', pad=10)

                # --- HEATMAP ---
                im_hm = ax_hm.imshow(matrix_data, cmap=cmap_obj, aspect='auto', origin='lower', norm=norm_shared)
                
                ax_hm.set_xticks(x_positions)
                ax_hm.set_xticklabels(member_ids, rotation=90, fontsize=6)
                ax_hm.set_yticks(np.arange(n_quantiles))
                ax_hm.set_yticklabels(group_names, fontsize=11, fontweight='bold')
                
                ax_hm.grid(axis='x', linestyle='--', alpha=0.3, color='white')
                ax_hm.set_xlabel("Ensemble Member ID", fontsize=12, fontweight='bold')
                ax_hm.set_ylabel("PC1 Quantiles", fontsize=12, fontweight='bold')
                
                # --- COLORBAR ---
                cbar = fig.colorbar(im_hm, cax=cbar_ax)
                cbar_label = 'Quantile SNR Energy' if not is_diverging else r'Pattern Amplitude ($\beta$)'
                cbar.set_label(cbar_label, fontsize=10)
                
                out_path = os.path.join(base_home, f"Member_Heatmap_{metric_name}_{domain_name}_{key}.png")
                fig.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"  -> {metric_name} ({domain_name}) sauvegardé.")

            # Tracé des deux métriques
            plot_matrix(matrix_energy, "SNR_Energy", "magma", is_diverging=False)
            plot_matrix(matrix_beta, "Pattern_Amplitude", "RdBu_r", is_diverging=True)

    sst_keys = [k for k in all_keys if 'sst' in k]
    slp_keys = [k for k in all_keys if 'slp' in k]

    # Re-génération des poids pour garantir le broadcast (1, lat, 1) vers (lat, lon)
    wgts_slp = None
    if lat_weight:
        sample_slp_map = composites_mean[0]['slp_0']
        lats_slp = np.linspace(20, 70, sample_slp_map.shape[0])
        coslat_slp = np.cos(np.deg2rad(lats_slp)).clip(0., 1.)
        wgts_slp = coslat_slp.reshape(1, len(lats_slp), 1)

    wgts_sst = None
    if lat_weight and len(sst_keys) > 0:
        sample_sst_map = composites_mean[0][sst_keys[0]]
        lats_sst = np.linspace(-15, 70, sample_sst_map.shape[0])
        coslat_sst = np.cos(np.deg2rad(lats_sst)).clip(0., 1.)
        wgts_sst = coslat_sst.reshape(1, len(lats_sst), 1)

    generate_member_heatmap_per_lag(sst_keys, "SST", wgts_sst)
    generate_member_heatmap_per_lag(slp_keys, "SLP", wgts_slp)


    # =========================================================================
    # HEATMAPS 2D SÉPARÉES (SST vs SLP) DU SNR SPATIAL + SCORE GLOBAL
    # =========================================================================
    print("\nGénération des heatmaps 2D récapitulatives des SNR (SST et SLP séparées)...")
    
    sst_keys = [k for k in all_keys if 'sst' in k]
    slp_keys = [k for k in all_keys if 'slp' in k]

    # Formatage propre des mois pour le titre (ex: "Nov-Dec-Jan-Feb" ou "Jan")
    months_str = "-".join([calendar.month_abbr[m] for m in winter_months])

    def generate_domain_heatmap(keys_subset, domain_name, wgts_domain):
        if not keys_subset:
            return 0.0
            
        summary_matrix = np.zeros((n_quantiles, len(keys_subset)))
        x_labels = []
        unit_label_time = "months" if monthly_reduction else "days"
        
        for col_idx, key in enumerate(keys_subset):
            lag_val = int(key.split('_')[-1])
            
            # Formalisme propre académique (Lag X [units])
            if lag_val < 0:    label = f"Lag +{abs(lag_val)} [{unit_label_time}]"
            elif lag_val == 0: label = f"Lag 0 [{unit_label_time}]"
            else:              label = f"Lag -{lag_val} [{unit_label_time}]"
            x_labels.append(label)
            
            for i, k in enumerate(sorted_groups):
                snr_map = composites_mean[k][key] / np.maximum(global_std[key], 1e-5)
                
                # Norme L2 Extensive (Énergie Totale du Signal)
                if lat_weight and wgts_domain is not None:
                    total_snr = np.sqrt(np.sum((snr_map**2) * wgts_domain[0, :, :]))
                else:
                    total_snr = np.sqrt(np.sum(snr_map**2))
                summary_matrix[i, col_idx] = total_snr

        # --- CALCUL DU SCORE SUPER-GLOBAL (Moyenne sur tous les quantiles et lags) ---
        super_global_score = np.mean(summary_matrix)
        peak_score = np.max(summary_matrix)

        # 1. Définition des bornes fixes selon le domaine physique
        vmin_hm = 0.0
        if domain_name == "SST":
            vmax_hm = 27.0  # Échelle fixée sur le pic de février
        elif domain_name == "SLP":
            vmax_hm = None  
        else:
            vmax_hm = None

        fig_hm, ax_hm = plt.subplots(figsize=(max(6, len(keys_subset) * 0.85), 0.8 * n_quantiles + 2.2))
        
        # ---> CHANGEMENT ICI : cmap='magma' <---
        im_hm = ax_hm.imshow(summary_matrix, cmap='magma', aspect='auto', origin='lower', vmin=vmin_hm, vmax=vmax_hm)
        
        ax_hm.set_xticks(np.arange(len(keys_subset)))
        ax_hm.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
        ax_hm.set_yticks(np.arange(n_quantiles))
        ax_hm.set_yticklabels(group_names, fontsize=11, fontweight='bold')
        
        title_text = f"Global SNR ({domain_name}) — {months_str}\nMean Overall Score: {super_global_score:.2f}"
        ax_hm.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
        ax_hm.set_xlabel("Time Lags", fontsize=12, fontweight='bold', labelpad=10)
        ax_hm.set_ylabel("PC1 Quantiles", fontsize=12, fontweight='bold')

        # 3. Référence de couleur INVERSÉE pour la palette Magma !
        ref_max_color = vmax_hm if vmax_hm is not None else (np.max(summary_matrix) if summary_matrix.size > 0 else 1.0)
        for i in range(n_quantiles):
            for j in range(len(keys_subset)):
                val = summary_matrix[i, j]
                # Avec magma, si c'est très élevé c'est clair (jaune/blanc) -> texte en NOIR. 
                # Si c'est bas c'est sombre (violet/noir) -> texte en BLANC.
                text_color = "black" if val > ref_max_color * 0.75 else "white"
                ax_hm.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')
                
        cbar_hm = fig_hm.colorbar(im_hm, ax=ax_hm, fraction=0.03, pad=0.04)
        cbar_hm.set_label('Total Signal Energy (L2 Norm)', fontsize=11)
        
        fig_hm.tight_layout()
        fig_name_hm = f"{base_home}/Summary_Heatmap_SNR_{domain_name}.png"
        fig_hm.savefig(fig_name_hm, dpi=200, bbox_inches='tight')
        plt.close(fig_hm)
        
        print(f"✅ Heatmap globale {domain_name} sauvegardée (Magma) | Score SNR Global ({months_str}) : {super_global_score:.4f}")
        return super_global_score

    # Re-génération des poids pour garantir le broadcast (1, lat, 1) vers (lat, lon)
    wgts_slp = None
    if lat_weight:
        sample_slp_map = composites_mean[0]['slp_0']
        lats_slp = np.linspace(20, 70, sample_slp_map.shape[0])
        coslat_slp = np.cos(np.deg2rad(lats_slp)).clip(0., 1.)
        wgts_slp = coslat_slp.reshape(1, len(lats_slp), 1)

    wgts_sst = None
    if lat_weight and len(sst_keys) > 0:
        sample_sst_map = composites_mean[0][sst_keys[0]]
        lats_sst = np.linspace(-15, 70, sample_sst_map.shape[0])
        coslat_sst = np.cos(np.deg2rad(lats_sst)).clip(0., 1.)
        wgts_sst = coslat_sst.reshape(1, len(lats_sst), 1)

    # Exécution
    score_sst = generate_domain_heatmap(sst_keys, "SST", wgts_sst)
    score_slp = generate_domain_heatmap(slp_keys, "SLP", wgts_slp)


    # =========================================================
    # ÉVALUATION PAR ANALOGUES (Test sur les membres non utilisés)
    # =========================================================
    print("\n=== ÉVALUATION : Prédiction par Analogues (SST -> SLP PC1) ===")
    
    test_members = [m for m in all_members if m not in members_used]
    
    if len(test_members) == 0:
        print("⚠️ Aucun membre de test disponible (number_of_members = 89). Évaluation ignorée.")
    else:
        target_lags = [-7,-5,-3,-1,0,1,3,5,7,10,14] if not monthly_reduction else [1,2,3]
        for target_lag in target_lags:
            target_key = f'sst_lag_{target_lag}'
            print(f"👉 Lag cible utilisé pour la prédiction SST : {target_lag} {'mois' if monthly_reduction else 'jours'}")
            
            ref_sst_maps = np.array([composites_mean[k][target_key] for k in sorted_groups])
            
            dates_list = []
            members_list = []
            trues_list = []
            preds_list = []
            
            start_time = time.time()
            for mem in test_members:
                file_slp, file_sst = get_file_paths(path_slp, path_sst, mem, duree_lissage, monthly_reduction)
                if not os.path.exists(file_slp) or not os.path.exists(file_sst):
                    continue
                    
                with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
                    ds_winter = ds_slp.sel(
                        time=slice(None, "2014-12-31"),
                        drop=False, 
                    )
                    ds_winter = ds_slp.sel(time=ds_slp['time'].dt.month.isin(winter_months))
                    ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15,70))
                    
                    if lat_weight:
                        lats_sst_eval = ds_sst['lat'].values
                        coslat_sst_eval = np.cos(np.deg2rad(lats_sst_eval)).clip(0., 1.)
                        wgts_sst_eval = coslat_sst_eval.reshape(1, len(lats_sst_eval), 1)
                    else:
                        wgts_sst_eval = 1.0

                    dates = ds_winter.time.values
                    slp_array = ds_winter['PSL'].values / slp_std
                    slp_array = np.nan_to_num(slp_array, nan=0.0)
                    if lat_weight:
                        lats = ds_winter['lat'].values
                        coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
                        wgts = np.sqrt(coslat).reshape(1, len(lats), 1)
                        slp_array = slp_array * wgts
                    
                    true_latents = projector.transform(slp_array.reshape(len(dates), -1))[:, 0]
                    
                    for i, d in enumerate(dates):
                        if not monthly_reduction:
                            t_sst = d - timedelta(days=target_lag)
                        else:
                            y_shift = (d.month - target_lag - 1) // 12
                            new_month = (d.month - target_lag - 1) % 12 + 1
                            t_sst = d.replace(year=d.year + y_shift, month=new_month)
                            
                        if t_sst in ds_sst.time.values:
                            sst_map = ds_sst['SST'].sel(time=t_sst).values
                            sst_map = np.nan_to_num(sst_map, nan=0.0)
                            
                            diff_sq = (ref_sst_maps - sst_map[np.newaxis, :, :])**2
                            mse_errors = np.mean(diff_sq * wgts_sst_eval, axis=(1, 2))
                            
                            best_k = np.argmin(mse_errors)
                            predicted_pc1 = pc1_means_per_quantile[best_k]
                            
                            dates_list.append(d)
                            members_list.append(mem)
                            trues_list.append(true_latents[i])
                            preds_list.append(predicted_pc1)

                print(f"✅ Prédictions terminées en {time.time() - start_time:.2f}s ({len(trues_list)} échantillons évalués)")
                
                df = pd.DataFrame({
                    'time': pd.to_datetime([str(d) for d in dates_list]),
                    'member': members_list,
                    'true_pc1': trues_list,
                    'pred_pc1': preds_list
                })
                
                freq_label = "Monthly" if monthly_reduction else "Daily"
                print(f"\nTime Series Frequency Set To: {freq_label}")
                
                unique_members = df['member'].unique()
                n_bootstraps = 1000 
                
                for member in unique_members:
                    print(f"\n{'='*40}")
                    print(f"Evaluating Validation Member (Analogues): {member}")
                    print(f"{'='*40}")

                    df_member = df[df['member'] == member].copy()
                    ds_member = df_member.set_index('time').to_xarray()

                    key = str(member) + '_test' + f"_{target_lag}lag"
                    member_outdir = os.path.join(base_home, "evaluation_plots_analogues_model", key)
                    os.makedirs(member_outdir, exist_ok=True)

                    pc_idx = 1
                    print(f"--- PC {pc_idx} ---")
                    
                    if monthly_reduction:
                        pred_series = ds_member['pred_pc1'].resample(time='1M').mean().dropna(dim="time")
                        true_series = ds_member['true_pc1'].resample(time='1M').mean().dropna(dim="time")
                    else:
                        pred_series = ds_member['pred_pc1'].dropna(dim="time")
                        true_series = ds_member['true_pc1'].dropna(dim="time")

                    pcs_true_dict, pcs_pred_dict, stats_dict = {}, {}, {}

                    for m in winter_months:
                        true_m = true_series.where(true_series.time.dt.month == m, drop=True)
                        pred_m = pred_series.where(pred_series.time.dt.month == m, drop=True)
                        
                        if len(true_m) < 2:
                            continue
                            
                        s = stats(true_m, pred_m, n_bootstraps)
                        
                        pcs_true_dict[m] = true_m
                        pcs_pred_dict[m] = pred_m
                        stats_dict[m] = s
                        
                        month_name = calendar.month_abbr[m]

                        orig_corr, orig_r2, orig_ss_l1, corr_tp_boot, corr_tt_boot, r2_tp_boot, ss_l1_tp_boot = compute_two_bootstraps(
                            true_m.values, 
                            pred_m.values, 
                            n_bootstraps
                        )
                        
                        if not np.isnan(orig_corr):
                            plot_two_bootstrap_histograms(
                                corr_tp_boot, 
                                corr_tt_boot, 
                                r2_tp_boot,
                                ss_l1_tp_boot,
                                orig_corr, 
                                orig_r2,
                                orig_ss_l1,
                                month_name, 
                                pc_idx, 
                                member, 
                                member_outdir, 
                                freq_label
                            )
                    
                    if stats_dict: 
                        valid_months = [m for m in winter_months if m in stats_dict]
                        plot_time_series(
                            pcs_true_dict=pcs_true_dict, 
                            pcs_pred_dict=pcs_pred_dict, 
                            stats_dict=stats_dict, 
                            months_num=valid_months, 
                            member=member, 
                            outdir=member_outdir,
                            pc_idx=pc_idx,
                            freq_label=freq_label,
                            quantiles_pred_dict=None
                        )

    # =========================================================================
    # EXPORT DES CARTES INDIVIDUELLES PAR MEMBRE (Pour création de GIF)
    # =========================================================================
    def generate_individual_member_maps_for_gif(target_key):
        print(f"\nGénération des cartes individuelles par membre pour le lag : {target_key}")
        
        dir_gif = os.path.join(base_home, f"GIF_Members_Maps_{target_key}")
        os.makedirs(dir_gif, exist_ok=True)
        
        is_slp = 'slp' in target_key
        extent = [-100, 40, 20, 70] if is_slp else [-180, 180, -15, 70]
        unit_label = "Pa" if is_slp else "K"
        
        # --- Verrouillage de l'échelle des couleurs sur la globale ---
        # Cela permet d'avoir une échelle fixe pour le GIF (pas de scintillement)
        if is_slp:
            vmax_val = 2 * magnitude_slp
            my_norm = mcolors.Normalize(vmin=-vmax_val, vmax=vmax_val)
            cbar_ticks = cbar_ticks_mean_slp
        else:
            vmax_val = 2 * magnitude_sst
            linthresh_val = 0.4 * magnitude_sst
            my_norm = mcolors.SymLogNorm(linthresh=linthresh_val, vmin=-vmax_val, vmax=vmax_val, base=10)
            cbar_ticks = cbar_ticks_mean_sst

        for mem in member_ids:
            # Création d'une figure 1x4 (une case par quantile)
            fig, axes = plt.subplots(1, n_quantiles, figsize=(5.5 * n_quantiles, 4), subplot_kw={'projection': ccrs.PlateCarree()})
            if n_quantiles == 1:
                axes = [axes] # Sécurité si jamais n_quantiles=1
                
            has_data = False
            
            for i, k in enumerate(sorted_groups):
                ax = axes[i]
                N_mem = mem_valid_counts[mem][k][target_key]
                
                if N_mem > 0:
                    has_data = True
                    # On calcule la moyenne individuelle pour CE membre et CE quantile
                    mem_map = mem_sums[mem][k][target_key] / N_mem
                    im = ax.imshow(mem_map, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, norm=my_norm)
                else:
                    # S'il n'y a eu aucun événement pour ce quantile sur ce membre
                    mem_map = np.zeros_like(composites_mean[k][target_key])
                    im = ax.imshow(mem_map, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent, norm=my_norm)
                    ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center', va='center', color='black', fontweight='bold', fontsize=12)

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(color='black', linewidth=0.8, alpha=0.7)
                ax.set_title(f"{group_names[i]} (N={N_mem})", fontweight='bold')
                
            if has_data:
                # Ajout d'une Colorbar unique à droite
                fig.subplots_adjust(right=0.92)
                cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax, ticks=cbar_ticks, format="%g")
                cbar.set_label(f'Anomaly ({unit_label})')
                
                # Titre propre de la figure (Nettoyage de l'underscore)
                clean_target_name = target_key.replace('_', ' ')
                months_str = "-".join([calendar.month_abbr[m] for m in winter_months])
                fig.suptitle(f"Member {mem} Composite — {clean_target_name} ({months_str})", fontsize=16, fontweight='bold', y=1.05)
                
                out_path = os.path.join(dir_gif, f"Composite_{target_key}_{mem}.png")
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
            
            plt.close(fig)
            
        print(f"✅ Cartes individuelles sauvegardées dans : {dir_gif}")
    # --- Appel de la fonction pour le lag de ton choix (ici sst_lag_-1) ---
    if 'sst_lag_-1' in all_keys:
        print("\n--- Génération des cartes individuelles pour le lag sst_lag_-1 ---")
        generate_individual_member_maps_for_gif('sst_lag_-1')
    if 'sst_lag_1' in all_keys:
        print("\n--- Génération des cartes individuelles pour le lag sst_lag_1 ---")
        generate_individual_member_maps_for_gif('sst_lag_1')


if __name__ == "__main__":
    # ============================================================
    # ARGUMENTS & PATHS
    # ============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
    parser.add_argument('--model_path', type=str, required=True, help="Chemin vers le modèle PCA (.pkl) pré-entraîné")
    parser.add_argument('--n_quantiles', type=int, default=4, help="Nombre de quantiles pour la PC1")
    parser.add_argument('--duree_lissage', type=int, help="Durée du lissage en jours (10 ou 30)")
    parser.add_argument('--normalize', action='store_true', help="PC normalisée (True) ou non (False)")
    parser.add_argument('--number_of_members', type=int, default=89, help="Nombre de membres à utiliser (doit être <= 89)")
    parser.add_argument('--lat_weight', action='store_true', help='Appliquer la pondération spatiale sqrt(cos(lat))')
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données mensuelles (_1mo.nc)')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[11, 12, 1, 2], help='Mois d\'hiver à considérer')
    # --------------------------
    args = parser.parse_args()

    number_of_members = args.number_of_members
    if not args.monthly_reduction:
        sst_lags = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7]
        slp_lags = []
    else:
        sst_lags = [9, 8, 7, 6, 5, 4, 3, 2, 1] 
        slp_lags = []

    duree_lissage = args.duree_lissage
    winter_months = args.winter_months
    lat_weight = args.lat_weight
    monthly_reduction = args.monthly_reduction

    if not monthly_reduction:
        folder_name = f"pc1_per_member_{args.n_quantiles}_quantiles_winter_months{'_'.join(map(str, winter_months))}_{number_of_members}members_normalize{args.normalize}_duree_lissage{duree_lissage}_lags_{'_'.join(str(lag) for lag in sst_lags)}_sst_{'_'.join(map(str, slp_lags))}_slp_lat_weight_{lat_weight}"
    else:
        folder_name = f"pc1_per_member{args.n_quantiles}_quantiles_winter_months{'_'.join(map(str, winter_months))}_{number_of_members}members_normalize{args.normalize}_monthly_reduction_{monthly_reduction}_lags_{'_'.join(str(lag) for lag in sst_lags)}_sst_{'_'.join(map(str, slp_lags))}_slp_lat_weight_{lat_weight}"

    if args.machine == 'hacienda':
        base_home = f"/home/moysan/stage_isir_jz/data_analysis/composites_pc1_quantiles/{folder_name}/"
        data_dir = "/data/moysan/data/"
    elif args.machine == 'jean-zay-work': 
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/composites_pc1_quantiles/{folder_name}/"
        data_dir = "/lustre/fswork/projects/rech/uxg/uca57ub/data/"
    elif args.machine == 'jean-zay-scratch':
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/composites_pc1_quantiles/{folder_name}/"
        data_dir = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/"

    os.makedirs(base_home, exist_ok=True)
    path_sst = os.path.join(data_dir, "SST/")
    path_slp = os.path.join(data_dir, "SLP/")

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    members_used = all_members[:number_of_members]
    print(f"len(members_used) = {len(members_used)}")

    dynamic_slp_std = 596.0  

    if args.model_path:
        match = re.search(r'slp_std([0-9.]+)', args.model_path)
        if match:
            dynamic_slp_std = float(match.group(1))
            print(f"\n✅ slp_std extrait avec succès du chemin PCA : {dynamic_slp_std}")
        else:
            print(f"\n⚠️ 'slp_std' introuvable dans le nom du dossier. Utilisation du fallback : {dynamic_slp_std}")
    else:
        print(f"\n⚠️ Aucun modèle pré-entraîné fourni. Utilisation du slp_std par défaut : {dynamic_slp_std}")
    
    generate_master_reference(
        member_ids = members_used,
        sst_lags = sst_lags,
        slp_lags = slp_lags,
        n_quantiles = args.n_quantiles,
        winter_months = winter_months,
        duree_lissage = duree_lissage,
        monthly_reduction = monthly_reduction,
        lat_weight = lat_weight,
        base_home = base_home,
        slp_std = dynamic_slp_std,
    )

    