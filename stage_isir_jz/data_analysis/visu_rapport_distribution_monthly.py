import os
import re
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import joblib
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns

# ==========================================
# 1. UTILITIES & WEIGHTING
# ==========================================
def get_file_paths(path_slp, path_sst, mem):
    file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}_1mo.nc')
    file_sst = os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid_1mo.nc')
    return file_slp, file_sst

def apply_pca_lat_weights(data_3d, lats):
    w_sqrt = np.sqrt(np.cos(np.deg2rad(lats)))[None, :, None]
    weighted = data_3d * w_sqrt
    return np.nan_to_num(weighted, nan=0.0).reshape(data_3d.shape[0], -1)

def extract_std_from_path(path, var_name):
    pattern = rf"{var_name.lower()}_std([\d.]+)"
    match = re.search(pattern, path)
    if match: return float(match.group(1))
    else: raise ValueError(f"❌ Impossible de trouver le motif '{var_name.lower()}_std...' dans : {path}")

# ==========================================
# 2. EXTRACTION DES STATS, CARTES & EMBEDDINGS
# ==========================================
def process_ensemble_slp_sst(member_ids, months_slp, months_sst, path_slp, path_sst, pca_path_slp, pca_path_sst, output_dir, max_components):
    print(f"\n=======================================================")
    print(f" 🚀 STARTING MONTHLY ANALYSIS (SLP: NDJF | SST: ASONDJF)")
    print(f"=======================================================", flush=True)
    
    pca_slp = joblib.load(pca_path_slp)
    pca_sst = joblib.load(pca_path_sst)
    
    slp_std = extract_std_from_path(pca_path_slp, 'slp')
    sst_std = extract_std_from_path(pca_path_sst, 'sst')
    print(f" ℹ️ Extracted SLP scaling std: {slp_std}")
    print(f" ℹ️ Extracted SST scaling std: {sst_std}", flush=True)
    
    indiv_maps_dir = os.path.join(output_dir, "Individual_Member_Maps")
    os.makedirs(indiv_maps_dir, exist_ok=True)

    stats = {
        'mem': [],
        'slp_pixel_mae_0': [], 'slp_pixel_mae_med': [],
        'sst_pixel_mae_0': [], 'sst_pixel_mae_med': [],
        'slp_pca_mae_0': [], 'slp_pca_mae_med': [],
        'sst_pca_mae_0': [], 'sst_pca_mae_med': []
    }
    latent_embeddings = {'SLP': {}, 'SST': {}}
    
    sum_mean_slp, sum_med_slp, sum_factor_slp = 0.0, 0.0, 0.0
    sum_mean_sst, sum_med_sst, sum_factor_sst = 0.0, 0.0, 0.0
    valid_members_count = 0
    lats_slp, lons_slp, weights_2d_slp = None, None, None
    lats_sst, lons_sst, weights_2d_sst = None, None, None

    for i, mem in enumerate(member_ids):
        file_slp, file_sst = get_file_paths(path_slp, path_sst, mem)
        if not os.path.exists(file_slp) or not os.path.exists(file_sst): continue
        
        with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
            ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15, 70))

            cond_win_slp = ds_slp.time.dt.month.isin(months_slp)
            cond_win_sst = ds_sst.time.dt.month.isin(months_sst)
            cond_not_jan2015_slp = ~((ds_slp.time.dt.year == 2015) & (ds_slp.time.dt.month == 1))
            cond_not_jan2015_sst = ~((ds_sst.time.dt.year == 2015) & (ds_sst.time.dt.month == 1))
            
            slp_val = ds_slp['PSL'].sel(time=(cond_win_slp & cond_not_jan2015_slp))
            sst_val = ds_sst['SST'].sel(time=(cond_win_sst & cond_not_jan2015_sst))
            
            if valid_members_count == 0:
                lats_slp, lons_slp = slp_val.lat.values, slp_val.lon.values
                lats_sst, lons_sst = sst_val.lat.values, sst_val.lon.values
                mask_slp = ~np.isnan(slp_val.isel(time=0).values); mask_sst = ~np.isnan(sst_val.isel(time=0).values)
                weights_2d_slp = np.cos(np.deg2rad(lats_slp))[:, None] * mask_slp
                weights_2d_sst = np.cos(np.deg2rad(lats_sst))[:, None] * mask_sst

            v_slp_3d = np.nan_to_num(slp_val.values, nan=0.0).astype(np.float32, copy=False)
            v_sst_3d = np.nan_to_num(sst_val.values, nan=0.0).astype(np.float32, copy=False)

            mean_slp = np.mean(v_slp_3d, axis=0); med_slp = np.median(v_slp_3d, axis=0)
            mean_sst = np.mean(v_sst_3d, axis=0); med_sst = np.median(v_sst_3d, axis=0)

            # Cartes L1 pour ce membre
            mae_0_slp_map = np.mean(np.abs(v_slp_3d), axis=0)
            mae_m_slp_map = np.mean(np.abs(v_slp_3d - med_slp), axis=0)
            factor_slp_map = mae_0_slp_map / np.where(mae_m_slp_map == 0, 1e-10, mae_m_slp_map)
            
            mae_0_sst_map = np.mean(np.abs(v_sst_3d), axis=0)
            mae_m_sst_map = np.mean(np.abs(v_sst_3d - med_sst), axis=0)
            factor_sst_map = mae_0_sst_map / np.where(mae_m_sst_map == 0, 1e-10, mae_m_sst_map)

            v_slp_pca_input = v_slp_3d / slp_std
            v_sst_pca_input = v_sst_3d / sst_std
            
            z_slp = pca_slp.transform(apply_pca_lat_weights(v_slp_pca_input, lats_slp)).astype(np.float32, copy=False)
            z_sst = pca_sst.transform(apply_pca_lat_weights(v_sst_pca_input, lats_sst)).astype(np.float32, copy=False)
            latent_embeddings['SLP'][mem] = z_slp; latent_embeddings['SST'][mem] = z_sst

            mae_0_slp_pca = np.mean(np.abs(z_slp), axis=0)
            mae_m_slp_pca = np.mean(np.abs(z_slp - np.median(z_slp, axis=0)), axis=0)
            mae_0_sst_pca = np.mean(np.abs(z_sst), axis=0)
            mae_m_sst_pca = np.mean(np.abs(z_sst - np.median(z_sst, axis=0)), axis=0)

            # --- CARTES INDIVIDUELLES ---
            if valid_members_count < 5:
                fig_indiv, axes_indiv = plt.subplots(2, 3, figsize=(22, 10), subplot_kw={'projection': ccrs.PlateCarree()})
                ext_slp = [lons_slp.min(), lons_slp.max(), lats_slp.min(), lats_slp.max()]
                ext_sst = [lons_sst.min(), lons_sst.max(), lats_sst.min(), lats_sst.max()]
                
                # Cadrage spécifique corrigé
                zoom_slp = [-100, 40, 20, 70]; zoom_sst = [-180, 180, -15, 70]
                ext_slp = zoom_slp; ext_sst = zoom_sst  # Pour éviter les bords blancs avec arrondis
                
                maps_to_plot = [
                    (axes_indiv[0,0], mean_slp.copy(), ext_slp, zoom_slp, 'RdBu_r', f"SLP Mean - Mem {mem}", True, slp_std, 'Pa'),
                    (axes_indiv[0,1], med_slp.copy(), ext_slp, zoom_slp, 'RdBu_r', f"SLP Median - Mem {mem}", True, slp_std, 'Pa'),
                    (axes_indiv[0,2], factor_slp_map.copy(), ext_slp, zoom_slp, 'YlOrRd', f"SLP L1 Factor - Mem {mem}", False, None, 'Factor'),
                    (axes_indiv[1,0], mean_sst.copy(), ext_sst, zoom_sst, 'RdBu_r', f"SST Mean - Mem {mem}", True, sst_std, 'K'),
                    (axes_indiv[1,1], med_sst.copy(), ext_sst, zoom_sst, 'RdBu_r', f"SST Median - Mem {mem}", True, sst_std, 'K'),
                    (axes_indiv[1,2], factor_sst_map.copy(), ext_sst, zoom_sst, 'YlOrRd', f"SST L1 Factor - Mem {mem}", False, None, 'Factor')
                ]
                
                for ax, data, ext, zoom, cmap, title, is_diverging, ref_std, unit in maps_to_plot:
                    data[data == 0.0] = np.nan 
                    if is_diverging:
                        vmax = ref_std * 0.5
                        vmin = -vmax
                    else:
                        vmin, vmax = 1.0, 1.05
                        
                    im = ax.imshow(data, transform=ccrs.PlateCarree(), cmap=cmap, origin='lower', extent=ext, vmin=vmin, vmax=vmax)
                    ax.set_extent(zoom, crs=ccrs.PlateCarree()); ax.coastlines(color='black', alpha=0.5)
                    ax.set_title(title, fontweight='bold', fontsize=11)
                    cbar = fig_indiv.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
                    cbar.set_label(unit)

                fig_indiv.tight_layout()
                fig_indiv.savefig(os.path.join(indiv_maps_dir, f"Member_{mem}_Spatial_Stats.jpg"), dpi=150, pil_kwargs={'quality': 85}, bbox_inches='tight')
                plt.close(fig_indiv)

                factor_slp_pca = mae_0_slp_pca[:max_components] / np.where(mae_m_slp_pca[:max_components] == 0, 1e-10, mae_m_slp_pca[:max_components])
                factor_sst_pca = mae_0_sst_pca[:max_components] / np.where(mae_m_sst_pca[:max_components] == 0, 1e-10, mae_m_sst_pca[:max_components])
                
                fig_pca, axes_pca = plt.subplots(1, 2, figsize=(14, 5))
                x_labels = [f"PC{k+1}" for k in range(max_components)]
                
                axes_pca[0].bar(x_labels, factor_slp_pca, color='#d95f02', edgecolor='black')
                axes_pca[0].axhline(1.0, color='black', linestyle='--')
                axes_pca[0].set_ylim(0.95, 1.10)
                axes_pca[0].set_title(f"SLP L1 Scaling Factor per PC - Mem {mem}", fontweight='bold')
                axes_pca[0].set_ylabel("Factor (MAE_0 / MAE_Median)")
                
                axes_pca[1].bar(x_labels, factor_sst_pca, color='#7570b3', edgecolor='black')
                axes_pca[1].axhline(1.0, color='black', linestyle='--')
                axes_pca[1].set_ylim(0.95, 1.10)
                axes_pca[1].set_title(f"SST L1 Scaling Factor per PC - Mem {mem}", fontweight='bold')
                axes_pca[1].set_ylabel("Factor (MAE_0 / MAE_Median)")
                
                fig_pca.tight_layout()
                fig_pca.savefig(os.path.join(indiv_maps_dir, f"Member_{mem}_PCA_L1_Factor.jpg"), dpi=150, pil_kwargs={'quality': 85})
                plt.close(fig_pca)

            sum_mean_slp += mean_slp; sum_med_slp += med_slp; sum_factor_slp += factor_slp_map
            sum_mean_sst += mean_sst; sum_med_sst += med_sst; sum_factor_sst += factor_sst_map

            sum_w_slp = np.sum(weights_2d_slp); T_slp = v_slp_3d.shape[0]
            sum_w_sst = np.sum(weights_2d_sst); T_sst = v_sst_3d.shape[0]

            mae_0_slp_pix = np.sum(np.abs(v_slp_3d) * weights_2d_slp) / (T_slp * sum_w_slp)
            mae_m_slp_pix = np.sum(np.abs(v_slp_3d - med_slp) * weights_2d_slp) / (T_slp * sum_w_slp)
            mae_0_sst_pix = np.sum(np.abs(v_sst_3d) * weights_2d_sst) / (T_sst * sum_w_sst)
            mae_m_sst_pix = np.sum(np.abs(v_sst_3d - med_sst) * weights_2d_sst) / (T_sst * sum_w_sst)

            stats['mem'].append(mem)
            stats['slp_pixel_mae_0'].append(mae_0_slp_pix); stats['slp_pixel_mae_med'].append(mae_m_slp_pix)
            stats['sst_pixel_mae_0'].append(mae_0_sst_pix); stats['sst_pixel_mae_med'].append(mae_m_sst_pix)
            stats['slp_pca_mae_0'].append(mae_0_slp_pca); stats['slp_pca_mae_med'].append(mae_m_slp_pca)
            stats['sst_pca_mae_0'].append(mae_0_sst_pca); stats['sst_pca_mae_med'].append(mae_m_sst_pca)

            valid_members_count += 1
        print(f" -> Member {mem} processed.", end='\r', flush=True)
    
    print(f"\n -> Extraction complete for {valid_members_count} members.", flush=True)

    # --- C. TRACÉ DES CARTES DE CONSENSUS (MEAN, MEDIAN & FACTOR) ---
    print(" -> Saving Ensemble Consensus Maps...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(22, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    ext_slp = [-100, 40, 20, 70]
    ext_sst = [-180, 180, -15, 70]
    zoom_slp = [-100, 40, 20, 70]; zoom_sst = [-180, 180, -15, 70]
    
    maps_to_plot = [
        (axes[0,0], sum_mean_slp / valid_members_count, ext_slp, zoom_slp, 'RdBu_r', "SLP Ensemble Mean of Means", True, slp_std, 'Pa'),
        (axes[0,1], sum_med_slp / valid_members_count,  ext_slp, zoom_slp, 'RdBu_r', "SLP Ensemble Mean of Medians", True, slp_std, 'Pa'),
        (axes[0,2], sum_factor_slp / valid_members_count, ext_slp, zoom_slp, 'YlOrRd', "SLP Ensemble Mean L1 Factor", False, None, 'Factor'),
        (axes[1,0], sum_mean_sst / valid_members_count, ext_sst, zoom_sst, 'RdBu_r', "SST Ensemble Mean of Means", True, sst_std, 'K'),
        (axes[1,1], sum_med_sst / valid_members_count,  ext_sst, zoom_sst, 'RdBu_r', "SST Ensemble Mean of Medians", True, sst_std, 'K'),
        (axes[1,2], sum_factor_sst / valid_members_count, ext_sst, zoom_sst, 'YlOrRd', "SST Ensemble Mean L1 Factor", False, None, 'Factor')
    ]

    for ax, data, ext, zoom, cmap, title, is_diverging, ref_std, unit in maps_to_plot:
        data[data == 0.0] = np.nan 
        if is_diverging:
            vmax = ref_std * 0.5 
            vmin = -vmax
        else:
            vmin, vmax = 1.0, 1.05
            
        im = ax.imshow(data, transform=ccrs.PlateCarree(), cmap=cmap, origin='lower', extent=ext, vmin=vmin, vmax=vmax)
        ax.set_extent(zoom, crs=ccrs.PlateCarree()); ax.coastlines(color='black', alpha=0.5)
        ax.set_title(title, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
        cbar.set_label(unit)

    fig.tight_layout()
    map_path = os.path.join(output_dir, "Ensemble_Consensus_Mean_Median_Factor_Maps.jpg")
    fig.savefig(map_path, dpi=200, pil_kwargs={'quality': 90, 'subsampling': 0}, bbox_inches='tight')
    plt.close(fig)

    return stats, latent_embeddings

# ==========================================
# 3. ANALYSE DU FACTEUR D'ERREUR L1
# ==========================================
def evaluate_l1_skill_score_scaling(stats, max_components, output_dir):
    print(f"\n--- Quantifying L1 Baseline Error Scaling Factor ---", flush=True)
    
    df_pixels = pd.DataFrame({
        'Member': stats['mem'],
        'Factor_SLP_Pixel': np.array(stats['slp_pixel_mae_0']) / np.array(stats['slp_pixel_mae_med']),
        'Factor_SST_Pixel': np.array(stats['sst_pixel_mae_0']) / np.array(stats['sst_pixel_mae_med'])
    })
    
    df_pca = pd.DataFrame({'Member': stats['mem']})
    arr_slp_0 = np.array(stats['slp_pca_mae_0'])[:, :max_components]
    arr_slp_m = np.array(stats['slp_pca_mae_med'])[:, :max_components]
    arr_sst_0 = np.array(stats['sst_pca_mae_0'])[:, :max_components]
    arr_sst_m = np.array(stats['sst_pca_mae_med'])[:, :max_components]
    
    for c in range(max_components):
        df_pca[f'Factor_SLP_PC{c+1}'] = arr_slp_0[:, c] / np.where(arr_slp_m[:, c] == 0, 1e-10, arr_slp_m[:, c])
        df_pca[f'Factor_SST_PC{c+1}'] = arr_sst_0[:, c] / np.where(arr_sst_m[:, c] == 0, 1e-10, arr_sst_m[:, c])
    
    df_pixels.to_csv(os.path.join(output_dir, "L1_Scaling_Pixels.csv"), index=False)
    df_pca.to_csv(os.path.join(output_dir, f"L1_Scaling_PCA_Top{max_components}.csv"), index=False)
    
    print(f" 📊 SAVED TWO CSVs:")
    print(f"    - Global Pixels Factor (SLP mean): {df_pixels['Factor_SLP_Pixel'].mean():.4f}x")
    print(f"    - PCA Factors (PC1 SLP mean)     : {df_pca['Factor_SLP_PC1'].mean():.4f}x", flush=True)

# ==========================================
# 4. MATRICES WASSERSTEIN (CSVs + HEATMAPS INDIVIDUELLES)
# ==========================================
def compute_pairwise_wasserstein_matrices(latent_embeddings, max_components, var_name, output_dir):
    print(f"\n--- Computing Pairwise Wasserstein-1 Matrices for Top {max_components} PCs ({var_name}) ---", flush=True)
    
    members = sorted(list(latent_embeddings.keys()))
    M = len(members)
    Z_stacked = np.array([latent_embeddings[m] for m in members])
    Z_sorted = np.sort(Z_stacked[:, :, :max_components], axis=1)
    
    out_subfolder_csv = os.path.join(output_dir, f"Wasserstein_CSVs_{var_name}")
    out_subfolder_img = os.path.join(output_dir, f"Wasserstein_Heatmaps_{var_name}")
    os.makedirs(out_subfolder_csv, exist_ok=True); os.makedirs(out_subfolder_img, exist_ok=True)

    for c in range(max_components):
        mat_c = np.zeros((M, M), dtype=np.float64)
        for i in range(M):
            for j in range(i + 1, M):
                w_ij = np.mean(np.abs(Z_sorted[i, :, c] - Z_sorted[j, :, c]))
                mat_c[i, j] = w_ij; mat_c[j, i] = w_ij
                
        df_w = pd.DataFrame(mat_c, index=members, columns=members)
        csv_path = os.path.join(out_subfolder_csv, f"Wasserstein_{var_name}_PC_{c+1:02d}.csv")
        df_w.to_csv(csv_path)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(mat_c, ax=ax, cmap='magma_r', 
                    xticklabels=members, 
                    yticklabels=members, 
                    cbar_kws={'label': 'Wasserstein-1 Distance'})
        
        # Adaptation de la taille de police pour faire rentrer les 89 labels
        ax.tick_params(axis='x', rotation=90, labelsize=3, length=0)
        ax.tick_params(axis='y', labelsize=3, length=0)

        ax.set_title(f"Pairwise Wasserstein-1 Distance — PC {c+1} ({var_name})", fontweight='bold', fontsize=12)
        ax.set_xlabel("Member ID"); ax.set_ylabel("Member ID")
            
        fig.tight_layout()
        plot_path = os.path.join(out_subfolder_img, f"Wasserstein_Heatmap_{var_name}_PC_{c+1:02d}.jpg")
        fig.savefig(plot_path, dpi=200, pil_kwargs={'quality': 90, 'subsampling': 0})
        plt.close(fig)
        print(f" -> Computed and plotted {var_name} matrix for PC {c+1}/{max_components}", end='\r', flush=True)
        
    print(f"\n ✅ {max_components} CSVs and Heatmaps generated for {var_name}!", flush=True)

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
    parser.add_argument('--number_of_members', type=int, default=89)
    parser.add_argument('--max_components', type=int, default=10, help="Nombre de PC pour Wasserstein")
    parser.add_argument('--pca_slp', type=str, required=True, help="Path to SLP best_pca_model.joblib")
    parser.add_argument('--pca_sst', type=str, required=True, help="Path to SST best_pca_model.joblib")
    args = parser.parse_args()

    folder_name = "report_monthly_distributions_wasserstein"
    if args.machine == 'hacienda':
        base_home = f"/home/moysan/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = "/data/moysan/data/"
    else:
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = f"/lustre/{'fswork' if args.machine == 'jean-zay-work' else 'fsn1'}/projects/rech/uxg/uca57ub/data/"

    os.makedirs(base_home, exist_ok=True)
    path_sst = os.path.join(data_dir, "SST/")
    path_slp = os.path.join(data_dir, "SLP/")

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    members_used = all_members[:args.number_of_members]
    print(f"\n✅ Using {len(members_used)} members for analysis")
    
    months_slp = [11, 12, 1, 2]           # NDJF
    months_sst = [8, 9, 10, 11, 12, 1, 2] # ASONDJF

    start_time = time.time()
    
    stats_m, emb_m = process_ensemble_slp_sst(
        members_used, months_slp, months_sst, 
        path_slp, path_sst, 
        args.pca_slp, args.pca_sst, 
        base_home, args.max_components
    )
    
    evaluate_l1_skill_score_scaling(stats_m, args.max_components, base_home)
    compute_pairwise_wasserstein_matrices(emb_m['SLP'], args.max_components, 'SLP', base_home)
    compute_pairwise_wasserstein_matrices(emb_m['SST'], args.max_components, 'SST', base_home)
    
    print(f"\n=== ANALYSIS PIPELINE COMPLETED in {time.time() - start_time:.2f} seconds ===", flush=True)