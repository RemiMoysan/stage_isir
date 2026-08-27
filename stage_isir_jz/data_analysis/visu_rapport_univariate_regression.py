import os
import re
import argparse
import numpy as np
import xarray as xr
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import stats  # AJOUT : Pour calculer la p-value très rapidement

# ==========================================
# 1. UTILITIES
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
    else: raise ValueError(f"❌ Could not find '{var_name.lower()}_std...' in path: {path}")

def get_lagged_ym(y, m, lag):
    """Calcule l'année et le mois cibles après application du lag (en mois)."""
    total_m = (y * 12 + m) - lag - 1
    lag_y = total_m // 12
    lag_m = (total_m % 12) + 1
    return lag_y, lag_m

# ==========================================
# 2. MAIN PROCESSING
# ==========================================
def run_univariate_regression_safe(member_ids, months_slp, lags, path_slp, path_sst, pca_path_slp, output_dir):
    print(f"\n=======================================================")
    print(f" 🚀 MEMORY-SAFE UNIVARIATE REGRESSION: SST vs SLP PC1")
    print(f"=======================================================", flush=True)
    
    pca_slp = joblib.load(pca_path_slp)
    slp_std = extract_std_from_path(pca_path_slp, 'slp')
    print(f" ℹ️ Loaded SLP PCA Model. Extracted scaling std: {slp_std}")

    month_names = {11: 'November', 12: 'December', 1: 'January', 2: 'February'}
    extent_sst = [-180, 180, -15, 70]

    # --- BOUCLE SUR CHAQUE MOIS CIBLE SÉPARÉMENT ---
    for target_m in months_slp:
        print(f"\n---> 📅 Processing Target Month: {month_names[target_m]} <---", flush=True)
        
        accumulators = {
            lag: {
                'sum_p': 0.0, 'sum_p2': 0.0, 
                'sum_s': None, 'sum_s2': None, 'sum_sp': None, 
                'n': 0
            } for lag in lags
        }
        
        land_mask = None
        lats_slp = None
        lon2d, lat2d = None, None  # AJOUT : Grilles nécessaires pour hachurer

        # --- ÉTAPE A : ACCUMULATION EN LIGNE ---
        valid_members = 0
        for mem in member_ids:
            file_slp, file_sst = get_file_paths(path_slp, path_sst, mem)
            if not os.path.exists(file_slp) or not os.path.exists(file_sst): continue
            
            with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
                ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15, 70))
                
                if lats_slp is None:
                    lats_slp = ds_slp['lat'].values
                    land_mask = np.isnan(ds_sst['SST'].isel(time=0).values)
                    # AJOUT : Récupération des coordonnées pour le contourf (hachures)
                    lons_sst = ds_sst['lon'].values
                    lats_sst = ds_sst['lat'].values
                    lon2d, lat2d = np.meshgrid(lons_sst, lats_sst)

                slp_years, slp_months = ds_slp.time.dt.year.values, ds_slp.time.dt.month.values
                sst_years, sst_months = ds_sst.time.dt.year.values, ds_sst.time.dt.month.values

                sst_ym_to_idx = {(y, m): i for i, (y, m) in enumerate(zip(sst_years, sst_months))}

                for lag in lags:
                    idx_slp, idx_sst = [], []
                    
                    for i_slp, (y, m) in enumerate(zip(slp_years, slp_months)):
                        if m == target_m and not (y == 2015 and m == 1):
                            lag_y, lag_m = get_lagged_ym(y, m, lag)
                            if (lag_y, lag_m) in sst_ym_to_idx:
                                idx_slp.append(i_slp)
                                idx_sst.append(sst_ym_to_idx[(lag_y, lag_m)])
                    
                    if not idx_slp: continue
                    
                    slp_arr = ds_slp['PSL'].isel(time=idx_slp).values
                    sst_arr = ds_sst['SST'].isel(time=idx_sst).values
                    
                    v_slp_3d = np.nan_to_num(slp_arr, nan=0.0).astype(np.float32, copy=False)
                    X_scaled = apply_pca_lat_weights(v_slp_3d / slp_std, lats_slp)
                    Z = pca_slp.transform(X_scaled)
                    P = Z[:, 0].astype(np.float64)
                    
                    S = np.nan_to_num(sst_arr, nan=0.0).astype(np.float64)
                    
                    acc = accumulators[lag]
                    if acc['sum_s'] is None:
                        acc['sum_s'] = np.zeros_like(S[0])
                        acc['sum_s2'] = np.zeros_like(S[0])
                        acc['sum_sp'] = np.zeros_like(S[0])
                        
                    acc['sum_p'] += np.sum(P)
                    acc['sum_p2'] += np.sum(P**2)
                    acc['sum_s'] += np.sum(S, axis=0)
                    acc['sum_s2'] += np.sum(S**2, axis=0)
                    acc['sum_sp'] += np.sum(S * P[:, None, None], axis=0)
                    acc['n'] += len(P)

            valid_members += 1
            print(f"    - Member {mem} accumulated.", end='\r', flush=True)

        print(f"\n    -> Done accumulating for {month_names[target_m]}. Computing metrics...", flush=True)

        # --- ÉTAPE B : CALCUL DES MATRICES (Corrélation, p-value et R2) ---
        maps_to_plot = {}
        
        for lag in lags:
            acc = accumulators[lag]
            n = acc['n']
            if n < 3: continue # Besoin d'au moins 3 points pour un test de Student
            
            mean_p = acc['sum_p'] / n
            var_p = (acc['sum_p2'] / n) - mean_p**2
            std_p = np.sqrt(np.maximum(var_p, 1e-10))
            
            mean_s = acc['sum_s'] / n
            var_s = (acc['sum_s2'] / n) - mean_s**2
            std_s = np.sqrt(np.maximum(var_s, 1e-10))
            
            cov_sp = (acc['sum_sp'] / n) - (mean_s * mean_p)
            
            # 1. Coefficient de corrélation de Pearson
            corr = np.divide(cov_sp, (std_s * std_p), out=np.zeros_like(cov_sp), where=(std_s > 1e-6))
            
            # AJOUT : 2. Calcul vectorisé très rapide de la significativité (p-value)
            df = n - 2
            # Eviter la division par zéro si corrélation parfaite = 1
            t_stat = corr * np.sqrt(df / np.maximum(1 - corr**2, 1e-10))
            # Test bilatéral (survival function de Scipy)
            p_val = stats.t.sf(np.abs(t_stat), df) * 2 
            
            # 3. R^2 (Coefficient de détermination)
            r2 = corr ** 2
            
            # Application du masque océan/terre
            r2[land_mask] = np.nan
            corr[land_mask] = np.nan
            p_val[land_mask] = np.nan
            
            maps_to_plot[lag] = {
                'corr': corr,
                'p_val': p_val,
                'r2': r2
            }

        if not maps_to_plot:
            continue

        vmax_r2 = max([np.nanmax(m['r2']) for m in maps_to_plot.values()])
        n_lags = len(lags)

       # ---------------------------------------------------------
        # FIGURE 1: Pearson Correlation (Avec hachures intelligentes)
        # ---------------------------------------------------------
        # Affiner l'épaisseur des hachures pour ne pas alourdir la carte
        plt.rcParams['hatch.linewidth'] = 0.5 
        
        fig_c, axes_c = plt.subplots(1, n_lags, figsize=(6 * n_lags, 2.5), subplot_kw={'projection': ccrs.PlateCarree()})
        ax_list_c = [axes_c] if n_lags == 1 else axes_c.flatten()
        
        for ax, lag in zip(ax_list_c, lags):
            lag_label = -lag if lag > 0 else 0
            m_corr = maps_to_plot[lag]['corr']
            m_pval = maps_to_plot[lag]['p_val']
            
            im_c = ax.imshow(m_corr, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent_sst, vmin=-0.4, vmax=0.4)
            
            # CRÉATION DU MASQUE : Significatif (p < 0.01) ET pertinent physiquement (|r| > 0.15)
            # On remplace les NaNs par False pour éviter les warnings
            # sig_mask = (m_pval < 0.01) & (np.abs(m_corr) > 0.15)
            sig_mask = m_pval < 0.05 
            sig_mask = np.nan_to_num(sig_mask, nan=False)
            
            # Hachures uniquement là où le masque vaut 1 (True)
            ax.contourf(lon2d, lat2d, sig_mask.astype(int), levels=[0.5, 2], hatches=['///'], colors='none', transform=ccrs.PlateCarree())

            ax.set_extent(extent_sst, crs=ccrs.PlateCarree())
            ax.coastlines(color='black', linewidth=0.8)
            ax.set_title(f"Pearson Correlation — Lag {lag_label} [months]", fontweight='bold', fontsize=12)
            
        # CORRECTION COLORBAR : fraction beaucoup plus petite (0.01 au lieu de 0.03) pour éviter le chevauchement
        cbar_c = fig_c.colorbar(im_c, ax=ax_list_c, fraction=0.012, pad=0.02, shrink=0.7)
        cbar_c.set_label("Pearson $r$")
        
        fig_c.suptitle(f"Target Month: {month_names[target_m]} — SST vs SLP PC1 Correlation (Hatched: p < 0.05)", fontsize=14, fontweight='bold', y=1.05)
        
        # On ajuste les marges pour que le titre ne soit pas écrasé
        fig_c.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path_c = os.path.join(output_dir, f"SST_PC1_Correlation_{month_names[target_m]}.jpg")
        fig_c.savefig(save_path_c, dpi=200, pil_kwargs={'quality': 90}, bbox_inches='tight')
        plt.close(fig_c)
        
        # Réinitialiser le paramètre global pour ne pas affecter d'autres scripts éventuels
        plt.rcParams['hatch.linewidth'] = 1.0

        # ---------------------------------------------------------
        # FIGURE 2: Coefficient of Determination (R^2)
        # ---------------------------------------------------------
        fig_r, axes_r = plt.subplots(1, n_lags, figsize=(6 * n_lags, 2.5), subplot_kw={'projection': ccrs.PlateCarree()})
        ax_list_r = [axes_r] if n_lags == 1 else axes_r.flatten()
        
        for ax, lag in zip(ax_list_r, lags):
            lag_label = -lag if lag > 0 else 0
            m_r2 = maps_to_plot[lag]['r2']
            max_r2_lag = np.nanmax(m_r2)
            
            im_r = ax.imshow(m_r2, transform=ccrs.PlateCarree(), cmap='viridis', origin='lower', extent=extent_sst, vmin=0, vmax=vmax_r2)
            ax.set_extent(extent_sst, crs=ccrs.PlateCarree())
            ax.coastlines(color='black', linewidth=0.8)
            ax.set_title(f"Lag {lag_label} [months] | Max $R^2$: {max_r2_lag:.3f}", fontweight='bold', fontsize=12)
            
        cbar_r = fig_r.colorbar(im_r, ax=ax_list_r, fraction=0.03, pad=0.04, shrink=0.6)
        cbar_r.set_label("$R^2$ Score")
        fig_r.suptitle(f"Target Month: {month_names[target_m]} — Coefficient of Determination ($R^2$)", fontsize=14, fontweight='bold', y=1.02)
        fig_r.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path_r = os.path.join(output_dir, f"SST_PC1_R2_{month_names[target_m]}.jpg")
        fig_r.savefig(save_path_r, dpi=200, pil_kwargs={'quality': 90}, bbox_inches='tight')
        plt.close(fig_r)
        
        print(f"    ✅ Saved Correlation and R2 plots for {month_names[target_m]}.", flush=True)

    print("\n✅ All months processed successfully!", flush=True)

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
    parser.add_argument('--pca_slp', type=str, required=True, help="Path to SLP best_pca_model.joblib")
    parser.add_argument('--lags', type=int, nargs='+', default=[1, 2, 3], help="Lags in months for SST (e.g. 1 2 3)")
    parser.add_argument('--number_of_members', type=int, default=89)
    args = parser.parse_args()

    folder_name = "report_univariate_regression_pvalue" + ("_lags_" + "_".join(map(str, args.lags)))
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
    
    months_slp = [11, 12, 1, 2] # NDJF

    run_univariate_regression_safe(members_used, months_slp, args.lags, path_slp, path_sst, args.pca_slp, base_home)