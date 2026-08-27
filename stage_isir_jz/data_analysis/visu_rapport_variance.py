import os
import argparse
import time
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image

# ==========================================
# 1. PATH UTILITIES & GIF CREATION
# ==========================================
def get_file_paths(path_slp, path_sst, mem, is_monthly=False):
    suffix = "_1mo" if is_monthly else ""
    file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}{suffix}.nc')
    file_sst = os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid{suffix}.nc')
    return file_slp, file_sst

def create_gif_from_folder(image_folder, output_gif_path, duration_ms=300):
    jpg_files = sorted(glob.glob(os.path.join(image_folder, "acf_lag_*.jpg")))
    if not jpg_files: return
    print(f" -> Creating/Updating animation ({len(jpg_files)} frames): {output_gif_path}", flush=True)
    frames = [Image.open(f).convert('RGB') for f in jpg_files]
    frames[0].save(output_gif_path, format='GIF', append_images=frames[1:],
                   save_all=True, duration=duration_ms, loop=0)

# ==========================================
# 2. INTRA-MONTHLY VARIANCE HISTOGRAMS
# ==========================================
def plot_intramonthly_variance_histogram(member_ids, months_slp, months_sst, path_slp, path_sst, output_dir):
    print(f"\n=== [1/3] Computing Intra-Monthly Variances (Histograms) ===", flush=True)
    intra_vars_slp = {m: [] for m in months_slp}
    intra_vars_sst = {m: [] for m in months_sst}
    
    for i, mem in enumerate(member_ids):
        file_slp, file_sst = get_file_paths(path_slp, path_sst, mem, is_monthly=False)
        if not os.path.exists(file_slp): continue
            
        with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
            if i == 0:
                print(f" ℹ️ Date check for Member {mem}:")
                print(f"    - SLP: Start = {str(ds_slp.time.values[0])[:10]} | End = {str(ds_slp.time.values[-1])[:10]}")
                print(f"    - SST: Start = {str(ds_sst.time.values[0])[:10]} | End = {str(ds_sst.time.values[-1])[:10]}", flush=True)
            ds_slp = ds_slp.sel(
                    time=slice(None, "2014-12-31"),
                    drop=False,  # Sécurité
                )
            ds_sst = ds_sst.sel(
                    time=slice(None, "2014-12-31"),
                    drop=False,  # Sécurité
                )


            ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15, 70))
            slp_sel = ds_slp['PSL'].sel(time=ds_slp.time.dt.month.isin(months_slp))
            sst_sel = ds_sst['SST'].sel(time=ds_sst.time.dt.month.isin(months_sst))
            
            weights_slp = np.cos(np.deg2rad(slp_sel.lat))
            weights_sst = np.cos(np.deg2rad(sst_sel.lat))
            
            var_slp_monthly = slp_sel.resample(time='1MS').var(dim='time').weighted(weights_slp).mean(dim=['lat', 'lon'])
            var_sst_monthly = sst_sel.resample(time='1MS').var(dim='time').weighted(weights_sst).mean(dim=['lat', 'lon'])
            
            for m in months_slp:
                vals = var_slp_monthly.sel(time=var_slp_monthly.time.dt.month == m).values
                vals = vals[~np.isnan(vals)]
                intra_vars_slp[m].extend(vals[vals > 1e-6])
            for m in months_sst:
                vals = var_sst_monthly.sel(time=var_sst_monthly.time.dt.month == m).values
                vals = vals[~np.isnan(vals)]
                intra_vars_sst[m].extend(vals[vals > 1e-6])
        print(f" -> Member {mem} processed.", end='\r', flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    month_names_en = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    
    month_colors_en = {
        8: '#edd400', 9: '#f7973a', 10: '#f26122', 
        11: '#c82538', 12: '#80143f', 1: '#2b5c8f', 2: '#3888c0'
    }
    
    labels_slp = [month_names_en.get(m, str(m)) for m in months_slp]
    colors_slp = [month_colors_en.get(m, '#333333') for m in months_slp]
    data_slp = [intra_vars_slp[m] for m in months_slp]
    axes[0].hist(data_slp, bins=30, stacked=True, label=labels_slp, color=colors_slp, edgecolor='black', linewidth=0.5)
    axes[0].set_title(f"SLP Intra-Monthly Spatial Variance ({'-'.join(labels_slp)})", fontweight='bold')
    axes[0].set_xlabel("Mean Spatial Variance (Pa²)"); axes[0].legend(loc='upper right')

    labels_sst = [month_names_en.get(m, str(m)) for m in months_sst]
    colors_sst = [month_colors_en.get(m, '#333333') for m in months_sst]
    data_sst = [intra_vars_sst[m] for m in months_sst]
    axes[1].hist(data_sst, bins=30, stacked=True, label=labels_sst, color=colors_sst, edgecolor='black', linewidth=0.5)
    axes[1].set_title(f"SST Intra-Monthly Spatial Variance ({'-'.join(labels_sst)})", fontweight='bold')
    axes[1].set_xlabel("Mean Spatial Variance (K²)"); axes[1].legend(loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Intramonthly_Variance_Histogram.jpg")
    plt.savefig(save_path, dpi=200, pil_kwargs={'quality': 85})
    plt.close()
    print(f"\n✅ Histogram saved: {save_path}", flush=True)

# ==========================================
# 3. DAILY vs MONTHLY VARIANCE MAPS
# ==========================================
def compute_variance_maps(member_ids, months_slp, months_sst, path_slp, path_sst, output_dir):
    print(f"\n=== [2/3] Computing Daily vs Monthly Spatial Variances ===", flush=True)
    w_slp_2d, w_sst_2d = None, None
    ext_data_slp, ext_data_sst = None, None

    def accumulate_variance(is_monthly):
        nonlocal w_slp_2d, w_sst_2d, ext_data_slp, ext_data_sst
        sum_slp, sum_sq_slp, n_slp = 0.0, 0.0, 0
        sum_sst, sum_sq_sst, n_sst = 0.0, 0.0, 0
        for mem in member_ids:
            file_slp, file_sst = get_file_paths(path_slp, path_sst, mem, is_monthly)
            if not os.path.exists(file_slp): continue
            with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
                ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15, 70))
                slp_w = ds_slp['PSL'].sel(time=ds_slp.time.dt.month.isin(months_slp))
                sst_w = ds_sst['SST'].sel(time=ds_sst.time.dt.month.isin(months_sst))
                if w_slp_2d is None:
                    ext_data_slp = [slp_w.lon.min().item(), slp_w.lon.max().item(), slp_w.lat.min().item(), slp_w.lat.max().item()]
                    ext_data_sst = [sst_w.lon.min().item(), sst_w.lon.max().item(), sst_w.lat.min().item(), sst_w.lat.max().item()]
                    mask_slp = ~np.isnan(slp_w.isel(time=0).values); mask_sst = ~np.isnan(sst_w.isel(time=0).values)
                    w_slp_2d = np.cos(np.deg2rad(slp_w.lat.values))[:, None] * mask_slp
                    w_sst_2d = np.cos(np.deg2rad(sst_w.lat.values))[:, None] * mask_sst

                v_slp = np.nan_to_num(slp_w.values, nan=0.0).astype(np.float32, copy=False)
                sum_slp += v_slp.sum(axis=0, dtype=np.float64); sum_sq_slp += (v_slp**2).sum(axis=0, dtype=np.float64); n_slp += v_slp.shape[0]
                v_sst = np.nan_to_num(sst_w.values, nan=0.0).astype(np.float32, copy=False)
                sum_sst += v_sst.sum(axis=0, dtype=np.float64); sum_sq_sst += (v_sst**2).sum(axis=0, dtype=np.float64); n_sst += v_sst.shape[0]
        var_slp = (sum_sq_slp / max(1, n_slp)) - (sum_slp / max(1, n_slp))**2
        var_sst = (sum_sq_sst / max(1, n_sst)) - (sum_sst / max(1, n_sst))**2
        return np.maximum(0, var_slp), np.maximum(0, var_sst)

    var_slp_daily, var_sst_daily = accumulate_variance(is_monthly=False)
    var_slp_monthly, var_sst_monthly = accumulate_variance(is_monthly=True)

    mean_var_slp_daily = np.average(var_slp_daily, weights=w_slp_2d); mean_var_slp_monthly = np.average(var_slp_monthly, weights=w_slp_2d)
    mean_var_sst_daily = np.average(var_sst_daily, weights=w_sst_2d); mean_var_sst_monthly = np.average(var_sst_monthly, weights=w_sst_2d)
    ratio_global_slp = (mean_var_slp_monthly / mean_var_slp_daily) * 100
    ratio_global_sst = (mean_var_sst_monthly / mean_var_sst_daily) * 100

    print("\n" + "="*50, flush=True)
    print("📊 GLOBAL SPATIALLY-WEIGHTED VARIANCE METRICS:", flush=True)
    print(f" - SLP Daily: {mean_var_slp_daily:.0f} Pa² | Monthly: {mean_var_slp_monthly:.0f} Pa² => Retained: {ratio_global_slp:.1f}%", flush=True)
    print(f" - SST Daily: {mean_var_sst_daily:.3f} K²  | Monthly: {mean_var_sst_monthly:.3f} K²  => Retained: {ratio_global_sst:.1f}%", flush=True)
    print("="*50 + "\n", flush=True)

    ratio_slp_map = (var_slp_monthly / np.where(var_slp_daily == 0, 1e-10, var_slp_daily)) * 100
    ratio_sst_map = (var_sst_monthly / np.where(var_sst_daily == 0, 1e-10, var_sst_daily)) * 100

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), subplot_kw={'projection': ccrs.PlateCarree()},gridspec_kw={'width_ratios': [1, 1.5]})
    
    zoom_box_slp, zoom_box_sst = [-100, 40, 20, 70], [-180, 180, -15, 70]
    ext_data_slp, ext_data_sst = zoom_box_slp, zoom_box_sst # au final c'est ça qu'il faut pour pas avoir du blanc au bord avec arrondis. 
    
    plots = [
        (axes[0,0], np.sqrt(var_slp_daily), ext_data_slp, zoom_box_slp, 'Reds', 2500, f"SLP Daily Standard Deviation (Nov, Dec, Jan, Feb)", 'Pa'),
        (axes[0,1], np.sqrt(var_sst_daily), ext_data_sst, zoom_box_sst, 'Reds', 2.0, f"SST Daily Standard Deviation (Aug, Sep, Oct, Nov, Dec, Jan, Feb)", 'K'),
        (axes[1,0], np.sqrt(var_slp_monthly), ext_data_slp, zoom_box_slp, 'Reds', 2500, f"SLP Monthly Standard Deviation (Nov, Dec, Jan, Feb)", 'Pa'),
        (axes[1,1], np.sqrt(var_sst_monthly), ext_data_sst, zoom_box_sst, 'Reds', 2.0, f"SST Monthly Standard Deviation (Aug, Sep, Oct, Nov, Dec, Jan, Feb)", 'K'),
        (axes[2,0], ratio_slp_map, ext_data_slp, zoom_box_slp, 'magma', 50, f"SLP Retained Variance Ratio (Global = {ratio_global_slp:.1f}%)", '%'),
        (axes[2,1], ratio_sst_map, ext_data_sst, zoom_box_sst, 'magma', 100, f"SST Retained Variance Ratio (Global = {ratio_global_sst:.1f}%)", '%')
    ]

    for ax, data, ext_data, zoom_box, cmap, vmax, title, unit in plots:
        im = ax.imshow(data, transform=ccrs.PlateCarree(), cmap=cmap, origin='lower', extent=ext_data, vmax=vmax)
        ax.set_extent(zoom_box, crs=ccrs.PlateCarree()); ax.coastlines(color='black', linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontweight='bold', fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
        cbar.set_label(unit)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "Variance_Daily_vs_Monthly.jpg")
    fig.savefig(save_path, dpi=200, pil_kwargs={'quality': 85, 'subsampling': 0}, bbox_inches='tight')
    plt.close()
    print(f"✅ Variance maps saved: {save_path}", flush=True)

# ==========================================
# 4. SLICED AUTOCORRELATION & CUMULATIVE DATABASE
# ==========================================
def process_autocorrelation_slice(member_ids, lag_min, lag_max, is_monthly, months_slp, months_sst, path_slp, path_sst, output_dir):
    time_unit = "Months" if is_monthly else "Days"
    print(f"\n--- Computing ACF ({time_unit}) Slice: Lags [{lag_min} -> {lag_max}] ---", flush=True)
    out_subfolder = os.path.join(output_dir, f"acf_maps_{'monthly' if is_monthly else 'daily'}")
    os.makedirs(out_subfolder, exist_ok=True)

    num_slp, den1_slp, den2_slp = None, None, None
    num_sst, den1_sst, den2_sst = None, None, None
    weights_slp_2d, weights_sst_2d = None, None
    ext_data_slp, ext_data_sst = None, None
    num_lags = (lag_max - lag_min) + 1

    for mem in member_ids:
        file_slp, file_sst = get_file_paths(path_slp, path_sst, mem, is_monthly)
        if not os.path.exists(file_slp): continue
        with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
            ds_slp = ds_slp.sel(
                    time=slice(None, "2014-12-31"),
                    drop=False,  # Sécurité
                )
            ds_sst = ds_sst.sel(
                    time=slice(None, "2014-12-31"),
                    drop=False,  # Sécurité
                )
            ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15, 70))
            slp_w = ds_slp.sel(time=ds_slp.time.dt.month.isin(months_slp))
            sst_w = ds_sst.sel(time=ds_sst.time.dt.month.isin(months_sst))
            slp_years = slp_w.time.dt.year.values; slp_months = slp_w.time.dt.month.values
            season_year_slp = np.where(slp_months <= 7, slp_years - 1, slp_years)
            split_idx_slp = np.where(np.diff(season_year_slp) != 0)[0] + 1

            sst_years = sst_w.time.dt.year.values; sst_months = sst_w.time.dt.month.values
            season_year_sst = np.where(sst_months <= 7, sst_years - 1, sst_years)
            split_idx_sst = np.where(np.diff(season_year_sst) != 0)[0] + 1
            v_slp = np.nan_to_num(slp_w['PSL'].values, nan=0.0).astype(np.float32, copy=False)
            v_sst = np.nan_to_num(sst_w['SST'].values, nan=0.0).astype(np.float32, copy=False)

            T_slp, S_slp = v_slp.shape[0], v_slp.shape[1] * v_slp.shape[2]
            T_sst, S_sst = v_sst.shape[0], v_sst.shape[1] * v_sst.shape[2]
            v_slp_2d = v_slp.reshape(T_slp, S_slp); v_slp_sq_2d = v_slp_2d**2
            v_sst_2d = v_sst.reshape(T_sst, S_sst); v_sst_sq_2d = v_sst_2d**2

            seasons_slp = np.split(v_slp_2d, split_idx_slp); seasons_slp_sq = np.split(v_slp_sq_2d, split_idx_slp)
            if len(months_sst) < 12:
                seasons_sst = np.split(v_sst_2d, split_idx_sst); seasons_sst_sq = np.split(v_sst_sq_2d, split_idx_sst)
            else:
                seasons_sst = [v_sst_2d]; seasons_sst_sq = [v_sst_sq_2d]

            if num_slp is None:
                ext_data_slp = [slp_w.lon.min().item(), slp_w.lon.max().item(), slp_w.lat.min().item(), slp_w.lat.max().item()]
                ext_data_sst = [sst_w.lon.min().item(), sst_w.lon.max().item(), sst_w.lat.min().item(), sst_w.lat.max().item()]
                shape_slp = (num_lags, v_slp.shape[1], v_slp.shape[2]); shape_sst = (num_lags, v_sst.shape[1], v_sst.shape[2])
                num_slp, den1_slp, den2_slp = np.zeros(shape_slp, dtype=np.float64), np.zeros(shape_slp, dtype=np.float64), np.zeros(shape_slp, dtype=np.float64)
                num_sst, den1_sst, den2_sst = np.zeros(shape_sst, dtype=np.float64), np.zeros(shape_sst, dtype=np.float64), np.zeros(shape_sst, dtype=np.float64)
                mask_slp = ~np.isnan(slp_w['PSL'].isel(time=0).values); mask_sst = ~np.isnan(sst_w['SST'].isel(time=0).values)
                weights_slp_2d = np.cos(np.deg2rad(slp_w.lat.values))[:, None] * mask_slp
                weights_sst_2d = np.cos(np.deg2rad(sst_w.lat.values))[:, None] * mask_sst

            for arr, sq in zip(seasons_slp, seasons_slp_sq):
                n_days = arr.shape[0]
                for idx, k in enumerate(range(lag_min, lag_max + 1)):
                    if k >= n_days: continue
                    x0, xk = (arr, arr) if k == 0 else (arr[:-k], arr[k:])
                    sq0, sqk = (sq, sq) if k == 0 else (sq[:-k], sq[k:])
                    num_slp[idx] += np.einsum('ts,ts->s', x0, xk).reshape(shape_slp[1], shape_slp[2])
                    den1_slp[idx] += np.sum(sq0, axis=0, dtype=np.float64).reshape(shape_slp[1], shape_slp[2])
                    den2_slp[idx] += np.sum(sqk, axis=0, dtype=np.float64).reshape(shape_slp[1], shape_slp[2])

            for arr, sq in zip(seasons_sst, seasons_sst_sq):
                n_days = arr.shape[0]
                for idx, k in enumerate(range(lag_min, lag_max + 1)):
                    if k >= n_days: continue
                    x0, xk = (arr, arr) if k == 0 else (arr[:-k], arr[k:])
                    sq0, sqk = (sq, sq) if k == 0 else (sq[:-k], sq[k:])
                    num_sst[idx] += np.einsum('ts,ts->s', x0, xk).reshape(shape_sst[1], shape_sst[2])
                    den1_sst[idx] += np.sum(sq0, axis=0, dtype=np.float64).reshape(shape_sst[1], shape_sst[2])
                    den2_sst[idx] += np.sum(sqk, axis=0, dtype=np.float64).reshape(shape_sst[1], shape_sst[2])

        print(f" -> Member {mem} accumulated.", end='\r', flush=True)

    print("\n -> Saving JPG maps and updating cumulative database...", flush=True)
    slice_slp_glob, slice_sst_glob, slice_slp_mean, slice_sst_mean = {}, {}, {}, {}
    zoom_box_slp, zoom_box_sst = [-100, 40, 20, 70], [-180, 180, -15, 70]

    for idx, k in enumerate(range(lag_min, lag_max + 1)):
        r_map_slp = num_slp[idx] / np.sqrt(np.where(den1_slp[idx]*den2_slp[idx] == 0, 1e-10, den1_slp[idx]*den2_slp[idx]))
        r_map_sst = num_sst[idx] / np.sqrt(np.where(den1_sst[idx]*den2_sst[idx] == 0, 1e-10, den1_sst[idx]*den2_sst[idx]))

        cov_g_slp = np.sum(num_slp[idx] * weights_slp_2d); cov_g_sst = np.sum(num_sst[idx] * weights_sst_2d)
        v1_g_slp = np.sum(den1_slp[idx] * weights_slp_2d); v2_g_slp = np.sum(den2_slp[idx] * weights_slp_2d)
        v1_g_sst = np.sum(den1_sst[idx] * weights_sst_2d); v2_g_sst = np.sum(den2_sst[idx] * weights_sst_2d)

        R_glob_slp = cov_g_slp / np.sqrt(v1_g_slp * v2_g_slp) if (v1_g_slp > 0 and v2_g_slp > 0) else np.nan
        R_glob_sst = cov_g_sst / np.sqrt(v1_g_sst * v2_g_sst) if (v1_g_sst > 0 and v2_g_sst > 0) else np.nan
        R_mean_slp = np.average(np.nan_to_num(r_map_slp, nan=0.0), weights=weights_slp_2d) if (v1_g_slp > 0 and v2_g_slp > 0) else np.nan
        R_mean_sst = np.average(np.nan_to_num(r_map_sst, nan=0.0), weights=weights_sst_2d) if (v1_g_sst > 0 and v2_g_sst > 0) else np.nan

        slice_slp_glob[k] = R_glob_slp; slice_sst_glob[k] = R_glob_sst
        slice_slp_mean[k] = R_mean_slp; slice_sst_mean[k] = R_mean_sst

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()},gridspec_kw={'width_ratios': [1, 1.5]})
        ax1 = axes[0]
        im1 = ax1.imshow(r_map_slp, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=ext_data_slp, vmin=-1, vmax=1)
        ax1.set_extent(zoom_box_slp, crs=ccrs.PlateCarree()); ax1.coastlines(color='black', alpha=0.5)
        ax1.set_title(f"SLP Autocorrelation — Lag {k} {time_unit}", fontweight='bold', fontsize=11)
        str_slp = f"Global R: {R_glob_slp:.3f} | Spatial Mean R: {R_mean_slp:.3f}" if not np.isnan(R_glob_slp) else "Lag > Season Length (N/A)"
        ax1.text(0.5, -0.16, str_slp, transform=ax1.transAxes, ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#cccccc'))
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.025, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1]); cbar1.set_label('Pearson r')

        ax2 = axes[1]
        im2 = ax2.imshow(r_map_sst, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=ext_data_sst, vmin=-1, vmax=1)
        ax2.set_extent(zoom_box_sst, crs=ccrs.PlateCarree()); ax2.coastlines(color='black', alpha=0.5)
        ax2.set_title(f"SST Autocorrelation — Lag {k} {time_unit}", fontweight='bold', fontsize=11)
        str_sst = f"Global R: {R_glob_sst:.3f} | Spatial Mean R: {R_mean_sst:.3f}" if not np.isnan(R_glob_sst) else "Lag > Season Length (N/A)"
        ax2.text(0.5, -0.16, str_sst, transform=ax2.transAxes, ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#cccccc'))
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.025, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1]); cbar2.set_label('Pearson r')

        fig.tight_layout()
        fig.savefig(os.path.join(out_subfolder, f"acf_lag_{k:02d}.jpg"), dpi=150, pil_kwargs={'quality': 95, 'subsampling': 0}, bbox_inches='tight')
        plt.close(fig)

    gif_name = "Autocorrelation_Animation_Monthly.gif" if is_monthly else "Autocorrelation_Animation_Daily.gif"
    create_gif_from_folder(out_subfolder, os.path.join(output_dir, gif_name))

    # --- UPDATING CUMULATIVE DATABASE (.NPZ) ---
    db_file = os.path.join(output_dir, f"database_acf_{'monthly' if is_monthly else 'daily'}.npz")
    db = {}
    if os.path.exists(db_file):
        loaded = np.load(db_file, allow_pickle=True)
        db = {k: loaded[k].item() for k in loaded.files}
    for k in range(lag_min, lag_max + 1):
        db[f"lag_{k}"] = {
            'slp_glob': slice_slp_glob[k], 'sst_glob': slice_sst_glob[k],
            'slp_mean': slice_slp_mean[k], 'sst_mean': slice_sst_mean[k]
        }
    np.savez(db_file, **db)
    return db

def plot_all_acf_analyses(member_ids, months_slp, months_sst, path_slp, path_sst, output_dir, args):
    print(f"\n=== [3/3] Merging and Plotting Cumulative Autocorrelations ===", flush=True)
    db_daily = process_autocorrelation_slice(member_ids, args.lag_min_daily, args.lag_max_daily, False, months_slp, months_sst, path_slp, path_sst, output_dir)
    db_monthly = process_autocorrelation_slice(member_ids, args.lag_min_monthly, args.lag_max_monthly, True, months_slp, months_sst, path_slp, path_sst, output_dir)

    lags_d = sorted([int(k.split('_')[1]) for k in db_daily.keys()])
    slp_d_g = [db_daily[f"lag_{k}"]['slp_glob'] for k in lags_d]; sst_d_g = [db_daily[f"lag_{k}"]['sst_glob'] for k in lags_d]
    slp_d_m = [db_daily[f"lag_{k}"]['slp_mean'] for k in lags_d]; sst_d_m = [db_daily[f"lag_{k}"]['sst_mean'] for k in lags_d]

    lags_m = sorted([int(k.split('_')[1]) for k in db_monthly.keys()])
    slp_m_g = [db_monthly[f"lag_{k}"]['slp_glob'] for k in lags_m]; sst_m_g = [db_monthly[f"lag_{k}"]['sst_glob'] for k in lags_m]
    slp_m_m = [db_monthly[f"lag_{k}"]['slp_mean'] for k in lags_m]; sst_m_m = [db_monthly[f"lag_{k}"]['sst_mean'] for k in lags_m]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    if lags_d:
        axes[0].plot(lags_d, slp_d_g, '-', color='#d95f02', linewidth=2.2, label='SLP — Global R (Pooled)')
        axes[0].plot(lags_d, slp_d_m, '--', color='#d95f02', linewidth=1.8, alpha=0.8, label='SLP — Spatial Mean R')
        axes[0].plot(lags_d, sst_d_g, '-', color='#7570b3', linewidth=2.2, label='SST — Global R (Pooled)')
        axes[0].plot(lags_d, sst_d_m, '--', color='#7570b3', linewidth=1.8, alpha=0.8, label='SST — Spatial Mean R')
        # axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8); axes[0].axhline(1/np.e, color='gray', linestyle=':', label='e-folding Timescale')
        axes[0].set_title(f"Daily Persistence (SLP: Nov, Dec, Jan, Feb) / SST: Aug, Sep, Oct, Nov, Dec, Jan, Feb)", fontweight='bold')
        axes[0].set_xlabel("Lag (Days)"); axes[0].set_ylabel("Autocorrelation (r)"); axes[0].set_xlim(min(lags_d), max(lags_d)); axes[0].set_ylim(-0.1, 1.05)
        axes[0].grid(True, linestyle='--', alpha=0.5); axes[0].legend(fontsize=9.5)

    if lags_m:
        axes[1].plot(lags_m, slp_m_g, 'o-', color='#d95f02', linewidth=2.2, label='SLP — Global R (Pooled)')
        axes[1].plot(lags_m, slp_m_m, '--', color='#d95f02', linewidth=1.8, alpha=0.8, label='SLP — Spatial Mean R')
        axes[1].plot(lags_m, sst_m_g, 's-', color='#7570b3', linewidth=2.2, label='SST — Global R (Pooled)')
        axes[1].plot(lags_m, sst_m_m, '--', color='#7570b3', linewidth=1.8, alpha=0.8, label='SST — Spatial Mean R')
        # axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8); axes[1].axhline(1/np.e, color='gray', linestyle=':', label='e-folding Timescale')
        axes[1].set_title(f"Monthly Persistence (SLP: Nov, Dec, Jan, Feb) / SST: Aug, Sep, Oct, Nov, Dec, Jan, Feb)", fontweight='bold')
        axes[1].set_xlabel("Lag (Months)"); axes[1].set_ylabel("Autocorrelation (r)"); axes[1].set_xlim(min(lags_m), max(lags_m)); axes[1].set_ylim(-0.2, 1.05)
        axes[1].set_xticks(lags_m); axes[1].grid(True, linestyle='--', alpha=0.5); axes[1].legend(fontsize=9.5)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "ACF_Curves_Daily_Monthly.jpg")
    fig.savefig(save_path, dpi=200, pil_kwargs={'quality': 85})
    plt.close()
    print(f"\n✅ ACF curves saved: {save_path}", flush=True)

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
    parser.add_argument('--number_of_members', type=int, default=10)
    parser.add_argument('--lag_min_daily', type=int, default=0, help="Minimum daily lag to compute")
    parser.add_argument('--lag_max_daily', type=int, default=30, help="Maximum daily lag to compute")
    parser.add_argument('--lag_min_monthly', type=int, default=0, help="Minimum monthly lag to compute")
    parser.add_argument('--lag_max_monthly', type=int, default=6, help="Maximum monthly lag to compute")
    args = parser.parse_args()

    folder_name = "report_plots_variance_autocorr_updatable_english_89_members"
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
    print(f"Using {len(members_used)} members")
    months_slp = [11, 12, 1, 2]           # NDJF for atmosphere
    months_sst = [3,4,5,6,7, 8, 9, 10, 11, 12, 1, 2] # ASONDJF for ocean

    plot_intramonthly_variance_histogram(members_used, months_slp, months_sst, path_slp, path_sst, base_home)
    compute_variance_maps(members_used, months_slp, months_sst, path_slp, path_sst, base_home)
    plot_all_acf_analyses(members_used, months_slp, months_sst, path_slp, path_sst, base_home, args)