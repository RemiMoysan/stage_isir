import os
import argparse
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import random
import calendar
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
import re
from matplotlib.colors import SymLogNorm, Normalize, ListedColormap
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec

import torch
import torch.nn as nn

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# import des dossiers siblings
import sys
from pathlib import Path



def compute_shap_regression_slope(shap_array, input_array):
    shap_mean = np.mean(shap_array, axis=0)
    input_mean = np.mean(input_array, axis=0)
    shap_centered = shap_array - shap_mean
    input_centered = input_array - input_mean
    numerator = np.sum(shap_centered * input_centered, axis=0)
    denominator = np.sum(input_centered**2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope_map = numerator / denominator
        slope_map = np.nan_to_num(slope_map, nan=0.0)
    return slope_map

def plot_attribution_maps(mean_attr_array, lags, extent, display_title, filename_prefix, outdir, feature_name="SST", negative_value=False, time_unit="d", cbar_label=""):
    # Tri chronologique : Les grands lags (plus loin dans le passé) à gauche, les récents à droite
    sorted_indices = np.argsort(lags)[::-1]
    lags_ordered = [lags[idx] for idx in sorted_indices]
    mean_attr_array_ordered = mean_attr_array[sorted_indices]

    num_lags = len(lags_ordered)
    fig, axes = plt.subplots(1, num_lags, figsize=(6 * num_lags, 4), subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
    if num_lags == 1: axes = [axes]

    if negative_value:
        vmax = np.percentile(np.abs(mean_attr_array_ordered), 99) or 1e-6
        vmin = -vmax
        cmap = 'RdBu_r'
    else:
        vmax = np.percentile(mean_attr_array_ordered, 99) or 1e-6
        vmin = 0
        cmap = 'Reds' 

    for i, lag in enumerate(lags_ordered):
        ax = axes[i]
        ax.set_facecolor('white')
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.8)
        im = ax.imshow(mean_attr_array_ordered[i], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent, interpolation='nearest')
        ax.set_title(f"Lag {-lag}{time_unit}", fontsize=12)
        
        cb = fig.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.08)
        cb.locator = ticker.MaxNLocator(nbins=4) 
        cb.formatter = ticker.ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((-2, 2)) 
        cb.update_ticks()
        
        if cbar_label:
            cb.set_label(cbar_label, fontsize=11)

    plt.suptitle(display_title, fontsize=16, y=1.05)
    plt.tight_layout()
    filename = f"{filename_prefix}_{feature_name}.png"
    plt.savefig(os.path.join(outdir, filename), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()


def generate_summary_plots(shap_np, inputs_np, lags, extent, outdir, display_name, file_prefix, time_unit, pc1_std):
    shap_norm = shap_np / pc1_std
    inputs_k = inputs_np * 0.707  # Conversion en Kelvin
    
    mean_abs_shap = np.mean(np.abs(shap_norm), axis=0)
    plot_attribution_maps(mean_abs_shap, lags, extent, f"Absolute Importance - {display_name}", f"Importance_Absolue_{file_prefix}", outdir, "SST", negative_value=False, time_unit=time_unit, cbar_label="SHAP (Unitless)")
    
    slope_map = compute_shap_regression_slope(shap_norm, inputs_k)
    plot_attribution_maps(slope_map, lags, extent, f"Sensibility (SST $\\rightarrow$ SLP) - {display_name}", f"Sensibilite_{file_prefix}", outdir, "SST_Slope", negative_value=True, time_unit=time_unit, cbar_label="Sensitivity (Unitless / K)")

    std_inputs_k = np.std(inputs_k, axis=0)
    plot_attribution_maps(slope_map * std_inputs_k, lags, extent, f"Typical Impact (SST $\\rightarrow$ SLP) - {display_name}", f"Impact_Typique_{file_prefix}", outdir, "SST_Typical", negative_value=True, time_unit=time_unit, cbar_label="Typical Impact (Unitless)")


def plot_individual_sample(input_sst, shap_sst, pred_val, target_val, pred_map, target_map_recon, true_map, lags, extent_sst, extent_slp, member, date, dim_c, outdir, time_unit, pc1_std, dynamic_slp_std):
    # Passage aux unités physiques
    input_sst_k = input_sst * 0.707
    pred_map_pa = pred_map * dynamic_slp_std
    target_map_recon_pa = target_map_recon * dynamic_slp_std
    true_map_pa = true_map * dynamic_slp_std

    # Tri chronologique
    sorted_indices = np.argsort(lags)[::-1]
    lags_ordered = [lags[idx] for idx in sorted_indices]
    input_sst_ordered = input_sst_k[sorted_indices]
    shap_sst_ordered = shap_sst[sorted_indices]

    num_lags = len(lags_ordered)
        # Hauteur passée de 11 à 7.5 pour supprimer l'espace vide
    fig = plt.figure(figsize=(max(12, 4.5 * num_lags), 7.5), facecolor='white')

    
    width_ratios_top = [1] * num_lags + [0.03]
    # top abaissé, bottom remonté, hspace réduit
    gs_top = GridSpec(2, num_lags + 1, figure=fig, top=0.82, bottom=0.45, hspace=0.1, wspace=0.15, width_ratios=width_ratios_top)
    
    width_ratios_bottom = [1, 1, 1, 0.05]
    # top remonté pour coller au bloc du haut
    gs_bottom = GridSpec(1, 4, figure=fig, top=0.35, bottom=0.05, wspace=0.15, width_ratios=width_ratios_bottom)



    p_val_norm = pred_val / pc1_std
    t_val_norm = target_val / pc1_std if target_val is not None else np.nan
    
    vmax_in = np.percentile(np.abs(input_sst_ordered), 99) or 1e-6
    vmax_shap = np.percentile(np.abs(shap_sst_ordered / pc1_std), 99) or 1e-6
    vmax_slp = np.percentile(np.abs(true_map_pa), 99) or 1e-6
    
    cmap_shap_scalar = plt.get_cmap('PiYG_r')
    norm_shap_scalar = mcolors.Normalize(vmin=-3, vmax=3)

    def get_color_and_contrast(val, norm, cmap):
        rgba = cmap(norm(val))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = 'white' if lum < 0.55 else 'black'
        return rgba, text_color

    for i, lag in enumerate(lags_ordered):
        ax_in = fig.add_subplot(gs_top[0, i], projection=ccrs.PlateCarree())
        ax_in.set_extent(extent_sst, crs=ccrs.PlateCarree())
        ax_in.coastlines(linewidth=0.8)
        im_in = ax_in.imshow(input_sst_ordered[i], cmap='RdBu_r', origin='lower', vmin=-vmax_in, vmax=vmax_in, transform=ccrs.PlateCarree(), extent=extent_sst)
        ax_in.set_title(f"SST Input Lag {-lag}{time_unit}", fontsize=12)
        
        ax_sh = fig.add_subplot(gs_top[1, i], projection=ccrs.PlateCarree())
        ax_sh.set_extent(extent_sst, crs=ccrs.PlateCarree())
        ax_sh.coastlines(linewidth=0.8)
        im_sh = ax_sh.imshow((shap_sst_ordered[i] / pc1_std), cmap='PiYG_r', origin='lower', vmin=-vmax_shap, vmax=vmax_shap, transform=ccrs.PlateCarree(), extent=extent_sst)
        ax_sh.set_title(f"SHAP Lag {-lag}{time_unit} (Unitless)", fontsize=12)

    cax_in = fig.add_subplot(gs_top[0, -1])
    cbar_in = fig.colorbar(im_in, cax=cax_in, orientation='vertical')
    cbar_in.set_label("SST Anomaly (K)", fontsize=11)
    
    cax_sh = fig.add_subplot(gs_top[1, -1])
    cbar_sh = fig.colorbar(im_sh, cax=cax_sh, orientation='vertical')
    cbar_sh.set_label("SHAP (Unitless)", fontsize=11)

    titles_slp = [f"Predicted Embedding Reconstruction\n{p_val_norm:.2f} $\sigma$", 
                  f"Target Embedding Reconstruction\n{t_val_norm:.2f} $\sigma$" if not np.isnan(t_val_norm) else "Target Embedding Reconstruction (N/A)", 
                  "True Target SLP Map"]
    maps_slp_pa = [pred_map_pa, target_map_recon_pa, true_map_pa]
    
    for i in range(3):
        ax_slp = fig.add_subplot(gs_bottom[0, i], projection=ccrs.PlateCarree())
        ax_slp.set_extent(extent_slp, crs=ccrs.PlateCarree())
        ax_slp.coastlines(linewidth=0.8)
        
        map_2d = np.squeeze(maps_slp_pa[i])
        im_slp = ax_slp.imshow(map_2d, cmap='RdBu_r', origin='lower', vmin=-vmax_slp, vmax=vmax_slp, transform=ccrs.PlateCarree(), extent=extent_slp)
        
        if i == 0:
            bg_col, text_col = get_color_and_contrast(p_val_norm, norm_shap_scalar, cmap_shap_scalar)
            bbox_props = dict(boxstyle="round,pad=0.3", fc=bg_col, ec="gray", alpha=0.9)
        elif i == 1 and not np.isnan(t_val_norm):
            bg_col, text_col = get_color_and_contrast(t_val_norm, norm_shap_scalar, cmap_shap_scalar)
            bbox_props = dict(boxstyle="round,pad=0.3", fc=bg_col, ec="gray", alpha=0.9)
        else:
            text_col, bbox_props = 'black', None
            
        ax_slp.set_title(titles_slp[i], fontsize=12, color=text_col, bbox=bbox_props, pad=12)

    gs_cb = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_bottom[0, -1], height_ratios=[0.15, 0.7, 0.15])
    cax_slp = fig.add_subplot(gs_cb[1, 0])
    
    cbar_slp = fig.colorbar(im_slp, cax=cax_slp, orientation='vertical')
    cbar_slp.set_label("SLP Anomaly (Pa)", fontsize=12)

    target_str = f" | Target : {t_val_norm:.3f} $\sigma$" if not np.isnan(t_val_norm) else ""
    plt.suptitle(f"Member : {member}  |  Date : {date}  |  Component {dim_c}\nPrediction : {p_val_norm:.3f} $\sigma${target_str}", fontsize=12, fontweight='bold', y=0.96)

    mem_dir = os.path.join(outdir, str(member))
    os.makedirs(mem_dir, exist_ok=True)
    
    date_str_clean = str(date).replace(" ", "_").replace(":", "-")
    filename = f"shap_sample_{date_str_clean}_dim{dim_c}.png"
    plt.savefig(os.path.join(mem_dir, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================
# STATISTICAL & PLOTTING FUNCTIONS EMBEDDING : Désuet car le cas spatial avec compute_map_metrics_and_bootstraps englobe tout ceci 
# Celle la fonction plot_combined_pcs_time_series n'est pas désuète car elle a la mise en page intuitive pour l'évaluation embedding. 
# Pour cette même raison plot_latent_timeseries_raw_metrics n'est pas désuet
# ============================================================

def compute_latent_metrics_and_bootstraps(Z_t, Z_p, dates, n_bootstraps=300):
    """
    Moteur unifié pour l'espace latent (T, D). Sans pondération spatiale.
    Calcule en une seule passe :
    1. Les séries temporelles instantanées (pour plot_latent_timeseries_raw_metrics)
    2. Les 6 métriques globales et moyennées par composante + p-values sous H0
    3. Les statistiques et p-values individuelles pour CHAQUE composante PC1..PCD
    """
    T_len, D = Z_t.shape
    
    # --- 1. SÉRIES TEMPORELLES INSTANTANÉES (à chaque pas de temps t) ---
    mse_inst = np.mean((Z_p - Z_t)**2, axis=1)
    var_inst = np.mean(Z_t**2, axis=1) + 1e-8
    mae_inst = np.mean(np.abs(Z_p - Z_t), axis=1)
    base_mae_inst = np.mean(np.abs(Z_t), axis=1) + 1e-8
    
    Z_p_sub = Z_p - np.mean(Z_p, axis=1, keepdims=True)
    Z_t_sub = Z_t - np.mean(Z_t, axis=1, keepdims=True)
    cov_inst = np.mean(Z_p_sub * Z_t_sub, axis=1)
    std_p_inst = np.sqrt(np.mean(Z_p_sub**2, axis=1) + 1e-8)
    std_t_inst = np.sqrt(np.mean(Z_t_sub**2, axis=1) + 1e-8)
    corr_inst = cov_inst / (std_p_inst * std_t_inst)
    
    df_lat_ts = pd.DataFrame({
        'time': dates, 'mse_pred': mse_inst, 'base_var': var_inst,
        'mae_pred': mae_inst, 'base_mae': base_mae_inst, 'corr_inst': corr_inst
    })

    # --- 2. STATISTIQUES PAR COMPOSANTE (PC1 .. PCD) ---
    t_mean = Z_t.mean(axis=0); p_mean = Z_p.mean(axis=0)
    var_t_c = np.var(Z_t, axis=0) + 1e-8; var_p_c = np.var(Z_p, axis=0) + 1e-8
    cov_c = np.mean((Z_t - t_mean) * (Z_p - p_mean), axis=0)
    mse_c = np.mean((Z_t - Z_p)**2, axis=0); rmse_c = np.sqrt(mse_c)
    nrmse_c = rmse_c / np.sqrt(var_t_c)
    
    r2_c = 1.0 - (mse_c / var_t_c)
    corr_c = cov_c / np.sqrt(var_t_c * var_p_c)
    mae_c = np.mean(np.abs(Z_t - Z_p), axis=0); mae_ref_c = np.mean(np.abs(Z_t), axis=0) + 1e-8
    l1_c = 1.0 - (mae_c / mae_ref_c)

    # --- 3. LES 6 MÉTRIQUES GLOBALES & MOYENNÉES PAR COMPOSANTE ---
    mc_r2_orig   = np.mean(r2_c);   mc_l1_orig   = np.mean(l1_c);   mc_corr_orig = np.mean(corr_c)
    g_r2_orig    = 1.0 - (np.mean(mse_c) / np.mean(var_t_c))
    g_l1_orig    = 1.0 - (np.mean(mae_c) / np.mean(mae_ref_c))
    
    z_p_gmean = np.mean(Z_p); z_t_gmean = np.mean(Z_t)
    cov_g = np.mean((Z_p - z_p_gmean) * (Z_t - z_t_gmean))
    g_corr_orig  = cov_g / np.sqrt((np.mean((Z_p - z_p_gmean)**2) + 1e-8) * (np.mean((Z_t - z_t_gmean)**2) + 1e-8))

    # --- 4. BOOTSTRAP SOUS H0 (PERMUTATION TEMPORELLE) ---
    boot_r2_cnt = np.zeros(D, dtype=int); boot_l1_cnt = np.zeros(D, dtype=int); boot_corr_cnt = np.zeros(D, dtype=int)
    cnt = {k: 0 for k in ['g_r2', 'mc_r2', 'g_l1', 'mc_l1', 'g_corr', 'mc_corr']}
    
    print(f"    Running {n_bootstraps} bootstraps for latent vector...", end="", flush=True)
    for _ in range(n_bootstraps):
        idx = np.random.choice(T_len, size=T_len, replace=True)
        Z_t_b = Z_t[idx]
        
        # Par composante sur le tirage
        t_b_mean = Z_t_b.mean(axis=0); var_t_c_b = np.var(Z_t_b, axis=0) + 1e-8
        cov_c_b = np.mean((Z_t_b - t_b_mean) * (Z_p - p_mean), axis=0); mse_c_b = np.mean((Z_t_b - Z_p)**2, axis=0)
        mae_c_b = np.mean(np.abs(Z_t_b - Z_p), axis=0); mae_ref_c_b = np.mean(np.abs(Z_t_b), axis=0) + 1e-8
        
        r2_c_b = 1.0 - (mse_c_b / var_t_c_b); corr_c_b = cov_c_b / np.sqrt(var_t_c_b * var_p_c); l1_c_b = 1.0 - (mae_c_b / mae_ref_c_b)
        
        boot_r2_cnt += (r2_c_b >= r2_c); boot_l1_cnt += (l1_c_b >= l1_c); boot_corr_cnt += (np.abs(corr_c_b) >= np.abs(corr_c))
        
        # Global & Mean-Comp sur le tirage
        mc_r2_b = np.mean(r2_c_b); mc_l1_b = np.mean(l1_c_b); mc_corr_b = np.mean(corr_c_b)
        g_r2_b  = 1.0 - (np.mean(mse_c_b) / np.mean(var_t_c_b)); g_l1_b  = 1.0 - (np.mean(mae_c_b) / np.mean(mae_ref_c_b))
        
        t_g_mean_b = np.mean(Z_t_b); cov_gb = np.mean((Z_p - z_p_gmean) * (Z_t_b - t_g_mean_b))
        g_corr_b   = cov_gb / np.sqrt((np.mean((Z_p - z_p_gmean)**2) + 1e-8) * (np.mean((Z_t_b - t_g_mean_b)**2) + 1e-8))
        
        cnt['g_r2'] += (g_r2_b >= g_r2_orig); cnt['mc_r2'] += (mc_r2_b >= mc_r2_orig)
        cnt['g_l1'] += (g_l1_b >= g_l1_orig); cnt['mc_l1'] += (mc_l1_b >= mc_l1_orig)
        cnt['g_corr'] += (np.abs(g_corr_b) >= np.abs(g_corr_orig)); cnt['mc_corr'] += (np.abs(mc_corr_b) >= np.abs(mc_corr_orig))
        
    print(" Done!")

    # --- 5. FORMATTAGE DES RETOURS ---
    stats_global_dict = {
        'global_r2': (g_r2_orig, cnt['g_r2']/n_bootstraps), 'mean_comp_r2': (mc_r2_orig, cnt['mc_r2']/n_bootstraps),
        'global_l1': (g_l1_orig, cnt['g_l1']/n_bootstraps), 'mean_comp_l1': (mc_l1_orig, cnt['mc_l1']/n_bootstraps),
        'global_corr': (g_corr_orig, cnt['g_corr']/n_bootstraps), 'mean_comp_corr': (mc_corr_orig, cnt['mc_corr']/n_bootstraps)
    }
    
    stats_per_pc = {}
    for i in range(D):
        stats_per_pc[i+1] = (
            corr_c[i], rmse_c[i], nrmse_c[i], r2_c[i], l1_c[i],
            boot_corr_cnt[i]/n_bootstraps, boot_r2_cnt[i]/n_bootstraps, boot_l1_cnt[i]/n_bootstraps
        )

    return df_lat_ts, stats_global_dict, stats_per_pc

def plot_combined_pcs_time_series(df_month, stats_per_pc, stats_global, month_num, member, outdir, max_pcs=5, freq_label="Daily", quantiles_dict=None):
    """
    Version corrigée : 
    - Supprime le suptitle redondant quand k=1.
    - Marges ajustées dynamiquement.
    """
    month_name = calendar.month_name[month_num]
    k = min(max_pcs, len(stats_per_pc))
    
    fig, axes = plt.subplots(k, 1, figsize=(16, 4.0 * k), sharex=True)
    if k == 1: axes = [axes]

    x_idx = np.arange(len(df_month))
    x_labels = df_month['time'].dt.strftime('%Y-%m-%d' if freq_label == 'Daily' else '%Y-%m').tolist()
    lw = 0.7 if freq_label == 'Daily' else 1.5; mk = None if freq_label == 'Daily' else '.'; alpha_v = 0.85

    for i in range(k):
        pc_idx = i + 1
        ax = axes[i]
        true_vals = df_month[f'true_pc{pc_idx}'].values
        pred_vals = df_month[f'pred_pc{pc_idx}'].values
        std_true = np.std(true_vals) if np.std(true_vals) > 0 else 1.0

        # --- FOND COLORÉ GRADUEL ---
        ymin = min(-2.5, np.min(true_vals/std_true) * 1.2)
        ymax = max(2.5, np.max(true_vals/std_true) * 1.2)
        
        y_gradient = np.linspace(ymin, ymax, 256).reshape(-1, 1)
        ax.imshow(y_gradient, aspect='auto', cmap='PiYG_r', origin='lower',
                  extent=[x_idx[0]-0.5, x_idx[-1]+0.5, ymin, ymax], 
                  alpha=0.15, zorder=0, vmin=-3, vmax=3)
        
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(x_idx[0]-0.5, x_idx[-1]+0.5)

        # --- STATS INDIVIDUELLES ---
        corr, rmse, nrmse, r2, ss_l1, pval_corr, pval_r2, pval_l1 = stats_per_pc[pc_idx]

        r2_str = f"R² = {r2:.3f} (p = N/A)" if r2 < 0 else f"R² = {r2:.3f}{'**' if pval_r2 < 0.01 else ('*' if pval_r2 < 0.05 else '')} (p={pval_r2:.3f})"
        ss_l1_str = f"L1 = {ss_l1:.3f} (p = N/A)" if ss_l1 < 0 else f"L1 = {ss_l1:.3f}{'**' if pval_l1 < 0.01 else ('*' if pval_l1 < 0.05 else '')} (p={pval_l1:.3f})"
        corr_str = f"Corr = {corr:.2f} (p = N/A)" if corr < 0 else f"Corr = {corr:.2f}{'**' if pval_corr < 0.01 else ('*' if pval_corr < 0.05 else '')} (p={pval_corr:.3f})"

        # --- GESTION INTELLIGENTE DES TITRES ---
        if k == 1:
            title_pc = (f"Latent Space Evaluation - Member {member} - {month_name} ({freq_label})\n"
                        f"Normalized PC1 |  {r2_str}  |  {ss_l1_str}  |  {corr_str}")
            ax.set_title(title_pc, fontsize=13, pad=12, fontweight='bold')
        else:
            title_pc = f"Normalized Component {pc_idx} (Original Magnitude: {std_true:.2f})\n{r2_str}   |   {ss_l1_str}   |   {corr_str}"
            ax.set_title(title_pc, fontsize=11, pad=10)

        # --- COURBES ---
        if quantiles_dict and pc_idx in quantiles_dict:
            q_keys = sorted(list(quantiles_dict[pc_idx].keys()))
            q_lower = sorted([q for q in q_keys if q < 0.5], reverse=True); q_upper = sorted([q for q in q_keys if q > 0.5])
            for idx_q in range(min(len(q_lower), len(q_upper))):
                ql = quantiles_dict[pc_idx][q_lower[idx_q]].values; qu = quantiles_dict[pc_idx][q_upper[idx_q]].values
                ax.fill_between(x_idx, ql/std_true, qu/std_true, color='tab:blue', alpha=0.15, lw=0, zorder=1, label=f"Quantiles ({min(q_lower)}-{max(q_upper)})" if (i==0 and idx_q==0) else "")
        
        label = "Predicted (Median)" if quantiles_dict else "Predicted"
        ax.plot(x_idx, true_vals/std_true, color="black", lw=lw, marker=mk, alpha=alpha_v, zorder=2, label="True" if i==0 else "")
        ax.plot(x_idx, pred_vals/std_true, color="navy", lw=lw, marker=mk, alpha=alpha_v, zorder=2, label=label if i==0 else "")
        
        ax.set_ylabel(f"Norm. Comp. {pc_idx}")
        ax.grid(True, linestyle=':', alpha=0.5, zorder=1)

    n_ticks = min(15, len(x_idx))
    tick_indices = np.linspace(0, len(x_idx) - 1, n_ticks, dtype=int)
    axes[-1].set_xticks(tick_indices)
    axes[-1].set_xticklabels([x_labels[idx] for idx in tick_indices], rotation=45, ha="right")
    axes[-1].set_xlabel("Time", fontsize=12)

    # --- AJUSTEMENT STRICT DES MARGES ET DU SUPTITLE ---
    if k > 1:
        g_r2, p_gr2 = stats_global['global_r2']; mc_r2, p_mcr2 = stats_global['mean_comp_r2']
        g_l1, p_gl1 = stats_global['global_l1']; mc_l1, p_mcl1 = stats_global['mean_comp_l1']
        g_c, p_gc   = stats_global['global_corr']; mc_c, p_mcc   = stats_global['mean_comp_corr']

        sup_str = (f"Latent Space Evaluation - Member {member} - {month_name} ({freq_label})\n"
                   f"L2 Skill: Global R² = {g_r2:.2f} (p={p_gr2:.3f})  |  Mean Comp R² = {mc_r2:.2f} (p={p_mcr2:.3f})\n"
                   f"L1 Skill: Global L1 = {g_l1:.2f} (p={p_gl1:.3f})  |  Mean Comp L1 = {mc_l1:.2f} (p={p_mcl1:.3f})\n"
                   f"Correlation: Global Corr = {g_c:.2f} (p={p_gc:.3f})  |  Mean Comp Corr = {mc_c:.2f} (p={p_mcc:.3f})")
        fig.suptitle(sup_str, fontsize=14, fontweight='bold', y=0.96)
        fig.subplots_adjust(top=0.82, bottom=0.10, hspace=0.60)
    else:
        # Moins d'espace vide en haut pour un seul plot
        fig.subplots_adjust(top=0.88, bottom=0.18)

    # Légende repositionnée pour éviter de sortir du cadre si top est grand
    # fig.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=3, fontsize=10)
        # On accroche la légende au coin supérieur droit du premier sous-graphique
    axes[0].legend(loc="upper right", ncol=3, fontsize=10, framealpha=0.9)

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'Combined_PCs_Time_Series_{calendar.month_abbr[month_num]}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_latent_timeseries_raw_metrics(df_lat_ts, member, outdir, freq_label, stats_lat):
    """
    Trace 3 subplots (MSE latente, MAE latente, Corrélation instantanée).
    Le titre affiche le score GLOBAL et le score MOYENNÉ PAR COMPOSANTE (Mean Comp) avec p-values.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    
    df_plot = df_lat_ts.copy()
    lw = 0.7 if freq_label == 'Daily' else 1.5
    mk = None if freq_label == 'Daily' else '.'
    alpha_v = 0.85 if freq_label == 'Daily' else 1.0

    x_idx = np.arange(len(df_plot))
    x_labels = df_plot['time'].dt.strftime('%Y-%m-%d' if freq_label == 'Daily' else '%Y-%m').tolist() if 'time' in df_plot else [str(i) for i in x_idx]

    # --- SUBPLOT 1 : MSE vs BASELINE (VARIANCE 0) ---
    g_r2, p_gr2 = stats_lat['global_r2']; mc_r2, p_mcr2 = stats_lat['mean_comp_r2']
    title_mse = (f"1. Latent MSE vs Baseline Variance (0) - Member {member}\n"
                 f"Global R²={g_r2:.2f} (p={p_gr2:.3f}) | Mean-Comp R²={mc_r2:.2f} (p={p_mcr2:.3f})")
    
    ax1.plot(x_idx, df_plot['mse_pred'], label="Model Latent MSE", color="firebrick", lw=lw, marker=mk, alpha=alpha_v)
    ax1.plot(x_idx, df_plot['base_var'], label="Baseline 0 (Latent Variance)", color="darkred", lw=1.5, linestyle="--", alpha=0.9)
    ax1.set_title(title_mse, fontsize=11.5, pad=8); ax1.set_ylabel("MSE"); ax1.grid(True, linestyle=':', alpha=0.7); ax1.legend(loc="upper right", fontsize=10)

    # --- SUBPLOT 2 : MAE (NORME L1) vs BASELINE MAE ---
    g_l1, p_gl1 = stats_lat['global_l1']; mc_l1, p_mcl1 = stats_lat['mean_comp_l1']
    title_mae = (f"2. Latent MAE vs Baseline MAE (0) - Member {member}\n"
                 f"Global SS_L1={g_l1:.2f} (p={p_gl1:.3f}) | Mean-Comp SS_L1={mc_l1:.2f} (p={p_mcl1:.3f})")
    
    ax2.plot(x_idx, df_plot['mae_pred'], label="Model Latent MAE", color="firebrick", lw=lw, marker=mk, alpha=alpha_v)
    ax2.plot(x_idx, df_plot['base_mae'], label="Baseline 0 (Latent MAE)", color="darkred", lw=1.5, linestyle="--", alpha=0.9)
    ax2.set_title(title_mae, fontsize=11.5, pad=8); ax2.set_ylabel("MAE"); ax2.grid(True, linestyle=':', alpha=0.7); ax2.legend(loc="upper right", fontsize=10)

    # --- SUBPLOT 3 : CORRÉLATION INSTANTANÉE SUR LE VECTEUR LATENT ---
    g_c, p_gc = stats_lat['global_corr']; mc_c, p_mcc = stats_lat['mean_comp_corr']
    title_corr = (f"3. Instantaneous Latent Vector Correlation ($r_{{latent}}$) - Member {member}\n"
                  f"Global Corr={g_c:.2f} (p={p_gc:.3f}) | Mean-Comp Corr={mc_c:.2f} (p={p_mcc:.3f})")
    
    ax3.plot(x_idx, df_plot['corr_inst'], label="Model Latent Correlation", color="firebrick", lw=lw, marker=mk, alpha=alpha_v)
    ax3.axhline(0, color='black', linestyle='--', lw=1.5, alpha=0.9, label="Baseline 0 Correlation")
    ax3.set_title(title_corr, fontsize=11.5, pad=8); ax3.set_ylabel("$r$"); ax3.grid(True, linestyle=':', alpha=0.7); ax3.legend(loc="upper right", fontsize=10)

    n_ticks = min(15, len(x_idx))
    tick_indices = np.linspace(0, len(x_idx) - 1, n_ticks, dtype=int)
    ax3.set_xticks(tick_indices); ax3.set_xticklabels([x_labels[idx] for idx in tick_indices], rotation=45, ha="right")
    ax3.set_xlabel("Time", fontsize=14)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'Latent_Raw_Metrics_Timeseries_{freq_label}_Member_{member}.png'), dpi=200)
    plt.close(fig)

# ============================================================
# PLOTTING FUNCTIONS SPATIAL
# ============================================================

def plot_spatial_timeseries_raw_metrics(df_member, member, outdir, freq_label, stats_true, stats_rec):
    """
    Trace 3 subplots représentant la physique brute du modèle au fil du temps :
    1. MSE (Modèle vs Baseline Variance 0)
    2. MAE / Norme L1 (Modèle vs Baseline MAE 0)
    3. Corrélation spatiale instantanée
    Les titres incluent les métriques globales et leurs p-values associées.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    
    # 100% des données tracées (pas de sous-échantillonnage)
    df_plot = df_member.copy()
    lw = 0.7 if freq_label == 'Daily' else 1.5
    mk = None if freq_label == 'Daily' else '.'
    alpha_v = 0.85 if freq_label == 'Daily' else 1.0

    x_idx = np.arange(len(df_plot))
    x_labels = df_plot['time'].dt.strftime('%Y-%m-%d' if freq_label == 'Daily' else '%Y-%m').tolist()

    # --- SUBPLOT 1 : MSE vs BASELINE (VARIANCE) ---
    g_r2_t, p_gr2_t = stats_true['global_r2']; g_r2_r, p_gr2_r = stats_rec['global_r2']
    title_mse = (f"1. Mean Squared Error (MSE) vs Baseline Variance - Member {member}\n"
                 f"True Target: Global R²={g_r2_t:.2f} (p={p_gr2_t:.3f})   ||   "
                 f"Rec Target: Global R²={g_r2_r:.2f} (p={p_gr2_r:.3f})")
    
    ax1.plot(x_idx, df_plot['mse_true'], label="Model MSE (vs True Target)", color="firebrick", lw=lw, marker=mk, alpha=alpha_v)
    ax1.plot(x_idx, df_plot['base_var_true'], label="Baseline 0 (True Variance)", color="lightcoral", lw=1.5, linestyle="--", alpha=0.9)
    ax1.plot(x_idx, df_plot['mse_rec'], label="Model MSE (vs Reconstructed)", color="teal", lw=lw, marker=mk, alpha=alpha_v)
    ax1.plot(x_idx, df_plot['base_var_rec'], label="Baseline 0 (Rec Variance)", color="mediumturquoise", lw=1.5, linestyle="--", alpha=0.9)
    
    ax1.set_title(title_mse, fontsize=11.5, pad=8)
    ax1.set_ylabel("MSE"); ax1.grid(True, linestyle=':', alpha=0.7); ax1.legend(loc="upper right", ncol=2, fontsize=10)

    # --- SUBPLOT 2 : MAE (NORME L1) vs BASELINE ---
    g_l1_t, p_gl1_t = stats_true['global_l1']; g_l1_r, p_gl1_r = stats_rec['global_l1']
    title_mae = (f"2. L1 Norm (MAE) vs Baseline MAE - Member {member}\n"
                 f"True Target: Global SS_L1={g_l1_t:.2f} (p={p_gl1_t:.3f})   ||   "
                 f"Rec Target: Global SS_L1={g_l1_r:.2f} (p={p_gl1_r:.3f})")
    
    ax2.plot(x_idx, df_plot['mae_true'], label="Model MAE (vs True Target)", color="firebrick", lw=lw, marker=mk, alpha=alpha_v)
    ax2.plot(x_idx, df_plot['base_mae_true'], label="Baseline 0 (True MAE)", color="lightcoral", lw=1.5, linestyle="--", alpha=0.9)
    ax2.plot(x_idx, df_plot['mae_rec'], label="Model MAE (vs Reconstructed)", color="teal", lw=lw, marker=mk, alpha=alpha_v)
    ax2.plot(x_idx, df_plot['base_mae_rec'], label="Baseline 0 (Rec MAE)", color="mediumturquoise", lw=1.5, linestyle="--", alpha=0.9)
    
    ax2.set_title(title_mae, fontsize=11.5, pad=8)
    ax2.set_ylabel("MAE"); ax2.grid(True, linestyle=':', alpha=0.7); ax2.legend(loc="upper right", ncol=2, fontsize=10)

    # --- SUBPLOT 3 : CORRÉLATION SPATIALE INSTANTANÉE ---
    g_c_t, p_gc_t = stats_true['global_corr']; g_c_r, p_gc_r = stats_rec['global_corr']
    title_corr = (f"3. Instantaneous Spatial Correlation ($r_{{spatial}}$) - Member {member}\n"
                  f"True Target: Global Corr={g_c_t:.2f} (p={p_gc_t:.3f})   ||   "
                  f"Rec Target: Global Corr={g_c_r:.2f} (p={p_gc_r:.3f})")
    
    ax3.plot(x_idx, df_plot['spatial_corr_true'], label="vs True Target", color="firebrick", lw=lw, marker=mk, alpha=alpha_v)
    ax3.plot(x_idx, df_plot['spatial_corr_rec'], label="vs Reconstructed", color="teal", lw=lw, marker=mk, alpha=alpha_v)
    ax3.axhline(0, color='black', linestyle='--', lw=1.5, alpha=0.9, label="Baseline 0 Correlation")
    ax3.set_title(title_corr, fontsize=11.5, pad=8)
    ax3.set_ylabel("$r$"); ax3.grid(True, linestyle=':', alpha=0.7); ax3.legend(loc="upper right", fontsize=10)

    # Axe X : 15 étiquettes épurées pour la lisibilité
    n_ticks = min(15, len(x_idx))
    tick_indices = np.linspace(0, len(x_idx) - 1, n_ticks, dtype=int)
    ax3.set_xticks(tick_indices)
    ax3.set_xticklabels([x_labels[idx] for idx in tick_indices], rotation=45, ha="right")
    ax3.set_xlabel("Time", fontsize=14)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'Raw_Metrics_Timeseries_{freq_label}_Member_{member}.png'), dpi=200)
    plt.close(fig)


def plot_metric_with_pvalue_map(metric_map, pval_map, outdir, filename, metric_type="l2", title="", global_stat=None, mean_pixel_stat=None):
    """
    Génère une figure à 2 panneaux côte à côte.
    Les chevauchements de la colorbar sont résolus via une rotation des labels.
    """
    os.makedirs(outdir, exist_ok=True)
    extent_slp = [-100, 40, 20, 70]

    if metric_type in ["l2", "l1"]:
        norm = SymLogNorm(linthresh=0.2, linscale=1.0, vmin=-1.0, vmax=1.0, base=10)
        ticks = [-1.0, -0.2, -0.1, 0, 0.05, 0.1, 0.2, 1.0]
        cmap_name = "RdBu_r"
        cbar_label = r"$R^2$ (L2 Skill Score)" if metric_type == "l2" else r"$SS_{L1}$ (L1 Skill Score)"
    else:
        norm = Normalize(vmin=-1.0, vmax=1.0)
        ticks = [-1.0, -0.6, -0.3, 0, 0.3, 0.6, 1.0]
        cmap_name = "RdBu_r"
        cbar_label = r"Temporal Correlation ($r$)"

    pval_cmap = "YlGnBu_r"
    pval_norm = Normalize(vmin=0.0, vmax=0.30)

    if HAS_CARTOPY:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        im1 = ax1.imshow(metric_map, cmap=cmap_name, norm=norm, origin='lower', extent=extent_slp, transform=ccrs.PlateCarree())
        im2 = ax2.imshow(pval_map, cmap=pval_cmap, norm=pval_norm, origin='lower', extent=extent_slp, transform=ccrs.PlateCarree())
        ax1.set_title(f"{title} (Max: {np.nanmax(metric_map):.3f})", fontsize=12.5)
        ax1.coastlines()
        ax2.set_title(r"Bootstrap Pixel Significance ($p$-value under $H_0$)", fontsize=12.5)
        ax2.coastlines()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        im1 = ax1.imshow(metric_map, cmap=cmap_name, norm=norm, origin='lower', aspect='auto')
        im2 = ax2.imshow(pval_map, cmap=pval_cmap, norm=pval_norm, origin='lower', aspect='auto')
        ax1.set_title(title, fontsize=12.5)
        ax2.set_title(r"Bootstrap Pixel Significance ($p$-value under $H_0$)", fontsize=12.5)

    title_parts = [title]
    if global_stat is not None:
        g_val, g_pval = global_stat
        title_parts.append(f"Global: {g_val:.3f} (p={g_pval:.3f})")
    if mean_pixel_stat is not None:
        mp_val, mp_pval = mean_pixel_stat
        title_parts.append(f"Mean Pixel: {mp_val:.3f} (p={mp_pval:.3f})")
    ax1.set_title(" | ".join(title_parts), fontsize=11.5)

    cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.08)
    cbar1.set_label(cbar_label, fontsize=12)
    if ticks is not None:
        cbar1.set_ticks(ticks); cbar1.set_ticklabels([str(t) for t in ticks])

    cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.08)
    cbar2.set_label(r"$p$-value (Dark = Significant, Light = Non-Significant)", fontsize=11)
    
    # Modification cruciale ici pour éviter le chevauchement
    cbar2.set_ticks([0.0, 0.05, 0.10, 0.15, 0.20])
    labels = ['0.00', '0.05 (5% Thresh)', '0.10 (10% Thresh)', '0.15', '≥ 0.20 (NS)']
    cbar2.ax.set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{filename}.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)


def compute_map_metrics_and_bootstraps(t_arr, p_arr, spatial_weights, n_bootstraps=300):
    """
    Calcule les 3 cartes géographiques (R2, SS_L1, Corr), leurs p-values au pixel par Bootstrap,
    ET les 6 métriques de synthèse (Globales et Moyennées par pixel) avec p-values sous H0.
    """
    T, H, W = t_arr.shape
    
    # --- 1. CARTES PIXEL PAR PIXEL ---
    t_mean = t_arr.mean(axis=0); p_mean = p_arr.mean(axis=0)
    var_t = np.var(t_arr, axis=0) + 1e-8; var_p = np.var(p_arr, axis=0) + 1e-8
    cov = np.mean((t_arr - t_mean) * (p_arr - p_mean), axis=0)
    mse = np.mean((t_arr - p_arr)**2, axis=0)
    
    r2_map = 1.0 - (mse / var_t)
    corr_map = cov / np.sqrt(var_t * var_p)
    mae = np.mean(np.abs(t_arr - p_arr), axis=0)
    mae_ref = np.mean(np.abs(t_arr), axis=0) + 1e-8
    l1_map = 1.0 - (mae / mae_ref)
    
    # --- 2. LES 6 MÉTRIQUES DE SYNTHÈSE DE RÉFÉRENCE ---
    mean_pixel_r2_orig   = np.sum(r2_map * spatial_weights)
    mean_pixel_l1_orig   = np.sum(l1_map * spatial_weights)
    mean_pixel_corr_orig = np.sum(corr_map * spatial_weights)
    
    mse_global_orig = np.sum(mse * spatial_weights); var_global_orig = np.sum(var_t * spatial_weights)
    global_r2_orig  = 1.0 - (mse_global_orig / var_global_orig)
    
    mae_global_orig     = np.sum(mae * spatial_weights); mae_ref_global_orig = np.sum(mae_ref * spatial_weights)
    global_l1_orig      = 1.0 - (mae_global_orig / mae_ref_global_orig)
    
    w_3d = np.broadcast_to(spatial_weights, t_arr.shape)
    t_g_mean = np.sum(t_arr * w_3d) / np.sum(w_3d); p_g_mean = np.sum(p_arr * w_3d) / np.sum(w_3d)
    cov_g = np.sum((t_arr - t_g_mean) * (p_arr - p_g_mean) * w_3d)
    var_t_g = np.sum((t_arr - t_g_mean)**2 * w_3d) + 1e-8; var_p_g = np.sum((p_arr - p_g_mean)**2 * w_3d) + 1e-8
    global_corr_orig = cov_g / np.sqrt(var_t_g * var_p_g)
    
    # --- 3. BOOTSTRAP SOUS H0 ---
    boot_r2_count = np.zeros((H, W), dtype=int); boot_l1_count = np.zeros((H, W), dtype=int); boot_corr_count = np.zeros((H, W), dtype=int)
    cnt = {k: 0 for k in ['g_r2', 'mp_r2', 'g_l1', 'mp_l1', 'g_corr', 'mp_corr']}
    
    print(f"    Running {n_bootstraps} bootstraps for spatial & global p-values...", end="", flush=True)
    for b in range(n_bootstraps):
        idx = np.random.choice(T, size=T, replace=True)
        t_boot = t_arr[idx]
        
        t_b_mean = t_boot.mean(axis=0); var_t_b = np.var(t_boot, axis=0) + 1e-8
        cov_b = np.mean((t_boot - t_b_mean) * (p_arr - p_mean), axis=0)
        mse_b = np.mean((t_boot - p_arr)**2, axis=0)
        mae_b = np.mean(np.abs(t_boot - p_arr), axis=0); mae_ref_b = np.mean(np.abs(t_boot), axis=0) + 1e-8
        
        r2_b = 1.0 - (mse_b / var_t_b); corr_b = cov_b / np.sqrt(var_t_b * var_p); l1_b = 1.0 - (mae_b / mae_ref_b)
        
        boot_r2_count += (r2_b >= r2_map); boot_l1_count += (l1_b >= l1_map); boot_corr_count += (np.abs(corr_b) >= np.abs(corr_map))
        
        mp_r2_b   = np.sum(r2_b * spatial_weights);   mp_l1_b   = np.sum(l1_b * spatial_weights);   mp_corr_b = np.sum(corr_b * spatial_weights)
        g_r2_b    = 1.0 - (np.sum(mse_b * spatial_weights) / np.sum(var_t_b * spatial_weights))
        g_l1_b    = 1.0 - (np.sum(mae_b * spatial_weights) / np.sum(mae_ref_b * spatial_weights))
        
        t_g_mean_b = np.sum(t_boot * w_3d) / np.sum(w_3d)
        cov_g_b    = np.sum((t_boot - t_g_mean_b) * (p_arr - p_g_mean) * w_3d)
        var_t_g_b  = np.sum((t_boot - t_g_mean_b)**2 * w_3d) + 1e-8
        g_corr_b   = cov_g_b / np.sqrt(var_t_g_b * var_p_g)
        
        cnt['g_r2']   += (g_r2_b >= global_r2_orig);       cnt['mp_r2']   += (mp_r2_b >= mean_pixel_r2_orig)
        cnt['g_l1']   += (g_l1_b >= global_l1_orig);       cnt['mp_l1']   += (mp_l1_b >= mean_pixel_l1_orig)
        cnt['g_corr'] += (np.abs(g_corr_b) >= np.abs(global_corr_orig)); cnt['mp_corr'] += (np.abs(mp_corr_b) >= np.abs(mean_pixel_corr_orig))
        
    print(" Done!")
    
    pval_r2 = boot_r2_count / n_bootstraps; pval_l1 = boot_l1_count / n_bootstraps; pval_corr = boot_corr_count / n_bootstraps
    
    stats_dict = {
        'global_r2':       (global_r2_orig,       cnt['g_r2']    / n_bootstraps),
        'mean_pixel_r2':   (mean_pixel_r2_orig,   cnt['mp_r2']   / n_bootstraps),
        'global_l1':       (global_l1_orig,       cnt['g_l1']    / n_bootstraps),
        'mean_pixel_l1':   (mean_pixel_l1_orig,   cnt['mp_l1']   / n_bootstraps),
        'global_corr':     (global_corr_orig,     cnt['g_corr']  / n_bootstraps),
        'mean_pixel_corr': (mean_pixel_corr_orig, cnt['mp_corr'] / n_bootstraps)
    }
    
    return r2_map, pval_r2, l1_map, pval_l1, corr_map, pval_corr, stats_dict


## DESUET POUR L'ESPACE LATENT. Laisser quand même pour certains code de visualisations uniquement en PC1 qui les utilise (ex: pc1_composites)

def compute_two_bootstraps(true_values, pred_values, n_iterations):
    """
    Calcule de distributions bootstrap (Target permutée vs Prédiction
    pour r, R^2 et le Skill Score L1 (SS_L1). Pour r, il y a aussi  (Target permutée vs Target) qui me semblait aussi pertinent comme baseline (d'où le nom historique de la fonction).
    """
    true_values = np.asarray(true_values).flatten()
    pred_values = np.asarray(pred_values).flatten()
    
    if len(true_values) < 2:
        return np.nan, np.nan, np.nan, None, None, None, None
        
    original_corr, _ = pearsonr(true_values, pred_values)
    
    # R2 original
    mse_orig = np.mean((true_values - pred_values)**2)
    var_orig = np.var(true_values)
    original_r2 = 1 - (mse_orig / var_orig) if var_orig > 0 else np.nan
    
    # NOUVEAU : SS_L1 original
    mae_orig = np.mean(np.abs(true_values - pred_values))
    mae_ref_orig = np.mean(np.abs(true_values))
    original_ss_l1 = 1 - (mae_orig / mae_ref_orig) if mae_ref_orig > 0 else np.nan
    
    n = len(true_values)
    corr_target_pred_boot = np.zeros(n_iterations)
    corr_target_target_boot = np.zeros(n_iterations)
    r2_target_pred_boot = np.zeros(n_iterations)
    ss_l1_target_pred_boot = np.zeros(n_iterations) # NOUVEAU

    for i in range(n_iterations):
        sampled_true = np.random.choice(true_values, size=n, replace=True)
        
        # 1. Corrélations
        corr_target_pred_boot[i], _ = pearsonr(sampled_true, pred_values)
        corr_target_target_boot[i], _ = pearsonr(sampled_true, true_values)

        # 2. R2
        mse_boot = np.mean((sampled_true - pred_values)**2)
        var_boot = np.var(sampled_true)
        r2_target_pred_boot[i] = 1 - (mse_boot / var_boot) if var_boot > 0 else np.nan
        
        # NOUVEAU : 3. SS_L1
        mae_boot = np.mean(np.abs(sampled_true - pred_values))
        mae_ref_boot = np.mean(np.abs(sampled_true))
        ss_l1_target_pred_boot[i] = 1 - (mae_boot / mae_ref_boot) if mae_ref_boot > 0 else np.nan

    return original_corr, original_r2, original_ss_l1, corr_target_pred_boot, corr_target_target_boot, r2_target_pred_boot, ss_l1_target_pred_boot

def plot_two_bootstrap_histograms(corr_tp_boot, corr_tt_boot, r2_tp_boot, ss_l1_tp_boot, original_corr, original_r2, original_ss_l1, month_name, pc_idx, member, outdir, freq_label):
    """Trace les 4 histogrammes bootstrap avec les lignes de mesure réelles. Nom historique venant du fait qu'au départ on avait 2 histogrammes (corrélation permutée vs prédiction et corrélation permutée vs target)."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    
    # 1. Corrélation : Permutée vs Pred
    axes[0].hist(corr_tp_boot, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(original_corr, color='red', linestyle='dashed', linewidth=2, label=f'Mesure ({original_corr:.2f})')
    axes[0].set_title('Corr: Permutée vs Pred', fontsize=12); axes[0].set_xlabel('r'); axes[0].legend(); axes[0].grid(axis='y', alpha=0.5)
    
    # 2. Corrélation : Permutée vs Target
    axes[1].hist(corr_tt_boot, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(original_corr, color='red', linestyle='dashed', linewidth=2, label=f'Mesure ({original_corr:.2f})')
    axes[1].set_title('Corr: Permutée vs Target', fontsize=12); axes[1].set_xlabel('r'); axes[1].legend(); axes[1].grid(axis='y', alpha=0.5)

    # 3. R2 : Permutée vs Pred
    valid_r2 = r2_tp_boot[~np.isnan(r2_tp_boot)]
    axes[2].hist(valid_r2, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    axes[2].axvline(original_r2, color='red', linestyle='dashed', linewidth=2, label=f'Mesure ({original_r2:.2f})')
    axes[2].set_title('R²: Permutée vs Pred', fontsize=12); axes[2].set_xlabel('R²'); axes[2].legend(); axes[2].grid(axis='y', alpha=0.5)
    
    # NOUVEAU : 4. SS_L1 : Permutée vs Pred
    valid_l1 = ss_l1_tp_boot[~np.isnan(ss_l1_tp_boot)]
    axes[3].hist(valid_l1, bins=30, color='plum', edgecolor='black', alpha=0.7)
    axes[3].axvline(original_ss_l1, color='red', linestyle='dashed', linewidth=2, label=f'Mesure ({original_ss_l1:.2f})')
    axes[3].set_title(r'$SS_{L1}$: Permutée vs Pred', fontsize=12); axes[3].set_xlabel(r'$SS_{L1}$'); axes[3].legend(); axes[3].grid(axis='y', alpha=0.5)
    
    plt.suptitle(f'Distributions Bootstrap - Comp {pc_idx} - {month_name} - Membre {member}', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(outdir, f'Hist_Boot_Comp{pc_idx}_{month_name}_{freq_label}_Member_{member}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def bootstrap_correlation(true_values, pred_values, n_iterations): 
    true_values = np.asarray(true_values).flatten()
    pred_values = np.asarray(pred_values).flatten()
    
    if len(true_values) < 2:
        return np.nan
        
    original_corr, _ = pearsonr(true_values, pred_values) 
    correlations = np.zeros(n_iterations) 

    for i in range(n_iterations): 
        sample_indices = np.random.choice(len(true_values), size=len(true_values), replace=True) 
        sampled_true = true_values[sample_indices] 
        corr, _ = pearsonr(sampled_true, pred_values) 
        correlations[i] = corr 

    p_value = np.mean(np.abs(correlations) >= np.abs(original_corr))  
    return p_value

def bootstrap_r2(true_values, pred_values, n_iterations): 
    true_values = np.asarray(true_values).flatten()
    pred_values = np.asarray(pred_values).flatten()
    if len(true_values) < 2: return np.nan
    
    mse_orig = np.mean((true_values - pred_values)**2)
    var_orig = np.var(true_values)
    original_r2 = 1 - (mse_orig / var_orig) if var_orig > 0 else np.nan
    
    r2_scores = np.zeros(n_iterations) 
    for i in range(n_iterations): 
        samp = true_values[np.random.choice(len(true_values), size=len(true_values), replace=True)]
        mse_boot = np.mean((samp - pred_values)**2)
        var_boot = np.var(samp)
        r2_scores[i] = 1 - (mse_boot / var_boot) if var_boot > 0 else np.nan
        # r2_scores[i] = 1 - (mse_boot / var_orig) # plutôt que d'utiliser var_boot? 
    return np.mean(r2_scores >= original_r2)

def bootstrap_ss_l1(true_values, pred_values, n_iterations): 
    true_values = np.asarray(true_values).flatten()
    pred_values = np.asarray(pred_values).flatten()
    if len(true_values) < 2: return np.nan
    
    mae_orig = np.mean(np.abs(true_values - pred_values))
    mae_ref_orig = np.mean(np.abs(true_values))
    original_ss_l1 = 1 - (mae_orig / mae_ref_orig) if mae_ref_orig > 0 else np.nan
    
    l1_scores = np.zeros(n_iterations) 
    for i in range(n_iterations): 
        samp = true_values[np.random.choice(len(true_values), size=len(true_values), replace=True)]
        mae_boot = np.mean(np.abs(samp - pred_values))
        mae_ref_boot = np.mean(np.abs(samp))
        l1_scores[i] = 1 - (mae_boot / mae_ref_boot) if mae_ref_boot > 0 else np.nan
        
    return np.mean(l1_scores >= original_ss_l1)

def stats(pcs_true, pcs_pred, n_iterations):
    pval_corr = xr.apply_ufunc(
        bootstrap_correlation, pcs_true, pcs_pred,
        input_core_dims=[["time"], ["time"]], output_core_dims=[[]],
        vectorize=True, kwargs={"n_iterations": n_iterations}, output_dtypes=[float]
    )
    
    pval_r2 = xr.apply_ufunc(
        bootstrap_r2, pcs_true, pcs_pred,
        input_core_dims=[["time"], ["time"]], output_core_dims=[[]],
        vectorize=True, kwargs={"n_iterations": n_iterations}, output_dtypes=[float]
    )
    
    # NOUVEAU : P-value pour SS_L1
    pval_ss_l1 = xr.apply_ufunc(
        bootstrap_ss_l1, pcs_true, pcs_pred,
        input_core_dims=[["time"], ["time"]], output_core_dims=[[]],
        vectorize=True, kwargs={"n_iterations": n_iterations}, output_dtypes=[float]
    )
    
    corr = xr.corr(pcs_true, pcs_pred, dim='time')
    std_true = pcs_true.std(dim='time')
    rmse = np.sqrt(((pcs_true - pcs_pred)**2).mean(dim='time'))
    nrmse = rmse / std_true
    r2 = 1 - nrmse**2 
    
    # NOUVEAU : Calcul natif du SS_L1 sur Xarray
    mae = np.abs(pcs_true - pcs_pred).mean(dim='time')
    mae_ref = np.abs(pcs_true).mean(dim='time')
    ss_l1 = 1 - (mae / mae_ref)

    return corr, rmse, nrmse, r2, ss_l1, pval_corr, pval_r2, pval_ss_l1

def plot_time_series(pcs_true_dict, pcs_pred_dict, stats_dict, months_num, member, outdir, pc_idx, freq_label, quantiles_pred_dict=None):
    "Ancienne version qui faisait 1 PC à la fois et des subplots par mois"
    n_months = len(months_num)
    fig_width = 25 if freq_label == "Daily" else 10
    
    fig, axes = plt.subplots(n_months, 1, figsize=(fig_width, 5 * n_months))
    if n_months == 1: axes = [axes] 

    fig_res, axes_res = plt.subplots(n_months, 1, figsize=(fig_width, 5 * n_months))
    if n_months == 1: axes_res = [axes_res]
        
    for i, m in enumerate(months_num):
        ax = axes[i]; ax_res = axes_res[i] 
        true_m = pcs_true_dict[m]; pred_m = pcs_pred_dict[m]
        
        # NOUVEAU : Unpacking à 8 variables
        corr, rmse, nrmse, r2, ss_l1, pval_corr, pval_r2, pval_ss_l1 = stats_dict[m]
        
        p_c = float(pval_corr.values) if not np.isnan(pval_corr.values) else np.nan
        p_r = float(pval_r2.values) if not np.isnan(pval_r2.values) else np.nan
        p_l1 = float(pval_ss_l1.values) if not np.isnan(pval_ss_l1.values) else np.nan # NOUVEAU
        
        # Corrélation
        if np.isnan(p_c):
            corr_str = f"r={corr.values:.2f} (p=NaN)"
        elif corr.values < 0:
            corr_str = f"r={corr.values:.2f} (p= N/A)"
        else:
            sign_c = "**" if p_c < 0.05 else ("*" if p_c < 0.1 else "")
            corr_str = f"r={corr.values:.2f}{sign_c} (p={p_c:.3f})"

        # R²
        if np.isnan(p_r):
            r2_str = f"R²={r2.values:.3f} (p=NaN)"
        elif r2.values < 0:
            r2_str = f"R²={r2.values:.3f} (p= N/A)"
        else:
            sign_r = "**" if p_r < 0.05 else ("*" if p_r < 0.1 else "")
            r2_str = f"R²={r2.values:.3f}{sign_r} (p={p_r:.3f})"

        # Skill L1
        if np.isnan(p_l1):
            ssl1_str = f"SS_L1={ss_l1.values:.3f} (p=NaN)"
        elif ss_l1.values < 0:
            ssl1_str = f"SS_L1={ss_l1.values:.3f} (p= N/A)"
        else:
            sign_l1 = "**" if p_l1 < 0.05 else ("*" if p_l1 < 0.1 else "")
            ssl1_str = f"SS_L1={ss_l1.values:.3f}{sign_l1} (p={p_l1:.3f})"
        
        month_name = calendar.month_abbr[m] 
        
        # NOUVEAU : Titre enrichi avec SS_L1
        title_str = f"{month_name} | {r2_str} | {ssl1_str} | {corr_str}"

        std_true = np.std(true_m)
        if std_true == 0: std_true = 1 
        
        x_idx = np.arange(len(true_m.time.values))
        x_labels = pd.to_datetime(true_m.time.values).strftime('%Y-%m-%d').tolist()

        lw = 0.4 if freq_label == "Daily" else 1.5; ms = 1 if freq_label == "Daily" else 6

        if quantiles_pred_dict and m in quantiles_pred_dict and len(quantiles_pred_dict[m]) > 0:
            q_keys = sorted(list(quantiles_pred_dict[m].keys()))
            q_lower = sorted([q for q in q_keys if q < 0.5], reverse=True); q_upper = sorted([q for q in q_keys if q > 0.5])
            n_bands = min(len(q_lower), len(q_upper))
            if n_bands > 0:
                for idx in reversed(range(n_bands)):
                    ql = q_lower[idx]; qu = q_upper[idx]
                    label_str = f"Quantiles ({min(q_lower)}-{max(q_upper)})" if (idx == n_bands - 1 and i == 0) else ""
                    ax.fill_between(x_idx, quantiles_pred_dict[m][ql]/std_true, quantiles_pred_dict[m][qu]/std_true, color='tab:blue', alpha=0.15, linewidth=0.0, label=label_str)

        ax.plot(x_idx, true_m/std_true, color="black", marker='.', linewidth=lw, markersize=ms, label="True" if i==0 else "")
        label = "Predicted (Median)" if quantiles_pred_dict else "Predicted"
        ax.plot(x_idx, pred_m/std_true, color="navy", marker='.', linewidth=lw, markersize=ms, label=label if i==0 else "")
        ax.set_xlabel("Time", fontsize=14); ax.set_ylabel(f"Normalized component {pc_idx}", fontsize=14); ax.set_title(title_str, fontsize=15); ax.grid(True)
        
        n_ticks = min(12, len(x_idx))
        tick_indices = np.linspace(0, len(x_idx) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices); ax.set_xticklabels([x_labels[idx] for idx in tick_indices], rotation=45, ha="right")

        # RÉSIDUS
        residuals = (pred_m - true_m) / std_true
        ax_res.plot(x_idx, residuals, color="firebrick", marker='.', linewidth=lw, markersize=ms, label="Résidus (Pred - True)" if i==0 else "")
        ax_res.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax_res.set_xlabel("Time", fontsize=14); ax_res.set_ylabel(f"Résidus (component {pc_idx}) (normalisé)", fontsize=14); ax_res.set_title(f"Résidus : {title_str}", fontsize=15); ax_res.grid(True)
        ax_res.set_xticks(tick_indices); ax_res.set_xticklabels([x_labels[idx] for idx in tick_indices], rotation=45, ha="right")

    top_margin = 0.75 if n_months == 1 else 0.85
    legend_y = 1.00 if n_months == 1 else 0.95

    fig.subplots_adjust(hspace=0.8, top=top_margin); fig.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=3, fontsize=14)
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'Perf_comp{pc_idx}_{freq_label}_Member_{member}.png'), dpi=300, bbox_inches='tight'); plt.close(fig)

    fig_res.subplots_adjust(hspace=0.8, top=top_margin); fig_res.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=1, fontsize=14)
    fig_res.savefig(os.path.join(outdir, f'Residuals_comp{pc_idx}_{freq_label}_Member_{member}.png'), dpi=300, bbox_inches='tight'); plt.close(fig_res)

