import argparse
import calendar
from datetime import timedelta
import os
import time
import cartopy.crs as ccrs
import joblib
import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np
import pandas as pd
import random
import xarray as xr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, LinearRegression
import shutil
import gc

# ============================================================
# 1. FONCTIONS DE PRÉPARATION DE DONNÉES
# ============================================================

def compute_mca_stds(
    members, path_SST, path_SLP, selected_months, sst_lags,
    duree_lissage=10, roll_sst=False, monthly_reduction=False, lat_weight=False,
):
    print("Calcul dynamique de sst_std et slp_std rigoureux (Pass 1/2 sur Train Set)...")
    total_sum_sq_sst, total_sum_sq_slp = 0.0, 0.0
    total_weights_sst, total_weights_slp = 0.0, 0.0
    map_weight_sum_sst, map_weight_sum_slp = None, None

    for member in members:
        X_sst, Y_slp, _, _, _, w_sst, w_slp = load_member_mca_data(
            member, path_SST, path_SLP, selected_months, sst_lags=sst_lags,
            duree_lissage=duree_lissage, sst_std=1.0, slp_std=1.0,
            roll_sst=roll_sst, monthly_reduction=monthly_reduction, lat_weight=lat_weight,
        )
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
    print(f"--> sst_std calculé (Train) : {sst_std_rigoureux:.4f}")
    print(f"--> slp_std calculé (Train) : {slp_std_rigoureux:.4f}")
    return sst_std_rigoureux, slp_std_rigoureux

def load_member_mca_data(
    member, path_SST, path_SLP, selected_months, sst_lags=[35],
    duree_lissage=10, sst_std=0.707, slp_std=596.0, roll_sst=False,
    monthly_reduction=False, lat_weight=False,
):
    if not monthly_reduction:
        file_slp = os.path.join(path_SLP, f"PSL_anom_LE2-{member}_{duree_lissage}d.nc" if duree_lissage != 0 else f"PSL_anom_LE2-{member}.nc")
        file_sst = os.path.join(path_SST, f"SST_anom_LE2-{member}_T_regrid.nc")
    else:
        file_slp = os.path.join(path_SLP, f"PSL_anom_LE2-{member}_1mo.nc")
        file_sst = os.path.join(path_SST, f"SST_anom_LE2-{member}_T_regrid_1mo.nc")

    with xr.open_dataset(file_slp) as ds_slp_raw, xr.open_dataset(file_sst) as ds_sst_raw:
        ds_slp = ds_slp_raw["PSL"].load()
        ds_sst = ds_sst_raw["SST"].load()

    if roll_sst:
        ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby("lon")

    slp_winter = ds_slp.sel(time=slice(None, "2014-12-31"))
    slp_winter = slp_winter.sel(time=slp_winter["time"].dt.month.isin(selected_months))
    years = slp_winter["time"].dt.year
    valid_mask = (years > years.min()) & (years < years.max())
    target_dates = slp_winter.sel(time=valid_mask).time.values

    ds_sst_cropped = ds_sst.sel(lat=slice(-15, 70))
    shape_sst = ds_sst_cropped.shape[1:]
    shape_slp = ds_slp.shape[1:]
    wgts_sst_flat, wgts_slp_flat, wgts_sst_3d, wgts_slp_2d = None, None, None, None

    if lat_weight:
        lats_slp = ds_slp["lat"].values
        coslat_slp = np.cos(np.deg2rad(lats_slp)).clip(0.0, 1.0)
        wgts_slp_2d = np.sqrt(coslat_slp).reshape(shape_slp[0], 1)
        wgts_slp_flat = np.broadcast_to(wgts_slp_2d, shape_slp).flatten()

        lats_sst = ds_sst_cropped["lat"].values
        coslat_sst = np.cos(np.deg2rad(lats_sst)).clip(0.0, 1.0)
        wgts_sst_1d = np.sqrt(coslat_sst).reshape(shape_sst[0], 1)
        wgts_sst_3d = wgts_sst_1d.reshape(1, shape_sst[0], 1)
        wgts_sst_flat = np.broadcast_to(wgts_sst_3d, (len(sst_lags), shape_sst[0], shape_sst[1])).flatten()

    X_sst_list, Y_slp_list, valid_target_dates = [], [], []
    for t_target in target_dates:
        t_obj = pd.Timestamp(t_target) if isinstance(t_target, np.datetime64) else t_target
        if not monthly_reduction:
            dates_sst = [t_obj - timedelta(days=d) for d in sst_lags]
        else:
            dates_sst = []
            for m in sst_lags:
                y_shift = (t_obj.month - m - 1) // 12
                new_month = (t_obj.month - m - 1) % 12 + 1
                dates_sst.append(t_obj.replace(year=t_obj.year + y_shift, month=new_month))
        try:
            slp_t = ds_slp.sel(time=t_obj)
            sst_lags_ds = ds_sst.sel(time=dates_sst, lat=slice(-15, 70))
            slp_np = np.nan_to_num(slp_t.values, nan=0.0) / slp_std
            sst_np = np.nan_to_num(sst_lags_ds.values, nan=0.0) / sst_std
            if lat_weight:
                slp_np *= wgts_slp_2d
                sst_np *= wgts_sst_3d
            Y_slp_list.append(slp_np.flatten())
            X_sst_list.append(sst_np.flatten())
            valid_target_dates.append(t_target)
        except KeyError:
            continue

    return np.array(X_sst_list), np.array(Y_slp_list), valid_target_dates, shape_sst, shape_slp, wgts_sst_flat, wgts_slp_flat

# ============================================================
# 2. FONCTIONS DE VISUALISATION
# ============================================================

def plot_scf(s, outdir, n_modes_plot=30):
    scf = (s**2) / np.sum(s**2)
    scf_cum = np.cumsum(scf)
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, n_modes_plot + 1), scf_cum[:n_modes_plot], marker="o", color="#d62728", linewidth=1.5, label="Cumulative SCF")
    plt.bar(range(1, n_modes_plot + 1), scf[:n_modes_plot], alpha=0.6, color="#1f77b4", label="Individual Mode SCF")
    plt.xlabel("Mode Number", fontweight="bold")
    plt.ylabel("Explained Squared Covariance (Fraction)", fontweight="bold")
    plt.title("Maximum Covariance Analysis — Singular Value Spectrum (Train Set)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mca_scf_variance.png"), dpi=200, bbox_inches="tight")
    plt.close()

def plot_mca_modes(U, V, s, shape_sst, shape_slp, sst_lags, outdir, sigma_A1, sigma_B1, sst_std=1.0, slp_std=1.0, n_modes=1, roll_sst=False, wgts_sst_flat=None, wgts_slp_flat=None):
    sorted_indices = np.argsort(sst_lags)[::-1]
    sorted_lags = [sst_lags[idx] for idx in sorted_indices]
    U_phys, V_phys = np.copy(U), np.copy(V)
    if wgts_sst_flat is not None: U_phys /= np.maximum(wgts_sst_flat, 1e-5)[:, None]
    if wgts_slp_flat is not None: V_phys /= np.maximum(wgts_slp_flat, 1e-5)[None, :]
    extent_slp = [-100, 40, 20, 70]
    extent_sst = [-180, 180, -15, 70] if roll_sst else [0, 359.9, -15, 70]
    scf = (s**2) / np.sum(s**2)
    h_sst, w_sst, h_slp, w_slp, num_lags = *shape_sst, *shape_slp, len(sorted_lags)
    col_widths = [1.5] * num_lags + [1.0]
    fig, axes = plt.subplots(n_modes, num_lags + 1, figsize=(5.5 * (num_lags + 1), 3.5 * n_modes), subplot_kw={"projection": ccrs.PlateCarree()}, gridspec_kw={"width_ratios": col_widths}, squeeze=False)
    fig.suptitle(f"Maximum Covariance Analysis — Leading Coupled Mode", fontsize=16, fontweight="bold", y=0.98)
    unit_label = "months" if "monthly" in outdir else "days"

    mode_sst_full = U_phys[:, 0].reshape((num_lags, h_sst, w_sst))
    mode_slp = V_phys[0, :].reshape((h_slp, w_slp))
    vlim_sst, vlim_slp = np.max(np.abs(mode_sst_full)), np.max(np.abs(mode_slp))

    for col_idx, orig_idx in enumerate(sorted_indices):
        ax_sst = axes[0, col_idx]
        im_sst = ax_sst.imshow(mode_sst_full[orig_idx], cmap="RdBu_r", origin="lower", vmin=-vlim_sst, vmax=vlim_sst, transform=ccrs.PlateCarree(), extent=extent_sst)
        ax_sst.set_extent(extent_sst, crs=ccrs.PlateCarree())
        ax_sst.coastlines(color="black", linewidth=0.8, alpha=0.7)
        ax_sst.set_title(f"Mode 1 SST (Lag -{sorted_lags[col_idx]} [{unit_label}])", fontweight="normal" if col_idx == 0 else "normal", pad=8)
        if col_idx == num_lags - 1:
            fig.colorbar(im_sst, ax=ax_sst, fraction=0.035, shrink=0.65, pad=0.04).set_label("SST Mode Magnitude (unitless)")
    ax_slp = axes[0, num_lags]
    im_slp = ax_slp.imshow(mode_slp, cmap="RdBu_r", origin="lower", vmin=-vlim_slp, vmax=vlim_slp, transform=ccrs.PlateCarree(), extent=extent_slp)
    ax_slp.set_extent(extent_slp, crs=ccrs.PlateCarree())
    ax_slp.coastlines(color="black", linewidth=0.8, alpha=0.7)
    ax_slp.set_title(f"Mode 1 SLP (Lag 0 — SCF: {scf[0]:.4f})", fontweight="normal", pad=8)
    fig.colorbar(im_slp, ax=ax_slp, fraction=0.035, shrink=0.85, pad=0.04).set_label("SLP Mode Magnitude (unitless)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(outdir, "mca_coupled_modes.png"), dpi=200, bbox_inches="tight")
    plt.close()

# ============================================================
# 3. DIAGNOSTIC TRAIN vs VALIDATION & MCA vs PCA
# ============================================================

def evaluate_train_val_modes(
    train_members, val_members, path_SST, path_SLP, winter_months, sst_lags,
    duree_lissage, roll_sst, monthly_reduction, lat_weight, sst_std, slp_std,
    U, V, shape_sst, shape_slp, outdir, global_wgts_sst, global_wgts_slp, max_modes_in=200, max_modes_out=60, model_path=None
):
    print(f"\n{'='*75}\n🏆 ÉVALUATION RIGOUREUSE TRAIN vs VALIDATION & MCA vs PCA\n{'='*75}")
    max_modes = max_modes_in
    def extract_X_and_Y(members_list, name):
        print(f"--> Extraction des matrices brutes (X) et (Y) pour le set [{name}]...")
        X_list, Y_list = [], []
        for mem in members_list:
            X_m, Y_m, _, _, _, _, _ = load_member_mca_data(
                mem, path_SST, path_SLP, winter_months, sst_lags=sst_lags,
                duree_lissage=duree_lissage, roll_sst=roll_sst, monthly_reduction=monthly_reduction,
                lat_weight=lat_weight, sst_std=sst_std, slp_std=slp_std,
            )
            X_list.append(X_m)
            Y_list.append(Y_m)
        return np.concatenate(X_list), np.concatenate(Y_list)

    # --- 1. Chargement Train complet & PCA ultra-rapide ---
    X_train, Y_train = extract_X_and_Y(train_members, "TRAIN")
    X_train_mean = np.mean(X_train, axis=0)
    Y_train_mean = np.mean(Y_train, axis=0)
    X_train_centered = X_train - X_train_mean
    Y_train_centered = Y_train - Y_train_mean

    var_total_train_X = np.sum(np.var(X_train_centered, axis=0))
    var_total_train_Y = np.sum(np.var(Y_train_centered, axis=0))

    A_mca_train = X_train_centered @ U[:, :max_modes]
    
    print("\n--- Calcul de la PCA sur l'océan (SST) via TruncatedSVD (Rapide) ---")
    pca_sst = TruncatedSVD(n_components=max_modes, random_state=42)
    A_pca_train = pca_sst.fit_transform(X_train_centered)
    PCA_U1 = pca_sst.components_[0, :]

    # Sauvegarde du Modèle PCA
    if model_path is not None and os.path.exists(model_path):
        model_dict = joblib.load(model_path)
        model_dict["pca_sst"] = pca_sst
        model_dict["X_train_mean"] = X_train_mean
        model_dict["Y_train_mean"] = Y_train_mean
        joblib.dump(model_dict, model_path)
        print(f"💾 Modèle PCA et moyennes de centrage ajoutés avec succès au fichier joblib !")

    X_val, Y_val = extract_X_and_Y(val_members, "VALIDATION")
    Y_val_centered = Y_val - Y_train_mean
    X_val_centered = X_val - X_train_mean
    
    A_mca_val = X_val_centered @ U[:, :max_modes]
    A_pca_val = pca_sst.transform(X_val_centered)
    
    var_total_val_Y = np.sum(Y_val_centered**2)
    
    extent_slp = [-100, 40, 20, 70]
    extent_sst = [-180, 180, -15, 70] if roll_sst else [0, 359.9, -15, 70]

    # ==========================================================
    # 3.1 COMPARAISON SPATIALE : PCA vs MCA (SST - Tous les Lags)
    # ==========================================================
    print("\n--- Comparaison Spatiale SST : MCA vs PCA ---")
    MCA_U1 = U[:, 0]
    
    spat_corr_sst = np.corrcoef(MCA_U1, PCA_U1)[0, 1]
    if spat_corr_sst < 0:
        PCA_U1 = -PCA_U1
        A_pca_train[:, 0] = -A_pca_train[:, 0]
        A_pca_val[:, 0] = -A_pca_val[:, 0]
        spat_corr_sst = -spat_corr_sst
    print(f"-> Corrélation spatiale SST (MCA Mode 1 vs PCA Mode 1) : {spat_corr_sst:.4f}")

    phys_MCA_U1 = np.copy(MCA_U1)
    phys_PCA_U1 = np.copy(PCA_U1)
    if global_wgts_sst is not None:
        phys_MCA_U1 /= np.maximum(global_wgts_sst, 1e-5)
        phys_PCA_U1 /= np.maximum(global_wgts_sst, 1e-5)

    map_MCA_SST = phys_MCA_U1 
    map_PCA_SST = phys_PCA_U1 
    map_diff_SST = map_MCA_SST - map_PCA_SST

    num_lags = len(sst_lags)
    fig_sst, axes_sst = plt.subplots(2, num_lags, figsize=(4.5 * num_lags, 10), subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    fig_sst.suptitle(f"Mode 1 SST Comparison : PCA vs MCA [Spatial Corr = {spat_corr_sst:.3f}]", fontsize=15, fontweight="bold")
    vmax_sst = max(np.max(np.abs(map_PCA_SST)), np.max(np.abs(map_MCA_SST)))
    
    map_PCA_3D = map_PCA_SST.reshape((num_lags, shape_sst[0], shape_sst[1]))
    map_MCA_3D = map_MCA_SST.reshape((num_lags, shape_sst[0], shape_sst[1]))
    map_diff_3D = map_diff_SST.reshape((num_lags, shape_sst[0], shape_sst[1]))
    
    titles_row = ["PCA Mode 1", "MCA Mode 1"]
    sorted_indices = np.argsort(sst_lags)[::-1]
    
    for row_idx, (maps_3d, row_title) in enumerate(zip([map_PCA_3D, map_MCA_3D], titles_row)):
        for col_idx, orig_idx in enumerate(sorted_indices):
            ax = axes_sst[row_idx, col_idx]
            im = ax.imshow(maps_3d[orig_idx], cmap="RdBu_r", origin="lower", extent=extent_sst, transform=ccrs.PlateCarree(), vmin=-vmax_sst, vmax=vmax_sst)
            ax.set_extent(extent_sst, crs=ccrs.PlateCarree())
            ax.coastlines(color="black", linewidth=0.8)
            if row_idx == 0:
                ax.set_title(f"Lag -{sst_lags[orig_idx]}", fontweight="bold")
            if col_idx == 0:
                ax.text(-0.05, 0.5, row_title, va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=ax.transAxes, fontweight='bold', fontsize=12)

    cbar = fig_sst.colorbar(im, ax=axes_sst.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label("SST Mode Magnitude (unitless)", fontweight="bold")
    plt.savefig(os.path.join(outdir, "mca_vs_pca_maps_SST.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # 3.2 COMPARAISON SPATIALE : PCA vs MCA (SLP - Vecteurs purs)
    # ==========================================================
    print("\n--- Comparaison Spatiale SLP (Vecteurs de Projection) : MCA vs PCA ---")
    U_pca_slp, S_pca_slp, Vt_pca_slp = svd(Y_train_centered, full_matrices=False)
    PCA_V1 = Vt_pca_slp[0, :]
    MCA_V1 = V[:, 0]
    
    if np.corrcoef(MCA_V1, PCA_V1)[0, 1] < 0:
        PCA_V1 = -PCA_V1
    spat_corr_slp = np.corrcoef(MCA_V1, PCA_V1)[0, 1]
    print(f"-> Corrélation spatiale SLP (MCA Mode 1 vs PCA Mode 1) : {spat_corr_slp:.4f}")

    phys_PCA_V1 = np.copy(PCA_V1)
    phys_MCA_V1 = np.copy(MCA_V1)
    if global_wgts_slp is not None:
        phys_PCA_V1 /= np.maximum(global_wgts_slp, 1e-5)
        phys_MCA_V1 /= np.maximum(global_wgts_slp, 1e-5)

    map_PCA_SLP_V = phys_PCA_V1 
    map_MCA_SLP_V = phys_MCA_V1
    map_diff_SLP_V = map_MCA_SLP_V - map_PCA_SLP_V

    fig_slp, axes_slp = plt.subplots(1, 2, figsize=(16, 4.5), subplot_kw={"projection": ccrs.PlateCarree()})
    fig_slp.suptitle(f"Mode 1 {month_label} SLP Comparison : PCA vs MCA [Spatial Corr = {spat_corr_slp:.3f}]", fontsize=15, fontweight="bold")
    vmax_slp = max(np.max(np.abs(map_PCA_SLP_V)), np.max(np.abs(map_MCA_SLP_V)))
    titles_slp = ["PCA Mode 1", "MCA Mode 1"]
    maps_slp = [map_PCA_SLP_V.reshape(shape_slp), map_MCA_SLP_V.reshape(shape_slp)]
    
    for ax, title, data in zip(axes_slp, titles_slp, maps_slp):
        im = ax.imshow(data, cmap="RdBu_r", origin="lower", extent=extent_slp, transform=ccrs.PlateCarree(), vmin=-vmax_slp, vmax=vmax_slp)
        ax.set_extent(extent_slp, crs=ccrs.PlateCarree())
        ax.coastlines(color="black", linewidth=0.8)
        ax.set_title(title, fontweight="bold", fontsize=11)
    
    cbar = fig_slp.colorbar(im, ax=axes_slp.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label("SLP Mode Magnitude (unitless)", fontweight="bold")
    plt.savefig(os.path.join(outdir, "mca_vs_pca_maps_SLP.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # 3.3 PROJECTIONS MULTI-MODES & PRÉDICTIONS (BASELINE)
    # ==========================================================
    B_mca_train = Y_train_centered @ V[:, :max_modes]
    B_mca_val = Y_val_centered @ V[:, :max_modes]
    
    PC_pca_train = Y_train_centered @ Vt_pca_slp.T[:, :max_modes]
    PC_pca_val = Y_val_centered @ Vt_pca_slp.T[:, :max_modes]

    vf_val_mca_list, vf_val_pca_list = [], []
    hvf_sst_mca_list, hvf_sst_pca_list = [], []
    hvf_slp_mca_list, hvf_slp_pca_list = [], []
    
    r2_predMCA_targMCA, r2_predMCA_targPCA = [], []
    r2_predPCA_targMCA, r2_predPCA_targPCA = [], []
    
    print(f"\nModes | HVF SST(MCA)| HVF SST(PCA)| HVF SLP(MCA)| HVF SLP(PCA)| Q2 Val(MCA)| Q2 Val(PCA)| R2 M->M | R2 M->P | R2 P->M | R2 P->P")
    print("-" * 125)

    anim_dir = os.path.join(outdir, "animation_R2_frames")
    os.makedirs(anim_dir, exist_ok=True)
    var_local_real_va = np.var(Y_val_centered, axis=0)

    for k in range(1, max_modes + 1):
        A_mca_tr_k, A_mca_va_k = A_mca_train[:, :k], A_mca_val[:, :k]
        A_pca_tr_k, A_pca_va_k = A_pca_train[:, :k], A_pca_val[:, :k]

        hvf_sst_mca = np.sum(np.var(A_mca_tr_k, axis=0)) / var_total_train_X
        hvf_sst_pca = np.sum(np.var(A_pca_tr_k, axis=0)) / var_total_train_X
        hvf_sst_mca_list.append(hvf_sst_mca)
        hvf_sst_pca_list.append(hvf_sst_pca)
        
        hvf_slp_mca = np.sum(np.var(B_mca_train[:, :k], axis=0)) / var_total_train_Y
        hvf_slp_pca = np.sum(np.var(PC_pca_train[:, :k], axis=0)) / var_total_train_Y
        hvf_slp_mca_list.append(hvf_slp_mca)
        hvf_slp_pca_list.append(hvf_slp_pca)

        W_mca, _, _, _ = np.linalg.lstsq(A_mca_tr_k, Y_train_centered, rcond=None)
        vf_va_mca = 1.0 - (np.sum((Y_val_centered - (A_mca_va_k @ W_mca))**2) / var_total_val_Y)
        vf_val_mca_list.append(vf_va_mca)

        W_pca, _, _, _ = np.linalg.lstsq(A_pca_tr_k, Y_train_centered, rcond=None)
        vf_va_pca = 1.0 - (np.sum((Y_val_centered - (A_pca_va_k @ W_pca))**2) / var_total_val_Y)
        vf_val_pca_list.append(vf_va_pca)

        w1, _, _, _ = np.linalg.lstsq(A_mca_tr_k, B_mca_train[:, 0], rcond=None)
        r2_1 = 1.0 - np.var(B_mca_val[:, 0] - (A_mca_va_k @ w1)) / np.var(B_mca_val[:, 0])
        r2_predMCA_targMCA.append(r2_1)

        w2, _, _, _ = np.linalg.lstsq(A_mca_tr_k, PC_pca_train[:, 0], rcond=None)
        r2_2 = 1.0 - np.var(PC_pca_val[:, 0] - (A_mca_va_k @ w2)) / np.var(PC_pca_val[:, 0])
        r2_predMCA_targPCA.append(r2_2)
        
        w3, _, _, _ = np.linalg.lstsq(A_pca_tr_k, B_mca_train[:, 0], rcond=None)
        r2_3 = 1.0 - np.var(B_mca_val[:, 0] - (A_pca_va_k @ w3)) / np.var(B_mca_val[:, 0])
        r2_predPCA_targMCA.append(r2_3)
        
        w4, _, _, _ = np.linalg.lstsq(A_pca_tr_k, PC_pca_train[:, 0], rcond=None)
        r2_4 = 1.0 - np.var(PC_pca_val[:, 0] - (A_pca_va_k @ w4)) / np.var(PC_pca_val[:, 0])
        r2_predPCA_targPCA.append(r2_4)

        print(f" #{k:02d}   |   {hvf_sst_mca:5.4f}    |   {hvf_sst_pca:5.4f}    |   {hvf_slp_mca:5.4f}    |   {hvf_slp_pca:5.4f}    |   {vf_va_mca:5.4f}   |   {vf_va_pca:5.4f}   | {r2_1:6.4f} | {r2_2:6.4f} | {r2_3:6.4f} | {r2_4:6.4f}")

        var_local_pred_va = np.var(A_mca_va_k @ W_mca, axis=0)
        r2_local_val = np.divide(var_local_pred_va, np.maximum(var_local_real_va, 1e-10), out=np.zeros_like(var_local_pred_va), where=var_local_real_va > 0)
        
        fig_map, ax_map = plt.subplots(figsize=(7, 4.5), subplot_kw={"projection": ccrs.PlateCarree()})
        im = ax_map.imshow(r2_local_val.reshape(shape_slp), cmap="YlOrRd", origin="lower", extent=extent_slp, transform=ccrs.PlateCarree(), vmin=0, vmax=0.25) 
        ax_map.set_extent(extent_slp, crs=ccrs.PlateCarree())
        ax_map.coastlines(color="black", linewidth=0.8)
        ax_map.set_title(f"Pixel Val $R^2$ — {k:02d} Input MCA Modes ({month_label} SLP)", fontweight="bold", fontsize=11)
        fig_map.colorbar(im, ax=ax_map, fraction=0.035, pad=0.04)
        plt.savefig(os.path.join(anim_dir, f"frame_R2_K{k:02d}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        var_local_pred_va_pca = np.var(A_pca_va_k @ W_pca, axis=0)
        r2_local_val_pca = np.divide(var_local_pred_va_pca, np.maximum(var_local_real_va, 1e-10), out=np.zeros_like(var_local_pred_va_pca), where=var_local_real_va > 0)
        
        fig_map_pca, ax_map_pca = plt.subplots(figsize=(7, 4.5), subplot_kw={"projection": ccrs.PlateCarree()})
        im_pca = ax_map_pca.imshow(r2_local_val_pca.reshape(shape_slp), cmap="YlOrRd", origin="lower", extent=extent_slp, transform=ccrs.PlateCarree(), vmin=0, vmax=0.25) 
        ax_map_pca.set_extent(extent_slp, crs=ccrs.PlateCarree())
        ax_map_pca.coastlines(color="black", linewidth=0.8)
        ax_map_pca.set_title(f"Pixel Val $R^2$ — {k:02d} Input PCA Modes ({month_label} SLP)", fontweight="bold", fontsize=11)
        fig_map_pca.colorbar(im_pca, ax=ax_map_pca, fraction=0.035, pad=0.04)
        plt.savefig(os.path.join(anim_dir, f"frame_R2_PCA_K{k:02d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_map_pca)

    # ==========================================================
    # 3.4 PLOTS GLOBAUX
    # ==========================================================
    modes_x = range(1, max_modes + 1)
    best_mode_idx = np.argmax(vf_val_mca_list)
    best_k = best_mode_idx + 1
    
    step = max(1, max_modes // 10)
    base_ticks = [1] + list(range(step, max_modes + 1, step))
    
    # Seuil de tolérance (ex: on efface les ticks à moins de 4% de distance du max)
    threshold = max(1, max_modes * 0.04) 
    ticks = [t for t in base_ticks if abs(t - best_k) > threshold]
    
    # On ajoute le meilleur K et on trie
    ticks.append(best_k)
    ticks = sorted(list(set(ticks)))
    
    shutil.copy(os.path.join(anim_dir, f"frame_R2_K{best_mode_idx+1:02d}.png"), os.path.join(outdir, f"mca_val_local_R2_optK{best_mode_idx+1:02d}.png"))

    plt.figure(figsize=(8, 5))
    plt.plot(modes_x, vf_val_mca_list, marker="o", color="#d62728", linewidth=1.5, markersize=2, label="MCA Modes")
    plt.plot(modes_x, vf_val_pca_list, marker="s", color="#7f7f7f", linewidth=1.5, markersize=2, linestyle="--", label="PCA Modes")
    plt.title(f"{month_label} SLP Spatial Skill Score (MCA input vs PCA input)", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Input Modes (SST)", fontweight="bold")
    plt.ylabel("$R^2$", fontweight="bold")
    # plt.xticks(ticks)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axvline(best_mode_idx + 1, color="black", linestyle=":", label=f"Optimal MCA K={best_mode_idx+1}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mca_vs_pca_target_Q2.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(modes_x, hvf_sst_mca_list, marker="o", color="#1f77b4", linewidth=1.5, markersize=2, label="MCA Modes")
    plt.plot(modes_x, hvf_sst_pca_list, marker="s", color="#ff7f0e", linewidth=1.5, markersize=2, linestyle="--", label="PCA Modes")
    plt.title("Homogeneous Variance Fraction reconstructed by MCA/PCA, SST", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Ocean Modes Included", fontweight="bold")
    plt.ylabel("Explained SST Variance Fraction", fontweight="bold")
    # plt.xticks(ticks)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mca_vs_pca_hvf_sst.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(modes_x, hvf_slp_mca_list, marker="o", color="#2ca02c", linewidth=1.5, markersize=2, label="MCA Modes")
    plt.plot(modes_x, hvf_slp_pca_list, marker="s", color="#9467bd", linewidth=1.5, markersize=2, linestyle="--", label="PCA Modes")
    plt.title(f"Homogeneous Variance Fraction reconstructed by MCA/PCA ({month_label} SLP)", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Atmosphere Modes Included", fontweight="bold")
    plt.ylabel("Explained SLP Variance Fraction", fontweight="bold")
    # plt.xticks(ticks)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mca_vs_pca_hvf_slp.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.plot(modes_x, r2_predMCA_targMCA, marker="o", color="#1f77b4", linewidth=1.5, markersize = 2,label="Pred: SST MCA $\\rightarrow$ Targ: SLP MCA Mode 1")
    plt.plot(modes_x, r2_predMCA_targPCA, marker="s", color="#9467bd", linewidth=1.5, markersize = 2, label="Pred: SST MCA $\\rightarrow$ Targ: SLP PCA Mode 1")
    plt.plot(modes_x, r2_predPCA_targMCA, marker="^", color="#2ca02c", linewidth=1.5, markersize = 2, linestyle="--", label="Pred: SST PCA $\\rightarrow$ Targ: SLP MCA Mode 1")
    plt.plot(modes_x, r2_predPCA_targPCA, marker="x", color="#d62728", linewidth=1.5, markersize = 2, linestyle="--", label="Pred: SST PCA $\\rightarrow$ Targ: SLP PCA Mode 1")
    plt.title(f"First SLP mode (MCA/PCA) Val Skill Score ({month_label})", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Input Modes (SST)", fontweight="bold")
    plt.ylabel("$R^2$", fontweight="bold")
    #plt.xticks(ticks)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mca_target_R2_skill_4curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # 3.5 EXHAUSTIVE LINEAR BENCHMARK (Les 4 architectures viables)
    # ==========================================================
    print(f"\n{'='*75}\n🚀 3.5 GRID SEARCH (ÉVALUATION 2D DES ESPACES LATENTS)\n{'='*75}")
    
    t0_grid = time.time()
    
    # 1. Préparation des espaces latents orthonormés cibles (SLP)
    var_trunc_mca = np.zeros(max_modes_out + 1)
    var_trunc_pca = np.zeros(max_modes_out + 1)
    for k in range(1, max_modes_out + 1):
        var_trunc_mca[k] = var_total_val_Y - np.sum((Y_val_centered @ V[:, :k])**2)
        var_trunc_pca[k] = var_total_val_Y - np.sum((Y_val_centered @ Vt_pca_slp.T[:, :k])**2)

    # 2. Matrices 2D asymétriques [K_out, K_in]
    q2_mca2mca = np.zeros((max_modes_out, max_modes_in))
    q2_mca2pca = np.zeros((max_modes_out, max_modes_in))
    q2_pca2mca = np.zeros((max_modes_out, max_modes_in))
    q2_pca2pca = np.zeros((max_modes_out, max_modes_in))

    # --- L'ASTUCE X100 : On sort la résolution OLS de la boucle des K_out ---
    print("Calcul de la grille 2D en cours (Vectorisation massive)...", end=" ", flush=True)
    for i, kin in enumerate(range(1, max_modes_in + 1)):
        
        # Un seul entraînement pour prédire TOUS les max_modes_out d'un coup
        W_mca2mca, _, _, _ = np.linalg.lstsq(A_mca_train[:, :kin], B_mca_train[:, :max_modes_out], rcond=None)
        W_mca2pca, _, _, _ = np.linalg.lstsq(A_mca_train[:, :kin], PC_pca_train[:, :max_modes_out], rcond=None)
        
        W_pca2mca, _, _, _ = np.linalg.lstsq(A_pca_train[:, :kin], B_mca_train[:, :max_modes_out], rcond=None)
        W_pca2pca, _, _, _ = np.linalg.lstsq(A_pca_train[:, :kin], PC_pca_train[:, :max_modes_out], rcond=None)
        
        # Prédiction unique sur tout le set de validation
        Pred_mca2mca = A_mca_val[:, :kin] @ W_mca2mca
        Pred_mca2pca = A_mca_val[:, :kin] @ W_mca2pca
        Pred_pca2mca = A_pca_val[:, :kin] @ W_pca2mca
        Pred_pca2pca = A_pca_val[:, :kin] @ W_pca2pca

        # Remplissage instantané de la colonne i pour tous les K_out
        for j, kout in enumerate(range(1, max_modes_out + 1)):
            
            mse_latent = np.sum((B_mca_val[:, :kout] - Pred_mca2mca[:, :kout])**2)
            q2_mca2mca[j, i] = 1.0 - (mse_latent + var_trunc_mca[kout]) / var_total_val_Y

            mse_latent = np.sum((PC_pca_val[:, :kout] - Pred_mca2pca[:, :kout])**2)
            q2_mca2pca[j, i] = 1.0 - (mse_latent + var_trunc_pca[kout]) / var_total_val_Y

            mse_latent = np.sum((B_mca_val[:, :kout] - Pred_pca2mca[:, :kout])**2)
            q2_pca2mca[j, i] = 1.0 - (mse_latent + var_trunc_mca[kout]) / var_total_val_Y

            mse_latent = np.sum((PC_pca_val[:, :kout] - Pred_pca2pca[:, :kout])**2)
            q2_pca2pca[j, i] = 1.0 - (mse_latent + var_trunc_pca[kout]) / var_total_val_Y

    print(f"✅ Terminé en {time.time() - t0_grid:.1f} s")

    # 3. Tracé des Heatmaps (4 graphiques 2D)
    # (Le reste du code pour fig_grid, etc... reste strictement identique)

    # 3. Tracé des Heatmaps (4 graphiques 2D)
    fig_grid, axes_grid = plt.subplots(2, 2, figsize=(14, 12))
    fig_grid.suptitle(f"Embedding Linear Models Val Spatial Skill Score Comparison ({month_label} SLP)", fontsize=16, fontweight="bold")
    
    grids = [q2_mca2mca, q2_mca2pca, q2_pca2mca, q2_pca2pca]
    titles = ["Pred: MCA $\\rightarrow$ Targ: MCA", "Pred: MCA $\\rightarrow$ Targ: PCA", 
              "Pred: PCA $\\rightarrow$ Targ: MCA", "Pred: PCA $\\rightarrow$ Targ: PCA"]
    
    vmax_grid = max(np.max(g) for g in grids)
    vmin_grid = min(0.0, min(np.min(g) for g in grids))
    
    for ax, grid, title in zip(axes_grid.ravel(), grids, titles):
        im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto", vmin=vmin_grid, vmax=vmax_grid,
                       extent=[0.5, max_modes_in+0.5, 0.5, max_modes_out+0.5])
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Number of Input Modes (SST)")
        ax.set_ylabel("Number of Output Modes (SLP)")
        idx_max = np.unravel_index(np.argmax(grid, axis=None), grid.shape)
        ax.plot(idx_max[1] + 1, idx_max[0] + 1, 'r*', markersize=12, markeredgecolor='white', label=f"Max: {grid[idx_max]:.4f}")
        ax.legend(loc="lower right")

    fig_grid.subplots_adjust(right=0.9)
    cbar_ax = fig_grid.add_axes([0.93, 0.15, 0.02, 0.7])
    fig_grid.colorbar(im, cax=cbar_ax).set_label("$R^2$", fontweight="bold")
    plt.savefig(os.path.join(outdir, "benchmark_2D_heatmaps.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Plot récapitulatif 1D final : Comparaison directe (6 architectures viables)
    plt.figure(figsize=(18, 8)) # Figure plus large pour s'adapter aux 128 graduations
    
    # Pour chaque nombre de modes en entrée (Kin), on récupère la meilleure prédiction d'espace latent (max sur l'axe Kout)
    best_mca_mca = np.max(q2_mca2mca, axis=0) 
    best_mca_pca = np.max(q2_mca2pca, axis=0)
    best_pca_mca = np.max(q2_pca2mca, axis=0)
    best_pca_pca = np.max(q2_pca2pca, axis=0)

    # --- 1. LIGNES PLEINES : ENTRÉE = MCA ---
    plt.plot(modes_x, best_mca_mca, color="#d62728", marker="o", linewidth=1.5, markersize=2,
             label=f"Optimal MCA $\\rightarrow$ MCA (Max: {np.max(best_mca_mca):.5f})")
    plt.plot(modes_x, best_mca_pca, color="#9467bd", marker="d", linewidth=2.5, markersize=4,
             label=f"Optimal MCA $\\rightarrow$ PCA (Max: {np.max(best_mca_pca):.5f})")
    plt.plot(modes_x, vf_val_mca_list, color="#1f77b4", marker="^", linewidth=1.5, markersize=2,
             label=f"MCA $\\rightarrow$ Pixels (Max: {np.max(vf_val_mca_list):.5f})")
    
    # --- 2. LIGNES POINTILLÉES : ENTRÉE = PCA ---
    plt.plot(modes_x, best_pca_mca, color="#2ca02c", marker="p", linestyle="--", linewidth=1.5, markersize=2,
             label=f"Optimal PCA $\\rightarrow$ MCA (Max: {np.max(best_pca_mca):.5f})")
    plt.plot(modes_x, best_pca_pca, color="#7f7f7f", marker="s", linestyle="--", linewidth=2.5, markersize=4,
             label=f"Optimal PCA $\\rightarrow$ PCA (Max: {np.max(best_pca_pca):.5f})")
    plt.plot(modes_x, vf_val_pca_list, color="#ff7f0e", marker="v", linestyle="--", linewidth=1.5, markersize=2,
             label=f"PCA $\\rightarrow$ Pixels (Max: {np.max(vf_val_pca_list):.5f})")
    
    plt.title(f"Linear Models Val Spatial Skill Score Comparison ({month_label} SLP)", fontsize=15, fontweight="bold")
    plt.xlabel("Number of Input Modes (SST)", fontweight="bold", fontsize=12)
    plt.ylabel("$R^2$", fontweight="bold", fontsize=12)
    # plt.xticks(ticks)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Légende placée en bas à droite, avec un fond légèrement opaque pour être très lisible
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "benchmark_1D_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # FIN - LIBÉRATION MÉMOIRE
    # ==========================================================
    del X_train, X_train_centered, X_val, X_val_centered
    gc.collect()

# ============================================================
# 4. SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="hacienda")
    parser.add_argument("--duree_lissage", type=int, default=10)
    parser.add_argument("--sst_lags", type=int, nargs="+", default=[35], help="Lags en jours")
    parser.add_argument("--roll_sst", action="store_true", help="Lissage glissant océan Atlantique")
    parser.add_argument("--monthly_reduction", action="store_true", help="Données mensuelles")
    parser.add_argument("--lat_weight", action="store_true", help="Pondération spatiale cos(lat)")
    parser.add_argument("--winter_months", type=int, nargs="+", default=[11, 12, 1, 2], help="Mois d'hiver")
    parser.add_argument("--nb_val", type=int, default=5, help="Nombre de membres pour validation")
    parser.add_argument("--max_modes_in", type=int, default=20, help="Nombre maximum de modes à considérer")
    parser.add_argument("--max_modes_out", type=int, default=20, help="Nombre maximum de modes à considérer")
    args = parser.parse_args()

    all_members = [
        '1001.001',"1041.003", "1061.004", "1081.005", "1101.006", "1121.007", "1141.008", "1161.009", "1181.010",
        "1231.001", "1231.002", "1231.003", "1231.004", "1231.005", "1231.006", "1231.007", "1231.008",
        "1231.009", "1231.010", "1231.011", "1231.012", "1231.013", "1231.014", "1231.015", "1231.016",
        "1231.017", "1231.018", "1231.019", "1231.020", "1251.001", "1251.002", "1251.003", "1251.004",
        "1251.005", "1251.006", "1251.007", "1251.008", "1251.009", "1251.010", "1251.011", "1251.012",
        "1251.013", "1251.014", "1251.015", "1251.016", "1251.017", "1251.018", "1251.019", "1251.020",
        "1281.001", "1281.002", "1281.003", "1281.004", "1281.005", "1281.006", "1281.007", "1281.008",
        "1281.009", "1281.010", "1281.011", "1281.012", "1281.013", "1281.014", "1281.015", "1281.016",
        "1281.017", "1281.018", "1281.019", "1281.020", "1301.001", "1301.002", "1301.003", "1301.004",
        "1301.005", "1301.006", "1301.007", "1301.008", "1301.009", "1301.010", "1301.011", "1301.012",
        "1301.013", "1301.014", "1301.015", "1301.016", "1301.017", "1301.018", "1301.019", "1301.020",
    ]
    rng = random.Random(1)
    rng.shuffle(all_members)

    # n_val = int(len(all_members) * args.val_ratio)
    n_val = args.nb_val
    train_members = all_members[:-2*n_val] if n_val > 0 else all_members
    train2_members = all_members[-n_val:] if n_val > 0 else []
    # train_members = train_members + train2_members # on enlève pour comparer aux runs optuna
    val_members = all_members[-2*n_val:-n_val] if n_val > 0 else []
    print(f"\n📦 SPLIT DU JEU DE DONNÉES : {len(train_members)} membres Train | {len(val_members)} membres Validation")
    print(f"val_members: {val_members}")
    

    if args.machine == "hacienda":
        path_SLP, path_SST = "/data/moysan/data/SLP/", "/data/moysan/data/SST/"
        base_home = "/home/moysan/stage_isir_jz/data_analysis/mca_slp_sst/"
    elif args.machine == "jean-zay-work":
        path_SLP = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/"
        path_SST = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/mca_slp_sst/"
    else:
        path_SLP = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SLP/"
        path_SST = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SST/"
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/mca_slp_sst/"

    dynamic_sst_std, dynamic_slp_std = compute_mca_stds(
        train_members, path_SST, path_SLP, args.winter_months, sst_lags=args.sst_lags,
        duree_lissage=args.duree_lissage, roll_sst=args.roll_sst,
        monthly_reduction=args.monthly_reduction, lat_weight=args.lat_weight,
    )

    temp_res_str = "monthly" if args.monthly_reduction else f"{args.duree_lissage}d"
    outdir_name = f'Complete_MCA_month{"_".join(map(str, args.winter_months))}_{temp_res_str}_TrainVal_{len(train_members)}tr_lags_{"_".join(map(str, args.sst_lags))}_roll_{args.roll_sst}_latw_{args.lat_weight}_maxin_{args.max_modes_in}_maxout_{args.max_modes_out}_seed1val{args.nb_val}'
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    month_label = ""
    if 11 in args.winter_months:
        month_label += "November"
    if 12 in args.winter_months:
        month_label += "December"
    if 1 in args.winter_months:
        month_label += "January"
    if 2 in args.winter_months:
        month_label += "February"

    start_time = time.time()
    C = None
    N_total = 0
    global_wgts_sst, global_wgts_slp = None, None

    print("\nCalcul de la matrice de covariance croisée C (sans Cxx pour éviter le Segfault)...")
    for i, member in enumerate(train_members):
        X, Y, _, shape_sst, shape_slp, w_sst, w_slp = load_member_mca_data(
            member, path_SST, path_SLP, args.winter_months, sst_lags=args.sst_lags,
            duree_lissage=args.duree_lissage, roll_sst=args.roll_sst,
            monthly_reduction=args.monthly_reduction, lat_weight=args.lat_weight,
            sst_std=dynamic_sst_std, slp_std=dynamic_slp_std,
        )
        if w_sst is not None and global_wgts_sst is None:
            global_wgts_sst, global_wgts_slp = w_sst, w_slp
        N_total += X.shape[0]
        if C is None:
            C = X.T @ Y
        else:
            C += X.T @ Y

    C /= (N_total - 1)

    print("Calcul SVD en cours sur le jeu d'entraînement...")
    U, s, Vt = svd(C, full_matrices=False)
    V = Vt.T

    print("Extraction des variances temporelles du Mode 1 pour la cartographie...")
    var_A1, var_B1 = 0.0, 0.0
    for member in train_members:
        X, Y, _, _, _, _, _ = load_member_mca_data(
            member, path_SST, path_SLP, args.winter_months, sst_lags=args.sst_lags,
            duree_lissage=args.duree_lissage, roll_sst=args.roll_sst,
            monthly_reduction=args.monthly_reduction, lat_weight=args.lat_weight,
            sst_std=dynamic_sst_std, slp_std=dynamic_slp_std,
        )
        A1_mem = X @ U[:, 0]
        B1_mem = Y @ V[:, 0]
        var_A1 += np.sum(A1_mem**2)
        var_B1 += np.sum(B1_mem**2)
    sigma_A1 = np.sqrt(var_A1 / (N_total - 1))
    sigma_B1 = np.sqrt(var_B1 / (N_total - 1))

    plot_scf(s, outdir, n_modes_plot=30)
    plot_mca_modes(
        U, V.T, s, shape_sst, shape_slp, args.sst_lags, outdir, 
        sigma_A1=sigma_A1, sigma_B1=sigma_B1,
        sst_std=dynamic_sst_std, slp_std=dynamic_slp_std, n_modes=1,
        roll_sst=args.roll_sst, wgts_sst_flat=global_wgts_sst, wgts_slp_flat=global_wgts_slp,
    )

    model_filepath = os.path.join(outdir, f"mca_model_slp_std{dynamic_slp_std:.4f}_sst_std{dynamic_sst_std:.4f}.joblib")
    joblib.dump({"U": U, "V": V, "s": s, "shape_sst": shape_sst, "shape_slp": shape_slp}, model_filepath)

    if len(val_members) > 0:
        evaluate_train_val_modes(
            train_members, val_members, path_SST, path_SLP, args.winter_months, args.sst_lags,
            args.duree_lissage, args.roll_sst, args.monthly_reduction, args.lat_weight,
            dynamic_sst_std, dynamic_slp_std, U, V, shape_sst, shape_slp, outdir, 
            global_wgts_sst=global_wgts_sst, global_wgts_slp=global_wgts_slp, max_modes_in=min(args.max_modes_in, len(s)), max_modes_out=min(args.max_modes_out, len(s)),
            model_path=model_filepath 
        )

    print(f"\n⏱️ Exécution complète en {(time.time() - start_time) / 60:.2f} minutes.")