import argparse
import calendar
from datetime import timedelta
import os
import time
import cartopy.crs as ccrs
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.linalg import svd
import numpy as np
import pandas as pd
import random
import xarray as xr
from sklearn.decomposition import TruncatedSVD
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
# 2. LOOCV PLOTTING FUNCTION
# ============================================================

def generate_1d_loocv_barplot(all_members, scores, output_dir, metric_name, val_member_used=""):
    print(f"--- Génération du Barplot 1D Global ({metric_name}) ---", flush=True)
    M = len(all_members)
    global_mean = np.nanmean(scores)
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(15, 3.5))
    cmap = plt.get_cmap("magma")
    norm = mcolors.Normalize(vmin=np.nanmin(scores), vmax=np.nanmax(scores))
    bar_colors = [cmap(norm(val)) if not np.isnan(val) else (0,0,0,0) for val in scores]

    x_positions = np.arange(M)
    ax.bar(x_positions, scores, color=bar_colors, edgecolor='black', linewidth=0.3, width=0.9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_members, rotation=90, fontsize=5)
    ax.set_xlim(-0.5, M - 0.5) 
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Score", fontsize=10, labelpad=10)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5) 
    
    if val_member_used:
        ax.text(-0.01, 0.5, f"Val:\n{val_member_used}", transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='center', ha='right')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.08)
    cbar.set_label(f'Score')
    
    ax.set_title(f"1D LOOCV per Test Member — {metric_name}\nMean Score: {global_mean:.4f}", fontweight='bold', fontsize=12)
    ax.set_xlabel("Test Member ID")
    
    fig.tight_layout()
    # Nettoyage robuste pour que le LaTeX ne casse pas le nom de fichier
    safe_name = metric_name.replace(' ', '_').replace('$\\rightarrow$', 'to').replace('(', '').replace(')', '').replace('\\', '').replace('$', '')
    
    barplot_path = os.path.join(output_dir, f"LOOCV_1D_BarPlot_{safe_name}.jpg")
    fig.savefig(barplot_path, dpi=200, bbox_inches='tight', pil_kwargs={'quality': 90})
    plt.close(fig)

# ============================================================
# 3. OPTIMISATION RAPIDE DES MODES
# ============================================================

def optimize_1d_nao(A_tr, A_te, nao_tr, nao_te, max_kin):
    """Optimisation 1D pour la NAO."""
    var_nao = np.var(nao_te)
    scores_1d = []
    best_kin, max_score = 1, -np.inf
    
    for kin in range(1, max_kin + 1):
        W, _, _, _ = np.linalg.lstsq(A_tr[:, :kin], nao_tr, rcond=None)
        pred_nao = A_te[:, :kin] @ W
        score = 1.0 - np.var(nao_te - pred_nao) / var_nao
        scores_1d.append(score)
        if score > max_score:
            max_score, best_kin = score, kin
            
    return best_kin, max_score, scores_1d

def optimize_1d_pixels(A_tr, A_te, Y_tr, Y_te, max_kin):
    """Optimisation 1D pour la prédiction directe des Pixels."""
    var_Y = np.sum(Y_te**2)
    scores_1d = []
    best_kin, max_score = 1, -np.inf
    
    for kin in range(1, max_kin + 1):
        W, _, _, _ = np.linalg.lstsq(A_tr[:, :kin], Y_tr, rcond=None)
        pred_Y = A_te[:, :kin] @ W
        score = 1.0 - np.sum((Y_te - pred_Y)**2) / var_Y
        scores_1d.append(score)
        if score > max_score:
            max_score, best_kin = score, kin
            
    return best_kin, max_score, scores_1d

def optimize_2d_modes(A_tr, A_te, B_tr, B_te, var_Y, max_kin, max_kout):
    """Optimisation 2D ultra-rapide avec l'astuce de l'erreur latente."""
    scores_1d_profile = []
    best_kin, best_kout, max_score = 1, 1, -np.inf
    
    for kin in range(1, max_kin + 1):
        # Calcul de W pour tous les modes 'out' d'un coup
        W_all, _, _, _ = np.linalg.lstsq(A_tr[:, :kin], B_tr[:, :max_kout], rcond=None)
        pred_latent = A_te[:, :kin] @ W_all
        
        kin_scores = []
        for kout in range(1, max_kout + 1):
            # 1. Erreur sur les kout modes prédits
            mse_latent = np.sum((B_te[:, :kout] - pred_latent[:, :kout])**2)
            
            # 2. Erreur de troncature (Variance totale - Variance des kout vrais modes)
            var_trunc = var_Y - np.sum(B_te[:, :kout]**2)
            
            # 3. Score Q2 exact sans jamais reconstruire les pixels
            score = 1.0 - (mse_latent + var_trunc) / var_Y
            kin_scores.append(score)
            
            if score > max_score:
                max_score, best_kin, best_kout = score, kin, kout
                
        scores_1d_profile.append(np.max(kin_scores))
        
    return best_kin, best_kout, max_score, scores_1d_profile

# ============================================================
# 4. ÉVALUATION ET LOOCV GLOBALE EXHAUSTIVE
# ============================================================

def evaluate_and_loocv(
    data_cache_X, data_cache_Y, all_members, train_members, val_members,
    outdir, max_modes_in, max_modes_out, month_label
):
    print(f"\n{'='*75}\n🏆 PHASE 1: EXHAUSTIVE MODE OPTIMIZATION (TRAIN vs VAL)\n{'='*75}")
    
    train_idx = [all_members.index(m) for m in train_members]
    val_idx = [all_members.index(m) for m in val_members]
    
    X_train = np.concatenate([data_cache_X[i] for i in train_idx])
    Y_train = np.concatenate([data_cache_Y[i] for i in train_idx])
    X_val = np.concatenate([data_cache_X[i] for i in val_idx])
    Y_val = np.concatenate([data_cache_Y[i] for i in val_idx])
    
    X_train_mean, Y_train_mean = np.mean(X_train, axis=0), np.mean(Y_train, axis=0)
    X_tr_c, Y_tr_c = X_train - X_train_mean, Y_train - Y_train_mean
    X_val_c, Y_val_c = X_val - X_train_mean, Y_val - Y_train_mean

    # 👇 AJOUTE CETTE LIGNE ICI 👇
    var_total_val_Y = np.sum(Y_val_c**2)

    # Bases globales pour le Train Set
    U_mca, _, Vt_mca = svd(X_tr_c.T @ Y_tr_c, full_matrices=False)
    V_mca = Vt_mca.T
    
    pca_sst = TruncatedSVD(n_components=max_modes_in, random_state=42)
    A_pca_tr = pca_sst.fit_transform(X_tr_c)
    A_pca_val = pca_sst.transform(X_val_c)
    
    A_mca_tr = X_tr_c @ U_mca[:, :max_modes_in]
    A_mca_val = X_val_c @ U_mca[:, :max_modes_in]
    
    pca_slp = TruncatedSVD(n_components=max_modes_out, random_state=42)
    B_pca_tr = pca_slp.fit_transform(Y_tr_c)
    V_pca = pca_slp.components_.T
    
    # NAO (Mode 1 PCA SLP)
    nao_tr = B_pca_tr[:, 0]
    nao_val = pca_slp.transform(Y_val_c)[:, 0]

    # Latents MCA SLP Out
    B_mca_tr = Y_tr_c @ V_mca[:, :max_modes_out]

    # --- EXECUTION DES 8 RECHERCHES D'OPTIMISATION ---
    opt_results = {}
    
    # 1-2: Cible NAO
    k1, s1, prof1 = optimize_1d_nao(A_mca_tr, A_mca_val, nao_tr, nao_val, max_modes_in)
    opt_results["MCA_NAO"] = {"kin": k1, "kout": None, "score": s1, "profile": prof1}
    
    k2, s2, prof2 = optimize_1d_nao(A_pca_tr, A_pca_val, nao_tr, nao_val, max_modes_in)
    opt_results["PCA_NAO"] = {"kin": k2, "kout": None, "score": s2, "profile": prof2}

    # 3-4: Cible Pixels (Direct)
    k3, s3, prof3 = optimize_1d_pixels(A_mca_tr, A_mca_val, Y_tr_c, Y_val_c, max_modes_in)
    opt_results["MCA_Pix"] = {"kin": k3, "kout": None, "score": s3, "profile": prof3}

    k4, s4, prof4 = optimize_1d_pixels(A_pca_tr, A_pca_val, Y_tr_c, Y_val_c, max_modes_in)
    opt_results["PCA_Pix"] = {"kin": k4, "kout": None, "score": s4, "profile": prof4}

    # Cible MCA(SLP) -> Pixels
    B_mca_val = Y_val_c @ V_mca[:, :max_modes_out]
    k5_in, k5_out, s5, prof5 = optimize_2d_modes(A_mca_tr, A_mca_val, B_mca_tr, B_mca_val, var_total_val_Y, max_modes_in, max_modes_out)
    opt_results["MCA_MCA"] = {"kin": k5_in, "kout": k5_out, "score": s5, "profile": prof5}

    k6_in, k6_out, s6, prof6 = optimize_2d_modes(A_pca_tr, A_pca_val, B_mca_tr, B_mca_val, var_total_val_Y, max_modes_in, max_modes_out)
    opt_results["PCA_MCA"] = {"kin": k6_in, "kout": k6_out, "score": s6, "profile": prof6}

    # Cible PCA(SLP) -> Pixels
    B_pca_val = pca_slp.transform(Y_val_c)
    k7_in, k7_out, s7, prof7 = optimize_2d_modes(A_mca_tr, A_mca_val, B_pca_tr, B_pca_val, var_total_val_Y, max_modes_in, max_modes_out)
    opt_results["MCA_PCA"] = {"kin": k7_in, "kout": k7_out, "score": s7, "profile": prof7}

    k8_in, k8_out, s8, prof8 = optimize_2d_modes(A_pca_tr, A_pca_val, B_pca_tr, B_pca_val, var_total_val_Y, max_modes_in, max_modes_out)
    opt_results["PCA_PCA"] = {"kin": k8_in, "kout": k8_out, "score": s8, "profile": prof8}

    # --- SAUVEGARDE CSV DES OPTIMUMS ---
    records = []
    for conf, data in opt_results.items():
        records.append({
            "Configuration": conf, "Optimal_K_in": data["kin"], 
            "Optimal_K_out": data["kout"] if data["kout"] is not None else "N/A", 
            "Max_Val_Score": data["score"]
        })
    df_opt = pd.DataFrame(records)
    df_opt.to_csv(os.path.join(outdir, f"Optimal_Modes_Summary_{month_label}.csv"), index=False)
    print(df_opt.to_string(index=False))

    # --- PLOT 1: NAO (2 Curves) ---
    plt.figure(figsize=(10, 5))
    x_ax = range(1, max_modes_in + 1)
    plt.plot(x_ax, opt_results["MCA_NAO"]["profile"], color="#9467bd", marker="d", markevery=10, 
             label=f"MCA $\\rightarrow$ NAO (Max: {opt_results['MCA_NAO']['score']:.4f} at K={opt_results['MCA_NAO']['kin']})")
    plt.plot(x_ax, opt_results["PCA_NAO"]["profile"], color="#d62728", marker="x", linestyle="--", markevery=10, 
             label=f"PCA $\\rightarrow$ NAO (Max: {opt_results['PCA_NAO']['score']:.4f} at K={opt_results['PCA_NAO']['kin']})")
    plt.title(f"NAO Target Val Skill Score ({month_label})", fontweight="bold")
    plt.xlabel("Number of Input Modes (SST)")
    plt.ylabel("Explained Variance Fraction ($R^2$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "nao_target_R2_skill.png"), dpi=200)
    plt.close()

    # --- PLOT 2: SPATIAL (6 Curves) ---
    plt.figure(figsize=(18, 8)) # Format allongé rétabli
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    styles = ['-', '--', '-', '--', '-', '--']
    markers = ['o', 'x', 's', '^', 'D', 'v']
    
    spatial_keys = ["MCA_Pix", "PCA_Pix", "MCA_MCA", "PCA_MCA", "MCA_PCA", "PCA_PCA"]
    for i, key in enumerate(spatial_keys):
        d = opt_results[key]
        in_t, out_t = key.split('_')
        out_t_str = "Pixels" if out_t == "Pix" else out_t
        
        # Titre propre avec flèche LaTeX
        lbl = f"{in_t} $\\rightarrow$ {out_t_str} (Max: {d['score']:.4f} | in={d['kin']}"
        if d['kout'] is not None: lbl += f", out={d['kout']}"
        lbl += ")"
        
        plt.plot(x_ax, d["profile"], color=colors[i], linestyle=styles[i], marker=markers[i], markevery=10, label=lbl)

    # Nouveaux titres propres
    plt.title(f"Spatial Target Val Skill Score (Max over optimal number of output modes if applicable) - {month_label}", fontweight="bold")
    plt.xlabel("Number of Input Modes (SST)")
    plt.ylabel("Explained Spatial Variance ($R^2$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Légende en bas à droite
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spatial_target_R2_skill_6curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*75}\n🚀 PHASE 2: EXHAUSTIVE LOOCV ON 8 CONFIGURATIONS\n{'='*75}")
    loocv_scores = {k: np.zeros(len(all_members)) for k in opt_results.keys()}

    for i, test_mem in enumerate(all_members):
        print(f"[{i+1}/{len(all_members)}] Running strict LOOCV on member {test_mem}...", end="\r", flush=True)
        
        X_test, Y_test = data_cache_X[i], data_cache_Y[i]
        X_train_loo = np.concatenate([data_cache_X[j] for j in range(len(all_members)) if j != i])
        Y_train_loo = np.concatenate([data_cache_Y[j] for j in range(len(all_members)) if j != i])
        
        X_mean_loo, Y_mean_loo = X_train_loo.mean(axis=0), Y_train_loo.mean(axis=0)
        X_tr_c, Y_tr_c = X_train_loo - X_mean_loo, Y_train_loo - Y_mean_loo
        X_te_c, Y_te_c = X_test - X_mean_loo, Y_test - Y_mean_loo
        
        var_nao_te = np.var(Y_te_c) # Default if NAO isn't isolated, updated below
        var_Y_te = np.sum(Y_te_c**2)

        # ---------------------------------------------------------
        # RECALCUL TOTAL DES BASES SPATIALES SUR N-1 (NO LEAKAGE)
        # ---------------------------------------------------------
        # 1. MCA
        U_loo, _, Vt_loo = svd(X_tr_c.T @ Y_tr_c, full_matrices=False)
        V_loo = Vt_loo.T
        
        # 2. PCA SST
        max_req_pca_in = max([opt_results[k]["kin"] for k in opt_results.keys() if "PCA_" in k])
        pca_sst_loo = TruncatedSVD(n_components=max_req_pca_in, random_state=42)
        A_pca_tr = pca_sst_loo.fit_transform(X_tr_c)
        A_pca_te = pca_sst_loo.transform(X_te_c)
        
        # 3. PCA SLP
        max_req_pca_out = max([opt_results[k]["kout"] for k in opt_results.keys() if "PCA" in k and opt_results[k]["kout"] is not None])
        max_req_pca_out = max(1, max_req_pca_out) # Au moins 1 pour la NAO
        
        pca_slp_loo = TruncatedSVD(n_components=max_req_pca_out, random_state=42)
        B_pca_tr = pca_slp_loo.fit_transform(Y_tr_c)
        V_pca_loo = pca_slp_loo.components_.T
        
        nao_tr_loo = B_pca_tr[:, 0]
        nao_te_loo = pca_slp_loo.transform(Y_te_c)[:, 0]
        var_nao_te = np.var(nao_te_loo)

        # Pre-compute MCA Latents
        A_mca_tr = X_tr_c @ U_loo
        A_mca_te = X_te_c @ U_loo
        B_mca_tr = Y_tr_c @ V_loo

        # --- EXECUTION DES 8 PREDICTIONS ---
        for conf, data in opt_results.items():
            kin, kout = data["kin"], data["kout"]
            is_pca_in = "PCA_" in conf
            A_tr = A_pca_tr[:, :kin] if is_pca_in else A_mca_tr[:, :kin]
            A_te = A_pca_te[:, :kin] if is_pca_in else A_mca_te[:, :kin]

            if "NAO" in conf:
                W, _, _, _ = np.linalg.lstsq(A_tr, nao_tr_loo, rcond=None)
                pred = A_te @ W
                loocv_scores[conf][i] = 1.0 - np.var(nao_te_loo - pred) / var_nao_te
            
            elif "Pix" in conf:
                W, _, _, _ = np.linalg.lstsq(A_tr, Y_tr_c, rcond=None)
                pred = A_te @ W
                loocv_scores[conf][i] = 1.0 - np.sum((Y_te_c - pred)**2) / var_Y_te
                
            elif "MCA" in conf.split("_")[1]: # x -> MCA -> Pix
                W, _, _, _ = np.linalg.lstsq(A_tr, B_mca_tr[:, :kout], rcond=None)
                pred = (A_te @ W) @ V_loo[:, :kout].T
                loocv_scores[conf][i] = 1.0 - np.sum((Y_te_c - pred)**2) / var_Y_te
                
            elif "PCA" in conf.split("_")[1]: # x -> PCA -> Pix
                W, _, _, _ = np.linalg.lstsq(A_tr, B_pca_tr[:, :kout], rcond=None)
                pred = (A_te @ W) @ V_pca_loo[:, :kout].T
                loocv_scores[conf][i] = 1.0 - np.sum((Y_te_c - pred)**2) / var_Y_te

    print("\n✅ LOOCV exhaustive terminée avec succès !")

    # --- SAUVEGARDE DES SCORES LOOCV ---
    df_scores = pd.DataFrame(loocv_scores, index=all_members)
    df_scores.index.name = "Test_Member"
    csv_scores_path = os.path.join(outdir, f"LOOCV_Scores_Summary_{month_label}.csv")
    df_scores.to_csv(csv_scores_path)
    print(f"📊 Tableau des scores LOOCV sauvegardé : {csv_scores_path}")

    # --- GENERATION DES 8 BARPLOTS ---
    for conf in opt_results.keys():
        in_t, out_t = conf.split("_")
        out_t_str = "PIXELS" if out_t == "Pix" else out_t
        
        # Création d'un titre très clair pour le rapport
        if "NAO" in conf:
            plot_title = f"NAO Explained Variance ({in_t} $\\rightarrow$ {out_t_str})"
        else:
            plot_title = f"SLP Spatial Explained Variance ({in_t} $\\rightarrow$ {out_t_str})"
            
        generate_1d_loocv_barplot(all_members, loocv_scores[conf], outdir, plot_title)

# ============================================================
# 4. SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="hacienda")
    parser.add_argument("--duree_lissage", type=int, default=10)
    parser.add_argument("--sst_lags", type=int, nargs="+", default=[35], help="Lags en jours ou mois")
    parser.add_argument("--roll_sst", action="store_true", help="Lissage glissant océan Atlantique")
    parser.add_argument("--monthly_reduction", action="store_true", help="Données mensuelles")
    parser.add_argument("--lat_weight", action="store_true", help="Pondération spatiale cos(lat)")
    parser.add_argument("--winter_months", type=int, nargs="+", default=[11, 12, 1, 2], help="Mois d'hiver")
    parser.add_argument("--nb_val", type=int, default=5, help="Nombre de membres pour validation (Graphe)")
    
    # NOUVEAUX ARGUMENTS GLOBAUX POUR L'OPTIMISATION
    parser.add_argument("--max_modes_in", type=int, default=300, help="Max modes SST (PCA/MCA) à tester")
    parser.add_argument("--max_modes_out", type=int, default=100, help="Max modes SLP (PCA/MCA) à tester")
    
    args = parser.parse_args()

    ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    ALL_MEMBERS_copy = ALL_MEMBERS.copy()
    rng = random.Random(1)
    rng.shuffle(ALL_MEMBERS_copy)

    n_val = args.nb_val
    train_members = ALL_MEMBERS_copy[:-2*n_val] if n_val > 0 else ALL_MEMBERS_copy
    val_members = ALL_MEMBERS_copy[-2*n_val:-n_val] if n_val > 0 else []
    print(f"membres validation: {val_members}")

    if args.machine == "hacienda":
        path_SLP, path_SST = "/data/moysan/data/SLP/", "/data/moysan/data/SST/"
        base_home = "/home/moysan/stage_isir_jz/data_analysis/mca_slp_sst/"
    elif args.machine == "jean-zay-work":
        path_SLP = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/"
        path_SST = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/mca_slp_sst/"
    else:
        path_SLP, path_SST = "./SLP/", "./SST/"
        base_home = "./"

    dynamic_sst_std, dynamic_slp_std = compute_mca_stds(
        train_members, path_SST, path_SLP, args.winter_months, sst_lags=args.sst_lags,
        duree_lissage=args.duree_lissage, roll_sst=args.roll_sst,
        monthly_reduction=args.monthly_reduction, lat_weight=args.lat_weight,
    )

    temp_res_str = "monthly" if args.monthly_reduction else f"{args.duree_lissage}d"
    outdir_name = f'LOOCV_month{"_".join(map(str, args.winter_months))}_{temp_res_str}_lags_{"_".join(map(str, args.sst_lags))}_latw_{args.lat_weight}_Exhaustive'
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    month_label = ""
    for m, name in zip([11, 12, 1, 2], ["November", "December", "January", "February"]):
        if m in args.winter_months: month_label += name

    print("\n📦 MISE EN CACHE des données des 89 membres...")
    t0 = time.time()
    data_cache_X, data_cache_Y = [], []
    for member in ALL_MEMBERS:
        X, Y, _, _, _, _, _ = load_member_mca_data(
            member, path_SST, path_SLP, args.winter_months, sst_lags=args.sst_lags,
            duree_lissage=args.duree_lissage, roll_sst=args.roll_sst,
            monthly_reduction=args.monthly_reduction, lat_weight=args.lat_weight,
            sst_std=dynamic_sst_std, slp_std=dynamic_slp_std,
        )
        data_cache_X.append(X)
        data_cache_Y.append(Y)
        
    print(f"✅ Données préchargées en RAM en {time.time() - t0:.1f} s !")
    
    evaluate_and_loocv(
        data_cache_X, data_cache_Y, ALL_MEMBERS, train_members, val_members,
        outdir, args.max_modes_in, args.max_modes_out, month_label
    )
    print(f"Temps total d'exécution : {time.time() - t0:.1f} s")