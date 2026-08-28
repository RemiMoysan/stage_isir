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
from sklearn.cross_decomposition import PLSRegression
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
# 3. OPTIMISATIONS PLS (TRAIN vs VAL)
# ============================================================

def optimize_pls1_nao(X_tr, X_val, nao_tr, nao_val, max_kin):
    """Optimisation 1D pour la PLS1 sur la NAO."""
    var_nao = np.sum(nao_val**2)
    scores_1d = []
    best_kin, max_score = 1, -np.inf
    
    for kin in range(1, max_kin + 1):
        pls = PLSRegression(n_components=kin, scale=False)
        pls.fit(X_tr, nao_tr)
        pred_nao = pls.predict(X_val).squeeze()
        score = 1.0 - np.sum((nao_val - pred_nao)**2) / var_nao
        scores_1d.append(score)
        if score > max_score:
            max_score, best_kin = score, kin
            
    return best_kin, max_score, scores_1d

def optimize_multi_pls1_pca(X_tr, X_val, Y_tr, Y_val, max_kin, max_kout):
    """Entraîne une PLS1 indépendante pour chaque PC atmosphérique, puis optimise K_out."""
    # 1. PCA sur l'atmosphère (Cible orthogonale)
    pca_slp = TruncatedSVD(n_components=max_kout, random_state=42)
    B_tr = pca_slp.fit_transform(Y_tr)
    B_val = pca_slp.transform(Y_val)
    var_total_Y_val = np.sum(Y_val**2)
    
    best_kins_per_pc = []
    best_preds_val = np.zeros_like(B_val)
    
    # 2. PLS1 indépendante pour chaque mode atmosphérique (Spécialisation)
    for j in range(max_kout):
        best_r2_pc, best_k_pc, best_pred_pc = -np.inf, 1, None
        var_pc = np.sum(B_val[:, j]**2)
        
        for kin in range(1, max_kin + 1):
            pls = PLSRegression(n_components=kin, scale=False)
            pls.fit(X_tr, B_tr[:, j])
            pred_pc = pls.predict(X_val).squeeze()
            
            # Éviter la division par zéro si var_pc est infime
            if var_pc > 1e-10:
                r2 = 1.0 - np.sum((B_val[:, j] - pred_pc)**2) / var_pc
            else:
                r2 = 0.0
                
            if r2 > best_r2_pc:
                best_r2_pc, best_k_pc, best_pred_pc = r2, kin, pred_pc
                
        best_kins_per_pc.append(best_k_pc)
        best_preds_val[:, j] = best_pred_pc

    # 3. Reconstruction spatiale et balayage sur K_out
    scores_spatial = []
    best_kout, max_score_spatial = 1, -np.inf
    
    for kout in range(1, max_kout + 1):
        mse_latent = np.sum((B_val[:, :kout] - best_preds_val[:, :kout])**2)
        var_trunc = var_total_Y_val - np.sum(B_val[:, :kout]**2)
        score = 1.0 - (mse_latent + var_trunc) / var_total_Y_val
        scores_spatial.append(score)
        
        if score > max_score_spatial:
            max_score_spatial, best_kout = score, kout

    return best_kins_per_pc, best_kout, max_score_spatial, scores_spatial

def optimize_pls2_pixels(X_tr, X_val, Y_tr, Y_val, max_k):
    """Optimisation PLS2 globale (Prédiction multivariée simultanée)."""
    var_total_Y_val = np.sum(Y_val**2)
    scores_1d = []
    best_k, max_score = 1, -np.inf
    
    for k in range(1, max_k + 1):
        pls2 = PLSRegression(n_components=k, scale=False)
        pls2.fit(X_tr, Y_tr)
        pred_Y = pls2.predict(X_val)
        score = 1.0 - np.sum((Y_val - pred_Y)**2) / var_total_Y_val
        scores_1d.append(score)
        if score > max_score:
            max_score, best_k = score, k
            
    return best_k, max_score, scores_1d

# ============================================================
# 4. ÉVALUATION ET LOOCV GLOBALE EXHAUSTIVE (PLS)
# ============================================================

def evaluate_and_loocv_pls(
    data_cache_X, data_cache_Y, all_members, train_members, val_members,
    outdir, max_modes_in, max_modes_out, month_label
):
    print(f"\n{'='*75}\n🏆 PHASE 1: EXHAUSTIVE PLS OPTIMIZATION (TRAIN vs VAL)\n{'='*75}")
    
    train_idx = [all_members.index(m) for m in train_members]
    val_idx = [all_members.index(m) for m in val_members]
    
    X_train = np.concatenate([data_cache_X[i] for i in train_idx])
    Y_train = np.concatenate([data_cache_Y[i] for i in train_idx])
    X_val = np.concatenate([data_cache_X[i] for i in val_idx])
    Y_val = np.concatenate([data_cache_Y[i] for i in val_idx])
    
    X_train_mean, Y_train_mean = np.mean(X_train, axis=0), np.mean(Y_train, axis=0)
    X_tr_c, Y_tr_c = X_train - X_train_mean, Y_train - Y_train_mean
    X_val_c, Y_val_c = X_val - X_train_mean, Y_val - Y_train_mean

    # NAO : Mode 1 de la PCA sur SLP
    pca_slp = TruncatedSVD(n_components=1, random_state=42)
    nao_tr = pca_slp.fit_transform(Y_tr_c)[:, 0]
    nao_val = pca_slp.transform(Y_val_c)[:, 0]

    opt_results = {}

    # --- 1. PLS1 -> NAO ---
    k1, s1, prof1 = optimize_pls1_nao(X_tr_c, X_val_c, nao_tr, nao_val, max_modes_in)
    opt_results["PLS1_NAO"] = {"kin": k1, "kout": None, "score": s1, "profile": prof1, "kins_array": None}

    # --- 2. Multi-PLS1 -> PCA(SLP) -> Pixels ---
    kins_array, k2_out, s2, prof2 = optimize_multi_pls1_pca(X_tr_c, X_val_c, Y_tr_c, Y_val_c, max_modes_in, max_modes_out)
    opt_results["Multi-PLS1_PCA"] = {"kin": "Multiple", "kout": k2_out, "score": s2, "profile": prof2, "kins_array": kins_array}

    # --- 3. PLS2 -> SLP Pixels ---
    k3, s3, prof3 = optimize_pls2_pixels(X_tr_c, X_val_c, Y_tr_c, Y_val_c, max_modes_out) # max_modes_out pilote les dimensions ici
    opt_results["PLS2_Pix"] = {"kin": k3, "kout": k3, "score": s3, "profile": prof3, "kins_array": None}

    # --- SAUVEGARDE CSV DES OPTIMUMS ---
    records = []
    for conf, data in opt_results.items():
        # Pour le CSV principal, on convertit l'array en string lisible si c'est le modèle Multi
        kin_display = str(data["kins_array"]) if data["kins_array"] is not None else data["kin"]
        
        records.append({
            "Configuration": conf, 
            "Optimal_K_in": kin_display, 
            "Optimal_K_out": data["kout"] if data["kout"] is not None else "N/A", 
            "Max_Val_Score": data["score"]
        })
    df_opt = pd.DataFrame(records)
    df_opt.to_csv(os.path.join(outdir, f"Optimal_Modes_Summary_PLS_{month_label}.csv"), index=False)
    print(df_opt.to_string(index=False))

    # --- SAUVEGARDE DÉTAILLÉE DES K_IN POUR MULTI-PLS1 ---
    kins_multi = opt_results["Multi-PLS1_PCA"]["kins_array"]
    df_kins = pd.DataFrame({
        "Atmospheric_PC": np.arange(1, len(kins_multi) + 1),
        "Optimal_K_in_PLS": kins_multi
    })
    csv_kins_path = os.path.join(outdir, f"Multi_PLS1_Kin_per_PC_{month_label}.csv")
    df_kins.to_csv(csv_kins_path, index=False)
    print(f"📊 Tableau des K_in optimaux par PC sauvegardé : {csv_kins_path}")

    # --- PLOT 1: PLS1 -> NAO ---
    plt.figure(figsize=(10, 5))
    x_ax_nao = range(1, max_modes_in + 1)
    plt.plot(x_ax_nao, opt_results["PLS1_NAO"]["profile"], color="#d62728", marker="x", linestyle="-", markevery=2, 
             label=f"PLS1 $\\rightarrow$ NAO (Max: {opt_results['PLS1_NAO']['score']:.4f} at K={opt_results['PLS1_NAO']['kin']})")
    plt.title(f"NAO Target Val Skill Score - PLS ({month_label})", fontweight="bold")
    plt.xlabel("Number of Latent Modes (PLS1)")
    plt.ylabel("Explained Variance Fraction ($R^2$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pls1_nao_target_R2_skill.png"), dpi=200)
    plt.close()

    # --- PLOT 2: SPATIAL (Multi-PLS1 & PLS2) ---
    plt.figure(figsize=(12, 6))
    x_ax_spatial = range(1, max_modes_out + 1)
    
    lbl_multi = f"Multi-PLS1 $\\rightarrow$ PCA (Max: {opt_results['Multi-PLS1_PCA']['score']:.4f} | out={opt_results['Multi-PLS1_PCA']['kout']})"
    plt.plot(x_ax_spatial, opt_results["Multi-PLS1_PCA"]["profile"], color="#1f77b4", linestyle="-", marker="o", markevery=2, label=lbl_multi)
    
    lbl_pls2 = f"PLS2 $\\rightarrow$ Pixels (Max: {opt_results['PLS2_Pix']['score']:.4f} | K={opt_results['PLS2_Pix']['kin']})"
    plt.plot(x_ax_spatial, opt_results["PLS2_Pix"]["profile"], color="#ff7f0e", linestyle="--", marker="s", markevery=2, label=lbl_pls2)

    plt.title(f"Spatial Target Val Skill Score - PLS - {month_label}", fontweight="bold")
    plt.xlabel("Number of Output/Latent Modes ($K_{out}$)")
    plt.ylabel("Explained Spatial Variance ($R^2$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spatial_target_R2_skill_PLS.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*75}\n🚀 PHASE 2: EXHAUSTIVE LOOCV ON PLS CONFIGURATIONS (AUTO-RESUME)\n{'='*75}", flush=True)
    
    csv_scores_path = os.path.join(outdir, f"LOOCV_Scores_Summary_PLS_{month_label}.csv")
    
    # --- 1. INITIALISATION OU REPRISE (CHECKPOINTING) ---
    if os.path.exists(csv_scores_path):
        print(f"🔄 Fichier de sauvegarde trouvé ! Reprise de l'exécution...")
        df_scores = pd.read_csv(csv_scores_path, index_col="Test_Member")
        # S'assure que toutes les colonnes existent
        for k in opt_results.keys():
            if k not in df_scores.columns:
                df_scores[k] = np.nan
    else:
        print(f"🆕 Aucun fichier de sauvegarde. Démarrage de zéro...")
        df_scores = pd.DataFrame(index=all_members, columns=list(opt_results.keys()))

    # --- 2. BOUCLE LOOCV AVEC SAUT DES MEMBRES DÉJÀ CALCULÉS ---
    for i, test_mem in enumerate(all_members):
        # Vérifie si le membre a déjà été calculé (pas de NaN dans sa ligne)
        if not pd.isna(df_scores.loc[test_mem, "PLS1_NAO"]):
            print(f"[{i+1}/{len(all_members)}] ⏩ Membre {test_mem} déjà calculé. On passe.", flush=True)
            continue
            
        t_start_loo = time.time()
        print(f"[{i+1}/{len(all_members)}] Running strict LOOCV on member {test_mem}...", flush=True)
        
        X_test, Y_test = data_cache_X[i], data_cache_Y[i]
        X_train_loo = np.concatenate([data_cache_X[j] for j in range(len(all_members)) if j != i])
        Y_train_loo = np.concatenate([data_cache_Y[j] for j in range(len(all_members)) if j != i])
        
        Y_mean_loo = Y_train_loo.mean(axis=0)
        Y_tr_c = Y_train_loo - Y_mean_loo
        Y_te_c = Y_test - Y_mean_loo
        
        var_Y_te = np.sum(Y_te_c**2)

        # ---------------------------------------------------------
        # RECALCUL TOTAL DES BASES SPATIALES SUR N-1
        # ---------------------------------------------------------
        pca_slp_loo = TruncatedSVD(n_components=max_modes_out, random_state=42)
        B_pca_tr = pca_slp_loo.fit_transform(Y_tr_c)
        V_pca_loo = pca_slp_loo.components_.T
        
        nao_tr_loo = B_pca_tr[:, 0]
        nao_te_loo = pca_slp_loo.transform(Y_te_c)[:, 0]
        var_nao_te = np.sum(nao_te_loo**2)

        # --- PREDICTION 1: PLS1 -> NAO ---
        kin_pls1 = opt_results["PLS1_NAO"]["kin"]
        pls1 = PLSRegression(n_components=kin_pls1, scale=False)
        pls1.fit(X_train_loo, nao_tr_loo) 
        pred_nao = pls1.predict(X_test).squeeze() 
        df_scores.loc[test_mem, "PLS1_NAO"] = 1.0 - np.sum((nao_te_loo - pred_nao)**2) / var_nao_te
        
        # --- PREDICTION 2: Multi-PLS1 -> PCA ---
        kout_multi = opt_results["Multi-PLS1_PCA"]["kout"]
        kins_array = opt_results["Multi-PLS1_PCA"]["kins_array"]
        pred_latent_loo = np.zeros((X_test.shape[0], kout_multi))
        
        for j in range(kout_multi):
            pls_multi = PLSRegression(n_components=kins_array[j], scale=False)
            pls_multi.fit(X_train_loo, B_pca_tr[:, j])
            pred_latent_loo[:, j] = pls_multi.predict(X_test).squeeze()
            
        pred_Y_multi = pred_latent_loo @ V_pca_loo[:, :kout_multi].T
        df_scores.loc[test_mem, "Multi-PLS1_PCA"] = 1.0 - np.sum((Y_te_c - pred_Y_multi)**2) / var_Y_te
        
        # --- PREDICTION 3: PLS2 -> Pixels ---
        k_pls2 = opt_results["PLS2_Pix"]["kin"]
        pls2 = PLSRegression(n_components=k_pls2, scale=False)
        pls2.fit(X_train_loo, Y_tr_c) 
        pred_Y_pls2 = pls2.predict(X_test)
        df_scores.loc[test_mem, "PLS2_Pix"] = 1.0 - np.sum((Y_te_c - pred_Y_pls2)**2) / var_Y_te

        print(f"   -> Membre {test_mem} complété en {time.time() - t_start_loo:.1f} sec", flush=True)

        # --- 3. SAUVEGARDE IMMÉDIATE (CHECKPOINT) ---
        df_scores.to_csv(csv_scores_path)
        print(f"   💾 Progression sauvegardée.", flush=True)

        # Nettoyage mémoire
        del X_train_loo, Y_train_loo, Y_tr_c, Y_te_c
        gc.collect()

    print("\n✅ LOOCV exhaustive terminée avec succès !", flush=True)

    # --- 4. GENERATION DES GRAPHIQUES (Seulement quand tout est fini) ---
    generate_1d_loocv_barplot(all_members, df_scores["PLS1_NAO"].values, outdir, "NAO Explained Variance (PLS1 $\\rightarrow$ NAO)")
    generate_1d_loocv_barplot(all_members, df_scores["Multi-PLS1_PCA"].values, outdir, "SLP Spatial Explained Variance (Multi-PLS1 $\\rightarrow$ PCA)")
    generate_1d_loocv_barplot(all_members, df_scores["PLS2_Pix"].values, outdir, "SLP Spatial Explained Variance (PLS2 $\\rightarrow$ PIXELS)")
    
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
    
    parser.add_argument("--max_modes_in", type=int, default=20, help="Max modes PLS (k_in) à tester")
    parser.add_argument("--max_modes_out", type=int, default=20, help="Max modes SLP (k_out) à tester")
    
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
        base_home = "/home/moysan/stage_isir_jz/data_analysis/pls/"
    elif args.machine == "jean-zay-work":
        path_SLP = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/"
        path_SST = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pls/"
    else:
        path_SLP, path_SST = "./SLP/", "./SST/"
        base_home = "./"

    dynamic_sst_std, dynamic_slp_std = compute_mca_stds(
        train_members, path_SST, path_SLP, args.winter_months, sst_lags=args.sst_lags,
        duree_lissage=args.duree_lissage, roll_sst=args.roll_sst,
        monthly_reduction=args.monthly_reduction, lat_weight=args.lat_weight,
    )

    temp_res_str = "monthly" if args.monthly_reduction else f"{args.duree_lissage}d"
    outdir_name = f'LOOCV_month{"_".join(map(str, args.winter_months))}_{temp_res_str}_lags_{"_".join(map(str, args.sst_lags))}_latw_{args.lat_weight}_PLS_Exhaustive'
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
    
    evaluate_and_loocv_pls(
        data_cache_X, data_cache_Y, ALL_MEMBERS, train_members, val_members,
        outdir, args.max_modes_in, args.max_modes_out, month_label
    )
    print(f"Temps total d'exécution : {time.time() - t0:.1f} s")