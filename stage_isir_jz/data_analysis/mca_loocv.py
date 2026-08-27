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
    """
    Génère un Barplot 1D coloré pour les scores de la LOOCV.
    """
    print(f"\n--- Génération du Barplot 1D Global ({metric_name}) ---", flush=True)
    
    M = len(all_members)
    global_mean = np.nanmean(scores)
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(15, 3.5))
    
    # 🔴 CORRECTION ICI : on force la colormap 'magma' pour matcher tes autres plots
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
    
    # Texte de validation (vide si test pur)
    if val_member_used:
        ax.text(-0.01, 0.5, f"Val:\n{val_member_used}", transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='center', ha='right')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.08)
    cbar.set_label(f'{metric_name}')
    
    ax.set_title(f"1D LOOCV per Test Member — {metric_name}\nMean {metric_name}: {global_mean:.4f}", fontweight='bold', fontsize=12)
    ax.set_xlabel("Test Member ID")
    
    fig.tight_layout()
    barplot_path = os.path.join(output_dir, f"LOOCV_1D_BarPlot_{metric_name.replace(' ', '_')}.jpg")
    
    fig.savefig(barplot_path, dpi=200, bbox_inches='tight', pil_kwargs={'quality': 90})
    plt.close(fig)
    print(f"✅ Barplot 1D généré : {barplot_path}")


# ============================================================
# 3. ÉVALUATION ET LOOCV GLOBALE
# ============================================================

def evaluate_and_loocv(
    data_cache_X, data_cache_Y, all_members, train_members, val_members,
    outdir, max_modes_in, month_label, in_type="pca"
):
    print(f"\n{'='*75}\n🏆 PHASE 1: OPTIMISATION DU NOMBRE DE MODES (TRAIN vs VAL)\n{'='*75}")
    
    # --- 1.1 Préparation du split Train/Val Standard ---
    train_idx = [all_members.index(m) for m in train_members]
    val_idx = [all_members.index(m) for m in val_members]
    
    X_train = np.concatenate([data_cache_X[i] for i in train_idx])
    Y_train = np.concatenate([data_cache_Y[i] for i in train_idx])
    X_val = np.concatenate([data_cache_X[i] for i in val_idx])
    Y_val = np.concatenate([data_cache_Y[i] for i in val_idx])
    
    X_train_mean, Y_train_mean = np.mean(X_train, axis=0), np.mean(Y_train, axis=0)
    X_train_c, Y_train_c = X_train - X_train_mean, Y_train - Y_train_mean
    X_val_c, Y_val_c = X_val - X_train_mean, Y_val - Y_train_mean
    
    var_total_val_Y = np.sum(Y_val_c**2)

    # MCA Train
    C = X_train_c.T @ Y_train_c
    U, s, Vt = svd(C, full_matrices=False)
    V = Vt.T
    
    # PCA Train
    pca_sst = TruncatedSVD(n_components=max_modes_in, random_state=42)
    A_pca_train = pca_sst.fit_transform(X_train_c)
    A_pca_val = pca_sst.transform(X_val_c)
    
    A_mca_train = X_train_c @ U[:, :max_modes_in]
    A_mca_val = X_val_c @ U[:, :max_modes_in]
    
    # PCA SLP (pour la Cible NAO pure)
    U_pca_slp, S_pca_slp, Vt_pca_slp = svd(Y_train_c, full_matrices=False)
    PC_pca_train = Y_train_c @ Vt_pca_slp.T[:, :max_modes_in]
    PC_pca_val = Y_val_c @ Vt_pca_slp.T[:, :max_modes_in]

    # Sauvegarde des modèles globaux
    model_filepath = os.path.join(outdir, f"models_pca_mca_{month_label}.joblib")
    joblib.dump({
        "X_train_mean": X_train_mean, "Y_train_mean": Y_train_mean,
        "mca_U": U, "mca_V": V, "mca_s": s,
        "pca_sst": pca_sst, "pca_slp_Vt": Vt_pca_slp
    }, model_filepath)

    # --- 1.2 RECHERCHE DES MODES OPTIMAUX ---
    r2_mca_nao, r2_pca_nao = [], []
    q2_mca_global, q2_pca_global = [], []
    
    for k in range(1, max_modes_in + 1):
        # -- Cible NAO --
        w_mca, _, _, _ = np.linalg.lstsq(A_mca_train[:, :k], PC_pca_train[:, 0], rcond=None)
        r2_mca_nao.append(1.0 - np.var(PC_pca_val[:, 0] - (A_mca_val[:, :k] @ w_mca)) / np.var(PC_pca_val[:, 0]))
        
        w_pca, _, _, _ = np.linalg.lstsq(A_pca_train[:, :k], PC_pca_train[:, 0], rcond=None)
        r2_pca_nao.append(1.0 - np.var(PC_pca_val[:, 0] - (A_pca_val[:, :k] @ w_pca)) / np.var(PC_pca_val[:, 0]))
        
        # -- Cible Pixels (Spatial) --
        w_mca_sp, _, _, _ = np.linalg.lstsq(A_mca_train[:, :k], Y_train_c, rcond=None)
        q2_mca_global.append(1.0 - np.sum((Y_val_c - (A_mca_val[:, :k] @ w_mca_sp))**2) / var_total_val_Y)
        
        w_pca_sp, _, _, _ = np.linalg.lstsq(A_pca_train[:, :k], Y_train_c, rcond=None)
        q2_pca_global.append(1.0 - np.sum((Y_val_c - (A_pca_val[:, :k] @ w_pca_sp))**2) / var_total_val_Y)
        
    # Identification des meilleurs K
    best_k_mca_nao = np.argmax(r2_mca_nao) + 1
    best_k_pca_nao = np.argmax(r2_pca_nao) + 1
    best_k_mca_global = np.argmax(q2_mca_global) + 1
    best_k_pca_global = np.argmax(q2_pca_global) + 1

    # --- 1.3 SAUVEGARDE DES VALEURS OPTIMALES DANS UN TABLEAU ---
    df_optimal = pd.DataFrame({
        "Configuration": ["MCA -> NAO", "PCA -> NAO", "MCA -> Pixels", "PCA -> Pixels"],
        "Best_K": [best_k_mca_nao, best_k_pca_nao, best_k_mca_global, best_k_pca_global],
        "Max_Score": [np.max(r2_mca_nao), np.max(r2_pca_nao), np.max(q2_mca_global), np.max(q2_pca_global)],
        "Metric": ["R2", "R2", "Q2", "Q2"]
    })
    csv_opt_path = os.path.join(outdir, f"Optimal_Modes_Summary_{month_label}.csv")
    df_optimal.to_csv(csv_opt_path, index=False)
    print(f"📊 Tableau des modes optimaux sauvegardé : {csv_opt_path}")

    # Tracé du graphe NAO (optionnel, on garde l'existant)
    plt.figure(figsize=(12, 6))
    modes_x = range(1, max_modes_in + 1)
    plt.plot(modes_x, r2_mca_nao, color="#9467bd", marker="d", markevery=5,
             label=f"MCA $\\rightarrow$ NAO (Max: {np.max(r2_mca_nao):.4f} at K={best_k_mca_nao})")
    plt.plot(modes_x, r2_pca_nao, color="#d62728", marker="x", linestyle="--", markevery=5,
             label=f"PCA $\\rightarrow$ NAO (Max: {np.max(r2_pca_nao):.4f} at K={best_k_pca_nao})")
    plt.title(f"Target: NAO Val Skill Score ({month_label})", fontweight="bold")
    plt.xlabel("Number of Input Modes (SST)")
    plt.ylabel("Explained Variance Fraction ($R^2$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(outdir, "nao_target_R2_skill_2curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- SÉLECTION DES MODES POUR LA LOOCV EN FONCTION DE IN_TYPE ---
    if in_type == "pca":
        opt_kin_global = best_k_pca_global
        opt_kin_nao = best_k_pca_nao
    else:
        opt_kin_global = best_k_mca_global
        opt_kin_nao = best_k_mca_nao

    print(f"\n{'='*75}\n🚀 PHASE 2: LOOCV ({in_type.upper()} -> PIXELS) SUR {len(all_members)} MEMBRES\n{'='*75}")
    print(f"Modes retenus pour la LOOCV -> Global: {opt_kin_global} | NAO: {opt_kin_nao}")
    
    scores_q2_global = np.zeros(len(all_members))
    scores_r2_nao = np.zeros(len(all_members))
    max_k_needed = max(opt_kin_global, opt_kin_nao)

    for i, test_mem in enumerate(all_members):
        print(f"[{i+1}/{len(all_members)}] LOOCV - Testing on member {test_mem}...", end="\r", flush=True)
        
        X_test, Y_test = data_cache_X[i], data_cache_Y[i]
        X_train_loo = np.concatenate([data_cache_X[j] for j in range(len(all_members)) if j != i])
        Y_train_loo = np.concatenate([data_cache_Y[j] for j in range(len(all_members)) if j != i])
        
        X_mean_loo, Y_mean_loo = X_train_loo.mean(axis=0), Y_train_loo.mean(axis=0)
        X_tr_c, Y_tr_c = X_train_loo - X_mean_loo, Y_train_loo - Y_mean_loo
        X_te_c, Y_te_c = X_test - X_mean_loo, Y_test - Y_mean_loo
        
        if in_type == "pca":
            pca_sst_loo = TruncatedSVD(n_components=max_k_needed, random_state=42)
            A_tr_all = pca_sst_loo.fit_transform(X_tr_c)
            A_te_all = pca_sst_loo.transform(X_te_c)
            A_tr_global, A_te_global = A_tr_all[:, :opt_kin_global], A_te_all[:, :opt_kin_global]
            A_tr_nao, A_te_nao = A_tr_all[:, :opt_kin_nao], A_te_all[:, :opt_kin_nao]
        else: # mca
            C_loo = X_tr_c.T @ Y_tr_c
            U_loo, _, _ = svd(C_loo, full_matrices=False)
            A_tr_global, A_te_global = X_tr_c @ U_loo[:, :opt_kin_global], X_te_c @ U_loo[:, :opt_kin_global]
            A_tr_nao, A_te_nao = X_tr_c @ U_loo[:, :opt_kin_nao], X_te_c @ U_loo[:, :opt_kin_nao]

        # -- PREDICTION 1 : GLOBALE (Pixels) --
        W_global, _, _, _ = np.linalg.lstsq(A_tr_global, Y_tr_c, rcond=None)
        pred_pixels_te = A_te_global @ W_global
        
        scores_q2_global[i] = 1.0 - (np.sum((Y_te_c - pred_pixels_te)**2) / np.sum(Y_te_c**2))

        # -- PREDICTION 2 : NAO PURE --
        pca_slp_nao = TruncatedSVD(n_components=1, random_state=42)
        nao_tr = pca_slp_nao.fit_transform(Y_tr_c)[:, 0]
        nao_te = pca_slp_nao.transform(Y_te_c)[:, 0]

        W_nao, _, _, _ = np.linalg.lstsq(A_tr_nao, nao_tr, rcond=None)
        pred_nao_te = A_te_nao @ W_nao
        
        scores_r2_nao[i] = 1.0 - (np.var(nao_te - pred_nao_te) / np.var(nao_te))
        
    print("\n✅ LOOCV terminée avec succès !")

    generate_1d_loocv_barplot(all_members, scores_q2_global, outdir, f"SLP Spatial Explained Variance({in_type.upper()} -> PIXELS)")
    generate_1d_loocv_barplot(all_members, scores_r2_nao, outdir, f"NAO Explained Variance ({in_type.upper()} -> PIXELS)")

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
    parser.add_argument("--max_modes_in", type=int, default=200, help="Max modes à tester pour l'optimisation")
    parser.add_argument("--in_type", type=str, default="pca", choices=["pca", "mca"], help="Type de réduction en entrée (SST)")
    args = parser.parse_args()

    ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    ALL_MEMBERS_copy = ALL_MEMBERS.copy()
    rng = random.Random(1)
    rng.shuffle(ALL_MEMBERS_copy)

    n_val = args.nb_val
    train_members = ALL_MEMBERS_copy[:-2*n_val] if n_val > 0 else ALL_MEMBERS_copy
    val_members = ALL_MEMBERS_copy[-2*n_val:-n_val] if n_val > 0 else []
    print(f"membres de val : {val_members}")
    print(f"months d'hiver : {args.winter_months}, lags : {args.sst_lags}, roll_sst : {args.roll_sst}, monthly_reduction : {args.monthly_reduction}, lat_weight : {args.lat_weight}")

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
    outdir_name = f'LOOCV_month{"_".join(map(str, args.winter_months))}_{temp_res_str}_lags_{"_".join(map(str, args.sst_lags))}_latw_{args.lat_weight}_in_{args.in_type}_out_pixel'
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    month_label = ""
    for m, name in zip([11, 12, 1, 2], ["November", "December", "January", "February"]):
        if m in args.winter_months: month_label += name

    # --- MISE EN CACHE MASSIVE (Préchargement des 89 membres) ---
    print("\n📦 MISE EN CACHE des données des 89 membres (pour accélérer la LOOCV)...")
    t0 = time.time()
    data_cache_X = []
    data_cache_Y = []
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
        outdir, args.max_modes_in, month_label, args.in_type
    )
    print(f"Temps total d'exécution : {time.time() - t0:.1f} s")