import os
import argparse
import glob
import numpy as np
import pandas as pd
import xarray as xr
import joblib
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. UTILITIES & WEIGHTING
# ==========================================
def get_file_path_slp(path_slp, mem, is_monthly=False):
    suffix = "_1mo" if is_monthly else ""
    return os.path.join(path_slp, f'PSL_anom_LE2-{mem}{suffix}.nc')

def apply_pca_lat_weights(data_3d, lats):
    """
    Applique la pondération sqrt(cos(lat)) requise avant projection PCA.
    data_3d: shape (T, Lat, Lon)
    """
    w_sqrt = np.sqrt(np.cos(np.deg2rad(lats)))[:, None]
    weighted = data_3d * w_sqrt
    return weighted.reshape(data_3d.shape[0], -1)

# ==========================================
# 2. EXTRACTION DES STATS, BASELINES & EMBEDDINGS
# ==========================================
def process_ensemble_slp(member_ids, months_slp, path_slp, pca_model_path, is_monthly=False):
    time_str = "Monthly" if is_monthly else "Daily"
    print(f"\n=======================================================")
    print(f" 🚀 STARTING {time_str.upper()} ANALYSIS (SLP Winter NDJF)")
    print(f"=======================================================", flush=True)
    
    if not os.path.exists(pca_model_path):
        raise FileNotFoundError(f"❌ PCA model not found at: {pca_model_path}")
    
    pca_model = joblib.load(pca_model_path)
    n_components = pca_model.n_components_
    print(f" ℹ️ Loaded pre-trained PCA model: {pca_model_path} (n_components={n_components})", flush=True)

    stats = {
        'mem': [],
        'pixel_mae_0': [], 'pixel_mae_med': [],
        'pca_mae_0': [], 'pca_mae_med': []
    }
    
    # Dictionnaire pour stocker les embeddings temporels (T, n_components) par membre
    latent_embeddings = {}
    weights_2d = None

    for i, mem in enumerate(member_ids):
        file_slp = get_file_path_slp(path_slp, mem, is_monthly)
        if not os.path.exists(file_slp): continue
        
        with xr.open_dataset(file_slp) as ds_slp:
            # --- 1. FILTRAGE DES MOIS NDJF & EXCLUSION JANVIER 2015 ---
            cond_winter = ds_slp.time.dt.month.isin(months_slp)
            cond_not_jan2015 = ~((ds_slp.time.dt.year == 2015) & (ds_slp.time.dt.month == 1))
            
            slp_val = ds_slp['PSL'].sel(time=(cond_winter & cond_not_jan2015))
            
            if i == 0:
                print(f" ℹ️ Date Check (Member {mem}) — Excluded Jan 2015:")
                print(f"    - First Timestamp : {str(slp_val.time.values[0])[:10]}")
                print(f"    - Total Timestamps: {len(slp_val.time.values)} ({time_str} steps)", flush=True)
                
                # Masque géométrique 2D pour le calcul de la MAE physique pondérée
                lats = slp_val.lat.values
                mask_valid = ~np.isnan(slp_val.isel(time=0).values)
                weights_2d = np.cos(np.deg2rad(lats))[:, None] * mask_valid
                sum_weights = np.sum(weights_2d)

            # --- 2. ESPACE PIXEL : MOYENNE, MÉDIANE & MAE PONDÉRÉE ---
            v_slp_3d = np.nan_to_num(slp_val.values, nan=0.0).astype(np.float32, copy=False)
            T_steps = v_slp_3d.shape[0]

            # Moyenne et Médiane temporelle par pixel (Shape: Lat, Lon)
            pixel_mean = np.mean(v_slp_3d, axis=0)
            pixel_med = np.median(v_slp_3d, axis=0)

            # MAE spatialement pondérée par cos(lat) sur tout le domaine et tout le temps
            mae_0_pix = np.sum(np.abs(v_slp_3d - 0.0) * weights_2d) / (T_steps * sum_weights)
            mae_med_pix = np.sum(np.abs(v_slp_3d - pixel_med) * weights_2d) / (T_steps * sum_weights)

            # --- 3. ESPACE LATENT (PCA) : PROJECTION & BASELINES ---
            slp_flat_weighted = apply_pca_lat_weights(v_slp_3d, lats)
            Z = pca_model.transform(slp_flat_weighted).astype(np.float32, copy=False) # Shape: (T, n_components)
            
            # Sauvegarde pour le calcul ultérieur de Wasserstein
            latent_embeddings[mem] = Z

            pca_mean = np.mean(Z, axis=0)
            pca_med = np.median(Z, axis=0)

            mae_0_pca = np.mean(np.abs(Z - 0.0))
            mae_med_pca = np.mean(np.abs(Z - pca_med))

            stats['mem'].append(mem)
            stats['pixel_mae_0'].append(mae_0_pix); stats['pixel_mae_med'].append(mae_med_pix)
            stats['pca_mae_0'].append(mae_0_pca); stats['pca_mae_med'].append(mae_med_pca)

        print(f" -> Member {mem} processed.", end='\r', flush=True)
    
    print(f"\n -> Extraction complete for {len(stats['mem'])} members.", flush=True)
    return stats, latent_embeddings

# ==========================================
# 3. ANALYSE DU FACTEUR D'ERREUR L1 (1 - SKILL SCORE)
# ==========================================
def evaluate_l1_skill_score_scaling(stats, output_dir, is_monthly=False):
    time_str = "Monthly" if is_monthly else "Daily"
    print(f"\n--- [1/2] Quantifying L1 Baseline Error Scaling Factor ({time_str}) ---", flush=True)
    
    df = pd.DataFrame(stats)
    
    # Facteur d'échelle : MAE(0) / MAE(Median)
    df['Factor_Pixel_L1'] = df['pixel_mae_0'] / df['pixel_mae_med']
    df['Factor_PCA_L1'] = df['pca_mae_0'] / df['pca_mae_med']
    
    print(f" 📊 SUMMARY OF L1 ERROR SCALING (1 - Skill Score Multiplier):")
    print(f"    [Physical Pixels (Lat-Weighted)]:")
    print(f"    - Mean Naive MAE(0)    : {df['pixel_mae_0'].mean():.2f} Pa")
    print(f"    - Mean Optimal MAE(Med): {df['pixel_mae_med'].mean():.2f} Pa")
    print(f"    => Scaling Factor      : {df['Factor_Pixel_L1'].mean():.4f}x ({(df['Factor_Pixel_L1'].mean()-1)*100:.2f}% artificial L1 penalty with Pred=0)")
    print(f"    ---------------------------------------------------")
    print(f"    [Latent Space (PCA Embeddings)]:")
    print(f"    - Mean Naive MAE(0)    : {df['pca_mae_0'].mean():.4f}")
    print(f"    - Mean Optimal MAE(Med): {df['pca_mae_med'].mean():.4f}")
    print(f"    => Scaling Factor      : {df['Factor_PCA_L1'].mean():.4f}x ({(df['Factor_PCA_L1'].mean()-1)*100:.2f}% artificial L1 penalty with Pred=0)", flush=True)
    
    # Export CSV du tableau par membre
    csv_path = os.path.join(output_dir, f"L1_Skill_Score_Scaling_{time_str}.csv")
    df.to_csv(csv_path, index=False)
    print(f" ✅ L1 scaling summary table saved: {csv_path}", flush=True)

# ==========================================
# 4. MATRICES DE WASSERSTEIN PAR COMPOSANTE (ZÉRO BOUCLE TEMPS)
# ==========================================
def compute_pairwise_wasserstein_matrices(latent_embeddings, max_components, output_dir, is_monthly=False):
    time_str = "Monthly" if is_monthly else "Daily"
    print(f"\n--- [2/2] Computing Pairwise Wasserstein-1 Matrices for Top {max_components} PCs ({time_str}) ---", flush=True)
    
    members = sorted(list(latent_embeddings.keys()))
    M = len(members)
    
    # Empilement dans un tableau 3D -> Shape: (M_members, T_steps, n_components)
    Z_stacked = np.array([latent_embeddings[m] for m in members])
    
    # L'ASTUCE MATHÉMATIQUE : En 1D, Wasserstein-1 est la moyenne L1 des séries TRIÉES !
    print(" -> Sorting temporal embeddings along time axis for instant 1D optimal transport...", flush=True)
    Z_sorted = np.sort(Z_stacked[:, :, :max_components], axis=1) # Shape: (M, T, max_components)
    
    # Dictionnaire de stockage des matrices M x M pour chaque PC
    w1_matrices = {}
    
    for c in range(max_components):
        mat_c = np.zeros((M, M), dtype=np.float64)
        for i in range(M):
            for j in range(i + 1, M):
                # Distance W1 exacte entre le membre i et j pour la composante c : simple moyenne !
                w_ij = np.mean(np.abs(Z_sorted[i, :, c] - Z_sorted[j, :, c]))
                mat_c[i, j] = w_ij
                mat_c[j, i] = w_ij
        w1_matrices[f"PC_{c+1}"] = mat_c
        print(f" -> Computed Wasserstein matrix for PC {c+1}/{max_components}", end='\r', flush=True)
        
    print(f"\n -> Done computing all {max_components} Wasserstein matrices!", flush=True)

    # --- SAUVEGARDE COMPLÈTE EN BASE DE DONNÉES (.NPZ) ---
    npz_path = os.path.join(output_dir, f"Wasserstein_Matrices_Top{max_components}PCs_{time_str}.npz")
    np.savez(npz_path, members=np.array(members), **w1_matrices)
    print(f" ✅ Complete set of W1 matrices saved to: {npz_path}", flush=True)

    # --- TRACÉ VISUEL DES 4 PREMIÈRES PC ---
    n_plot = min(4, max_components)
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    axes = axes.flatten()
    
    for idx in range(n_plot):
        pc_key = f"PC_{idx+1}"
        mat = w1_matrices[pc_key]
        ax = axes[idx]
        sns.heatmap(mat, ax=ax, cmap='magma_r', xticklabels=members, yticklabels=members, cbar_kws={'label': 'Wasserstein-1 Distance'})
        ax.set_title(f"Pairwise Wasserstein-1 Distance — {pc_key} ({time_str})", fontweight='bold', fontsize=11)
        ax.set_xlabel("Member ID"); ax.set_ylabel("Member ID")
        
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"Wasserstein_Heatmaps_Top{n_plot}PCs_{time_str}.jpg")
    fig.savefig(plot_path, dpi=200, pil_kwargs={'quality': 90, 'subsampling': 0})
    plt.close(fig)
    print(f" ✅ Visual heatmaps for top {n_plot} PCs saved: {plot_path}", flush=True)
    return w1_matrices

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
    parser.add_argument('--number_of_members', type=int, default=10)
    parser.add_argument('--max_components', type=int, default=10, help="Nombre de PC pour calculer Wasserstein 2 à 2")
    parser.add_argument('--pca_monthly', type=str, required=True, help="Path to monthly best_pca_model.joblib")
    parser.add_argument('--pca_daily', type=str, required=True, help="Path to daily best_pca_model.joblib")
    args = parser.parse_args()

    folder_name = "report_pca_baselines_wasserstein"
    if args.machine == 'hacienda':
        base_home = f"/home/moysan/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = "/data/moysan/data/"
    else:
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = f"/lustre/{'fswork' if args.machine == 'jean-zay-work' else 'fsn1'}/projects/rech/uxg/uca57ub/data/"

    os.makedirs(base_home, exist_ok=True)
    path_slp = os.path.join(data_dir, "SLP/")

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020']
    members_used = all_members[:args.number_of_members]
    months_slp = [11, 12, 1, 2] # NDJF pour SLP

    start_time = time.time()
    print(f"\n=== STARTING SLP ANALYSIS PIPELINE ===", flush=True)
    # --- 1. RUN MONTHLY ---
    stats_m, emb_m = process_ensemble_slp(members_used, months_slp, path_slp, args.pca_monthly, is_monthly=True)
    evaluate_l1_skill_score_scaling(stats_m, base_home, is_monthly=True)
    compute_pairwise_wasserstein_matrices(emb_m, args.max_components, base_home, is_monthly=True)
    mid_time = time.time()
    print(f"\n=== MONTHLY ANALYSIS COMPLETED in {mid_time - start_time:.2f} seconds ===", flush=True)
    # --- 2. RUN DAILY ---
    stats_d, emb_d = process_ensemble_slp(members_used, months_slp, path_slp, args.pca_daily, is_monthly=False)
    evaluate_l1_skill_score_scaling(stats_d, base_home, is_monthly=False)
    compute_pairwise_wasserstein_matrices(emb_d, args.max_components, base_home, is_monthly=False)
    print(f"\n=== DAILY ANALYSIS COMPLETED in {time.time() - mid_time:.2f} seconds ===", flush=True)