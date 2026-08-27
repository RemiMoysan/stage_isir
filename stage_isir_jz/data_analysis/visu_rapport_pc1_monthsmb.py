import os
import re
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==========================================
# 1. UTILITIES
# ==========================================
def get_file_path_slp(path_slp, mem):
    return os.path.join(path_slp, f'PSL_anom_LE2-{mem}_1mo.nc')

def apply_pca_lat_weights(data_3d, lats):
    w_sqrt = np.sqrt(np.cos(np.deg2rad(lats)))[None, :, None]
    weighted = data_3d * w_sqrt
    return np.nan_to_num(weighted, nan=0.0).reshape(data_3d.shape[0], -1)

def extract_std_from_path(path, var_name):
    pattern = rf"{var_name.lower()}_std([\d.]+)"
    match = re.search(pattern, path)
    if match: return float(match.group(1))
    else: raise ValueError(f"❌ Could not find '{var_name.lower()}_std...' in path: {path}")

# ==========================================
# 2. MAIN PROCESSING
# ==========================================
def analyze_pc1_distributions(member_ids, months_slp, path_slp, pca_path_slp, output_dir):
    print(f"\n=======================================================")
    print(f" 🚀 PHYSICAL PC1 DISTRIBUTION & EXPLAINED VARIANCE")
    print(f"=======================================================", flush=True)
    
    pca_slp = joblib.load(pca_path_slp)
    slp_std = extract_std_from_path(pca_path_slp, 'slp')
    print(f" ℹ️ Loaded SLP PCA Model. Extracted scaling std: {slp_std}", flush=True)

    month_str = '_'.join(map(str, months_slp))

    # Dictionnaires de stockage
    pc1_distributions = {}
    pc1_stds = {} 
    explained_variances_per_member = {m: [] for m in months_slp}
    
    global_pc1_sum = {m: 0.0 for m in months_slp}
    global_pc1_sq_sum = {m: 0.0 for m in months_slp}
    global_X_sum = {m: 0.0 for m in months_slp}
    global_X_sq_sum = {m: 0.0 for m in months_slp}
    global_count = {m: 0 for m in months_slp}
    global_min_pc1 = float('inf')
    global_max_pc1 = float('-inf')

    # --- ÉTAPE 1 : EXTRACTION DES DONNÉES ---
    print(f"\n -> Extracting physical PC1 scores for months {months_slp}...", flush=True)
    valid_members = []

    for mem in member_ids:
        file_slp = get_file_path_slp(path_slp, mem)
        if not os.path.exists(file_slp): continue
        with xr.open_dataset(file_slp) as ds_slp:
            cond_win_slp = ds_slp.time.dt.month.isin(months_slp)
            cond_not_jan2015 = ~((ds_slp.time.dt.year == 2015) & (ds_slp.time.dt.month == 1))
            slp_val = ds_slp['PSL'].sel(time=(cond_win_slp & cond_not_jan2015))
            lats = slp_val.lat.values
            time_months = slp_val.time.dt.month.values

            v_slp_3d = np.nan_to_num(slp_val.values, nan=0.0).astype(np.float32, copy=False)
            X_scaled = apply_pca_lat_weights(v_slp_3d / slp_std, lats)
            
            Z = pca_slp.transform(X_scaled)
            
            n_pixels = X_scaled.shape[1]
            pc1_scores_phys = (Z[:, 0] * slp_std) / np.sqrt(n_pixels)
            
            pc1_distributions[mem] = pc1_scores_phys
            pc1_stds[mem] = np.std(pc1_scores_phys)
            valid_members.append(mem)
            global_min_pc1 = min(global_min_pc1, pc1_scores_phys.min())
            global_max_pc1 = max(global_max_pc1, pc1_scores_phys.max())

            for m in months_slp:
                idx_m = (time_months == m)
                X_m = X_scaled[idx_m] 
                pc1_m = pc1_scores_phys[idx_m]
                n_t = len(pc1_m)
                if n_t == 0: continue
                
                var_pc1_m_latent = np.var(Z[idx_m, 0])
                total_var_m_latent = np.sum(np.var(X_m, axis=0))
                ev_ratio = var_pc1_m_latent / total_var_m_latent if total_var_m_latent > 0 else 0.0
                explained_variances_per_member[m].append(ev_ratio)
                
                global_pc1_sum[m] += np.sum(pc1_m, dtype=np.float64)
                global_pc1_sq_sum[m] += np.sum(pc1_m**2, dtype=np.float64)
                
                X_m_phys = X_m * slp_std
                if isinstance(global_X_sum[m], float): 
                    global_X_sum[m] = np.zeros(X_m_phys.shape[1], dtype=np.float64)
                    global_X_sq_sum[m] = np.zeros(X_m_phys.shape[1], dtype=np.float64)
                    
                global_X_sum[m] += np.sum(X_m_phys, axis=0, dtype=np.float64)
                global_X_sq_sum[m] += np.sum(X_m_phys**2, axis=0, dtype=np.float64)
                global_count[m] += n_t

        print(f"    - Member {mem} processed.", end='\r', flush=True)

    print(f"\n -> Extraction complete for {len(valid_members)} members.")

    # --- ÉTAPE 2 : CALCUL FINAL DE LA VARIANCE GLOBALE ---
    global_explained_variance = {}
    for m in months_slp:
        n = global_count[m]
        if n == 0: continue
        mean_pc1 = global_pc1_sum[m] / n
        var_pc1_global = (global_pc1_sq_sum[m] / n) - mean_pc1**2
        mean_X = global_X_sum[m] / n
        var_X_global = (global_X_sq_sum[m] / n) - mean_X**2
        
        global_explained_variance[m] = (var_pc1_global * n_pixels) / np.sum(var_X_global)

    # --- ÉTAPE 3 : EXPORT DES HISTOGRAMMES ANIMATION ---
    print("\n -> Generating individual PC1 physical distribution plots...", flush=True)
    anim_dir = os.path.join(output_dir, f"PC1_Distributions_Animation_m{month_str}")
    os.makedirs(anim_dir, exist_ok=True)

    for mem in valid_members:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(pc1_distributions[mem], bins=50, stat='density', kde=False, color='#2b5c8f', edgecolor='black', ax=ax)
        ax.set_xlim(global_min_pc1 * 1.1, global_max_pc1 * 1.1)
        
        mem_std = pc1_stds[mem]
        ax.set_title(f"PC1 Physical Distribution (m{month_str}) — Member {mem}\n(Std: {mem_std:.0f} Pa)", fontweight='bold', fontsize=12)
        ax.set_xlabel("PC1 Amplitude (Pa)")
        ax.set_ylabel("Density")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(anim_dir, f"PC1_Dist_Mem_{mem}.jpg"), dpi=150, pil_kwargs={'quality': 85})
        plt.close(fig)

    # --- ÉTAPE 4 : HEATMAP 1D DE L'ÉCART-TYPE PAR MEMBRE ---
    std_array = np.array([pc1_stds[mem] for mem in valid_members]).reshape(1, -1)
    df_std = pd.DataFrame(std_array, index=["Std (Pa)"], columns=valid_members)
    
    fig, ax = plt.subplots(figsize=(15, 2))
    sns.heatmap(df_std, ax=ax, cmap='magma_r', 
                xticklabels=valid_members, yticklabels=False, 
                cbar_kws={'label': 'PC1 Std Dev (Pa)'})
    ax.tick_params(axis='x', rotation=90, labelsize=5, length=0)
    ax.tick_params(axis='y', rotation=0, labelsize=9, length=0)
    ax.set_title(f"Physical Standard Deviation of PC1 per Ensemble Member (m{month_str})", fontweight='bold', fontsize=12)
    fig.tight_layout()
    heatmap_path = os.path.join(output_dir, f"PC1_Std_1D_Heatmap_m{month_str}.jpg")
    fig.savefig(heatmap_path, dpi=200, pil_kwargs={'quality': 90})
    plt.close(fig)

    # --- ÉTAPE 4.5 : BARPLOT 1D DU RATIO D'ÉCART-TYPE ---
    print("\n -> Generating 1D Barplot of PC1 Std Dev Ratios...", flush=True)
    all_pc1_scores = np.concatenate([pc1_distributions[mem] for mem in valid_members])
    global_std = np.std(all_pc1_scores)
    ratios = np.array([pc1_stds[mem] / global_std for mem in valid_members])
    
    fig, ax = plt.subplots(figsize=(15, 3.5))
    cmap = plt.get_cmap('magma')
    norm = mcolors.Normalize(vmin=0.8, vmax=1.2)
    bar_colors = [cmap(norm(val)) for val in ratios]

    M = len(valid_members)
    x_positions = np.arange(M)
    ax.bar(x_positions, ratios, color=bar_colors, edgecolor='black', linewidth=0.3, width=0.9)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(valid_members, rotation=90, fontsize=5)
    ax.set_xlim(-0.5, M - 0.5) 
    ax.set_ylim(0, 1.3)
    
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Std Ratio\n(Member / Global Std)", fontsize=10, labelpad=10)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5) 
    
    ax.text(-0.01, 0.5, f"Global Std:\n{global_std:.0f} Pa", transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='center', ha='right')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.08)
    cbar.set_label('Std Ratio')
    
    ax.set_title(f"PC1 Standard Deviation Ratio per Member (m{month_str})\nGlobal Pooled Std: {global_std:.0f} Pa", fontweight='bold', fontsize=12)
    ax.set_xlabel("Ensemble Member ID")
    
    fig.tight_layout()
    barplot_path = os.path.join(output_dir, f"PC1_Std_Ratio_1D_BarPlot_m{month_str}.jpg")
    fig.savefig(barplot_path, dpi=200, bbox_inches='tight', pil_kwargs={'quality': 90})
    plt.close(fig)

    # --- ÉTAPE 4.6 : EXPORT CSV DES RATIOS DE VARIANCE ---
    print("\n -> Exporting Std ratios to CSV...", flush=True)
    df_ratios = pd.DataFrame({
        'Member': valid_members,
        'Member_Std_Pa': [pc1_stds[mem] for mem in valid_members],
        'Global_Std_Pa': [global_std] * len(valid_members),
        'Ratio_Std_Member_vs_Global': ratios,
        'Ratio_Variance_Member_vs_Global': ratios**2 # C'est ce chiffre qui divise directement ton R2
    })
    
    csv_path = os.path.join(output_dir, f"PC1_Std_Ratios_m{month_str}.csv")
    df_ratios.to_csv(csv_path, index=False, float_format='%.4f')
    print(f" ✅ CSV saved: {csv_path}", flush=True)

    # --- ÉTAPE 5 : ANALYSE DES VARIANCES EXPLIQUÉES ---
    print("\n -> Generating Monthly Explained Variance summary...", flush=True)
    month_names = {11: 'November', 12: 'December', 1: 'January', 2: 'February'}
    month_colors = {11: '#d95f02', 12: '#7570b3', 1: '#1b9e77', 2: '#e7298a'}

    print("\n 📊 PC1 EXPLAINED VARIANCE BY MONTH:")
    for m in months_slp:
        mean_ev_members = np.mean(explained_variances_per_member[m]) * 100.0
        global_ev = global_explained_variance[m] * 100.0
        print(f"    - {month_names[m]:<10} | Avg of Members: {mean_ev_members:.2f}% | GLOBAL POOLED: {global_ev:.2f}%")

    fig, ax = plt.subplots(figsize=(11, 6))
    data_to_plot = [[val * 100.0 for val in explained_variances_per_member[m]] for m in months_slp]
    labels = [f"{month_names[m]} (Global $R^2$: {global_explained_variance[m]*100:.1f}%)" for m in months_slp]
    colors = [month_colors[m] for m in months_slp]

    ax.hist(data_to_plot, bins=15, label=labels, color=colors, edgecolor='black', alpha=0.85)
    ax.set_title(f"PC1 Explained Variance Distribution across Ensemble Members (m{month_str})", fontweight='bold', fontsize=13)
    ax.set_xlabel("Explained Variance by PC1 (%)")
    ax.set_ylabel("Number of Members")
    ax.legend(
        title="Winter Months", title_fontsize='11', loc='upper right', 
        fontsize=10, framealpha=0.95, edgecolor='#333333', fancybox=True
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    ev_plot_path = os.path.join(output_dir, f"PC1_Monthly_Explained_Variance_m{month_str}.jpg")
    fig.savefig(ev_plot_path, dpi=200, pil_kwargs={'quality': 90})
    plt.close(fig)

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
    parser.add_argument('--pca_slp', type=str, required=True, help="Path to SLP best_pca_model.joblib")
    args = parser.parse_args()

    folder_name = "report_pc1_distributions"
    if args.machine == 'hacienda':
        base_home = f"/home/moysan/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = "/data/moysan/data/"
    else:
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = f"/lustre/{'fswork' if args.machine == 'jean-zay-work' else 'fsn1'}/projects/rech/uxg/uca57ub/data/"

    os.makedirs(base_home, exist_ok=True)
    path_slp = os.path.join(data_dir, "SLP/")

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    
    # RESTRICTION EXCLUSIVE À FÉVRIER
    months_slp = [2]

    analyze_pc1_distributions(all_members, months_slp, path_slp, args.pca_slp, base_home)