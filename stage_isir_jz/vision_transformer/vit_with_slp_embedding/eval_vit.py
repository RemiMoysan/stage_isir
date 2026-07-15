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

import torch
import torch.nn as nn

# import des dossiers siblings
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.datasets import Dataset, Dataset_mensuel
from tools.models import ConvVAE, ViT_Latent_SLP_Multimodal

# ============================================================
# STATISTICAL & PLOTTING FUNCTIONS
# ============================================================

def compute_two_bootstraps(true_values, pred_values, n_iterations):
    """
    Calcule deux distributions bootstrap : 
    1. Target permutée vs Prédiction
    2. Target permutée vs Target initiale

    pour r et pour R^2
    """
    true_values = np.asarray(true_values).flatten()
    pred_values = np.asarray(pred_values).flatten()
    
    if len(true_values) < 2:
        return np.nan, np.nan,None, None, None
        
    original_corr, _ = pearsonr(true_values, pred_values)
    mse_orig = np.mean((true_values - pred_values)**2)
    var_orig = np.var(true_values)
    original_r2 = 1 - (mse_orig / var_orig) if var_orig > 0 else np.nan
    n = len(true_values)
    
    corr_target_pred_boot = np.zeros(n_iterations)
    corr_target_target_boot = np.zeros(n_iterations)
    r2_target_pred_boot = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Tirage avec remise de la target (bootstrap)
        sampled_true = np.random.choice(true_values, size=n, replace=True)
        
        # 1. Corrélation : Target permutée vs Prédiction
        corr_tp, _ = pearsonr(sampled_true, pred_values)
        corr_target_pred_boot[i] = corr_tp
        
        # 2. Corrélation : Target permutée vs Target initiale
        corr_tt, _ = pearsonr(sampled_true, true_values)
        corr_target_target_boot[i] = corr_tt

        # 2. R2
        mse_boot = np.mean((sampled_true - pred_values)**2)
        var_boot = np.var(sampled_true)
        r2_target_pred_boot[i] = 1 - (mse_boot / var_boot) if var_boot > 0 else np.nan

    return original_corr, original_r2, corr_target_pred_boot, corr_target_target_boot, r2_target_pred_boot

def plot_two_bootstrap_histograms(corr_tp_boot, corr_tt_boot, r2_tp_boot, original_corr, original_r2, month_name, pc_idx, member, outdir, freq_label):
    """Trace les deux histogrammes bootstrap avec la ligne de corrélation mesurée."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # --- Plot 1 : Target permutée vs Prédiction ---
    axes[0].hist(corr_tp_boot, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(original_corr, color='red', linestyle='dashed', linewidth=2, label=f'Mesure ({original_corr:.2f})')
    axes[0].set_title('Target permutée vs Prédiction', fontsize=14)
    axes[0].set_xlabel('Corrélation de Pearson')
    axes[0].set_ylabel('Fréquence')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.5)
    
    # --- Plot 2 : Target permutée vs Target initiale ---
    axes[1].hist(corr_tt_boot, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(original_corr, color='red', linestyle='dashed', linewidth=2, label=f'Mesure ({original_corr:.2f})')
    axes[1].set_title('Target permutée vs Target initiale', fontsize=14)
    axes[1].set_xlabel('Corrélation de Pearson')
    axes[1].set_ylabel('Fréquence')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.5)

    # --- Plot 3 : R2 (Permutée vs Pred) ---
    # On filtre les éventuels NaN générés par des variances nulles
    valid_r2_boot = r2_tp_boot[~np.isnan(r2_tp_boot)]
    axes[2].hist(valid_r2_boot, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    axes[2].axvline(original_r2, color='red', linestyle='dashed', linewidth=2, label=f'Mesure ({original_r2:.2f})')
    axes[2].set_title('R²: Target permutée vs Prédiction', fontsize=12)
    axes[2].set_xlabel('R²')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.5)
    
    plt.suptitle(f'Distributions Bootstrap - PC {pc_idx} - {month_name} - Membre {member}', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(outdir, f'Hist_Boot_PC{pc_idx}_{month_name}_{freq_label}_Member_{member}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def bootstrap_correlation(true_values, pred_values, n_iterations): 
    """Computes a bootstrap-based p-value for the Pearson correlation."""
    # Ensure inputs are 1D numpy arrays
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
        # r2_scores[i] = 1 - (mse_boot / var_boot) if var_boot > 0 else np.nan
        r2_scores[i] = 1 - (mse_boot / var_orig) # plutôt que d'utiliser var_boot? 
        
    # Test unilatéral : la probabilité que le bruit fasse mieux ou aussi bien que notre R2
    return np.mean(r2_scores >= original_r2)

def stats(pcs_true, pcs_pred, n_iterations):
    """Computes ACC, nRMSE, R2 and their respective significance."""
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
    
    corr = xr.corr(pcs_true, pcs_pred, dim='time')
    std_true = pcs_true.std(dim='time')
    rmse = np.sqrt(((pcs_true - pcs_pred)**2).mean(dim='time'))
    nrmse = rmse / std_true
    r2 = 1 - nrmse**2 # Déduction pure du R2 depuis la nRMSE

    return corr, rmse, nrmse, r2, pval_corr, pval_r2

def plot_time_series(pcs_true_dict, pcs_pred_dict, stats_dict, months_num, member, outdir, pc_idx, freq_label, quantiles_pred_dict=None):
    """Plot the predicted and targeted index time series dynamically for any number of months."""
    n_months = len(months_num)
    fig_width = 25 if freq_label == "Daily" else 10
    
    # =======================================================
    # 1. Figure Principale (Prédictions vs Vrai)
    # Hauteur augmentée (5 * n_months) pour aérer les subplots
    # =======================================================
    fig, axes = plt.subplots(n_months, 1, figsize=(fig_width, 5 * n_months))
    if n_months == 1:
        axes = [axes] 

    # =======================================================
    # 2. Figure Secondaire (Résidus)
    # =======================================================
    fig_res, axes_res = plt.subplots(n_months, 1, figsize=(fig_width, 5 * n_months))
    if n_months == 1:
        axes_res = [axes_res]
        
    for i, m in enumerate(months_num):
        ax = axes[i]
        ax_res = axes_res[i] 
        
        true_m = pcs_true_dict[m]
        pred_m = pcs_pred_dict[m]
        corr, rmse, nrmse, r2, pval_corr, pval_r2 = stats_dict[m]
        
        p_c = float(pval_corr.values) if not np.isnan(pval_corr.values) else np.nan
        p_r = float(pval_r2.values) if not np.isnan(pval_r2.values) else np.nan
        
        sign_c = "**" if p_c < 0.05 else ("*" if p_c < 0.1 else "")
        sign_r = "**" if p_r < 0.05 else ("*" if p_r < 0.1 else "")
        month_name = calendar.month_abbr[m] 
        
        title_str = f"{month_name} - RMSE={rmse.values:.2f}, nRMSE={nrmse.values:.2f}, R²={r2.values:.2f}{sign_r} (p={p_r:.3f}), r={corr.values:.2f}{sign_c} (p={p_c:.3f})"

        std_true = np.std(true_m)
        if std_true == 0: std_true = 1 
        
        # Axe X régulier pour éviter les "trous"
        x_idx = np.arange(len(true_m.time.values))
        x_labels = pd.to_datetime(true_m.time.values).strftime('%Y-%m-%d').tolist()

        lw = 0.4 if freq_label == "Daily" else 1.5
        ms = 1 if freq_label == "Daily" else 6

        # Tracé des quantiles en dégradé (Fan Chart)
        if quantiles_pred_dict and m in quantiles_pred_dict and len(quantiles_pred_dict[m]) > 0:
            q_keys = sorted(list(quantiles_pred_dict[m].keys()))
            
            q_lower = sorted([q for q in q_keys if q < 0.5], reverse=True)
            q_upper = sorted([q for q in q_keys if q > 0.5])
            
            n_bands = min(len(q_lower), len(q_upper))
            
            if n_bands > 0:
                for idx in reversed(range(n_bands)):
                    ql = q_lower[idx]
                    qu = q_upper[idx]
                    
                    label_str = f"Quantiles ({min(q_lower)}-{max(q_upper)})" if (idx == n_bands - 1 and i == 0) else ""
                    
                    ax.fill_between(
                        x_idx, 
                        quantiles_pred_dict[m][ql]/std_true, 
                        quantiles_pred_dict[m][qu]/std_true, 
                        color='tab:blue', 
                        alpha=0.15, 
                        linewidth=0.0,
                        label=label_str
                    )

        # Plot des lignes True et Median
        ax.plot(x_idx, true_m/std_true, color="black", marker='.', linewidth=lw, markersize=ms, label="True" if i==0 else "")
        label = "Predicted (Median)" if quantiles_pred_dict else "Predicted"
        ax.plot(x_idx, pred_m/std_true, color="navy", marker='.', linewidth=lw, markersize=ms, label=label if i==0 else "")
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel(f"PC {pc_idx}", fontsize=14)
        ax.set_title(title_str, fontsize=16)
        ax.grid(True)
        
        n_ticks = min(12, len(x_idx))
        tick_indices = np.linspace(0, len(x_idx) - 1, n_ticks, dtype=int)
        
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([x_labels[idx] for idx in tick_indices], rotation=45, ha="right")

        # =======================================================
        # PLOT 2 : RÉSIDUS (Prédiction - Réalité)
        # =======================================================
        residuals = (pred_m - true_m) / std_true
        
        ax_res.plot(x_idx, residuals, color="firebrick", marker='.', linewidth=lw, markersize=ms, label="Résidus (Pred - True)" if i==0 else "")
        ax_res.axhline(y=0, color='black', linestyle='--', linewidth=1.5) # Ligne du zéro
        
        ax_res.set_xlabel("Time", fontsize=14)
        ax_res.set_ylabel(f"Résidus (PC {pc_idx})", fontsize=14)
        ax_res.set_title(f"Résidus : {title_str}", fontsize=16)
        ax_res.grid(True)
        
        ax_res.set_xticks(tick_indices)
        ax_res.set_xticklabels([x_labels[idx] for idx in tick_indices], rotation=45, ha="right")

    # --- CORRECTION DE L'ESPACEMENT (Figure Principale) ---
    top_margin = 0.75 if n_months == 1 else 0.85
    legend_y = 1.00 if n_months == 1 else 0.95

    fig.subplots_adjust(hspace=0.8, top=top_margin)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=3, fontsize=14)
    
    os.makedirs(outdir, exist_ok=True)
    
    save_path = os.path.join(outdir, f'Perf_PC{pc_idx}_{freq_label}_Member_{member}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- CORRECTION DE L'ESPACEMENT ET SAUVEGARDE (Figure Résidus) ---
    fig_res.subplots_adjust(hspace=0.8, top=top_margin)
    fig_res.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=1, fontsize=14)
    save_path_res = os.path.join(outdir, f'Residuals_PC{pc_idx}_{freq_label}_Member_{member}.png')
    fig_res.savefig(save_path_res, dpi=300, bbox_inches='tight')
    plt.close(fig_res)


# ============================================================
# MAIN EVALUATION SCRIPT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='vae')
    parser.add_argument('--vit_dir', type=str, required=True, help='Chemin du dossier contenant le modèle entraîné (final_model_ViT...)')
    parser.add_argument('--model_type', type=str, choices=['best', 'final'], default='best', help='Évaluer le meilleur modèle (best) ou le dernier (final)')
    parser.add_argument('--embed_path', type=str, required=True, help='Chemin du dossier contenant le modèle d\'embedding (pca_model.joblib ou vae_model.pth)')
    parser.add_argument('--nb_members_train', type=int, default=10)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_members_test', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--monthly_mean', action='store_true', help='Appliquer une moyenne mensuelle. Si absent, garde la résolution journalière.')
    parser.add_argument('--n_bootstraps', type=int, default=1000, help='Itérations pour la p-value de corrélation')
    parser.add_argument('--use_lags_attention', action='store_true', help='Activer l\'attention temporelle entre les lags (spécifique au ViT)')
    
    # --- NOUVEAUX ARGUMENTS POUR LES QUANTILES ---
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile','correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données mensuelles (_1mo.nc)')
    parser.add_argument('--lat_weight', action='store_true', help='Pondération spatiale sqrt(cos(lat))')

    args = parser.parse_args()

    if args.loss_type == 'quantile' and 0.5 not in args.quantiles:
        raise ValueError("Erreur: 0.5 doit être inclus dans la liste des quantiles pour extraire la médiane.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = time.time()

    # ============================================================
    # 1. SETUP DATASET & MEMBERS
    # ============================================================
    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)
    val_early_members = all_members[-args.nb_members_val:]
    test_members = all_members[args.nb_members_train:args.nb_members_train + args.nb_members_test]
    val_members = val_early_members + test_members

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)

    dynamic_slp_std = 596.0
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))
            print(f"✅ slp_std extrait du chemin PCA/VAE : {dynamic_slp_std}")
        else:
            print(f"⚠️ 'slp_std' introuvable dans le chemin. Utilisation du fallback : {dynamic_slp_std}")

    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)


    # ============================================================
    # PRÉPARATION DES POIDS SPATIAUX (POUR PROJECTION CIBLE PCA)
    # ============================================================
    wgts_flat = None
    if args.lat_weight and args.embed_method == 'pca':
        if args.machine == 'hacienda':
            base_data = "/home/moysan/stage_isir_jz/data/"
        elif args.machine in ['jean-zay-work', 'jean-zay-scratch']:
            base_data = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data/"
        elif args.machine == 'mac_local':
            base_data = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data/"
            
        sample_member = val_members[0]
        sample_path = os.path.join(base_data, f"SLP/PSL_anom_LE2-{sample_member}_1mo.nc")
        
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            h, w = len(lats), len(ds_sample['lon'].values)
            wgts = np.sqrt(coslat).reshape(h, 1)
            wgts_flat = np.broadcast_to(wgts, (h, w)).flatten()
            ds_sample.close()
            print("✅ Grille de poids de latitude chargée pour l'évaluation PCA.")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la grille de latitude : {e}")
    # ============================================================
    # 2. LOAD MODELS
    # ============================================================
    pca_model, vae_model = None, None
    if args.embed_method == 'pca':
        pca_path = args.embed_path
        pca_model = joblib.load(pca_path)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=args.latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

    # --- MODIFICATION: Taille dynamique de sortie ---
    out_features = args.latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else args.latent_dim
    active_sst_lags = args.sst_lags_months if args.monthly_reduction else args.sst_lags_days
    active_slp_lags = args.slp_lags_months if args.monthly_reduction else args.slp_lags_days

    # Initialisation spécifique au ViT
    model = ViT_Latent_SLP_Multimodal(
        sst_size=(85, 360), 
        slp_size=(53, 113), 
        patch_size_sst=(5, 10), 
        patch_size_slp=(5, 10), 
        in_chans_sst=len(active_sst_lags), 
        in_chans_slp=len(active_slp_lags), 
        embed_dim=128, 
        depth=4, 
        num_heads=4, 
        dr=0., # Dropout inactif en inférence
        nb_out=out_features, 
        use_lags_attention=args.use_lags_attention
    ).to(device)

    # Dummy forward pour vérifier/initialiser les dimensions si besoin
    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(active_sst_lags), 85, 360).to(device) if len(active_sst_lags) > 0 else None
        dummy_slp = torch.zeros(1, len(active_slp_lags), 53, 113).to(device) if len(active_slp_lags) > 0 else None
        _ = model(dummy_sst, dummy_slp)

    # Chargement dynamique selon le type (best vs final)
    if args.model_type == 'best':
        vit_path = os.path.join(args.vit_dir, f"best_val_ViT_bs{args.bs}.pth")
        if not os.path.exists(vit_path):
            print("⚠️ Fichier best_val introuvable. Fallback sur le final_model.")
            vit_path = os.path.join(args.vit_dir, f"final_model_ViT_bs{args.bs}.pth")
    elif args.model_type == 'final':
        vit_path = os.path.join(args.vit_dir, f"final_model_ViT_bs{args.bs}.pth")
        if not os.path.exists(vit_path):
            print("⚠️ Fichier final_model introuvable. Fallback sur le best_val.")
            vit_path = os.path.join(args.vit_dir, f"best_val_ViT_bs{args.bs}.pth")
        
    checkpoint = torch.load(vit_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Models loaded successfully from {vit_path}.")

    # ============================================================
    # 3. INFERENCE LOOP
    # ============================================================
    dates_list, members_list = [], []
    preds_list, trues_list = [], []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(valloader):
            X_sst = X_sst.to(device)
            X_slp = X_slp.to(device)
            
            if args.embed_method == 'pca':
                slp_flat = y_target.view(y_target.size(0), -1).numpy()
                if wgts_flat is not None:
                    slp_flat = slp_flat * wgts_flat
                target_embed = pca_model.transform(slp_flat)[:, :args.latent_dim]
            elif args.embed_method == 'vae':
                y_target = y_target.to(device)
                target_tensor, _ = vae_model.encode(y_target)
                target_embed = target_tensor.cpu().numpy()

            predicted_latent = model(X_sst, X_slp).cpu().numpy()

            preds_list.append(predicted_latent)
            trues_list.append(target_embed)
            dates_list.extend([str(d) for d in dates])
            
            for m in members:
                m_str = m if isinstance(m, str) else (m.item().decode() if isinstance(m.item(), bytes) else str(m.item()))
                members_list.append(m_str)

    preds_arr = np.concatenate(preds_list, axis=0)
    trues_arr = np.concatenate(trues_list, axis=0)

    # --- NOUVEAU : Transformation pour la Modalité Quantile ---
    if args.loss_type == 'quantile':
        # (N, latent_dim * n_quantiles) -> (N, latent_dim, n_quantiles)
        preds_arr = preds_arr.reshape(-1, args.latent_dim, len(args.quantiles))

    df = pd.DataFrame({'time': pd.to_datetime(dates_list), 'member': members_list})
    for i in range(args.latent_dim):
        df[f'true_pc{i+1}'] = trues_arr[:, i]
        
        if args.loss_type == 'quantile':
            # Extraction de la médiane et de tous les quantiles
            median_idx = args.quantiles.index(0.5)
            df[f'pred_pc{i+1}'] = preds_arr[:, i, median_idx]  # La médiane sert de PC principale pour les calculs d'erreurs
            for q_idx, q in enumerate(args.quantiles):
                df[f'pred_pc{i+1}_q{q}'] = preds_arr[:, i, q_idx]
        else:
            df[f'pred_pc{i+1}'] = preds_arr[:, i]

    # ============================================================
    # 4. STATISTIQUES & PLOTS PAR MEMBRE
    # ============================================================
    freq_label = "Monthly" if args.monthly_mean else "Daily"
    print(f"\nTime Series Frequency Set To: {freq_label}")
    
    unique_members = df['member'].unique()
    max_pcs_to_plot = min(2, args.latent_dim)

    for member in unique_members:
        print(f"\n{'='*40}")
        print(f"Evaluating Validation Member: {member}")
        print(f"{'='*40}")

        df_member = df[df['member'] == member].copy()
        ds_member = df_member.set_index('time').to_xarray()

        # Séparation dans les dossiers 'val' ou 'test'
        if member in val_early_members:
            key = str(member) + '_val'
        else:
            key = str(member) + '_test'
        
        member_outdir = os.path.join(args.vit_dir, f"evaluation_plots_{args.model_type}_model", key)
        os.makedirs(member_outdir, exist_ok=True)

        for pc_idx in range(1, max_pcs_to_plot + 1):
            print(f"--- PC {pc_idx} ---")
            
            if args.monthly_mean:
                pred_series = ds_member[f'pred_pc{pc_idx}'].resample(time='1M').mean().dropna(dim="time")
                true_series = ds_member[f'true_pc{pc_idx}'].resample(time='1M').mean().dropna(dim="time")
            else:
                pred_series = ds_member[f'pred_pc{pc_idx}'].dropna(dim="time")
                true_series = ds_member[f'true_pc{pc_idx}'].dropna(dim="time")

            pcs_true_dict, pcs_pred_dict, stats_dict,quantiles_pred_dict = {}, {}, {}, {}

            for m in args.winter_months:
                true_m = true_series.where(true_series.time.dt.month == m, drop=True)
                pred_m = pred_series.where(pred_series.time.dt.month == m, drop=True)
                
                if len(true_m) < 2:
                    print(f"  Month {m}: Not enough valid dates to compute statistics.")
                    continue
                    
                s = stats(true_m, pred_m, args.n_bootstraps)
                
                pcs_true_dict[m] = true_m
                pcs_pred_dict[m] = pred_m
                stats_dict[m] = s

                # --- NOUVEAU : Préparation des séries temporelles des quantiles ---
                if args.loss_type == 'quantile':
                    quantiles_pred_dict[m] = {}
                    for q in args.quantiles:
                        if q == 0.5: continue
                        if args.monthly_mean:
                            q_series = ds_member[f'pred_pc{pc_idx}_q{q}'].resample(time='1M').mean().dropna(dim="time")
                        else:
                            q_series = ds_member[f'pred_pc{pc_idx}_q{q}'].dropna(dim="time")
                        quantiles_pred_dict[m][q] = q_series.where(q_series.time.dt.month == m, drop=True)
                
                month_name = calendar.month_abbr[m]
                print(f"  {month_name}: ACC={s[0].values:.3f}, RMSE={s[1].values:.3f}, nRMSE={s[2].values:.3f}, R2={s[3].values:.3f}, p-val={s[4].values:.3f}")
            
                # ============================================================
                # NOUVEAU : Calcul et tracé des deux histogrammes bootstrap
                # ============================================================
                orig_corr, orig_r2, corr_tp_boot, corr_tt_boot, r2_tp_boot = compute_two_bootstraps(
                    true_m.values, 
                    pred_m.values, 
                    args.n_bootstraps
                )
                
                if not np.isnan(orig_corr):
                    plot_two_bootstrap_histograms(
                        corr_tp_boot, 
                        corr_tt_boot, 
                        r2_tp_boot,       # <-- NOUVEAU
                        orig_corr, 
                        orig_r2,          # <-- NOUVEAU
                        month_name, 
                        pc_idx, 
                        member, 
                        member_outdir, 
                        freq_label
                    )
            
            # Plot pour cette composante principale et ce membre
            if stats_dict: 
                valid_months = [m for m in args.winter_months if m in stats_dict]
                plot_time_series(
                    pcs_true_dict=pcs_true_dict, 
                    pcs_pred_dict=pcs_pred_dict, 
                    stats_dict=stats_dict, 
                    months_num=valid_months, 
                    member=member, 
                    outdir=member_outdir,
                    pc_idx=pc_idx,
                    freq_label=freq_label,
                    quantiles_pred_dict=quantiles_pred_dict if args.loss_type == 'quantile' else None # <-- L'ARGUMENT MANQUANT
                )



    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Evaluation Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("\nÉvaluation terminée avec succès !")