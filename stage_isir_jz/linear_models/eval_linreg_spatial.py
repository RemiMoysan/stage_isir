import os
import argparse
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt
import time
import calendar
import torch.nn as nn
import re

import sys
from pathlib import Path

# Setup des chemins d'importation vers la racine
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

from shared_tools.datasets import Dataset, Dataset_mensuel
from shared_tools.models import ConvVAE
from shared_tools.optuna_loop_helpers import encode_to_latent_gpu
# Importation stricte des fonctions d'évaluation et de tracé cartographique du CNN
from shared_tools.evaluation_functions import (
    plot_spatial_timeseries_raw_metrics, 
    plot_metric_with_pvalue_map, 
    compute_map_metrics_and_bootstraps
)

# ============================================================
# ARCHITECTURE DU MODÈLE DE RÉGRESSION LINÉAIRE
# ============================================================
class LinearRegressionPredictor(nn.Module):
    def __init__(self, in_features, out_dim=128):
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim)

    def forward(self, x):
        return self.linear(x)

# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca')
    parser.add_argument('--linreg_dir', type=str, required=True, help="Chemin vers le dossier d'entraînement LinReg")
    parser.add_argument('--model_type', type=str, choices=['best', 'final'], default='best')
    
    # --- PCA PATHS ---
    parser.add_argument('--embed_path', type=str, required=True, help="Chemin vers le modèle VAE/PCA SLP")
    parser.add_argument('--embed_path_sst', type=str, default=None, help="Chemin vers le modèle PCA SST")
    
    # --- SPLIT ---
    parser.add_argument('--nb_members_train', type=int, default=10)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_members_test', type=int, default=5)
    parser.add_argument('--force_val_members', type=str, nargs='*', default=None, help='Forcer une liste spécifique de membres pour la val')
    parser.add_argument('--force_test_members', type=str, nargs='*', default=None, help='Forcer une liste spécifique de membres pour le test')
    parser.add_argument('--seed', type=int, default=42)
    
    # --- HYPERPARAMETRES ---
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--monthly_mean', action='store_true')
    parser.add_argument('--monthly_reduction', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--n_bootstraps', type=int, default=1000)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile','correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='*', default=[])
    parser.add_argument('--input_format', type=str, choices=['raw', 'pca'], default='raw')
    parser.add_argument('--sst_pca_dim', type=int, default=0)

    args = parser.parse_args()
    
    if args.loss_type == 'quantile' and 0.5 not in args.quantiles:
        raise ValueError("Erreur: 0.5 doit être inclus dans la liste des quantiles pour extraire la médiane.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================
    # 1. SETUP DATASET & MEMBERS
    # ============================================================
    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    
    import random
    rng = random.Random(args.seed)
    rng.shuffle(all_members)
    if args.force_val_members is not None or args.force_test_members is not None:
        print("⚠️ OVERRIDE ACTIF : Utilisation des listes de membres forcées.")
        val_early_members = args.force_val_members if args.force_val_members else []
        test_members = args.force_test_members if args.force_test_members else []
        remaining = [m for m in all_members if m not in val_early_members and m not in test_members]
        train_members = remaining[:args.nb_members_train]
        nb_members_train = len(train_members)
        nb_members_val = len(val_early_members)
        nb_members_test = len(test_members)
    else:
        nb_members_train = args.nb_members_train
        nb_members_val = args.nb_members_val
        nb_members_test = args.nb_members_test
        train_members = all_members[:nb_members_train]
        val_early_members = all_members[-nb_members_val:]
        test_members = all_members[nb_members_train:nb_members_train + nb_members_test] if nb_members_test > 0 else []

    val_members = val_early_members + test_members
    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1)

    dynamic_slp_std = 596.0
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match: dynamic_slp_std = float(match.group(1))

    dynamic_sst_std = 0.707 
    if args.embed_path_sst:
        match = re.search(r'sst_std([0-9.]+)', args.embed_path_sst)
        if match: dynamic_sst_std = float(match.group(1))

    active_sst_lags = sorted(args.sst_lags_months if args.monthly_reduction else args.sst_lags_days, reverse=True) 
    active_slp_lags = sorted(args.slp_lags_months if args.monthly_reduction else args.slp_lags_days, reverse=True)

    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
        
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # 2. PRÉPARATION DES POIDS SPATIAUX ET MODÈLES (SLP & SST)
    # ============================================================
    coslat_2d = None
    slp_pca_model, vae_model = None, None
    slp_pca_mean_gpu, slp_pca_components_gpu = None, None

    if args.embed_method == 'pca':
        slp_pca_model = joblib.load(args.embed_path)
        slp_pca_mean_gpu = torch.tensor(slp_pca_model.mean_, dtype=torch.float32, device=device)
        slp_pca_components_gpu = torch.tensor(slp_pca_model.components_[:max(args.latent_dim, 1)], dtype=torch.float32, device=device)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=args.latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

    wgts_slp_gpu, wgts_slp_flat, safe_wgts_slp = None, None, None
    if args.lat_weight:
        sample_path = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-1001.001_1mo.nc"      
        try:
            with xr.open_dataset(sample_path) as ds_sample:
                lats = ds_sample['lat'].values
                coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
                h, w = len(lats), len(ds_sample['lon'].values)
                coslat_2d = np.broadcast_to(coslat.reshape(-1, 1), (h, w))
                wgts_slp_flat = np.broadcast_to(np.sqrt(coslat).reshape(h, 1), (h, w)).flatten()
                wgts_slp_gpu = torch.tensor(wgts_slp_flat, dtype=torch.float32, device=device)
                safe_wgts_slp = np.maximum(wgts_slp_flat, 1e-5)
            print("✅ Grille de poids de latitude SLP générée avec succès.")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la grille de latitude SLP : {e}")

    sst_pca_model = None
    sst_pca_mean_gpu, sst_pca_components_gpu = None, None
    if args.input_format == 'pca' and args.embed_path_sst:
        sst_pca_model = joblib.load(args.embed_path_sst)
        sst_pca_mean_gpu = torch.tensor(sst_pca_model.mean_, dtype=torch.float32, device=device)
        sst_pca_components_gpu = torch.tensor(sst_pca_model.components_, dtype=torch.float32, device=device)

    wgts_sst_gpu, wgts_sst_flat = None, None
    if args.lat_weight:
        sample_path_sst = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/SST_anom_LE2-1001.001_T_regrid_1mo.nc"
        try:
            with xr.open_dataset(sample_path_sst) as ds_sample_sst:
                ds_sample_sst = ds_sample_sst.sel(lat=slice(-15, 70))
                coslat_sst = np.cos(np.deg2rad(ds_sample_sst['lat'].values)).clip(0., 1.)
                h_sst, w_sst = len(coslat_sst), len(ds_sample_sst['lon'].values)
                wgts_sst_flat = np.broadcast_to(np.sqrt(coslat_sst).reshape(h_sst, 1), (h_sst, w_sst)).flatten()
                wgts_sst_gpu = torch.tensor(wgts_sst_flat, dtype=torch.float32, device=device)
            print("✅ Grille de poids de latitude SST générée avec succès.")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la grille de latitude SST : {e}")

    # ============================================================
    # 3. INITIALISATION DU MODÈLE DE RÉGRESSION
    # ============================================================
    if args.input_format == 'pca':
        in_features_sst = len(active_sst_lags) * args.sst_pca_dim
        in_features_slp = len(active_slp_lags) * 1
    else:
        in_features_sst = len(active_sst_lags) * 85 * 360
        in_features_slp = len(active_slp_lags) * 53 * 113

    out_features = args.latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else args.latent_dim

    model = LinearRegressionPredictor(in_features=in_features_sst + in_features_slp, out_dim=out_features).to(device)

    if args.model_type == 'best':
        linreg_path = os.path.join(args.linreg_dir, "best_val_LinReg.pth")
        if not os.path.exists(linreg_path):
            linreg_path = os.path.join(args.linreg_dir, f"best_val_LinReg_bs{args.bs}.pth")
    elif args.model_type == 'final':
        linreg_path = os.path.join(args.linreg_dir, "final_model_LinReg.pth")
        if not os.path.exists(linreg_path):
            linreg_path = os.path.join(args.linreg_dir, f"final_model_LinReg_bs{args.bs}.pth")

    try:
        checkpoint = torch.load(linreg_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Models loaded successfully from {linreg_path}.")
    except Exception as e:
        print(f"Erreur de chargement du modèle : {e}")

    model.eval()

    def format_inputs(X_sst, X_slp):
        B, L, H, W = X_sst.shape
        if args.input_format == 'pca':
            X_sst_2d = X_sst.view(B * L, H, W)
            sst_embed = encode_to_latent_gpu(X_sst_2d, 'pca', args.sst_pca_dim, sst_pca_components_gpu[:args.sst_pca_dim], sst_pca_mean_gpu, wgts_sst_gpu, None)
            X_sst_tensor = sst_embed.view(B, L * args.sst_pca_dim)
            if len(active_slp_lags) > 0:
                X_slp_2d = X_slp.view(B * len(active_slp_lags), 53, 113)
                slp_embed_entree = encode_to_latent_gpu(X_slp_2d, 'pca', 1, slp_pca_components_gpu[:1], slp_pca_mean_gpu, wgts_slp_gpu, None)
                X_slp_tensor = slp_embed_entree.view(B, len(active_slp_lags) * 1)
                return torch.cat((X_sst_tensor, X_slp_tensor), dim=1)
            return X_sst_tensor
        else:
            X_sst_tensor = X_sst.view(B, -1)
            if len(active_slp_lags) > 0:
                X_slp_tensor = X_slp.view(B, -1)
                return torch.cat((X_sst_tensor, X_slp_tensor), dim=1)
            return X_sst_tensor

    # ============================================================
    # 4. INFERENCE & SPATIAL DECODING
    # ============================================================
    preds_list, targets_list, recs_list = [], [], []
    dates_list, members_list = [], []

    print("Running inference and decoding maps...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(valloader):
            X_sst = X_sst.to(device)
            if len(active_slp_lags) > 0:
                X_slp = X_slp.to(device)
            
            y_target_np = y_target.numpy() 
            if len(y_target_np.shape) == 3:
                y_target_np = np.expand_dims(y_target_np, axis=1)
            B, C, H, W = y_target_np.shape

            X_combined = format_inputs(X_sst, X_slp)
            predicted_raw = model(X_combined).cpu().numpy()
            
            if args.loss_type == 'quantile':
                predicted_raw = predicted_raw.reshape(B, args.latent_dim, len(args.quantiles))
                pred_latent = predicted_raw[:, :, args.quantiles.index(0.5)]
            else:
                pred_latent = predicted_raw

            if args.embed_method == 'pca':
                slp_flat = y_target_np.reshape(B, -1)
                if args.lat_weight and safe_wgts_slp is not None:
                    slp_flat = slp_flat * wgts_slp_flat
                true_latent = slp_pca_model.transform(slp_flat)[:, :args.latent_dim]
                
                pca_expected_dim = slp_pca_model.n_components_
                if args.latent_dim < pca_expected_dim:
                    pad_width = pca_expected_dim - args.latent_dim
                    pred_latent_padded = np.pad(pred_latent, ((0, 0), (0, pad_width)), mode='constant')
                    true_latent_padded = np.pad(true_latent, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    pred_latent_padded = pred_latent
                    true_latent_padded = true_latent
                
                pred_map_flat = slp_pca_model.inverse_transform(pred_latent_padded)
                rec_map_flat = slp_pca_model.inverse_transform(true_latent_padded)
                if args.lat_weight and safe_wgts_slp is not None:
                    pred_map_flat = pred_map_flat / safe_wgts_slp
                    rec_map_flat = rec_map_flat / safe_wgts_slp
                
                pred_map = pred_map_flat.reshape(B, C, H, W)
                rec_map = rec_map_flat.reshape(B, C, H, W)
            elif args.embed_method == 'vae':
                y_target_tensor = torch.tensor(y_target_np, dtype=torch.float32).to(device)
                true_latent_tensor, _ = vae_model.encode(y_target_tensor)
                rec_map = vae_model.decode(true_latent_tensor).cpu().numpy()
                pred_map = vae_model.decode(torch.tensor(pred_latent, dtype=torch.float32).to(device)).cpu().numpy()

            preds_list.append(pred_map)
            targets_list.append(y_target_np)
            recs_list.append(rec_map)
            dates_list.extend([str(d) for d in dates])
            for m in members:
                members_list.append(m if isinstance(m, str) else (m.item().decode() if isinstance(m.item(), bytes) else str(m.item())))
                
    print(f"Decoding finished in {time.time() - start_time:.2f} seconds.")

    # ============================================================
    # 5. ÉVALUATION SPATIALE PAR MEMBRE ET PAR MOIS
    # ============================================================
    preds_arr = np.concatenate(preds_list, axis=0).squeeze()
    targets_arr = np.concatenate(targets_list, axis=0).squeeze()
    recs_arr = np.concatenate(recs_list, axis=0).squeeze()
    dates_arr = pd.to_datetime(dates_list); members_arr = np.array(members_list)
    
    freq_label = "Monthly" if args.monthly_mean else "Daily"
    unique_members = np.unique(members_arr)
    
    if len(targets_arr.shape) == 2:
        H, W = targets_arr.shape
    else:
        _, H, W = targets_arr.shape
        
    spatial_weights = coslat_2d / coslat_2d.sum() if (args.lat_weight and coslat_2d is not None) else np.ones((H, W)) / (H * W)

    for member in unique_members:
        print(f"\n{'='*40}\nEvaluating Spatial Member: {member}\n{'='*40}")
        split_name = 'val' if member in val_early_members else 'test'
        member_base_dir = os.path.join(args.linreg_dir, f"spatial_eval_{args.model_type}", f"{member}_{split_name}", freq_label)

        mask_mem = (members_arr == member)
        ds_member = xr.Dataset(
            {"pred": (["time", "h", "w"], preds_arr[mask_mem]),
             "target": (["time", "h", "w"], targets_arr[mask_mem]),
             "rec": (["time", "h", "w"], recs_arr[mask_mem])},
            coords={"time": dates_arr[mask_mem]}
        )
        if args.monthly_mean: ds_member = ds_member.resample(time='1M').mean().dropna(dim="time")

        for m in args.winter_months:
            month_name = calendar.month_name[m]
            print(f"  --- Month: {month_name} ---")
            
            ds_month = ds_member.where(ds_member["time"].dt.month == m, drop=True)
            if ds_month.sizes["time"] < 2: continue

            month_outdir = os.path.join(member_base_dir, f"month_{m}_{calendar.month_abbr[m]}")
            os.makedirs(month_outdir, exist_ok=True)

            p_m = ds_month["pred"].values
            t_m = ds_month["target"].values
            r_m = ds_month["rec"].values

            # 1. Calcul des cartes & des métriques spatiales bootstrapées
            r2_map, pval_r2, l1_map, pval_l1, corr_map, pval_corr, stats_true = compute_map_metrics_and_bootstraps(t_m, p_m, spatial_weights, n_bootstraps=min(300, args.n_bootstraps))
            r2_map_r, pval_r2_r, l1_map_r, pval_l1_r, corr_map_r, pval_corr_r, stats_rec = compute_map_metrics_and_bootstraps(r_m, p_m, spatial_weights, n_bootstraps=min(300, args.n_bootstraps))

            # 2. Séries temporelles brutes
            mse_t_true = np.sum((p_m - t_m)**2 * spatial_weights, axis=(1, 2)); var_t_true = np.sum(t_m**2 * spatial_weights, axis=(1, 2))
            mse_t_rec  = np.sum((p_m - r_m)**2 * spatial_weights, axis=(1, 2)); var_t_rec  = np.sum(r_m**2 * spatial_weights, axis=(1, 2))
            mae_t_true = np.sum(np.abs(p_m - t_m) * spatial_weights, axis=(1, 2)); ref_mae_true = np.sum(np.abs(t_m) * spatial_weights, axis=(1, 2))
            mae_t_rec  = np.sum(np.abs(p_m - r_m) * spatial_weights, axis=(1, 2)); ref_mae_rec  = np.sum(np.abs(r_m) * spatial_weights, axis=(1, 2))

            p_sub = p_m - np.sum(p_m * spatial_weights, axis=(1, 2), keepdims=True)
            t_sub = t_m - np.sum(t_m * spatial_weights, axis=(1, 2), keepdims=True)
            r_sub = r_m - np.sum(r_m * spatial_weights, axis=(1, 2), keepdims=True)
            cov_sp_true = np.sum(p_sub * t_sub * spatial_weights, axis=(1, 2)); std_p = np.sqrt(np.sum(p_sub**2 * spatial_weights, axis=(1, 2)) + 1e-8)
            std_t = np.sqrt(np.sum(t_sub**2 * spatial_weights, axis=(1, 2)) + 1e-8); std_r = np.sqrt(np.sum(r_sub**2 * spatial_weights, axis=(1, 2)) + 1e-8)
            cov_sp_rec  = np.sum(p_sub * r_sub * spatial_weights, axis=(1, 2))

            df_month_ts = pd.DataFrame({
                "time": ds_month["time"].values,
                "mse_true": mse_t_true, "mse_rec": mse_t_rec, "base_var_true": var_t_true, "base_var_rec": var_t_rec,
                "mae_true": mae_t_true, "mae_rec": mae_t_rec, "base_mae_true": ref_mae_true, "base_mae_rec": ref_mae_rec,
                "spatial_corr_true": cov_sp_true / (std_p * std_t), "spatial_corr_rec": cov_sp_rec / (std_p * std_r)
            })
            
            plot_spatial_timeseries_raw_metrics(df_month_ts, member, month_outdir, freq_label, stats_true, stats_rec)

            # 3. Tracé des cartes
            plot_metric_with_pvalue_map(r2_map, pval_r2, month_outdir, f"Map_R2_vs_True_{calendar.month_abbr[m]}", "l2", f"L2 Skill Score vs True Target - {month_name}", global_stat=stats_true['global_r2'], mean_pixel_stat=stats_true['mean_pixel_r2'])
            plot_metric_with_pvalue_map(l1_map, pval_l1, month_outdir, f"Map_L1_vs_True_{calendar.month_abbr[m]}", "l1", f"L1 Skill Score vs True Target - {month_name}", global_stat=stats_true['global_l1'], mean_pixel_stat=stats_true['mean_pixel_l1'])
            plot_metric_with_pvalue_map(corr_map, pval_corr, month_outdir, f"Map_Corr_vs_True_{calendar.month_abbr[m]}", "corr", f"Temporal Correlation vs True Target - {month_name}", global_stat=stats_true['global_corr'], mean_pixel_stat=stats_true['mean_pixel_corr'])

            plot_metric_with_pvalue_map(r2_map_r, pval_r2_r, month_outdir, f"Map_R2_vs_Rec_{calendar.month_abbr[m]}", "l2", f"L2 Skill Score vs Reconstructed - {month_name}", global_stat=stats_rec['global_r2'], mean_pixel_stat=stats_rec['mean_pixel_r2'])
            plot_metric_with_pvalue_map(l1_map_r, pval_l1_r, month_outdir, f"Map_L1_vs_Rec_{calendar.month_abbr[m]}", "l1", f"L1 Skill Score vs Reconstructed - {month_name}", global_stat=stats_rec['global_l1'], mean_pixel_stat=stats_rec['mean_pixel_l1'])
            plot_metric_with_pvalue_map(corr_map_r, pval_corr_r, month_outdir, f"Map_Corr_vs_Rec_{calendar.month_abbr[m]}", "corr", f"Temporal Correlation vs Reconstructed - {month_name}", global_stat=stats_rec['global_corr'], mean_pixel_stat=stats_rec['mean_pixel_corr'])

    print(f"\n✅ Évaluation spatiale par membre terminée avec succès ! Tous les graphiques sont dans : {args.linreg_dir}/spatial_eval_{args.model_type}")