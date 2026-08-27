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

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

from shared_tools.datasets import Dataset, Dataset_mensuel
from shared_tools.models import ConvVAE
from shared_tools.optuna_loop_helpers import encode_to_latent_gpu
from shared_tools.evaluation_functions import compute_latent_metrics_and_bootstraps, plot_combined_pcs_time_series, plot_latent_timeseries_raw_metrics 

# ============================================================
# ARCHITECTURE ADAPTÉE (Entrée 1D générique)
# ============================================================
class LinearRegressionPredictor(nn.Module):
    def __init__(self, in_features, out_dim=128):
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim)

    def forward(self, x):
        return self.linear(x)

# ============================================================
# MAIN EVALUATION SCRIPT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca')
    parser.add_argument('--linreg_dir', type=str, required=True, help='Chemin du dossier contenant le modèle entraîné de régression linéaire')
    parser.add_argument('--model_type', type=str, choices=['best', 'final'], default='best', help='Évaluer le meilleur (best) ou le dernier (final)')
    parser.add_argument('--embed_path', type=str, required=True, help='Chemin du dossier contenant le modèle d\'embedding SLP')
    parser.add_argument('--embed_path_sst', type=str, default=None, help='Chemin du dossier contenant le modèle d\'embedding SST')
    parser.add_argument('--nb_members_train', type=int, default=10)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_members_test', type=int, default=5)
    parser.add_argument('--force_val_members', type=str, nargs='*', default=None, help='Forcer une liste spécifique de membres pour la val')
    parser.add_argument('--force_test_members', type=str, nargs='*', default=None, help='Forcer une liste spécifique de membres pour le test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--monthly_reduction', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--monthly_mean', action='store_true', help='Appliquer une moyenne mensuelle.')
    parser.add_argument('--n_bootstraps', type=int, default=1000)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile', 'correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='*', default=[])
    parser.add_argument('--input_format', type=str, choices=['raw', 'pca'], default='raw')
    parser.add_argument('--sst_pca_dim', type=int, default=0)
    
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
    if args.force_val_members is not None or args.force_test_members is not None:
        print("⚠️ OVERRIDE ACTIF : Utilisation des listes de membres forcées.")
        val_early_members = args.force_val_members if args.force_val_members else []
        test_members = args.force_test_members if args.force_test_members else []
        remaining = [m for m in all_members if m not in val_early_members and m not in test_members]
        train_members = remaining[:args.nb_members_train]
    else:
        train_members = all_members[:args.nb_members_train]
        val_early_members = all_members[-args.nb_members_val:]
        test_members = all_members[args.nb_members_train:args.nb_members_train + args.nb_members_test] if args.nb_members_test > 0 else []
        
    val_members = val_early_members + test_members

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)

    dynamic_slp_std = 596.0
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match: dynamic_slp_std = float(match.group(1))

    dynamic_sst_std = 0.707 
    if args.embed_path_sst:
        match = re.search(r'sst_std([0-9.]+)', args.embed_path_sst)
        if match: dynamic_sst_std = float(match.group(1))

    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
        active_sst_lags = args.sst_lags_days
        active_slp_lags = args.slp_lags_days
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
        active_sst_lags = args.sst_lags_months
        active_slp_lags = args.slp_lags_months
        
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # 2. PRÉPARATION DES POIDS SPATIAUX ET DES MODÈLES PCA
    # ============================================================
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

    wgts_slp_gpu = None
    if args.lat_weight:
        sample_path = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-1001.001_1mo.nc"
        try:
            with xr.open_dataset(sample_path) as ds_sample:
                coslat = np.cos(np.deg2rad(ds_sample['lat'].values)).clip(0., 1.)
                h, w = len(coslat), len(ds_sample['lon'].values)
                wgts_slp_flat = np.broadcast_to(np.sqrt(coslat).reshape(h, 1), (h, w)).flatten()
                wgts_slp_gpu = torch.tensor(wgts_slp_flat, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"⚠️ Erreur latitude SLP : {e}")

    sst_pca_model = None
    sst_pca_mean_gpu, sst_pca_components_gpu = None, None
    if args.input_format == 'pca' and args.embed_path_sst:
        sst_pca_model = joblib.load(args.embed_path_sst)
        sst_pca_mean_gpu = torch.tensor(sst_pca_model.mean_, dtype=torch.float32, device=device)
        sst_pca_components_gpu = torch.tensor(sst_pca_model.components_, dtype=torch.float32, device=device)

    wgts_sst_gpu = None
    if args.lat_weight:
        sample_path_sst = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/SST_anom_LE2-1001.001_T_regrid_1mo.nc"
        try:
            with xr.open_dataset(sample_path_sst) as ds_sample_sst:
                ds_sample_sst = ds_sample_sst.sel(lat=slice(-15, 70))
                coslat_sst = np.cos(np.deg2rad(ds_sample_sst['lat'].values)).clip(0., 1.)
                h_sst, w_sst = len(coslat_sst), len(ds_sample_sst['lon'].values)
                wgts_sst_flat = np.broadcast_to(np.sqrt(coslat_sst).reshape(h_sst, 1), (h_sst, w_sst)).flatten()
                wgts_sst_gpu = torch.tensor(wgts_sst_flat, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"⚠️ Erreur latitude SST : {e}")

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
        linreg_path = os.path.join(args.linreg_dir, f"best_model_LinReg.pth")
        if not os.path.exists(linreg_path):
            linreg_path = os.path.join(args.linreg_dir, f"best_val_Linreg_bs{args.bs}.pth")
        if not os.path.exists(linreg_path):
            linreg_path = os.path.join(args.linreg_dir, f"best_val_LinReg.pth")
    elif args.model_type == 'final':
        linreg_path = os.path.join(args.linreg_dir, f"final_model_LinReg.pth")
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
    # 4. INFERENCE LOOP
    # ============================================================
    dates_list, members_list = [], []
    preds_list, trues_list = [], []

    print("Running inference on validation & test set...")
    with torch.no_grad():
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(valloader):
            X_sst = X_sst.to(device)
            if len(active_slp_lags) > 0:
                X_slp = X_slp.to(device)
            
            X_combined = format_inputs(X_sst, X_slp)
            
            if args.embed_method == 'pca':
                target_embed_tensor = encode_to_latent_gpu(y_target.to(device), 'pca', args.latent_dim, slp_pca_components_gpu, slp_pca_mean_gpu, wgts_slp_gpu, None)
                target_embed = target_embed_tensor.cpu().numpy()
            elif args.embed_method == 'vae':
                target_tensor, _ = vae_model.encode(y_target.to(device))
                target_embed = target_tensor.cpu().numpy()

            predicted_latent = model(X_combined).cpu().numpy()

            preds_list.append(predicted_latent)
            trues_list.append(target_embed)
            dates_list.extend([str(d) for d in dates])
            
            for m in members:
                m_str = m if isinstance(m, str) else (m.item().decode() if isinstance(m.item(), bytes) else str(m.item()))
                members_list.append(m_str)

    preds_arr = np.concatenate(preds_list, axis=0)
    trues_arr = np.concatenate(trues_list, axis=0)

    if args.loss_type == 'quantile':
        preds_arr = preds_arr.reshape(-1, args.latent_dim, len(args.quantiles))

    df = pd.DataFrame({'time': pd.to_datetime(dates_list), 'member': members_list})
    for i in range(args.latent_dim):
        df[f'true_pc{i+1}'] = trues_arr[:, i]
        if args.loss_type == 'quantile':
            median_idx = args.quantiles.index(0.5)
            df[f'pred_pc{i+1}'] = preds_arr[:, i, median_idx] 
            for q_idx, q in enumerate(args.quantiles):
                df[f'pred_pc{i+1}_q{q}'] = preds_arr[:, i, q_idx]
        else:
            df[f'pred_pc{i+1}'] = preds_arr[:, i]

    # ============================================================
    # 5. STATISTIQUES & PLOTS PAR MEMBRE ET PAR MOIS
    # ============================================================
    freq_label = "Monthly" if args.monthly_mean else "Daily"
    print(f"\nTime Series Frequency Set To: {freq_label}")
    
    unique_members = df['member'].unique()
    max_pcs_to_plot = min(5, args.latent_dim)

    for member in unique_members:
        print(f"\n{'='*40}\nEvaluating Latent Member: {member}\n{'='*40}")
        df_member = df[df['member'] == member].copy()
        
        split_name = 'val' if member in val_early_members else 'test'
        member_base_dir = os.path.join(args.linreg_dir, f"latent_eval_{args.model_type}", f"{member}_{split_name}", freq_label)

        for m in args.winter_months:
            month_name = calendar.month_name[m]
            print(f"  --- Month: {month_name} ---")
            
            df_month = df_member[df_member['time'].dt.month == m].copy().reset_index(drop=True)
            if len(df_month) < 2:
                print(f"    Not enough data for month {m}. Skipping.")
                continue

            month_outdir = os.path.join(member_base_dir, f"month_{m}_{calendar.month_abbr[m]}")
            os.makedirs(month_outdir, exist_ok=True)

            pred_cols = [f'pred_pc{i+1}' for i in range(args.latent_dim)]
            true_cols = [f'true_pc{i+1}' for i in range(args.latent_dim)]
            
            if args.monthly_mean:
                df_m_res = df_month.set_index('time')[pred_cols + true_cols].resample('1M').mean().dropna().reset_index()
            else:
                df_m_res = df_month[['time'] + pred_cols + true_cols].dropna().reset_index(drop=True)
                
            Z_p_m = df_m_res[pred_cols].values
            Z_t_m = df_m_res[true_cols].values
            
            print(f"    Computing all latent metrics & bootstraps for {month_name}...")
            df_lat_ts, stats_global_month, stats_per_pc = compute_latent_metrics_and_bootstraps(
                Z_t_m, Z_p_m, df_m_res['time'], n_bootstraps=min(300, args.n_bootstraps)
            )

            plot_latent_timeseries_raw_metrics(df_lat_ts, member, month_outdir, freq_label, stats_global_month)

            quantiles_dict = {}
            if args.loss_type == 'quantile':
                for pc_idx in range(1, max_pcs_to_plot + 1):
                    quantiles_dict[pc_idx] = {}
                    for q in args.quantiles:
                        if q == 0.5: continue
                        col_q = f'pred_pc{pc_idx}_q{q}'
                        if col_q in df_month:
                            quantiles_dict[pc_idx][q] = df_m_res[col_q] if col_q in df_m_res else df_month[col_q]

            plot_combined_pcs_time_series(
                df_m_res, stats_per_pc, stats_global_month, m, member, 
                month_outdir, max_pcs=max_pcs_to_plot, freq_label=freq_label, quantiles_dict=quantiles_dict
            )
            print(f"    -> Plots générés avec succès dans : {month_outdir}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Evaluation Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("\nÉvaluation terminée avec succès !")