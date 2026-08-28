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
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from shared_tools.datasets import Dataset, Dataset_mensuel
from shared_tools.models import ConvVAE
# Utilisation stricte des mêmes fonctions d'évaluation que le CNN
from shared_tools.evaluation_functions import compute_latent_metrics_and_bootstraps, plot_combined_pcs_time_series, plot_latent_timeseries_raw_metrics 

# Import du modèle ViT Tunable
from tools.models import ViT_Latent_SLP_Multimodal_tunable

# ============================================================
# MAIN EVALUATION SCRIPT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca')
    parser.add_argument('--vit_dir', type=str, required=True, help='Chemin du dossier contenant le modèle entraîné de ViT')
    parser.add_argument('--model_type', type=str, choices=['best', 'final'], default='best', help='Évaluer le meilleur (best) ou le dernier (final)')
    parser.add_argument('--embed_path', type=str, required=True, help='Chemin du dossier contenant le modèle d\'embedding')
    
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
    parser.add_argument('--monthly_mean', action='store_true', help='Appliquer une moyenne mensuelle.')
    parser.add_argument('--monthly_reduction', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--use_lags_attention', action='store_true')
    
    parser.add_argument('--n_bootstraps', type=int, default=1000)
    
    # --- ARGUMENTS POUR LES QUANTILES ET LA LOSS ---
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile', 'correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # --- ARGUMENTS TUNABLES DU VIT ---
    parser.add_argument('--dr', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--transformer_act', type=str, choices=['gelu', 'relu'], default='gelu')
    parser.add_argument('--pool_strategy', type=str, choices=['cls', 'gap'], default='cls')
    parser.add_argument('--head_hidden_dim', type=int, default=0) # 0 se transforme en None
    parser.add_argument('--head_act', type=str, choices=['tanh', 'relu'], default='tanh')
    parser.add_argument('--patch_size_sst', type=int, nargs=2, default=[5, 10])
    parser.add_argument('--patch_size_slp', type=int, nargs=2, default=[5, 5])
    parser.add_argument('--norm_first', action='store_true')
    
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

    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        active_sst_lags = args.sst_lags_days
        active_slp_lags = args.slp_lags_days
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        active_sst_lags = args.sst_lags_months
        active_slp_lags = args.slp_lags_months

    # Ordre chronologique strict
    active_sst_lags = sorted(active_sst_lags, reverse=True)
    active_slp_lags = sorted(active_slp_lags, reverse=True)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # 1.5 PRÉPARATION DES POIDS SPATIAUX (POUR PCA)
    # ============================================================
    wgts_flat = None
    if args.lat_weight and args.embed_method == 'pca':
        sample_member = val_members[0]
        sample_path = os.path.join(f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-{sample_member}_1mo.nc")        
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            h, w = len(lats), len(ds_sample['lon'].values)
            wgts = np.sqrt(coslat).reshape(h, 1)
            wgts_flat = np.broadcast_to(wgts, (h, w)).flatten()
            ds_sample.close()
            print("Grille de poids de latitude générée pour le décodage PCA.")
        except Exception as e:
            print(f"Erreur lors du chargement de la grille de latitude : {e}")

    # ============================================================
    # 2. LOAD MODELS (EMBEDDER + VIT)
    # ============================================================
    pca_model, vae_model = None, None
    if args.embed_method == 'pca':
        pca_model = joblib.load(args.embed_path)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=args.latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

    out_features = args.latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else args.latent_dim
    h_dim = args.head_hidden_dim if args.head_hidden_dim > 0 else None

    # Initialisation du ViT Tunable
    model = ViT_Latent_SLP_Multimodal_tunable(
        sst_size=(85, 360), patch_size_sst=tuple(args.patch_size_sst), in_chans_sst=len(active_sst_lags), 
        slp_size=(53, 113), patch_size_slp=tuple(args.patch_size_slp), in_chans_slp=len(active_slp_lags), 
        nb_out=out_features, 
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, 
        mlp_ratio=args.mlp_ratio, transformer_act=args.transformer_act, dr=args.dr, 
        use_lags_attention=args.use_lags_attention, pool_strategy=args.pool_strategy, 
        head_hidden_dim=h_dim, head_act=args.head_act, norm_first=args.norm_first
    ).to(device)

    # Initialisation avec Dummy
    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(active_sst_lags), 85, 360).to(device) if len(active_sst_lags) > 0 else None
        dummy_slp = torch.zeros(1, len(active_slp_lags), 53, 113).to(device) if len(active_slp_lags) > 0 else None
        _ = model(dummy_sst, dummy_slp)

    if args.model_type == 'best':
        vit_path = os.path.join(args.vit_dir, "best_val_ViT.pth")
        if not os.path.exists(vit_path):
            vit_path = os.path.join(args.vit_dir, f"best_val_ViT_bs{args.bs}.pth")
    elif args.model_type == 'final':
        vit_path = os.path.join(args.vit_dir, "final_model_ViT.pth")
        if not os.path.exists(vit_path):
            vit_path = os.path.join(args.vit_dir, f"final_model_ViT_bs{args.bs}.pth")
        
    try:
        checkpoint = torch.load(vit_path, map_location=device)
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        print(f"Models loaded successfully from {vit_path}.")
    except Exception as e:
        print(f"Erreur critique lors du chargement : {e}")

    model.eval()

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
            
            # Cible latente
            if args.embed_method == 'pca':
                slp_flat = y_target.view(y_target.size(0), -1).numpy()
                if args.lat_weight and wgts_flat is not None:
                    slp_flat *= wgts_flat
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
    # 4. STATISTIQUES & PLOTS PAR MEMBRE ET PAR MOIS
    # ============================================================
    freq_label = "Monthly" if args.monthly_mean else "Daily"
    print(f"\nTime Series Frequency Set To: {freq_label}")
    
    unique_members = df['member'].unique()
    max_pcs_to_plot = min(5, args.latent_dim) # Tracé groupé de PC1 à PC5

    for member in unique_members:
        print(f"\n{'='*40}\nEvaluating Latent Member: {member}\n{'='*40}")
        df_member = df[df['member'] == member].copy()
        
        split_name = 'val' if member in val_early_members else 'test'
        member_base_dir = os.path.join(args.vit_dir, f"latent_eval_{args.model_type}", f"{member}_{split_name}", freq_label)

        # BOUCLE UNIFIÉE PAR MOIS D'HIVER
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