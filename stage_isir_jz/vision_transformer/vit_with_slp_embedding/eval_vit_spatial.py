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
import re

import sys
from pathlib import Path

# Setup des chemins d'importation vers la racine
project_root = Path(__file__).resolve().parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from shared_tools.datasets import Dataset, Dataset_mensuel
from shared_tools.models import ConvVAE

from shared_tools.evaluation_functions import (
    plot_spatial_timeseries_raw_metrics, 
    plot_metric_with_pvalue_map, 
    compute_map_metrics_and_bootstraps
)

# Import du modèle ViT Tunable
from tools.models import ViT_Latent_SLP_Multimodal_tunable

# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='vae')
    parser.add_argument('--vit_dir', type=str, required=True, help="Chemin vers le dossier d'entraînement ViT")
    parser.add_argument('--model_type', type=str, choices=['best', 'final'], default='best')
    parser.add_argument('--embed_path', type=str, required=True, help="Chemin vers le modèle VAE/PCA")
    
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
    parser.add_argument('--monthly_mean', action='store_true')
    parser.add_argument('--monthly_reduction', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--use_lags_attention', action='store_true')
    
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile','correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # --- ARGUMENTS TUNABLES DU VIT ---
    parser.add_argument('--dr', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--transformer_act', type=str, choices=['gelu', 'relu'], default='gelu')
    parser.add_argument('--pool_strategy', type=str, choices=['cls', 'gap'], default='cls')
    parser.add_argument('--head_hidden_dim', type=int, default=0) # 0 se traduit par None
    parser.add_argument('--head_act', type=str, choices=['tanh', 'relu'], default='tanh')
    parser.add_argument('--patch_size_sst', type=int, nargs=2, default=[5, 10])
    parser.add_argument('--patch_size_slp', type=int, nargs=2, default=[5, 5])
    parser.add_argument('--norm_first', action='store_true')

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
    val_early_members = all_members[-args.nb_members_val:]
    test_members = all_members[args.nb_members_train:args.nb_members_train + args.nb_members_test]
    val_members = val_early_members + test_members

    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1)

    dynamic_slp_std = 596.0
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))

    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        active_sst_lags = args.sst_lags_days; active_slp_lags = args.slp_lags_days
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        active_sst_lags = args.sst_lags_months; active_slp_lags = args.slp_lags_months

    # Tri décroissant pour être raccord
    active_sst_lags = sorted(active_sst_lags, reverse=True)
    active_slp_lags = sorted(active_slp_lags, reverse=True)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # 1.5 PRÉPARATION DES POIDS SPATIAUX
    # ============================================================
    wgts_flat = None; coslat_2d = None
    if args.lat_weight:
        sample_member = val_members[0]
        sample_path = os.path.join(f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-{sample_member}_1mo.nc")        
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            h, w = len(lats), len(ds_sample['lon'].values)
            coslat_2d = np.broadcast_to(coslat.reshape(-1, 1), (h, w))
            if args.embed_method == 'pca':
                wgts = np.sqrt(coslat).reshape(h, 1)
                wgts_flat = np.broadcast_to(wgts, (h, w)).flatten()
                safe_wgts = np.maximum(wgts_flat, 1e-5)
            ds_sample.close()
            print("✅ Grille de poids de latitude (cos(lat)) générée avec succès.")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la grille de latitude : {e}")

    # ============================================================
    # 2. LOAD MODELS (Embedder + ViT)
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

    model = ViT_Latent_SLP_Multimodal_tunable(
        sst_size=(85, 360), patch_size_sst=tuple(args.patch_size_sst), in_chans_sst=len(active_sst_lags), 
        slp_size=(53, 113), patch_size_slp=tuple(args.patch_size_slp), in_chans_slp=len(active_slp_lags), 
        nb_out=out_features, 
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, 
        mlp_ratio=args.mlp_ratio, transformer_act=args.transformer_act, dr=args.dr, 
        use_lags_attention=args.use_lags_attention, pool_strategy=args.pool_strategy, 
        head_hidden_dim=h_dim, head_act=args.head_act, norm_first=args.norm_first
    ).to(device)

    # Chargement dynamique selon le type (best vs final)
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
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Models loaded successfully from {vit_path}.")
    except Exception as e:
        print(f"Erreur de chargement du modèle : {e}")

    model.eval()

    # ============================================================
    # 3. INFERENCE & SPATIAL DECODING
    # ============================================================
    preds_list, targets_list, recs_list = [], [], []
    dates_list, members_list = [], []

    print("Running inference and decoding maps...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(valloader):
            X_sst = X_sst.to(device); X_slp = X_slp.to(device)
            y_target_np = y_target.numpy() 
            if len(y_target_np.shape) == 3:
                y_target_np = np.expand_dims(y_target_np, axis=1)
            B, C, H, W = y_target_np.shape

            predicted_raw = model(X_sst, X_slp).cpu().numpy()
            if args.loss_type == 'quantile':
                predicted_raw = predicted_raw.reshape(B, args.latent_dim, len(args.quantiles))
                pred_latent = predicted_raw[:, :, args.quantiles.index(0.5)]
            else:
                pred_latent = predicted_raw

            if args.embed_method == 'pca':
                slp_flat = y_target_np.reshape(B, -1)
                if args.lat_weight and 'safe_wgts' in locals():
                    slp_flat = slp_flat * wgts_flat
                true_latent = pca_model.transform(slp_flat)[:, :args.latent_dim]
                
                pca_expected_dim = pca_model.n_components_
                if args.latent_dim < pca_expected_dim:
                    pad_width = pca_expected_dim - args.latent_dim
                    pred_latent_padded = np.pad(pred_latent, ((0, 0), (0, pad_width)), mode='constant')
                    true_latent_padded = np.pad(true_latent, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    pred_latent_padded = pred_latent; true_latent_padded = true_latent
                
                pred_map_flat = pca_model.inverse_transform(pred_latent_padded)
                rec_map_flat = pca_model.inverse_transform(true_latent_padded)
                
                if args.lat_weight and 'safe_wgts' in locals():
                    pred_map_flat = pred_map_flat / safe_wgts; rec_map_flat = rec_map_flat / safe_wgts
                
                pred_map = pred_map_flat.reshape(B, C, H, W); rec_map = rec_map_flat.reshape(B, C, H, W)
                
            elif args.embed_method == 'vae':
                y_target_tensor = torch.tensor(y_target_np, dtype=torch.float32).to(device)
                true_latent_tensor, _ = vae_model.encode(y_target_tensor)
                rec_map = vae_model.decode(true_latent_tensor).cpu().numpy()
                pred_map = vae_model.decode(torch.tensor(pred_latent, dtype=torch.float32).to(device)).cpu().numpy()

            preds_list.append(pred_map); targets_list.append(y_target_np); recs_list.append(rec_map)
            dates_list.extend([str(d) for d in dates])
            for m in members:
                members_list.append(m if isinstance(m, str) else (m.item().decode() if isinstance(m.item(), bytes) else str(m.item())))
                
    print(f"Decoding finished in {time.time() - start_time:.2f} seconds.")

    # ============================================================
    # 4. ÉVALUATION SPATIALE PAR MEMBRE ET PAR MOIS
    # ============================================================
    preds_arr = np.concatenate(preds_list, axis=0).squeeze()
    targets_arr = np.concatenate(targets_list, axis=0).squeeze()
    recs_arr = np.concatenate(recs_list, axis=0).squeeze()
    dates_arr = pd.to_datetime(dates_list); members_arr = np.array(members_list)
    
    freq_label = "Monthly" if args.monthly_mean else "Daily"
    unique_members = np.unique(members_arr)
    _, H, W = targets_arr.shape
    spatial_weights = coslat_2d / coslat_2d.sum() if (args.lat_weight and coslat_2d is not None) else np.ones((H, W)) / (H * W)

    for member in unique_members:
        print(f"\n{'='*40}\nEvaluating Spatial Member: {member}\n{'='*40}")
        split_name = 'val' if member in val_early_members else 'test'
        member_base_dir = os.path.join(args.vit_dir, f"spatial_eval_{args.model_type}", f"{member}_{split_name}", freq_label)

        mask_mem = (members_arr == member)
        ds_member = xr.Dataset(
            {"pred": (["time", "h", "w"], preds_arr[mask_mem]),
             "target": (["time", "h", "w"], targets_arr[mask_mem]),
             "rec": (["time", "h", "w"], recs_arr[mask_mem])},
            coords={"time": dates_arr[mask_mem]}
        )
        if args.monthly_mean: ds_member = ds_member.resample(time='1M').mean().dropna(dim="time")

        # BOUCLE UNIFIÉE PAR MOIS D'HIVER
        for m in args.winter_months:
            month_name = calendar.month_name[m]
            print(f"  --- Month: {month_name} ---")
            
            ds_month = ds_member.where(ds_member["time"].dt.month == m, drop=True)
            if ds_month.sizes["time"] < 2: continue

            month_outdir = os.path.join(member_base_dir, f"month_{m}_{calendar.month_abbr[m]}")
            os.makedirs(month_outdir, exist_ok=True)

            p_m = ds_month["pred"].values; t_m = ds_month["target"].values; r_m = ds_month["rec"].values

            # 1. Calcul des cartes & des métriques spatiales bootstrapées
            r2_map, pval_r2, l1_map, pval_l1, corr_map, pval_corr, stats_true = compute_map_metrics_and_bootstraps(t_m, p_m, spatial_weights, n_bootstraps=300)
            r2_map_r, pval_r2_r, l1_map_r, pval_l1_r, corr_map_r, pval_corr_r, stats_rec = compute_map_metrics_and_bootstraps(r_m, p_m, spatial_weights, n_bootstraps=300)

            # 2. Séries temporelles brutes (MSE, MAE, Corr) pour le mois ciblé
            mse_t_true = np.sum((p_m - t_m)**2 * spatial_weights, axis=(1, 2)); var_t_true = np.sum(t_m**2 * spatial_weights, axis=(1, 2))
            mse_t_rec  = np.sum((p_m - r_m)**2 * spatial_weights, axis=(1, 2)); var_t_rec  = np.sum(r_m**2 * spatial_weights, axis=(1, 2))
            mae_t_true = np.sum(np.abs(p_m - t_m) * spatial_weights, axis=(1, 2)); ref_mae_true = np.sum(np.abs(t_m) * spatial_weights, axis=(1, 2))
            mae_t_rec  = np.sum(np.abs(p_m - r_m) * spatial_weights, axis=(1, 2)); ref_mae_rec  = np.sum(np.abs(r_m) * spatial_weights, axis=(1, 2))

            p_sub = p_m - np.sum(p_m * spatial_weights, axis=(1, 2), keepdims=True)
            t_sub = t_m - np.sum(t_m * spatial_weights, axis=(1, 2), keepdims=True); r_sub = r_m - np.sum(r_m * spatial_weights, axis=(1, 2), keepdims=True)
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

            # 3. Tracé des 6 cartes (3 True + 3 Rec) avec p-values
            plot_metric_with_pvalue_map(r2_map, pval_r2, month_outdir, f"Map_R2_vs_True_{calendar.month_abbr[m]}", "l2", f"L2 Skill Score vs True Target - {month_name}", global_stat=stats_true['global_r2'], mean_pixel_stat=stats_true['mean_pixel_r2'])
            plot_metric_with_pvalue_map(l1_map, pval_l1, month_outdir, f"Map_L1_vs_True_{calendar.month_abbr[m]}", "l1", f"L1 Skill Score vs True Target - {month_name}", global_stat=stats_true['global_l1'], mean_pixel_stat=stats_true['mean_pixel_l1'])
            plot_metric_with_pvalue_map(corr_map, pval_corr, month_outdir, f"Map_Corr_vs_True_{calendar.month_abbr[m]}", "corr", f"Temporal Correlation vs True Target - {month_name}", global_stat=stats_true['global_corr'], mean_pixel_stat=stats_true['mean_pixel_corr'])

            plot_metric_with_pvalue_map(r2_map_r, pval_r2_r, month_outdir, f"Map_R2_vs_Rec_{calendar.month_abbr[m]}", "l2", f"L2 Skill Score vs Reconstructed - {month_name}", global_stat=stats_rec['global_r2'], mean_pixel_stat=stats_rec['mean_pixel_r2'])
            plot_metric_with_pvalue_map(l1_map_r, pval_l1_r, month_outdir, f"Map_L1_vs_Rec_{calendar.month_abbr[m]}", "l1", f"L1 Skill Score vs Reconstructed - {month_name}", global_stat=stats_rec['global_l1'], mean_pixel_stat=stats_rec['mean_pixel_l1'])
            plot_metric_with_pvalue_map(corr_map_r, pval_corr_r, month_outdir, f"Map_Corr_vs_Rec_{calendar.month_abbr[m]}", "corr", f"Temporal Correlation vs Reconstructed - {month_name}", global_stat=stats_rec['global_corr'], mean_pixel_stat=stats_rec['mean_pixel_corr'])

    print(f"\n✅ Évaluation spatiale par membre terminée avec succès ! Tous les graphiques sont dans : {args.vit_dir}/spatial_eval_{args.model_type}")