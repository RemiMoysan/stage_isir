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

# Setup des chemins d'importation
project_root = Path(__file__).resolve().parent.parent

vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.datasets import Dataset, Dataset_mensuel
from tools.models import ConvVAE

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_mse_timeseries(df_member, member, outdir, freq_label, lat_weight):
    """Plot the MSE time series comparing Pred vs True and Pred vs Reconstructed."""
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Gestion de la densité des points pour la lisibilité
    if freq_label == 'Daily':
        step = 15           # On ne garde qu'un point sur 15
        df_plot = df_member.iloc[::step].copy()
        lw = 0.8
        mk = None
        alpha_v = 0.8
    else:
        df_plot = df_member.copy()
        step = 1
        lw = 1.5
        mk = '.'
        alpha_v = 1.0

    # Création d'un index régulier pour le plot
    x_idx = np.arange(len(df_plot))
    x_labels = df_plot['time'].dt.strftime('%Y-%m-%d' if freq_label == 'Daily' else '%Y-%m').tolist()

    # Tracé des courbes temporelles
    ax.plot(x_idx, df_plot['mse_true'], label="MSE (Pred vs Target)", color="firebrick", linewidth=lw, marker=mk, alpha=alpha_v)
    ax.plot(x_idx, df_plot['mse_rec'], label="MSE (Pred vs Reconstructed Target)", color="teal", linewidth=lw, marker=mk, alpha=alpha_v)
    
    # Lignes horizontales
    mean_mse = df_member['mse_true'].mean()
    mean_mse_rec = df_member['mse_rec'].mean()
    mean_var = df_member['var_true'].mean()
    mean_var_rec = df_member['var_rec'].mean()

    ax.axhline(mean_mse, color='darkred', linestyle='--', linewidth=2, 
               label=f"Mean MSE ({mean_mse:.3f})")
    ax.axhline(mean_mse_rec, color='darkgreen', linestyle='--', linewidth=2, 
               label=f"Mean MSE (vs Reconstructed) ({mean_mse_rec:.3f})")
    ax.axhline(mean_var, color='red', linestyle='-.', linewidth=2, 
               label=f"Baseline 0 / Variance ({mean_var:.3f})")
    ax.axhline(mean_var_rec, color='green', linestyle='--', linewidth=2, 
               label=f"Baseline Reconstructed ({mean_var_rec:.3f})")
    
    weight_str = " (Lat-Weighted)" if lat_weight else ""

    ax.set_title(f"Time Series of Spatial MSE{weight_str} ({freq_label}) - Member {member}", fontsize=16)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.15, 1))

    n_ticks = min(15, len(x_idx))
    tick_indices = np.linspace(0, len(x_idx) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([x_labels[idx] for idx in tick_indices], rotation=45, ha="right")

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'MSE_Timeseries_{freq_label}_Member_{member}.png'), dpi=300)
    plt.close(fig)

def plot_spatial_mse_map(mean_mse_map, outdir, suffix_name):
    """Plot a 2D heatmap of the mean MSE per pixel."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(mean_mse_map, cmap='YlOrRd', origin='lower')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Squared Error', fontsize=12)
    
    ax.set_title(f"Average Spatial MSE per Pixel (no weight) ({suffix_name})", fontsize=14)
    ax.set_xlabel("Longitude Index")
    ax.set_ylabel("Latitude Index")
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'Spatial_MSE_Map_{suffix_name}.png'), dpi=300)
    plt.close(fig)

def plot_spatial_difference_map(diff_map, outdir, title, filename, score=False):
    """Plot a 2D heatmap using a diverging colormap centered on 0."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    vmax = np.max(np.abs(diff_map))
    vmin = -vmax
    
    im = ax.imshow(diff_map, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    if not score:
        cbar.set_label('Difference (Variance - MSE)', fontsize=12)
    else:
        cbar.set_label('MSE Skill Score', fontsize=12)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude Index")
    ax.set_ylabel("Latitude Index")
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'{filename}.png'), dpi=300)
    plt.close(fig)

# ============================================================
# ARCHITECTURE DU MODÈLE DE RÉGRESSION LINÉAIRE
# ============================================================
class LinearRegressionPredictor(nn.Module):
    def __init__(self, sst_shape=(85, 360), slp_shape=(53, 113), in_chans_sst=3, in_chans_slp=0, out_dim=128):
        super().__init__()
        self.sst_size = in_chans_sst * sst_shape[0] * sst_shape[1]
        self.slp_size = in_chans_slp * slp_shape[0] * slp_shape[1]
        self.total_input_size = self.sst_size + self.slp_size
        self.linear = nn.Linear(self.total_input_size, out_dim)

    def forward(self, x_sst, x_slp):
        batch_size = x_sst.size(0)
        x_sst_flat = x_sst.view(batch_size, -1)
        if self.slp_size > 0:
            x_slp_flat = x_slp.view(batch_size, -1)
            x = torch.cat([x_sst_flat, x_slp_flat], dim=1)
        else:
            x = x_sst_flat
        return self.linear(x)

# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='vae')
    parser.add_argument('--linreg_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['best', 'final'], default='best')
    parser.add_argument('--embed_path', type=str, required=True)
    
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
    
    # Gestion de la loss pour extraire correctement la médiane si quantiles utilisés
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile','correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

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
        active_sst_lags = args.sst_lags_days
        active_slp_lags = args.slp_lags_days
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        active_sst_lags = args.sst_lags_months
        active_slp_lags = args.slp_lags_months
        
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # 1.5 PRÉPARATION DES POIDS SPATIAUX
    # ============================================================
    wgts_flat = None
    coslat_2d = None
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
    # 2. LOAD MODELS (Embedder + LinReg)
    # ============================================================
    pca_model, vae_model = None, None
    if args.embed_method == 'pca':
        pca_model = joblib.load(args.embed_path)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=args.latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

    out_features = args.latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else args.latent_dim

    # Initialisation spécifique de la régression linéaire
    model = LinearRegressionPredictor(
        sst_shape=(85, 360), 
        slp_shape=(53, 113), 
        in_chans_sst=len(active_sst_lags), 
        in_chans_slp=len(active_slp_lags), 
        out_dim=out_features
    ).to(device)

    # Chargement dynamique selon le type (best vs final)
    if args.model_type == 'best':
        linreg_path = os.path.join(args.linreg_dir, "best_model_LinReg.pth")
        if not os.path.exists(linreg_path):
            print("⚠️ Fichier best_model_LinReg introuvable. Fallback possible.")
            linreg_path = os.path.join(args.linreg_dir, f"best_val_LinReg_bs{args.bs}.pth")
    elif args.model_type == 'final':
        linreg_path = os.path.join(args.linreg_dir, f"final_model_LinReg_bs{args.bs}.pth")

    try:
        checkpoint = torch.load(linreg_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Models loaded successfully from {linreg_path}.")
    except Exception as e:
        print(f"Erreur de chargement du modèle : Vérifie le nom du fichier sauvegardé. Erreur: {e}")

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
            X_sst = X_sst.to(device)
            X_slp = X_slp.to(device)
            y_target_np = y_target.numpy() 
            
            if len(y_target_np.shape) == 3:
                y_target_np = np.expand_dims(y_target_np, axis=1)
                
            B, C, H, W = y_target_np.shape

            # 1. Obtenir les prédictions latentes brutes
            predicted_raw = model(X_sst, X_slp).cpu().numpy()

            if args.loss_type == 'quantile':
                predicted_raw = predicted_raw.reshape(B, args.latent_dim, len(args.quantiles))
                median_idx = args.quantiles.index(0.5)
                pred_latent = predicted_raw[:, :, median_idx] # On utilise la médiane pour reconstruire la map
            else:
                pred_latent = predicted_raw

            # 2. Obtenir les latents réels & Décoder
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
                    pred_latent_padded = pred_latent
                    true_latent_padded = true_latent
                
                pred_map_flat = pca_model.inverse_transform(pred_latent_padded)
                rec_map_flat = pca_model.inverse_transform(true_latent_padded)

                if args.lat_weight and 'safe_wgts' in locals():
                    pred_map_flat = pred_map_flat / safe_wgts
                    rec_map_flat = rec_map_flat / safe_wgts
                
                pred_map = pred_map_flat.reshape(B, C, H, W)
                rec_map = rec_map_flat.reshape(B, C, H, W)

            elif args.embed_method == 'vae':
                y_target_tensor = torch.tensor(y_target_np, dtype=torch.float32).to(device)
                
                true_latent_tensor, _ = vae_model.encode(y_target_tensor)
                rec_map_tensor = vae_model.decode(true_latent_tensor)
                
                pred_latent_tensor = torch.tensor(pred_latent, dtype=torch.float32).to(device)
                pred_map_tensor = vae_model.decode(pred_latent_tensor)
                
                pred_map = pred_map_tensor.cpu().numpy()
                rec_map = rec_map_tensor.cpu().numpy()

            # 3. Stockage
            preds_list.append(pred_map)
            targets_list.append(y_target_np)
            recs_list.append(rec_map)
            
            dates_list.extend([str(d) for d in dates])
            for m in members:
                m_str = m if isinstance(m, str) else (m.item().decode() if isinstance(m.item(), bytes) else str(m.item()))
                members_list.append(m_str)
                
    print(f"Decoding finished in {time.time() - start_time:.2f} seconds.")

    # ============================================================
    # 4. AGGREGATION, XARRAY RESAMPLING & PLOTS
    # ============================================================
    preds_arr = np.concatenate(preds_list, axis=0)
    targets_arr = np.concatenate(targets_list, axis=0)
    recs_arr = np.concatenate(recs_list, axis=0)
    
    dates_arr = pd.to_datetime(dates_list)
    members_arr = np.array(members_list)

    freq_label = "Monthly" if args.monthly_mean else "Daily"
    print(f"\nProcessing spatial metrics via Xarray ({freq_label})...")
    
    unique_members = np.unique(members_arr)

    metrics = {
        'val':  {'sq_true': 0, 'var_true': 0, 'sq_rec': 0, 'var_rec': 0, 'time_steps': 0},
        'test': {'sq_true': 0, 'var_true': 0, 'sq_rec': 0, 'var_rec': 0, 'time_steps': 0}
    }

    _, _, H, W = targets_arr.shape
    if args.lat_weight and coslat_2d is not None:
        spatial_weights = xr.DataArray(coslat_2d, dims=["h", "w"])
    else:
        spatial_weights = xr.DataArray(np.ones((H, W)), dims=["h", "w"])

    sum_weights = spatial_weights.sum().values

    for member in unique_members:
        split_name = 'val' if member in val_early_members else 'test'
        print(f"--- Evaluating Spatial Maps Member: {member} ({split_name.upper()}) ---")

        out_eval_dir = os.path.join(args.linreg_dir, f"spatial_{split_name}_{args.model_type}_{freq_label}")
        os.makedirs(out_eval_dir, exist_ok=True)

        mask = (members_arr == member)
        
        ds_member = xr.Dataset(
            {
                "pred": (["time", "c", "h", "w"], preds_arr[mask]),
                "target": (["time", "c", "h", "w"], targets_arr[mask]),
                "rec": (["time", "c", "h", "w"], recs_arr[mask]),
            },
            coords={"time": dates_arr[mask]}
        )
        
        if args.monthly_mean:
            ds_member = ds_member.resample(time='1M').mean().dropna(dim="time")

        sq_err_true = (ds_member["pred"] - ds_member["target"])**2
        sq_err_rec = (ds_member["pred"] - ds_member["rec"])**2
        var_true = (ds_member["target"])**2 
        var_rec = (ds_member["rec"])**2 

        sq_err_true_w = sq_err_true * spatial_weights
        sq_err_rec_w = sq_err_rec * spatial_weights
        var_true_w = var_true * spatial_weights
        var_rec_w = var_rec * spatial_weights

        metrics[split_name]['sq_true'] += sq_err_true.sum(dim="time").values
        metrics[split_name]['var_true'] += var_true.sum(dim="time").values
        metrics[split_name]['sq_rec'] += sq_err_rec.sum(dim="time").values
        metrics[split_name]['var_rec'] += var_rec.sum(dim="time").values
        metrics[split_name]['time_steps'] += ds_member.sizes["time"]

        df_member_ts = pd.DataFrame({
            "time": ds_member["time"].values,
            "mse_true": (sq_err_true_w.sum(dim=["c", "h", "w"]) / sum_weights).values,
            "mse_rec": (sq_err_rec_w.sum(dim=["c", "h", "w"]) / sum_weights).values,
            "var_true": (var_true_w.sum(dim=["c", "h", "w"]) / sum_weights).values,
            "var_rec": (var_rec_w.sum(dim=["c", "h", "w"]) / sum_weights).values,
        })
        
        plot_mse_timeseries(df_member_ts, member, out_eval_dir, freq_label, args.lat_weight)
        
        print(f"  -> Graphique Time Series généré avec succès !")
        print(f"  -> Moyenne MSE = {df_member_ts['mse_true'].mean():.4f} | Baseline = {df_member_ts['var_true'].mean():.4f}")

    # ============================================================
    # 5. CARTES SPATIALES GLOBALES DE FIN
    # ============================================================
    
    for split in ['val', 'test']:
        if metrics[split]['time_steps'] == 0:
            continue

        out_eval_dir = os.path.join(args.linreg_dir, f"spatial_{split}_{args.model_type}_{freq_label}")
        t_steps = metrics[split]['time_steps']

        mean_mse_true_map = (metrics[split]['sq_true'] / t_steps).squeeze()
        mean_mse_rec_map = (metrics[split]['sq_rec'] / t_steps).squeeze()
        mean_var_true_map = (metrics[split]['var_true'] / t_steps).squeeze()
        mean_var_rec_map = (metrics[split]['var_rec'] / t_steps).squeeze()

        plot_spatial_mse_map(mean_mse_true_map, out_eval_dir, "MSE_vs_True_Target")
        plot_spatial_mse_map(mean_mse_rec_map, out_eval_dir, "MSE_vs_Reconstructed_Target")

        plot_spatial_difference_map(mean_var_true_map - mean_mse_true_map, out_eval_dir, 
            "Model Beat Baseline? (Var - MSE) vs True Target", "Spatial_Diff_True_Target")
        
        plot_spatial_difference_map(mean_var_rec_map - mean_mse_rec_map, out_eval_dir, 
            "Model Beat Baseline? (Var - MSE) vs Reconstructed Target", "Spatial_Diff_Reconstructed_Target")

        epsilon = 1e-8
        plot_spatial_difference_map(np.clip(1.0 - (mean_mse_true_map / (mean_var_true_map + epsilon)), -1, 1), out_eval_dir, 
            "MSE Skill Score vs True Target", "Spatial_MSESS_True_Target", score=True)
        
        plot_spatial_difference_map(np.clip(1.0 - (mean_mse_rec_map / (mean_var_rec_map + epsilon)), -1, 1), out_eval_dir, 
            "MSE Skill Score vs Reconstructed Target", "Spatial_MSESS_Reconstructed_Target", score=True)

        print(f"\n✅ Évaluation spatiale terminée pour le split {split.upper()}. Les résultats sont dans : {out_eval_dir}")