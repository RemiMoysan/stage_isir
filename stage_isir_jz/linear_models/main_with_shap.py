import os
import time
import argparse
import copy
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import random 
import re
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import cartopy.crs as ccrs

import sys 
from pathlib import Path
import subprocess

project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

from shared_tools.visualizations import (
    loss_figure, loss_first_epoch, plot_and_save_maps_with_reconstruction_light, 
    plot_correlation_evolution, plot_r2_R2_evolution, MapMetricTracker, 
    LatentMetricTracker, save_r2_pixel_map_and_plot, plot_map_r2_evolution, 
    plot_spatial_corr_evolution, plot_latent_l1_ss_evolution
)
from shared_tools.datasets import Dataset, Dataset_mensuel
from shared_tools.models import (
    compute_loss, get_median_prediction, spatial_penalty_tikhonov, 
    spatial_penalty_laplacian, decode_latent_to_map
)
from shared_tools.optuna_loop_helpers import encode_to_latent_gpu

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
# DEVICE & ARGUMENTS CONFIGURATION
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', type=int, default=0) 
    parser.add_argument('--machine', type=str, default='jean-zay-work', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    
    # --- PCA PATHS ---
    parser.add_argument('--embed_method', type=str, default='pca')
    parser.add_argument('--embed_path', type=str, required=True, help="Chemin vers le PCA SLP (Target)")
    parser.add_argument('--embed_path_sst', type=str, default=None, help="Chemin vers le PCA SST (Optionnel)")
    
    # --- SPLIT ---
    parser.add_argument('--nb_members_train', type=int, default=10)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_members_test', type=int, default=0)
    parser.add_argument('--force_val_members', type=str, nargs='*', default=None)
    parser.add_argument('--force_test_members', type=str, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    # --- HYPERPARAMETRES ---
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--nb_epochs', type=int, default=20)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--alpha_l1', type=float, default=0.0)
    parser.add_argument('--alpha_tik', type=float, default=0.0)
    parser.add_argument('--alpha_lap', type=float, default=0.0)
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--gradient_clip', type=float, default=100.0)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile', 'correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='*', default=[])
    parser.add_argument('--input_format', type=str, choices=['raw', 'pca'], default='raw')
    parser.add_argument('--sst_pca_dim', type=int, default=0)
    
    # --- DATA PARAMS ---
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[1])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--monthly_reduction', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--nb_intra_evals', type=int, default=15)
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--normalize', action='store_true', help='PCA normalisé ou non')  

    args = parser.parse_args()

    # Routage dynamique des dossiers
    if args.machine == 'hacienda': base_home = "/home/moysan/stage_isir_jz/linear_models/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']: base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/"
    elif args.machine == 'mac_local': base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/linear_models/"

    active_sst_lags = sorted(args.sst_lags_months if args.monthly_reduction else args.sst_lags_days, reverse=True) 
    active_slp_lags = sorted(args.slp_lags_months if args.monthly_reduction else args.slp_lags_days, reverse=True)

    # --- SPLIT DES MEMBRES ---
    patience = 10000
    target_indices = {100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000} if not args.monthly_reduction else {1, 10, 20,30,40,45,50,60,70, 80} 
    
    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)
    
    if args.force_val_members is not None or args.force_test_members is not None:
        val_members = args.force_val_members if args.force_val_members else []
        test_members = args.force_test_members if args.force_test_members else []
        remaining = [m for m in all_members if m not in val_members and m not in test_members]
        train_members = remaining[:args.nb_members_train]
        nb_members_train = len(train_members)
        nb_members_val = len(val_members)
        nb_members_test = len(test_members)
    else:
        nb_members_train = args.nb_members_train
        nb_members_val = args.nb_members_val
        nb_members_test = args.nb_members_test
        train_members = all_members[:nb_members_train]
        val_members = all_members[-nb_members_val:]
        test_members = all_members[nb_members_train:nb_members_train + nb_members_test] if nb_members_test > 0 else []

    # --- STD EXTRACTION ---
    dynamic_slp_std = 596.0 
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match: dynamic_slp_std = float(match.group(1))

    dynamic_sst_std = 0.707 
    if args.embed_path_sst:
        match = re.search(r'sst_std([0-9.]+)', args.embed_path_sst)
        if match: dynamic_sst_std = float(match.group(1))

    loss_tag = args.loss_type
    base_outdir_name = f"LinReg_loss_{loss_tag}_L1-{args.alpha_l1}_Tik-{args.alpha_tik}_Lap-{args.alpha_lap}_emb_{args.latent_dim}_lat_{args.lat_weight}_inp_{args.input_format}_bs{args.bs}_lr{args.lr}_months_{''.join(map(str, args.winter_months))}_seed{args.seed}_train{nb_members_train}_val{nb_members_val}_{nb_members_test}_norm{args.normalize}"
    
    if not args.monthly_reduction:
        outdir_name = f"{base_outdir_name}_sst_{''.join(map(str, active_sst_lags))}_slp_{''.join(map(str, active_slp_lags))}_{args.duree_lissage}d_roll_{args.roll_sst}_slp_std{dynamic_slp_std}"
    else:
        outdir_name = f"{base_outdir_name}_sst_{''.join(map(str, active_sst_lags))}_slp_{''.join(map(str, active_slp_lags))}_monthly_roll_{args.roll_sst}_slp_std{dynamic_slp_std}"

    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    intra_workers = min(2, n_workers)

    # ============================================================
    # PRÉPARATION POIDS DE LATITUDE ET MODELES PCA (GPU)
    # ============================================================
    slp_pca_model = joblib.load(args.embed_path)
    slp_pca_mean_gpu = torch.tensor(slp_pca_model.mean_, dtype=torch.float32, device=device)
    slp_pca_components_gpu = torch.tensor(slp_pca_model.components_[:max(args.latent_dim, 1)], dtype=torch.float32, device=device)

    wgts_slp_gpu, wgts_slp_flat, safe_wgts_slp, area_weights_2d = None, None, None, None
    if args.lat_weight:
        sample_path = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-1001.001_1mo.nc"
        try:
            with xr.open_dataset(sample_path) as ds_sample:
                coslat = np.cos(np.deg2rad(ds_sample['lat'].values)).clip(0., 1.)
                h, w = len(coslat), len(ds_sample['lon'].values)
                wgts_slp_flat = np.broadcast_to(np.sqrt(coslat).reshape(h, 1), (h, w)).flatten()
                wgts_slp_gpu = torch.tensor(wgts_slp_flat, dtype=torch.float32, device=device)
                safe_wgts_slp = np.maximum(wgts_slp_flat, 1e-5)
                area_weights_2d = torch.tensor(np.broadcast_to(coslat.reshape(h, 1), (h, w)), dtype=torch.float64, device=device)
        except Exception as e:
            print(f"⚠️ Erreur latitude SLP : {e}")

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
        except Exception as e:
            print(f"⚠️ Erreur latitude SST : {e}")

    # ============================================================
    # DATALOADERS
    # ============================================================
    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
        training_set = Dataset(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=True, noise_std=args.noise_std)
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
        training_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=True, noise_std=args.noise_std)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=True, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=True, num_workers=intra_workers, pin_memory=True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=args.bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    if nb_members_test > 0:
        if not args.monthly_reduction:
            test_set = Dataset(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
        else:
            test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)
        testloader_intra = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=intra_workers, pin_memory=True)
    else:
        testloader, testloader_intra = None, None

    # ============================================================
    # INITIALISATION DU MODÈLE DE RÉGRESSION
    # ============================================================
    if args.input_format == 'pca':
        in_features_sst = len(active_sst_lags) * args.sst_pca_dim
        in_features_slp = len(active_slp_lags) * 1
    else:
        in_features_sst = len(active_sst_lags) * 85 * 360
        in_features_slp = len(active_slp_lags) * 53 * 113

    out_features = args.latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else args.latent_dim

    model = LinearRegressionPredictor(in_features=in_features_sst + in_features_slp, out_dim=out_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Fonction utilitaire de formatage (Identique LOOCV)
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

    # Suivi
    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float('inf') 
    best_model_path = ""
    val_losses_per_member_history = defaultdict(list)
    
    train_lat_mR2, val_lat_mR2, test_lat_mR2 = [], [], []
    train_lat_gR2, val_lat_gR2, test_lat_gR2 = [], [], []
    train_lat_mCorr, val_lat_mCorr, test_lat_mCorr = [], [], []
    train_lat_gCorr, val_lat_gCorr, test_lat_gCorr = [], [], []
    train_lat_mk, val_lat_mk, test_lat_mk = [], [], []
    train_lat_gk, val_lat_gk, test_lat_gk = [], [], []
    train_lat_gL1, val_lat_gL1, test_lat_gL1 = [], [], []
    train_lat_mL1, val_lat_mL1, test_lat_mL1 = [], [], []

    train_map_gR2, val_map_gR2, test_map_gR2 = [], [], []
    train_map_mR2, val_map_mR2, test_map_mR2 = [], [], []
    train_map_sCorr, val_map_sCorr, test_map_sCorr = [], [], []
    train_map_tCorr, val_map_tCorr, test_map_tCorr = [], [], []
    train_map_gCorr, val_map_gCorr, test_map_gCorr = [], [], []
    train_map_gL1, val_map_gL1, test_map_gL1 = [], [], []
    train_map_mL1, val_map_mL1, test_map_mL1 = [], [], []

    total_batches = len(trainloader)
    epoch1_batch_losses, epoch1_baseline_losses = [], []
    eval_steps_set = set(np.insert(np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int), 0, 0)) if args.nb_intra_evals > 0 else set()
    eval_steps_epoch2_set = set(np.insert(np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int), 0, 0)) if args.nb_intra_evals > 0 else set()

    intra_epoch1_steps, intra_epoch1_val_losses, intra_epoch1_test_losses = [], [], []
    intra_epoch1_val_lat_mR2, intra_epoch1_val_lat_gR2, intra_epoch1_val_lat_mCorr, intra_epoch1_val_lat_gCorr, intra_epoch1_val_lat_mk, intra_epoch1_val_lat_gk, intra_epoch1_val_lat_gL1, intra_epoch1_val_lat_mL1 = [], [], [], [], [], [], [], []
    intra_epoch1_val_map_gR2, intra_epoch1_val_map_mR2, intra_epoch1_val_map_sCorr, intra_epoch1_val_map_tCorr, intra_epoch1_val_map_gCorr, intra_epoch1_val_map_gL1, intra_epoch1_val_map_mL1 = [], [], [], [], [], [], []
    intra_epoch1_test_lat_mR2, intra_epoch1_test_lat_gR2, intra_epoch1_test_lat_mCorr, intra_epoch1_test_lat_gCorr, intra_epoch1_test_lat_mk, intra_epoch1_test_lat_gk, intra_epoch1_test_lat_gL1, intra_epoch1_test_lat_mL1 = [], [], [], [], [], [], [], []
    intra_epoch1_test_map_gR2, intra_epoch1_test_map_mR2, intra_epoch1_test_map_sCorr, intra_epoch1_test_map_tCorr, intra_epoch1_test_map_gCorr, intra_epoch1_test_map_gL1, intra_epoch1_test_map_mL1 = [], [], [], [], [], [], []

    intra_epoch2_steps, intra_epoch2_val_losses, intra_epoch2_test_losses = [], [], []
    intra_epoch2_val_lat_mR2, intra_epoch2_val_lat_gR2, intra_epoch2_val_lat_mCorr, intra_epoch2_val_lat_gCorr, intra_epoch2_val_lat_mk, intra_epoch2_val_lat_gk, intra_epoch2_val_lat_gL1, intra_epoch2_val_lat_mL1 = [], [], [], [], [], [], [], []
    intra_epoch2_val_map_gR2, intra_epoch2_val_map_mR2, intra_epoch2_val_map_sCorr, intra_epoch2_val_map_tCorr, intra_epoch2_val_map_gCorr, intra_epoch2_val_map_gL1, intra_epoch2_val_map_mL1 = [], [], [], [], [], [], []
    intra_epoch2_test_lat_mR2, intra_epoch2_test_lat_gR2, intra_epoch2_test_lat_mCorr, intra_epoch2_test_lat_gCorr, intra_epoch2_test_lat_mk, intra_epoch2_test_lat_gk, intra_epoch2_test_lat_gL1, intra_epoch2_test_lat_mL1 = [], [], [], [], [], [], [], []
    intra_epoch2_test_map_gR2, intra_epoch2_test_map_mR2, intra_epoch2_test_map_sCorr, intra_epoch2_test_map_tCorr, intra_epoch2_test_map_gCorr, intra_epoch2_test_map_gL1, intra_epoch2_test_map_mL1 = [], [], [], [], [], [], []

    if args.update == 1:
        initial_params = torch.load(f"{outdir}/final_model_LinReg.pth")
        model.load_state_dict(initial_params['state_dict'])
        optimizer.load_state_dict(initial_params['optimizer'])
        train_losses = initial_params['train_losses']
        val_losses = initial_params['val_losses']
        test_losses = initial_params.get('test_losses', [])
        best_val_loss = np.min(val_losses) if len(val_losses)>0 else float('inf')

    # ============================================================
    # TRAINING & EVALUATION LOOP
    # ============================================================
    start_time = time.time()
    epoch_times = []
    patience_counter = 0
    best_model_state = None

    for epoch in range(args.nb_epochs):
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0
        epoch_grad_norms = []
        
        train_latent_tracker = LatentMetricTracker(device=device)
        train_map_tracker = MapMetricTracker(shape=(53, 113), device=device)
        
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
            if batch_idx % 30 == 0:
                print(f" {100 * batch_idx / len(trainloader):.1f}% training complete", end='\r')
                
            optimizer.zero_grad()
            X_sst = X_sst.to(device, non_blocking=True) 
            if len(active_slp_lags) > 0:
                X_slp = X_slp.to(device, non_blocking=True) 

            X_combined = format_inputs(X_sst, X_slp)
            target_embed = encode_to_latent_gpu(y_target.to(device, non_blocking=True), 'pca', args.latent_dim, slp_pca_components_gpu, slp_pca_mean_gpu, wgts_slp_gpu, None)
                    
            predicted_latent = model(X_combined)            
            base_loss = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')

            # Pénalités
            penalty = 0.0
            if args.alpha_l1 > 0: penalty += args.alpha_l1 * torch.norm(model.linear.weight, p=1)
            if args.input_format == 'raw':
                w_sst = model.linear.weight[:, :in_features_sst]
                if args.alpha_tik > 0: penalty += args.alpha_tik * spatial_penalty_tikhonov(w_sst, len(active_sst_lags), 85, 360)
                if args.alpha_lap > 0: penalty += args.alpha_lap * spatial_penalty_laplacian(w_sst, len(active_sst_lags), 85, 360)
                if len(active_slp_lags) > 0:
                    w_slp = model.linear.weight[:, in_features_sst:]
                    if args.alpha_tik > 0: penalty += args.alpha_tik * spatial_penalty_tikhonov(w_slp, len(active_slp_lags), 53, 113)
                    if args.alpha_lap > 0: penalty += args.alpha_lap * spatial_penalty_laplacian(w_slp, len(active_slp_lags), 53, 113)
                    
            loss_value = base_loss + penalty
            loss_value.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
            epoch_grad_norms.append(grad_norm.item())
            optimizer.step()
            
            running_train_loss += base_loss.item() * X_sst.size(0)
            total_train_samples += X_sst.size(0)

            med_pred = get_median_prediction(predicted_latent, args.loss_type, args.quantiles, args.latent_dim) if args.loss_type == 'quantile' else predicted_latent 
            train_latent_tracker.update(med_pred.detach(), target_embed.detach())
            
            # Reconstruction Map pour R2 Spatial
            decoded_pred_map = decode_latent_to_map(med_pred, args, args.latent_dim, slp_pca_model, None, safe_wgts_slp)
            train_map_tracker.update(y_target.detach(), decoded_pred_map.detach())

            if epoch == 0:
                epoch1_batch_losses.append(loss_value.item())
                with torch.no_grad():
                    zeros_pred = torch.zeros_like(predicted_latent)
                    epoch1_baseline_losses.append(compute_loss(zeros_pred, target_embed, args.loss_type, args.quantiles, reduction='mean').item())

            # ---------------- INTRA-EPOCH VALIDATION ----------------
            if args.nb_intra_evals > 0 and (epoch == 0 or epoch == 1):  
                current_eval_steps = eval_steps_set if epoch == 0 else eval_steps_epoch2_set
                if batch_idx in current_eval_steps or batch_idx == len(trainloader) - 1:
                    eval_phases = ['val', 'test'] if nb_members_test > 0 else ['val']
                    for key in eval_phases:
                        loader = valloader_intra if key == 'val' else testloader_intra
                        model.eval()
                        
                        intra_val_loss, intra_n_samples = 0.0, 0
                        intra_latent_tracker = LatentMetricTracker(device=device)
                        intra_map_tracker = MapMetricTracker(shape=(53, 113), device=device)      
                        
                        with torch.no_grad():
                            for v_X_sst, v_X_slp, v_y_target, _, _, _ in loader:
                                v_X_sst = v_X_sst.to(device, non_blocking=True)
                                if len(active_slp_lags) > 0: v_X_slp = v_X_slp.to(device, non_blocking=True)
                                
                                v_X_combined = format_inputs(v_X_sst, v_X_slp)
                                v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), 'pca', args.latent_dim, slp_pca_components_gpu, slp_pca_mean_gpu, wgts_slp_gpu, None)
                                
                                v_pred = model(v_X_combined)
                                intra_val_loss += compute_loss(v_pred, v_target_embed, args.loss_type, args.quantiles, reduction='mean').item() * v_X_sst.size(0)
                                
                                p = get_median_prediction(v_pred, args.loss_type, args.quantiles, args.latent_dim) if args.loss_type == 'quantile' else v_pred
                                intra_latent_tracker.update(p.detach(), v_target_embed.detach())
                                intra_n_samples += p.size(0)

                                decoded_v_pred = decode_latent_to_map(p, args, args.latent_dim, slp_pca_model, None, safe_wgts_slp)
                                intra_map_tracker.update(v_y_target.detach(), decoded_v_pred.detach())

                        lat_gR2, lat_mR2, lat_gCorr, lat_mCorr, lat_gK, lat_mK, lat_gL1, lat_mL1 = intra_latent_tracker.compute()
                        map_gR2, map_mR2, map_r2_np, map_sCorr, map_tCorr, map_gCorr, map_corr_np, map_gL1, map_mL1, map_l1_np = intra_map_tracker.compute(area_weights=area_weights_2d)
                        
                        save_r2_pixel_map_and_plot(map_r2_np, outdir, f"L2_intra_{key}_ep{epoch+1}_step{batch_idx}", metric_type="l2")
                        save_r2_pixel_map_and_plot(map_l1_np, outdir, f"L1_intra_{key}_ep{epoch+1}_step{batch_idx}", metric_type="l1")
                        save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_intra_{key}_ep{epoch+1}_step{batch_idx}", metric_type="corr")

                        if epoch == 0:
                            if key == 'val':
                                intra_epoch1_steps.append(batch_idx) 
                                intra_epoch1_val_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch1_val_lat_gR2.append(lat_gR2); intra_epoch1_val_lat_mR2.append(lat_mR2)
                                intra_epoch1_val_lat_gCorr.append(lat_gCorr); intra_epoch1_val_lat_mCorr.append(lat_mCorr)
                                intra_epoch1_val_lat_mk.append(lat_mK); intra_epoch1_val_lat_gk.append(lat_gK)
                                intra_epoch1_val_lat_gL1.append(lat_gL1); intra_epoch1_val_lat_mL1.append(lat_mL1)
                                intra_epoch1_val_map_gR2.append(map_gR2); intra_epoch1_val_map_mR2.append(map_mR2)
                                intra_epoch1_val_map_sCorr.append(map_sCorr); intra_epoch1_val_map_tCorr.append(map_tCorr)
                                intra_epoch1_val_map_gCorr.append(map_gCorr); intra_epoch1_val_map_gL1.append(map_gL1)
                                intra_epoch1_val_map_mL1.append(map_mL1)
                            else:
                                intra_epoch1_test_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch1_test_lat_gR2.append(lat_gR2); intra_epoch1_test_lat_mR2.append(lat_mR2)
                                intra_epoch1_test_lat_gCorr.append(lat_gCorr); intra_epoch1_test_lat_mCorr.append(lat_mCorr)
                                intra_epoch1_test_lat_mk.append(lat_mK); intra_epoch1_test_lat_gk.append(lat_gK)
                                intra_epoch1_test_lat_gL1.append(lat_gL1); intra_epoch1_test_lat_mL1.append(lat_mL1)
                                intra_epoch1_test_map_gR2.append(map_gR2); intra_epoch1_test_map_mR2.append(map_mR2)
                                intra_epoch1_test_map_sCorr.append(map_sCorr); intra_epoch1_test_map_tCorr.append(map_tCorr)
                                intra_epoch1_test_map_gCorr.append(map_gCorr); intra_epoch1_test_map_gL1.append(map_gL1)
                                intra_epoch1_test_map_mL1.append(map_mL1)
                        elif epoch == 1:
                            if key == 'val':
                                intra_epoch2_steps.append(batch_idx)
                                intra_epoch2_val_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch2_val_lat_gR2.append(lat_gR2); intra_epoch2_val_lat_mR2.append(lat_mR2)
                                intra_epoch2_val_lat_gCorr.append(lat_gCorr); intra_epoch2_val_lat_mCorr.append(lat_mCorr)
                                intra_epoch2_val_lat_mk.append(lat_mK); intra_epoch2_val_lat_gk.append(lat_gK)
                                intra_epoch2_val_lat_gL1.append(lat_gL1); intra_epoch2_val_lat_mL1.append(lat_mL1)
                                intra_epoch2_val_map_gR2.append(map_gR2); intra_epoch2_val_map_mR2.append(map_mR2)
                                intra_epoch2_val_map_sCorr.append(map_sCorr); intra_epoch2_val_map_tCorr.append(map_tCorr)
                                intra_epoch2_val_map_gCorr.append(map_gCorr); intra_epoch2_val_map_gL1.append(map_gL1)
                                intra_epoch2_val_map_mL1.append(map_mL1)
                            else:
                                intra_epoch2_test_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch2_test_lat_gR2.append(lat_gR2); intra_epoch2_test_lat_mR2.append(lat_mR2)
                                intra_epoch2_test_lat_gCorr.append(lat_gCorr); intra_epoch2_test_lat_mCorr.append(lat_mCorr)
                                intra_epoch2_test_lat_mk.append(lat_mK); intra_epoch2_test_lat_gk.append(lat_gK)
                                intra_epoch2_test_lat_gL1.append(lat_gL1); intra_epoch2_test_lat_mL1.append(lat_mL1)
                                intra_epoch2_test_map_gR2.append(map_gR2); intra_epoch2_test_map_mR2.append(map_mR2)
                                intra_epoch2_test_map_sCorr.append(map_sCorr); intra_epoch2_test_map_tCorr.append(map_tCorr)
                                intra_epoch2_test_map_gCorr.append(map_gCorr); intra_epoch2_test_map_gL1.append(map_gL1)
                                intra_epoch2_test_map_mL1.append(map_mL1)

                        current_intra_loss = intra_val_loss / intra_n_samples
                        if key == 'val' and current_intra_loss < best_val_loss:
                            best_val_loss = current_intra_loss
                            best_model_state = copy.deepcopy(model.state_dict())
                            if best_model_path and os.path.exists(best_model_path): os.remove(best_model_path)
                            best_model_path = os.path.join(outdir, f'best_val_LinReg_ep{epoch + 1}_step{batch_idx}_loss{best_val_loss:.4f}.pth')
                            torch.save(model.state_dict(), best_model_path)
                        model.train()

        # Finalisation des calculs d'époque Train
        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)
        gn = np.array(epoch_grad_norms)
        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}, Grad Norms: mean={gn.mean():.4f}, std={gn.std():.4f}, max={gn.max():.4f}')

        lat_gR2, lat_mR2, lat_gCorr, lat_mCorr, lat_gK, lat_mK, lat_gL1, lat_mL1 = train_latent_tracker.compute()
        train_lat_gR2.append(lat_gR2); train_lat_mR2.append(lat_mR2)
        train_lat_gCorr.append(lat_gCorr); train_lat_mCorr.append(lat_mCorr)
        train_lat_mk.append(lat_mK); train_lat_gk.append(lat_gK)
        train_lat_gL1.append(lat_gL1); train_lat_mL1.append(lat_mL1)

        map_gR2, map_mR2, map_r2_np, map_sCorr, map_tCorr, map_gCorr, map_corr_np, map_gL1, map_mL1, map_l1_np = train_map_tracker.compute(area_weights=area_weights_2d)
        train_map_gR2.append(map_gR2); train_map_mR2.append(map_mR2)
        train_map_sCorr.append(map_sCorr); train_map_tCorr.append(map_tCorr)
        train_map_gCorr.append(map_gCorr); train_map_gL1.append(map_gL1)
        train_map_mL1.append(map_mL1)

        save_r2_pixel_map_and_plot(map_r2_np, outdir, f"L2_train_ep{epoch+1}", metric_type="l2")
        save_r2_pixel_map_and_plot(map_l1_np, outdir, f"L1_train_ep{epoch+1}", metric_type="l1")
        save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_train_ep{epoch+1}", metric_type="corr")

        if epoch == 0 or epoch == 1:
            if epoch == 0: loss_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, outdir, label="Train")
            if args.nb_intra_evals > 0:
                c_steps = intra_epoch1_steps if epoch == 0 else intra_epoch2_steps
                c_losses = intra_epoch1_val_losses if epoch == 0 else intra_epoch2_val_losses
                c_mR2 = intra_epoch1_val_lat_mR2 if epoch == 0 else intra_epoch2_val_lat_mR2
                c_gR2 = intra_epoch1_val_lat_gR2 if epoch == 0 else intra_epoch2_val_lat_gR2
                c_mCorr = intra_epoch1_val_lat_mCorr if epoch == 0 else intra_epoch2_val_lat_mCorr
                c_gCorr = intra_epoch1_val_lat_gCorr if epoch == 0 else intra_epoch2_val_lat_gCorr
                c_mk = intra_epoch1_val_lat_mk if epoch == 0 else intra_epoch2_val_lat_mk
                c_gK = intra_epoch1_val_lat_gk if epoch == 0 else intra_epoch2_val_lat_gk
                c_mL1 = intra_epoch1_val_lat_mL1 if epoch == 0 else intra_epoch2_val_lat_mL1
                c_gL1 = intra_epoch1_val_lat_gL1 if epoch == 0 else intra_epoch2_val_lat_gL1

                c_mR2_map = intra_epoch1_val_map_mR2 if epoch == 0 else intra_epoch2_val_map_mR2
                c_gR2_map = intra_epoch1_val_map_gR2 if epoch == 0 else intra_epoch2_val_map_gR2
                c_sCorr_map = intra_epoch1_val_map_sCorr if epoch == 0 else intra_epoch2_val_map_sCorr
                c_tCorr_map = intra_epoch1_val_map_tCorr if epoch == 0 else intra_epoch2_val_map_tCorr
                c_gCorr_map = intra_epoch1_val_map_gCorr if epoch == 0 else intra_epoch2_val_map_gCorr
                c_gL1_map = intra_epoch1_val_map_gL1 if epoch == 0 else intra_epoch2_val_map_gL1
                c_mL1_map = intra_epoch1_val_map_mL1 if epoch == 0 else intra_epoch2_val_map_mL1

                c_test_losses = intra_epoch1_test_losses if epoch == 0 else intra_epoch2_test_losses
                c_mR2_test = intra_epoch1_test_lat_mR2 if epoch == 0 else intra_epoch2_test_lat_mR2
                c_gR2_test = intra_epoch1_test_lat_gR2 if epoch == 0 else intra_epoch2_test_lat_gR2
                c_mCorr_test = intra_epoch1_test_lat_mCorr if epoch == 0 else intra_epoch2_test_lat_mCorr
                c_gCorr_test = intra_epoch1_test_lat_gCorr if epoch == 0 else intra_epoch2_test_lat_gCorr
                c_mk_test = intra_epoch1_test_lat_mk if epoch == 0 else intra_epoch2_test_lat_mk
                c_gK_test = intra_epoch1_test_lat_gk if epoch == 0 else intra_epoch2_test_lat_gk
                c_mL1_test = intra_epoch1_test_lat_mL1 if epoch == 0 else intra_epoch2_test_lat_mL1
                c_gL1_test = intra_epoch1_test_lat_gL1 if epoch == 0 else intra_epoch2_test_lat_gL1

                c_mR2_map_test = intra_epoch1_test_map_mR2 if epoch == 0 else intra_epoch2_test_map_mR2
                c_gR2_map_test = intra_epoch1_test_map_gR2 if epoch == 0 else intra_epoch2_test_map_gR2
                c_sCorr_map_test = intra_epoch1_test_map_sCorr if epoch == 0 else intra_epoch2_test_map_sCorr
                c_tCorr_map_test = intra_epoch1_test_map_tCorr if epoch == 0 else intra_epoch2_test_map_tCorr
                c_gCorr_map_test = intra_epoch1_test_map_gCorr if epoch == 0 else intra_epoch2_test_map_gCorr
                c_gL1_map_test = intra_epoch1_test_map_gL1 if epoch == 0 else intra_epoch2_test_map_gL1
                c_mL1_map_test = intra_epoch1_test_map_mL1 if epoch == 0 else intra_epoch2_test_map_mL1
                
                loss_first_epoch(c_losses, [np.mean(epoch1_baseline_losses)]*len(c_losses), outdir, label="Intra-Val", batch_indexes=c_steps, epoch_num=epoch+1, batch_test_losses=c_test_losses)
                epoch_1 = True if epoch == 0 else False
                epoch_2 = True if epoch == 1 else False

                plot_correlation_evolution([], c_mCorr, outdir, val_ks=c_mk, test_corrs=c_mCorr_test, test_ks=c_mk_test, epoch_1=epoch_1, epoch_2=epoch_2, batch_indexes=c_steps, suffix="_Latent_Mean")
                plot_correlation_evolution([], c_gCorr, outdir, val_ks=c_gK, test_corrs=c_gCorr_test, test_ks=c_gK_test, epoch_1=epoch_1, epoch_2=epoch_2, batch_indexes=c_steps, suffix="_Latent_Global")
                plot_r2_R2_evolution([], c_mCorr, [], c_mR2, outdir, epoch_1=epoch_1, epoch_2=epoch_2, batch_indexes=c_steps, test_R2=c_mR2_test, test_corrs=c_mCorr_test, suffix="_Latent_Mean")
                plot_r2_R2_evolution([], c_gCorr, [], c_gR2, outdir, epoch_1=epoch_1, epoch_2=epoch_2, batch_indexes=c_steps, test_R2=c_gR2_test, test_corrs=c_gCorr_test, suffix="_Latent_Global")
                plot_latent_l1_ss_evolution([], c_gL1, [], c_mL1, outdir, test_g=c_gL1_test, test_m=c_mL1_test, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1)

                plot_map_r2_evolution([], c_gR2_map, [], c_mR2_map, outdir, test_map=c_gR2_map_test, test_pix=c_mR2_map_test, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1, norm="l2")
                plot_map_r2_evolution([], c_gL1_map, [], c_mL1_map, outdir, test_map=c_gL1_map_test, test_pix=c_mL1_map_test, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1, norm="l1")
                plot_spatial_corr_evolution([], c_sCorr_map, [], c_tCorr_map, [], c_gCorr_map, outdir, test_sc=c_sCorr_map_test, test_tc=c_tCorr_map_test, test_gc=c_gCorr_map_test, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1)

        # ---------------- VALIDATION & TEST ----------------
        per_member_loss = defaultdict(lambda: {'loss_sum': 0.0, 'count': 0})
        per_member_plots = defaultdict(lambda: {'time': [], 'slp_true': [], 'slp_recon_true': [], 'slp_pred': []})
        eval_phases = ['val', 'test'] if nb_members_test > 0 else ['val']
        
        for key in eval_phases:
            loader = valloader if key == 'val' else testloader
            model.eval()
            
            running_val_loss = 0.0
            total_val_samples = 0 
            eval_latent_tracker = LatentMetricTracker(device=device)
            eval_map_tracker = MapMetricTracker(shape=(53, 113), device=device)

            with torch.no_grad():
                for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(loader):
                    X_sst = X_sst.to(device, non_blocking=True)
                    if len(active_slp_lags) > 0: X_slp = X_slp.to(device, non_blocking=True)
                    
                    X_combined = format_inputs(X_sst, X_slp)
                    target_embed = encode_to_latent_gpu(y_target.to(device, non_blocking=True), 'pca', args.latent_dim, slp_pca_components_gpu, slp_pca_mean_gpu, wgts_slp_gpu, None)

                    predicted_latent = model(X_combined)
                    loss_value = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')
                    running_val_loss += loss_value.item() * X_sst.size(0)
                    total_val_samples += X_sst.size(0)

                    per_sample_losses = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='none').cpu().numpy()
                    median_pred_latent = get_median_prediction(predicted_latent, args.loss_type, args.quantiles, args.latent_dim) if args.loss_type == 'quantile' else predicted_latent

                    eval_latent_tracker.update(median_pred_latent.detach(), target_embed.detach())
                    decoded_eval_map = decode_latent_to_map(median_pred_latent, args, args.latent_dim, slp_pca_model, None, safe_wgts_slp)
                    eval_map_tracker.update(y_target.detach(), decoded_eval_map.detach())

                    if key == 'val':
                        pred_np = median_pred_latent.cpu().numpy()
                        target_np = target_embed.cpu().numpy()
                        padded_pred = np.zeros((pred_np.shape[0], slp_pca_model.n_components_))
                        padded_target = np.zeros((target_np.shape[0], slp_pca_model.n_components_))
                        padded_pred[:, :args.latent_dim] = pred_np; padded_target[:, :args.latent_dim] = target_np

                        predicted_slp_flat_polluted = slp_pca_model.inverse_transform(padded_pred)
                        recon_true_slp_flat_polluted = slp_pca_model.inverse_transform(padded_target)
                        
                        if args.lat_weight and safe_wgts_slp is not None:
                            predicted_slp_flat = predicted_slp_flat_polluted / safe_wgts_slp
                            recon_true_slp_flat = recon_true_slp_flat_polluted / safe_wgts_slp
                        else:
                            predicted_slp_flat = predicted_slp_flat_polluted
                            recon_true_slp_flat = recon_true_slp_flat_polluted

                        predicted_slp = predicted_slp_flat.reshape(-1, 53, 113) 
                        recon_true_slp = recon_true_slp_flat.reshape(-1, 53, 113)
                        y_map_np = y_map.numpy()
                        members_list = [m if isinstance(m, str) else m.item().decode() if isinstance(m.item(), bytes) else str(m.item()) for m in members]
                        dates_list = [d if isinstance(d, str) else str(d) for d in dates]

                        for i, mem in enumerate(members_list):
                            current_idx = per_member_loss[mem]['count']
                            per_member_loss[mem]['loss_sum'] += float(per_sample_losses[i])
                            per_member_loss[mem]['count'] += 1
                            if current_idx in target_indices:
                                per_member_plots[mem]['time'].append(dates_list[i])
                                per_member_plots[mem]['slp_true'].append(y_map_np[i])
                                per_member_plots[mem]['slp_recon_true'].append(recon_true_slp[i])
                                per_member_plots[mem]['slp_pred'].append(predicted_slp[i])

            if key == 'val':
                for mem, d in per_member_loss.items():
                    avg_loss = d['loss_sum'] / d['count'] if d['count'] > 0 else float('nan')
                    val_losses_per_member_history[mem].append(avg_loss)

            val_loss = running_val_loss / total_val_samples 
            val_losses.append(val_loss) if key == 'val' else test_losses.append(val_loss)
            
            lat_gR2, lat_mR2, lat_gCorr, lat_mCorr, lat_gK, lat_mK, lat_gL1, lat_mL1 = eval_latent_tracker.compute()
            map_gR2, map_mR2, map_r2_np, map_sCorr, map_tCorr, map_gCorr, map_corr_np, map_gL1, map_mL1, map_l1_np = eval_map_tracker.compute(area_weights=area_weights_2d)
            
            if key == 'val':
                val_map_gR2.append(map_gR2); val_map_mR2.append(map_mR2)
                val_map_sCorr.append(map_sCorr); val_map_tCorr.append(map_tCorr)
                val_map_gCorr.append(map_gCorr); val_map_gL1.append(map_gL1)
                val_map_mL1.append(map_mL1)
                val_lat_gR2.append(lat_gR2); val_lat_mR2.append(lat_mR2)
                val_lat_gCorr.append(lat_gCorr); val_lat_mCorr.append(lat_mCorr)
                val_lat_mk.append(lat_mK); val_lat_gk.append(lat_gK)
                val_lat_gL1.append(lat_gL1); val_lat_mL1.append(lat_mL1)
                save_r2_pixel_map_and_plot(map_r2_np, outdir, f"L2_val_ep{epoch+1}", metric_type="l2")
                save_r2_pixel_map_and_plot(map_l1_np, outdir, f"L1_val_ep{epoch+1}", metric_type="l1")
                save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_val_ep{epoch+1}", metric_type="corr")
            else:
                test_map_gR2.append(map_gR2); test_map_mR2.append(map_mR2)
                test_map_sCorr.append(map_sCorr); test_map_tCorr.append(map_tCorr)
                test_map_gCorr.append(map_gCorr); test_map_gL1.append(map_gL1)
                test_map_mL1.append(map_mL1)
                test_lat_gR2.append(lat_gR2); test_lat_mR2.append(lat_mR2)
                test_lat_gCorr.append(lat_gCorr); test_lat_mCorr.append(lat_mCorr)
                test_lat_mk.append(lat_mK); test_lat_gk.append(lat_gK)
                test_lat_gL1.append(lat_gL1); test_lat_mL1.append(lat_mL1)
                save_r2_pixel_map_and_plot(map_r2_np, outdir, f"L2_test_ep{epoch+1}", metric_type="l2")
                save_r2_pixel_map_and_plot(map_l1_np, outdir, f"L1_test_ep{epoch+1}", metric_type="l1")
                save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_test_ep{epoch+1}", metric_type="corr")
            
            print(f'Epoch {epoch + 1} {key} Loss: {val_loss:.6f} | {key} Latent Mean R2: {lat_mR2:.4f} | {key} Latent Mean Corr: {lat_mCorr:.4f}')

            if key == 'val':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    if best_model_path and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    best_model_path = os.path.join(outdir, f'best_val_LinReg_ep{epoch + 1}_end_loss{best_val_loss:.4f}.pth')
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1

        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)

        if patience_counter >= patience:
            break
        
        # ---------------- AFFICHAGE CHAK 2 EPOCHS ----------------
        if (epoch + 1) % 2 == 0:
            state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses}
            torch.save(state, f'{outdir}/final_model_LinReg.pth')
            loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times, per_member_val_losses=val_losses_per_member_history, test_losses=test_losses)
                        
            plot_r2_R2_evolution(train_lat_mCorr, val_lat_mCorr, train_lat_mR2, val_lat_mR2, outdir, test_R2=test_lat_mR2, test_corrs=test_lat_mCorr, suffix="_Latent_Mean")
            plot_r2_R2_evolution(train_lat_gCorr, val_lat_gCorr, train_lat_gR2, val_lat_gR2, outdir, test_R2=test_lat_gR2, test_corrs=test_lat_gCorr, suffix="_Latent_Global")
            plot_correlation_evolution(train_lat_mCorr, val_lat_mCorr, outdir, test_corrs=test_lat_mCorr, train_ks=train_lat_mk, val_ks=val_lat_mk, test_ks=test_lat_mk, suffix="_Latent_Mean")
            plot_correlation_evolution(train_lat_gCorr, val_lat_gCorr, outdir, test_corrs=test_lat_gCorr, train_ks=train_lat_gk, val_ks=val_lat_gk, test_ks=test_lat_gk, suffix="_Latent_Global")
            plot_latent_l1_ss_evolution(train_lat_gL1, val_lat_gL1, train_lat_mL1, val_lat_mL1, outdir, test_g=test_lat_gL1 if nb_members_test > 0 else None, test_m=test_lat_mL1 if nb_members_test > 0 else None)
            
            plot_map_r2_evolution(train_map_gR2, val_map_gR2, train_map_mR2, val_map_mR2, outdir, test_map=test_map_gR2, test_pix=test_map_mR2, norm="l2")
            plot_map_r2_evolution(train_map_gL1, val_map_gL1, train_map_mL1, val_map_mL1, outdir, test_map=test_map_gL1, test_pix=test_map_mL1, norm="l1")
            plot_spatial_corr_evolution(train_map_sCorr, val_map_sCorr, train_map_tCorr, val_map_tCorr, train_map_gCorr, val_map_gCorr, outdir, test_sc=test_map_sCorr, test_tc=test_map_tCorr, test_gc=test_map_gCorr)
            
            for mem, d in per_member_plots.items():
                member_outdir = os.path.join(outdir, "per_member", mem)
                os.makedirs(member_outdir, exist_ok=True)
                plot_and_save_maps_with_reconstruction_light(
                    slp_true_list=[np.array(d['slp_true'])],
                    slp_recon_true_list=[np.array(d['slp_recon_true'])],
                    slp_pred_list=[np.array(d['slp_pred'])],
                    time_list=d['time'], outdir=member_outdir, epoch=(epoch + 1)
                )
        
    print(f"Best Val Loss : {best_val_loss:.6f}")

    # Sauvegarde et plots finaux
    loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times, per_member_val_losses=val_losses_per_member_history, test_losses=test_losses)

    plot_r2_R2_evolution(train_lat_mCorr, val_lat_mCorr, train_lat_mR2, val_lat_mR2, outdir, test_R2=test_lat_mR2, test_corrs=test_lat_mCorr, suffix="_Latent_Mean")
    plot_r2_R2_evolution(train_lat_gCorr, val_lat_gCorr, train_lat_gR2, val_lat_gR2, outdir, test_R2=test_lat_gR2, test_corrs=test_lat_gCorr, suffix="_Latent_Global")
    plot_correlation_evolution(train_lat_mCorr, val_lat_mCorr, outdir, test_corrs=test_lat_mCorr, train_ks=train_lat_mk, val_ks=val_lat_mk, test_ks=test_lat_mk, suffix="_Latent_Mean")
    plot_correlation_evolution(train_lat_gCorr, val_lat_gCorr, outdir, test_corrs=test_lat_gCorr, train_ks=train_lat_gk, val_ks=val_lat_gk, test_ks=test_lat_gk, suffix="_Latent_Global")
    plot_latent_l1_ss_evolution(train_lat_gL1, val_lat_gL1, train_lat_mL1, val_lat_mL1, outdir, test_g=test_lat_gL1 if nb_members_test > 0 else None, test_m=test_lat_mL1 if nb_members_test > 0 else None)

    plot_map_r2_evolution(train_map_gR2, val_map_gR2, train_map_mR2, val_map_mR2, outdir, test_map=test_map_gR2 if nb_members_test > 0 else None, test_pix=test_map_mR2 if nb_members_test > 0 else None, norm="l2")
    plot_map_r2_evolution(train_map_gL1, val_map_gL1, train_map_mL1, val_map_mL1, outdir, test_map=test_map_gL1 if nb_members_test > 0 else None, test_pix=test_map_mL1 if nb_members_test > 0 else None, norm="l1")
    plot_spatial_corr_evolution(train_map_sCorr, val_map_sCorr, train_map_tCorr, val_map_tCorr, train_map_gCorr, val_map_gCorr, outdir, test_sc=test_map_sCorr, test_tc=test_map_tCorr, test_gc=test_map_gCorr)

    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses}
    torch.save(state, f'{outdir}/final_model_LinReg.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_LinReg.pth')

    # ============================================================
    # EXPLICABILITÉ (Covariances et Régression) EN RAW SPATIAL
    # ============================================================
    print("\n--- CALCUL DE LA VARIABILITÉ SPATIALE ET DES CORRÉLATIONS (PRED & TARGET) ---")
    
    sum_sst, sum_sq_sst = 0.0, 0.0
    sum_slp, sum_sq_slp = 0.0, 0.0
    sum_y_pred, sum_sq_y_pred = 0.0, 0.0
    sum_sst_y_pred, sum_slp_y_pred = 0.0, 0.0
    sum_y_true, sum_sq_y_true = 0.0, 0.0
    sum_sst_y_true, sum_slp_y_true = 0.0, 0.0
    sum_ypred_ytrue = 0.0
    
    n_samples_std = 0
    weights_linear = model.linear.weight.detach().cpu().numpy()

    with torch.no_grad():
        for X_sst, X_slp, y_target, _, _, _ in valloader:
            B = X_sst.size(0)
            n_samples_std += B
            
            # Les corrélations doivent toujours se faire sur les pixels RAW spatiaux
            X_sst_np = X_sst.cpu().numpy()
            sum_sst += X_sst_np.sum(axis=0); sum_sq_sst += (X_sst_np ** 2).sum(axis=0)
            X_sst_flat = X_sst_np.reshape(B, -1)
            
            if X_slp.numel() > 0:
                X_slp_np = X_slp.cpu().numpy()
                sum_slp += X_slp_np.sum(axis=0); sum_sq_slp += (X_slp_np ** 2).sum(axis=0)
                X_slp_flat = X_slp_np.reshape(B, -1)
            else:
                X_slp_np = None

            # Prediction
            X_sst_torch = X_sst.to(device)
            X_slp_torch = X_slp.to(device) if X_slp_np is not None else None
            X_combined = format_inputs(X_sst_torch, X_slp_torch)
            Y_pred = model(X_combined).cpu().numpy()
            
            sum_y_pred += Y_pred.sum(axis=0); sum_sq_y_pred += (Y_pred ** 2).sum(axis=0)
            
            # Target
            target_embed = encode_to_latent_gpu(y_target.to(device, non_blocking=True), 'pca', args.latent_dim, slp_pca_components_gpu, slp_pca_mean_gpu, wgts_slp_gpu, None)
            Y_true_latent = target_embed.cpu().numpy()
            sum_y_true += Y_true_latent.sum(axis=0); sum_sq_y_true += (Y_true_latent ** 2).sum(axis=0)
            
            # Accumulations Croisées
            sum_sst_y_pred += X_sst_flat.T @ Y_pred; sum_sst_y_true += X_sst_flat.T @ Y_true_latent
            sum_ypred_ytrue += (Y_pred * Y_true_latent).sum(axis=0)

            if X_slp_np is not None:
                sum_slp_y_pred += X_slp_flat.T @ Y_pred; sum_slp_y_true += X_slp_flat.T @ Y_true_latent

    # Stats 1D/2D
    sst_mean = sum_sst / n_samples_std; sst_std = np.sqrt(np.maximum((sum_sq_sst / n_samples_std) - (sst_mean ** 2), 0))
    sst_mean_flat = sst_mean.reshape(-1, 1)
    
    y_pred_mean = sum_y_pred / n_samples_std; y_pred_std = np.sqrt(np.maximum((sum_sq_y_pred / n_samples_std) - (y_pred_mean ** 2), 0))
    y_pred_mean_flat = y_pred_mean.reshape(1, -1)
    
    y_true_mean = sum_y_true / n_samples_std; y_true_std = np.sqrt(np.maximum((sum_sq_y_true / n_samples_std) - (y_true_mean ** 2), 0))
    y_true_mean_flat = y_true_mean.reshape(1, -1)

    cov_ypred_ytrue = (sum_ypred_ytrue / n_samples_std) - (y_pred_mean * y_true_mean)
    corr_model = np.divide(cov_ypred_ytrue, (y_pred_std * y_true_std), out=np.zeros_like(cov_ypred_ytrue), where=(y_pred_std * y_true_std)!=0)
    
    cov_sst_ypred = (sum_sst_y_pred / n_samples_std) - (sst_mean_flat @ y_pred_mean_flat)
    denom_sst_pred = sst_std.reshape(-1, 1) @ y_pred_std.reshape(1, -1)
    corr_sst_pred = np.divide(cov_sst_ypred, denom_sst_pred, out=np.zeros_like(cov_sst_ypred), where=denom_sst_pred!=0)
    
    cov_sst_ytrue = (sum_sst_y_true / n_samples_std) - (sst_mean_flat @ y_true_mean_flat)
    denom_sst_true = sst_std.reshape(-1, 1) @ y_true_std.reshape(1, -1)
    corr_sst_target = np.divide(cov_sst_ytrue, denom_sst_true, out=np.zeros_like(cov_sst_ytrue), where=denom_sst_true!=0)
    
    if len(active_slp_lags) > 0:
        slp_mean = sum_slp / n_samples_std; slp_std = np.sqrt(np.maximum((sum_sq_slp / n_samples_std) - (slp_mean ** 2), 0))
        slp_mean_flat = slp_mean.reshape(-1, 1)
        
        cov_slp_ypred = (sum_slp_y_pred / n_samples_std) - (slp_mean_flat @ y_pred_mean_flat)
        denom_slp_pred = slp_std.reshape(-1, 1) @ y_pred_std.reshape(1, -1)
        corr_slp_pred = np.divide(cov_slp_ypred, denom_slp_pred, out=np.zeros_like(cov_slp_ypred), where=denom_slp_pred!=0)
        
        cov_slp_ytrue = (sum_slp_y_true / n_samples_std) - (slp_mean_flat @ y_true_mean_flat)
        denom_slp_true = slp_std.reshape(-1, 1) @ y_true_std.reshape(1, -1)
        corr_slp_target = np.divide(cov_slp_ytrue, denom_slp_true, out=np.zeros_like(cov_slp_ytrue), where=denom_slp_true!=0)
    else:
        slp_std, corr_slp_pred, corr_slp_target, cov_slp_ypred, cov_slp_ytrue = None, None, None, None, None

    # --- REPROJECTION SPATIALE DES POIDS LINEAIRES ---
    sst_weights_spatial = np.zeros((out_features, len(active_sst_lags), 85 * 360))
    slp_weights_spatial = np.zeros((out_features, len(active_slp_lags), 53 * 113))

    if args.input_format == 'pca':
        sst_comps = sst_pca_model.components_[:args.sst_pca_dim]
        slp_comps = slp_pca_model.components_[:1]
        
        for c in range(out_features):
            for l in range(len(active_sst_lags)):
                w_pca = weights_linear[c, l * args.sst_pca_dim : (l+1) * args.sst_pca_dim]
                w_spat = w_pca @ sst_comps
                if args.lat_weight and wgts_sst_flat is not None:
                    w_spat = w_spat * wgts_sst_flat
                sst_weights_spatial[c, l, :] = w_spat
                
            for l in range(len(active_slp_lags)):
                offset = len(active_sst_lags) * args.sst_pca_dim
                w_pca = weights_linear[c, offset + l : offset + l + 1]
                w_spat = w_pca @ slp_comps
                if args.lat_weight and wgts_slp_flat is not None:
                    w_spat = w_spat * wgts_slp_flat
                slp_weights_spatial[c, l, :] = w_spat
    else:
        sst_w_raw = weights_linear[:, :len(active_sst_lags)*30600]
        sst_weights_spatial = sst_w_raw.reshape(out_features, len(active_sst_lags), 30600)
        if len(active_slp_lags) > 0:
            slp_w_raw = weights_linear[:, len(active_sst_lags)*30600:]
            slp_weights_spatial = slp_w_raw.reshape(out_features, len(active_slp_lags), 5989)

    print("\n--- GÉNÉRATION DES CARTES D'EXPLICABILITÉ ---")
    
    def plot_explainability_separated_rows(outdir, sst_lags, slp_lags, sst_std, slp_std, 
                                           sst_weights_spat, slp_weights_spat,
                                           corr_sst_p, corr_sst_t, corr_slp_p, corr_slp_t, 
                                           cov_sst_p, cov_sst_t, cov_slp_p, cov_slp_t, corr_model,
                                           y_true_std, sst_shape=(85, 360), slp_shape=(53, 113), max_components_to_plot=10):
        extent_sst = [-180, 180, -15, 70] if args.roll_sst else [0, 359.9, -15, 70]
        extent_slp = [-100, 40, 20, 70] 
        
        comp_outdir = os.path.join(outdir, "components_explainability")
        os.makedirs(comp_outdir, exist_ok=True)
        num_plots = min(out_features, max_components_to_plot)

        for comp_idx in range(num_plots):
            max_pixel_r = 0.0
            if len(sst_lags) > 0: max_pixel_r = np.max(np.abs(corr_sst_t[:, comp_idx]))
            if len(slp_lags) > 0: max_pixel_r = max(max_pixel_r, np.max(np.abs(corr_slp_t[:, comp_idx])))
            
            model_r = corr_model[comp_idx]
            print(f"Latent {comp_idx:03d} | R Modèle complet: {model_r:.4f} | R Meilleur Pixel: {max_pixel_r:.4f}")

            # ---------------- HELD-OUT FUNCTION ----------------
            def save_single_row_diagnostic(data_3d, title_prefix, mod_name, comp_idx, lags, extent, vmin, vmax, cmap='RdBu_r'):
                n_lags = len(lags)
                fig, axes = plt.subplots(1, n_lags, figsize=(6 * n_lags, 4.5), subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
                if n_lags == 1: axes = [axes]
                for idx, lag in enumerate(lags):
                    ax = axes[idx]
                    im = ax.imshow(data_3d[idx], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent, interpolation='nearest')
                    ax.set_title(rf"{title_prefix} - Lag {-lag}m", fontsize=13)
                    ax.coastlines(color='black', linewidth=0.8)
                fig.colorbar(im, ax=axes, shrink=0.8, orientation='vertical', pad=0.02)
                fig.suptitle(rf"{mod_name} Diagnostic : {title_prefix} - Latent {comp_idx}", fontsize=15, y=1.05)
                save_dir = os.path.join(comp_outdir, mod_name, title_prefix.replace(" ", "_").replace("(", "").replace(")", "").replace("\\", "").replace("$", ""))
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"comp_{comp_idx:03d}.png"), dpi=150, bbox_inches='tight')
                plt.close()

            # L'écart-type est un vecteur 1D, on utilise juste [comp_idx]
            target_std_val = y_true_std[comp_idx] if y_true_std[comp_idx] != 0 else 1.0

            # ---------------- DIAGNOSTICS SST ----------------
            if len(sst_lags) > 0:
                sst_w_raw = sst_weights_spat[comp_idx].reshape(len(sst_lags), sst_shape[0], sst_shape[1])
                # Correction du reshape pour prendre en compte le nombre de lags
                sst_w_eff = sst_w_raw * sst_std.reshape(len(sst_lags), sst_shape[0], sst_shape[1])
                sst_w_norm = sst_w_eff / target_std_val  # Impact Typique en Sigmas
                
                sst_c_pred = corr_sst_p[:, comp_idx].reshape(len(sst_lags), sst_shape[0], sst_shape[1])
                sst_c_targ = corr_sst_t[:, comp_idx].reshape(len(sst_lags), sst_shape[0], sst_shape[1])
                sst_cov_pred = cov_sst_p[:, comp_idx].reshape(len(sst_lags), sst_shape[0], sst_shape[1])
                sst_cov_targ = cov_sst_t[:, comp_idx].reshape(len(sst_lags), sst_shape[0], sst_shape[1])
                
                vmax_raw = np.max(np.abs(sst_w_raw))
                vmax_eff = np.max(np.abs(sst_w_eff))
                vmax_norm = np.max(np.abs(sst_w_norm))
                vmax_cov = max(np.max(np.abs(sst_cov_pred)), np.max(np.abs(sst_cov_targ))) 

                save_single_row_diagnostic(sst_w_raw, "Raw Coefs", "SST", comp_idx, sst_lags, extent_sst, -vmax_raw, vmax_raw)
                save_single_row_diagnostic(sst_w_eff, "Typical Impact (Absolute)", "SST", comp_idx, sst_lags, extent_sst, -vmax_eff, vmax_eff)
                save_single_row_diagnostic(sst_w_norm, "Normalized Typical Impact (Std)", "SST", comp_idx, sst_lags, extent_sst, -vmax_norm, vmax_norm)
                save_single_row_diagnostic(sst_c_pred, "Correlation (Pixel vs Pred)", "SST", comp_idx, sst_lags, extent_sst, -1, 1)
                save_single_row_diagnostic(sst_c_targ, "Correlation (Pixel vs Target)", "SST", comp_idx, sst_lags, extent_sst, -1, 1)

            # ---------------- DIAGNOSTICS SLP ----------------
            if len(slp_lags) > 0:
                slp_w_raw = slp_weights_spat[comp_idx].reshape(len(slp_lags), slp_shape[0], slp_shape[1])
                # Correction du reshape pour prendre en compte le nombre de lags
                slp_w_eff = slp_w_raw * slp_std.reshape(len(slp_lags), slp_shape[0], slp_shape[1])
                slp_w_norm = slp_w_eff / target_std_val 
                
                slp_c_pred = corr_slp_p[:, comp_idx].reshape(len(slp_lags), slp_shape[0], slp_shape[1])
                slp_c_targ = corr_slp_t[:, comp_idx].reshape(len(slp_lags), slp_shape[0], slp_shape[1])
                slp_cov_pred = cov_slp_p[:, comp_idx].reshape(len(slp_lags), slp_shape[0], slp_shape[1])
                slp_cov_targ = cov_slp_t[:, comp_idx].reshape(len(slp_lags), slp_shape[0], slp_shape[1])
                
                vmax_raw_slp = np.max(np.abs(slp_w_raw))
                vmax_eff_slp = np.max(np.abs(slp_w_eff))
                vmax_norm_slp = np.max(np.abs(slp_w_norm))
                vmax_cov_slp = max(np.max(np.abs(slp_cov_pred)), np.max(np.abs(slp_cov_targ)))

                save_single_row_diagnostic(slp_w_raw, "Raw Coefs", "SLP", comp_idx, slp_lags, extent_slp, -vmax_raw_slp, vmax_raw_slp)
                save_single_row_diagnostic(slp_w_eff, "Typical Impact (Absolute)", "SLP", comp_idx, slp_lags, extent_slp, -vmax_eff_slp, vmax_eff_slp)
                save_single_row_diagnostic(slp_w_norm, "Normalized Typical Impact (Std)", "SLP", comp_idx, slp_lags, extent_slp, -vmax_norm_slp, vmax_norm_slp)
                save_single_row_diagnostic(slp_c_pred, "Correlation (Pixel vs Pred)", "SLP", comp_idx, slp_lags, extent_slp, -1, 1)
                save_single_row_diagnostic(slp_c_targ, "Correlation (Pixel vs Target)", "SLP", comp_idx, slp_lags, extent_slp, -1, 1)

    plot_explainability_separated_rows(outdir, active_sst_lags, active_slp_lags, sst_std, slp_std, 
                                       sst_weights_spatial, slp_weights_spatial,
                                       corr_sst_pred, corr_sst_target, corr_slp_pred, corr_slp_target, 
                                       cov_sst_ypred, cov_sst_ytrue, cov_slp_ypred, cov_slp_ytrue, corr_model,
                                       y_true_std, max_components_to_plot=10)

    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation SHAP automatique...")
    print("="*50)

    eval_shap_script_path = os.path.join(os.path.dirname(__file__), "eval_linreg_shap.py")
    
    # On évalue généralement SHAP uniquement sur le "best" model pour gagner du temps
    for model_type in ["best"]:  
        print(f"\n--- Évaluation SHAP du modèle : {model_type} ---")
        eval_shap_command = [
            sys.executable, eval_shap_script_path,
            "--linreg_dir", str(outdir), "--model_type", str(model_type), "--machine", str(args.machine), '--embed_method', str(args.embed_method),"--top_k_components", str(min(args.latent_dim,5)),
            "--nb_members_train", str(args.nb_members_train), "--nb_members_val", str(args.nb_members_val),
            "--nb_members_test", str(args.nb_members_test), "--seed", str(args.seed),
            "--latent_dim", str(args.latent_dim), "--duree_lissage", str(args.duree_lissage),"--loss_type", str(args.loss_type),
            "--input_format", str(args.input_format), "--sst_pca_dim", str(args.sst_pca_dim), "--bs", str(args.bs),
            "--method", "gradient", "--bg_type", "zeros"
        ]

        if args.embed_path: eval_shap_command.extend(["--embed_path", str(args.embed_path)])
        if args.embed_path_sst: eval_shap_command.extend(["--embed_path_sst", str(args.embed_path_sst)])
        if args.sst_lags_days is not None: eval_shap_command.extend(["--sst_lags_days"] + [str(x) for x in args.sst_lags_days])
        if args.slp_lags_days is not None: eval_shap_command.extend(["--slp_lags_days"] + [str(x) for x in args.slp_lags_days])
        if hasattr(args, 'sst_lags_months') and args.sst_lags_months is not None: eval_shap_command.extend(["--sst_lags_months"] + [str(x) for x in args.sst_lags_months])
        if hasattr(args, 'slp_lags_months') and args.slp_lags_months is not None: eval_shap_command.extend(["--slp_lags_months"] + [str(x) for x in args.slp_lags_months])
        if args.winter_months is not None: eval_shap_command.extend(["--winter_months"] + [str(x) for x in args.winter_months])
        if args.quantiles: eval_shap_command.extend(["--quantiles"] + [str(x) for x in args.quantiles])
        # --- TRANSFERT DES ARGUMENTS DE FORÇAGE ---
        if args.force_val_members is not None: eval_shap_command.extend(["--force_val_members"] + args.force_val_members)
        if args.force_test_members is not None: eval_shap_command.extend(["--force_test_members"] + args.force_test_members)
        # ------------------------------------------

        if args.roll_sst: eval_shap_command.append("--roll_sst")
        if args.monthly_reduction: eval_shap_command.append("--monthly_reduction")
        if args.lat_weight: eval_shap_command.append("--lat_weight")

        try:
            subprocess.run(eval_shap_command, check=True, text=True)
            print(f"\n✅ Évaluation SHAP de {model_type} terminée avec succès !")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Erreur lors de l'exécution de l'évaluation SHAP. Code de retour : {e.returncode}")


    # ============================================================
    # LANCEMENT DES ÉVALUATIONS AUTOMATIQUES FINALES
    # ============================================================
    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation automatique...")
    print("="*50)

    eval_script_path = os.path.join(os.path.dirname(__file__), "eval_linreg_embedding.py")
    eval_spatial_script_path = os.path.join(os.path.dirname(__file__), "eval_linreg_spatial.py")

    for model_type in ["final", "best"]:
        for monthly_mean in [False, True]:
            print(f"\n--- Évaluation globale du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            eval_command = [
                sys.executable, eval_script_path,
                "--machine", str(args.machine), "--embed_method", str(args.embed_method),
                "--linreg_dir", str(outdir),
                "--model_type", str(model_type), "--nb_members_train", str(args.nb_members_train),
                "--nb_members_val", str(args.nb_members_val), "--nb_members_test", str(args.nb_members_test), "--seed", str(args.seed),
                "--latent_dim", str(args.latent_dim), "--duree_lissage", str(args.duree_lissage),
                "--bs", str(args.bs)
            ]
            if args.embed_path: eval_command.extend(["--embed_path", str(args.embed_path)])
            if args.embed_path_sst: eval_command.extend(["--embed_path_sst", str(args.embed_path_sst)])
            if args.sst_lags_days: eval_command.extend(["--sst_lags_days"] + [str(x) for x in args.sst_lags_days])
            if args.sst_lags_months: eval_command.extend(["--sst_lags_months"] + [str(x) for x in args.sst_lags_months])
            if args.slp_lags_days: eval_command.extend(["--slp_lags_days"] + [str(x) for x in args.slp_lags_days])
            if args.slp_lags_months: eval_command.extend(["--slp_lags_months"] + [str(x) for x in args.slp_lags_months])
            if args.winter_months: eval_command.extend(["--winter_months"] + [str(x) for x in args.winter_months])
            
            # --- TRANSFERT DES ARGUMENTS D'ENTRAÎNEMENT ---
            eval_command.extend(["--loss_type", str(args.loss_type)])
            eval_command.extend(["--input_format", str(args.input_format)])
            eval_command.extend(["--sst_pca_dim", str(args.sst_pca_dim)])
            if hasattr(args, 'quantiles') and args.quantiles: eval_command.extend(["--quantiles"] + [str(x) for x in args.quantiles])
            
            if args.force_val_members is not None: eval_command.extend(["--force_val_members"] + args.force_val_members)
            if args.force_test_members is not None: eval_command.extend(["--force_test_members"] + args.force_test_members)
            if args.roll_sst: eval_command.append("--roll_sst")
            if monthly_mean: eval_command.append("--monthly_mean")
            if args.monthly_reduction: eval_command.append("--monthly_reduction")
            if args.lat_weight: eval_command.append("--lat_weight")

            if os.path.exists(eval_script_path):
                try:
                    subprocess.run(eval_command, check=True, text=True)
                    print(f"✅ Évaluation de {model_type} terminée avec succès !")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Erreur lors de l'exécution de l'évaluation globale. Code: {e.returncode}")

            # --- EVAL SPATIALE ---
            print(f"\n--- Évaluation spatiale du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            eval_spatial_command = eval_command.copy()
            eval_spatial_command[1] = eval_spatial_script_path

            if os.path.exists(eval_spatial_script_path):
                try:
                    subprocess.run(eval_spatial_command, check=True, text=True)
                    print(f"✅ Évaluation spatiale de {model_type} terminée avec succès !")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Erreur lors de l'exécution de l'évaluation spatiale. Code: {e.returncode}")