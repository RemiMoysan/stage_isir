import os
import time
import argparse
import copy
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import random 
import re
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import cartopy.crs as ccrs

import sys 
from pathlib import Path
import subprocess

project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

# Utilisation stricte des mêmes outils que pour le CNN
from shared_tools.visualizations import (
    loss_figure, loss_first_epoch, plot_and_save_maps_with_reconstruction_light, 
    plot_reconstruction_check, plot_correlation_evolution, plot_r2_R2_evolution, 
    MapMetricTracker, LatentMetricTracker, save_r2_pixel_map_and_plot, 
    plot_map_r2_evolution, plot_spatial_corr_evolution, plot_latent_l1_ss_evolution
)
from shared_tools.datasets import Dataset, Dataset_mensuel
from shared_tools.models import (
    ConvVAE, vae_loss, compute_loss, get_median_prediction_full_slp, 
    decode_latent_to_map, spatial_penalty_tikhonov, spatial_penalty_laplacian
)

# ============================================================
# MODÈLE DE RÉGRESSION LINÉAIRE
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
# DEVICE & ARGUMENTS CONFIGURATION & OUTPUT DIRECTORY SETUP
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', type=int, required=False, default=0, help='Loading of previous parameters (1) or start fresh (0)') 

    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca', help='Méthode pour l\'espace latent')
    parser.add_argument('--embed_path', type=str, default='', help='Chemin vers le VAE/PCA pré-entraîné.')
    parser.add_argument('--machine', type=str, default='mac_local', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    
    parser.add_argument('--nb_members_train', type=int, default=10)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_members_test', type=int, default=0, help='Nombre de membres à utiliser pour le test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--alpha_penalty', type=float, default=1e-5, help='Poids de la pénalité L1, L2 ou spatiale')
    parser.add_argument('--penalty_type', type=str, choices=['none', 'l1', 'l2', 'tikhonov', 'laplacian'], default='l2', help='Type de régularisation à appliquer aux poids')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--beta_kld', type=float, default=1.0)
    parser.add_argument('--normalize', action='store_true')

    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--roll_sst', action='store_true', help='Appliquer un roll sur les données SST')
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Nombre de points de validation intra-époque')
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données sous-échantillonnées mensuellement')
    parser.add_argument('--lat_weight', action='store_true', help='Applique la pondération spatiale sqrt(cos(lat))')

    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile', 'correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    args = parser.parse_args()
    if args.loss_type == 'quantile':
        if 0.5 not in args.quantiles:
            raise ValueError("Erreur: Pour la quantile loss, la liste des quantiles (--quantiles) DOIT inclure la médiane (0.5).")

    # Routage dynamique des dossiers
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/linear_models/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']:
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/"
    elif args.machine == 'mac_local':
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/linear_models/"

    latent_dim = args.latent_dim
    sst_lags_days = args.sst_lags_days
    slp_lags_days = args.slp_lags_days
    sst_lags_months = args.sst_lags_months
    slp_lags_months = args.slp_lags_months
    bs = args.bs
    lr = args.lr
    winter_months = args.winter_months
    duree_lissage = args.duree_lissage
    nb_epochs = args.nb_epochs
    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val
    nb_members_test = args.nb_members_test
    active_sst_lags = sst_lags_months if args.monthly_reduction else sst_lags_days
    active_slp_lags = slp_lags_months if args.monthly_reduction else slp_lags_days

    # ORDRE CHRONOLOGIQUE : On trie dans l'ordre décroissant (ex: 95 -> 65 -> 35 jours / 4 -> 3 -> 2 mois), pour le plot d'explicabilité
    active_sst_lags = sorted(active_sst_lags, reverse=True) 
    active_slp_lags = sorted(active_slp_lags, reverse=True)

    print("Arg Parameters:")
    print(f"  Latent Dim: {latent_dim}", f"SST Lags: {active_sst_lags}", f"SLP Lags: {active_slp_lags}", f"Batch Size: {bs}", f"Learning Rate: {lr}", f"Winter Months: {winter_months}", f"Smoothing Duration: {duree_lissage}", f"Number of Epochs: {nb_epochs}", f"Train Members: {nb_members_train}", f"Val Members: {nb_members_val}\n")

    patience = 10000
    target_indices = {100, 1000, 2000, 3000, 4000, 4500, 5000, 6000, 7000, 8000} if not args.monthly_reduction else {1, 10, 20, 30, 40, 45, 50, 60, 70, 80} 

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[-nb_members_val:]
    test_members = all_members[nb_members_train:nb_members_train + nb_members_test] if nb_members_test > 0 else []

    dynamic_slp_std = 596.0 
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))
            print(f"\n✅ slp_std extrait avec succès du chemin : {dynamic_slp_std}")
        else:
            print(f"\n⚠️ 'slp_std' introuvable dans le nom du dossier. Utilisation du fallback : {dynamic_slp_std}")
    else:
        print(f"\n⚠️ Aucun modèle pré-entraîné fourni. Utilisation du slp_std par défaut : {dynamic_slp_std}")

    loss_tag = args.loss_type
    if args.loss_type == 'quantile':
        loss_tag += "_" + "".join([str(q).replace('.','') for q in args.quantiles])

    base_outdir_name = f"LinReg_loss_{loss_tag}_{args.penalty_type}_{args.alpha_penalty}_{args.embed_method}_emb_{latent_dim}_lat_{args.lat_weight}_norm_{args.normalize}_bs{bs}_lr{lr}_months_{''.join(map(str, winter_months))}_seed{args.seed}_train{nb_members_train}_val{nb_members_val}_{nb_members_test}"
    
    if not args.monthly_reduction:
        outdir_name = f"{base_outdir_name}_sst_{''.join(map(str, active_sst_lags))}_slp_{''.join(map(str, active_slp_lags))}_{duree_lissage}d_roll_{args.roll_sst}_slp_std{dynamic_slp_std}"
    else:
        outdir_name = f"{base_outdir_name}_sst_{''.join(map(str, active_sst_lags))}_slp_{''.join(map(str, active_slp_lags))}_monthly_roll_{args.roll_sst}_slp_std{dynamic_slp_std}"

    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    intra_workers = min(2, n_workers)
    print(f"Using {n_workers} workers for data loading")

    # ============================================================
    # DATALOADERS
    # ============================================================
    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        training_set = Dataset(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        training_set = Dataset_mensuel(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=intra_workers, pin_memory=True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    if nb_members_test > 0:
        if not args.monthly_reduction:
            test_set = Dataset(members=test_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        else:
            test_set = Dataset_mensuel(members=test_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
        testloader_intra = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=intra_workers, pin_memory=True)
    else:
        testloader, testloader_intra = None, None

    # ============================================================
    # PRÉPARATION DES POIDS SPATIAUX (POUR DÉCODAGE PCA PONDÉRÉ)
    # ============================================================
    wgts_flat = None
    safe_wgts = None
    area_weights_2d = None  
    if args.lat_weight and args.embed_method == 'pca':
        sample_member = train_members[0]
        sample_path = os.path.join(base_home.replace("stage_isir_jz/linear_models/", ""), f"data/SLP/PSL_anom_LE2-{sample_member}_1mo.nc")
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            h, w = len(lats), len(ds_sample['lon'].values)
            wgts = np.sqrt(coslat).reshape(h, 1)
            wgts_flat = np.broadcast_to(wgts, (h, w)).flatten()
            safe_wgts = np.maximum(wgts_flat, 1e-5)
            area_weights_2d = torch.tensor(np.broadcast_to(coslat.reshape(h, 1), (h, w)), dtype=torch.float64, device=device)
            ds_sample.close()
            print("Grille de poids de latitude générée pour le décodage PCA.")
        except Exception as e:
            print(f"Erreur lors du chargement de la grille de latitude : {e}")

    # ============================================================
    # PRÉPARATION DE L'EMBEDDER (PCA ou VAE)
    # ============================================================
    pca_model = None
    vae_model = None
    start_embed_time = time.time()

    if args.embed_method == 'pca':
        if args.embed_path and os.path.exists(args.embed_path):
            print(f"Loading PCA from {args.embed_path}")
            pca_model = joblib.load(args.embed_path)
        else:
            print("Training PCA from scratch on TrainLoader...")
            pca_model = PCA(n_components=latent_dim, whiten=args.normalize) 
            slp_list = []
            for X_sst, X_slp, y_target, y_map, dates, members in trainloader:
                slp_data_raw = y_target.view(y_target.size(0), -1).numpy()
                if args.lat_weight and wgts_flat is not None:
                    slp_data_raw = slp_data_raw * wgts_flat
                slp_list.append(slp_data_raw)
            slp_data = np.concatenate(slp_list, axis=0)
            pca_model.fit(slp_data)
            joblib.dump(pca_model, os.path.join(outdir, "pca_model.joblib"))

    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=latent_dim).to(device)
        if args.embed_path and os.path.exists(args.embed_path):
            print(f"Loading VAE model from {args.embed_path}")
            vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        else:
            print("Training VAE from scratch on TrainLoader (10 epochs)...")
            optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
            vae_model.train()
            for v_epoch in range(10): 
                total_loss = 0
                for X_sst, X_slp, y_target, y_map, dates, members in trainloader:
                    y_target = y_target.to(device)
                    optimizer_vae.zero_grad()
                    recon, mu, logvar = vae_model(y_target)
                    loss = vae_loss(recon, y_target, mu, logvar, beta=args.beta_kld)
                    loss.backward()
                    optimizer_vae.step()
                    total_loss += loss.item()
                print(f"VAE Epoch {v_epoch+1}/10, Loss: {total_loss/len(trainloader):.2f}")
            torch.save(vae_model.state_dict(), os.path.join(outdir, "vae_model.pth"))
            print("VAE training complete")
        vae_model.eval()
        for param in vae_model.parameters():
            param.requires_grad = False

    # ============================================================
    # SANITY CHECK DE L'EMBEDDER (PCA ou VAE)
    # ============================================================
    print("\n--- VÉRIFICATION DE LA QUALITÉ DE RECONSTRUCTION ---")
    X_sst_val, X_slp_val, y_target_val, y_map_val, dates_val, members_val = next(iter(valloader))

    if args.embed_method == 'pca':
        explained_var = np.sum(pca_model.explained_variance_ratio_)
        print(f"-> Variance expliquée par les {latent_dim} composantes PCA : {explained_var * 100:.2f}%")
        
        slp_flat_val = y_target_val.view(y_target_val.size(0), -1).cpu().numpy()
        if wgts_flat is not None:
            slp_flat_val *= wgts_flat
            
        latent_val = pca_model.transform(slp_flat_val)[:, :latent_dim]
        padded_latent = np.zeros((latent_val.shape[0], pca_model.n_components_))
        padded_latent[:, :latent_dim] = latent_val 
        
        recon_flat_val = pca_model.inverse_transform(padded_latent)
        if wgts_flat is not None:
            recon_flat_val /= safe_wgts

        recon_slp_val = recon_flat_val.reshape(-1, 1, 53, 113) 
        true_slp_val = y_target_val.cpu().numpy()

    elif args.embed_method == 'vae':
        y_target_val = y_target_val.to(device)
        with torch.no_grad():
            recon_slp_tensor, _, _ = vae_model(y_target_val)
        recon_slp_val = recon_slp_tensor.cpu().numpy()
        true_slp_val = y_target_val.cpu().numpy()

    rmse = np.sqrt(np.mean((true_slp_val - recon_slp_val)**2))
    print(f"-> Erreur RMSE moyenne de reconstruction : {rmse:.4f}")

    end_embed_time = time.time()
    print(f"Embedding completed in {(end_embed_time - start_embed_time) / 60:.2f} minutes")

    plot_reconstruction_check(true_slp_val, recon_slp_val, dates_val, outdir, args.embed_method, num_samples=min(10, true_slp_val.shape[0]))
    print("-> Plot de vérification sauvegardé.\n")

    # ============================================================
    # INITIALISATION DU MODÈLE DE RÉGRESSION ET SUIVI STANDARD
    # ============================================================
    out_features = latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else latent_dim

    model = LinearRegressionPredictor(
        sst_shape=(85, 360), 
        slp_shape=(53, 113), 
        in_chans_sst=len(active_sst_lags), 
        in_chans_slp=len(active_slp_lags), 
        out_dim=out_features
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Number of Linear parameters : {sum(p.numel() for p in model.parameters())}")

    ### SUIVI STANDARD IDENTIQUE AU CNN
    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float('inf') 
    best_model_path = ""
    val_losses_per_member_history = defaultdict(list)

    # Listes pour l'Espace Latent (Moyenné et Global)
    train_lat_mR2, val_lat_mR2, test_lat_mR2 = [], [], []
    train_lat_gR2, val_lat_gR2, test_lat_gR2 = [], [], []
    train_lat_mCorr, val_lat_mCorr, test_lat_mCorr = [], [], []
    train_lat_gCorr, val_lat_gCorr, test_lat_gCorr = [], [], []
    train_lat_mk, val_lat_mk, test_lat_mk = [], [], []
    train_lat_gk, val_lat_gk, test_lat_gk = [], [], []
    train_lat_gL1, val_lat_gL1, test_lat_gL1 = [], [], []
    train_lat_mL1, val_lat_mL1, test_lat_mL1 = [], [], []
    
    # Listes pour l'Espace Spatial (R2)
    train_map_gR2, val_map_gR2, test_map_gR2 = [], [], []
    train_map_mR2, val_map_mR2, test_map_mR2 = [], [], []
    
    # Listes pour l'Espace Spatial (Corrélations)
    train_map_sCorr, val_map_sCorr, test_map_sCorr = [], [], []
    train_map_tCorr, val_map_tCorr, test_map_tCorr = [], [], []
    train_map_gCorr, val_map_gCorr, test_map_gCorr = [], [], []

    # Listes pour l'Espace Spatial (L1)
    train_map_gL1, val_map_gL1, test_map_gL1 = [], [], []
    train_map_mL1, val_map_mL1, test_map_mL1 = [], [], []

    ### SUIVI INTRA-ÉPOQUE 1
    total_batches = len(trainloader)
    epoch1_batch_losses, epoch1_baseline_losses = [], []
    eval_steps = np.insert(np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int), 0, 0) if args.nb_intra_evals > 0 else np.array([])
    eval_steps_set = set(eval_steps)
    eval_steps_epoch2 = np.insert(np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int), 0, 0) if args.nb_intra_evals > 0 else np.array([])
    eval_steps_epoch2_set = set(eval_steps_epoch2)

    intra_epoch1_steps, intra_epoch1_val_losses, intra_epoch1_test_losses = [], [], []
    intra_epoch1_train_lat_mR2, intra_epoch1_val_lat_mR2, intra_epoch1_test_lat_mR2 = [], [], []
    intra_epoch1_train_lat_gR2, intra_epoch1_val_lat_gR2, intra_epoch1_test_lat_gR2 = [], [], []
    intra_epoch1_train_lat_mCorr, intra_epoch1_val_lat_mCorr, intra_epoch1_test_lat_mCorr = [], [], []
    intra_epoch1_train_lat_gCorr, intra_epoch1_val_lat_gCorr, intra_epoch1_test_lat_gCorr = [], [], []
    intra_epoch1_train_lat_mk, intra_epoch1_val_lat_mk, intra_epoch1_test_lat_mk = [], [], [] 
    intra_epoch1_train_lat_gk, intra_epoch1_val_lat_gk, intra_epoch1_test_lat_gk = [], [], []
    intra_epoch1_train_lat_gL1, intra_epoch1_val_lat_gL1, intra_epoch1_test_lat_gL1 = [], [], []
    intra_epoch1_train_lat_mL1, intra_epoch1_val_lat_mL1, intra_epoch1_test_lat_mL1 = [], [], []
    intra_epoch1_train_map_gR2, intra_epoch1_val_map_gR2, intra_epoch1_test_map_gR2 = [], [], []
    intra_epoch1_train_map_mR2, intra_epoch1_val_map_mR2, intra_epoch1_test_map_mR2 = [], [], [] 
    intra_epoch1_train_map_sCorr, intra_epoch1_val_map_sCorr, intra_epoch1_test_map_sCorr = [], [], []
    intra_epoch1_train_map_tCorr, intra_epoch1_val_map_tCorr, intra_epoch1_test_map_tCorr = [], [], []
    intra_epoch1_train_map_gCorr, intra_epoch1_val_map_gCorr, intra_epoch1_test_map_gCorr = [], [], []
    intra_epoch1_train_map_gL1, intra_epoch1_val_map_gL1, intra_epoch1_test_map_gL1 = [], [], []
    intra_epoch1_train_map_mL1, intra_epoch1_val_map_mL1, intra_epoch1_test_map_mL1 = [], [], []

    ### SUIVI INTRA-ÉPOQUE 2
    intra_epoch2_steps, intra_epoch2_val_losses, intra_epoch2_test_losses = [], [], []
    intra_epoch2_train_lat_mR2, intra_epoch2_val_lat_mR2, intra_epoch2_test_lat_mR2 = [], [], []
    intra_epoch2_train_lat_gR2, intra_epoch2_val_lat_gR2, intra_epoch2_test_lat_gR2 = [], [], []
    intra_epoch2_train_lat_mCorr, intra_epoch2_val_lat_mCorr, intra_epoch2_test_lat_mCorr = [], [], []
    intra_epoch2_train_lat_gCorr, intra_epoch2_val_lat_gCorr, intra_epoch2_test_lat_gCorr = [], [], []
    intra_epoch2_train_lat_mk, intra_epoch2_val_lat_mk, intra_epoch2_test_lat_mk = [], [], [] 
    intra_epoch2_train_lat_gk, intra_epoch2_val_lat_gk, intra_epoch2_test_lat_gk = [], [], []
    intra_epoch2_train_lat_gL1, intra_epoch2_val_lat_gL1, intra_epoch2_test_lat_gL1 = [], [], []
    intra_epoch2_train_lat_mL1, intra_epoch2_val_lat_mL1, intra_epoch2_test_lat_mL1 = [], [], [] 
    intra_epoch2_train_map_gR2, intra_epoch2_val_map_gR2, intra_epoch2_test_map_gR2 = [], [], []
    intra_epoch2_train_map_mR2, intra_epoch2_val_map_mR2, intra_epoch2_test_map_mR2 = [], [], [] 
    intra_epoch2_train_map_sCorr, intra_epoch2_val_map_sCorr, intra_epoch2_test_map_sCorr = [], [], []
    intra_epoch2_train_map_tCorr, intra_epoch2_val_map_tCorr, intra_epoch2_test_map_tCorr = [], [], []
    intra_epoch2_train_map_gCorr, intra_epoch2_val_map_gCorr, intra_epoch2_test_map_gCorr = [], [], []
    intra_epoch2_train_map_gL1, intra_epoch2_val_map_gL1, intra_epoch2_test_map_gL1 = [], [], []
    intra_epoch2_train_map_mL1, intra_epoch2_val_map_mL1, intra_epoch2_test_map_mL1 = [], [], [] 

    if args.update == 1:
        initial_params = torch.load(f"{outdir}/final_model_LinReg.pth")
        model.load_state_dict(initial_params['state_dict'])
        optimizer.load_state_dict(initial_params['optimizer'])
        train_losses = initial_params['train_losses']
        val_losses = initial_params['val_losses']
        test_losses = initial_params.get('test_losses', [])
        best_val_loss = np.min(val_losses) if len(val_losses)>0 else float('inf')
        print("Model state updated")
    else:
        print("Initiated first LinReg training")

    # ============================================================
    # TRAINING & EVALUATION LOOP
    # ============================================================
    start_time = time.time()
    epoch_times = []
    best_model_state = None
    patience_counter = 0

    for epoch in range(nb_epochs):
        # ---------------- TRAINING ----------------
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0
        
        train_latent_tracker = LatentMetricTracker(device=device)
        train_map_tracker = MapMetricTracker(shape=(53, 113), device=device)
        
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
            if batch_idx % 30 == 0:
                print(f" {100 * batch_idx / len(trainloader):.1f}% training complete", end='\r')
                
            optimizer.zero_grad()
            X_sst = X_sst.to(device, non_blocking=True) 
            X_slp = X_slp.to(device, non_blocking=True) 
            
            if args.embed_method == 'pca':
                slp_flat = y_target.view(y_target.size(0), -1).numpy()
                if args.lat_weight and wgts_flat is not None:
                    slp_flat = slp_flat * wgts_flat
                embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
            elif args.embed_method == 'vae':
                y_target = y_target.to(device, non_blocking=True) 
                with torch.no_grad():
                    target_embed, _ = vae_model.encode(y_target)
                    
            predicted_latent = model(X_sst, X_slp)            
            base_loss = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')

            if args.penalty_type == 'l1':
                penalty = torch.norm(model.linear.weight, p=1)
            elif args.penalty_type == 'l2':
                penalty = torch.sum(model.linear.weight ** 2)
            elif args.penalty_type in ['tikhonov', 'laplacian']:
                sst_weights = model.linear.weight[:, :model.sst_size]
                slp_weights = model.linear.weight[:, model.sst_size:]
                penalty_fn = spatial_penalty_tikhonov if args.penalty_type == 'tikhonov' else spatial_penalty_laplacian
                penalty_sst = penalty_fn(sst_weights, len(active_sst_lags), 85, 360)
                penalty_slp = penalty_fn(slp_weights, len(active_slp_lags), 53, 113) if model.slp_size > 0 else 0.0
                penalty = penalty_sst + penalty_slp
            else:
                penalty = 0.0
                
            loss_value = base_loss + args.alpha_penalty * penalty
            loss_value.backward()
            optimizer.step()
            
            running_train_loss += base_loss.item() * X_sst.size(0)
            total_train_samples += X_sst.size(0)

            # Accumulation pour corrélations/R2 dans l'espace latent et spatial
            med_pred = get_median_prediction_full_slp(predicted_latent, args.loss_type, args.quantiles) if args.loss_type == 'quantile' else predicted_latent 
            p, t = med_pred.detach(), target_embed.detach()
            train_latent_tracker.update(p, t)
            decoded_pred_map = decode_latent_to_map(predicted_latent, args, latent_dim, pca_model, vae_model, safe_wgts)
            train_map_tracker.update(y_target.detach(), decoded_pred_map.detach())

            if epoch == 0:
                epoch1_batch_losses.append(loss_value.item())
                with torch.no_grad():
                    zeros_pred = torch.zeros_like(predicted_latent)
                    baseline_loss = compute_loss(zeros_pred, target_embed, args.loss_type, args.quantiles, reduction='mean').item()
                    epoch1_baseline_losses.append(baseline_loss)

            # ---------------- INTRA-EPOCH VALIDATION ----------------
            if args.nb_intra_evals > 0 and (epoch == 0 or epoch == 1):  
                current_eval_steps_set = eval_steps_set if epoch == 0 else eval_steps_epoch2_set
                if batch_idx in current_eval_steps_set or batch_idx == len(trainloader) - 1:
                    print(f"\n--- Intra-epoch validation at step {batch_idx}/{len(trainloader)} ---")
                    eval_phases = ['val', 'test'] if nb_members_test > 0 else ['val']
                    
                    for key in eval_phases:
                        loader = valloader_intra if key == 'val' else testloader_intra
                        model.eval()
                        
                        intra_val_loss = 0.0
                        intra_n_samples = 0
                        intra_latent_tracker = LatentMetricTracker(device=device)
                        intra_map_tracker = MapMetricTracker(shape=(53, 113), device=device)      
                        
                        with torch.no_grad():
                            for v_X_sst, v_X_slp, v_y_target, _, _, _ in loader:
                                v_X_sst = v_X_sst.to(device, non_blocking=True)
                                v_X_slp = v_X_slp.to(device, non_blocking=True)
                                v_y_target = v_y_target.to(device, non_blocking=True)
                                
                                if args.embed_method == 'pca':
                                    slp_flat = v_y_target.view(v_y_target.size(0), -1).cpu().numpy()
                                    if args.lat_weight and wgts_flat is not None:
                                        slp_flat = slp_flat * wgts_flat
                                    embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                                    v_target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                                elif args.embed_method == 'vae':
                                    v_target_embed, _ = vae_model.encode(v_y_target)
                                    
                                v_pred = model(v_X_sst, v_X_slp)
                                loss_val = compute_loss(v_pred, v_target_embed, args.loss_type, args.quantiles, reduction='mean')
                                intra_val_loss += loss_val.item() * v_X_sst.size(0)
                                
                                p = get_median_prediction_full_slp(v_pred, args.loss_type, args.quantiles) if args.loss_type == 'quantile' else v_pred
                                t = v_target_embed
                                intra_latent_tracker.update(p.detach(), t.detach())
                                intra_n_samples += p.size(0)

                                decoded_v_pred = decode_latent_to_map(v_pred, args, latent_dim, pca_model, vae_model, safe_wgts)
                                intra_map_tracker.update(v_y_target.detach(), decoded_v_pred.detach())

                        lat_gR2, lat_mR2, lat_gCorr, lat_mCorr, lat_gK, lat_mK, lat_gL1, lat_mL1 = intra_latent_tracker.compute()
                        map_gR2, map_mR2, map_r2_np, map_sCorr, map_tCorr, map_gCorr, map_corr_np, map_gL1, map_mL1, map_l1_np = intra_map_tracker.compute(area_weights=area_weights_2d)
                        
                        prefix_l2 = f"L2_intra_{key}_ep{epoch+1}_step{batch_idx}"
                        prefix_l1 = f"L1_intra_{key}_ep{epoch+1}_step{batch_idx}"
                        save_r2_pixel_map_and_plot(map_r2_np, outdir, prefix_l2, metric_type="l2")
                        save_r2_pixel_map_and_plot(map_l1_np, outdir, prefix_l1, metric_type="l1")
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
                        print(f"-> Intra-{key} Loss: {current_intra_loss:.4f} | Latent gR2: {lat_gR2:.4f} | Map R2: {map_gR2:.4f} | Pixel Mean R2: {map_mR2:.4f}")
                        
                        if key == 'val' and current_intra_loss < best_val_loss:
                            best_val_loss = current_intra_loss
                            best_model_state = copy.deepcopy(model.state_dict())
                            if best_model_path and os.path.exists(best_model_path):
                                os.remove(best_model_path)
                            best_model_path = os.path.join(outdir, f'best_val_LinReg_ep{epoch + 1}_step{batch_idx}_loss{best_val_loss:.4f}.pth')
                            torch.save(model.state_dict(), best_model_path)
                            print(f"   *** Nouveau Best Model (Intra) sauvegardé : {os.path.basename(best_model_path)} ***")

                        model.train()

        # Finalisation des calculs d'époque Train
        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)
        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}')

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
        print(f"-> Train Map R2: {map_gR2:.4f} | Train Pixel Mean R2: {map_mR2:.4f}")

        if epoch == 0 or epoch == 1:
            if epoch == 0:
                loss_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, outdir, label="Train")
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
                    if batch_idx % 30 == 0:
                        print(f" {100 * batch_idx / len(loader):.1f}% {key} complete", end='\r')
                        
                    X_sst = X_sst.to(device, non_blocking=True)
                    X_slp = X_slp.to(device, non_blocking=True)
                    y_target = y_target.to(device, non_blocking=True)
                    
                    if args.embed_method == 'pca':
                        slp_flat = y_target.view(y_target.size(0), -1).cpu().numpy()
                        if args.lat_weight and wgts_flat is not None:
                            slp_flat = slp_flat * wgts_flat
                        embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                        target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                    elif args.embed_method == 'vae':
                        target_embed, _ = vae_model.encode(y_target)

                    predicted_latent = model(X_sst, X_slp)
                    loss_value = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')
                    running_val_loss += loss_value.item() * X_sst.size(0)
                    total_val_samples += X_sst.size(0)

                    per_sample_losses = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='none').cpu().numpy()
                    median_pred_latent = get_median_prediction_full_slp(predicted_latent, args.loss_type, args.quantiles) if args.loss_type == 'quantile' else predicted_latent

                    eval_latent_tracker.update(median_pred_latent.detach(), target_embed.detach())
                    decoded_eval_map = decode_latent_to_map(predicted_latent, args, latent_dim, pca_model, vae_model, safe_wgts)
                    eval_map_tracker.update(y_target.detach(), decoded_eval_map.detach())

                    if key == 'val':
                        if args.embed_method == 'pca':
                            pred_np = median_pred_latent.cpu().numpy()
                            target_np = target_embed.cpu().numpy()
                            padded_pred = np.zeros((pred_np.shape[0], pca_model.n_components_))
                            padded_target = np.zeros((target_np.shape[0], pca_model.n_components_))
                            padded_pred[:, :latent_dim] = pred_np; padded_target[:, :latent_dim] = target_np

                            predicted_slp_flat_polluted = pca_model.inverse_transform(padded_pred)
                            recon_true_slp_flat_polluted = pca_model.inverse_transform(padded_target)
                            
                            if args.lat_weight and safe_wgts is not None:
                                predicted_slp_flat = predicted_slp_flat_polluted / safe_wgts
                                recon_true_slp_flat = recon_true_slp_flat_polluted / safe_wgts
                            else:
                                predicted_slp_flat = predicted_slp_flat_polluted
                                recon_true_slp_flat = recon_true_slp_flat_polluted

                            predicted_slp = predicted_slp_flat.reshape(-1, 1, 53, 113) 
                            recon_true_slp = recon_true_slp_flat.reshape(-1, 1, 53, 113)
                        elif args.embed_method == 'vae':
                            predicted_slp = vae_model.decode(median_pred_latent).cpu().numpy()
                            recon_true_slp = vae_model.decode(target_embed).cpu().numpy()

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
            print(f'Epoch {epoch + 1} {key} Loss: {val_loss:.6f} | {key} Map global R2: {map_gR2:.4f} | {key} Pixel Mean R2: {map_mR2:.4f} | {key} Latent global R2: {lat_gR2:.4f} | {key} Latent Mean R2: {lat_mR2:.4f} | {key} Map global Corr: {map_gCorr:.4f} | {key} Map Mean Corr: {map_tCorr:.4f} | {key} Latent global Corr: {lat_gCorr:.4f} | {key} Latent Mean Corr: {lat_mCorr:.4f} | {key} Map global L1: {map_gL1:.4f} | {key} Map Mean L1: {map_mL1:.4f} | {key} Latent global L1: {lat_gL1:.4f} | {key} Latent Mean L1: {lat_mL1:.4f}')

            # ---------------- EARLY STOPPING & SAVING ----------------
            if key == 'val':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    if best_model_path and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    best_model_path = os.path.join(outdir, f'best_val_LinReg_ep{epoch + 1}_end_loss{best_val_loss:.4f}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"   *** Nouveau Best Model (Fin d'époque) sauvegardé : {os.path.basename(best_model_path)} ***")
                else:
                    patience_counter += 1

        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        print(f"--> Fin de l'époque {epoch + 1} - Elapsed Time: {current_time_min:.2f} minutes")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
            break
        
        # ---------------- AFFICHAGE ET SAUVEGARDE CHAK 2 EPOCHS ----------------
        if (epoch + 1) % 2 == 0:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses,
            }
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

            print(f"Saved checkpoint and spatial R2 plots at epoch {epoch + 1}")
            
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

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses
    }
    torch.save(state, f'{outdir}/final_model_LinReg.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_LinReg.pth')

    # ============================================================
    # EXPLICABILITÉ (1 diagnostic = 1 ligne de plot = sauvegardé séparément)
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
    weights = model.linear.weight.detach().cpu().numpy()

    with torch.no_grad():
        for X_sst, X_slp, y_target, _, _, _ in valloader:
            B = X_sst.size(0)
            n_samples_std += B
            
            X_sst_np = X_sst.cpu().numpy()
            sum_sst += X_sst_np.sum(axis=0); sum_sq_sst += (X_sst_np ** 2).sum(axis=0)
            X_sst_flat = X_sst_np.reshape(B, -1)
            
            if X_slp.numel() > 0:
                X_slp_np = X_slp.cpu().numpy()
                sum_slp += X_slp_np.sum(axis=0); sum_sq_slp += (X_slp_np ** 2).sum(axis=0)
                X_slp_flat = X_slp_np.reshape(B, -1)
                X_in = np.concatenate([X_sst_flat, X_slp_flat], axis=1)
            else:
                X_in = X_sst_flat
                X_slp_np = None

            Y_pred = X_in @ weights.T
            sum_y_pred += Y_pred.sum(axis=0); sum_sq_y_pred += (Y_pred ** 2).sum(axis=0)
            
            if args.embed_method == 'pca':
                slp_flat = y_target.view(B, -1).cpu().numpy()
                if args.lat_weight and wgts_flat is not None:
                    slp_flat = slp_flat * wgts_flat
                Y_true_latent = pca_model.transform(slp_flat)[:, :latent_dim]
            elif args.embed_method == 'vae':
                Y_true_latent_tensor, _ = vae_model.encode(y_target.to(device))
                Y_true_latent = Y_true_latent_tensor.cpu().numpy()
                
            sum_y_true += Y_true_latent.sum(axis=0); sum_sq_y_true += (Y_true_latent ** 2).sum(axis=0)
            
            sum_sst_y_pred += X_sst_flat.T @ Y_pred; sum_sst_y_true += X_sst_flat.T @ Y_true_latent
            sum_ypred_ytrue += (Y_pred * Y_true_latent).sum(axis=0)

            if X_slp_np is not None:
                sum_slp_y_pred += X_slp_flat.T @ Y_pred; sum_slp_y_true += X_slp_flat.T @ Y_true_latent

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
        slp_std = None
        corr_slp_pred = corr_slp_target = None
        cov_slp_ypred = cov_slp_ytrue = None

    print("\n--- GÉNÉRATION DES CARTES D'EXPLICABILITÉ ---")
    
    def plot_explainability_separated_rows(model, outdir, sst_lags, slp_lags, sst_std, slp_std, 
                                           corr_sst_p, corr_sst_t, corr_slp_p, corr_slp_t, 
                                           cov_sst_p, cov_sst_t, cov_slp_p, cov_slp_t, corr_model,
                                           sst_shape=(85, 360), slp_shape=(53, 113), max_components_to_plot=10):
        extent_sst = [-180, 180, -15, 70] if args.roll_sst else [0, 359.9, -15, 70]
        extent_slp = [-100, 40, 20, 70] 
        
        weights = model.linear.weight.detach().cpu().numpy()
        latent_dimension = weights.shape[0]
        in_chans_sst = len(sst_lags)
        in_chans_slp = len(slp_lags)
        sst_size_total = in_chans_sst * sst_shape[0] * sst_shape[1]
        
        if in_chans_sst > 0:
            sst_weights_raw = weights[:, :sst_size_total]
            sst_weights_eff = sst_weights_raw * sst_std.reshape(1, -1)
            
        if in_chans_slp > 0:
            slp_weights_raw = weights[:, sst_size_total:]
            slp_weights_eff = slp_weights_raw * slp_std.reshape(1, -1)

        comp_outdir = os.path.join(outdir, "components_explainability")
        os.makedirs(comp_outdir, exist_ok=True)
        num_plots = min(latent_dimension, max_components_to_plot)

        for comp_idx in range(num_plots):
            max_pixel_r = 0.0
            if in_chans_sst > 0:
                max_pixel_r = np.max(np.abs(corr_sst_t[:, comp_idx]))
            if in_chans_slp > 0:
                max_pixel_r = max(max_pixel_r, np.max(np.abs(corr_slp_t[:, comp_idx])))
            
            model_r = corr_model[comp_idx]
            print(f"Latent {comp_idx:03d} | R Modèle complet: {model_r:.4f} | R Meilleur Pixel: {max_pixel_r:.4f}")

            # ---------------- HISTOGRAMME ----------------
            plt.figure(figsize=(8, 5), facecolor='white')
            if in_chans_sst > 0:
                plt.hist(corr_sst_t[:, comp_idx].flatten(), bins=60, color='royalblue', edgecolor='black', alpha=0.7, label='SST Pixels vs Target')
            if in_chans_slp > 0:
                plt.hist(corr_slp_t[:, comp_idx].flatten(), bins=60, color='forestgreen', edgecolor='black', alpha=0.6, label='SLP Pixels vs Target')

            plt.axvline(x=model_r, color='crimson', linestyle='dashed', linewidth=2.5, label=f'Prédiction Modèle ($r={model_r:.3f}$)')
            plt.title(f'Distribution des corrélations spatiales - Latent {comp_idx}', fontsize=14)
            plt.xlabel(rf'Coefficient de corrélation de Pearson ($r$)', fontsize=12)
            plt.ylabel('Nombre de pixels', fontsize=12)
            plt.xlim(-1, 1)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(os.path.join(comp_outdir, f"histogram_corr_comp_{comp_idx:03d}.png"), dpi=150, bbox_inches='tight')
            plt.close()

            # ---------------- HELD-OUT FUNCTION POUR TRACER UNE LIGNE HORIZONTALE SÉPARÉE ----------------
            def save_single_row_diagnostic(data_3d, title_prefix, mod_name, comp_idx, lags, extent, vmin, vmax, cmap='RdBu_r'):
                n_lags = len(lags)
                fig, axes = plt.subplots(1, n_lags, figsize=(6 * n_lags, 4.5), subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
                if n_lags == 1: axes = [axes]
                
                for idx, lag in enumerate(lags):
                    ax = axes[idx]
                    im = ax.imshow(data_3d[idx], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent, interpolation='nearest')
                    ax.set_title(rf"{title_prefix} - Lag {lag}", fontsize=13)
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    
                fig.colorbar(im, ax=axes, shrink=0.8, orientation='vertical', pad=0.02)
                fig.suptitle(rf"{mod_name} Diagnostic : {title_prefix} - Latent {comp_idx}", fontsize=15, y=1.05)
                
                save_dir = os.path.join(comp_outdir, mod_name, title_prefix.replace(" ", "_").replace("(", "").replace(")", "").replace("\\", "").replace("$", ""))
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"comp_{comp_idx:03d}.png"), dpi=150, bbox_inches='tight')
                plt.close()

            # ---------------- DIAGNOSTICS SST ----------------
            if in_chans_sst > 0:
                sst_w_raw = sst_weights_raw[comp_idx].reshape(in_chans_sst, sst_shape[0], sst_shape[1])
                sst_w_eff = sst_weights_eff[comp_idx].reshape(in_chans_sst, sst_shape[0], sst_shape[1])
                sst_c_pred = corr_sst_p[:, comp_idx].reshape(in_chans_sst, sst_shape[0], sst_shape[1])
                sst_c_targ = corr_sst_t[:, comp_idx].reshape(in_chans_sst, sst_shape[0], sst_shape[1])
                sst_cov_pred = cov_sst_p[:, comp_idx].reshape(in_chans_sst, sst_shape[0], sst_shape[1])
                sst_cov_targ = cov_sst_t[:, comp_idx].reshape(in_chans_sst, sst_shape[0], sst_shape[1])
                
                vmax_raw = np.max(np.abs(sst_w_raw))
                vmax_eff = np.max(np.abs(sst_w_eff))
                vmax_cov = max(np.max(np.abs(sst_cov_pred)), np.max(np.abs(sst_cov_targ))) 

                save_single_row_diagnostic(sst_w_raw, "Raw Coefs", "SST", comp_idx, sst_lags, extent_sst, -vmax_raw, vmax_raw)
                save_single_row_diagnostic(sst_w_eff, "Effective Sensitivity", "SST", comp_idx, sst_lags, extent_sst, -vmax_eff, vmax_eff)
                save_single_row_diagnostic(sst_c_pred, "Correlation (Pixel vs Pred)", "SST", comp_idx, sst_lags, extent_sst, -1, 1)
                save_single_row_diagnostic(sst_c_targ, "Correlation (Pixel vs Target)", "SST", comp_idx, sst_lags, extent_sst, -1, 1)
                save_single_row_diagnostic(sst_cov_pred, "Covariance (Pixel vs Pred)", "SST", comp_idx, sst_lags, extent_sst, -vmax_cov, vmax_cov)
                save_single_row_diagnostic(sst_cov_targ, "Covariance (Pixel vs Target)", "SST", comp_idx, sst_lags, extent_sst, -vmax_cov, vmax_cov)

            # ---------------- DIAGNOSTICS SLP ----------------
            if in_chans_slp > 0:
                slp_w_raw = slp_weights_raw[comp_idx].reshape(in_chans_slp, slp_shape[0], slp_shape[1])
                slp_w_eff = slp_weights_eff[comp_idx].reshape(in_chans_slp, slp_shape[0], slp_shape[1])
                slp_c_pred = corr_slp_p[:, comp_idx].reshape(in_chans_slp, slp_shape[0], slp_shape[1])
                slp_c_targ = corr_slp_t[:, comp_idx].reshape(in_chans_slp, slp_shape[0], slp_shape[1])
                slp_cov_pred = cov_slp_p[:, comp_idx].reshape(in_chans_slp, slp_shape[0], slp_shape[1])
                slp_cov_targ = cov_slp_t[:, comp_idx].reshape(in_chans_slp, slp_shape[0], slp_shape[1])
                
                vmax_raw_slp = np.max(np.abs(slp_w_raw))
                vmax_eff_slp = np.max(np.abs(slp_w_eff))
                vmax_cov_slp = max(np.max(np.abs(slp_cov_pred)), np.max(np.abs(slp_cov_targ)))

                save_single_row_diagnostic(slp_w_raw, "Raw Coefs", "SLP", comp_idx, slp_lags, extent_slp, -vmax_raw_slp, vmax_raw_slp)
                save_single_row_diagnostic(slp_w_eff, "Effective Sensitivity", "SLP", comp_idx, slp_lags, extent_slp, -vmax_eff_slp, vmax_eff_slp)
                save_single_row_diagnostic(slp_c_pred, "Correlation (Pixel vs Pred)", "SLP", comp_idx, slp_lags, extent_slp, -1, 1)
                save_single_row_diagnostic(slp_c_targ, "Correlation (Pixel vs Target)", "SLP", comp_idx, slp_lags, extent_slp, -1, 1)
                save_single_row_diagnostic(slp_cov_pred, "Covariance (Pixel vs Pred)", "SLP", comp_idx, slp_lags, extent_slp, -vmax_cov_slp, vmax_cov_slp)
                save_single_row_diagnostic(slp_cov_targ, "Covariance (Pixel vs Target)", "SLP", comp_idx, slp_lags, extent_slp, -vmax_cov_slp, vmax_cov_slp)

    plot_explainability_separated_rows(model, outdir, active_sst_lags, active_slp_lags, sst_std, slp_std, 
                                       corr_sst_pred, corr_sst_target, corr_slp_pred, corr_slp_target, 
                                       cov_sst_ypred, cov_sst_ytrue, cov_slp_ypred, cov_slp_ytrue, corr_model,
                                       max_components_to_plot=10)
    print(f"Training complete, elapsed time: {(time.time() - start_time) / 60:.2f} minutes")

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
                "--latent_dim", str(latent_dim), "--duree_lissage", str(args.duree_lissage),
                "--bs", str(args.bs)
            ]
            if args.embed_path: eval_command.extend(["--embed_path", str(args.embed_path)])
            if args.sst_lags_days: eval_command.extend(["--sst_lags_days"] + [str(x) for x in args.sst_lags_days])
            if args.sst_lags_months: eval_command.extend(["--sst_lags_months"] + [str(x) for x in args.sst_lags_months])
            if args.slp_lags_days: eval_command.extend(["--slp_lags_days"] + [str(x) for x in args.slp_lags_days])
            if args.slp_lags_months: eval_command.extend(["--slp_lags_months"] + [str(x) for x in args.slp_lags_months])
            if args.winter_months: eval_command.extend(["--winter_months"] + [str(x) for x in args.winter_months])
            if hasattr(args, 'loss_type'): eval_command.extend(["--loss_type", str(args.loss_type)])
            if hasattr(args, 'quantiles') and args.quantiles: eval_command.extend(["--quantiles"] + [str(x) for x in args.quantiles])
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
            else:
                print(f"⚠️ Script ignoré (non trouvé) : {eval_script_path}")

            # --- EVAL SPATIALE ---
            print(f"\n--- Évaluation spatiale du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            eval_spatial_command = [
                sys.executable, eval_spatial_script_path,
                "--machine", str(args.machine), "--embed_method", str(args.embed_method),
                "--linreg_dir", str(outdir),
                "--model_type", str(model_type), "--nb_members_train", str(args.nb_members_train), "--nb_members_val", str(args.nb_members_val), "--nb_members_test", str(args.nb_members_test),
                "--seed", str(args.seed), "--latent_dim", str(latent_dim),
                "--duree_lissage", str(args.duree_lissage), "--bs", str(args.bs)
            ]
            if args.embed_path:
                eval_spatial_command.extend(["--embed_path", str(args.embed_path)])
            else:
                ext = "joblib" if args.embed_method == 'pca' else "pth"
                eval_spatial_command.extend(["--embed_path", os.path.join(outdir, f"{args.embed_method}_model.{ext}")])

            if args.sst_lags_days: eval_spatial_command.extend(["--sst_lags_days"] + [str(x) for x in args.sst_lags_days])
            if args.slp_lags_days: eval_spatial_command.extend(["--slp_lags_days"] + [str(x) for x in args.slp_lags_days])
            if args.sst_lags_months: eval_spatial_command.extend(["--sst_lags_months"] + [str(x) for x in args.sst_lags_months])
            if args.slp_lags_months: eval_spatial_command.extend(["--slp_lags_months"] + [str(x) for x in args.slp_lags_months])
            if args.winter_months: eval_spatial_command.extend(["--winter_months"] + [str(x) for x in args.winter_months])
            if hasattr(args, 'loss_type'): eval_spatial_command.extend(["--loss_type", str(args.loss_type)])
            if hasattr(args, 'quantiles') and args.quantiles: eval_spatial_command.extend(["--quantiles"] + [str(x) for x in args.quantiles])
            if args.roll_sst: eval_spatial_command.append("--roll_sst")
            if args.monthly_reduction: eval_spatial_command.append("--monthly_reduction")
            if args.lat_weight: eval_spatial_command.append("--lat_weight")
            if monthly_mean: eval_spatial_command.append("--monthly_mean")

            if os.path.exists(eval_spatial_script_path):
                try:
                    subprocess.run(eval_spatial_command, check=True, text=True)
                    print(f"✅ Évaluation spatiale de {model_type} terminée avec succès !")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Erreur lors de l'exécution de l'évaluation spatiale. Code: {e.returncode}")
            else:
                print(f"⚠️ Script ignoré (non trouvé) : {eval_spatial_script_path}")