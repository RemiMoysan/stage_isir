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

# Ajouter le dossier "tools" de vision_transformer au sys.path pour les imports
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

from tools.visualizations import loss_figure, loss_first_epoch, plot_and_save_maps_with_reconstruction, plot_and_save_maps_with_reconstruction_light, plot_reconstruction_check, plot_correlation_evolution,plot_r2_R2_evolution
from tools.datasets import Dataset, Dataset_mensuel
from tools.models import ConvVAE, vae_loss, compute_loss, get_median_prediction, spatial_penalty_tikhonov, spatial_penalty_laplacian

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
    parser.add_argument('--alpha_penalty', type=float, default=1e-5, help='Poids de la pénalité L1 ou L2')
    parser.add_argument('--penalty_type', type=str, choices=['l1', 'l2', 'tikhonov', 'laplacian'], default='l2', help='Type de pénalité à utiliser')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--beta_kld', type=float, default=1.0)
    parser.add_argument('--normalize', action='store_true')

    parser.add_argument('--exact_solver', action='store_true', help="Utiliser la formule mathématique exacte (Ridge/OLS)")
    parser.add_argument('--max_samples_exact', type=int, default=2000, help="Limite d'échantillons N pour éviter le Out-Of-Memory")
    
    # --- NOUVEAUX ARGUMENTS ---
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--roll_sst', action='store_true', help='Appliquer un roll sur les données SST')
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Nombre de points de validation intra-époque')
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données sous-échantillonnées mensuellement')
    parser.add_argument('--lat_weight', action='store_true', help='Applique la pondération spatiale sqrt(cos(lat))')

    # --- NOUVEAUX ARGUMENTS DE LOSS ---
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile', 'correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    args = parser.parse_args()
    if args.loss_type == 'quantile':
        if 0.5 not in args.quantiles:
            raise ValueError("Erreur: Pour la quantile loss, la liste des quantiles (--quantiles) DOIT inclure la médiane (0.5) pour permettre les reconstructions.")
    # --- NOUVELLE VÉRIFICATION ICI ---
    if args.exact_solver and args.loss_type != 'mse' and args.penalty_type not in ['l2','none']:
        raise ValueError("Erreur: Le solveur analytique exact n'est mathématiquement valide que pour la loss 'mse'. Désactive --exact_solver ou change la loss.")

    # Routage dynamique des dossiers
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/linear_models/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']:
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/"
    elif args.machine == 'mac_local':
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/linear_models/"

    # Variables globales & args
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

    print("Arg Parameters:")
    print(f"  Latent Dim: {latent_dim}", f"SST Lags: {active_sst_lags}", f"SLP Lags: {active_slp_lags}", f"Batch Size: {bs}", f"Learning Rate: {lr}", f"Winter Months: {winter_months}", f"Smoothing Duration: {duree_lissage}", f"Number of Epochs: {nb_epochs}", f"Train Members: {nb_members_train}", f"Val Members: {nb_members_val}\n")

    patience = 10000
    target_indices = {100, 1000, 2000, 3000, 4000, 4500, 5000, 6000, 7000, 8000} if not args.monthly_reduction else {10, 100, 200, 300, 400, 450, 500, 600, 700, 800} 

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[-nb_members_val:]
    test_members = all_members[args.nb_members_train:args.nb_members_train + args.nb_members_test] if args.nb_members_test > 0 else []

    # Nom du dossier adapté
    base_outdir_name = f"LinReg_loss_{args.loss_type}_{args.penalty_type}_{args.alpha_penalty}_{args.embed_method}_emb_{latent_dim}_lat_weight_{args.lat_weight}_normalize_{args.normalize}_bs{bs}_lr{lr}_months_{'_'.join(map(str, winter_months))}_seed_{args.seed}_train{nb_members_train}_val{nb_members_val}_{nb_members_test}"
    if args.exact_solver:
        base_outdir_name += f"_{args.exact_solver}_exact_{args.max_samples_exact}_samples"
        
    if not args.monthly_reduction:
        outdir_name = f"{base_outdir_name}_sst_{'_'.join(map(str, sst_lags_days))}_slp_{'_'.join(map(str, slp_lags_days))}_{duree_lissage}d_roll_sst_{args.roll_sst}"
    else:
        outdir_name = f"{base_outdir_name}_sst_{'_'.join(map(str, sst_lags_months))}_slp_{'_'.join(map(str, slp_lags_months))}_monthly_roll_sst_{args.roll_sst}"

    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    intra_workers = min(2, n_workers)
    print(f"Using {n_workers} workers for data loading")

    # Fallback et extraction du std pour l'embedding SLP
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

    if args.nb_members_test > 0:
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
            pca_model = joblib.load(args.embed_path)
        else:
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
            torch.save(vae_model.state_dict(), os.path.join(outdir, "vae_model.pth"))
            print(f"VAE training complete")
        vae_model.eval()
        for param in vae_model.parameters():
            param.requires_grad = False

    # ============================================================
    # 4.5 SANITY CHECK DE L'EMBEDDER (PCA ou VAE)
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
    print(f"-> Erreur RMSE moyenne de reconstruction : {rmse:.4f} hPa (environ)")

    end_embed_time = time.time()
    print(f"Embedding completed in {(end_embed_time - start_embed_time) / 60:.2f} minutes")

    plot_reconstruction_check(true_slp_val, recon_slp_val, dates_val, outdir, args.embed_method, num_samples=10)
    print("-> Plot de vérification sauvegardé (Vérifie l'image avant de lancer !)\n")

    # ============================================================
    # 5. INITIALISATION DU MODÈLE DE RÉGRESSION
    # ============================================================
    out_features = latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else latent_dim

    model = LinearRegressionPredictor(
        sst_shape=(85, 360), 
        slp_shape=(53, 113), 
        in_chans_sst=len(active_sst_lags), 
        in_chans_slp=len(active_slp_lags), 
        out_dim=out_features # <--- MODIFICATION ICI
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()

    print(f"Number of Linear parameters : {sum(p.numel() for p in model.parameters())}")

    # ============================================================
    # SOLUTION ANALYTIQUE EXACTE (SANITY CHECK, pas pour loss correlation ni quantile)
    # ============================================================
    if args.exact_solver:
            
        print(f"\n--- CALCUL DE LA SOLUTION EXACTE (Limité à {args.max_samples_exact} échantillons) ---")
        model.eval()
        X_list, Y_list = [], []
        current_samples = 0
        
        for X_sst, X_slp, y_target, _, _, _ in trainloader:
            batch_size = X_sst.size(0)
            X_sst_flat = X_sst.view(batch_size, -1).cpu()
            
            if X_slp.numel() > 0:
                X_slp_flat = X_slp.view(batch_size, -1).cpu()
                X_batch = torch.cat([X_sst_flat, X_slp_flat], dim=1)
            else:
                X_batch = X_sst_flat
                
            if args.embed_method == 'pca':
                slp_flat = y_target.view(batch_size, -1).numpy()
                if args.lat_weight and wgts_flat is not None:
                    slp_flat = slp_flat * wgts_flat
                embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                Y_batch = torch.tensor(embed_np, dtype=torch.float32)
            elif args.embed_method == 'vae':
                with torch.no_grad():
                    Y_batch, _ = vae_model.encode(y_target.to(device))
                    Y_batch = Y_batch.cpu()
                    
            X_list.append(X_batch)
            Y_list.append(Y_batch)
            current_samples += batch_size
            
            if current_samples >= args.max_samples_exact:
                print(f"Limite de {args.max_samples_exact} échantillons atteinte.")
                break
                
        X = torch.cat(X_list, dim=0)[:args.max_samples_exact] 
        Y = torch.cat(Y_list, dim=0)[:args.max_samples_exact] 
        
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        X_c = X - X_mean
        Y_c = Y - Y_mean
        
        N, D = X.shape
        lambda_ridge = args.alpha_penalty * N 
        print(f"Dimensions - Échantillons (N): {N}, Features spatiales (D): {D}")
        print("Résolution matricielle en cours (Formulation Duale)...")
        start_math = time.time()
        
        K_mat = X_c @ X_c.T + lambda_ridge * torch.eye(N) 
        dual_coef = torch.linalg.solve(K_mat, Y_c)
        W_exact = X_c.T @ dual_coef
        bias_exact = Y_mean.squeeze() - (X_mean @ W_exact).squeeze()
        
        print(f"Calcul terminé en {time.time() - start_math:.2f} secondes.")
        
        with torch.no_grad():
            model.linear.weight.copy_(W_exact.T.to(device))
            model.linear.bias.copy_(bias_exact.to(device))
            
        print("-> Solution mathématique injectée dans le modèle avec succès.")
        best_model_state = copy.deepcopy(model.state_dict())
        nb_epochs = 1

    # ============================================================
    # PRÉPARATION DU SUIVI (INTRA-EVAL & HISTORIQUE)
    # ============================================================
    val_losses_per_member_history = defaultdict(list)
    train_losses, val_losses = [], []
    train_corrs, val_corrs = [], []
    train_R2, val_R2 = [], []
    train_ks, val_ks, test_ks = [], [], []
    best_val_loss = float('inf') 
    best_model_path = ""

    epoch1_batch_losses, epoch1_baseline_losses = [], []
    intra_epoch1_steps, intra_epoch1_val_losses, intra_epoch1_val_corrs, intra_epoch1_val_R2 = [], [], [], []
    intra_epoch2_steps, intra_epoch2_val_losses, intra_epoch2_val_corrs, intra_epoch2_val_R2 = [], [], [], []

    test_losses, test_corrs, test_R2 = [], [], []
    intra_epoch1_test_losses, intra_epoch1_test_corrs, intra_epoch1_test_R2 = [], [], []
    intra_epoch2_test_losses, intra_epoch2_test_corrs, intra_epoch2_test_R2 = [], [], []

    total_batches = len(trainloader)
    eval_steps = np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int) if args.nb_intra_evals > 0 else []
    eval_steps = np.insert(eval_steps, 0, 0) if len(eval_steps) > 0 else np.array([])
    eval_steps_set = set(eval_steps)

    # Variables pour le suivi intra-époque 1
    intra_epoch1_val_ks, intra_epoch1_test_ks = [], []
    # Variables pour le suivi intra-époque 2
    intra_epoch2_val_ks, intra_epoch2_test_ks = [], []

    eval_steps_epoch2 = np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int) if args.nb_intra_evals > 0 else []
    eval_steps_epoch2 = np.insert(eval_steps_epoch2, 0, 0) if len(eval_steps_epoch2) > 0 else np.array([])
    eval_steps_epoch2_set = set(eval_steps_epoch2)

    if args.nb_intra_evals > 0:
        print(f"Validation intra-époque 1 aux steps : {sorted(list(eval_steps_set))}")

    # ============================================================
    # 6. TRAINING & EVALUATION LOOP
    # ============================================================
    start_time = time.time()
    epoch_times = []
    patience_counter = 0

    for epoch in range(nb_epochs):
        # ---------------- TRAINING ----------------
        if not args.exact_solver:
            model.train()
            running_train_loss = 0.0
            total_train_samples = 0
            
            # Variables pour corrélation
            sum_p, sum_t, sum_p2, sum_t2, sum_pt, sum_res = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
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
                    # Séparation des poids SST et SLP
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

                # Accumulation pour corrélation (sur l'époque)
                med_pred = get_median_prediction(predicted_latent, args.loss_type, args.quantiles, latent_dim)
                p = med_pred.detach()
                t = target_embed.detach()
                sum_p += p.sum(dim=0)
                sum_t += t.sum(dim=0)
                sum_p2 += (p ** 2).sum(dim=0)
                sum_t2 += (t ** 2).sum(dim=0)
                sum_pt += (p * t).sum(dim=0)
                sum_res += ((p - t) ** 2).sum(dim=0)

                # Suivi epoch 1
                if epoch == 0:
                    epoch1_batch_losses.append(loss_value.item())
                    with torch.no_grad():
                        zeros_pred = torch.zeros_like(predicted_latent) # <--- CORRECTION ICI
                        baseline_loss = compute_loss(zeros_pred, target_embed, args.loss_type, args.quantiles, reduction='mean').item()
                        epoch1_baseline_losses.append(baseline_loss)

                # Validation intra-époque
                if args.nb_intra_evals > 0 and (epoch == 0 or epoch == 1):
                    current_eval_steps_set = eval_steps_set if epoch == 0 else eval_steps_epoch2_set
                    if batch_idx in current_eval_steps_set or batch_idx == len(trainloader) - 1:
                        print(f"\n--- Intra-epoch validation at step {batch_idx}/{len(trainloader)} ---")
                        eval_phases = ['val', 'test'] if nb_members_test > 0 else ['val']
                        for key in eval_phases:
                            if key == 'val':
                                loader = valloader_intra
                            else:
                                loader = testloader_intra
                            model.eval()
                            intra_val_loss = 0.0
                            intra_n_samples = 0
                            v_sum_p, v_sum_t, v_sum_p2, v_sum_t2, v_sum_pt, v_sum_res = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                            
                            with torch.no_grad():
                                for v_X_sst, v_X_slp, v_y_target, _, _, _ in loader:
                                    v_X_sst = v_X_sst.to(device, non_blocking=True)
                                    v_X_slp = v_X_slp.to(device, non_blocking=True)
                                    v_y_target = v_y_target.to(device, non_blocking=True)
                                    
                                    if args.embed_method == 'pca':
                                        v_slp_flat = v_y_target.view(v_y_target.size(0), -1).cpu().numpy()
                                        if args.lat_weight and wgts_flat is not None:
                                            v_slp_flat = v_slp_flat * wgts_flat
                                        v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                                        v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                                    elif args.embed_method == 'vae':
                                        v_target_embed, _ = vae_model.encode(v_y_target.to(device))
                                        
                                    v_pred = model(v_X_sst, v_X_slp)
                                    intra_val_loss += compute_loss(v_pred, v_target_embed, args.loss_type, args.quantiles, reduction='mean').item() * v_X_sst.size(0)

                                    vp = get_median_prediction(v_pred, args.loss_type, args.quantiles, latent_dim) if args.loss_type == 'quantile' else v_pred
                                    vt = v_target_embed
                                    v_sum_p += vp.sum(dim=0)
                                    v_sum_t += vt.sum(dim=0)
                                    v_sum_p2 += (vp ** 2).sum(dim=0)
                                    v_sum_t2 += (vt ** 2).sum(dim=0)
                                    v_sum_pt += (vp * vt).sum(dim=0)
                                    v_sum_res += ((vp - vt) ** 2).sum(dim=0)
                                    intra_n_samples += vp.size(0)

                            v_mean_p = v_sum_p / intra_n_samples
                            v_mean_t = v_sum_t / intra_n_samples
                            v_var_p = (v_sum_p2 / intra_n_samples) - v_mean_p**2
                            v_var_t = (v_sum_t2 / intra_n_samples) - v_mean_t**2
                            v_cov_pt = (v_sum_pt / intra_n_samples) - (v_mean_p * v_mean_t)
                            v_corr = (v_cov_pt / torch.sqrt(v_var_p * v_var_t + 1e-8)).mean().item()

                            # AJOUT : Calcul de v_k = std(pred) / std(target)
                            v_k_vector = torch.sqrt(v_var_p / (v_var_t + 1e-8))
                            v_k = v_k_vector.mean().item()
                            
                            v_ss_tot = v_var_t * intra_n_samples  # Variance totale * N
                            v_r2_vector = 1 - (v_sum_res / (v_ss_tot + 1e-8))
                            v_epoch_train_r2 = v_r2_vector.mean().item()
                            
                            
                            if epoch == 0:
                                if key == 'val': intra_epoch1_steps.append(batch_idx)  
                                intra_epoch1_val_losses.append(intra_val_loss / intra_n_samples) if key == 'val' else intra_epoch1_test_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch1_val_corrs.append(v_corr) if key == 'val' else intra_epoch1_test_corrs.append(v_corr)
                                intra_epoch1_val_R2.append(v_epoch_train_r2) if key == 'val' else intra_epoch1_test_R2.append(v_epoch_train_r2)
                                intra_epoch1_val_ks.append(v_k) if key == 'val' else intra_epoch1_test_ks.append(v_k)
                            elif epoch == 1:
                                if key == 'val': intra_epoch2_steps.append(batch_idx)
                                intra_epoch2_val_losses.append(intra_val_loss / intra_n_samples) if key == 'val' else intra_epoch2_test_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch2_val_corrs.append(v_corr) if key == 'val' else intra_epoch2_test_corrs.append(v_corr)
                                intra_epoch2_val_R2.append(v_epoch_train_r2) if key == 'val' else intra_epoch2_test_R2.append(v_epoch_train_r2)
                                intra_epoch2_val_ks.append(v_k) if key == 'val' else intra_epoch2_test_ks.append(v_k)
                            
                            current_intra_loss = intra_val_loss / intra_n_samples
                            print(f"-> Intra-{key} Loss: {current_intra_loss:.4f} | Intra-{key} Corr: {v_corr:.4f} | Intra-{key} R2: {v_epoch_train_r2:.4f}")
                            
                            # NOUVEAU : Sauvegarde du best model en intra-époque
                            if key == 'val' and current_intra_loss < best_val_loss:
                                best_val_loss = current_intra_loss
                                best_model_state = copy.deepcopy(model.state_dict())
                                
                                # Suppression de l'ancien meilleur modèle s'il existe
                                if best_model_path and os.path.exists(best_model_path):
                                    os.remove(best_model_path)
                                    
                                # Formatage du nouveau nom dynamique
                                best_model_path = os.path.join(outdir, f'best_val_Linreg_bs{bs}_ep{epoch + 1}_step{batch_idx}_loss{best_val_loss:.4f}.pth')
                                torch.save(model.state_dict(), best_model_path)
                                print(f"   *** Nouveau Best Model (Intra) sauvegardé : {os.path.basename(best_model_path)} ***")

                            # 2. IMPORTANT : Ne pas oublier de repasser le modèle en mode train !
                            model.train()

            train_loss = running_train_loss / total_train_samples
            train_losses.append(train_loss)
            
            # Corrélation finale train
            mean_p = sum_p / total_train_samples
            mean_t = sum_t / total_train_samples
            var_p = (sum_p2 / total_train_samples) - mean_p**2
            var_t = (sum_t2 / total_train_samples) - mean_t**2
            cov_pt = (sum_pt / total_train_samples) - (mean_p * mean_t)
            train_corr_vector = cov_pt / torch.sqrt(var_p * var_t + 1e-8)
            train_corrs.append(train_corr_vector.mean().item())
            # NOUVEAU : Calcul du R2
            ss_tot = var_t * total_train_samples # Variance totale * N
            r2_vector = 1 - (sum_res / (ss_tot + 1e-8))
            epoch_train_r2 = r2_vector.mean().item()
            train_R2.append(epoch_train_r2)

            # AJOUT : Calcul du k d'entraînement
            train_k_vector = torch.sqrt(var_p / (var_t + 1e-8))
            epoch_train_k = train_k_vector.mean().item()
            train_ks.append(epoch_train_k)

            print(f'Epoch {epoch + 1} Training MSE Loss: {train_loss:.8f}')

            # ----- NOUVEAU : Appel de la fonction de visualisation -----
            if epoch == 0 or epoch == 1:
                if epoch == 0:
                    loss_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, outdir,label="Train")
                if args.nb_intra_evals > 0:

                    # Routage dynamique des variables selon l'époque
                    current_intra_losses = intra_epoch1_val_losses if epoch == 0 else intra_epoch2_val_losses
                    current_intra_steps = intra_epoch1_steps if epoch == 0 else intra_epoch2_steps
                    current_intra_losses_test = intra_epoch1_test_losses if epoch == 0 else intra_epoch2_test_losses

                    current_intra_corrs = intra_epoch1_val_corrs if epoch == 0 else intra_epoch2_val_corrs
                    current_intra_R2 = intra_epoch1_val_R2 if epoch == 0 else intra_epoch2_val_R2
                    current_intra_test_corrs = intra_epoch1_test_corrs if epoch == 0 else intra_epoch2_test_corrs
                    current_intra_test_R2 = intra_epoch1_test_R2 if epoch == 0 else intra_epoch2_test_R2

                    current_intra_ks = intra_epoch1_val_ks if epoch == 0 else intra_epoch2_val_ks
                    current_intra_test_ks = intra_epoch1_test_ks if epoch == 0 else intra_epoch2_test_ks

                    
                    loss_first_epoch(current_intra_losses, [np.mean(epoch1_baseline_losses)]*len(current_intra_losses), outdir, label="Intra-Val", batch_indexes=current_intra_steps, epoch_num=epoch+1,batch_test_losses = current_intra_losses_test)

                    # éventuellement peaufiner en calculant vraiment la baseline mais ok
                    if epoch == 0:
                        plot_correlation_evolution([], current_intra_corrs, outdir, val_ks=current_intra_ks, test_corrs=current_intra_test_corrs, test_ks=current_intra_test_ks, epoch_1=True, batch_indexes=intra_epoch1_steps)
                        plot_r2_R2_evolution([],current_intra_corrs, [], current_intra_R2, outdir, epoch_1 = True,batch_indexes = intra_epoch1_steps,test_R2 = current_intra_test_R2, test_corrs = current_intra_test_corrs)
                    if epoch == 1:
                        plot_correlation_evolution([],intra_epoch2_val_corrs, outdir, epoch_2 = True,batch_indexes = intra_epoch2_steps,test_corrs = current_intra_test_corrs, test_ks = current_intra_test_ks, val_ks = current_intra_ks)
                        plot_r2_R2_evolution([],intra_epoch2_val_corrs, [], intra_epoch2_val_R2, outdir, epoch_2 = True,batch_indexes = intra_epoch2_steps, test_R2 = current_intra_test_R2, test_corrs = current_intra_test_corrs)

        else:
            print("\n--- EXACT SOLVER: Bypass de l'entraînement itératif ---")
            train_losses.append(0.0)
            train_corrs.append(0.0)
            train_R2.append(0.0)
            train_ks.append(0.0)

        # ---------------- VALIDATION ----------------


        per_member_loss = defaultdict(lambda: {'loss_sum': 0.0, 'count': 0})
        per_member_plots = defaultdict(lambda: {'time': [], 'slp_true': [], 'slp_recon_true': [], 'slp_pred': []})

        eval_phases = ['val', 'test'] if nb_members_test > 0 else ['val']
        for key in eval_phases:
            if key == 'val':
                loader = valloader
            else:
                loader = testloader
            model.eval()
            
            running_val_loss = 0.0
            total_val_samples = 0 

            # pour le calcul de la corrélation : 
            sum_p, sum_t = 0.0, 0.0
            sum_p2, sum_t2 = 0.0, 0.0
            sum_pt = 0.0
            sum_res = 0.0
            with torch.no_grad():
                for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(loader):
                    X_sst = X_sst.to(device, non_blocking=True)
                    X_slp = X_slp.to(device, non_blocking=True)
                    y_target = y_target.to(device, non_blocking=True)
                    
                    # Encodage Cible
                    if args.embed_method == 'pca':
                        slp_flat = y_target.view(y_target.size(0), -1).cpu().numpy()
                        if args.lat_weight and wgts_flat is not None:
                            slp_flat = slp_flat * wgts_flat
                        embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                        target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                    elif args.embed_method == 'vae':
                        target_embed, _ = vae_model.encode(y_target)

                    # Prédiction
                    predicted_latent = model(X_sst, X_slp)
                    # mse_loss = criterion(predicted_latent, target_embed)
                    base_loss = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')
                    
                    if args.penalty_type == 'l1':
                        penalty = torch.norm(model.linear.weight, p=1)
                    elif args.penalty_type == 'l2':
                        penalty = torch.sum(model.linear.weight ** 2)
                    elif args.penalty_type in ['tikhonov', 'laplacian']:
                        # Séparation des poids SST et SLP
                        sst_weights = model.linear.weight[:, :model.sst_size]
                        slp_weights = model.linear.weight[:, model.sst_size:]
                        
                        penalty_fn = spatial_penalty_tikhonov if args.penalty_type == 'tikhonov' else spatial_penalty_laplacian
                        
                        penalty_sst = penalty_fn(sst_weights, len(active_sst_lags), 85, 360)
                        penalty_slp = penalty_fn(slp_weights, len(active_slp_lags), 53, 113) if model.slp_size > 0 else 0.0
                        
                        penalty = penalty_sst + penalty_slp
                    else:
                        penalty = 0.0
                        
                    loss_value = base_loss + args.alpha_penalty * penalty
                    running_val_loss += base_loss.item() * X_sst.size(0) 
                    total_val_samples += X_sst.size(0)

                    # NOUVEAU : Tableau des pertes par sample
                    per_sample_losses = (compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='none') + args.alpha_penalty * penalty).cpu().numpy()

                    # NOUVEAU : Médiane pour les reconstructions et la corrélation
                    median_pred_latent = get_median_prediction(predicted_latent, args.loss_type, args.quantiles, latent_dim) if args.loss_type == 'quantile' else predicted_latent

                    # Accumulation Corrélation Validation
                    p, t = median_pred_latent, target_embed
                    sum_p += p.sum(dim=0)
                    sum_t += t.sum(dim=0)
                    sum_p2 += (p ** 2).sum(dim=0)
                    sum_t2 += (t ** 2).sum(dim=0)
                    sum_pt += (p * t).sum(dim=0)
                    sum_res += ((p - t) ** 2).sum(dim=0) # NOUVEAU

                    if key == 'val':

                        # Décodage Double (Prédiction & Cible idéale)
                        if args.embed_method == 'pca':
                            pred_np = median_pred_latent.cpu().numpy()
                            target_np = target_embed.cpu().numpy()
                            
                            padded_pred = np.zeros((pred_np.shape[0], pca_model.n_components_))
                            padded_target = np.zeros((target_np.shape[0], pca_model.n_components_))
                            padded_pred[:, :latent_dim] = pred_np
                            padded_target[:, :latent_dim] = target_np

                            predicted_slp_flat_polluted = pca_model.inverse_transform(padded_pred)
                            recon_true_slp_flat_polluted = pca_model.inverse_transform(padded_target)
                            
                            if args.lat_weight and 'safe_wgts' in locals():
                                predicted_slp_flat = predicted_slp_flat_polluted / safe_wgts
                                recon_true_slp_flat = recon_true_slp_flat_polluted / safe_wgts
                            else:
                                predicted_slp_flat = predicted_slp_flat_polluted
                                recon_true_slp_flat = recon_true_slp_flat_polluted

                            predicted_slp = predicted_slp_flat.reshape(-1, 1, 53, 113) 
                            recon_true_slp = recon_true_slp_flat.reshape(-1, 1, 53, 113)
                            
                        elif args.embed_method == 'vae':
                            predicted_slp = vae_model.decode(predicted_latent).cpu().numpy()
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
                # Calcul de la perte de validation par membre
                for mem, d in per_member_loss.items():
                    avg_loss = d['loss_sum'] / d['count'] if d['count'] > 0 else float('nan')
                    val_losses_per_member_history[mem].append(avg_loss)
            val_loss = running_val_loss / total_val_samples 
            val_losses.append(val_loss) if key == 'val' else test_losses.append(val_loss)
            
            mean_p = sum_p / total_val_samples
            mean_t = sum_t / total_val_samples
            var_p = (sum_p2 / total_val_samples) - mean_p**2
            var_t = (sum_t2 / total_val_samples) - mean_t**2
            cov_pt = (sum_pt / total_val_samples) - (mean_p * mean_t)
            val_corr_vector = cov_pt / torch.sqrt(var_p * var_t + 1e-8)
            val_corrs.append(val_corr_vector.mean().item()) if key == 'val' else test_corrs.append(val_corr_vector.mean().item())

            # NOUVEAU : Calcul du R2
            ss_tot = var_t * total_val_samples # Variance totale * N
            r2_vector = 1 - (sum_res / (ss_tot + 1e-8))
            epoch_val_r2 = r2_vector.mean().item()
            val_R2.append(epoch_val_r2) if key == 'val' else test_R2.append(epoch_val_r2)

            # AJOUT : Calcul du k de validation / test
            val_k_vector = torch.sqrt(var_p / (var_t + 1e-8))
            epoch_val_k = val_k_vector.mean().item()
            val_ks.append(epoch_val_k) if key == 'val' else test_ks.append(epoch_val_k)

            # ---------------- EARLY STOPPING & SAVING ----------------
            if key == 'val':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    if best_model_path and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    best_model_path = os.path.join(outdir, f'best_model_LinReg_ep{epoch + 1}_end_loss{best_val_loss:.4f}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"   *** Nouveau Best Model sauvegardé : {os.path.basename(best_model_path)} ***")
                else:
                    patience_counter += 1

        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        print(f'Epoch {epoch + 1} - Elapsed Time: {current_time_min:.2f} minutes')

        # 2. Vrai déclenchement de l'Early Stopping (Quitte la boucle des époques)
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
            break
        
        # ---------------- AFFICHAGE ET SAUVEGARDE ----------------
        if (epoch + 1) % 1 == 0:
            state = {'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    'train_losses': train_losses, 'val_losses': val_losses, 'train_corrs': train_corrs, 'train_R2': train_R2, 'val_corrs': val_corrs, 'val_R2': val_R2, 'test_losses': test_losses, 'test_corrs': test_corrs, 'test_R2': test_R2,'train_ks': train_ks, 'val_ks': val_ks, 'test_ks': test_ks}
            torch.save(state, f'{outdir}/final_model_LinReg_bs{bs}.pth')
            
            loss_figure(len(train_losses), train_losses, val_losses, outdir,epoch_times,per_member_val_losses=val_losses_per_member_history,test_losses=test_losses)
            plot_correlation_evolution(train_corrs, val_corrs,outdir,test_corrs = test_corrs, val_ks=val_ks, test_ks=test_ks, train_ks=train_ks)
            plot_r2_R2_evolution(train_corrs, val_corrs, train_R2, val_R2, outdir,test_R2 = test_R2)
            print(f"Saved checkpoint at epoch {epoch + 1}")
            
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

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_model_LinReg.pth')

    # ============================================================
    # 7. EXPLICABILITÉ (Double Corrélation : Pred vs Target)
    # ============================================================
    print("\n--- CALCUL DE LA VARIABILITÉ SPATIALE ET DES CORRÉLATIONS (PRED & TARGET) ---")
    
    # Accumulateurs pour X
    sum_sst, sum_sq_sst = 0.0, 0.0
    sum_slp, sum_sq_slp = 0.0, 0.0
    
    # Accumulateurs pour Y_pred (Sorties du modèle)
    sum_y_pred, sum_sq_y_pred = 0.0, 0.0
    sum_sst_y_pred, sum_slp_y_pred = 0.0, 0.0
    
    # Accumulateurs pour Y_true (Targets dans l'espace latent)
    sum_y_true, sum_sq_y_true = 0.0, 0.0
    sum_sst_y_true, sum_slp_y_true = 0.0, 0.0

    # --- NOUVEAU : Accumulateur pour la corrélation du modèle ---
    sum_ypred_ytrue = 0.0
    
    n_samples_std = 0
    weights = model.linear.weight.detach().cpu().numpy()

    with torch.no_grad():
        for X_sst, X_slp, y_target, _, _, _ in valloader:
            B = X_sst.size(0)
            n_samples_std += B
            
            # --- 1. Préparation des inputs (X) ---
            X_sst_np = X_sst.cpu().numpy()
            sum_sst += X_sst_np.sum(axis=0)
            sum_sq_sst += (X_sst_np ** 2).sum(axis=0)
            X_sst_flat = X_sst_np.reshape(B, -1)
            
            if X_slp.numel() > 0:
                X_slp_np = X_slp.cpu().numpy()
                sum_slp += X_slp_np.sum(axis=0)
                sum_sq_slp += (X_slp_np ** 2).sum(axis=0)
                X_slp_flat = X_slp_np.reshape(B, -1)
                X_in = np.concatenate([X_sst_flat, X_slp_flat], axis=1)
            else:
                X_in = X_sst_flat
                X_slp_np = None

            # --- 2. Calcul de l'output prédit (Y_pred) ---
            # Y_pred shape : (Batch, Latent_Dim)
            Y_pred = X_in @ weights.T
            sum_y_pred += Y_pred.sum(axis=0)
            sum_sq_y_pred += (Y_pred ** 2).sum(axis=0)
            
            # --- 3. Encodage de la vraie Target dans l'espace latent (Y_true_latent) ---
            if args.embed_method == 'pca':
                slp_flat = y_target.view(B, -1).cpu().numpy()
                if args.lat_weight and wgts_flat is not None:
                    slp_flat = slp_flat * wgts_flat
                Y_true_latent = pca_model.transform(slp_flat)[:, :latent_dim]
            elif args.embed_method == 'vae':
                Y_true_latent_tensor, _ = vae_model.encode(y_target.to(device))
                Y_true_latent = Y_true_latent_tensor.cpu().numpy()
                
            sum_y_true += Y_true_latent.sum(axis=0)
            sum_sq_y_true += (Y_true_latent ** 2).sum(axis=0)
            
            # --- 4. Produits croisés (Moyennes temporelles des interactions) ---
            sum_sst_y_pred += X_sst_flat.T @ Y_pred
            sum_sst_y_true += X_sst_flat.T @ Y_true_latent
            
            # --- NOUVEAU : Produit croisé global (Pred vs Target) ---
            sum_ypred_ytrue += (Y_pred * Y_true_latent).sum(axis=0)

            if X_slp_np is not None:
                sum_slp_y_pred += X_slp_flat.T @ Y_pred
                sum_slp_y_true += X_slp_flat.T @ Y_true_latent

    # --- Finalisation des statistiques fondamentales ---
    sst_mean = sum_sst / n_samples_std
    sst_std = np.sqrt(np.maximum((sum_sq_sst / n_samples_std) - (sst_mean ** 2), 0))
    sst_mean_flat = sst_mean.reshape(-1, 1)
    
    # Stats Pred
    y_pred_mean = sum_y_pred / n_samples_std
    y_pred_std = np.sqrt(np.maximum((sum_sq_y_pred / n_samples_std) - (y_pred_mean ** 2), 0))
    y_pred_mean_flat = y_pred_mean.reshape(1, -1)
    
    # Stats Target
    y_true_mean = sum_y_true / n_samples_std
    y_true_std = np.sqrt(np.maximum((sum_sq_y_true / n_samples_std) - (y_true_mean ** 2), 0))
    y_true_mean_flat = y_true_mean.reshape(1, -1)

    # --- NOUVEAU : Corrélation globale du modèle ---
    cov_ypred_ytrue = (sum_ypred_ytrue / n_samples_std) - (y_pred_mean * y_true_mean)
    corr_model = np.divide(cov_ypred_ytrue, (y_pred_std * y_true_std), out=np.zeros_like(cov_ypred_ytrue), where=(y_pred_std * y_true_std)!=0)
    
    # --- Calcul des Corrélations ---
    # SST Vs Pred
    cov_sst_ypred = (sum_sst_y_pred / n_samples_std) - (sst_mean_flat @ y_pred_mean_flat)
    denom_sst_pred = sst_std.reshape(-1, 1) @ y_pred_std.reshape(1, -1)
    corr_sst_pred = np.divide(cov_sst_ypred, denom_sst_pred, out=np.zeros_like(cov_sst_ypred), where=denom_sst_pred!=0)
    
    # SST Vs Target
    cov_sst_ytrue = (sum_sst_y_true / n_samples_std) - (sst_mean_flat @ y_true_mean_flat)
    denom_sst_true = sst_std.reshape(-1, 1) @ y_true_std.reshape(1, -1)
    corr_sst_target = np.divide(cov_sst_ytrue, denom_sst_true, out=np.zeros_like(cov_sst_ytrue), where=denom_sst_true!=0)
    
    # SLP Corrélations
    if len(active_slp_lags) > 0:
        slp_mean = sum_slp / n_samples_std
        slp_std = np.sqrt(np.maximum((sum_sq_slp / n_samples_std) - (slp_mean ** 2), 0))
        slp_mean_flat = slp_mean.reshape(-1, 1)
        
        # Vs Pred
        cov_slp_ypred = (sum_slp_y_pred / n_samples_std) - (slp_mean_flat @ y_pred_mean_flat)
        denom_slp_pred = slp_std.reshape(-1, 1) @ y_pred_std.reshape(1, -1)
        corr_slp_pred = np.divide(cov_slp_ypred, denom_slp_pred, out=np.zeros_like(cov_slp_ypred), where=denom_slp_pred!=0)
        
        # Vs Target
        cov_slp_ytrue = (sum_slp_y_true / n_samples_std) - (slp_mean_flat @ y_true_mean_flat)
        denom_slp_true = slp_std.reshape(-1, 1) @ y_true_std.reshape(1, -1)
        corr_slp_target = np.divide(cov_slp_ytrue, denom_slp_true, out=np.zeros_like(cov_slp_ytrue), where=denom_slp_true!=0)
    else:
        slp_std = None
        corr_slp_pred = corr_slp_target = None
        cov_slp_ypred = cov_slp_ytrue = None

    print("\n--- GÉNÉRATION DES CARTES D'EXPLICABILITÉ ---")
    
    def plot_explainability_weights(model, outdir, sst_lags, slp_lags, sst_std, slp_std, 
                                    corr_sst_p, corr_sst_t, corr_slp_p, corr_slp_t, 
                                    cov_sst_p, cov_sst_t, cov_slp_p, cov_slp_t, corr_model,
                                    sst_shape=(85, 360), slp_shape=(53, 113), max_components_to_plot=5):
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
            # --- VÉRIFICATION THÉORIQUE ---
            max_pixel_r = 0.0
            if in_chans_sst > 0:
                max_pixel_r = np.max(np.abs(corr_sst_t[:, comp_idx]))
            if in_chans_slp > 0:
                max_pixel_r = max(max_pixel_r, np.max(np.abs(corr_slp_t[:, comp_idx])))
            
            model_r = corr_model[comp_idx]
            print(f"Latent {comp_idx:03d} | R Modèle complet: {model_r:.4f} | R Meilleur Pixel: {max_pixel_r:.4f}")

            # ==========================================================
            # --- PLOT HISTOGRAMME DES CORRÉLATIONS ---
            # ==========================================================
            plt.figure(figsize=(8, 5), facecolor='white')
            
            if in_chans_sst > 0:
                pixel_corrs_sst = corr_sst_t[:, comp_idx].flatten()
                plt.hist(pixel_corrs_sst, bins=60, color='royalblue', edgecolor='black', alpha=0.7, label='SST Pixels vs Target')
            
            if in_chans_slp > 0:
                pixel_corrs_slp = corr_slp_t[:, comp_idx].flatten()
                plt.hist(pixel_corrs_slp, bins=60, color='forestgreen', edgecolor='black', alpha=0.6, label='SLP Pixels vs Target')

            plt.axvline(x=model_r, color='crimson', linestyle='dashed', linewidth=2.5, label=f'Prédiction Modèle ($r={model_r:.3f}$)')
            
            plt.title(f'Distribution des corrélations spatiales - Latent {comp_idx}', fontsize=14)
            plt.xlabel(rf'Coefficient de corrélation de Pearson ($r$)', fontsize=12)
            plt.ylabel('Nombre de pixels', fontsize=12)
            plt.xlim(-1, 1)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            hist_outpath = os.path.join(comp_outdir, f"histogram_corr_comp_{comp_idx:03d}.png")
            plt.savefig(hist_outpath, dpi=150, bbox_inches='tight')
            plt.close()

            # ==========================================================
            # --- PLOT SST (6 Lignes) ---
            # ==========================================================
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

                fig, axes = plt.subplots(6, in_chans_sst, figsize=(6 * in_chans_sst, 24), subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
                if in_chans_sst == 1: axes = np.expand_dims(axes, axis=1)

                for i, lag in enumerate(sst_lags):
                    ax = axes[0, i]
                    im = ax.imshow(sst_w_raw[i], cmap='RdBu_r', origin='lower', vmin=-vmax_raw, vmax=vmax_raw, transform=ccrs.PlateCarree(), extent=extent_sst, interpolation='nearest')
                    ax.set_title(rf"Raw Coefs ($\beta$) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_sst,crs=ccrs.PlateCarree())
                    if i == in_chans_sst - 1: fig.colorbar(im, ax=axes[0, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)

                    ax = axes[1, i]
                    im = ax.imshow(sst_w_eff[i], cmap='RdBu_r', origin='lower', vmin=-vmax_eff, vmax=vmax_eff, transform=ccrs.PlateCarree(), extent=extent_sst, interpolation='nearest')
                    ax.set_title(rf"Effective Sensitivity ($\beta \times \sigma$) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_sst,crs=ccrs.PlateCarree())
                    if i == in_chans_sst - 1: fig.colorbar(im, ax=axes[1, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)
                    
                    ax = axes[2, i]
                    im = ax.imshow(sst_c_pred[i], cmap='RdBu_r', origin='lower', vmin=-1, vmax=1, transform=ccrs.PlateCarree(), extent=extent_sst, interpolation='nearest')
                    ax.set_title(rf"Correlation (Pixel vs Pred) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_sst,crs=ccrs.PlateCarree())
                    if i == in_chans_sst - 1: fig.colorbar(im, ax=axes[2, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)

                    ax = axes[3, i]
                    im = ax.imshow(sst_c_targ[i], cmap='RdBu_r', origin='lower', vmin=-1, vmax=1, transform=ccrs.PlateCarree(), extent=extent_sst, interpolation='nearest')
                    ax.set_title(rf"Correlation (Pixel vs Target) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_sst,crs=ccrs.PlateCarree())
                    if i == in_chans_sst - 1: fig.colorbar(im, ax=axes[3, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)

                    ax = axes[4, i]
                    im = ax.imshow(sst_cov_pred[i], cmap='RdBu_r', origin='lower', vmin=-vmax_cov, vmax=vmax_cov, transform=ccrs.PlateCarree(), extent=extent_sst, interpolation='nearest')
                    ax.set_title(rf"Covariance (Pixel vs Pred) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_sst,crs=ccrs.PlateCarree())
                    if i == in_chans_sst - 1: fig.colorbar(im, ax=axes[4, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)

                    ax = axes[5, i]
                    im = ax.imshow(sst_cov_targ[i], cmap='RdBu_r', origin='lower', vmin=-vmax_cov, vmax=vmax_cov, transform=ccrs.PlateCarree(), extent=extent_sst, interpolation='nearest')
                    ax.set_title(rf"Covariance (Pixel vs Target) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_sst,crs=ccrs.PlateCarree())
                    if i == in_chans_sst - 1: fig.colorbar(im, ax=axes[5, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)

                fig.suptitle(rf"SST Diagnostics - Latent {comp_idx}", fontsize=16, y=0.99)
                plt.savefig(os.path.join(comp_outdir, f"explainability_SST_comp_{comp_idx:03d}.png"), dpi=150, bbox_inches='tight')
                plt.close()

            # ==========================================================
            # --- PLOT SLP (6 Lignes, si existant) ---
            # ==========================================================
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

                fig, axes = plt.subplots(6, in_chans_slp, figsize=(6 * in_chans_slp, 24), subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
                if in_chans_slp == 1: axes = np.expand_dims(axes, axis=1)

                for i, lag in enumerate(slp_lags):
                    ax = axes[0, i]
                    im = ax.imshow(slp_w_raw[i], cmap='RdBu_r', origin='lower', vmin=-vmax_raw_slp, vmax=vmax_raw_slp, transform=ccrs.PlateCarree(), extent=extent_slp, interpolation='nearest')
                    ax.set_title(rf"Raw Coefs ($\beta$) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_slp,crs=ccrs.PlateCarree())
                    if i == in_chans_slp - 1: fig.colorbar(im, ax=axes[0, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)

                    ax = axes[1, i]
                    im = ax.imshow(slp_w_eff[i], cmap='RdBu_r', origin='lower', vmin=-vmax_eff_slp, vmax=vmax_eff_slp, transform=ccrs.PlateCarree(), extent=extent_slp, interpolation='nearest')
                    ax.set_title(rf"Effective Sensitivity ($\beta \times \sigma$) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_slp,crs=ccrs.PlateCarree())
                    if i == in_chans_slp - 1: fig.colorbar(im, ax=axes[1, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)
                    
                    ax = axes[2, i]
                    im = ax.imshow(slp_c_pred[i], cmap='RdBu_r', origin='lower', vmin=-1, vmax=1, transform=ccrs.PlateCarree(), extent=extent_slp, interpolation='nearest')
                    ax.set_title(rf"Correlation (Pixel vs Pred) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_slp,crs=ccrs.PlateCarree())
                    if i == in_chans_slp - 1: fig.colorbar(im, ax=axes[2, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)

                    ax = axes[3, i]
                    im = ax.imshow(slp_c_targ[i], cmap='RdBu_r', origin='lower', vmin=-1, vmax=1, transform=ccrs.PlateCarree(), extent=extent_slp, interpolation='nearest')
                    ax.set_title(rf"Correlation (Pixel vs Target) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_slp,crs=ccrs.PlateCarree())
                    if i == in_chans_slp - 1: fig.colorbar(im, ax=axes[3, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)
                    
                    ax = axes[4, i]
                    im = ax.imshow(slp_cov_pred[i], cmap='RdBu_r', origin='lower', vmin=-vmax_cov_slp, vmax=vmax_cov_slp, transform=ccrs.PlateCarree(), extent=extent_slp, interpolation='nearest')
                    ax.set_title(rf"Covariance (Pixel vs Pred) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_slp,crs=ccrs.PlateCarree())
                    if i == in_chans_slp - 1: fig.colorbar(im, ax=axes[4, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)

                    ax = axes[5, i]
                    im = ax.imshow(slp_cov_targ[i], cmap='RdBu_r', origin='lower', vmin=-vmax_cov_slp, vmax=vmax_cov_slp, transform=ccrs.PlateCarree(), extent=extent_slp, interpolation='nearest')
                    ax.set_title(rf"Covariance (Pixel vs Target) - Lag {lag}")
                    ax.coastlines(color='black', linewidth=0.8)
                    ax.set_extent(extent_slp,crs=ccrs.PlateCarree())
                    if i == in_chans_slp - 1: fig.colorbar(im, ax=axes[5, :].ravel().tolist(), shrink=0.7, orientation='vertical', pad=0.02)

                fig.suptitle(rf"SLP Diagnostics - Latent {comp_idx}", fontsize=16, y=0.99)
                plt.savefig(os.path.join(comp_outdir, f"explainability_SLP_comp_{comp_idx:03d}.png"), dpi=150, bbox_inches='tight')
                plt.close()
            
    plot_explainability_weights(model, outdir, active_sst_lags, active_slp_lags, sst_std, slp_std, 
                                corr_sst_pred, corr_sst_target, corr_slp_pred, corr_slp_target, 
                                cov_sst_ypred, cov_sst_ytrue, cov_slp_ypred, cov_slp_ytrue, corr_model,# <--- NOUVEAU
                                max_components_to_plot=10)
    print(f"Training complete, elapsed time: {(time.time() - start_time) / 60:.2f} minutes")

    # ============================================================
    # 8. LANCEMENT DES ÉVALUATIONS AUTOMATIQUES FINALES
    # ============================================================
    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation automatique...")
    print("="*50)

    # Assure-toi que ces scripts existent et sont nommés ainsi, ou adapte le nom !
    eval_script_path = os.path.join(os.path.dirname(__file__), "eval_linreg.py")
    eval_spatial_script_path = os.path.join(os.path.dirname(__file__), "eval_linreg_full_slp.py")

    for model_type in ["final", "best"]:
        for monthly_mean in [False, True]:
            print(f"\n--- Évaluation globale du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            
            # --- EVAL CLASSIQUE ---
            eval_command = [
                sys.executable, eval_script_path,
                "--machine", str(args.machine), "--embed_method", str(args.embed_method),
                "--linreg_dir", str(outdir), # Ou --cnn_dir selon ce qu'attend ton script !
                "--model_type", str(model_type), "--nb_members_train", str(args.nb_members_train),
                "--nb_members_val", str(args.nb_members_val), "--nb_members_test", str(args.nb_members_test),"--seed", str(args.seed),
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
                "--linreg_dir", str(outdir), # Ou --cnn_dir
                "--model_type", str(model_type), "--nb_members_train", str(args.nb_members_train),"--nb_members_val", str(args.nb_members_val), "--nb_members_test", str(args.nb_members_test),
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