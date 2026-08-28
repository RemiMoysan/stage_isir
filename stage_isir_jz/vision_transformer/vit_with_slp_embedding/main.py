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
from datetime import datetime
from collections import defaultdict
import re
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from shared_tools.visualizations import loss_figure, loss_first_epoch, plot_and_save_maps_with_reconstruction_light, plot_reconstruction_check, plot_correlation_evolution, plot_r2_R2_evolution, MapMetricTracker, LatentMetricTracker, save_r2_pixel_map_and_plot, plot_map_r2_evolution, plot_spatial_corr_evolution, plot_latent_l1_ss_evolution

from shared_tools.datasets import Dataset, Dataset_mensuel
from shared_tools.models import ConvVAE, vae_loss, compute_loss, get_median_prediction_full_slp, decode_latent_to_map

from tools.models import ViT_Latent_SLP_Multimodal_tunable

# ============================================================
# DEVICE & ARGUMENTS CONFIGURATION & OUTPUT DIRECTORY SETUP
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', type=int, required=True, help='Loading of previous parameters (1) or start fresh (0)')
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='vae', help='Méthode pour l\'espace latent')
    parser.add_argument('--embed_path', type=str, default='', help='Chemin vers le VAE/PCA pré-entraîné. Si vide, entraîne un nouveau.')
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'], help='Machine sur laquelle le code tourne')
    
    parser.add_argument('--nb_members_train', type=int, default=10, help='Nombre de membres à utiliser pour l\'entraînement')
    parser.add_argument('--nb_members_val', type=int, default=5, help='Nombre de membres à utiliser pour la validation')
    parser.add_argument('--nb_members_test', type=int, default=0, help='Nombre de membres à utiliser pour le test') 
    
    parser.add_argument('--seed', type=int, default=42, help='Seed pour le mélange inter membres')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension de l\'espace latent cible')
    parser.add_argument('--nb_epochs', type=int, default=30, help='Nombre d\'époques pour l\'entraînement du ViT')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours')
    parser.add_argument('--bs', type=int, default=128, help='Taille de batch pour l\'entraînement')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate pour l\'entraînement du ViT')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Poids de la régularisation L2 (Weight Decay)')

    # Arguments Tunables ViT
    parser.add_argument('--dr', type=float, default=0.1, help='Dropout rate pour le ViT')
    parser.add_argument('--embed_dim', type=int, default=128, help='Dimension interne du Transformer')
    parser.add_argument('--depth', type=int, default=4, help='Profondeur du réseau Transformer')
    parser.add_argument('--num_heads', type=int, default=4, help='Nombre de têtes d\'attention')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='Multiplicateur du FeedForward')
    parser.add_argument('--transformer_act', type=str, choices=['gelu', 'relu'], default='gelu', help='Activation interne')
    parser.add_argument('--pool_strategy', type=str, choices=['cls', 'gap'], default='cls', help='Stratégie de pooling finale')
    parser.add_argument('--head_hidden_dim', type=int, default=0, help='Dimension de la couche cachée de la tête (0 = embed_dim)')
    parser.add_argument('--head_act', type=str, choices=['tanh', 'relu'], default='tanh', help='Activation de la tête')
    parser.add_argument('--norm_first', action='store_true', help='Pre-Norm ou Post-Norm')
    parser.add_argument('--patch_size_sst', type=int, nargs=2, default=[5, 10], help='Taille des patches SST (y, x)')
    parser.add_argument('--patch_size_slp', type=int, nargs=2, default=[5, 5], help='Taille des patches SLP (y, x)')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95], help='Liste des lags pour SST')
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[], help='Liste des lags pour SLP')
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4], help='Liste des lags mois pour SST')
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[], help='Liste des lags mois pour SLP')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target')
    parser.add_argument('--beta_kld', type=float, default=1.0, help='Coefficient KL divergence VAE')
    parser.add_argument('--normalize', action='store_true', help='PCA normalisé ou non')
    parser.add_argument('--use_lags_attention', action='store_true', help='Attention temporelle entre les lags')
    parser.add_argument('--roll_sst', action='store_true', help='Appliquer un roll sur les données SST')
    
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile','correlation'], default='mse', help='Loss')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Points de validation intra-époque')
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données mensuelles')
    parser.add_argument('--lat_weight', action='store_true', help='Pondération spatiale sqrt(cos(lat))')
    args = parser.parse_args()

    if args.loss_type == 'quantile':
        if 0.5 not in args.quantiles:
            raise ValueError("Erreur: Pour la quantile loss, la médiane (0.5) DOIT être incluse.")

    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/vision_transformer/vit_with_slp_embedding/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']:
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/"
    elif args.machine == 'mac_local':
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/vision_transformer/vit_with_slp_embedding/"

    latent_dim = args.latent_dim
    sst_lags_days = args.sst_lags_days
    slp_lags_days = args.slp_lags_days
    sst_lags_months = args.sst_lags_months
    slp_lags_months = args.slp_lags_months
    bs = args.bs
    lr = args.lr
    dr = args.dr
    weight_decay = args.weight_decay
    winter_months = args.winter_months
    duree_lissage = args.duree_lissage
    nb_epochs = args.nb_epochs
    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val
    nb_members_test = args.nb_members_test

    patience = 10000
    target_indices = {100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000} if not args.monthly_reduction else {1, 10, 20,30,40,45,50,60,70, 80} 

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[-nb_members_val:]
    test_members = all_members[nb_members_train:nb_members_train + nb_members_test] if nb_members_test > 0 else []

    dynamic_slp_std = 596.0 
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match: dynamic_slp_std = float(match.group(1))

    loss_tag = args.loss_type
    if args.loss_type == 'quantile':
        loss_tag += "_" + "".join([str(q).replace('.','') for q in args.quantiles])

    model_spec = f"emb{args.embed_dim}d{args.depth}h{args.num_heads}pool{args.pool_strategy}wd{weight_decay}"
    
    if args.embed_method == 'pca':
        if not args.monthly_reduction:
            outdir_name = f"ViT_{model_spec}_{duree_lissage}d_att_{args.use_lags_attention}_loss_{loss_tag}_{args.embed_method}n{args.normalize}{latent_dim}_months{''.join(map(str, winter_months))}_lat{args.lat_weight}_sst{''.join(map(str, sst_lags_days))}slp{''.join(map(str, slp_lags_days))}_train{nb_members_train}val{nb_members_val}_bs{bs}lr{lr}dr{dr}roll{args.roll_sst}_std{dynamic_slp_std}"
        else:
            outdir_name = f"ViT_{model_spec}_monthly_att_{args.use_lags_attention}_loss_{loss_tag}_{args.embed_method}n{args.normalize}{latent_dim}_months{''.join(map(str, winter_months))}_lat{args.lat_weight}_sst{''.join(map(str, sst_lags_months))}slp{''.join(map(str, slp_lags_months))}_train{nb_members_train}val{nb_members_val}_bs{bs}lr{lr}dr{dr}roll{args.roll_sst}_std{dynamic_slp_std}"
    elif args.embed_method == 'vae':
        if not args.monthly_reduction:
            outdir_name = f"ViT_{model_spec}_{duree_lissage}d_att_{args.use_lags_attention}_loss_{loss_tag}_{args.embed_method}beta{args.beta_kld}{latent_dim}_months{''.join(map(str, winter_months))}_lat{args.lat_weight}_sst{''.join(map(str, sst_lags_days))}slp{''.join(map(str, slp_lags_days))}_train{nb_members_train}val{nb_members_val}_bs{bs}lr{lr}dr{dr}roll{args.roll_sst}_std{dynamic_slp_std}"
        else:
            outdir_name = f"ViT_{model_spec}_monthly_att_{args.use_lags_attention}_loss_{loss_tag}_{args.embed_method}beta{args.beta_kld}{latent_dim}_months{''.join(map(str, winter_months))}_lat{args.lat_weight}_sst{''.join(map(str, sst_lags_months))}slp{''.join(map(str, slp_lags_months))}_train{nb_members_train}val{nb_members_val}_bs{bs}lr{lr}dr{dr}roll{args.roll_sst}_std{dynamic_slp_std}"
            
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1)
    intra_workers = min(2, n_workers)

    # ============================================================
    # DATALOADERS
    # ============================================================
    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        training_set = Dataset(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        training_set = Dataset_mensuel(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)

    if nb_members_test > 0:
        if not args.monthly_reduction:
            test_set = Dataset(members=test_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)
        else:
            test_set = Dataset_mensuel(members=test_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
        testloader_intra = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=intra_workers, pin_memory=True)
    else:
        testloader, testloader_intra = None, None

    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=intra_workers, pin_memory=True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # PRÉPARATION DES POIDS SPATIAUX
    # ============================================================
    wgts_flat, safe_wgts, area_weights_2d = None, None, None  
    if args.lat_weight:
        sample_member = train_members[0]
        sample_path = os.path.join(base_home.replace("stage_isir_jz/vision_transformer/vit_with_slp_embedding/", ""), f"data/SLP/PSL_anom_LE2-{sample_member}_1mo.nc")
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            h, w = len(lats), len(ds_sample['lon'].values)
            if args.embed_method == 'pca':
                wgts = np.sqrt(coslat).reshape(h, 1)
                wgts_flat = np.broadcast_to(wgts, (h, w)).flatten()
                safe_wgts = np.maximum(wgts_flat, 1e-5)
            area_weights_2d = torch.tensor(np.broadcast_to(coslat.reshape(h, 1), (h, w)), dtype=torch.float64, device=device)
            ds_sample.close()
        except Exception as e:
            print(f"Erreur chargement grille latitude : {e}")

    # ============================================================
    # PRÉPARATION DE L'EMBEDDER (PCA ou VAE)
    # ============================================================
    pca_model, vae_model = None, None

    if args.embed_method == 'pca':
        if args.embed_path and os.path.exists(args.embed_path):
            pca_model = joblib.load(args.embed_path)
        else:
            pca_model = PCA(n_components=latent_dim, whiten=args.normalize) 
            slp_list = []
            for X_sst, X_slp, y_target, y_map, dates, members in trainloader:
                slp_data_raw = y_target.view(y_target.size(0), -1).numpy()
                if args.lat_weight and wgts_flat is not None: slp_data_raw *= wgts_flat
                slp_list.append(slp_data_raw)
            pca_model.fit(np.concatenate(slp_list, axis=0))
            joblib.dump(pca_model, os.path.join(outdir, "pca_model.joblib"))

    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=latent_dim).to(device)
        if args.embed_path and os.path.exists(args.embed_path):
            vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        else:
            optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
            vae_model.train()
            for v_epoch in range(10): 
                for X_sst, X_slp, y_target, y_map, dates, members in trainloader:
                    y_target = y_target.to(device)
                    optimizer_vae.zero_grad()
                    recon, mu, logvar = vae_model(y_target)
                    loss = vae_loss(recon, y_target, mu, logvar, beta=args.beta_kld)
                    loss.backward(); optimizer_vae.step()
            torch.save(vae_model.state_dict(), os.path.join(outdir, "vae_model.pth"))
        vae_model.eval()
        for param in vae_model.parameters(): param.requires_grad = False

    # ============================================================
    # INITIALISATION DU VISION TRANSFORMER (TUNABLE)
    # ============================================================
    out_features = latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else latent_dim
    active_sst_lags = sst_lags_months if args.monthly_reduction else sst_lags_days
    active_slp_lags = slp_lags_months if args.monthly_reduction else slp_lags_days

    h_dim = args.head_hidden_dim if args.head_hidden_dim > 0 else None

    model = ViT_Latent_SLP_Multimodal_tunable(
        sst_size=(85, 360), patch_size_sst=tuple(args.patch_size_sst), in_chans_sst=len(active_sst_lags), 
        slp_size=(53, 113), patch_size_slp=tuple(args.patch_size_slp), in_chans_slp=len(active_slp_lags), 
        nb_out=out_features, 
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, 
        mlp_ratio=args.mlp_ratio, transformer_act=args.transformer_act, dr=dr, 
        use_lags_attention=args.use_lags_attention, pool_strategy=args.pool_strategy, 
        head_hidden_dim=h_dim, head_act=args.head_act, norm_first=args.norm_first
    ).to(device)

    # NOUVEAU: Optimiseur avec weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("Number of ViT parameters : ", sum(p.numel() for p in model.parameters()))

    # ============================================================
    # PIPELINE DE SUIVI (Trackers Unifiés)
    # ============================================================
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

    ### SUIVI INTRA-ÉPOQUE
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
        initial_params = torch.load(f"{outdir}/final_model_ViT.pth")
        model.load_state_dict(initial_params['state_dict'])
        optimizer.load_state_dict(initial_params['optimizer'])
        train_losses = initial_params['train_losses']; val_losses = initial_params['val_losses']
        best_val_loss = np.min(val_losses) if len(val_losses)>0 else float('inf')
        print("Model state updated")
    else:
        print("Initiated first ViT training")

    # ============================================================
    # TRAINING & EVALUATION LOOP
    # ============================================================
    start_time = time.time()
    epoch_times = []
    best_model_state = None
    patience_counter = 0
    

    for epoch in range(nb_epochs):
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0
        train_latent_tracker = LatentMetricTracker(device=device)
        train_map_tracker = MapMetricTracker(shape=(53, 113), device=device)
        
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
            optimizer.zero_grad()
            X_sst = X_sst.to(device, non_blocking=True); X_slp = X_slp.to(device, non_blocking=True) 
            
            if args.embed_method == 'pca':
                slp_flat = y_target.view(y_target.size(0), -1).numpy()
                if args.lat_weight and wgts_flat is not None: slp_flat *= wgts_flat
                embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
            elif args.embed_method == 'vae':
                y_target = y_target.to(device, non_blocking=True) 
                with torch.no_grad(): target_embed, _ = vae_model.encode(y_target)
                    
            predicted_latent = model(X_sst, X_slp)            
            loss_value = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')

            loss_value.backward(); optimizer.step()
            running_train_loss += loss_value.item() * X_sst.size(0)
            total_train_samples += X_sst.size(0)

            med_pred = get_median_prediction_full_slp(predicted_latent, args.loss_type, args.quantiles) if args.loss_type == 'quantile' else predicted_latent 
            train_latent_tracker.update(med_pred.detach(), target_embed.detach())
            train_map_tracker.update(y_target.detach(), decode_latent_to_map(predicted_latent, args, latent_dim, pca_model, vae_model, safe_wgts).detach())

            if epoch == 0:
                epoch1_batch_losses.append(loss_value.item())
                with torch.no_grad():
                    epoch1_baseline_losses.append(compute_loss(torch.zeros_like(predicted_latent), target_embed, args.loss_type, args.quantiles, reduction='mean').item())

            # --- INTRA-EPOCH EVAL ---
            if args.nb_intra_evals > 0 and (epoch == 0 or epoch == 1):  
                current_eval_steps_set = eval_steps_set if epoch == 0 else eval_steps_epoch2_set
                if batch_idx in current_eval_steps_set or batch_idx == len(trainloader) - 1:
                    eval_phases = ['val', 'test'] if nb_members_test > 0 else ['val']
                    for key in eval_phases:
                        loader = valloader_intra if key == 'val' else testloader_intra
                        model.eval()
                        
                        intra_val_loss = 0.0; intra_n_samples = 0
                        intra_latent_tracker = LatentMetricTracker(device=device)
                        intra_map_tracker = MapMetricTracker(shape=(53, 113), device=device)      
                        
                        with torch.no_grad():
                            for v_X_sst, v_X_slp, v_y_target, _, _, _ in loader:
                                v_X_sst = v_X_sst.to(device, non_blocking=True); v_X_slp = v_X_slp.to(device, non_blocking=True); v_y_target = v_y_target.to(device, non_blocking=True)
                                
                                if args.embed_method == 'pca':
                                    slp_flat = v_y_target.view(v_y_target.size(0), -1).cpu().numpy()
                                    if args.lat_weight and wgts_flat is not None: slp_flat *= wgts_flat
                                    v_target_embed = torch.tensor(pca_model.transform(slp_flat)[:, :latent_dim], dtype=torch.float32).to(device, non_blocking=True)
                                elif args.embed_method == 'vae':
                                    v_target_embed, _ = vae_model.encode(v_y_target)
                                    
                                v_pred = model(v_X_sst, v_X_slp)
                                intra_val_loss += compute_loss(v_pred, v_target_embed, args.loss_type, args.quantiles, reduction='mean').item() * v_X_sst.size(0)
                                
                                p = get_median_prediction_full_slp(v_pred, args.loss_type, args.quantiles) if args.loss_type == 'quantile' else v_pred
                                intra_latent_tracker.update(p.detach(), v_target_embed.detach())
                                intra_map_tracker.update(v_y_target.detach(), decode_latent_to_map(v_pred, args, latent_dim, pca_model, vae_model, safe_wgts).detach())
                                intra_n_samples += p.size(0)

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
                                intra_epoch1_val_map_gCorr.append(map_gCorr); intra_epoch1_val_map_gL1.append(map_gL1); intra_epoch1_val_map_mL1.append(map_mL1)
                            else:
                                intra_epoch1_test_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch1_test_lat_gR2.append(lat_gR2); intra_epoch1_test_lat_mR2.append(lat_mR2)
                                intra_epoch1_test_lat_gCorr.append(lat_gCorr); intra_epoch1_test_lat_mCorr.append(lat_mCorr)
                                intra_epoch1_test_lat_mk.append(lat_mK); intra_epoch1_test_lat_gk.append(lat_gK)
                                intra_epoch1_test_lat_gL1.append(lat_gL1); intra_epoch1_test_lat_mL1.append(lat_mL1)
                                intra_epoch1_test_map_gR2.append(map_gR2); intra_epoch1_test_map_mR2.append(map_mR2)
                                intra_epoch1_test_map_sCorr.append(map_sCorr); intra_epoch1_test_map_tCorr.append(map_tCorr)
                                intra_epoch1_test_map_gCorr.append(map_gCorr); intra_epoch1_test_map_gL1.append(map_gL1); intra_epoch1_test_map_mL1.append(map_mL1)
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
                                intra_epoch2_val_map_gCorr.append(map_gCorr); intra_epoch2_val_map_gL1.append(map_gL1); intra_epoch2_val_map_mL1.append(map_mL1)
                            else:
                                intra_epoch2_test_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch2_test_lat_gR2.append(lat_gR2); intra_epoch2_test_lat_mR2.append(lat_mR2)
                                intra_epoch2_test_lat_gCorr.append(lat_gCorr); intra_epoch2_test_lat_mCorr.append(lat_mCorr)
                                intra_epoch2_test_lat_mk.append(lat_mK); intra_epoch2_test_lat_gk.append(lat_gK)
                                intra_epoch2_test_lat_gL1.append(lat_gL1); intra_epoch2_test_lat_mL1.append(lat_mL1)
                                intra_epoch2_test_map_gR2.append(map_gR2); intra_epoch2_test_map_mR2.append(map_mR2)
                                intra_epoch2_test_map_sCorr.append(map_sCorr); intra_epoch2_test_map_tCorr.append(map_tCorr)
                                intra_epoch2_test_map_gCorr.append(map_gCorr); intra_epoch2_test_map_gL1.append(map_gL1); intra_epoch2_test_map_mL1.append(map_mL1)

                        if key == 'val' and (intra_val_loss / intra_n_samples) < best_val_loss:
                            best_val_loss = (intra_val_loss / intra_n_samples)
                            best_model_state = copy.deepcopy(model.state_dict())
                            if best_model_path and os.path.exists(best_model_path): os.remove(best_model_path)
                            best_model_path = os.path.join(outdir, f'best_val_ViT_ep{epoch+1}_step{batch_idx}.pth')
                            torch.save(model.state_dict(), best_model_path)

                        model.train()

        # --- BILAN EPOQUE TRAIN ---
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
        train_map_gCorr.append(map_gCorr); train_map_gL1.append(map_gL1); train_map_mL1.append(map_mL1)

        save_r2_pixel_map_and_plot(map_r2_np, outdir, f"L2_train_ep{epoch+1}", metric_type="l2")
        save_r2_pixel_map_and_plot(map_l1_np, outdir, f"L1_train_ep{epoch+1}", metric_type="l1")
        save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_train_ep{epoch+1}", metric_type="corr")

        if epoch == 0 or epoch == 1:
            if epoch == 0: loss_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, outdir, label="Train")
            if args.nb_intra_evals > 0:
                c_steps = intra_epoch1_steps if epoch == 0 else intra_epoch2_steps
                c_losses = intra_epoch1_val_losses if epoch == 0 else intra_epoch2_val_losses
                c_test_losses = intra_epoch1_test_losses if epoch == 0 else intra_epoch2_test_losses
                
                loss_first_epoch(c_losses, [np.mean(epoch1_baseline_losses)]*len(c_losses), outdir, label="Intra-Val", batch_indexes=c_steps, epoch_num=epoch+1, batch_test_losses=c_test_losses)

                e1, e2 = (epoch == 0), (epoch == 1)
                plot_correlation_evolution([], intra_epoch1_val_lat_mCorr if e1 else intra_epoch2_val_lat_mCorr, outdir, val_ks=intra_epoch1_val_lat_mk if e1 else intra_epoch2_val_lat_mk, test_corrs=intra_epoch1_test_lat_mCorr if e1 else intra_epoch2_test_lat_mCorr, test_ks=intra_epoch1_test_lat_mk if e1 else intra_epoch2_test_lat_mk, epoch_1=e1, epoch_2=e2, batch_indexes=c_steps, suffix="_Latent_Mean")
                plot_correlation_evolution([], intra_epoch1_val_lat_gCorr if e1 else intra_epoch2_val_lat_gCorr, outdir, val_ks=intra_epoch1_val_lat_gk if e1 else intra_epoch2_val_lat_gk, test_corrs=intra_epoch1_test_lat_gCorr if e1 else intra_epoch2_test_lat_gCorr, test_ks=intra_epoch1_test_lat_gk if e1 else intra_epoch2_test_lat_gk, epoch_1=e1, epoch_2=e2, batch_indexes=c_steps, suffix="_Latent_Global")
                plot_r2_R2_evolution([], intra_epoch1_val_lat_mCorr if e1 else intra_epoch2_val_lat_mCorr, [], intra_epoch1_val_lat_mR2 if e1 else intra_epoch2_val_lat_mR2, outdir, epoch_1=e1, epoch_2=e2, batch_indexes=c_steps, test_R2=intra_epoch1_test_lat_mR2 if e1 else intra_epoch2_test_lat_mR2, test_corrs=intra_epoch1_test_lat_mCorr if e1 else intra_epoch2_test_lat_mCorr, suffix="_Latent_Mean")
                plot_r2_R2_evolution([], intra_epoch1_val_lat_gCorr if e1 else intra_epoch2_val_lat_gCorr, [], intra_epoch1_val_lat_gR2 if e1 else intra_epoch2_val_lat_gR2, outdir, epoch_1=e1, epoch_2=e2, batch_indexes=c_steps, test_R2=intra_epoch1_test_lat_gR2 if e1 else intra_epoch2_test_lat_gR2, test_corrs=intra_epoch1_test_lat_gCorr if e1 else intra_epoch2_test_lat_gCorr, suffix="_Latent_Global")
                plot_latent_l1_ss_evolution([], intra_epoch1_val_lat_gL1 if e1 else intra_epoch2_val_lat_gL1, [], intra_epoch1_val_lat_mL1 if e1 else intra_epoch2_val_lat_mL1, outdir, test_g=intra_epoch1_test_lat_gL1 if e1 else intra_epoch2_test_lat_gL1, test_m=intra_epoch1_test_lat_mL1 if e1 else intra_epoch2_test_lat_mL1, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1)

                plot_map_r2_evolution([], intra_epoch1_val_map_gR2 if e1 else intra_epoch2_val_map_gR2, [], intra_epoch1_val_map_mR2 if e1 else intra_epoch2_val_map_mR2, outdir, test_map=intra_epoch1_test_map_gR2 if e1 else intra_epoch2_test_map_gR2, test_pix=intra_epoch1_test_map_mR2 if e1 else intra_epoch2_test_map_mR2, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1, norm="l2")
                plot_map_r2_evolution([], intra_epoch1_val_map_gL1 if e1 else intra_epoch2_val_map_gL1, [], intra_epoch1_val_map_mL1 if e1 else intra_epoch2_val_map_mL1, outdir, test_map=intra_epoch1_test_map_gL1 if e1 else intra_epoch2_test_map_gL1, test_pix=intra_epoch1_test_map_mL1 if e1 else intra_epoch2_test_map_mL1, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1, norm="l1")
                plot_spatial_corr_evolution([], intra_epoch1_val_map_sCorr if e1 else intra_epoch2_val_map_sCorr, [], intra_epoch1_val_map_tCorr if e1 else intra_epoch2_val_map_tCorr, [], intra_epoch1_val_map_gCorr if e1 else intra_epoch2_val_map_gCorr, outdir, test_sc=intra_epoch1_test_map_sCorr if e1 else intra_epoch2_test_map_sCorr, test_tc=intra_epoch1_test_map_tCorr if e1 else intra_epoch2_test_map_tCorr, test_gc=intra_epoch1_test_map_gCorr if e1 else intra_epoch2_test_map_gCorr, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1)

        # --- EVALUATION COMPLETE VAL/TEST ---
        per_member_loss = defaultdict(lambda: {'loss_sum': 0.0, 'count': 0})
        per_member_plots = defaultdict(lambda: {'time': [], 'slp_true': [], 'slp_recon_true': [], 'slp_pred': []})
        
        for key in (['val', 'test'] if nb_members_test > 0 else ['val']):
            loader = valloader if key == 'val' else testloader
            model.eval()
            
            running_val_loss = 0.0; total_val_samples = 0 
            eval_latent_tracker = LatentMetricTracker(device=device)
            eval_map_tracker = MapMetricTracker(shape=(53, 113), device=device)

            with torch.no_grad():
                for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(loader):
                    X_sst = X_sst.to(device, non_blocking=True); X_slp = X_slp.to(device, non_blocking=True); y_target = y_target.to(device, non_blocking=True)
                    
                    if args.embed_method == 'pca':
                        slp_flat = y_target.view(y_target.size(0), -1).cpu().numpy()
                        if args.lat_weight and wgts_flat is not None: slp_flat *= wgts_flat
                        target_embed = torch.tensor(pca_model.transform(slp_flat)[:, :latent_dim], dtype=torch.float32).to(device, non_blocking=True)
                    elif args.embed_method == 'vae':
                        target_embed, _ = vae_model.encode(y_target)

                    predicted_latent = model(X_sst, X_slp)
                    loss_value = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')
                    running_val_loss += loss_value.item() * X_sst.size(0); total_val_samples += X_sst.size(0)
                    per_sample_losses = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='none').cpu().numpy()

                    med_pred = get_median_prediction_full_slp(predicted_latent, args.loss_type, args.quantiles) if args.loss_type == 'quantile' else predicted_latent
                    eval_latent_tracker.update(med_pred.detach(), target_embed.detach())
                    
                    decoded_map = decode_latent_to_map(predicted_latent, args, latent_dim, pca_model, vae_model, safe_wgts)
                    eval_map_tracker.update(y_target.detach(), decoded_map.detach())

                    if key == 'val':
                        recon_true_map = decode_latent_to_map(target_embed, args, latent_dim, pca_model, vae_model, safe_wgts)
                        y_map_np = y_map.numpy()
                        members_list = [m if isinstance(m, str) else m.item().decode() if isinstance(m.item(), bytes) else str(m.item()) for m in members]
                        
                        for i, mem in enumerate(members_list):
                            per_member_loss[mem]['loss_sum'] += float(per_sample_losses[i])
                            per_member_loss[mem]['count'] += 1
                            if per_member_loss[mem]['count'] - 1 in target_indices:
                                per_member_plots[mem]['time'].append(str(dates[i]))
                                per_member_plots[mem]['slp_true'].append(y_map_np[i])
                                per_member_plots[mem]['slp_recon_true'].append(recon_true_map[i].cpu().numpy())
                                per_member_plots[mem]['slp_pred'].append(decoded_map[i].cpu().numpy())

            if key == 'val':
                for mem, d in per_member_loss.items():
                    val_losses_per_member_history[mem].append(d['loss_sum'] / d['count'] if d['count'] > 0 else float('nan'))

            val_loss = running_val_loss / total_val_samples 
            val_losses.append(val_loss) if key == 'val' else test_losses.append(val_loss)
            
            lat_gR2, lat_mR2, lat_gCorr, lat_mCorr, lat_gK, lat_mK, lat_gL1, lat_mL1 = eval_latent_tracker.compute()
            map_gR2, map_mR2, map_r2_np, map_sCorr, map_tCorr, map_gCorr, map_corr_np, map_gL1, map_mL1, map_l1_np = eval_map_tracker.compute(area_weights=area_weights_2d)
            
            if key == 'val':
                val_map_gR2.append(map_gR2); val_map_mR2.append(map_mR2); val_map_sCorr.append(map_sCorr); val_map_tCorr.append(map_tCorr); val_map_gCorr.append(map_gCorr); val_map_gL1.append(map_gL1); val_map_mL1.append(map_mL1)
                val_lat_gR2.append(lat_gR2); val_lat_mR2.append(lat_mR2); val_lat_gCorr.append(lat_gCorr); val_lat_mCorr.append(lat_mCorr); val_lat_mk.append(lat_mK); val_lat_gk.append(lat_gK); val_lat_gL1.append(lat_gL1); val_lat_mL1.append(lat_mL1)
            else:
                test_map_gR2.append(map_gR2); test_map_mR2.append(map_mR2); test_map_sCorr.append(map_sCorr); test_map_tCorr.append(map_tCorr); test_map_gCorr.append(map_gCorr); test_map_gL1.append(map_gL1); test_map_mL1.append(map_mL1)
                test_lat_gR2.append(lat_gR2); test_lat_mR2.append(lat_mR2); test_lat_gCorr.append(lat_gCorr); test_lat_mCorr.append(lat_mCorr); test_lat_mk.append(lat_mK); test_lat_gk.append(lat_gK); test_lat_gL1.append(lat_gL1); test_lat_mL1.append(lat_mL1)
            
            save_r2_pixel_map_and_plot(map_r2_np, outdir, f"L2_{key}_ep{epoch+1}", metric_type="l2")
            save_r2_pixel_map_and_plot(map_l1_np, outdir, f"L1_{key}_ep{epoch+1}", metric_type="l1")
            save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_{key}_ep{epoch+1}", metric_type="corr")

            if key == 'val':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    if best_model_path and os.path.exists(best_model_path): os.remove(best_model_path)
                    best_model_path = os.path.join(outdir, f'best_val_ViT_ep{epoch + 1}_end.pth')
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1

        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        print(f"--> Fin de l'époque {epoch + 1} - Elapsed Time: {current_time_min:.2f} minutes")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
            break
        
        # ---------------- AFFICHAGE TOUTES LES 2 ÉPOQUES ----------------
        if (epoch + 1) % 2 == 0:
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses}, f'{outdir}/final_model_ViT.pth')
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
                    slp_true_list=[np.array(d['slp_true'])], slp_recon_true_list=[np.array(d['slp_recon_true'])],
                    slp_pred_list=[np.array(d['slp_pred'])], time_list=d['time'], outdir=member_outdir, epoch=(epoch + 1)
                )

    # ============================================================
    # SAUVEGARDE FINALE
    # ============================================================
    print(f"Best Val Loss : {best_val_loss:.6f}")

    loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times, per_member_val_losses=val_losses_per_member_history, test_losses=test_losses)
    plot_r2_R2_evolution(train_lat_mCorr, val_lat_mCorr, train_lat_mR2, val_lat_mR2, outdir, test_R2=test_lat_mR2, test_corrs=test_lat_mCorr, suffix="_Latent_Mean")
    plot_r2_R2_evolution(train_lat_gCorr, val_lat_gCorr, train_lat_gR2, val_lat_gR2, outdir, test_R2=test_lat_gR2, test_corrs=test_lat_gCorr, suffix="_Latent_Global")
    plot_correlation_evolution(train_lat_mCorr, val_lat_mCorr, outdir, test_corrs=test_lat_mCorr, train_ks=train_lat_mk, val_ks=val_lat_mk, test_ks=test_lat_mk, suffix="_Latent_Mean")
    plot_correlation_evolution(train_lat_gCorr, val_lat_gCorr, outdir, test_corrs=test_lat_gCorr, train_ks=train_lat_gk, val_ks=val_lat_gk, test_ks=test_lat_gk, suffix="_Latent_Global")
    plot_latent_l1_ss_evolution(train_lat_gL1, val_lat_gL1, train_lat_mL1, val_lat_mL1, outdir, test_g=test_lat_gL1 if nb_members_test > 0 else None, test_m=test_lat_mL1 if nb_members_test > 0 else None)

    plot_map_r2_evolution(train_map_gR2, val_map_gR2, train_map_mR2, val_map_mR2, outdir, test_map=test_map_gR2 if nb_members_test > 0 else None, test_pix=test_map_mR2 if nb_members_test > 0 else None, norm="l2")
    plot_map_r2_evolution(train_map_gL1, val_map_gL1, train_map_mL1, val_map_mL1, outdir, test_map=test_map_gL1 if nb_members_test > 0 else None, test_pix=test_map_mL1 if nb_members_test > 0 else None, norm="l1")
    plot_spatial_corr_evolution(train_map_sCorr, val_map_sCorr, train_map_tCorr, val_map_tCorr, train_map_gCorr, val_map_gCorr, outdir, test_sc=test_map_sCorr, test_tc=test_map_tCorr, test_gc=test_map_gCorr)

    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses}, f'{outdir}/final_model_ViT.pth')
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_ViT.pth')

    print(f"Training complete, elapsed time: {(time.time() - start_time) / 60:.2f} minutes")

    # ======================
    # AUTOMATIC EVALUATIONS 
    # ======================
    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation automatique...")
    print("="*50)

    eval_script_path = os.path.join(os.path.dirname(__file__), "eval_vit_embedding.py")
    for model_type in ["final", "best"]:
        for monthly_mean in [False, True]:
            print(f"\n--- Évaluation du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            eval_command = [
                sys.executable, eval_script_path,
                "--machine", str(args.machine), "--embed_method", str(args.embed_method),
                "--vit_dir", str(outdir), "--model_type", str(model_type),
                "--nb_members_train", str(args.nb_members_train), "--nb_members_val", str(args.nb_members_val),
                "--nb_members_test", str(args.nb_members_test), "--seed", str(args.seed),
                "--latent_dim", str(latent_dim), "--duree_lissage", str(args.duree_lissage),
                "--bs", str(args.bs), "--loss_type", str(args.loss_type),
                
                # --- TRANSFERT DES ARGUMENTS TUNABLES ---
                "--dr", str(args.dr),
                "--embed_dim", str(args.embed_dim),
                "--depth", str(args.depth),
                "--num_heads", str(args.num_heads),
                "--mlp_ratio", str(args.mlp_ratio),
                "--transformer_act", str(args.transformer_act),
                "--pool_strategy", str(args.pool_strategy),
                "--head_hidden_dim", str(args.head_hidden_dim),
                "--head_act", str(args.head_act),
                "--patch_size_sst", str(args.patch_size_sst[0]), str(args.patch_size_sst[1]),
                "--patch_size_slp", str(args.patch_size_slp[0]), str(args.patch_size_slp[1])
            ]
            if args.norm_first: eval_command.append("--norm_first")
            
            if args.embed_path: eval_command.extend(["--embed_path", str(args.embed_path)])
            if args.sst_lags_days is not None: eval_command.extend(["--sst_lags_days"] + [str(x) for x in args.sst_lags_days])
            if args.sst_lags_months is not None: eval_command.extend(["--sst_lags_months"] + [str(x) for x in args.sst_lags_months])
            if args.slp_lags_days is not None: eval_command.extend(["--slp_lags_days"] + [str(x) for x in args.slp_lags_days])
            if args.slp_lags_months is not None: eval_command.extend(["--slp_lags_months"] + [str(x) for x in args.slp_lags_months])
            if args.winter_months is not None: eval_command.extend(["--winter_months"] + [str(x) for x in args.winter_months])
            if args.quantiles: eval_command.extend(["--quantiles"] + [str(x) for x in args.quantiles])

            if args.roll_sst: eval_command.append("--roll_sst")
            if args.use_lags_attention: eval_command.append("--use_lags_attention")
            if monthly_mean: eval_command.append("--monthly_mean")
            if args.monthly_reduction: eval_command.append("--monthly_reduction")
            if args.lat_weight: eval_command.append("--lat_weight")

            try:
                result = subprocess.run(eval_command, check=True, text=True)
                print(f"✅ Évaluation de {model_type} terminée avec succès !")
            except subprocess.CalledProcessError as e:
                print(f"❌ Erreur lors de l'évaluation. Code: {e.returncode}")

    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation SPATIALE automatique...")
    print("="*50)

    eval_spatial_script_path = os.path.join(os.path.dirname(__file__), "eval_vit_spatial.py")
    for model_type in ["final", "best"]:
        for monthly_mean in [False, True]:
            print(f"\n--- Évaluation spatiale du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            eval_spatial_command = [
                sys.executable, eval_spatial_script_path,
                "--machine", str(args.machine), "--embed_method", str(args.embed_method),
                "--vit_dir", str(outdir), "--model_type", str(model_type),
                "--nb_members_train", str(args.nb_members_train), "--nb_members_val", str(args.nb_members_val),
                "--nb_members_test", str(args.nb_members_test), "--seed", str(args.seed),
                "--latent_dim", str(latent_dim), "--duree_lissage", str(args.duree_lissage),
                "--bs", str(args.bs), "--loss_type", str(args.loss_type),
                
                # --- TRANSFERT DES ARGUMENTS TUNABLES ---
                "--dr", str(args.dr),
                "--embed_dim", str(args.embed_dim),
                "--depth", str(args.depth),
                "--num_heads", str(args.num_heads),
                "--mlp_ratio", str(args.mlp_ratio),
                "--transformer_act", str(args.transformer_act),
                "--pool_strategy", str(args.pool_strategy),
                "--head_hidden_dim", str(args.head_hidden_dim),
                "--head_act", str(args.head_act),
                "--patch_size_sst", str(args.patch_size_sst[0]), str(args.patch_size_sst[1]),
                "--patch_size_slp", str(args.patch_size_slp[0]), str(args.patch_size_slp[1])
            ]
            if args.norm_first: eval_spatial_command.append("--norm_first")

            if args.embed_path:
                eval_spatial_command.extend(["--embed_path", str(args.embed_path)])
            else:
                ext = "joblib" if args.embed_method == 'pca' else "pth"
                eval_spatial_command.extend(["--embed_path", os.path.join(outdir, f"{args.embed_method}_model.{ext}")])

            if args.sst_lags_days is not None: eval_spatial_command.extend(["--sst_lags_days"] + [str(x) for x in args.sst_lags_days])
            if args.slp_lags_days is not None: eval_spatial_command.extend(["--slp_lags_days"] + [str(x) for x in args.slp_lags_days])
            if hasattr(args, 'sst_lags_months') and args.sst_lags_months is not None: eval_spatial_command.extend(["--sst_lags_months"] + [str(x) for x in args.sst_lags_months])
            if hasattr(args, 'slp_lags_months') and args.slp_lags_months is not None: eval_spatial_command.extend(["--slp_lags_months"] + [str(x) for x in args.slp_lags_months])
            if args.winter_months is not None: eval_spatial_command.extend(["--winter_months"] + [str(x) for x in args.winter_months])
            if args.quantiles: eval_spatial_command.extend(["--quantiles"] + [str(x) for x in args.quantiles])

            if args.roll_sst: eval_spatial_command.append("--roll_sst")
            if args.use_lags_attention: eval_spatial_command.append("--use_lags_attention")
            if args.monthly_reduction: eval_spatial_command.append("--monthly_reduction")
            if args.lat_weight: eval_spatial_command.append("--lat_weight")
            if monthly_mean: eval_spatial_command.append("--monthly_mean")

            try:
                result = subprocess.run(eval_spatial_command, check=True, text=True)
                print(f"✅ Évaluation spatiale de {model_type} terminée avec succès !")
            except subprocess.CalledProcessError as e:
                print(f"❌ Erreur lors de l'exécution de l'évaluation spatiale. Code de retour : {e.returncode}")