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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


import sys
from pathlib import Path
import subprocess

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
from tools_cnn.models import CNN_Latent_SLP_Multimodal1, CNN_Latent_SLP_Multimodal0, CNN_Latent_SLP_Multimodal1_tunable


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
    parser.add_argument('--machine', type=str, default='jean-zay-work', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'], help='Machine sur laquelle le code tourne')
    
    parser.add_argument('--nb_members_train', type=int, default=10, help='Nombre de membres à utiliser pour l\'entraînement')
    parser.add_argument('--nb_members_val', type=int, default=5, help='Nombre de membres à utiliser pour la validation')
    parser.add_argument('--nb_members_test', type=int, default=0, help='Nombre de membres à utiliser pour le test') 
    parser.add_argument('--force_val_members', type=str, nargs='*', default=None, help='Forcer une liste spécifique de membres pour la val')
    parser.add_argument('--force_test_members', type=str, nargs='*', default=None, help='Forcer une liste spécifique de membres pour le test')

    parser.add_argument('--seed', type=int, default=42, help='Seed pour le mélange inter membres')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension de l\'espace latent')
    parser.add_argument('--nb_epochs', type=int, default=30, help='Nombre d\'époques pour l\'entraînement du CNN')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours')
    parser.add_argument('--bs', type=int, default=128, help='Taille de batch pour l\'entraînement')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate pour l\'entraînement du CNN')
    
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95], help='Liste des lags pour SST')
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[], help='Liste des lags pour SLP')
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4], help='Liste des lags pour SST')
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[], help='Liste des lags pour SLP')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner')     
    
    parser.add_argument('--beta_kld', type=float, default=1.0, help='Coefficient de la composante KL divergence dans la loss du VAE')
    parser.add_argument('--normalize', action='store_true', help='PCA normalisé ou non')    

    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile','correlation'], default='mse', help='Fonction de coût pour l\'entraînement')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help='Quantiles à prédire')

    parser.add_argument('--roll_sst', action='store_true', help='Appliquer un roll sur les données SST pour centrer l\'océan Atlantique')
    parser.add_argument('--early_fusion_sst', action='store_true', help='Fusionner les lags SST dès les premières couches du CNN')
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Nombre de points de validation intra-époque')

    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données sous-échantillonnées mensuellement (_1mo.nc)')
    parser.add_argument('--lat_weight', action='store_true', help='Applique la pondération spatiale sqrt(cos(lat))')

    parser.add_argument('--dr_conv', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--dr_fc', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--fc_dim', type=int, default=20, help='Dimension de la couche fully connected')
    parser.add_argument('--n_feat', type=int, default=20, help='Nombre de filtres dans la première couche convolutive')
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh'], default='relu', help='Fonction d\'activation')
    parser.add_argument('--depth', type=int, default=3, help='Profondeur du réseau CNN')
    parser.add_argument('--filter_mult', type=float, default=1.0, help='Facteur de multiplication')
    parser.add_argument('--pool_strategy', type=str, choices=['progressive','standart'], default='progressive', help='Stratégie de pooling')
    parser.add_argument('--pool_type', type=str, choices=['avg','max'], default='max', help='Type de pooling')
    parser.add_argument('--sst_kx', type=int, default=3, help='Taille du kernel pour le pooling SST en x')
    parser.add_argument('--sst_ky', type=int, default=5, help='Taille du kernel pour le pooling SST en y')
    parser.add_argument('--sst_pool_x', type=int, default=2, help='Facteur de pooling pour SST en x')
    parser.add_argument('--sst_pool_y', type=int, default=2, help='Facteur de pooling pour SST en y')
    parser.add_argument('--use_gap', action='store_true', help='Utilise Global Average Pooling')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Poids de la régularisation L2')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Écart-type du bruit ajouté aux gradients pour la régularisation')
    parser.add_argument('--gradient_clip', type=float, default=float("inf"), help='Valeur de clipping des gradients pour la régularisation')

    args = parser.parse_args()

    if args.loss_type == 'quantile':
        if 0.5 not in args.quantiles:
            raise ValueError("Erreur: Pour la quantile loss, la liste des quantiles (--quantiles) DOIT inclure la médiane (0.5).")

    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/cnn/cnn_with_slp_embedding/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']:
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/"
    elif args.machine == 'mac_local':
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/cnn/cnn_with_slp_embedding/"

    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val
    nb_members_test = args.nb_members_test
    sst_lags_days = args.sst_lags_days
    slp_lags_days = args.slp_lags_days
    sst_lags_months = args.sst_lags_months
    slp_lags_months = args.slp_lags_months
    bs = args.bs
    lr = args.lr
    latent_dim = args.latent_dim
    nb_epochs = args.nb_epochs
    duree_lissage = args.duree_lissage
    winter_months = args.winter_months

    dr_conv = args.dr_conv
    dr_fc = args.dr_fc
    fc_dim = args.fc_dim
    n_feat = args.n_feat
    activation = args.activation
    depth = args.depth
    filter_mult = args.filter_mult
    pool_strategy = args.pool_strategy
    pool_type = args.pool_type
    sst_kx = args.sst_kx
    sst_ky = args.sst_ky
    sst_pool_x = args.sst_pool_x
    sst_pool_y = args.sst_pool_y
    use_gap = args.use_gap
    weight_decay = args.weight_decay

    patience = 10000
    target_indices = {100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000} if not args.monthly_reduction else {1, 10, 20,30,40,45,50,60,70, 80} 
    
    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    
    # SPLIT MEMBERS INTO TRAIN, VAL, TEST, prise en compte éventuelle des listes forcées
    rng = random.Random(args.seed)
    rng.shuffle(all_members)
    if args.force_val_members is not None or args.force_test_members is not None:
        print("⚠️ OVERRIDE ACTIF : Utilisation des listes de membres forcées.")
        val_members = args.force_val_members if args.force_val_members else []
        test_members = args.force_test_members if args.force_test_members else []
        remaining = [m for m in all_members if m not in val_members and m not in test_members]
        # On coupe selon nb_members_train dans l'ordre de la seed !
        train_members = remaining[:args.nb_members_train]
        nb_members_train = len(train_members)
        nb_members_val = len(val_members)
        nb_members_test = len(test_members)
    else:
        train_members = all_members[:nb_members_train]
        val_members = all_members[-nb_members_val:]
        test_members = all_members[nb_members_train:nb_members_train + nb_members_test] if nb_members_test > 0 else []
    # ---------------------------------

    print(f"val_members: {val_members}, test_members: {test_members}")
    dynamic_slp_std = 596.0 

    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))
            print(f"\n✅ slp_std extrait avec succès du chemin PCA : {dynamic_slp_std}")
        else:
            print(f"\n⚠️ 'slp_std' introuvable dans le nom du dossier. Utilisation du fallback : {dynamic_slp_std}")
    else:
        print(f"\n⚠️ Aucun modèle pré-entraîné fourni. Utilisation du slp_std par défaut : {dynamic_slp_std}")

    loss_tag = args.loss_type
    if args.loss_type == 'quantile':
        loss_tag += "_" + "".join([str(q).replace('.','') for q in args.quantiles])

    test_member_label = "_".join(args.force_test_members) if args.force_test_members else "_"
    if args.embed_method == 'pca':
        if args.monthly_reduction:
            outdir_name = f"Shap_test{test_member_label}_CNN_monthly_loss_{loss_tag}_{args.embed_method}n{args.normalize}{args.latent_dim}_months{''.join(map(str, args.winter_months))}_lat{args.lat_weight}_sst{''.join(map(str, args.sst_lags_months))}slp{''.join(map(str, args.slp_lags_months))}_train{nb_members_train}val{nb_members_val}_{nb_members_test}seed{args.seed}ep{args.nb_epochs}intra{args.nb_intra_evals}bs{args.bs}dr1{args.dr_conv:.3g}dr2{args.dr_fc:.3g}fc{args.fc_dim}fusion{args.early_fusion_sst}lr{args.lr:.3g}feat{args.n_feat}roll{args.roll_sst}act{args.activation}loss{args.loss_type}depth{args.depth}mult{args.filter_mult:.3g}pool{args.pool_type}{args.pool_strategy}{args.sst_pool_x}x{args.sst_pool_y}ker{args.sst_kx}x{args.sst_ky}gap{args.use_gap}decay{args.weight_decay:.3g}slp_std{dynamic_slp_std}"
        else:
            outdir_name = f"Shap_test{test_member_label}_CNN_{duree_lissage}d_loss_{loss_tag}_{args.embed_method}n{args.normalize}{args.latent_dim}_months{''.join(map(str, args.winter_months))}_lat{args.lat_weight}_sst{''.join(map(str, args.sst_lags_days))}slp{''.join(map(str, args.slp_lags_days))}_train{nb_members_train}val{nb_members_val}_{nb_members_test}seed{args.seed}ep{args.nb_epochs}intra{args.nb_intra_evals}bs{args.bs}dr1{args.dr_conv:.3g}dr2{args.dr_fc:.3g}fc{args.fc_dim}fusion{args.early_fusion_sst}lr{args.lr:.3g}feat{args.n_feat}roll{args.roll_sst}act{args.activation}loss{args.loss_type}depth{args.depth}mult{args.filter_mult:.3g}pool{args.pool_type}{args.pool_strategy}{args.sst_pool_x}x{args.sst_pool_y}ker{args.sst_kx}x{args.sst_ky}gap{args.use_gap}decay{args.weight_decay:.3g}slp_std{dynamic_slp_std}"
    elif args.embed_method == 'vae':
        if args.monthly_reduction:
            outdir_name = f"Shap_test{test_member_label}_CNN_monthly_loss_{loss_tag}_{args.embed_method}beta{args.beta_kld}{args.latent_dim}_months{''.join(map(str, args.winter_months))}_lat{args.lat_weight}_sst{''.join(map(str, args.sst_lags_months))}slp{''.join(map(str, args.slp_lags_months))}_train{nb_members_train}val{nb_members_val}_{nb_members_test}seed{args.seed}ep{args.nb_epochs}intra{args.nb_intra_evals}bs{args.bs}dr1{args.dr_conv:.3g}dr2{args.dr_fc:.3g}fc{args.fc_dim}fusion{args.early_fusion_sst}lr{args.lr:.3g}feat{args.n_feat}roll{args.roll_sst}act{args.activation}loss{args.loss_type}depth{args.depth}mult{args.filter_mult:.3g}pool{args.pool_type}{args.pool_strategy}{args.sst_pool_x}x{args.sst_pool_y}ker{args.sst_kx}x{args.sst_ky}gap{args.use_gap}decay{args.weight_decay:.3g}slp_std{dynamic_slp_std}"
        else:
            outdir_name = f"Shap_CNN_{duree_lissage}d_loss_{loss_tag}_{args.embed_method}beta{args.beta_kld}{args.latent_dim}_months{''.join(map(str, args.winter_months))}_lat{args.lat_weight}_sst{''.join(map(str, args.sst_lags_days))}slp{''.join(map(str, args.slp_lags_days))}_train{nb_members_train}val{nb_members_val}_{nb_members_test}seed{args.seed}ep{args.nb_epochs}intra{args.nb_intra_evals}bs{args.bs}dr1{args.dr_conv:.3g}dr2{args.dr_fc:.3g}fc{args.fc_dim}fusion{args.early_fusion_sst}lr{args.lr:.3g}feat{args.n_feat}roll{args.roll_sst}act{args.activation}loss{args.loss_type}depth{args.depth}mult{args.filter_mult:.3g}pool{args.pool_type}{args.pool_strategy}{args.sst_pool_x}x{args.sst_pool_y}ker{args.sst_kx}x{args.sst_ky}gap{args.use_gap}decay{args.weight_decay:.3g}slp_std{dynamic_slp_std}"
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")

    # ============================================================
    # DATALOADERS
    # ============================================================
    intra_workers = min(2, n_workers)

    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)
        training_set = Dataset(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst,slp_std=dynamic_slp_std, augment =True, noise_std = args.noise_std)
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)
        training_set = Dataset_mensuel(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst,slp_std=dynamic_slp_std, augment =True, noise_std = args.noise_std)

    if len(test_members) > 0:
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
    # PRÉPARATION DES POIDS SPATIAUX (POUR DÉCODAGE PCA PONDÉRÉ)
    # ============================================================
    wgts_flat = None
    safe_wgts = None
    area_weights_2d = None  
    if args.lat_weight and args.embed_method == 'pca':
        sample_member = train_members[0]
        sample_path = os.path.join(base_home.replace("stage_isir_jz/cnn/cnn_with_slp_embedding/", ""), f"data/SLP/PSL_anom_LE2-{sample_member}_1mo.nc")
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            h, w = len(lats), len(ds_sample['lon'].values)
            wgts = np.sqrt(coslat).reshape(h, 1)
            wgts_flat = np.broadcast_to(wgts, (h, w)).flatten()
            safe_wgts = np.maximum(wgts_flat, 1e-5)
            # NOUVEAU : Poids d'aire (cos(lat)) au format carte 2D sur GPU
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
            print("Training PCA from scratch on TrainLoader, no weights...")
            pca_model = PCA(n_components=latent_dim, whiten=args.normalize)
            slp_list = []
            for X_sst, X_slp, y_target, y_map, dates, members in trainloader:
                slp_list.append(y_target.view(y_target.size(0), -1).numpy())
            slp_data = np.concatenate(slp_list, axis=0)
            pca_model.fit(slp_data)
            joblib.dump(pca_model, os.path.join(outdir, "pca_model.joblib"))
            print("PCA training done and saved.")

    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=latent_dim).to(device)
        if args.embed_path and os.path.exists(args.embed_path):
            print(f"Loading VAE from {args.embed_path}")
            vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        else:
            print("Training VAE from scratch on TrainLoader (10 epochs), no weights...")
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
            print("VAE training done and saved.")
        
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
        print(f"-> Variance expliquée par entre guillements toutes le composantes PCA : {explained_var * 100:.2f}%")
        
        slp_flat_val = y_target_val.view(y_target_val.size(0), -1).cpu().numpy()
        if wgts_flat is not None:
            slp_flat_val *= wgts_flat
        
        latent_val = pca_model.transform(slp_flat_val)[:, :latent_dim]
        padded_latent = np.zeros((latent_val.shape[0], pca_model.n_components_))
        padded_latent[:, :latent_dim] = latent_val 
        
        recon_flat_val = pca_model.inverse_transform(padded_latent)
        if wgts_flat is not None:
            recon_flat_val /= wgts_flat
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
    # INITIALISATION DU CNN ET DEFINITION DU SUIVI
    # ============================================================
    out_features = latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else latent_dim
    active_sst_lags = sst_lags_months if args.monthly_reduction else sst_lags_days
    active_slp_lags = slp_lags_months if args.monthly_reduction else slp_lags_days

    model = CNN_Latent_SLP_Multimodal1_tunable(
        dr_conv=dr_conv,
        dr_fc=dr_fc,
        fc_dim=fc_dim,
        nb_out=out_features,
        in_chans_sst=len(active_sst_lags),
        in_chans_slp=len(active_slp_lags),
        n_feat=n_feat,
        early_fusion_sst=args.early_fusion_sst,
        depth=depth,
        filter_mult=filter_mult,
        sst_kx=sst_kx,
        sst_ky=sst_ky,
        sst_pool_x=sst_pool_x,
        sst_pool_y=sst_pool_y,
        pool_type=pool_type,
        pool_strategy=pool_strategy,
        activation=activation,
        use_gap=use_gap
    ).to(device)

    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(active_sst_lags), 85, 360).to(device) if len(active_sst_lags) > 0 else None
        dummy_slp = torch.zeros(1, len(active_slp_lags), 53, 113).to(device) if len(active_slp_lags) > 0 else None
        _ = model(dummy_sst, dummy_slp)

    print("Number of CNN parameters : ", sum(p.numel() for p in model.parameters()))
    # Séparation des paramètres pour le Weight Decay
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # On ne régularise pas les biais ni les tenseurs 1D (comme les poids de BatchNorm)
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    # Création de l'optimiseur avec des groupes de paramètres
    optimizer = torch.optim.AdamW([
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ], lr=lr)

    ### SUIVIE GENERALE
    # Initialisation des listes de suivi existantes
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

    # Listes ppour l'Espace Spatial (L1)
    train_map_gL1, val_map_gL1, test_map_gL1 = [], [], []
    train_map_mL1, val_map_mL1, test_map_mL1 = [], [], []

    ### SUIVI INTRA-ÉPOQUE 1

    total_batches = len(trainloader)
    epoch1_batch_losses, epoch1_baseline_losses = [], []
    eval_steps = np.insert(np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int), 0, 0)
    eval_steps_set = set(eval_steps)
    eval_steps_epoch2 = np.insert(np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int), 0, 0)
    eval_steps_epoch2_set = set(eval_steps_epoch2)

    intra_epoch1_steps, intra_epoch1_val_losses, intra_epoch1_test_losses = [], [], []
    # Listes pour l'Espace Latent (Moyenné et Global)
    intra_epoch1_train_lat_mR2, intra_epoch1_val_lat_mR2, intra_epoch1_test_lat_mR2 = [], [], []
    intra_epoch1_train_lat_gR2, intra_epoch1_val_lat_gR2, intra_epoch1_test_lat_gR2 = [], [], []
    intra_epoch1_train_lat_mCorr, intra_epoch1_val_lat_mCorr, intra_epoch1_test_lat_mCorr = [], [], []
    intra_epoch1_train_lat_gCorr, intra_epoch1_val_lat_gCorr, intra_epoch1_test_lat_gCorr = [], [], []
    intra_epoch1_train_lat_mk, intra_epoch1_val_lat_mk, intra_epoch1_test_lat_mk = [], [], [] 
    intra_epoch1_train_lat_gk, intra_epoch1_val_lat_gk, intra_epoch1_test_lat_gk = [], [], []
    intra_epoch1_train_lat_gL1, intra_epoch1_val_lat_gL1, intra_epoch1_test_lat_gL1 = [], [], []
    intra_epoch1_train_lat_mL1, intra_epoch1_val_lat_mL1, intra_epoch1_test_lat_mL1 = [], [], []
    # Listes pour l'Espace Spatial (R2)
    intra_epoch1_train_map_gR2, intra_epoch1_val_map_gR2, intra_epoch1_test_map_gR2 = [], [], []
    intra_epoch1_train_map_mR2, intra_epoch1_val_map_mR2, intra_epoch1_test_map_mR2 = [], [], [] 
    # Listes pour l'Espace Spatial (Corrélations)
    intra_epoch1_train_map_sCorr, intra_epoch1_val_map_sCorr, intra_epoch1_test_map_sCorr = [], [], []
    intra_epoch1_train_map_tCorr, intra_epoch1_val_map_tCorr, intra_epoch1_test_map_tCorr = [], [], []
    intra_epoch1_train_map_gCorr, intra_epoch1_val_map_gCorr, intra_epoch1_test_map_gCorr = [], [], []
    # Listes pour l'Espace Spatial (L1)
    intra_epoch1_train_map_gL1, intra_epoch1_val_map_gL1, intra_epoch1_test_map_gL1 = [], [], []
    intra_epoch1_train_map_mL1, intra_epoch1_val_map_mL1, intra_epoch1_test_map_mL1 = [], [], []

    ### SUIVI INTRA-ÉPOQUE 2
    intra_epoch2_steps, intra_epoch2_val_losses, intra_epoch2_test_losses = [], [], []
    # Listes pour l'Espace Latent (Moyenné et Global)
    intra_epoch2_train_lat_mR2, intra_epoch2_val_lat_mR2, intra_epoch2_test_lat_mR2 = [], [], []
    intra_epoch2_train_lat_gR2, intra_epoch2_val_lat_gR2, intra_epoch2_test_lat_gR2 = [], [], []
    intra_epoch2_train_lat_mCorr, intra_epoch2_val_lat_mCorr, intra_epoch2_test_lat_mCorr = [], [], []
    intra_epoch2_train_lat_gCorr, intra_epoch2_val_lat_gCorr, intra_epoch2_test_lat_gCorr = [], [], []
    intra_epoch2_train_lat_mk, intra_epoch2_val_lat_mk, intra_epoch2_test_lat_mk = [], [], [] 
    intra_epoch2_train_lat_gk, intra_epoch2_val_lat_gk, intra_epoch2_test_lat_gk = [], [], []
    intra_epoch2_train_lat_gL1, intra_epoch2_val_lat_gL1, intra_epoch2_test_lat_gL1 = [], [], []
    intra_epoch2_train_lat_mL1, intra_epoch2_val_lat_mL1, intra_epoch2_test_lat_mL1 = [], [], [] 
    # Listes pour l'Espace Spatial (R2)
    intra_epoch2_train_map_gR2, intra_epoch2_val_map_gR2, intra_epoch2_test_map_gR2 = [], [], []
    intra_epoch2_train_map_mR2, intra_epoch2_val_map_mR2, intra_epoch2_test_map_mR2 = [], [], [] 
    # Listes pour l'Espace Spatial (Corrélations)
    intra_epoch2_train_map_sCorr, intra_epoch2_val_map_sCorr, intra_epoch2_test_map_sCorr = [], [], []
    intra_epoch2_train_map_tCorr, intra_epoch2_val_map_tCorr, intra_epoch2_test_map_tCorr = [], [], []
    intra_epoch2_train_map_gCorr, intra_epoch2_val_map_gCorr, intra_epoch2_test_map_gCorr = [], [], []
    # Listes pour l'Espace Spatial (L1)
    intra_epoch2_train_map_gL1, intra_epoch2_val_map_gL1, intra_epoch2_test_map_gL1 = [], [], []
    intra_epoch2_train_map_mL1, intra_epoch2_val_map_mL1, intra_epoch2_test_map_mL1 = [], [], [] 



    if args.update == 1:
        initial_params = torch.load(f"{outdir}/final_model_CNN.pth")
        model.load_state_dict(initial_params['state_dict'])
        optimizer.load_state_dict(initial_params['optimizer'])
        train_losses = initial_params['train_losses']
        val_losses = initial_params['val_losses']
        test_corrs = initial_params.get('test_corrs', [])
        best_val_loss = np.min(val_losses)
        print("Model state updated")
    else:
        print("Initiated first CNN training")

    

    # ============================================================
    # TRAINING & EVALUATION LOOP (CNN)
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

        epoch_grad_norms = []
        
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
            loss_value = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')

            loss_value.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
            epoch_grad_norms.append(grad_norm.item())
            optimizer.step()
            running_train_loss += loss_value.item() * X_sst.size(0)
            total_train_samples += X_sst.size(0)

            # Accumulation pour corrélations/R2 dans l'espace latent
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
                    eval_phases = ['val', 'test'] if len(test_members) > 0 else ['val']
                    
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

                                # Décodage en ligne et accumulation spatiale intra-époque
                                decoded_v_pred = decode_latent_to_map(v_pred, args, latent_dim, pca_model, vae_model, safe_wgts)
                                intra_map_tracker.update(v_y_target.detach(), decoded_v_pred.detach())

                        # --- 1. Récupération Latente ---
                        lat_gR2, lat_mR2, lat_gCorr, lat_mCorr, lat_gK, lat_mK, lat_gL1, lat_mL1 = intra_latent_tracker.compute()
                        # --- 2. Récupération Spatiale ---
                        map_gR2, map_mR2, map_r2_np, map_sCorr, map_tCorr, map_gCorr, map_corr_np, map_gL1, map_mL1, map_l1_np = intra_map_tracker.compute(area_weights=area_weights_2d)
                        
        
                        # Sauvegarde de la carte de R^2 pixel intra
                        prefix_l2 = f"L2_intra_{key}_ep{epoch+1}_step{batch_idx}"
                        prefix_l1 = f"L1_intra_{key}_ep{epoch+1}_step{batch_idx}"
                        save_r2_pixel_map_and_plot(map_r2_np, outdir, prefix_l2, metric_type = "l2")
                        save_r2_pixel_map_and_plot(map_l1_np, outdir, prefix_l1, metric_type = "l1")
                        save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_intra_{key}_ep{epoch+1}_step{batch_idx}", metric_type = "corr")

                        if epoch == 0:
                            if key == 'val':
                                intra_epoch1_steps.append(batch_idx) 
                                intra_epoch1_val_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch1_val_lat_gR2.append(lat_gR2)
                                intra_epoch1_val_lat_mR2.append(lat_mR2)
                                intra_epoch1_val_lat_gCorr.append(lat_gCorr)
                                intra_epoch1_val_lat_mCorr.append(lat_mCorr)
                                intra_epoch1_val_lat_mk.append(lat_mK)
                                intra_epoch1_val_lat_gk.append(lat_gK)
                                intra_epoch1_val_lat_gL1.append(lat_gL1)
                                intra_epoch1_val_lat_mL1.append(lat_mL1)
                                intra_epoch1_val_map_gR2.append(map_gR2)
                                intra_epoch1_val_map_mR2.append(map_mR2)
                                intra_epoch1_val_map_sCorr.append(map_sCorr)
                                intra_epoch1_val_map_tCorr.append(map_tCorr)
                                intra_epoch1_val_map_gCorr.append(map_gCorr)
                                intra_epoch1_val_map_gL1.append(map_gL1)
                                intra_epoch1_val_map_mL1.append(map_mL1)

                            else:
                                intra_epoch1_test_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch1_test_lat_gR2.append(lat_gR2)
                                intra_epoch1_test_lat_mR2.append(lat_mR2)
                                intra_epoch1_test_lat_gCorr.append(lat_gCorr)
                                intra_epoch1_test_lat_mCorr.append(lat_mCorr)
                                intra_epoch1_test_lat_mk.append(lat_mK)
                                intra_epoch1_test_lat_gk.append(lat_gK)
                                intra_epoch1_test_lat_gL1.append(lat_gL1)
                                intra_epoch1_test_lat_mL1.append(lat_mL1)
                                intra_epoch1_test_map_gR2.append(map_gR2)
                                intra_epoch1_test_map_mR2.append(map_mR2)
                                intra_epoch1_test_map_sCorr.append(map_sCorr)
                                intra_epoch1_test_map_tCorr.append(map_tCorr)
                                intra_epoch1_test_map_gCorr.append(map_gCorr)
                                intra_epoch1_test_map_gL1.append(map_gL1)
                                intra_epoch1_test_map_mL1.append(map_mL1)
                        elif epoch == 1:
                            if key == 'val':
                                intra_epoch2_steps.append(batch_idx)
                                intra_epoch2_val_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch2_val_lat_gR2.append(lat_gR2)
                                intra_epoch2_val_lat_mR2.append(lat_mR2)
                                intra_epoch2_val_lat_gCorr.append(lat_gCorr)
                                intra_epoch2_val_lat_mCorr.append(lat_mCorr)
                                intra_epoch2_val_lat_mk.append(lat_mK)
                                intra_epoch2_val_lat_gk.append(lat_gK)
                                intra_epoch2_val_lat_gL1.append(lat_gL1)
                                intra_epoch2_val_lat_mL1.append(lat_mL1)
                                intra_epoch2_val_map_gR2.append(map_gR2)
                                intra_epoch2_val_map_mR2.append(map_mR2)
                                intra_epoch2_val_map_sCorr.append(map_sCorr)
                                intra_epoch2_val_map_tCorr.append(map_tCorr)
                                intra_epoch2_val_map_gCorr.append(map_gCorr)
                                intra_epoch2_val_map_gL1.append(map_gL1)
                                intra_epoch2_val_map_mL1.append(map_mL1)
                            else:
                                intra_epoch2_test_losses.append(intra_val_loss / intra_n_samples)
                                intra_epoch2_test_lat_gR2.append(lat_gR2)
                                intra_epoch2_test_lat_mR2.append(lat_mR2)
                                intra_epoch2_test_lat_gCorr.append(lat_gCorr)
                                intra_epoch2_test_lat_mCorr.append(lat_mCorr)
                                intra_epoch2_test_lat_mk.append(lat_mK)
                                intra_epoch2_test_lat_gk.append(lat_gK)
                                intra_epoch2_test_lat_gL1.append(lat_gL1)
                                intra_epoch2_test_lat_mL1.append(lat_mL1)
                                intra_epoch2_test_map_gR2.append(map_gR2)
                                intra_epoch2_test_map_mR2.append(map_mR2)
                                intra_epoch2_test_map_sCorr.append(map_sCorr)
                                intra_epoch2_test_map_tCorr.append(map_tCorr)
                                intra_epoch2_test_map_gCorr.append(map_gCorr)
                                intra_epoch2_test_map_gL1.append(map_gL1)
                                intra_epoch2_test_map_mL1.append(map_mL1)

                        current_intra_loss = intra_val_loss / intra_n_samples
                        print(f"-> Intra-{key} Loss: {current_intra_loss:.4f} | Latent gR2: {lat_gR2:.4f} | Map R2: {map_gR2:.4f} | Pixel Mean R2: {map_mR2:.4f}")
                        
                        if key == 'val' and current_intra_loss < best_val_loss:
                            best_val_loss = current_intra_loss
                            best_model_state = copy.deepcopy(model.state_dict())
                            if best_model_path and os.path.exists(best_model_path):
                                os.remove(best_model_path)
                            best_model_path = os.path.join(outdir, f'best_val_CNN_ep{epoch + 1}_step{batch_idx}_loss{best_val_loss:.4f}.pth')
                            torch.save(model.state_dict(), best_model_path)
                            print(f"   *** Nouveau Best Model (Intra) sauvegardé : {os.path.basename(best_model_path)} ***")

                        model.train()

        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)
        gn = np.array(epoch_grad_norms)
        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}, Grad Norms: mean={gn.mean():.4f}, std={gn.std():.4f}, max={gn.max():.4f}')

        lat_gR2, lat_mR2, lat_gCorr, lat_mCorr, lat_gK, lat_mK, lat_gL1, lat_mL1 = train_latent_tracker.compute()
        train_lat_gR2.append(lat_gR2)
        train_lat_mR2.append(lat_mR2)
        train_lat_gCorr.append(lat_gCorr)
        train_lat_mCorr.append(lat_mCorr)
        train_lat_mk.append(lat_mK)
        train_lat_gk.append(lat_gK)
        train_lat_gL1.append(lat_gL1)
        train_lat_mL1.append(lat_mL1)
        map_gR2, map_mR2, map_r2_np, map_sCorr, map_tCorr, map_gCorr, map_corr_np, map_gL1, map_mL1, map_l1_np = train_map_tracker.compute(area_weights=area_weights_2d)
        train_map_gR2.append(map_gR2)
        train_map_mR2.append(map_mR2)
        train_map_sCorr.append(map_sCorr)
        train_map_tCorr.append(map_tCorr)
        train_map_gCorr.append(map_gCorr)
        train_map_gL1.append(map_gL1)
        train_map_mL1.append(map_mL1)
        save_r2_pixel_map_and_plot(map_r2_np, outdir, f"L2_train_ep{epoch+1}", metric_type="l2")
        save_r2_pixel_map_and_plot(map_l1_np, outdir, f"L1_train_ep{epoch+1}", metric_type="l1")
        save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_train_ep{epoch+1}", metric_type="corr")
        print(f"-> Train Map R2: {map_gR2:.4f} | Train Pixel Mean R2: {map_mR2:.4f}")

        # Visualisations première époque / intra
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

                # Plot de l'évaluation latente : corrélation m et g, R^2 m et g
                plot_correlation_evolution([], c_mCorr, outdir, val_ks=c_mk, test_corrs=c_mCorr_test, test_ks=c_mk_test, epoch_1=epoch_1, epoch_2=epoch_2, batch_indexes=c_steps, suffix="_Latent_Mean")
                plot_correlation_evolution([], c_gCorr, outdir, val_ks=c_gK, test_corrs=c_gCorr_test, test_ks=c_gK_test, epoch_1=epoch_1, epoch_2=epoch_2, batch_indexes=c_steps, suffix="_Latent_Global")
                plot_r2_R2_evolution([], c_mCorr, [], c_mR2, outdir, epoch_1=epoch_1, epoch_2=epoch_2, batch_indexes=c_steps, test_R2=c_mR2_test, test_corrs=c_mCorr_test, suffix="_Latent_Mean")
                plot_r2_R2_evolution([], c_gCorr, [], c_gR2, outdir, epoch_1=epoch_1, epoch_2=epoch_2, batch_indexes=c_steps, test_R2=c_gR2_test, test_corrs=c_gCorr_test, suffix="_Latent_Global")
                plot_latent_l1_ss_evolution([], c_gL1, [], c_mL1, outdir, test_g=c_gL1_test, test_m=c_mL1_test, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1)

                
                # Plot de l'évaluation spatiale : corrélation s, t et g, R^2 m et g
                plot_map_r2_evolution([], c_gR2_map, [], c_mR2_map, outdir, test_map=c_gR2_map_test, test_pix=c_mR2_map_test, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1, norm = "l2")
                plot_map_r2_evolution([], c_gL1_map, [], c_mL1_map, outdir, test_map=c_gL1_map_test, test_pix=c_mL1_map_test, is_intra=True, batch_indexes=c_steps, epoch_num=epoch+1, norm = "l1")
                plot_spatial_corr_evolution([], c_sCorr_map, [], c_tCorr_map, [], c_gCorr_map, outdir, test_sc=c_sCorr_map_test, test_tc=c_tCorr_map_test, test_gc=c_gCorr_map_test, is_intra = True, batch_indexes=c_steps, epoch_num=epoch+1)

        # ---------------- VALIDATION & TEST ----------------
        per_member_loss = defaultdict(lambda: {'loss_sum': 0.0, 'count': 0})
        per_member_plots = defaultdict(lambda: {'time': [], 'slp_true': [], 'slp_recon_true': [], 'slp_pred': []})
        eval_phases = ['val', 'test'] if len(test_members) > 0 else ['val']
        
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

                    # --- NOUVEAU : Décodage en ligne et mise à jour du tracker spatial Val/Test ---
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
            
            # --- 1. Récupération Latente ---
            lat_gR2, lat_mR2, lat_gCorr, lat_mCorr, lat_gK, lat_mK, lat_gL1, lat_mL1 = eval_latent_tracker.compute()
            
            # --- 2. Récupération Spatiale ---
            map_gR2, map_mR2, map_r2_np, map_sCorr, map_tCorr, map_gCorr, map_corr_np, map_gL1, map_mL1, map_l1_np = eval_map_tracker.compute(area_weights=area_weights_2d)
            
            if key == 'val':
                val_map_gR2.append(map_gR2)
                val_map_mR2.append(map_mR2)
                val_map_sCorr.append(map_sCorr)
                val_map_tCorr.append(map_tCorr)
                val_map_gCorr.append(map_gCorr)
                val_map_gL1.append(map_gL1)
                val_map_mL1.append(map_mL1)
                val_lat_gR2.append(lat_gR2)
                val_lat_mR2.append(lat_mR2)
                val_lat_gCorr.append(lat_gCorr)
                val_lat_mCorr.append(lat_mCorr)
                val_lat_mk.append(lat_mK)
                val_lat_gk.append(lat_gK)
                val_lat_gL1.append(lat_gL1)
                val_lat_mL1.append(lat_mL1)
                save_r2_pixel_map_and_plot(map_r2_np, outdir, f"L2_val_ep{epoch+1}", metric_type = "l2")
                save_r2_pixel_map_and_plot(map_l1_np, outdir, f"L1_val_ep{epoch+1}", metric_type = "l1")
                save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_val_ep{epoch+1}", metric_type = "corr")

            else:
                test_map_gR2.append(map_gR2)
                test_map_mR2.append(map_mR2)
                test_map_sCorr.append(map_sCorr)
                test_map_tCorr.append(map_tCorr)
                test_map_gCorr.append(map_gCorr)
                test_map_gL1.append(map_gL1)
                test_map_mL1.append(map_mL1)
                test_lat_gR2.append(lat_gR2)
                test_lat_mR2.append(lat_mR2)
                test_lat_gCorr.append(lat_gCorr)
                test_lat_mCorr.append(lat_mCorr)
                test_lat_mk.append(lat_mK)
                test_lat_gk.append(lat_gK)
                test_lat_gL1.append(lat_gL1)
                test_lat_mL1.append(lat_mL1)
                save_r2_pixel_map_and_plot(map_r2_np, outdir, f"L2_test_ep{epoch+1}", metric_type = "l2")
                save_r2_pixel_map_and_plot(map_l1_np, outdir, f"L1_test_ep{epoch+1}", metric_type = "l1")
                save_r2_pixel_map_and_plot(map_corr_np, outdir, f"Corr_test_ep{epoch+1}", metric_type = "corr")
            print(f'Epoch {epoch + 1} {key} Loss: {val_loss:.6f} | {key} Map global R2: {map_gR2:.4f} | {key} Pixel Mean R2: {map_mR2:.4f} | {key} Latent global R2: {lat_gR2:.4f} | {key} Latent Mean R2: {lat_mR2:.4f} | {key} Map global Corr: {map_gCorr:.4f} | {key} Map Mean Corr: {map_tCorr:.4f} | {key} Latent global Corr: {lat_gCorr:.4f} | {key} Latent Mean Corr: {lat_mCorr:.4f} | {key} Map global L1: {map_gL1:.4f} | {key} Map Mean L1: {map_mL1:.4f} | {key} Latent global L1: {lat_gL1:.4f} | {key} Latent Mean L1: {lat_mL1:.4f}')

            # ---------------- EARLY STOPPING & SAVING ----------------
            if key == 'val':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    if best_model_path and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    best_model_path = os.path.join(outdir, f'best_val_CNN_ep{epoch + 1}_end_loss{best_val_loss:.4f}.pth')
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
        
        # ---------------- AFFICHAGE ET SAUVEGARDE ----------------
        if (epoch + 1) % 2 == 0:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses,
            }
            torch.save(state, f'{outdir}/final_model_CNN.pth')
            loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times, per_member_val_losses=val_losses_per_member_history, test_losses=test_losses)
                        
            # Plots d'évaluation latente 
            plot_r2_R2_evolution(train_lat_mCorr, val_lat_mCorr, train_lat_mR2, val_lat_mR2, outdir, test_R2=test_lat_mR2, test_corrs=test_lat_mCorr, suffix="_Latent_Mean")
            plot_r2_R2_evolution(train_lat_gCorr, val_lat_gCorr, train_lat_gR2, val_lat_gR2, outdir, test_R2=test_lat_gR2, test_corrs=test_lat_gCorr, suffix="_Latent_Global")
            plot_correlation_evolution(train_lat_mCorr, val_lat_mCorr, outdir, test_corrs=test_lat_mCorr, train_ks=train_lat_mk, val_ks=val_lat_mk, test_ks=test_lat_mk, suffix="_Latent_Mean")
            plot_correlation_evolution(train_lat_gCorr, val_lat_gCorr, outdir, test_corrs=test_lat_gCorr, train_ks=train_lat_gk, val_ks=val_lat_gk, test_ks=test_lat_gk, suffix="_Latent_Global")
            plot_latent_l1_ss_evolution(train_lat_gL1, val_lat_gL1, train_lat_mL1, val_lat_mL1, outdir, test_g=test_lat_gL1 if nb_members_test > 0 else None, test_m=test_lat_mL1 if nb_members_test > 0 else None)
            
            # Plots d'évaluation spatiale 
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
                    time_list=d['time'],
                    outdir=member_outdir,
                    epoch=(epoch + 1)
                )
        
    print(f"Best Val Loss : {best_val_loss:.6f}")

    # Sauvegarde et plots finaux
    loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times, per_member_val_losses=val_losses_per_member_history, test_losses=test_losses)

    # Plots d'évaluation latente 
    plot_r2_R2_evolution(train_lat_mCorr, val_lat_mCorr, train_lat_mR2, val_lat_mR2, outdir, test_R2=test_lat_mR2, test_corrs=test_lat_mCorr, suffix="_Latent_Mean")
    plot_r2_R2_evolution(train_lat_gCorr, val_lat_gCorr, train_lat_gR2, val_lat_gR2, outdir, test_R2=test_lat_gR2, test_corrs=test_lat_gCorr, suffix="_Latent_Global")
    plot_correlation_evolution(train_lat_mCorr, val_lat_mCorr, outdir, test_corrs=test_lat_mCorr, train_ks=train_lat_mk, val_ks=val_lat_mk, test_ks=test_lat_mk, suffix="_Latent_Mean")
    plot_correlation_evolution(train_lat_gCorr, val_lat_gCorr, outdir, test_corrs=test_lat_gCorr, train_ks=train_lat_gk, val_ks=val_lat_gk, test_ks=test_lat_gk, suffix="_Latent_Global")
    plot_latent_l1_ss_evolution(train_lat_gL1, val_lat_gL1, train_lat_mL1, val_lat_mL1, outdir, test_g=test_lat_gL1 if nb_members_test > 0 else None, test_m=test_lat_mL1 if nb_members_test > 0 else None)

    # Plots d'évaluation spatiale
    plot_map_r2_evolution(train_map_gR2, val_map_gR2, train_map_mR2, val_map_mR2, outdir, test_map=test_map_gR2 if nb_members_test > 0 else None, test_pix=test_map_mR2 if nb_members_test > 0 else None, norm="l2")
    plot_map_r2_evolution(train_map_gL1, val_map_gL1, train_map_mL1, val_map_mL1, outdir, test_map=test_map_gL1 if nb_members_test > 0 else None, test_pix=test_map_mL1 if nb_members_test > 0 else None, norm="l1")
    plot_spatial_corr_evolution(train_map_sCorr, val_map_sCorr, train_map_tCorr, val_map_tCorr, train_map_gCorr, val_map_gCorr, outdir, test_sc=test_map_sCorr, test_tc=test_map_tCorr, test_gc=test_map_gCorr)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses
    }
    torch.save(state, f'{outdir}/final_model_CNN.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_CNN.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training complete, elapsed time: {elapsed_time / 60:.2f} minutes")

    
    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation automatique...")
    print("="*50)

    eval_script_path = os.path.join(os.path.dirname(__file__), "eval_cnn_embedding.py")
    for model_type in ["best"]:
        for monthly_mean in [True]:
            print(f"\n--- Évaluation du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            eval_command = [
                sys.executable, eval_script_path,
                "--machine", str(args.machine), "--embed_method", str(args.embed_method),
                "--cnn_dir", str(outdir), "--model_type", str(model_type),
                "--nb_members_train", str(args.nb_members_train), "--nb_members_val", str(args.nb_members_val),
                "--nb_members_test", str(args.nb_members_test), "--seed", str(args.seed),
                "--latent_dim", str(latent_dim), "--duree_lissage", str(args.duree_lissage),
                "--bs", str(args.bs), "--loss_type", str(args.loss_type),
                "--dr_conv", str(args.dr_conv), "--dr_fc", str(args.dr_fc),
                "--fc_dim", str(args.fc_dim), "--n_feat", str(args.n_feat),
                "--depth", str(args.depth), "--filter_mult", str(args.filter_mult),
                "--sst_kx", str(args.sst_kx), "--sst_ky", str(args.sst_ky),
                "--sst_pool_x", str(args.sst_pool_x), "--sst_pool_y", str(args.sst_pool_y),
                "--pool_type", str(args.pool_type), "--pool_strategy", str(args.pool_strategy),
                "--activation", str(args.activation)
            ]
            if args.embed_path: eval_command.extend(["--embed_path", str(args.embed_path)])
            if args.sst_lags_days is not None: eval_command.extend(["--sst_lags_days"] + [str(x) for x in args.sst_lags_days])
            if args.sst_lags_months is not None: eval_command.extend(["--sst_lags_months"] + [str(x) for x in args.sst_lags_months])
            if args.slp_lags_days is not None: eval_command.extend(["--slp_lags_days"] + [str(x) for x in args.slp_lags_days])
            if args.slp_lags_months is not None: eval_command.extend(["--slp_lags_months"] + [str(x) for x in args.slp_lags_months])
            if args.winter_months is not None: eval_command.extend(["--winter_months"] + [str(x) for x in args.winter_months])
            if args.quantiles: eval_command.extend(["--quantiles"] + [str(x) for x in args.quantiles])

            if args.force_val_members is not None: eval_command.extend(["--force_val_members"] + args.force_val_members)
            if args.force_test_members is not None: eval_command.extend(["--force_test_members"] + args.force_test_members)

            if args.roll_sst: eval_command.append("--roll_sst")
            if args.early_fusion_sst: eval_command.append("--early_fusion_sst")
            if monthly_mean: eval_command.append("--monthly_mean")
            if args.monthly_reduction: eval_command.append("--monthly_reduction")
            if args.lat_weight: eval_command.append("--lat_weight")
            if args.use_gap: eval_command.append("--use_gap")

            try:
                result = subprocess.run(eval_command, check=True, text=True)
                print(f"\n✅ Évaluation de {model_type} terminée avec succès !")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Erreur lors de l'exécution de l'évaluation. Code de retour : {e.returncode}")

    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation SPATIALE automatique...")
    print("="*50)

    eval_spatial_script_path = os.path.join(os.path.dirname(__file__), "eval_cnn_spatial.py")
    for model_type in ["best"]:
        for monthly_mean in [True]:
            print(f"\n--- Évaluation spatiale du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            eval_spatial_command = [
                sys.executable, eval_spatial_script_path,
                "--machine", str(args.machine), "--embed_method", str(args.embed_method),
                "--cnn_dir", str(outdir), "--model_type", str(model_type),
                "--nb_members_train", str(args.nb_members_train), "--nb_members_val", str(args.nb_members_val),
                "--nb_members_test", str(args.nb_members_test), "--seed", str(args.seed),
                "--latent_dim", str(latent_dim), "--duree_lissage", str(args.duree_lissage),
                "--bs", str(args.bs), "--loss_type", str(args.loss_type),
                "--dr_conv", str(args.dr_conv), "--dr_fc", str(args.dr_fc),
                "--fc_dim", str(args.fc_dim), "--n_feat", str(args.n_feat),
                "--depth", str(args.depth), "--filter_mult", str(args.filter_mult),
                "--sst_kx", str(args.sst_kx), "--sst_ky", str(args.sst_ky),
                "--sst_pool_x", str(args.sst_pool_x), "--sst_pool_y", str(args.sst_pool_y),
                "--pool_type", str(args.pool_type), "--pool_strategy", str(args.pool_strategy),
                "--activation", str(args.activation)
            ]
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


            if args.force_val_members is not None: eval_spatial_command.extend(["--force_val_members"] + args.force_val_members)
            if args.force_test_members is not None: eval_spatial_command.extend(["--force_test_members"] + args.force_test_members)

            if args.roll_sst: eval_spatial_command.append("--roll_sst")
            if args.early_fusion_sst: eval_spatial_command.append("--early_fusion_sst")
            if args.monthly_reduction: eval_spatial_command.append("--monthly_reduction")
            if args.lat_weight: eval_spatial_command.append("--lat_weight")
            if monthly_mean: eval_spatial_command.append("--monthly_mean")
            if args.use_gap: eval_spatial_command.append("--use_gap")

            try:
                result = subprocess.run(eval_spatial_command, check=True, text=True)
                print(f"\n✅ Évaluation spatiale de {model_type} terminée avec succès !")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Erreur lors de l'exécution de l'évaluation spatiale. Code de retour : {e.returncode}")

    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation SHAP automatique...")
    print("="*50)

    eval_shap_script_path = os.path.join(os.path.dirname(__file__), "eval_cnn_shap.py")
    
    # On évalue SHAP uniquement sur le "best" model 
    for model_type in ["best"]:  
        print(f"\n--- Évaluation SHAP du modèle : {model_type} ---")
        eval_shap_command = [
            sys.executable, eval_shap_script_path,
            "--cnn_dir", str(outdir), "--model_type", str(model_type), "--machine", str(args.machine), '--embed_method', str(args.embed_method),"--top_k_components", str(min(args.latent_dim,5)),
            "--nb_members_train", str(args.nb_members_train), "--nb_members_val", str(args.nb_members_val),
            "--nb_members_test", str(args.nb_members_test), "--seed", str(args.seed),
            "--latent_dim", str(latent_dim), "--duree_lissage", str(args.duree_lissage),"--loss_type", str(args.loss_type),
            "--dr_conv", str(args.dr_conv), "--dr_fc", str(args.dr_fc),
            "--fc_dim", str(args.fc_dim), "--n_feat", str(args.n_feat),
            "--depth", str(args.depth), "--filter_mult", str(args.filter_mult),
            "--sst_kx", str(args.sst_kx), "--sst_ky", str(args.sst_ky),
            "--sst_pool_x", str(args.sst_pool_x), "--sst_pool_y", str(args.sst_pool_y),
            "--pool_type", str(args.pool_type), "--pool_strategy", str(args.pool_strategy),
            "--activation", str(args.activation),
            "--method", "gradient", "--bg_type", "zeros"
        ]

        if args.embed_path: eval_shap_command.extend(["--embed_path", str(args.embed_path)])
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
        if args.early_fusion_sst: eval_shap_command.append("--early_fusion_sst")
        if args.monthly_reduction: eval_shap_command.append("--monthly_reduction")
        if args.lat_weight: eval_shap_command.append("--lat_weight")
        if args.use_gap: eval_shap_command.append("--use_gap")

        try:
            subprocess.run(eval_shap_command, check=True, text=True)
            print(f"\n✅ Évaluation SHAP de {model_type} terminée avec succès !")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Erreur lors de l'exécution de l'évaluation SHAP. Code de retour : {e.returncode}")
