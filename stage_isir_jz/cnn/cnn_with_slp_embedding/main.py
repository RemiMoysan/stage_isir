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

# import des dossiers siblings
import sys
from pathlib import Path
import subprocess

project_root = Path(__file__).resolve().parent.parent.parent

# Ajouter le dossier "tools" de vision_transformer au sys.path pour tes imports de modèles
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.visualizations import loss_figure, loss_first_epoch, plot_and_save_maps_with_reconstruction_light, plot_reconstruction_check, plot_correlation_evolution, plot_r2_R2_evolution
from tools.datasets import Dataset, Dataset_mensuel
from tools.models import ConvVAE, vae_loss, compute_loss, get_median_prediction
from tools_cnn.models import CNN_Latent_SLP_Multimodal1, CNN_Latent_SLP_Multimodal0


# Eventuellement normaliser les embeddings pour interpréter la MSE, mais en soit cette variance a un sens physique. 

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
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension de l\'espace latent')
    parser.add_argument('--nb_epochs', type=int, default=30, help='Nombre d\'époques pour l\'entraînement du CNN')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours')
    parser.add_argument('--bs', type=int, default=128, help='Taille de batch pour l\'entraînement')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate pour l\'entraînement du CNN')
    parser.add_argument('--dr', type=float, default=0.2, help='Dropout rate pour le CNN')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95], help='Liste des lags pour SST')
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[], help='Liste des lags pour SLP')
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4], help='Liste des lags pour SST')
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[], help='Liste des lags pour SLP')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner')    
    
    parser.add_argument('--beta_kld', type=float, default=1.0, help='Coefficient de la composante KL divergence dans la loss du VAE (si utilisé)')
    parser.add_argument('--normalize', action='store_true', help='PCA normalisé ou non')    

# --- NOUVEAUX ARGUMENTS DE LOSS ---
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile','correlation'], default='mse', help='Fonction de coût pour l\'entraînement')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help='Quantiles à prédire (la médiane 0.5 est obligatoire pour la reconstruction)')

    parser.add_argument('--roll_sst', action='store_true', help='Appliquer un roll sur les données SST pour centrer l\'océan Atlantique')
    parser.add_argument('--early_fusion_sst', action='store_true', help='Fusionner les lags SST dès les premières couches du CNN (au lieu de fusion tardive)')
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Nombre de points de validation intra-époque (espacement logarithmique epoch 1, espacement liénaire epoch 2)')

    # --- NOUVEAUX ARGUMENTS ---
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données sous-échantillonnées mensuellement (_1mo.nc)')
    parser.add_argument('--lat_weight', action='store_true', help='Applique la pondération spatiale sqrt(cos(lat))')
    args = parser.parse_args()

    # Vérification stricte des quantiles
    if args.loss_type == 'quantile':
        if 0.5 not in args.quantiles:
            raise ValueError("Erreur: Pour la quantile loss, la liste des quantiles (--quantiles) DOIT inclure la médiane (0.5) pour permettre les reconstructions.")

    # Routage dynamique des dossiers
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/cnn/cnn_with_slp_embedding/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']:
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/"
    elif args.machine == 'mac_local':
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/cnn/cnn_with_slp_embedding/"


    # ============================================================
    # GLOBAL CONSTANTS
    # ============================================================

    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val
    nb_members_test = args.nb_members_test
    sst_lags_days = args.sst_lags_days
    slp_lags_days = args.slp_lags_days
    sst_lags_months = args.sst_lags_months
    slp_lags_months = args.slp_lags_months
    bs = args.bs
    lr = args.lr
    dr = args.dr
    latent_dim = args.latent_dim
    nb_epochs = args.nb_epochs
    duree_lissage = args.duree_lissage
    winter_months = args.winter_months

    print("Arg Parameters:")
    print(f"  Latent Dim: {latent_dim}", f" SST Lags: {sst_lags_days}", f" SLP Lags: {slp_lags_days}", f" Batch Size: {bs}", f" Learning Rate: {lr}", f" Dropout Rate: {dr}", f" Winter Months: {winter_months}", f" Smoothing Duration: {duree_lissage}", f" Number of Epochs: {nb_epochs}", f" Number of Training Members: {nb_members_train}", f" Number of Validation Members: {nb_members_val}", f" Number of Test Members: {nb_members_test}\n")

    patience = 10000
    target_indices = {100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000} if not args.monthly_reduction else {10, 100, 200,300,400,450,500,600,700, 800} 
    

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]

    # La validation reste exactement comme dans tes anciens scripts (les derniers de la liste)
    val_members = all_members[-nb_members_val:]

    # Le test prend les membres disponibles juste après l'entraînement
    test_members = all_members[nb_members_train:nb_members_train + nb_members_test] if nb_members_test > 0 else []

    # Nom du dossier adapté avec la loss
    loss_tag = args.loss_type
    if args.loss_type == 'quantile':
        loss_tag += "_" + "".join([str(q).replace('.','') for q in args.quantiles])

    if args.embed_method == 'pca':
        if not args.monthly_reduction:
            outdir_name = f"CNN_Latent_early_fusion_sst_{args.early_fusion_sst}_{args.embed_method}_normalize_{args.normalize}_lat_weight_{args.lat_weight}_loss_{loss_tag}_embedding_{latent_dim}_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_{nb_members_test}_members_seed_{args.seed}_{duree_lissage}d_roll_sst_{args.roll_sst}"
        else:
            outdir_name = f"CNN_Latent_early_fusion_sst_{args.early_fusion_sst}_{args.embed_method}_normalize_{args.normalize}_lat_weight_{args.lat_weight}_loss_{loss_tag}_embedding_{latent_dim}_lags_{'_'.join(map(str, sst_lags_months))}_sst_{'_'.join(map(str, slp_lags_months))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_{nb_members_test}_members_seed_{args.seed}_monthly_roll_sst_{args.roll_sst}"
    elif args.embed_method == 'vae':
        # éventuellement ajouter le lat weight ici mais n'a pas beaucoup d'intérêt je pense 
        if not args.monthly_reduction:
            outdir_name = f"CNN_Latent_early_fusion_sst_{args.early_fusion_sst}_{args.embed_method}_beta_{args.beta_kld}_loss_{loss_tag}_embedding_{latent_dim}_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_{nb_members_test}_members_seed_{args.seed}_{duree_lissage}d_roll_sst_{args.roll_sst}"
        else:
            outdir_name = f"CNN_Latent_early_fusion_sst_{args.early_fusion_sst}_{args.embed_method}_beta_{args.beta_kld}_loss_{loss_tag}_embedding_{latent_dim}_lags_{'_'.join(map(str, sst_lags_months))}_sst_{'_'.join(map(str, slp_lags_months))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_{nb_members_test}_members_seed_{args.seed}_monthly_roll_sst_{args.roll_sst}"
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")

    # On réduit la target à partir de la std qui est dans le nom
    # à voir si c'est compatible avec le VAE, mais pour l'instant on va juste faire ça pour le PCA.
    # Intérêt : la MSE baseline sur la reconstruction est à peu près 1. 
    # Pour la sst, on garde la normalisation par 0.707 (valeur par défaut du dataset) pour simplifier (donc pas exactement normalisé)

    dynamic_slp_std = 596.0  # Valeur de repli (fallback) par sécurité

    if args.embed_path:
        # On cherche le motif "slp_std" suivi de chiffres et d'un point
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))
            print(f"\n✅ slp_std extrait avec succès du chemin PCA : {dynamic_slp_std}")
        else:
            print(f"\n⚠️ 'slp_std' introuvable dans le nom du dossier. Utilisation du fallback : {dynamic_slp_std}")
    else:
        print(f"\n⚠️ Aucun modèle pré-entraîné fourni. Utilisation du slp_std par défaut : {dynamic_slp_std}")

    # ============================================================
    # DATALOADERS
    # ============================================================
    intra_workers = min(2, n_workers) # éventuellement diminuer encore plus le nombre de workers pour le valloader intra-époque si la RAM est limitée / si plus de lags en input

    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)
        training_set = Dataset(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)
        
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)
        training_set = Dataset_mensuel(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)

    
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
    # PRÉPARATION DES POIDS SPATIAUX (POUR DÉCODAGE PCA PONDÉRÉ) : à mettre avant le sanity check de l'embedder non??
    # ============================================================
    wgts_flat = None
    if args.lat_weight and args.embed_method == 'pca':
        # On lit un fichier SLP n'importe lequel juste pour extraire la grille de latitude
        sample_member = train_members[0]
        sample_path = os.path.join(base_home.replace("stage_isir_jz/cnn/cnn_with_slp_embedding/", ""), f"data/SLP/PSL_anom_LE2-{sample_member}_1mo.nc")
        
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            h, w = len(lats), len(ds_sample['lon'].values)
            wgts = np.sqrt(coslat).reshape(h, 1)
            wgts_flat = np.broadcast_to(wgts, (h, w)).flatten()
            safe_wgts = np.maximum(wgts_flat, 1e-5) # Pour éviter division par zéro
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
            pca_model = PCA(n_components=latent_dim, whiten = args.normalize)
            slp_list = []
            # Unpacking des 6 variables !
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
                # Unpacking des 6 variables !
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
    # 4.5 SANITY CHECK DE L'EMBEDDER (PCA ou VAE)
    # ============================================================
    print("\n--- VÉRIFICATION DE LA QUALITÉ DE RECONSTRUCTION ---")
    X_sst_val, X_slp_val, y_target_val, y_map_val, dates_val, members_val = next(iter(valloader))

    if args.embed_method == 'pca':
        explained_var = np.sum(pca_model.explained_variance_ratio_)
        print(f"-> Variance expliquée par les {latent_dim} composantes PCA : {explained_var * 100:.2f}%")
        
        slp_flat_val = y_target_val.view(y_target_val.size(0), -1).cpu().numpy()
        if wgts_flat is not None:
            slp_flat_val *= wgts_flat  # Appliquer la pondération spatiale si nécessaire
        
        # Tronquature pour s'adapter à la dimension latente ciblée
        latent_val = pca_model.transform(slp_flat_val)[:, :latent_dim]
        padded_latent = np.zeros((latent_val.shape[0], pca_model.n_components_))
        padded_latent[:, :latent_dim] = latent_val 
        
        recon_flat_val = pca_model.inverse_transform(padded_latent)
        if wgts_flat is not None:
            recon_flat_val /= wgts_flat  # Re-appliquer la pondération inverse pour la reconstruction
        recon_slp_val = recon_flat_val.reshape(-1, 1, 53, 113) 
        true_slp_val = y_target_val.cpu().numpy()

    elif args.embed_method == 'vae':
        y_target_val = y_target_val.to(device)
        with torch.no_grad():
            recon_slp_tensor, _, _ = vae_model(y_target_val)
        recon_slp_val = recon_slp_tensor.cpu().numpy()
        true_slp_val = y_target_val.cpu().numpy()

    rmse = np.sqrt(np.mean((true_slp_val - recon_slp_val)**2))
    print(f"-> Erreur RMSE moyenne de reconstruction (pas de prise en compte du weight) : {rmse:.4f} (environ)")

    end_embed_time = time.time()
    print(f"Embedding completed in {(end_embed_time - start_embed_time) / 60:.2f} minutes")

    plot_reconstruction_check(true_slp_val, recon_slp_val, dates_val, outdir, args.embed_method, num_samples=10)
    print("-> Plot de vérification sauvegardé (Vérifie l'image avant de lancer 20 epochs !)\n")

    # ============================================================
    # 5. INITIALISATION DU CNN
    # ============================================================

    # --- MODIFICATION : Taille de sortie dynamique ---
    out_features = latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else latent_dim
    # NOUVEAU : On sélectionne la bonne liste de lags selon le mode pour avoir le bon nombre de canaux
    active_sst_lags = sst_lags_months if args.monthly_reduction else sst_lags_days
    active_slp_lags = slp_lags_months if args.monthly_reduction else slp_lags_days


    model = CNN_Latent_SLP_Multimodal1(
        dr=dr, 
        nb_out=out_features, 
        in_chans_sst=len(active_sst_lags), 
        in_chans_slp=len(active_slp_lags), 
        n_feat=8, 
        early_fusion_sst=args.early_fusion_sst
    ).to(device)

    # Petit passage factice (dummy forward) requis par le nn.LazyLinear 
    # pour qu'il calcule sa taille avant que l'optimiseur ne l'enregistre
    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(active_sst_lags), 85, 360).to(device) if len(active_sst_lags) > 0 else None
        dummy_slp = torch.zeros(1, len(active_slp_lags), 53, 113).to(device) if len(active_slp_lags) > 0 else None
        _ = model(dummy_sst, dummy_slp)

    print("Number of CNN parameters : ", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if args.update == 1:
        initial_params = torch.load(f"{outdir}/final_model_CNN_bs{bs}.pth")
        model.load_state_dict(initial_params['state_dict'])
        optimizer.load_state_dict(initial_params['optimizer'])
        train_losses = initial_params['train_losses']
        val_losses = initial_params['val_losses']
        best_val_loss = np.min(val_losses)
        train_corrs = initial_params.get('train_corrs', [])
        val_corrs = initial_params.get('val_corrs', [])
        train_R2 = initial_params.get('train_R2', [])
        val_R2 = initial_params.get('val_R2', [])
        test_losses = initial_params.get('test_losses', [])
        test_corrs = initial_params.get('test_corrs', [])
        test_R2 = initial_params.get('test_R2', [])
        train_ks = initial_params.get('train_ks', [])
        val_ks = initial_params.get('val_ks', [])
        test_ks = initial_params.get('test_ks', [])
        print("Model state updated")
    else:
        train_losses, val_losses, train_corrs, val_corrs,train_R2,val_R2,test_losses,test_corrs,test_R2 = [], [], [], [], [], [], [], [], []
        train_ks, val_ks, test_ks = [], [], []
        best_val_loss = float('inf') 
        print("Initiated first CNN training")

    best_model_path = "" # pour supprimer au fur à mesure les best qui sont save

    val_losses_per_member_history = defaultdict(list)

    # NOUVEAU : Suivi de la loss par batch pour l'époque 1
    epoch1_batch_losses = []
    epoch1_baseline_losses = []

    # Variables pour le suivi intra-époque 1
    intra_epoch1_steps = []
    intra_epoch1_val_losses = []
    intra_epoch1_val_corrs = []
    intra_epoch1_val_R2 = []
    intra_epoch1_test_losses = []
    intra_epoch1_test_corrs = []
    intra_epoch1_test_R2 = []

    # Variables pour le suivi intra-époque 2
    intra_epoch2_steps = []
    intra_epoch2_val_losses = []
    intra_epoch2_val_corrs = []
    intra_epoch2_val_R2 = []
    intra_epoch2_test_losses = []
    intra_epoch2_test_corrs = []
    intra_epoch2_test_R2 = []

    # Variables pour le suivi intra-époque 1
    intra_epoch1_val_ks, intra_epoch1_test_ks = [], []
    # Variables pour le suivi intra-époque 2
    intra_epoch2_val_ks, intra_epoch2_test_ks = [], []

    # ============================================================
    # CALCUL DES STEPS DE VALIDATION INTRA-ÉPOQUE (Espacement Logarithmique epoch 1)
    # ============================================================
    nb_intra_evals = args.nb_intra_evals  # Le nombre total de points que tu veux sur ta courbe, si c'est 0, on ne fait pas de validation intra-époque
    total_batches = len(trainloader)
    
    # geomspace génère des points espacés exponentiellement (ex: 1, 3, 10, 31, 100...)
    # On va de 1 à la fin de l'époque
    eval_steps = np.geomspace(1, total_batches - 1, num=nb_intra_evals, dtype=int)
    
    # On ajoute le step 0, et on utilise un "set" pour éviter les doublons au début
    eval_steps = np.insert(eval_steps, 0, 0)
    eval_steps_set = set(eval_steps)
    
    print(f"Validation intra-époque aux steps : {sorted(list(eval_steps_set))}")

    eval_steps_epoch2 = np.linspace(0, total_batches - 1, num=nb_intra_evals, dtype=int)
    eval_steps_epoch2 = np.insert(eval_steps_epoch2, 0, 0)
    eval_steps_epoch2_set = set(eval_steps_epoch2)

    # ============================================================
    # 6. TRAINING & EVALUATION LOOP (CNN)
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
        # pour le calcul de la corrélation : 
        sum_p, sum_t = 0.0, 0.0
        sum_p2, sum_t2 = 0.0, 0.0
        sum_pt = 0.0
        sum_res = 0.0
        
        # CORRECTION : Unpacking complet des 6 variables !
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
            if batch_idx % 30 == 0:
                print(f" {100 * batch_idx / len(trainloader):.1f}% training complete", end='\r')
                
            optimizer.zero_grad()
            X_sst = X_sst.to(device, non_blocking=True) 
            X_slp = X_slp.to(device, non_blocking=True) 
            
            # --- GENERATION DE LA CIBLE LATENTE ---
            if args.embed_method == 'pca':
                slp_flat = y_target.view(y_target.size(0), -1).numpy()

                # CORRECTION : On pondère la cible avant de la projeter !
                if args.lat_weight and wgts_flat is not None:
                    slp_flat = slp_flat * wgts_flat
                embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
            elif args.embed_method == 'vae':
                y_target = y_target.to(device, non_blocking=True) 
                with torch.no_grad():
                    target_embed, _ = vae_model.encode(y_target)
                    
            # --- PREDICTION ET LOSS ---
            predicted_latent = model(X_sst, X_slp) # Remplacement de 'inputs'            
            loss_value = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')

            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item() * X_sst.size(0)
            total_train_samples += X_sst.size(0)

            # Calcul pour la corrélation sur l'epoch
            med_pred = get_median_prediction(predicted_latent, args.loss_type, args.quantiles, latent_dim) if args.loss_type == 'quantile' else predicted_latent 
            # On somme sur la dimension du batch (dim=0)
            p = med_pred.detach()
            t = target_embed.detach()
            
            sum_p += p.sum(dim=0)
            sum_t += t.sum(dim=0)
            sum_p2 += (p ** 2).sum(dim=0)
            sum_t2 += (t ** 2).sum(dim=0)
            sum_pt += (p * t).sum(dim=0)
            sum_res += ((p - t) ** 2).sum(dim=0)

            # ----- NOUVEAU : Enregistrer la loss du batch et la baseline (Époque 1 uniquement) -----
            if epoch == 0:
                epoch1_batch_losses.append(loss_value.item())
                
                # Calcul de la loss de la baseline (prédiction constante = 0)
                with torch.no_grad():
                    zeros_pred = torch.zeros_like(predicted_latent)
                    baseline_loss = compute_loss(zeros_pred, target_embed, args.loss_type, args.quantiles, reduction='mean').item()
                    epoch1_baseline_losses.append(baseline_loss)

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
                    
                        # 1. On passe le modèle en mode eval
                        model.eval()
                        
                        intra_val_loss = 0.0
                        intra_n_samples = 0
                        
                        # Variables pour la corrélation avec la méthode Zéro Mémoire
                        v_sum_p, v_sum_t = 0.0, 0.0
                        v_sum_p2, v_sum_t2 = 0.0, 0.0
                        v_sum_pt = 0.0
                        v_sum_res = 0.0
                        
                        with torch.no_grad():
                            # On utilise un sous-ensemble ou tout le valloader
                            for v_batch_idx, (v_X_sst, v_X_slp, v_y_target, _, _, _) in enumerate(loader):
                                v_X_sst = v_X_sst.to(device, non_blocking=True)
                                v_X_slp = v_X_slp.to(device, non_blocking=True)
                                v_y_target = v_y_target.to(device, non_blocking=True)
                                
                                # Encodage (à adapter selon PCA/VAE)
                                # --- GENERATION DE LA CIBLE LATENTE ---
                                if args.embed_method == 'pca':
                                    slp_flat = v_y_target.view(v_y_target.size(0), -1).cpu().numpy()
                                    if args.lat_weight and wgts_flat is not None:
                                        slp_flat = slp_flat * wgts_flat
                                    embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                                    v_target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                                elif args.embed_method == 'vae':
                                    v_y_target = v_y_target.to(device, non_blocking=True) 
                                    with torch.no_grad():
                                        v_target_embed, _ = vae_model.encode(v_y_target)
                                v_pred = model(v_X_sst, v_X_slp)
                                
                                loss_val = compute_loss(v_pred, v_target_embed, args.loss_type, args.quantiles, reduction='mean')
                                intra_val_loss += loss_val.item() * v_X_sst.size(0)
                                
                                # Accumulation pour la corrélation
                                p = get_median_prediction(v_pred, args.loss_type, args.quantiles, latent_dim) if args.loss_type == 'quantile' else v_pred
                                t = v_target_embed
                                
                                v_sum_p += p.sum(dim=0)
                                v_sum_t += t.sum(dim=0)
                                v_sum_p2 += (p ** 2).sum(dim=0)
                                v_sum_t2 += (t ** 2).sum(dim=0)
                                v_sum_pt += (p * t).sum(dim=0)
                                v_sum_res += ((p - t) ** 2).sum(dim=0)
                                intra_n_samples += p.size(0)

                        # Calcul final intra-époque
                        v_mean_p = v_sum_p / intra_n_samples
                        v_mean_t = v_sum_t / intra_n_samples
                        v_var_p = (v_sum_p2 / intra_n_samples) - v_mean_p**2
                        v_var_t = (v_sum_t2 / intra_n_samples) - v_mean_t**2
                        v_cov_pt = (v_sum_pt / intra_n_samples) - (v_mean_p * v_mean_t)
                        
                        v_corr = (v_cov_pt / torch.sqrt(v_var_p * v_var_t + 1e-8)).mean().item()
                        v_ss_tot = v_var_t * intra_n_samples  # Variance totale * N
                        v_r2_vector = 1 - (v_sum_res / (v_ss_tot + 1e-8))
                        v_epoch_train_r2 = v_r2_vector.mean().item()

                        # AJOUT : Calcul de v_k = std(pred) / std(target)
                        v_k_vector = torch.sqrt(v_var_p / (v_var_t + 1e-8))
                        v_k = v_k_vector.mean().item()
                        
                        
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
                            best_model_path = os.path.join(outdir, f'best_val_CNN_bs{bs}_ep{epoch + 1}_step{batch_idx}_loss{best_val_loss:.4f}.pth')
                            torch.save(model.state_dict(), best_model_path)
                            print(f"   *** Nouveau Best Model (Intra) sauvegardé : {os.path.basename(best_model_path)} ***")

                        # 2. IMPORTANT : Ne pas oublier de repasser le modèle en mode train !
                        model.train()

            # -----------------------------------------------------------------------------------------

        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)
        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}')

        # À LA FIN DE L'ÉPOQUE TRAIN :
        mean_p = sum_p / total_train_samples
        mean_t = sum_t / total_train_samples
        var_p = (sum_p2 / total_train_samples) - mean_p**2
        var_t = (sum_t2 / total_train_samples) - mean_t**2
        cov_pt = (sum_pt / total_train_samples) - (mean_p * mean_t)

        # Calcul de la corrélation finale (avec un epsilon pour éviter la division par zéro)
        train_corr_vector = cov_pt / torch.sqrt(var_p * var_t + 1e-8)
        epoch_train_corr = train_corr_vector.mean().item() # Moyenne sur les dimensions latentes
        train_corrs.append(epoch_train_corr)

        # NOUVEAU : Calcul du R2
        ss_tot = var_t * total_train_samples # Variance totale * N
        r2_vector = 1 - (sum_res / (ss_tot + 1e-8))
        epoch_train_r2 = r2_vector.mean().item()
        train_R2.append(epoch_train_r2)

        # AJOUT : Calcul du k d'entraînement
        train_k_vector = torch.sqrt(var_p / (var_t + 1e-8))
        epoch_train_k = train_k_vector.mean().item()
        train_ks.append(epoch_train_k)

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


        # -----------------------------------------------------------

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
                # Unpacking complet !
                for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(loader):
                    if batch_idx % 30 == 0:
                        print(f" {100 * batch_idx / len(loader):.1f}% {key} complete", end='\r')
                    
                    X_sst = X_sst.to(device, non_blocking=True)
                    X_slp = X_slp.to(device, non_blocking=True)
                    y_target = y_target.to(device, non_blocking=True)
                    
                    # --- 1. ENCODAGE DE LA VRAIE SLP ---
                    if args.embed_method == 'pca':
                        slp_flat = y_target.view(y_target.size(0), -1).cpu().numpy()
                        if args.lat_weight and wgts_flat is not None:
                            slp_flat = slp_flat * wgts_flat
                        embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                        target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                    elif args.embed_method == 'vae':
                        target_embed, _ = vae_model.encode(y_target)

                    # --- 2. PRÉDICTION DU CNN ET CALCUL DE LA LOSS ---
                    predicted_latent = model(X_sst, X_slp)
                    loss_value = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='mean')
                    running_val_loss += loss_value.item() * X_sst.size(0)
                    total_val_samples += X_sst.size(0)

                    per_sample_losses = compute_loss(predicted_latent, target_embed, args.loss_type, args.quantiles, reduction='none').cpu().numpy()
                    # --- 3. DÉCODAGE DOUBLE (Basé sur la médiane si quantile loss) ---
                    median_pred_latent = get_median_prediction(predicted_latent, args.loss_type, args.quantiles, latent_dim)

                    # On somme sur la dimension du batch (dim=0)
                    p = median_pred_latent.detach()
                    t = target_embed.detach()
                    
                    sum_p += p.sum(dim=0)
                    sum_t += t.sum(dim=0)
                    sum_p2 += (p ** 2).sum(dim=0)
                    sum_t2 += (t ** 2).sum(dim=0)
                    sum_pt += (p * t).sum(dim=0)
                    sum_res += ((p - t) ** 2).sum(dim=0) # NOUVEAU

                    if key == 'val':

                        # --- 3. DÉCODAGE DOUBLE ---
                        if args.embed_method == 'pca':
                            # pred_np = predicted_latent.cpu().numpy()
                            pred_np = median_pred_latent.cpu().numpy()
                            target_np = target_embed.cpu().numpy()
                            
                            padded_pred = np.zeros((pred_np.shape[0], pca_model.n_components_))
                            padded_target = np.zeros((target_np.shape[0], pca_model.n_components_))
                            
                            padded_pred[:, :latent_dim] = pred_np
                            padded_target[:, :latent_dim] = target_np

                            # Cartes pondérées (polluées)
                            predicted_slp_flat_polluted = pca_model.inverse_transform(padded_pred)
                            recon_true_slp_flat_polluted = pca_model.inverse_transform(padded_target)
                            
                            # RETOUR À LA PHYSIQUE : Division par sqrt(cos(lat))
                            if args.lat_weight and 'safe_wgts' in locals():
                                predicted_slp_flat = predicted_slp_flat_polluted / safe_wgts
                                recon_true_slp_flat = recon_true_slp_flat_polluted / safe_wgts
                            else:
                                predicted_slp_flat = predicted_slp_flat_polluted
                                recon_true_slp_flat = recon_true_slp_flat_polluted

                            predicted_slp = predicted_slp_flat.reshape(-1, 1, 53, 113) 
                            recon_true_slp = recon_true_slp_flat.reshape(-1, 1, 53, 113)
                            
                        elif args.embed_method == 'vae':
                            # predicted_slp = vae_model.decode(predicted_latent).cpu().numpy()
                            predicted_slp = vae_model.decode(median_pred_latent).cpu().numpy()
                            recon_true_slp = vae_model.decode(target_embed).cpu().numpy()

                        # --- 4. STOCKAGE DES LISTES PAR MEMBRES ---
                        y_map_np = y_map.numpy()
                        members_list = []
                        for m in members:
                            try:
                                members_list.append(m if isinstance(m, str) else m.item().decode() if isinstance(m.item(), bytes) else str(m.item()))
                            except:
                                members_list.append(str(m))

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
            
            # Corrélation sur la validation :
            mean_p = sum_p / total_val_samples
            mean_t = sum_t / total_val_samples
            var_p = (sum_p2 / total_val_samples) - mean_p**2
            var_t = (sum_t2 / total_val_samples) - mean_t**2
            cov_pt = (sum_pt / total_val_samples) - (mean_p * mean_t)

            # Calcul de la corrélation finale (avec un epsilon pour éviter la division par zéro)
            val_corr_vector = cov_pt / torch.sqrt(var_p * var_t + 1e-8)
            epoch_val_corr = val_corr_vector.mean().item() # Moyenne sur les dimensions latentes
            val_corrs.append(epoch_val_corr) if key == 'val' else test_corrs.append(epoch_val_corr)

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

                    # Suppression de l'ancien meilleur modèle s'il existe
                    if best_model_path and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                        
                    # Formatage du nouveau nom dynamique (on met 'end' pour signifier fin d'époque)
                    best_model_path = os.path.join(outdir, f'best_val_CNN_bs{bs}_ep{epoch + 1}_end_loss{best_val_loss:.4f}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"   *** Nouveau Best Model (Fin d'époque) sauvegardé : {os.path.basename(best_model_path)} ***")
                else:
                    patience_counter += 1

            # On print la loss de la phase en cours (Val ou Test)
            print(f'Epoch {epoch + 1} {key} Loss: {val_loss:.6f}')
        
        # 1. Gestion du temps globale de l'époque (Une seule fois !)
        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        print(f"--> Fin de l'époque {epoch + 1} - Elapsed Time: {current_time_min:.2f} minutes")

        # 2. Vrai déclenchement de l'Early Stopping (Quitte la boucle des époques)
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
            break
        
        # ---------------- AFFICHAGE ET SAUVEGARDE ----------------
        if (epoch + 1) % 2 == 0:
            state = {'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    'train_losses': train_losses, 'val_losses': val_losses, 'train_corrs': train_corrs, 'train_R2': train_R2, 'val_corrs': val_corrs, 'val_R2': val_R2, 'test_losses': test_losses, 'test_corrs': test_corrs, 'test_R2': test_R2, 'train_ks': train_ks, 'val_ks': val_ks, 'test_ks': test_ks}
            torch.save(state, f'{outdir}/final_model_CNN_bs{bs}.pth')
            loss_figure(len(train_losses), train_losses, val_losses, outdir,epoch_times,per_member_val_losses=val_losses_per_member_history,test_losses=test_losses)
            plot_correlation_evolution(train_corrs, val_corrs,outdir,test_corrs = test_corrs, train_ks=train_ks, val_ks=val_ks, test_ks=test_ks)
            plot_r2_R2_evolution(train_corrs, val_corrs, train_R2, val_R2, outdir,test_R2 = test_R2)
            print(f"Saved checkpoint at epoch {epoch + 1}")
            
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

    loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times,per_member_val_losses=val_losses_per_member_history,test_losses=test_losses)
    plot_correlation_evolution(train_corrs, val_corrs,outdir,test_corrs = test_corrs, train_ks=train_ks, val_ks=val_ks, test_ks=test_ks)
    plot_r2_R2_evolution(train_corrs, val_corrs, train_R2, val_R2, outdir,test_R2 = test_R2,test_corrs = test_corrs)
    # Save final model
    state = {'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses, 'train_corrs': train_corrs, 'train_R2': train_R2, 'val_corrs': val_corrs, 'val_R2': val_R2, 'test_corrs': test_corrs, 'test_R2': test_R2, 'train_ks': train_ks, 'val_ks': val_ks, 'test_ks': test_ks}
    torch.save(state, f'{outdir}/final_model_CNN_bs{bs}.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_CNN_bs{bs}.pth')

    # ============================================================
    # END OF TRAINING

    # ============================================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training complete, elapsed time: {elapsed_time / 60:.2f} minutes")

    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation automatique...")
    print("="*50)

    eval_script_path = os.path.join(os.path.dirname(__file__), "eval_cnn.py")

    for model_type in ["final", "best"]:
        for monthly_mean in [False, True]:
            print(f"\n--- Évaluation du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            
            # 1. On initialise la commande avec les arguments simples (convertis en strings)
            eval_command = [
                sys.executable, 
                eval_script_path,
                "--machine", str(args.machine),
                "--embed_method", str(args.embed_method),
                "--cnn_dir", str(outdir),      
                "--model_type", str(model_type),
                "--nb_members_train", str(args.nb_members_train),
                "--nb_members_val", str(args.nb_members_val),
                "--nb_members_test", str(args.nb_members_test),
                "--seed", str(args.seed),
                "--latent_dim", str(latent_dim),
                "--duree_lissage", str(args.duree_lissage),
                "--bs", str(args.bs),
                "--loss_type", str(args.loss_type)
            ]
            
            if args.embed_path:
                eval_command.extend(["--embed_path", str(args.embed_path)])

            # 2. On ajoute les listes (nargs='*') en les aplatissant
            if args.sst_lags_days is not None:
                eval_command.extend(["--sst_lags_days"] + [str(x) for x in args.sst_lags_days])
            if args.sst_lags_months is not None:
                eval_command.extend(["--sst_lags_months"] + [str(x) for x in args.sst_lags_months])
            if args.slp_lags_days is not None:
                eval_command.extend(["--slp_lags_days"] + [str(x) for x in args.slp_lags_days])
            if args.slp_lags_months is not None:
                eval_command.extend(["--slp_lags_months"] + [str(x) for x in args.slp_lags_months])
            if args.winter_months is not None:
                eval_command.extend(["--winter_months"] + [str(x) for x in args.winter_months])
            if args.quantiles:
                eval_command.extend(["--quantiles"] + [str(x) for x in args.quantiles])

            # 3. On ajoute les flags booléens (action='store_true') seulement s'ils sont True
            if args.roll_sst:
                eval_command.append("--roll_sst")
            if args.early_fusion_sst:
                eval_command.append("--early_fusion_sst")
            if monthly_mean:
                eval_command.append("--monthly_mean")
            if args.monthly_reduction:
                eval_command.append("--monthly_reduction")
            if args.lat_weight:
                eval_command.append("--lat_weight")

            # Exécution de la commande
            try:
                result = subprocess.run(eval_command, check=True, text=True)
                print(f"\n✅ Évaluation de {model_type} terminée avec succès !")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Erreur lors de l'exécution de l'évaluation. Code de retour : {e.returncode}")

    print("\n" + "="*50)
    print("🚀 Lancement de l'évaluation SPATIALE automatique...")
    print("="*50)

    # Remplace par le nom exact de ton deuxième script d'évaluation
    eval_spatial_script_path = os.path.join(os.path.dirname(__file__), "eval_cnn_full_slp.py")

    for model_type in ["final", "best"]:
        for monthly_mean in [False, True]:
            print(f"\n--- Évaluation spatiale du modèle : {model_type} | Moyenne mensuelle : {monthly_mean} ---")
            
            # 1. On initialise la commande avec les arguments simples
            eval_spatial_command = [
                sys.executable, 
                eval_spatial_script_path,
                "--machine", str(args.machine),
                "--embed_method", str(args.embed_method),
                "--cnn_dir", str(outdir),      
                "--model_type", str(model_type),
                "--nb_members_train", str(args.nb_members_train),
                "--nb_members_val", str(args.nb_members_val),
                "--nb_members_test", str(args.nb_members_test),
                "--seed", str(args.seed),
                "--latent_dim", str(latent_dim),
                "--duree_lissage", str(args.duree_lissage),
                "--bs", str(args.bs),
                "--loss_type", str(args.loss_type)
            ]
            
            # --- GESTION DU EMBED_PATH ---
            if args.embed_path:
                eval_spatial_command.extend(["--embed_path", str(args.embed_path)])
            else:
                ext = "joblib" if args.embed_method == 'pca' else "pth"
                eval_spatial_command.extend(["--embed_path", os.path.join(outdir, f"{args.embed_method}_model.{ext}")])

            # 2. On ajoute les listes (nargs='*') en les aplatissant
            if args.sst_lags_days is not None:
                eval_spatial_command.extend(["--sst_lags_days"] + [str(x) for x in args.sst_lags_days])
            if args.slp_lags_days is not None:
                eval_spatial_command.extend(["--slp_lags_days"] + [str(x) for x in args.slp_lags_days])
            if hasattr(args, 'sst_lags_months') and args.sst_lags_months is not None:
                eval_spatial_command.extend(["--sst_lags_months"] + [str(x) for x in args.sst_lags_months])
            if hasattr(args, 'slp_lags_months') and args.slp_lags_months is not None:
                eval_spatial_command.extend(["--slp_lags_months"] + [str(x) for x in args.slp_lags_months])
            if args.winter_months is not None:
                eval_spatial_command.extend(["--winter_months"] + [str(x) for x in args.winter_months])
            if args.quantiles:
                eval_spatial_command.extend(["--quantiles"] + [str(x) for x in args.quantiles])

            # 3. On ajoute les flags booléens
            if args.roll_sst:
                eval_spatial_command.append("--roll_sst")
            if args.early_fusion_sst:
                eval_spatial_command.append("--early_fusion_sst")
            if args.monthly_reduction:
                eval_spatial_command.append("--monthly_reduction")
            if args.lat_weight:
                eval_spatial_command.append("--lat_weight")
            if monthly_mean:
                eval_spatial_command.append("--monthly_mean")

            # Exécution
            try:
                result = subprocess.run(eval_spatial_command, check=True, text=True)
                print(f"\n✅ Évaluation spatiale de {model_type} terminée avec succès !")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Erreur lors de l'exécution de l'évaluation spatiale. Code de retour : {e.returncode}")