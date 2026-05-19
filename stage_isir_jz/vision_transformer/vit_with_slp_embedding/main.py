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

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

# import des dossiers siblings
import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.visualizations import loss_figure, loss_first_epoch, plot_and_save_maps_with_reconstruction, plot_and_save_maps_with_reconstruction_light, plot_reconstruction_check
from tools.datasets import Dataset, Dataset_faster2
from tools.models import ViT_Latent_SLP_Multimodal, ConvVAE, vae_loss


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
    
    parser.add_argument('--nb_members_train', type=int, default=10, help='Nombre de membres à utiliser pour l\'entraînement (sur les 89 disponibles)')
    parser.add_argument('--nb_members_val', type=int, default=5, help='Nombre de membres à utiliser pour la validation (sur les 89 disponibles)')
    parser.add_argument('--seed', type=int, default=42, help='Seed pour le mélange inter membres')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension de l\'espace latent (nombre de composantes PCA ou taille du code VAE)')
    parser.add_argument('--nb_epochs', type=int, default=30, help='Nombre d\'époques pour l\'entraînement du ViT')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours pour les cibles 10 ou 30')
    parser.add_argument('--bs', type=int, default=128, help='Taille de batch pour l\'entraînement')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate pour l\'entraînement du ViT')
    parser.add_argument('--dr', type=float, default=0.22, help='Dropout rate pour le ViT')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95], help='Liste des lags pour SST (ex: --sst_lags_days 30 60 90)')
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[], help='Liste des lags pour SLP (optionnel)')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner pour l\'entraînement (ex: --winter_months 1 2 pour janvier et février)')
    parser.add_argument('--beta_kld', type=float, default=1.0, help='Coefficient de la composante KL divergence dans la loss du VAE (si utilisé)')
    parser.add_argument('--normalize', action='store_true', help='PCA normalisé ou non')

    
    args = parser.parse_args()

    # Routage dynamique des dossiers
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/vision_transformer/vit_with_slp_embedding/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']:
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/"
    elif args.machine == 'mac_local':
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/vision_transformer/vit_with_slp_embedding/"


    # ============================================================
    # GLOBAL CONSTANTS
    # ============================================================

    #train_members_80 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013']
    #nb_members_train = 50
    #train_members = train_members_80[:nb_members_train]
    #val_members = ['1001.001', '1301.010','1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    #val_members = ["1301.010"]
    # sst_lags_days = [35,65,95]
    # slp_lags_days = []
    # bs = 128
    # lr = 5e-5
    # dr = 0.22
    # latent_dim = 128
    # nb_epochs = 30
    # winter_months = [1, 2]
    # duree_lissage = 30

    latent_dim = args.latent_dim
    sst_lags_days = args.sst_lags_days
    slp_lags_days = args.slp_lags_days
    bs = args.bs
    lr = args.lr
    dr = args.dr
    winter_months = args.winter_months
    duree_lissage = args.duree_lissage
    nb_epochs = args.nb_epochs
    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val

    print("Arg Parameters:")
    print(f"  Latent Dim: {latent_dim}", f" SST Lags: {sst_lags_days}", f" SLP Lags: {slp_lags_days}", f" Batch Size: {bs}", f" Learning Rate: {lr}", f" Dropout Rate: {dr}", f" Winter Months: {winter_months}", f" Smoothing Duration: {duree_lissage}", f" Number of Epochs: {nb_epochs}", f" Number of Training Members: {nb_members_train}", f" Number of Validation Members: {nb_members_val}\n")

    patience = 10000
    target_indices = {100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000} # pour les plots de comparaison de validation. 

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[-nb_members_val:]

    if args.embed_method == 'pca':
        outdir_name = f"ViT_Latent_{args.embed_method}_normalize_{args.normalize}_embedding_{latent_dim}_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_members_seed_{args.seed}_{duree_lissage}d"
    elif args.embed_method == 'vae':
        outdir_name = f"ViT_Latent_{args.embed_method}_beta_{args.beta_kld}_embedding_{latent_dim}_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_members_seed_{args.seed}_{duree_lissage}d"
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")

    # ============================================================
    # DATALOADERS
    # ============================================================
    val_set = Dataset(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    training_set = Dataset(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

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
            print("Training VAE from scratch on TrainLoader (10 epochs)...")
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
    # On récupère un seul batch du set de validation
    X_sst_val, X_slp_val, y_target_val, y_map_val, dates_val, members_val = next(iter(valloader))

    if args.embed_method == 'pca':
        explained_var = np.sum(pca_model.explained_variance_ratio_)
        print(f"-> Variance expliquée par les {latent_dim} composantes PCA : {explained_var * 100:.2f}%")
        
        slp_flat_val = y_target_val.view(y_target_val.size(0), -1).cpu().numpy()
        # Adapté à latent_dim < 128 (typiquement)
        # 1. On projette et on ne garde que les 'latent_dim' premières composantes
        latent_val = pca_model.transform(slp_flat_val)[:, :latent_dim]
        
        # 2. On recrée un vecteur vide de la taille d'origine du modèle PCA (ex: 128)
        padded_latent = np.zeros((latent_val.shape[0], pca_model.n_components_))
        padded_latent[:, :latent_dim] = latent_val # On insère nos composantes
        
        # 3. On reconstruit l'image
        recon_flat_val = pca_model.inverse_transform(padded_latent)

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
    print("-> Plot de vérification sauvegardé (Vérifie l'image avant de lancer 20 epochs !)\n")

    # ============================================================
    # 5. INITIALISATION DU VISION TRANSFORMER
    # ============================================================
    model = ViT_Latent_SLP_Multimodal(
        sst_size=(85, 360), slp_size=(53, 113), patch_size_sst=(5, 10), patch_size_slp=(5, 10), in_chans_sst=len(sst_lags_days), in_chans_slp=len(slp_lags_days), embed_dim=128, depth=4, num_heads=4, dr=dr, nb_out=latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # Loss classique car on prédit un vecteur

    print("Number of ViT parameters : ", sum(p.numel() for p in model.parameters()))

    if args.update == 1:
        initial_params = torch.load(f"{outdir}/final_model_ViT_bs{bs}.pth")
        model.load_state_dict(initial_params['state_dict'])
        optimizer.load_state_dict(initial_params['optimizer'])
        train_losses = initial_params['train_losses']
        val_losses = initial_params['val_losses']
        best_val_loss = np.min(val_losses)
        print("Model state updated")
    else:
        train_losses, val_losses = [], []
        best_val_loss = float('inf') 
        print("Initiated first ViT training")

    # --- placez ceci AVANT la boucle des epochs ---
    val_losses_per_member_history = defaultdict(list)

    # NOUVEAU : Suivi de la loss par batch pour l'époque 1
    epoch1_batch_losses = []
    epoch1_baseline_losses = []

    # ============================================================
    # 6. TRAINING & EVALUATION LOOP (ViT)
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
        
        # CORRECTION : Unpacking complet des 6 variables !
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
            if batch_idx % 30 == 0:
                print(f" {100 * batch_idx / len(trainloader):.1f}% training complete", end='\r')
                
            optimizer.zero_grad()
            X_sst = X_sst.to(device, non_blocking=True) 
            X_slp = X_slp.to(device, non_blocking=True) 
            
            # --- GENERATION DE LA CIBLE LATENTE ---
            if args.embed_method == 'pca':
                # Scikit-learn PCA fonctionne sur CPU avec NumPy
                slp_flat = y_target.view(y_target.size(0), -1).numpy() # Remplacement de 'targets_slp'
                # On tronque pour ne garder que les PC qui nous intéressent
                embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
            elif args.embed_method == 'vae':
                y_target = y_target.to(device, non_blocking=True) # Remplacement de 'targets_slp'
                with torch.no_grad():
                    target_embed, _ = vae_model.encode(y_target)
                    
            # --- PREDICTION ET LOSS ---
            predicted_latent = model(X_sst, X_slp) # Remplacement de 'inputs'
            loss_value = criterion(predicted_latent, target_embed)
            
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item() * X_sst.size(0)
            total_train_samples += X_sst.size(0)

            # ----- NOUVEAU : Enregistrer la loss du batch et la baseline (Époque 1 uniquement) -----
            if epoch == 0:
                epoch1_batch_losses.append(loss_value.item())
                
                # Calcul de la loss de la baseline (prédiction constante = 0)
                with torch.no_grad():
                    zeros_pred = torch.zeros_like(target_embed)
                    baseline_loss = criterion(zeros_pred, target_embed).item()
                    epoch1_baseline_losses.append(baseline_loss)

        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)
        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}')

        # ----- NOUVEAU : Appel de la fonction de visualisation -----
        if epoch == 0:
            loss_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, outdir)
        # -----------------------------------------------------------

    # ---------------- VALIDATION ----------------
        model.eval()
        
        # time_list, slp_true_list, slp_recon_true_list, slp_pred_list = [], [], [], [] 
        running_val_loss = 0.0
        total_val_samples = 0 

        # Plus élaboré : validation par membre



        # --- dans la boucle des epochs, remplacez la collecte globale par ceci ---
        # Validation (remplace le bloc qui construisait time_list/slp_true_list globaux)
        # per_member = defaultdict(lambda: {
        #     'time': [], 'slp_true': [], 'slp_recon_true': [], 'slp_pred': [],
        #     'loss_sum': 0.0, 'count': 0
        # })
        # pour la version light de la visualisation : 

        per_member_loss = defaultdict(lambda: {'loss_sum': 0.0, 'count': 0})
        per_member_plots = defaultdict(lambda: {'time': [], 'slp_true': [], 'slp_recon_true': [], 'slp_pred': []})

        with torch.no_grad():
            # Unpacking complet !
            for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(valloader):
                if batch_idx % 30 == 0:
                    print(f" {100 * batch_idx / len(valloader):.1f}% val complete", end='\r')
                
                X_sst = X_sst.to(device, non_blocking=True)
                X_slp = X_slp.to(device, non_blocking=True)
                y_target = y_target.to(device, non_blocking=True)
                
                # --- 1. ENCODAGE DE LA VRAIE SLP ---
                if args.embed_method == 'pca':
                    slp_flat = y_target.view(y_target.size(0), -1).cpu().numpy()
                    embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                    target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                elif args.embed_method == 'vae':
                    target_embed, _ = vae_model.encode(y_target)

                # --- 2. PRÉDICTION DU ViT ET CALCUL DE LA LOSS ---
                predicted_latent = model(X_sst, X_slp)
                loss_value = criterion(predicted_latent, target_embed)
                running_val_loss += loss_value.item()* X_sst.size(0) 
                total_val_samples += X_sst.size(0)

                per_sample_losses = torch.mean((predicted_latent - target_embed)**2, dim=1).cpu().numpy()

                # --- 3. DÉCODAGE DOUBLE (Prédiction ViT + Plafond de verre) ---
                if args.embed_method == 'pca':
                    pred_np = predicted_latent.cpu().numpy()
                    target_np = target_embed.cpu().numpy()
                    
                    # Création des vecteurs remplis de zéros (taille : batch_size x 128)
                    padded_pred = np.zeros((pred_np.shape[0], pca_model.n_components_))
                    padded_target = np.zeros((target_np.shape[0], pca_model.n_components_))
                    
                    # Insertion des prédictions (taille : batch_size x latent_dim)
                    padded_pred[:, :latent_dim] = pred_np
                    padded_target[:, :latent_dim] = target_np

                    # A. Décodage de la prédiction du ViT
                    predicted_slp_flat = pca_model.inverse_transform(padded_pred)
                    predicted_slp = predicted_slp_flat.reshape(-1, 1, 53, 113) 
                    
                    # B. Décodage de la cible idéale (Le plafond de verre du PCA)
                    recon_true_slp_flat = pca_model.inverse_transform(padded_target)
                    recon_true_slp = recon_true_slp_flat.reshape(-1, 1, 53, 113)
                    
                elif args.embed_method == 'vae':
                    # A. Décodage de la prédiction du ViT
                    predicted_slp = vae_model.decode(predicted_latent).cpu().numpy()
                    
                    # B. NOUVEAU : Décodage de la cible idéale (Le plafond de verre du VAE)
                    recon_true_slp = vae_model.decode(target_embed).cpu().numpy()

                # --- 4. STOCKAGE DES LISTES ---
                # time_list.extend(dates) 
                # slp_true_list.append(y_map.numpy())                # Colonne 1 : La réalité brute
                # slp_recon_true_list.append(recon_true_slp)         # Colonne 2 : La reconstruction idéale
                # slp_pred_list.append(predicted_slp)                # Colonne 3 : La prédiction du modèle

                y_map_np = y_map.numpy()

                # normaliser members & dates en listes de strings
                members_list = []
                for m in members:
                    try:
                        members_list.append(m if isinstance(m, str) else m.item().decode() if isinstance(m.item(), bytes) else str(m.item()))
                    except:
                        members_list.append(str(m))

                dates_list = [d if isinstance(d, str) else str(d) for d in dates]

                # remplir le dict par échantillon
                for i, mem in enumerate(members_list):
                    current_idx = per_member_loss[mem]['count']

                    per_member_loss[mem]['loss_sum'] += float(per_sample_losses[i])
                    per_member_loss[mem]['count'] += 1
                    if current_idx in target_indices:
                        per_member_plots[mem]['time'].append(dates_list[i])
                        per_member_plots[mem]['slp_true'].append(y_map_np[i])
                        per_member_plots[mem]['slp_recon_true'].append(recon_true_slp[i])
                        per_member_plots[mem]['slp_pred'].append(predicted_slp[i])

        # Après la boucle valloader : calculs par membre et sauvegardes
        for mem, d in per_member_loss.items():
            avg_loss = d['loss_sum'] / d['count'] if d['count'] > 0 else float('nan')
            val_losses_per_member_history[mem].append(avg_loss)

        val_loss = running_val_loss / total_val_samples 
        val_losses.append(val_loss)
        

        # ---------------- EARLY STOPPING & SAVING ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"Saved best val model at epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
                break

        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        print(f'Epoch {epoch + 1} Val Loss: {val_loss:.6f} - Elapsed Time: {current_time_min:.2f} minutes')
        
        # ---------------- AFFICHAGE ET SAUVEGARDE ----------------
        if (epoch + 1) % 1 == 0:
            state = {'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    'train_losses': train_losses, 'val_losses': val_losses}
            torch.save(state, f'{outdir}/final_model_ViT_bs{bs}.pth')
            loss_figure(len(train_losses), train_losses, val_losses, outdir,epoch_times,per_member_val_losses=val_losses_per_member_history)
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

    loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times, per_member_val_losses=val_losses_per_member_history)

    # Save final model
    state = {'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'train_losses': train_losses, 'val_losses': val_losses}
    torch.save(state, f'{outdir}/final_model_ViT_bs{bs}.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_ViT_bs{bs}.pth')

    # ============================================================
    # END OF TRAINING

    # ============================================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training complete, elapsed time: {elapsed_time / 60:.2f} minutes")