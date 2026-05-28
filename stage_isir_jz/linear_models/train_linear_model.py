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
import matplotlib.pyplot as plt # NOUVEAU : pour l'explicabilité

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import cartopy.crs as ccrs

import sys 
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# Ajouter le dossier "tools" de vision_transformer au sys.path pour les imports
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

from tools.visualizations import loss_figure, loss_first_epoch, plot_and_save_maps_with_reconstruction, plot_and_save_maps_with_reconstruction_light, plot_reconstruction_check
from tools.datasets import Dataset, Dataset_faster2
from tools.models import ConvVAE, vae_loss

# ============================================================
# NOUVEAU : MODÈLE DE RÉGRESSION LINÉAIRE
# ============================================================
class LinearRegressionPredictor(nn.Module):
    def __init__(self, sst_shape=(85, 360), slp_shape=(53, 113), in_chans_sst=3, in_chans_slp=0, out_dim=128):
        super().__init__()
        self.sst_size = in_chans_sst * sst_shape[0] * sst_shape[1]
        self.slp_size = in_chans_slp * slp_shape[0] * slp_shape[1]
        self.total_input_size = self.sst_size + self.slp_size
        
        # Une seule couche linéaire pour faire la régression
        self.linear = nn.Linear(self.total_input_size, out_dim)

    def forward(self, x_sst, x_slp):
        batch_size = x_sst.size(0)
        
        # Aplatir les dimensions spatiales
        x_sst_flat = x_sst.view(batch_size, -1)

        # du coup on part du principe qu'il y a au moins un lag de sst je crois
        
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
    parser.add_argument('--update', type=int, required=True, help='Loading of previous parameters (1) or start fresh (0)') # il me semble que cet argument n'est plus utilisé, à vérifier

    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca', help='Méthode pour l\'espace latent')
    parser.add_argument('--embed_path', type=str, default='', help='Chemin vers le VAE/PCA pré-entraîné.')
    parser.add_argument('--machine', type=str, default='mac_local', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    
    parser.add_argument('--nb_members_train', type=int, default=10)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4) # LR souvent plus grand pour régression
    parser.add_argument('--alpha_penalty', type=float, default=1e-5, help='Poids de la pénalité L1 ou L2')
    parser.add_argument('--penalty_type', type=str, choices=['l1', 'l2'], default='l2', help='Type de pénalité à utiliser')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--beta_kld', type=float, default=1.0)
    parser.add_argument('--normalize', action='store_true')

    parser.add_argument('--exact_solver', action='store_true', help="Utiliser la formule mathématique exacte (Ridge/OLS)")
    parser.add_argument('--max_samples_exact', type=int, default=2000, help="Limite d'échantillons N pour éviter le Out-Of-Memory")
    
    args = parser.parse_args()

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
    bs = args.bs
    lr = args.lr
    winter_months = args.winter_months
    duree_lissage = args.duree_lissage
    nb_epochs = args.nb_epochs
    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val

    print("Arg Parameters:")
    print(f"  Latent Dim: {latent_dim}", f"sst_lags_days: {sst_lags_days}", f"slp_lags_days: {slp_lags_days}", f"Batch Size: {bs}", f"Learning Rate: {lr}", f"Winter Months: {winter_months}", f"Smoothing Duration: {duree_lissage}", f"Number of Epochs: {nb_epochs}", f"Number of Training Members: {nb_members_train}", f"Number of Validation Members: {nb_members_val}\n")

    patience = 10000
    target_indices = {100, 1000, 2000, 3000, 4000, 4500, 5000, 6000, 7000, 8000}

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[-nb_members_val:]

    # Nom du dossier adapté pour la régression linéaire
    if args.exact_solver:
        outdir_name = f"LinReg_{args.penalty_type}_{args.alpha_penalty}_{args.exact_solver}_exact_{args.max_samples_exact}_samples_{args.embed_method}_emb_{latent_dim}_sst_{'_'.join(map(str, sst_lags_days))}_slp_{'_'.join(map(str, slp_lags_days))}_bs{bs}_lr{lr}_seed_{args.seed}_{duree_lissage}d_train{nb_members_train}_val{nb_members_val}"
    else:
        outdir_name = f"LinReg_{args.penalty_type}_{args.alpha_penalty}_{args.embed_method}_emb_{latent_dim}_sst_{'_'.join(map(str, sst_lags_days))}_slp_{'_'.join(map(str, slp_lags_days))}_bs{bs}_lr{lr}_seed_{args.seed}_{duree_lissage}d_train{nb_members_train}_val{nb_members_val}"
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")

    # ============================================================
    # DATALOADERS
    # ============================================================
    val_set = Dataset(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst = True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    training_set = Dataset(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst = True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

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
            pca_model = PCA(n_components=latent_dim, whiten = args.normalize) 
            slp_list = []
            for X_sst, X_slp, y_target, y_map, dates, members in trainloader:
                slp_list.append(y_target.view(y_target.size(0), -1).numpy())
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
    # 5. INITIALISATION DU MODÈLE DE RÉGRESSION
    # ============================================================
    model = LinearRegressionPredictor(
        sst_shape=(85, 360), 
        slp_shape=(53, 113), 
        in_chans_sst=len(sst_lags_days), 
        in_chans_slp=len(slp_lags_days), 
        out_dim=latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Number of Linear parameters : {sum(p.numel() for p in model.parameters())}")

    # ============================================================
    # SOLUTION ANALYTIQUE EXACTE (SANITY CHECK)
    # ============================================================
    if args.exact_solver:
        if args.penalty_type == 'l1':
            raise ValueError("Pas de solution analytique fermée pour la pénalité L1 (Lasso). Utilise L2.")
            
        print(f"\n--- CALCUL DE LA SOLUTION EXACTE (Limité à {args.max_samples_exact} échantillons) ---")
        model.eval()
        X_list, Y_list = [], []
        current_samples = 0
        
        # 1. Collecte d'un sous-ensemble des données
        for X_sst, X_slp, y_target, _, _, _ in trainloader:
            batch_size = X_sst.size(0)
            
            X_sst_flat = X_sst.view(batch_size, -1).cpu()
            if X_slp.numel() > 0:
                X_slp_flat = X_slp.view(batch_size, -1).cpu()
                X_batch = torch.cat([X_sst_flat, X_slp_flat], dim=1)
            else:
                X_batch = X_sst_flat
                
            # Encodage de la target
            if args.embed_method == 'pca':
                slp_flat = y_target.view(batch_size, -1).numpy()
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
                print(f"Limite de {args.max_samples_exact} échantillons atteinte. Arrêt de la collecte.")
                break
                
        X = torch.cat(X_list, dim=0)[:args.max_samples_exact] # Shape: (N, D)
        Y = torch.cat(Y_list, dim=0)[:args.max_samples_exact] # Shape: (N, K)
        
        # 2. Centrage (Obligatoire pour gérer le biais / l'ordonnée à l'origine)
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        X_c = X - X_mean
        Y_c = Y - Y_mean
        
        N, D = X.shape
        # La loss PyTorch MSE est une moyenne sur N. 
        # Pour retrouver la même pénalité exacte, on multiplie le lambda par N.
        lambda_ridge = args.alpha_penalty * N 
        
        print(f"Dimensions - Échantillons (N): {N}, Features spatiales (D): {D}")
        
        # 3. Résolution avec formulation duale (Woodbury)
        # Car avec N=2000 et D=91800, N est largement inférieur à D
        print("Résolution matricielle en cours (Formulation Duale)...")
        start_math = time.time()
        
        # Matrice de Gram (N x N)
        K_mat = X_c @ X_c.T + lambda_ridge * torch.eye(N) 
        
        # Inversion et calcul des poids
        dual_coef = torch.linalg.solve(K_mat, Y_c)
        W_exact = X_c.T @ dual_coef # Shape: (D, K)
        
        # Calcul du biais pour annuler le centrage initial
        bias_exact = Y_mean.squeeze() - (X_mean @ W_exact).squeeze()
        
        print(f"Calcul terminé en {time.time() - start_math:.2f} secondes.")
        
        # 4. Injection dans les poids du modèle PyTorch
        # Attention: nn.Linear attend un tenseur de shape (out_features, in_features) soit (K, D)
        with torch.no_grad():
            model.linear.weight.copy_(W_exact.T.to(device))
            model.linear.bias.copy_(bias_exact.to(device))
            
        print("-> Solution mathématique injectée dans le modèle avec succès.")
        
        # On définit ce modèle comme le meilleur et on bypass l'entraînement itératif
        best_model_state = copy.deepcopy(model.state_dict())
        nb_epochs = 1

    val_losses_per_member_history = defaultdict(list)
    train_losses, val_losses = [], []
    best_val_loss = float('inf') 

    # NOUVEAU : Suivi de la loss par batch pour l'époque 1
    epoch1_batch_losses = []
    epoch1_baseline_losses = []

    # ============================================================
    # 6. TRAINING & EVALUATION LOOP
    # ============================================================
    start_time = time.time()
    epoch_times = []
    best_model_state = None
    patience_counter = 0

    for epoch in range(nb_epochs):
        # ---------------- TRAINING ----------------
        if not args.exact_solver:
            model.train()
            running_train_loss = 0.0
            total_train_samples = 0
            
            for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
                optimizer.zero_grad()
                X_sst = X_sst.to(device, non_blocking=True) 
                X_slp = X_slp.to(device, non_blocking=True) 
                
                if args.embed_method == 'pca':
                    slp_flat = y_target.view(y_target.size(0), -1).numpy()
                    embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                    target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                elif args.embed_method == 'vae':
                    y_target = y_target.to(device, non_blocking=True)
                    with torch.no_grad():
                        target_embed, _ = vae_model.encode(y_target)
                        
                predicted_latent = model(X_sst, X_slp)
                
                # --- CALCUL DE LA LOSS AVEC PENALITE CORRIGEE ---
                mse_loss = criterion(predicted_latent, target_embed)
                
                if args.penalty_type == 'l1':
                    penalty = torch.norm(model.linear.weight, p=1)
                elif args.penalty_type == 'l2':
                    # Ridge utilise la norme L2 AU CARRÉ (évite la racine carrée de torch.norm)
                    penalty = torch.sum(model.linear.weight ** 2)
                else:
                    penalty = 0.0
                    
                loss_value = mse_loss + args.alpha_penalty * penalty
                
                loss_value.backward()
                optimizer.step()
                running_train_loss += mse_loss.item() * X_sst.size(0) # On garde que la MSE pour l'affichage
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
            print(f'Epoch {epoch + 1} Training MSE Loss: {train_loss:.8f}')

            # ----- NOUVEAU : Appel de la fonction de visualisation -----
            if epoch == 0:
                loss_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, outdir)
                
        else:
            print("\n--- EXACT SOLVER: Bypass de l'entraînement itératif ---")
            train_losses.append(0.0) # Valeur factice pour ne pas faire planter loss_figure
        # ---------------- VALIDATION ----------------
        model.eval()
        
        running_val_loss = 0.0
        total_val_samples = 0 

        # Plus élaboré : validation par membre

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

                # --- 2. PRÉDICTION PAR LE MODÈLE DE RÉGRESSION ---
                predicted_latent = model(X_sst, X_slp)
                # --- CALCUL DE LA LOSS AVEC PENALITE CORRIGEE ---
                mse_loss = criterion(predicted_latent, target_embed)
                
                if args.penalty_type == 'l1':
                    penalty = torch.norm(model.linear.weight, p=1)
                elif args.penalty_type == 'l2':
                    # Ridge utilise la norme L2 AU CARRÉ (évite la racine carrée de torch.norm)
                    penalty = torch.sum(model.linear.weight ** 2)
                else:
                    penalty = 0.0
                    
                loss_value = mse_loss + args.alpha_penalty * penalty
                running_val_loss += loss_value.item()* X_sst.size(0) 
                total_val_samples += X_sst.size(0)

                per_sample_losses = (torch.mean((predicted_latent - target_embed)**2, dim=1)+args.alpha_penalty * penalty).cpu().numpy()

                # --- 3. DÉCODAGE DOUBLE (Prédiction de la regression + Plafond de verre) ---
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

    # Load best model for explainability
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_model_LinReg.pth')

    # ============================================================
    # 7. EXPLICABILITÉ : VISUALISATION DES POIDS (PIXEL IMPORTANCE)
    # ============================================================
    print("\n--- GÉNÉRATION DES CARTES D'EXPLICABILITÉ ---")
    
    def plot_explainability_weights(model, outdir, sst_lags_days, slp_lags_days, sst_shape=(85, 360), slp_shape=(53, 113)):
        # Extents géographiques
        extent_sst = [-180, 180, -15, 70] 
        extent_slp = [-100, 40, 20, 70] 
        
        # Récupération des poids entraînés : shape = [latent_dim, total_input_size]
        weights = model.linear.weight.detach().cpu().numpy()
        latent_dimension = weights.shape[0]
        
        # TEST 1D vs Multi-D
        if latent_dimension == 1:
            print("-> Target 1D détectée : Conservation des signes des coefficients (Pas de valeur absolue).")
            global_importance = weights[0, :]
            is_1d = True
        else:
            print(f"-> Target Multi-D ({latent_dimension}) détectée : Calcul de la moyenne absolue des poids.")
            global_importance = np.mean(np.abs(weights), axis=0)
            is_1d = False
        
        # Dimensions pour le découpage
        in_chans_sst = len(sst_lags_days)
        in_chans_slp = len(slp_lags_days)
        sst_size_total = in_chans_sst * sst_shape[0] * sst_shape[1]
        
        # Découpage et reshape
        sst_weights = global_importance[:sst_size_total].reshape(in_chans_sst, sst_shape[0], sst_shape[1])

        if is_1d:
            global_vmax = np.max(np.abs(sst_weights))
            global_vmin = -global_vmax
            current_cmap = 'RdBu_r'
        else:
            global_vmax = sst_weights.max()
            global_vmin = 0.0
            current_cmap = 'Reds'
        
        # --------------------------------------------------------
        # PLOT SST AVEC COASTLINES
        # --------------------------------------------------------
        fig, axes = plt.subplots(
            1, in_chans_sst, 
            figsize=(6 * in_chans_sst, 4), 
            subplot_kw={'projection': ccrs.PlateCarree()},
            facecolor='white'
        )
        if in_chans_sst == 1: axes = [axes]
            
        for i, lag in enumerate(sst_lags_days):
            ax = axes[i]
            ax.set_facecolor('white')

            im = ax.imshow(
                sst_weights[i], 
                cmap=current_cmap, 
                origin='lower', 
                vmin=global_vmin, 
                vmax=global_vmax, 
                transform=ccrs.PlateCarree(), 
                extent=extent_sst,
                interpolation='nearest'
            )
            
            ax.set_title(f"SST Importance - Lag {lag} days", fontsize=12)
            ax.coastlines(resolution='110m', color='black', linewidth=0.8)
            
            # Plus besoin de bidouiller les limites de l'axe de la colorbar, matplotlib gère tout avec vmin/vmax
            fig.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.08)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "explainability_SST_weights.png"), dpi=150, facecolor='white')
        plt.close()
        
        # --------------------------------------------------------
        # PLOT SLP AVEC COASTLINES (si utilisé)
        # --------------------------------------------------------
        if in_chans_slp > 0:
            slp_weights = global_importance[sst_size_total:].reshape(in_chans_slp, slp_shape[0], slp_shape[1])

            if is_1d:
                global_vmax = np.max(np.abs(slp_weights))
                global_vmin = -global_vmax
                current_cmap = 'RdBu_r'
            else:
                global_vmax = slp_weights.max()
                global_vmin = 0.0
                current_cmap = 'Reds'

            fig, axes = plt.subplots(
                1, in_chans_slp, 
                figsize=(6 * in_chans_slp, 4), 
                subplot_kw={'projection': ccrs.PlateCarree()},
                facecolor='white'
            )
            if in_chans_slp == 1: axes = [axes]

                
            for i, lag in enumerate(slp_lags_days):
                ax = axes[i]
                ax.set_facecolor('white')
                
                    
                im = ax.imshow(
                    slp_weights[i], 
                    cmap=current_cmap, 
                    origin='lower', 
                    vmin=global_vmin, 
                    vmax=global_vmax, 
                    transform=ccrs.PlateCarree(), 
                    extent=extent_slp,
                    interpolation='nearest'
                )
                
                ax.set_title(f"SLP Importance - Lag {lag} days", fontsize=12)
                ax.coastlines(resolution='110m', color='black', linewidth=0.8)
                
                fig.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.08)

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "explainability_SLP_weights.png"), dpi=150, facecolor='white')
            plt.close()
            
        print(f"Cartes d'explicabilité sauvegardées dans {outdir}")

    # Appel de la fonction
    plot_explainability_weights(model, outdir, sst_lags_days, slp_lags_days)

    print(f"Training complete, elapsed time: {(time.time() - start_time) / 60:.2f} minutes")

