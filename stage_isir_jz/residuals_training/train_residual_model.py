import os
import time
import argparse
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Setup des chemins d'importation
project_root = Path(__file__).resolve().parent.parent
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

cnn_dir = os.path.join(project_root, "cnn")
if cnn_dir not in sys.path:
    sys.path.append(cnn_dir)

from tools.datasets import Dataset
from tools.models import ConvVAE, ViT_Latent_SLP_Multimodal
from tools_cnn.models import CNN_Latent_SLP_Multimodal1
from tools.visualizations import loss_figure

# ============================================================
# CLASSE RÉGRESSION LINÉAIRE (on importe pas le main de linreg pour éviter les conflits de path (meme nom))
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

def plot_residual_time_series(ds_member, member, outdir, pc_idx, freq_label):
    """Trace les séries temporelles latentes : Base vs Combined et Res_Target vs Res_Pred."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    time_vals = ds_member['time'].values
    x_idx = np.arange(len(time_vals))
    x_labels = pd.to_datetime(time_vals).strftime('%Y-%m-%d' if freq_label == "Daily" else '%Y-%m').tolist()

    lw = 0.8 if freq_label == "Daily" else 1.5
    ms = 2 if freq_label == "Daily" else 6

    # =======================================================
    # PLOT 1 : True Target vs Base vs Combined
    # =======================================================
    ax1 = axes[0]
    ax1.plot(x_idx, ds_member[f'true_pc{pc_idx}'], color="black", marker='.', linewidth=lw, markersize=ms, label="True Target")
    ax1.plot(x_idx, ds_member[f'base_pc{pc_idx}'], color="firebrick", linestyle="--", linewidth=lw, label="Base Prediction")
    ax1.plot(x_idx, ds_member[f'comb_pc{pc_idx}'], color="tab:blue", marker='.', linewidth=lw, markersize=ms, alpha=0.8, label="Combined (Base + Res) Prediction")
    
    # Calcul de la RMSE pour le titre
    rmse_base = np.sqrt(np.mean((ds_member[f'true_pc{pc_idx}'].values - ds_member[f'base_pc{pc_idx}'].values)**2))
    rmse_comb = np.sqrt(np.mean((ds_member[f'true_pc{pc_idx}'].values - ds_member[f'comb_pc{pc_idx}'].values)**2))
    
    ax1.set_title(f"PC {pc_idx} - Target vs Predictions ({freq_label}) | RMSE Base: {rmse_base:.3f} -> RMSE Comb: {rmse_comb:.3f}", fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # =======================================================
    # PLOT 2 : Residual Target vs Residual Prediction
    # =======================================================
    ax2 = axes[1]
    ax2.plot(x_idx, ds_member[f'res_target_pc{pc_idx}'], color="black", marker='.', linewidth=lw, markersize=ms, label="Residual Target (True - Base)")
    ax2.plot(x_idx, ds_member[f'res_pred_pc{pc_idx}'], color="mediumseagreen", marker='.', linewidth=lw, markersize=ms, label="Residual Prediction")
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1.5)
    
    corr = np.corrcoef(ds_member[f'res_target_pc{pc_idx}'].values, ds_member[f'res_pred_pc{pc_idx}'].values)[0, 1]
    
    ax2.set_title(f"PC {pc_idx} - Residuals ({freq_label}) | Correlation: {corr:.3f}", fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # =======================================================
    # FORMATTAGE DES AXES
    # =======================================================
    for ax in axes:
        n_ticks = min(15, len(x_idx))
        tick_indices = np.linspace(0, len(x_idx) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([x_labels[idx] for idx in tick_indices], rotation=45, ha="right")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(f"PC {pc_idx} Value", fontsize=12)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f'Latent_TimeSeries_PC{pc_idx}_{freq_label}_Member_{member}.png'), dpi=300)
    plt.close(fig)

# ============================================================
# FACTORY DE MODÈLES (FLEXIBILITÉ CNN / VIT / LINREG)
# ============================================================
def create_model(model_type, in_chans_sst, in_chans_slp, out_features, mix_sst=False, dr=0.0):
    """Instancie le modèle désiré de manière dynamique. TO DO : """
    if model_type == 'cnn':
        return CNN_Latent_SLP_Multimodal1(
            dr=dr, nb_out=out_features, in_chans_sst=in_chans_sst, in_chans_slp=in_chans_slp, n_feat=8, early_fusion_sst=mix_sst
        )
    elif model_type == 'vit':
        return ViT_Latent_SLP_Multimodal(
            sst_size=(85, 360), slp_size=(53, 113), patch_size_sst=(5, 10), patch_size_slp=(5, 10), 
            in_chans_sst=in_chans_sst, in_chans_slp=in_chans_slp, embed_dim=128, depth=4, num_heads=4, 
            dr=dr, nb_out=out_features, use_lags_attention=mix_sst
        )
    elif model_type == 'linreg':
        return LinearRegressionPredictor(
            sst_shape=(85, 360), slp_shape=(53, 113), in_chans_sst=in_chans_sst, in_chans_slp=in_chans_slp, out_dim=out_features
        )
    else:
        raise ValueError(f"Model type {model_type} non reconnu.")

# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # --- Arguments Globaux ---
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca')
    parser.add_argument('--embed_path', type=str, required=True, help="Chemin vers le PCA/VAE")
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--roll_sst', action='store_true', help="doit aussi matcher ce qui a été utilisé pour entraîner le modèle de base")
    parser.add_argument('--seed', type=int, default=42, help ="plus rigoureux ou interprétable si match avec le seed utilisé pour entraîner le modèle de base")

    # --- Arguments Modèle de Base (Prédicteur initial) ---
    # pas de drop out car on évalue juste un modèle pré-entraîné
    parser.add_argument('--base_model_type', type=str, choices=['cnn', 'vit', 'linreg'], required=True)
    parser.add_argument('--base_model_path', type=str, required=True, help="Poids du modèle de base à charger")
    parser.add_argument('--base_sst_lags', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--base_slp_lags', type=int, nargs='*', default=[])
    parser.add_argument('--base_model_use_lags_attention', action='store_true', help="Si activé, le ViT utilisera une attention spécifique pour les lags au lieu de les concaténer simplement.")
    parser.add_argument('--base_model_early_fusion_sst', action='store_true', help="Si activé, le modèle de base fusionnera les lags SST très tôt dans le réseau (ex: dès les premières couches) au lieu de les traiter séparément.")
    
    # --- Arguments Modèle Résiduel (Celui qu'on entraîne) ---
    parser.add_argument('--res_model_type', type=str, choices=['cnn', 'vit', 'linreg'], required=True)
    parser.add_argument('--res_sst_lags', type=int, nargs='*', default=[-7], help="Lags pour prédire l'erreur (ex: lag très court)")
    parser.add_argument('--res_slp_lags', type=int, nargs='*', default=[])
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--dr', type=float, default=0.22)
    parser.add_argument('--nb_members_train', type=int, default=10)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--use_lags_attention', action='store_true', help="Si activé, le ViT utilisera une attention spécifique pour les lags au lieu de les concaténer simplement.")
    parser.add_argument('--early_fusion_sst', action='store_true', help="Si activé, le modèle résiduel fusionnera les lags SST très tôt dans le réseau (ex: dès les premières couches) au lieu de les traiter séparément.")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================
    # 1. GESTION DES LAGS (L'astuce d'alignement)
    # ============================================================
    # On fait l'union de tous les lags pour que le DataLoader ait les mêmes dates.
    all_sst_lags = sorted(list(set(args.base_sst_lags + args.res_sst_lags)))
    all_slp_lags = sorted(list(set(args.base_slp_lags + args.res_slp_lags)))

    # On pré-calcule les indices pour découper le tenseur plus tard
    base_sst_idx = [all_sst_lags.index(lag) for lag in args.base_sst_lags]
    res_sst_idx  = [all_sst_lags.index(lag) for lag in args.res_sst_lags]
    base_slp_idx = [all_slp_lags.index(lag) for lag in args.base_slp_lags]
    res_slp_idx  = [all_slp_lags.index(lag) for lag in args.res_slp_lags]

    print(f"Combined SST Lags for DataLoader: {all_sst_lags}")
    print(f" -> Base Model uses indices: {base_sst_idx}")
    print(f" -> Residual Model uses indices: {res_sst_idx}")

    # ============================================================
    # 2. SETUP DOSSIER & DATALOADERS
    # ============================================================
    res_mix_sst = False
    base_mix_sst = False
    if args.res_model_type == "vit" and args.use_lags_attention:
        res_mix_sst = True
    if args.base_model_type == "vit" and args.base_model_use_lags_attention:
        base_mix_sst = True
    if args.res_model_type == "cnn" and args.early_fusion_sst:
        res_mix_sst = True
    if args.base_model_type == "cnn" and args.base_model_early_fusion_sst:
        base_mix_sst = True

    outdir_name = f"Residual_{args.res_model_type}_on_{args.base_model_type}_resLags_{'_'.join(map(str, args.res_sst_lags))}_baseLags_{'_'.join(map(str, args.base_sst_lags))}_lr{args.lr}_bs{args.bs}_months_{'_'.join(map(str, args.winter_months))}_seed_{args.seed}_{args.duree_lissage}d_train{args.nb_members_train}_val{args.nb_members_val}_roll_sst_{args.roll_sst}_mixSSTres_{res_mix_sst}_mixSSTbase_{base_mix_sst}_dr{args.dr}"
    outdir = os.path.join("output_residuals", outdir_name) # Dossier spécifique pour les résidus
    os.makedirs(outdir, exist_ok=True)

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    import random
    rng = random.Random(args.seed)
    rng.shuffle(all_members)
    train_members = all_members[:args.nb_members_train]
    val_members = all_members[-args.nb_members_val:]

    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1)
    
    # On donne l'union des lags au Dataset
    train_set = Dataset(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=all_sst_lags, slp_lags_days=all_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    val_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=all_sst_lags, slp_lags_days=all_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # 3. LOAD EMBEDDER & BASE MODEL
    # ============================================================
    if args.embed_method == 'pca':
        pca_model = joblib.load(args.embed_path)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=args.latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

    print(f"\n--- Chargement du Modèle de Base ({args.base_model_type.upper()}) ---")
    base_model = create_model(args.base_model_type, len(args.base_sst_lags), len(args.base_slp_lags), args.latent_dim, dr=0., mix_sst=base_mix_sst).to(device)
    
    checkpoint_base = torch.load(args.base_model_path, map_location=device)
    base_model.load_state_dict(checkpoint_base.get('state_dict', checkpoint_base))
    base_model.eval() # Toujours en eval
    for param in base_model.parameters():
        param.requires_grad = False # On gèle les poids du modèle de base

    # ============================================================
    # 4. INITIALISATION DU MODÈLE RÉSIDUEL
    # ============================================================
    print(f"\n--- Initialisation du Modèle Résiduel ({args.res_model_type.upper()}) ---")
    res_model = create_model(args.res_model_type, len(args.res_sst_lags), len(args.res_slp_lags), args.latent_dim, dr=args.dr, mix_sst=res_mix_sst).to(device)
    
    optimizer = torch.optim.Adam(res_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ============================================================
    # 5. TRAINING LOOP (Apprentissage du Résidu)
    # ============================================================
    train_losses, res_val_losses, base_val_losses, combined_val_losses = [], [], [], []
    best_val_loss = float('inf')
    start_time = time.time()
    epoch_times = []

    for epoch in range(args.nb_epochs):
        res_model.train()
        running_train_loss = 0.0
        total_train_samples = 0
        
        for X_sst_all, X_slp_all, y_target, _, _, _ in trainloader:
            X_sst_all = X_sst_all.to(device, non_blocking=True)
            X_slp_all = X_slp_all.to(device, non_blocking=True)
            
            # --- DÉCOUPAGE DES TENSEURS SELON LES LAGS ---
            X_sst_base = X_sst_all[:, base_sst_idx, :, :] if len(base_sst_idx) > 0 else None
            X_slp_base = X_slp_all[:, base_slp_idx, :, :] if len(base_slp_idx) > 0 else None
            
            X_sst_res = X_sst_all[:, res_sst_idx, :, :] if len(res_sst_idx) > 0 else None
            X_slp_res = X_slp_all[:, res_slp_idx, :, :] if len(res_slp_idx) > 0 else None
            
            # --- CIBLE LATENTE ---
            if args.embed_method == 'pca':
                target_latent = torch.tensor(pca_model.transform(y_target.view(y_target.size(0), -1).numpy())[:, :args.latent_dim], dtype=torch.float32).to(device)
            elif args.embed_method == 'vae':
                with torch.no_grad():
                    target_latent, _ = vae_model.encode(y_target.to(device))

            # --- CALCUL DU RÉSIDU (Cible de l'entraînement) ---
            with torch.no_grad():
                base_pred = base_model(X_sst_base, X_slp_base)
                residual_target = target_latent - base_pred  # L'erreur que le base model a fait

            # --- ENTRAÎNEMENT DU SECOND MODÈLE ---
            optimizer.zero_grad()
            res_pred = res_model(X_sst_res, X_slp_res) # Prédit l'erreur
            
            loss = criterion(res_pred, residual_target) # Loss sur l'erreur
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * X_sst_all.size(0)
            total_train_samples += X_sst_all.size(0)

        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)
        
        # ---------------- VALIDATION ----------------
        res_model.eval()
        running_val_res_loss = 0.0
        running_combined_loss = 0.0 # Score final = Base + Residu
        running_val_base_loss = 0.0 # Juste pour voir la performance du modèle de base seul sur la validation
        total_val_samples = 0
        
        with torch.no_grad():
            for X_sst_all, X_slp_all, y_target, _, _, _ in valloader:
                X_sst_all = X_sst_all.to(device, non_blocking=True)
                X_slp_all = X_slp_all.to(device, non_blocking=True)
                
                X_sst_base = X_sst_all[:, base_sst_idx, :, :] if len(base_sst_idx) > 0 else None
                X_slp_base = X_slp_all[:, base_slp_idx, :, :] if len(base_slp_idx) > 0 else None
                
                X_sst_res = X_sst_all[:, res_sst_idx, :, :] if len(res_sst_idx) > 0 else None
                X_slp_res = X_slp_all[:, res_slp_idx, :, :] if len(res_slp_idx) > 0 else None
                
                if args.embed_method == 'pca':
                    target_latent = torch.tensor(pca_model.transform(y_target.view(y_target.size(0), -1).numpy())[:, :args.latent_dim], dtype=torch.float32).to(device)
                elif args.embed_method == 'vae':
                    target_latent, _ = vae_model.encode(y_target.to(device))

                # Predictions
                base_pred = base_model(X_sst_base, X_slp_base)
                residual_target = target_latent - base_pred
                res_pred = res_model(X_sst_res, X_slp_res)
                
                # La vraie prédiction finale de ton système boosté
                final_combined_pred = base_pred + res_pred

                # Calcul des losses
                res_loss = criterion(res_pred, residual_target)
                comb_loss = criterion(final_combined_pred, target_latent)
                base_loss = criterion(base_pred, target_latent)
                
                running_val_res_loss += res_loss.item() * X_sst_all.size(0)
                running_combined_loss += comb_loss.item() * X_sst_all.size(0)
                running_val_base_loss += base_loss.item() * X_sst_all.size(0)
                total_val_samples += X_sst_all.size(0)

        val_res_loss = running_val_res_loss / total_val_samples
        val_comb_loss = running_combined_loss / total_val_samples
        val_base_loss = running_val_base_loss / total_val_samples
        res_val_losses.append(val_res_loss)
        base_val_losses.append(val_base_loss)
        combined_val_losses.append(val_comb_loss)
        
        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        
        print(f"Epoch {epoch+1:02d} | Train Res Loss: {train_loss:.5f} | Val Res Loss: {val_res_loss:.5f} || Final Combined Val MSE: {val_comb_loss:.5f}")

        # Sauvegarde
        if val_res_loss < best_val_loss:
            best_val_loss = val_res_loss
            torch.save({'state_dict': res_model.state_dict()}, os.path.join(outdir, f"best_val_ResModel.pth"))

    # Sauvegarde finale et plots
    torch.save({'state_dict': res_model.state_dict(), 'train_losses': train_losses, 'res_val_losses': res_val_losses, 'base_val_losses': base_val_losses, 'combined_val_losses': combined_val_losses}, os.path.join(outdir, f"final_ResModel.pth"))
    
    loss_figure(len(train_losses), train_losses, res_val_losses, outdir, epoch_times)
    loss_figure(len(train_losses), base_val_losses, combined_val_losses, outdir, epoch_times, train_loss_label="Base Model Loss", val_loss_label="Combined Model Loss", name="Fig_base_vs_combined_val_loss.png")
    print(f"\n✅ Entraînement terminé en {(time.time() - start_time) / 60:.2f} minutes.")
    print(f"Meilleure erreur sur les résidus: {best_val_loss:.5f}")
    print(f"Les modèles sont sauvegardés dans {outdir}")

    # ============================================================
    # 6. ÉVALUATION FINALE : PLOT DES SÉRIES TEMPORELLES LATENTES
    # ============================================================
    print("\n--- Génération des graphiques de Séries Temporelles Latentes ---")
    
    # Recharger le meilleur modèle de résidu
    best_checkpoint = torch.load(os.path.join(outdir, f"best_val_ResModel.pth"), map_location=device)
    res_model.load_state_dict(best_checkpoint['state_dict'])
    res_model.eval()

    dates_list, members_list = [], []
    true_latents, base_preds, res_targets, res_preds, comb_preds = [], [], [], [], []

    with torch.no_grad():
        for X_sst_all, X_slp_all, y_target, _, dates, members in valloader:
            X_sst_all, X_slp_all = X_sst_all.to(device), X_slp_all.to(device)
            
            X_sst_base = X_sst_all[:, base_sst_idx, :, :] if len(base_sst_idx) > 0 else None
            X_slp_base = X_slp_all[:, base_slp_idx, :, :] if len(base_slp_idx) > 0 else None
            X_sst_res = X_sst_all[:, res_sst_idx, :, :] if len(res_sst_idx) > 0 else None
            X_slp_res = X_slp_all[:, res_slp_idx, :, :] if len(res_slp_idx) > 0 else None
            
            # Target
            if args.embed_method == 'pca':
                tgt_lat = torch.tensor(pca_model.transform(y_target.view(y_target.size(0), -1).numpy())[:, :args.latent_dim], dtype=torch.float32).to(device)
            elif args.embed_method == 'vae':
                tgt_lat, _ = vae_model.encode(y_target.to(device))

            # Predictions
            b_pred = base_model(X_sst_base, X_slp_base)
            r_tgt = tgt_lat - b_pred
            r_pred = res_model(X_sst_res, X_slp_res)
            c_pred = b_pred + r_pred

            # Stockage Numpy
            true_latents.append(tgt_lat.cpu().numpy())
            base_preds.append(b_pred.cpu().numpy())
            res_targets.append(r_tgt.cpu().numpy())
            res_preds.append(r_pred.cpu().numpy())
            comb_preds.append(c_pred.cpu().numpy())
            
            dates_list.extend([str(d) for d in dates])
            for m in members:
                m_str = m if isinstance(m, str) else (m.item().decode() if isinstance(m.item(), bytes) else str(m.item()))
                members_list.append(m_str)

    # Concaténation globale
    true_arr = np.concatenate(true_latents, axis=0)
    base_arr = np.concatenate(base_preds, axis=0)
    rtgt_arr = np.concatenate(res_targets, axis=0)
    rpred_arr = np.concatenate(res_preds, axis=0)
    comb_arr = np.concatenate(comb_preds, axis=0)

    # Création du DataFrame Pandas
    df_eval = pd.DataFrame({'time': pd.to_datetime(dates_list), 'member': members_list})
    for i in range(args.latent_dim):
        df_eval[f'true_pc{i+1}'] = true_arr[:, i]
        df_eval[f'base_pc{i+1}'] = base_arr[:, i]
        df_eval[f'res_target_pc{i+1}'] = rtgt_arr[:, i]
        df_eval[f'res_pred_pc{i+1}'] = rpred_arr[:, i]
        df_eval[f'comb_pc{i+1}'] = comb_arr[:, i]

    # Définir les fréquences à plot (Daily et/ou Monthly)
    # Si tu veux forcer les deux, tu peux définir frequencies = ["Daily", "Monthly"] 
    # ou utiliser l'argument du parseur.
    frequencies = ["Daily", "Monthly"]
    
    max_pcs_to_plot = min(2, args.latent_dim) # On trace PC1 et PC2 max pour pas saturer
    unique_members = df_eval['member'].unique()

    for freq in frequencies:
        outdir_ts = os.path.join(outdir, f"latent_timeseries_{freq}")
        os.makedirs(outdir_ts, exist_ok=True)
        
        for member in unique_members:
            df_member = df_eval[df_eval['member'] == member].copy()
            ds_member = df_member.set_index('time').to_xarray()
            
            # Application de la fréquence demandée
            if freq == "Monthly":
                ds_member = ds_member.resample(time='1M').mean().dropna(dim="time")
            else:
                ds_member = ds_member.dropna(dim="time")

            # Tracer pour chaque PC
            for pc_idx in range(1, max_pcs_to_plot + 1):
                # Validation de sécurité (pas assez de data pour faire un plot)
                if len(ds_member['time']) < 2:
                    continue
                    
                member_outdir = os.path.join(outdir_ts, str(member))
                plot_residual_time_series(ds_member, member, member_outdir, pc_idx, freq)
                
    print(f"✅ Graphiques de séries temporelles générés dans {outdir}")