import xarray as xr
import numpy as np
from eofs.xarray import Eof
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
import torch
import math
from scipy.stats import pearsonr, norm
from tools.datasets import Dataset
from tqdm import tqdm 
import cftime
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from datetime import timedelta, datetime

extent_slp = [-100, 40, 20, 70] 
#extent_sst = [0, 360, -20, 80]  # ou [-180,180,-20,80] selon si on a fait rolling ou pas... EN FAIT je crois que c'est [-180, 180, -15, 70]
extent_sst = [-180, 180, -15, 70] # cf Dataset qui fait slice sur la latitude et qui décale de 180 les longitudes (si on décommente torch.roll dans Dataset) 
# je crois que .set_extent est superflu 

# def loss_figure(epochs, train_losses, val_losses, outdir_new, epoch_times=None):
#     """
#     Loss avec double axe des abscisses : epochs et temps d'entraînement cumulé en minutes.
#     """
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    
#     ax.plot(range(epochs), train_losses, label='Train Loss', color='C0')
#     ax.plot(range(epochs), val_losses, label='Validation Loss', color='C1')
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('Loss')
#     ax.legend()
#     ax.set_title("Loss Evolution during training\n", pad=20) # pad pour faire de la place au 2eme axe
    
#     # --- NOUVEAU : Création du 2ème axe des abscisses ---
#     if epoch_times is not None and len(epoch_times) == epochs:
#         ax2 = ax.twiny() # Crée un second axe X qui partage le même axe Y
        
#         # On force les limites à être strictement identiques
#         ax2.set_xlim(ax.get_xlim())
        
#         # On choisit combien de "ticks" on veut afficher (ex: 6 max pour que ce soit lisible)
#         num_ticks = min(6, epochs)
#         if epochs > 1:
#             tick_indices = np.linspace(0, epochs - 1, num_ticks, dtype=int)
#         else:
#             tick_indices = [0]
            
#         # On place les graduations aux mêmes endroits que les époques choisies
#         ax2.set_xticks(tick_indices)
#         # On écrit le temps formaté en minutes (ex: "15.2m")
#         ax2.set_xticklabels([f"{epoch_times[i]:.1f}m" for i in tick_indices])
#         ax2.set_xlabel("Temps d'entraînement cumulé (minutes)")
    
#     figs_file = "Fig_loss-evolution-during-training.png"
#     figs_filename = os.path.join(outdir_new, figs_file)
#     plt.tight_layout()
#     plt.savefig(figs_filename)
#     plt.close()

def loss_figure(epochs, train_losses, val_losses, outdir_new, epoch_times=None, per_member_val_losses=None):
    """
    Trace la courbe d'entraînement + courbe de validation globale (si fournie)
    et, optionnellement, une courbe par membre (per_member_val_losses: dict member -> list).
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='C0')
    
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss (global)', color='C1')
    
    # Traces par membre (optionnel)
    if per_member_val_losses:
        cmap = plt.get_cmap('tab20')
        for i, (member, hist) in enumerate(per_member_val_losses.items()):
            if hist is None or len(hist) == 0:
                continue
            x = range(1, min(len(hist), epochs) + 1)
            ax.plot(x, hist[:len(x)], label=f'Val {member}', color=cmap((i+2) % 20), linestyle='--', alpha=0.9)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='best', fontsize='small')
    ax.set_title("Loss Evolution during training\n", pad=20)
    
    # Second axe temporel en minutes (optionnel)
    if epoch_times is not None and len(epoch_times) == len(train_losses):
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        num_ticks = min(6, len(train_losses))
        tick_indices = np.linspace(1, len(train_losses), num_ticks, dtype=int) if len(train_losses) > 1 else [1]
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels([f"{epoch_times[i-1]:.1f}m" for i in tick_indices])
        ax2.set_xlabel("Temps d'entraînement cumulé (minutes)")
    
    figs_file = "Fig_loss-evolution-during-training.png"
    figs_filename = os.path.join(outdir_new, figs_file)
    plt.tight_layout()
    plt.savefig(figs_filename)
    plt.close()

def accuracy_figure(epochs, train_accuracies, val_accuracies, outdir_new, epoch_times=None, per_member_val_accuracies=None):
    """
    Trace la courbe d'accuracy d'entraînement et de validation globale,
    et, optionnellement, une courbe par membre de validation.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    
    # Courbe d'entraînement
    ax.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Acc', color='C0', linewidth=2)
    
    # Courbe de validation globale
    if val_accuracies is not None and len(val_accuracies) > 0:
        ax.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Acc (global)', color='C2', linewidth=2)
    
    # Traces par membre (optionnel)
    if per_member_val_accuracies:
        cmap = plt.get_cmap('tab20')
        for i, (member, hist) in enumerate(per_member_val_accuracies.items()):
            if hist is None or len(hist) == 0:
                continue
            x = range(1, min(len(hist), epochs) + 1)
            ax.plot(x, hist[:len(x)], label=f'Val {member}', color=cmap((i+2) % 20), linestyle='--', alpha=0.9)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='best', fontsize='small')
    ax.set_title("Accuracy Evolution during training\n", pad=20)
    
    # Second axe temporel en minutes (optionnel)
    if epoch_times is not None and len(epoch_times) == len(train_accuracies):
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        num_ticks = min(6, len(train_accuracies))
        tick_indices = np.linspace(1, len(train_accuracies), num_ticks, dtype=int) if len(train_accuracies) > 1 else [1]
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels([f"{epoch_times[i-1]:.1f}m" for i in tick_indices])
        ax2.set_xlabel("Temps d'entraînement cumulé (minutes)")
    
    # Sauvegarde
    figs_file = "Fig_accuracy-evolution-during-training.png"
    figs_filename = os.path.join(outdir_new, figs_file)
    plt.tight_layout()
    plt.savefig(figs_filename)
    plt.close()

def loss_first_epoch(batch_losses, baseline_losses, outdir):
    """
    Trace et sauvegarde l'évolution de la loss par batch lors de la première époque,
    en la comparant batch par batch avec une baseline (prédiction = 0).
    """
    plt.figure(figsize=(10, 6))
    
    # Courbe d'entraînement du modèle
    plt.plot(batch_losses, label='Train Loss (ViT)', color='blue')
    
    # Courbe de la baseline évaluée sur les mêmes batchs
    plt.plot(baseline_losses, label='Baseline (Pred = 0)', color='red', linestyle='--', alpha=0.5)
    
    # NOUVEAU : Ligne horizontale pour la moyenne globale de la baseline
    mean_baseline = np.mean(baseline_losses)
    plt.axhline(y=mean_baseline, color='red', linestyle='--', linewidth=1.5,label=f'Baseline Moyenne : {mean_baseline:.4f}')

    plt.xlabel('Batch Index')
    plt.ylabel('MSE Loss')
    plt.title('Train Loss vs Baseline per Batch - Epoch 1')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(outdir, "epoch1_batch_loss.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"-> Plot de la loss par batch pour l'époque 1 sauvegardé dans {plot_path}")

def loss_acc_first_epoch(batch_losses, baseline_losses, batch_accs, baseline_accs, outdir):
    """
    Trace et sauvegarde l'évolution de la loss ET de l'accuracy par batch 
    lors de la première époque, comparées à une baseline de classification.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot de la Loss ---
    ax1.plot(batch_losses, label='Train Loss (ViT)', color='blue')
    ax1.plot(baseline_losses, label='Baseline Loss (Priors)', color='red', linestyle='--', alpha=0.5)

    # NOUVEAU : Ligne horizontale pour la moyenne de la Baseline Loss
    mean_b_loss = np.mean(baseline_losses)
    ax1.axhline(y=mean_b_loss, color='red', linestyle='--', linewidth=1.5,label=f'Baseline Loss Moy : {mean_b_loss:.4f}')

    ax1.set_xlabel('Batch Index')
    ax1.set_ylabel('CrossEntropy Loss')
    ax1.set_title('Train Loss vs Baseline - Epoch 1')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Plot de l'Accuracy ---
    ax2.plot(batch_accs, label='Train Accuracy (ViT)', color='green')
    ax2.plot(baseline_accs, label='Baseline Acc (Majority Class)', color='orange', linestyle='--', alpha=0.5)

    # NOUVEAU : Ligne horizontale pour la moyenne de la Baseline Accuracy
    mean_b_acc = np.mean(baseline_accs)
    ax2.axhline(y=mean_b_acc, color='orange', linestyle='--', linewidth=1.5,label=f'Baseline Acc Moy : {mean_b_acc:.2f}%')
    ax2.set_xlabel('Batch Index')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Train Accuracy vs Baseline - Epoch 1')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(outdir, "epoch1_batch_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"-> Plot des métriques (Loss & Acc) par batch pour l'époque 1 sauvegardé dans {plot_path}")

def plot_and_save_maps(slp_true_list, slp_pred_list, time_list, outdir, epoch=None, fixed_indices=[100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000], duree_moyennage = 1):
    """
    Génère une figure pour des indices FIXES afin de voir l'évolution du modèle sur les mêmes dates.
    Normalement tant qu'on ne cherche pas plus loin que l'indice 9000 on ne risque pas de mélanger le premier membre du validation dataset avec la deuxième s'il y en a un
    Pas trop d'intérêt d'en mettre un deuxième surtout que ça va faire exploser la RAM qui stocke tout pour les tests de validations
    """
    y_true = np.concatenate(slp_true_list, axis=0)
    y_pred = np.concatenate(slp_pred_list, axis=0)

    if y_true.ndim == 4:
        y_true = y_true.squeeze(1)
    if y_pred.ndim == 4:
        y_pred = y_pred.squeeze(1)

    vmax_fixed = 2.

    N = y_true.shape[0]
    half_w = duree_moyennage // 2

    # On définit notre tolérance temporelle sous forme d'objet timedelta
    max_allowed_delta = timedelta(days=half_w)
    
    # On s'assure que les indices demandés ne dépassent pas la taille du dataset
    valid_indices = [idx for idx in fixed_indices if idx < N]
    num_samples = len(valid_indices)

    if num_samples == 0:
        print("Erreur : Aucun indice valide pour le plot.")
        return

    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4 * num_samples),subplot_kw={'projection': ccrs.PlateCarree()})
    title_suffix = f" (Lissage +/- {half_w}j)" if duree_moyennage > 1 else ""
    # Titre global avec l'époque
    epoch_str = f" (Epoch {epoch})" if epoch is not None else " (Final)"
    fig.suptitle(f"Évolution des prédictions SLP {epoch_str}{title_suffix}", fontsize=16)

    for i, idx in enumerate(valid_indices):
        target_date = time_list[idx]
        valid_window_indices = []

        for j in range(max(0, idx - half_w), min(N, idx + half_w + 1)):
            current_date = time_list[j]
            
            # On utilise strptime avec TON format exact '%Y-%m-%d'
            t_date = datetime.strptime(target_date, '%Y-%m-%d') if isinstance(target_date, str) else target_date
            c_date = datetime.strptime(current_date, '%Y-%m-%d') if isinstance(current_date, str) else current_date
            
            if abs(t_date - c_date) <= max_allowed_delta:
                valid_window_indices.append(j)

        # On moyenne uniquement sur les jours qui ont passé le test

        true_map = np.mean(y_true[valid_window_indices], axis=0)
        pred_map = np.mean(y_pred[valid_window_indices], axis=0)
        
        # Formatage propre de la date
        if hasattr(target_date, 'strftime'):
            date_str = target_date.strftime('%Y-%m-%d')
        else:
            date_str = str(target_date)

        ax_row = axes[i] if num_samples > 1 else axes

        # VRAIE SLP
        im1 = ax_row[0].imshow(true_map, cmap='RdBu_r', origin='lower', vmin=-vmax_fixed, vmax=vmax_fixed, transform = ccrs.PlateCarree(), extent=extent_slp)
        ax_row[0].set_title(f"Vraie SLP - {date_str} (Moy. sur {len(valid_window_indices)}j)")
        fig.colorbar(im1, ax=ax_row[0], fraction=0.046, pad=0.04)
        ax_row[0].coastlines()
        ax_row[0].set_extent(extent_slp, crs=ccrs.PlateCarree())


        # SLP PREDITE
        im2 = ax_row[1].imshow(pred_map, cmap='RdBu_r', origin='lower', vmin=-vmax_fixed, vmax=vmax_fixed, transform = ccrs.PlateCarree(), extent=extent_slp)
        ax_row[1].set_title(f"Prédite - {date_str} (Moy. sur {len(valid_window_indices)}j)")
        fig.colorbar(im2, ax=ax_row[1], fraction=0.046, pad=0.04)
        ax_row[1].coastlines()
        ax_row[1].set_extent(extent_slp, crs=ccrs.PlateCarree())

    plt.tight_layout()
    
    # On nomme le fichier avec l'époque pour ne pas écraser les précédents
    filename = f"val_maps_epoch_{epoch}_duree_moyennage_{duree_moyennage}.png" if epoch is not None else "val_maps_final.png"
    save_path = os.path.join(outdir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_and_save_maps_3_columns(slp_true_maps_list, pc_true_list, pc_pred_list, time_list, eof1_map, outdir, epoch=None, fixed_indices=[100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000], duree_moyennage=1):
    """
    Génère une figure à 3 colonnes : Vraie SLP | Reconstruite (Vraie PC) | Reconstruite (PC Prédite)
    A supprimer à terme et remplacer par la fonction plot_and_save_maps_with_reconstruction qui fait la même chose mais avec les cartes reconstruites via l'autoencodeur (PCA ou VAE) et pas juste via la PC1.
    """
    # 1. On concatène tout
    slp_true_maps = np.concatenate(slp_true_maps_list, axis=0).squeeze() # (N, H, W)
    pc_true = np.concatenate(pc_true_list, axis=0).squeeze()             # (N,)
    pc_pred_quantiles = np.concatenate(pc_pred_list, axis=0).squeeze()   # (N, 10)

    # 2. On extrait la médiane de la prédiction (Quantiles 0.45 et 0.55 -> indices 4 et 5)
    pc_pred_median = (pc_pred_quantiles[:, 4] + pc_pred_quantiles[:, 5]) / 2.0

    N = slp_true_maps.shape[0]
    half_w = duree_moyennage // 2
    
    valid_indices = [idx for idx in fixed_indices if idx < N]
    num_samples = len(valid_indices)

    if num_samples == 0:
        print("Erreur : Aucun indice valide pour le plot.")
        return

    # Figure à 3 colonnes
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4 * num_samples),subplot_kw={'projection': ccrs.PlateCarree()})
    
    title_suffix = f" (Lissage +/- {half_w}j)" if duree_moyennage > 1 else ""
    epoch_str = f" (Epoch {epoch})" if epoch is not None else " (Final)"
    fig.suptitle(f"Évaluation des prédictions (PC1) {epoch_str}{title_suffix}", fontsize=18, y=0.98)

    for i, idx in enumerate(valid_indices):
        start = max(0, idx - half_w)
        end = min(N, idx + half_w + 1)

        # --- A. CALCUL DES MOYENNES (Lissage) ---
        true_map_smoothed = np.mean(slp_true_maps[start:end], axis=0)
        mean_pc_true = np.mean(pc_true[start:end])
        mean_pc_pred = np.mean(pc_pred_median[start:end])

        # --- B. RECONSTRUCTION DES CARTES VIA L'EOF1 ---
        # Carte = PC * EOF
        recon_true_map = mean_pc_true * eof1_map
        recon_pred_map = mean_pc_pred * eof1_map

        # --- C. AFFICHAGE ---
        date = time_list[idx]
        
        # On définit une échelle commune pour les 3 cartes pour que les couleurs aient le même sens
        vmin = -2.
        vmax = 2.

        ax_row = axes[i] if num_samples > 1 else axes

        # Colonne 1 : La Vraie Carte (Complète)
        im1 = ax_row[0].imshow(true_map_smoothed, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_row[0].set_title(f"1. Vraie SLP Totale\n{date}")
        ax_row[0].coastlines()
        ax_row[0].set_extent(extent_slp, crs=ccrs.PlateCarree())
        fig.colorbar(im1, ax=ax_row[0], fraction=0.046, pad=0.04)

        # Colonne 2 : Reconstruction via la Vraie PC1
        im2 = ax_row[1].imshow(recon_true_map, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_row[1].set_title(f"2. Recon. (Vraie PC1 : {mean_pc_true:.2f})\nCompression PCA")
        fig.colorbar(im2, ax=ax_row[1], fraction=0.046, pad=0.04)
        ax_row[1].coastlines()
        ax_row[1].set_extent(extent_slp, crs=ccrs.PlateCarree())

        # Colonne 3 : Reconstruction via la PC1 Prédite
        im3 = ax_row[2].imshow(recon_pred_map, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_row[2].set_title(f"3. Recon. (PC1 Prédite : {mean_pc_pred:.2f})\nPrédiction Modèle")
        ax_row[2].coastlines()
        ax_row[2].set_extent(extent_slp, crs=ccrs.PlateCarree())

        fig.colorbar(im3, ax=ax_row[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"val_reconstruction_epoch_{epoch}_duree_moyennage_{duree_moyennage}.png" if epoch is not None else "val_reconstruction_final.png"
    plt.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close()

def plot_and_save_maps_with_reconstruction(slp_true_list, slp_recon_true_list, slp_pred_list, time_list, outdir, epoch=None, fixed_indices=[100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000], duree_moyennage=1):
    """
    Génère une figure à 3 colonnes : 
    1. Vraie SLP | 2. Vraie SLP encodée/décodée (PCA/VAE) | 3. SLP Prédite par le ViT
    Attention au moyennage qui pourrait être à cheval sur différentes années
    """
    # 1. On concatène et on enlève les dimensions superflues (ex: channel)
    slp_true = np.concatenate(slp_true_list, axis=0)      
    slp_recon_true = np.concatenate(slp_recon_true_list, axis=0)
    slp_pred = np.concatenate(slp_pred_list, axis=0)

    if slp_true.ndim == 4:
        slp_true = slp_true.squeeze(1)
    if slp_recon_true.ndim == 4:
        slp_recon_true = slp_recon_true.squeeze(1)
    if slp_pred.ndim == 4:
        slp_pred = slp_pred.squeeze(1)

    N = slp_true.shape[0]
    half_w = duree_moyennage // 2
    
    # Ne garder que les indices qui existent dans le batch de validation
    valid_indices = [idx for idx in fixed_indices if idx < N]
    num_samples = len(valid_indices)

    if num_samples == 0:
        print("Erreur : Aucun indice valide pour le plot.")
        return

    # Figure : 3 colonnes, num_samples lignes
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4 * num_samples), subplot_kw={'projection': ccrs.PlateCarree()})
    
    title_suffix = f" (Lissage +/- {half_w}j)" if duree_moyennage > 1 else ""
    epoch_str = f" (Epoch {epoch})" if epoch is not None else " (Final)"
    fig.suptitle(f"Reconstruction de la SLP via Embedding latent {epoch_str}{title_suffix}", fontsize=18, y=0.98)

    for i, idx in enumerate(valid_indices):
        start = max(0, idx - half_w)
        end = min(N, idx + half_w + 1)

        # --- CALCUL DES MOYENNES (Lissage temporel) ---
        map_true = np.mean(slp_true[start:end], axis=0)
        map_recon_true = np.mean(slp_recon_true[start:end], axis=0)
        map_pred = np.mean(slp_pred[start:end], axis=0)

        date = time_list[idx]
        
        # On définit une échelle commune pour les 3 cartes
        vmin = -2.0
        vmax = 2.0

        ax_row = axes[i] if num_samples > 1 else axes

        # Colonne 1 : La Réalité
        im1 = ax_row[0].imshow(map_true, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_row[0].set_title(f"1. Vraie SLP\n{date}")
        ax_row[0].coastlines()
        fig.colorbar(im1, ax=ax_row[0], fraction=0.046, pad=0.04)

        # Colonne 2 : La Réalité passée dans le PCA/VAE (Target Idéale)
        im2 = ax_row[1].imshow(map_recon_true, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_row[1].set_title(f"2. Auto-Encodage (Plafond de verre)\n(Realité -> Embed -> Decode)")
        ax_row[1].coastlines()
        fig.colorbar(im2, ax=ax_row[1], fraction=0.046, pad=0.04)

        # Colonne 3 : La Prédiction du ViT
        im3 = ax_row[2].imshow(map_pred, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_row[2].set_title(f"3. Prédiction Modèle\n(SST -> ViT -> Embed -> Decode)")
        ax_row[2].coastlines()
        fig.colorbar(im3, ax=ax_row[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"val_maps_epoch_{epoch}.png" if epoch is not None else "val_maps_final.png"
    plt.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close()


def plot_and_save_maps_with_reconstruction_light(slp_true_list, slp_recon_true_list, slp_pred_list, time_list, outdir, epoch=None, max_plots=30):
    """
    Génère une figure à 3 colonnes : 
    1. Vraie SLP | 2. Vraie SLP encodée/décodée (PCA/VAE) | 3. SLP Prédite par le ViT
    Version allégée (sans moyennage) pour le monitoring d'entraînement.
    On plot toutes les listes données jsuqu'à 30 lignes donc à utiliser en sélectionnant les indices dans la boucle d'entrainement, plus de moyennage après cout possible (équivalent à duree_moyennage = 1 dans la fonction précédente)
    """
    # 1. On concatène
    slp_true = np.concatenate(slp_true_list, axis=0)      
    slp_recon_true = np.concatenate(slp_recon_true_list, axis=0)
    slp_pred = np.concatenate(slp_pred_list, axis=0)

    # Nettoyage des dimensions (ex: [batch, 1, H, W] -> [batch, H, W])
    if slp_true.ndim == 4: slp_true = slp_true.squeeze(1)
    if slp_recon_true.ndim == 4: slp_recon_true = slp_recon_true.squeeze(1)
    if slp_pred.ndim == 4: slp_pred = slp_pred.squeeze(1)

    # 2. Déterminer combien de cartes on va tracer
    N = slp_true.shape[0]
    num_samples = min(N, max_plots)

    if num_samples == 0:
        print("Erreur : Aucun sample valide pour le plot.")
        return

    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4 * num_samples), subplot_kw={'projection': ccrs.PlateCarree()})
    
    epoch_str = f" (Epoch {epoch})" if epoch is not None else " (Final)"
    fig.suptitle(f"Reconstruction de la SLP via Embedding latent {epoch_str}", fontsize=18, y=0.98)

    # Gestion propre si on n'a qu'une seule ligne (1 seul plot)
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        date = time_list[i]
        vmin, vmax = -2.0, 2.0
        ax_row = axes[i]

        # Colonne 1 : La Réalité
        im1 = ax_row[0].imshow(slp_true[i], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_row[0].set_title(f"1. Vraie SLP\n{date}")
        ax_row[0].coastlines()
        fig.colorbar(im1, ax=ax_row[0], fraction=0.046, pad=0.04)

        # Colonne 2 : Plafond de verre (PCA/VAE)
        im2 = ax_row[1].imshow(slp_recon_true[i], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_row[1].set_title(f"2. Auto-Encodage (Plafond de verre)\n(Realité -> Embed -> Decode)")
        ax_row[1].coastlines()
        fig.colorbar(im2, ax=ax_row[1], fraction=0.046, pad=0.04)

        # Colonne 3 : Prédiction du ViT
        im3 = ax_row[2].imshow(slp_pred[i], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        ax_row[2].set_title(f"3. Prédiction Modèle\n(SST -> ViT -> Embed -> Decode)")
        ax_row[2].coastlines()
        fig.colorbar(im3, ax=ax_row[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"val_maps_epoch_{epoch}.png" if epoch is not None else "val_maps_final.png"
    plt.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close()



def plot_reconstruction_check(true_maps, recon_maps, dates, outdir, method_name, num_samples=3):
    """
    Trace num_samples cartes originales vs reconstruites pour vérifier l'embedder.
    Marche pour la PCA comme pour le VAE puisque de toute façon on donne les matrices reconstruites en input (juste de la visualisation).
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f"Sanity Check: Reconstruction via {method_name.upper()}", fontsize=14)
    
    for i in range(num_samples):
        true_m = true_maps[i].squeeze()
        recon_m = recon_maps[i].squeeze()
        
        vmin = min(true_m.min(), recon_m.min())
        vmax = max(true_m.max(), recon_m.max())
        
        # Carte Vraie
        im1 = axes[i, 0].imshow(true_m, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        axes[i, 0].set_title(f"Originale ({dates[i]})")
        axes[i, 0].coastlines()
        axes[i, 0].set_extent(extent_slp, crs=ccrs.PlateCarree())
        fig.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Carte Reconstruite
        im2 = axes[i, 1].imshow(recon_m, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent_slp)
        axes[i, 1].set_title(f"Reconstruite")
        axes[i, 1].coastlines()
        axes[i, 1].set_extent(extent_slp, crs=ccrs.PlateCarree())
        fig.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"check_reconstruction_{method_name}.png"), dpi=150)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, outdir, master_ref, filename='confusion_matrix.png'):
    
    # --- CALCUL DE L'ACCURACY ---
    # Sécurité: conversion en listes/numpy arrays d'une dimension
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) > 0:
        accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
    else:
        accuracy = 0.0

    # --- EXTRACTION DYNAMIQUE DES LABELS ---
    # On récupère les mêmes clés que dans get_fast_labels pour garantir l'ordre exact !
    target_keys = sorted([k for k in master_ref.keys() if k.endswith("_slp_0_mean") and not k.startswith("GLOBAL")])
    
    class_names = []
    
    if len(target_keys) > 0:
        for key in target_keys:
            # 1. On enlève le suffixe
            clean_name = key.replace("_slp_0_mean", "")
            
            # 2. Si c'est un régime (ex: "regime_1_NAO+"), on ne garde que "NAO+" pour faire joli
            if clean_name.startswith("regime_"):
                parts = clean_name.split('_')
                if len(parts) >= 3:
                    clean_name = parts[2] 
            
            class_names.append(clean_name)
    elif 'pc1_bins' in master_ref:
        # Fallback au cas où on utilise pc1_quantiles sans les matrices _slp_0_mean
        num_classes = len(master_ref['pc1_bins']) - 1
        class_names = [f"Q{i+1}" for i in range(num_classes)]
    else:
        print("Avertissement : Impossible de trouver les noms de classes dans master_ref. Utilisation de labels génériques. Pas très efficace en plus je crois")
        # Fallback de sécurité extrême
        num_classes = max(max(y_true, default=0), max(y_pred, default=0)) + 1
        class_names = [f"Class {i}" for i in range(int(num_classes))]

    # --- CALCUL DE LA MATRICE ---
    # On utilise range(len(class_names)) pour s'adapter à 4, 8, ou X classes automatiquement
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # On agrandit un peu la figure pour que les 8 classes rentrent bien
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Vraie Classe (Réalité)')
    plt.xlabel('Classe Prédite (Modèle)')
    
    # CORRECTION : f-string ajouté !
    plt.title(f'Matrice de Confusion \nAccuracy : {accuracy:.2f}%', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=200)
    plt.close()

def old_plot_confusion_matrix(y_true, y_pred, outdir, master_ref, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # --- CALCUL DE L'ACCURACY ---
    accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true) * 100

    # --- EXTRACTION DYNAMIQUE DES LABELS ---
    # On va chercher les noms directement dans les clés du dictionnaire
    class_names = ["", "", "", ""]
    for key in master_ref.keys():
        if key.startswith("regime_") and key.endswith("_slp_0"):
            # Exemple de key : 'regime_1_NAO+_slp_0'
            parts = key.split('_')
            # L'index du régime (1, 2, 3 ou 4) est à la position 1.
            # On fait -1 car les index des listes Python commencent à 0.
            regime_idx = int(parts[1]) - 1 
            # Le nom (NAO+, AR...) est à la position 2
            regime_name = parts[2] 
            class_names[regime_idx] = regime_name
            
    # Remplacement au cas où un nom serait vide (sécurité)
    class_names = [name if name else f"Regime {i+1}" for i, name in enumerate(class_names)]
    # ---------------------------------------

    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Vraie Classe (Réalité)')
    plt.xlabel('Classe Prédite (Modèle)')
    plt.title('Matrice de Confusion \nAccuracy : {accuracy:.2f}%', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=200)
    plt.close()