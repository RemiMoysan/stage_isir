import xarray as xr
import pandas as pd
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import joblib

# ============================================================
# 1. FONCTIONS ET CLASSES (Préparation, Visu)
# ============================================================

def load_member_data(member, file_path_SLP, selected_months, slp_std=596.0, duree_lissage = 10):
    """
    Charge un seul membre, le filtre sur les mois d'hiver, 
    gère les NaNs et aplatit les cartes pour la PCA.
    """
    ds = xr.open_dataset(os.path.join(file_path_SLP, f'PSL_anom_LE2-{member}_{duree_lissage}d.nc'))
    
    da = ds["PSL"]
        
    da = da.sel(time=da['time'].dt.month.isin(selected_months))
    dates = da.time.values
    
    # Récupération des données brutes
    data = da.values
    
    # Dimensions (Time, Lat, Lon) -> (Time, 53, 113)
    n_samples, h, w = data.shape
    
    # Remplacement des NaNs par 0 et normalisation (comme dans le VAE)
    data = np.nan_to_num(data, nan=0.0) / slp_std
    
    # Aplatissement spatial pour la PCA : (Time, 53*113)
    data_flat = data.reshape(n_samples, h * w)
    
    return data_flat, dates, (h, w)

def plot_pca_reconstructions(y_true_flat, y_recon_flat, time_list, shape_2d, outdir, latent_dim, fixed_indices=[100, 1000, 2000,3000,4000,4500,5000,6000,7000, 8000]):
    """Trace les vraies SLP vs les reconstructions de la PCA"""
    
    N = y_true_flat.shape[0]
    valid_indices = [idx for idx in fixed_indices if idx < N]
    num_samples = len(valid_indices)
    
    if num_samples == 0: return

    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4 * num_samples))
    fig.suptitle(f"PCA Reconstructions (Latent Dim = {latent_dim})", fontsize=16)

    h, w = shape_2d

    for i, idx in enumerate(valid_indices):
        # On reshape les vecteurs 1D en cartes 2D (53, 113)
        true_map = y_true_flat[idx].reshape(h, w)
        recon_map = y_recon_flat[idx].reshape(h, w)
        
        # Formatage de la date (gère les cftime climatiques)
        date_item = time_list[idx]
        if hasattr(date_item, 'strftime'):
            date_str = date_item.strftime('%Y-%m-%d')
        else:
            date_str = str(date_item)[:10] # Fallback (AAAA-MM-JJ)

        vmin, vmax = -2, 2
        ax_row = axes[i] if num_samples > 1 else axes

        # Vraie SLP
        im1 = ax_row[0].imshow(true_map, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        ax_row[0].set_title(f"Vraie SLP - {date_str}")
        fig.colorbar(im1, ax=ax_row[0], fraction=0.046, pad=0.04)

        # SLP Reconstruite
        im2 = ax_row[1].imshow(recon_map, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        ax_row[1].set_title(f"Reconstruction PCA (Latent Dim = {latent_dim})")
        fig.colorbar(im2, ax=ax_row[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_reconstructions.png"), dpi=150)
    plt.close()

def plot_explained_variance(ipca, outdir):
    """
    Remplace la courbe de 'loss'. Affiche la variance expliquée cumulative.
    """
    plt.figure(figsize=(8, 5))
    cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
    
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', markersize=4)
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Expliquée')
    plt.xlabel('Nombre de composantes principales (Latent Dim)')
    plt.ylabel('Variance expliquée cumulative')
    plt.title('Variance expliquée par la PCA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'Fig_explained_variance.png'))
    plt.close()


def plot_eof_patterns(ipca, shape_2d, outdir, n_eofs=10):
    """
    Affiche les n_eofs premières structures spatiales (EOFs).
    """
    h, w = shape_2d
    # Les EOFs sont dans ipca.components_ (shape: n_components, n_features)

    # On s'assure de ne pas demander plus d'EOFs qu'il n'y a de composantes calculées
    n_eofs = min(n_eofs, ipca.components_.shape[0])

    eofs = ipca.components_[:n_eofs]
    # Variance expliquée pour les titres
    var_exp = ipca.explained_variance_ratio_[:n_eofs] * 100

    rows = (n_eofs + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
    fig.suptitle(f"Top {n_eofs} EOF Patterns (Spatial Structures)", fontsize=20)

    axes = axes.flatten()

    for i in range(n_eofs):
        eof_map = eofs[i].reshape(h, w)
        
        # On centre l'échelle de couleur sur 0
        vlim = np.max(np.abs(eof_map)) 
        
        im = axes[i].imshow(eof_map, cmap='RdBu_r', origin='lower', vmin=-vlim, vmax=vlim)
        axes[i].set_title(f"EOF {i+1} ({var_exp[i]:.1f}% var. expliquée)")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Si n_eofs est impair, on cache le dernier axe vide
    if n_eofs % 2 != 0:
        axes[-1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(outdir, f"top_{n_eofs}_eofs.png"), dpi=150)
    plt.close()
    print(f"--> EOFs sauvegardées dans {outdir}")

# ============================================================
# 2. CONFIGURATION GÉNÉRALE
# ============================================================

if __name__ == "__main__":
    


    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch',"mac_local"])
    parser.add_argument('--normalize', action='store_true', help='Si on veut normaliser les PC, à omettre si on ne veut pas les normaliser')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours (10 ou 30)')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension de l\'espace latent (nombre de composantes principales)')
    args = parser.parse_args()

    # Hyperparamètre principal (équivalent de la taille du bottleneck du VAE)
    latent_dim = args.latent_dim
    normalize = args.normalize
    duree_lissage = args.duree_lissage

    train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']


    nb_members_train = 87
    train_members = train_members_87[:nb_members_train]
    val_members = ['1001.001']

    winter_months = [11, 12, 1, 2] # NDJF
    months_label = "NDJF"      

    if args.machine == 'hacienda':
        path_slp = "/data/moysan/data/SLP/"
        base_home = "/home/moysan/stage_isir_jz/data_analysis/pca_slp/"
    elif args.machine == 'jean-zay-work':
        path_slp = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/"
    elif args.machine == 'jean-zay-scratch':
        path_slp = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/SLP/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/"
    else: # mac_local
        path_slp = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SLP/"
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/pca_slp/"

    outdir_name = f'IPCA_latent{latent_dim}_{months_label}_{nb_members_train}members_normalize{normalize}_duree_lissage{duree_lissage}'
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    # ============================================================
    # 3. INITIALISATION DU MODÈLE PCA
    # ============================================================
    # IncrementalPCA permet de fitter le modèle chunk par chunk (membre par membre)
    # sans saturer la RAM.
    ipca = IncrementalPCA(n_components=latent_dim,whiten=normalize)
    
    print(f"Initialisation de l'Incremental PCA avec latent_dim = {latent_dim}")

    # ============================================================
    # 4. "TRAINING" LOOP (Ajustement de la PCA)
    # ============================================================
    start_time = time.time() 
    shape_2d = None

    print("\n--- Début du fit (Entraînement) ---")
    for i, member in enumerate(train_members):
        print(f"Chargement et fit du membre {member} ({i+1}/{nb_members_train})...")
        
        # On charge les données d'un seul membre
        X_train_chunk, _, shape_2d = load_member_data(member, path_slp, winter_months, duree_lissage=duree_lissage)
        
        # On met à jour la PCA avec ce membre
        ipca.partial_fit(X_train_chunk)

    print(f"Fit terminé. Variance totale expliquée par {latent_dim} composantes : {np.sum(ipca.explained_variance_ratio_)*100:.2f}%")
    plot_explained_variance(ipca, outdir)

    # NOUVEAU : Plot des EOFs
    plot_eof_patterns(ipca, shape_2d, outdir, n_eofs=10)

    # SAUVEGARDE DU MODÈLE LÉGER
    model_path = os.path.join(outdir, 'best_pca_model.joblib')
    joblib.dump(ipca, model_path)
    print(f"--> Modèle PCA sauvegardé sous : {model_path}")
    print("    (Tu pourras le recharger avec : pca = joblib.load('best_pca_model.joblib'))")

    # ============================================================
    # 5. VALIDATION & RECONSTRUCTION
    # ============================================================
    print("\n--- Début de la Validation (Reconstruction) ---")
    
    val_true_list = []
    val_recon_list = []
    val_dates_list = []

    for member in val_members:
        print(f"Évaluation sur le membre de validation : {member}")
        X_val, dates_val, _ = load_member_data(member, path_slp, winter_months, duree_lissage=duree_lissage)
        
        # Encodage (Projection dans l'espace latent) : shape (N, 128)
        latent_vectors = ipca.transform(X_val)
        
        # Décodage (Reconstruction depuis l'espace latent) : shape (N, 53*113)
        X_val_recon = ipca.inverse_transform(latent_vectors)
        
        # Calcul de la MSE pour comparer avec le VAE
        mse = np.mean((X_val - X_val_recon)**2)
        print(f"MSE de reconstruction sur la validation : {mse:.4f}")

        val_true_list.append(X_val)
        val_recon_list.append(X_val_recon)
        val_dates_list.extend(dates_val)

    # Concaténation pour le plot
    y_true_all = np.concatenate(val_true_list, axis=0)
    y_recon_all = np.concatenate(val_recon_list, axis=0)

    # Plot des reconstructions
    plot_pca_reconstructions(y_true_all, y_recon_all, val_dates_list, shape_2d, outdir, latent_dim=latent_dim)

    # 1. Récupération directe (zéro temps de calcul)
    variance_par_composante = ipca.explained_variance_ratio_
    variance_cumulative = np.cumsum(variance_par_composante)

    # 2. Création de la figure
    plt.figure(figsize=(10, 6))

    # Courbe 1 : La variance expliquée cumulée (en rouge)
    plt.plot(range(1, latent_dim + 1), variance_cumulative, 
            label='Variance cumulative', color='red', marker='o', markersize=3)

    # Courbe 2 (Barres) : La variance expliquée par chaque composante (en bleu)
    plt.bar(range(1, latent_dim + 1), variance_par_composante, 
            alpha=0.5, align='center', label='Variance par composante')

    # Ligne de repère (ex: objectif de 90% d'information gardée)
    plt.axhline(y=0.90, color='black', linestyle='--', label='Seuil de 90%')

    plt.xlabel('Nombre de composantes principales (Latent Dim)')
    plt.ylabel('Ratio de variance expliquée')
    plt.title('Évolution de la variance expliquée en fonction du nombre de composantes')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'variance_expliquee.png'), dpi=150)
    plt.close()

    elapsed_time = (time.time() - start_time) / 60
    print(f"\nExécution complète en {elapsed_time:.2f} minutes.")