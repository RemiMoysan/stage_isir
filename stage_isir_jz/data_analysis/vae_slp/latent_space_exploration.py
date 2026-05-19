import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import argparse
import os
import sys
from pathlib import Path
import cartopy.crs as ccrs

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from vae_on_slp import ConvVAE, Dataset_SLP_VAE

def explore_latent_space(model_path, dataloader, outdir, latent_dim=1, z_idx=0, device='cpu', num_steps=10,sampling_type='linear'):
    """
    1. Charge le modèle pré-entraîné.
    2. Calcule la distribution empirique de la composante latente `z_idx` et la sauvegarde.
    3. Génère une grille d'images (AVEC COASTLINES) balayant l'intervalle latent (Format 2D).
    4. Génère un GIF animé (AVEC COASTLINES) balayant l'intervalle (Ralenti).
    """
    print(f"Chargement du modèle depuis {model_path}...")
    model = ConvVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(outdir, exist_ok=True)
    extent_slp = [-100, 40, 20, 70] # Ta zone géographique

    # ==========================================
    # 1. Extraction de la distribution empirique
    # ==========================================
    print("Extraction des variables latentes sur le dataset...")
    mu_list = []
    
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            inputs = inputs.to(device)
            mu, _ = model.encode(inputs)
            mu_list.append(mu.cpu().numpy())
            
    all_mus = np.concatenate(mu_list, axis=0)
    z_dist = all_mus[:, z_idx]
    
    z_mean, z_std = np.mean(z_dist), np.std(z_dist)
    z_min, z_max = np.min(z_dist), np.max(z_dist)
    
    print(f"Composante latente {z_idx} : Moyenne = {z_mean:.3f}, Std = {z_std:.3f}")
    print(f"Intervalle balayé : [{z_min:.3f}, {z_max:.3f}]")

    # ==========================================
    # 2. Sauvegarde de la distribution (Histogramme)
    # ==========================================
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(z_dist, bins=300, density=True, alpha=0.6, color='b', label='Distribution empirique ($\mu$)')
    
    x_axis = np.linspace(z_min, z_max, 100)
    ax_hist.plot(x_axis, norm.pdf(x_axis, 0, 1), 'r--', lw=2, label='Prior cible $\mathcal{N}(0, 1)$')
    
    ax_hist.set_title(f"Distribution de la composante latente {z_idx}")
    ax_hist.set_xlabel("Valeur latente $z$")
    ax_hist.set_ylabel("Densité")
    ax_hist.legend()
    
    hist_path = os.path.join(outdir, f"distribution_z{z_idx}.png")
    plt.savefig(hist_path, dpi=150)
    plt.close(fig_hist)
    print(f"--> Histogramme sauvegardé : {hist_path}")

    # ==========================================
    # 3. Détermination des valeurs de z pour l'échantillonnage
    # ==========================================
    base_z = np.mean(all_mus, axis=0) 
    
    if sampling_type == 'gaussian':
        print(f"Échantillonnage de la grille : Densité GAUSSIENNE sur N(mu_emp, std_emp)")
        # On échantillonne linéairement les probabilités cumulées entre 1% et 99%
        # pour éviter d'aller chercher des valeurs aberrantes à +/- l'infini.
        probs = np.linspace(0.01, 0.99, num_steps)
        z_vals = norm.ppf(probs, loc=z_mean, scale=z_std)
    else:
        print(f"Échantillonnage de la grille : Linéaire entre {z_min:.2f} et {z_max:.2f}")
        z_vals = np.linspace(z_min, z_max, num_steps)


    # --- NOUVELLE LOGIQUE DE GRILLE ---
    ncols = 5 # Nombre maximum de colonnes par ligne
    nrows = int(np.ceil(num_steps / ncols))

    fig_grid, axes = plt.subplots(
        nrows, ncols, 
        figsize=(3.5 * ncols, 3.5 * nrows), # Taille ajustée dynamiquement
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Sécurité si jamais on demande 1 seule image (axes n'est pas un array dans ce cas)
    if num_steps == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten() # Permet d'itérer facilement même sur une grille 2D

    for i, val in enumerate(z_vals):
        current_z = base_z.copy()
        current_z[z_idx] = val
        
        z_tensor = torch.tensor(current_z).unsqueeze(0).float().to(device)
        with torch.no_grad():
            new_recon = model.decode(z_tensor).squeeze().cpu().numpy()

        ax = axes[i]
        im = ax.imshow(
            new_recon, cmap='RdBu_r', origin='lower', vmin=-2, vmax=2, 
            transform=ccrs.PlateCarree(), extent=extent_slp
        )
        ax.set_title(f"z = {val:.2f}", fontsize=12)
        ax.coastlines()
        ax.set_extent(extent_slp, crs=ccrs.PlateCarree())
        ax.axis('off')

    # On désactive les axes vides restants (si num_steps n'est pas un multiple de ncols)
    for j in range(len(z_vals), len(axes)):
        axes[j].axis('off')

    # Une seule colorbar pour toute la grille
    fig_grid.subplots_adjust(right=0.9, hspace=0.3, wspace=0.1)
    cbar_ax = fig_grid.add_axes([0.92, 0.15, 0.01, 0.7])
    fig_grid.colorbar(im, cax=cbar_ax)

    grid_path = os.path.join(outdir, f"traversal_grid_z{z_idx}.png")
    plt.savefig(grid_path, bbox_inches='tight', dpi=150)
    plt.close(fig_grid)
    print(f"--> Grille de traversée sauvegardée : {grid_path}")

    # ==========================================
    # 4. Génération d'un GIF (Animation fluide Cartographiée)
    # ==========================================
    print("Génération de l'animation GIF...")
    fig_anim, ax_anim = plt.subplots(figsize=(6, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Image vide pour initialiser l'animation
    im_anim = ax_anim.imshow(
        np.zeros((53, 113)), cmap='RdBu_r', origin='lower', vmin=-2, vmax=2,
        transform=ccrs.PlateCarree(), extent=extent_slp
    )
    ax_anim.coastlines()
    ax_anim.set_extent(extent_slp, crs=ccrs.PlateCarree())
    fig_anim.colorbar(im_anim, ax=ax_anim, fraction=0.046, pad=0.04)

    def update_frame(frame_val):
        current_z = base_z.copy()
        current_z[z_idx] = frame_val
        z_tensor = torch.tensor(current_z).unsqueeze(0).float().to(device)
        with torch.no_grad():
            new_recon = model.decode(z_tensor).squeeze().cpu().numpy()
        
        im_anim.set_data(new_recon)
        ax_anim.set_title(f"Reconstruction (z_{z_idx} = {frame_val:.2f})")
        return [im_anim]

    # Génération des frames selon le type d'échantillonnage choisi
    num_frames_half = 25
    if sampling_type == 'gaussian':
        probs_anim = np.linspace(0.01, 0.99, num_frames_half)
        frames_forward = norm.ppf(probs_anim, loc=z_mean, scale=z_std)
    else:
        frames_forward = np.linspace(z_min, z_max, num_frames_half)
        
    frames_backward = frames_forward[::-1] # Retour arrière
    z_frames = np.concatenate([frames_forward, frames_backward])

    ani = animation.FuncAnimation(fig_anim, update_frame, frames=z_frames, blit=False)
    
    gif_path = os.path.join(outdir, f"traversal_anim_z{z_idx}.gif")
    # --- MODIFICATION ICI : fps=4 pour ralentir l'animation ---
    ani.save(gif_path, writer='pillow', fps=4) 
    plt.close(fig_anim)
    print(f"--> Animation GIF sauvegardée : {gif_path}")


# ============================================================
# EXEMPLE D'UTILISATION
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch',"mac_local"])
    parser.add_argument('--number_of_members', type=int, default=1, help="Nombre de membres à analyser (ex: 10 pour les 10 premiers)")
    parser.add_argument('--z_idx', type=int, default=0, help="Index du paramètre latent à explorer")
    parser.add_argument('--model_name', type=str, default="ConvVAE_bs128_lr0.0001_NDJF_beta1.0_80members_latent1_duree30d", help="Nom du modèle à charger (doit correspondre au dossier dans base_home)")
    parser.add_argument('--duree_lissage', type=int, default=30, help="Durée de lissage des données SLP utilisées pour l'exploration (ex: 30 pour les données déjà lissées sur 30 jours)")
    parser.add_argument('--latent_dim', type=int, default=1, help="Dimension de l'espace latent du modèle (doit correspondre à celle utilisée lors de l'entraînement)")
    parser.add_argument('--sampling_type', type=str, default='linear', choices=['linear', 'gaussian'], help="Type d'espacement des échantillons dans l'espace latent")
    args = parser.parse_args()

    if args.machine == 'hacienda':
        path_slp = "/data/moysan/data/SLP/"
        base_home = "/home/moysan/stage_isir_jz/data_analysis/vae_slp/"
    elif args.machine == 'jean-zay-work':
        path_slp = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/vae_slp/"
    elif args.machine == 'jean-zay-scratch':
        path_slp = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/SLP/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/vae_slp/"
    else: # mac_local
        path_slp = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SLP/"
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/vae_slp/"

    model_name = args.model_name
    model_dir = os.path.join(base_home, model_name)
    model_path = os.path.join(model_dir, "best_vae.pth")
    
    # Création d'un dossier spécifique pour stocker les images d'exploration latente
    explore_outdir = os.path.join(model_dir, "latent_exploration")
    
    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    members_to_analyze = all_members[:args.number_of_members]

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")
    
    dataset_eval = Dataset_SLP_VAE(
        members=members_to_analyze, 
        selected_months=[11, 12, 1, 2], 
        file_path_SLP=path_slp, 
        duree_lissage=args.duree_lissage
    )
    
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=128, shuffle=False, num_workers=n_workers)
    
    explore_latent_space(
        model_path=model_path,
        dataloader=dataloader_eval,
        outdir=explore_outdir,
        latent_dim=args.latent_dim,
        z_idx=args.z_idx,
        device=device,
        num_steps=50, # Nombre d'images dans la grille
        sampling_type=args.sampling_type
    )