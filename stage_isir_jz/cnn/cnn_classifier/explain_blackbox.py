import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import torch

import sys
from pathlib import Path

# Importation des 3 algorithmes "Boîte Noire" de Captum
from captum.attr import Occlusion, KernelShap, Lime

project_root = Path(__file__).resolve().parent.parent.parent
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

from tools.datasets import Dataset
from tools_cnn.models import CNN_Classifier_Multimodal

def create_patch_feature_mask(shape, patch_size, device, start_id=0):
    """
    Crée un masque 2D où chaque patch de taille (patch_size x patch_size) 
    possède un ID unique. start_id permet de ne pas superposer les IDs 
    de la SST avec ceux de la SLP.
    """
    H, W = shape
    mask = torch.zeros((H, W), dtype=torch.long, device=device)
    patch_id = start_id
    
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            mask[i:i+patch_size, j:j+patch_size] = patch_id
            patch_id += 1
            
    return mask, patch_id

def plot_patch_explanations(attr_array, lags_days, extent, title_prefix, outdir, feature_name="SST"):
    num_lags = len(lags_days)
    fig, axes = plt.subplots(1, num_lags, figsize=(6 * num_lags, 4), subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
    if num_lags == 1: axes = [axes]

    vmax = np.max(np.abs(attr_array))
    vmin = -vmax

    for i, lag in enumerate(lags_days):
        ax = axes[i]
        ax.set_facecolor('white')
        ax.coastlines(resolution='110m', color='black', linewidth=0.8)
        
        im = ax.imshow(
            attr_array[i], 
            cmap='RdBu_r', 
            origin='lower', 
            vmin=vmin, 
            vmax=vmax, 
            transform=ccrs.PlateCarree(), 
            extent=extent,
            interpolation='nearest' 
        )
        
        ax.set_title(f"{feature_name} Lag {lag}d", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.08)

    plt.suptitle(title_prefix, fontsize=16, y=1.05)
    plt.tight_layout()
    
    filename = title_prefix.replace(" ", "_").replace("|", "").replace(":", "") + f"_{feature_name}.png"
    plt.savefig(os.path.join(outdir, filename), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--machine', type=str, default='jean-zay-work')
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=4)
    
    # Choix de la méthode
    parser.add_argument('--method', type=str, default='all', choices=['all', 'occlusion', 'kernelshap', 'lime'], help="Méthode à exécuter")
    
    # Paramètres d'explicabilité par patchs
    parser.add_argument('--patch_size', type=int, default=10, help='Taille du patch carré')
    parser.add_argument('--stride', type=int, default=5, help='Pas de glissement (uniquement pour Occlusion)')
    parser.add_argument('--n_samples', type=int, default=1000, help='Nombre d\'échantillons (uniquement pour KernelSHAP et LIME)')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. CHARGEMENT MODÈLE
    model = CNN_Classifier_Multimodal(
        num_classes=args.num_classes,
        in_chans_sst=len(args.sst_lags_days),
        in_chans_slp=len(args.slp_lags_days),
        n_feat=8,
        dr=0.0
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    # 2. CHARGEMENT D'UN PETIT BATCH DE TEST (Ces méthodes sont lourdes)
    test_set = Dataset(members=['1041.003'], selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=True)
    
    test_sst, test_slp, _, _, dates, _ = next(iter(test_loader))
    test_sst = test_sst.to(device)
    inputs = (test_sst,)
    
    # 3. PRÉPARATION DES MASQUES ET FENÊTRES (Communs aux méthodes)
    print(f"Préparation de la grille de patchs ({args.patch_size}x{args.patch_size})...")
    
    # - A. Pour KernelSHAP et LIME (Feature Masks)
    mask_sst_2d, next_id = create_patch_feature_mask((85, 360), args.patch_size, device, start_id=0)
    mask_sst = mask_sst_2d.unsqueeze(0).unsqueeze(0).expand(test_sst.shape[0], test_sst.shape[1], -1, -1)
    feature_mask = (mask_sst,)

    # - B. Pour Occlusion (Fenêtres glissantes)
    window_shapes = ((1, args.patch_size, args.patch_size),)
    strides = ((1, args.stride, args.stride),)

    if len(args.slp_lags_days) > 0:
        test_slp = test_slp.to(device)
        inputs = (test_sst, test_slp)
        
        # Masques LIME/KernelSHAP
        mask_slp_2d, _ = create_patch_feature_mask((53, 113), args.patch_size, device, start_id=next_id)
        mask_slp = mask_slp_2d.unsqueeze(0).unsqueeze(0).expand(test_slp.shape[0], test_slp.shape[1], -1, -1)
        feature_mask = (mask_sst, mask_slp)
        
        # Masques Occlusion
        window_shapes = ((1, args.patch_size, args.patch_size), (1, args.patch_size, args.patch_size))
        strides = ((1, args.stride, args.stride), (1, args.stride, args.stride))

    outdir = os.path.dirname(args.model_path)
    extent_sst = [-180, 180, -15, 70]
    extent_slp = [-100, 40, 20, 70] 

    # ============================================================
    # BOUCLE PRINCIPALE D'EXPLICABILITÉ
    # ============================================================

    methods_to_run = ['occlusion', 'kernelshap', 'lime'] if args.method == 'all' else [args.method]

    for method_name in methods_to_run:
        explain_dir = os.path.join(outdir, f"explain_{method_name}_patch{args.patch_size}")
        os.makedirs(explain_dir, exist_ok=True)
        print(f"\n{'='*40}\nLancement de {method_name.upper()}\n{'='*40}")

        # Instanciation de l'algorithme choisi
        if method_name == 'occlusion':
            explainer = Occlusion(model)
        elif method_name == 'kernelshap':
            explainer = KernelShap(model)
        elif method_name == 'lime':
            explainer = Lime(model)

        for c in range(args.num_classes):
            print(f"  -> Analyse du Régime {c+1}...")
            
            if method_name == 'occlusion':
                attributions = explainer.attribute(
                    inputs, target=c, baselines=0.0,
                    sliding_window_shapes=window_shapes, strides=strides
                )
            else:
                # KernelSHAP et LIME partagent exactement la même API fonctionnelle
                attributions = explainer.attribute(
                    inputs, target=c, baselines=0.0,
                    feature_mask=feature_mask, n_samples=args.n_samples
                )
            
            # Traitement SST
            attr_sst = attributions[0].cpu().numpy()
            mean_attr_sst = np.mean(attr_sst, axis=0) 
            plot_patch_explanations(mean_attr_sst, args.sst_lags_days, extent_sst, f"{method_name.upper()} Régime {c+1}", explain_dir, "SST")
            
            # Traitement SLP
            if len(args.slp_lags_days) > 0:
                attr_slp = attributions[1].cpu().numpy()
                mean_attr_slp = np.mean(attr_slp, axis=0)
                plot_patch_explanations(mean_attr_slp, args.slp_lags_days, extent_slp, f"{method_name.upper()} Régime {c+1}", explain_dir, "SLP")

    print("\nToutes les explications ont été générées avec succès !")