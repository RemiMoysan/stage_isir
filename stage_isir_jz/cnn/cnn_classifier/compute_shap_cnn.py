import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
import shap

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

# Ajouter le dossier "tools" de vision_transformer au sys.path pour tes imports de modèles
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)



from tools.datasets import Dataset
from tools_cnn.models import CNN_Classifier_Multimodal



def plot_mean_shap_maps(mean_shap_array, lags_days, extent, title_prefix, outdir, feature_name="SST"):
    """
    Génère les cartes SHAP moyennes pour chaque lag d'une feature donnée.
    mean_shap_array : shape (lags, H, W)
    """
    num_lags = len(lags_days)
    fig, axes = plt.subplots(1, num_lags, figsize=(6 * num_lags, 4), subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
    if num_lags == 1: axes = [axes]

    # Détermination d'une échelle symétrique globale pour tous les lags de cette classe
    vmax = np.max(np.abs(mean_shap_array))
    vmin = -vmax

    for i, lag in enumerate(lags_days):
        ax = axes[i]
        ax.set_facecolor('white')
        
        ax.coastlines(resolution='110m', color='black', linewidth=0.8)
        
        im = ax.imshow(
            mean_shap_array[i], 
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
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le fichier .pth du modèle')
    parser.add_argument('--machine', type=str, default='jean-zay-work', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=4, help='Nombre de régimes')
    
    # Paramètres spécifiques à SHAP
    parser.add_argument('--n_background', type=int, default=100, help='Nombre d\'échantillons pour le masque de fond')
    parser.add_argument('--n_test', type=int, default=300, help='Nombre d\'échantillons de validation à expliquer et moyenner')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")

    # ============================================================
    # 1. INITIALISATION ET CHARGEMENT DU MODÈLE
    # ============================================================
    model = CNN_Classifier_Multimodal(
        num_classes=args.num_classes,
        in_chans_sst=len(args.sst_lags_days),
        in_chans_slp=len(args.slp_lags_days),
        n_feat=8,
        dr=0.0 # Désactivé pour l'inférence
    ).to(device)

    print(f"Chargement des poids depuis : {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval() # Essentiel pour SHAP (désactive BatchNorm/Dropout)

    # ============================================================
    # 2. CHARGEMENT DES DONNÉES (BACKGROUND & TEST)
    # ============================================================
    # On utilise un membre arbitraire pour générer le background (le "bruit de fond" de l'océan) 
    # JE PENSE QU'ON POURRAIT CONSIDERER QUE C'EST 0 DE TOUTE FACON 
    bg_set = Dataset(members=['1001.001'], selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage)
    bg_loader = torch.utils.data.DataLoader(bg_set, batch_size=args.n_background, shuffle=True)
    
    # On utilise un autre membre pour extraire les prédictions à expliquer
    test_set = Dataset(members=['1041.003'], selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.n_test, shuffle=True)

    print("Extraction des données en cours...")
    bg_sst, bg_slp, _, _, _, _ = next(iter(bg_loader))
    test_sst, test_slp, _, _, dates, _ = next(iter(test_loader))

    bg_sst = bg_sst.to(device)
    test_sst = test_sst.to(device)
    background_data = [bg_sst]
    test_data = [test_sst]

    if len(args.slp_lags_days) > 0:
        bg_slp = bg_slp.to(device)
        test_slp = test_slp.to(device)
        background_data.append(bg_slp)
        test_data.append(test_slp)

    # ============================================================
    # 3. CALCUL DES SHAP VALUES (GRADIENT EXPLAINER)
    # ============================================================
    print(f"Instanciation de l'Explainer avec {args.n_background} échantillons de référence...")
    explainer = shap.GradientExplainer(model, background_data)

    print(f"Calcul des SHAP values sur {args.n_test} échantillons de test...")
    # Retourne une liste de taille [num_classes]. Chaque élément contient une liste [SST_shap_array, SLP_shap_array]
    shap_values = explainer.shap_values(test_data)

    # ============================================================
    # 4. MOYENNE ET VISUALISATION CARTOGRAPHIQUE
    # ============================================================
    outdir = os.path.dirname(args.model_path)
    shap_dir = os.path.join(outdir, f"shap_explanations_{args.n_test}_samples_{args.n_background}_background")
    os.makedirs(shap_dir, exist_ok=True)
    print(f"\nGénération des cartes dans : {shap_dir}")

    extent_sst = [-180, 180, -15, 70]
    extent_slp = [-100, 40, 20, 70] 

    for c in range(args.num_classes):
        print(f"  Traitement de la Classe {c} (Régime {c+1})...")
        
        # --- Traitement SST ---
        shap_sst_class = shap_values[c][0] # Shape: (n_test, n_lags_sst, 85, 360)
        # Moyenne signée sur tous les échantillons pour dégager les structures physiques stables
        mean_shap_sst = np.mean(shap_sst_class, axis=0)
        
        plot_mean_shap_maps(
            mean_shap_array=mean_shap_sst,
            lags_days=args.sst_lags_days,
            extent=extent_sst,
            title_prefix=f"Impact Moyen sur Régime {c+1} (SHAP Log-Odds)",
            outdir=shap_dir,
            feature_name="SST"
        )
        
        # --- Traitement SLP (Si présente) ---
        if len(args.slp_lags_days) > 0:
            shap_slp_class = shap_values[c][1] # Shape: (n_test, n_lags_slp, 53, 113)
            mean_shap_slp = np.mean(shap_slp_class, axis=0)
            
            plot_mean_shap_maps(
                mean_shap_array=mean_shap_slp,
                lags_days=args.slp_lags_days,
                extent=extent_slp,
                title_prefix=f"Impact Moyen sur Régime {c+1} (SHAP Log-Odds)",
                outdir=shap_dir,
                feature_name="SLP"
            )

    print("Génération terminée avec succès !")