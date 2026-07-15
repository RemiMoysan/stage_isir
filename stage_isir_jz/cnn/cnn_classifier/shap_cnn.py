import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import torch
import shap
import random
import time 
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

# Ajout des chemins
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.datasets import Dataset
from tools_cnn.models import CNN_Classifier_Multimodal

# ============================================================
# FONCTIONS DE VISUALISATION
# ============================================================

def compute_shap_regression_slope(shap_array, input_array):
    """
    Calcule le coefficient de régression linéaire (pente) beta : SHAP = beta * Input + Intercept
    pixel par pixel, le long de l'axe des échantillons (axis=0).
    shap_array et input_array : (n_test, lags, lat, lon).
    """
    # 1. Centrer les données
    shap_mean = np.mean(shap_array, axis=0)
    input_mean = np.mean(input_array, axis=0)
    
    shap_centered = shap_array - shap_mean
    input_centered = input_array - input_mean
    
    # 2. Covariance (Numérateur)
    numerator = np.sum(shap_centered * input_centered, axis=0)
    
    # 3. Variance de l'entrée (Dénominateur)
    # Remarque : on ne prend que la variance de l'input (SST), pas celle de SHAP
    denominator = np.sum(input_centered**2, axis=0)
    
    # 4. Pente Beta (avec gestion des divisions par zéro)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope_map = numerator / denominator
        slope_map = np.nan_to_num(slope_map, nan=0.0)
        
    return slope_map

def plot_attribution_maps(mean_attr_array, lags_days, extent, title_prefix, outdir, feature_name="SST", negative_value = False):
    """Génère les cartes d'explicabilité globales (Valeurs Absolues)."""
    num_lags = len(lags_days)
    fig, axes = plt.subplots(1, num_lags, figsize=(6 * num_lags, 4), subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
    if num_lags == 1: axes = [axes]

    if negative_value:
        # Échelle centrée sur zéro (pour Corrélation ou Pente)
        vmax = np.max(np.abs(mean_attr_array))
        if vmax == 0: vmax = 1e-6
        vmin = -vmax
        cmap = 'RdBu_r'
    else:
        # Échelle absolue (pour Importance)
        vmax = np.max(mean_attr_array)
        if vmax == 0: vmax = 1e-6
        vmin = 0
        cmap = 'Reds'

    for i, lag in enumerate(lags_days):
        ax = axes[i]
        ax.set_facecolor('white')
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='110m', color='black', linewidth=0.8)
        
        im = ax.imshow(
            mean_attr_array[i], 
            cmap=cmap,
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


def plot_pixel_beeswarm_with_locator(pixel_shap, pixel_sst, lag, extent, grid_shape, lat_idx, lon_idx, title_prefix, outdir):
    """
    Crée un scatter plot (pseudo-beeswarm) 1D pour un pixel précis, 
    accompagné d'une minimap pour localiser ce pixel.
    """
    H, W = grid_shape
    
    # Calcul des coordonnées géographiques approximatives du pixel
    # extent = [lon_min, lon_max, lat_min, lat_max]
    lon_val = extent[0] + (lon_idx / W) * (extent[1] - extent[0])
    lat_val = extent[2] + (lat_idx / H) * (extent[3] - extent[2])

    fig = plt.figure(figsize=(10, 4), facecolor='white')
    
    # --- 1. La Carte de Localisation (Minimap) ---
    ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax_map.set_extent(extent, crs=ccrs.PlateCarree())
    ax_map.coastlines(resolution='110m', color='black', linewidth=0.5)
    ax_map.plot(lon_val, lat_val, marker='*', color='yellow', markersize=12, markeredgecolor='black', transform=ccrs.PlateCarree())
    ax_map.set_title(f"Localisation Pixel ({lat_idx}, {lon_idx})", fontsize=10)

    # --- 2. Le Beeswarm Plot (Scatter 1D avec Jitter) ---
    ax_scatter = fig.add_subplot(1, 2, 2)
    # Jitter vertical aléatoire pour écarter les points et simuler un beeswarm
    jitter = np.random.normal(0, 0.05, size=len(pixel_shap)) 
    
    # Plot : X = SHAP, Y = Jitter, Couleur = Valeur d'entrée (SST)
    sc = ax_scatter.scatter(pixel_shap, jitter, c=pixel_sst, cmap='RdBu_r', alpha=0.8, edgecolor='none', s=20)
    
    ax_scatter.axvline(x=0, color='grey', linestyle='--', linewidth=1)
    ax_scatter.set_yticks([]) # On cache l'axe Y (le jitter n'a pas de sens physique)
    ax_scatter.set_xlabel("Valeur SHAP (Impact sur prédiction)")
    ax_scatter.set_title(f"Distribution des Impacts", fontsize=10)
    
    # Ajout de la colorbar pour la SST
    cbar = fig.colorbar(sc, ax=ax_scatter, orientation='vertical', pad=0.02)
    cbar.set_label('Anomalie SST (Entrée)')

    plt.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()
    
    # Sauvegarde
    filename = f"Beeswarm_{title_prefix.replace(' ', '_')}_Lag{lag}d_Lat{lat_idx}_Lon{lon_idx}.png"
    plt.savefig(os.path.join(outdir, filename), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()

def plot_background_samples(background_tensor, extent, outdir, num_samples=5, feature_name="SST"):
    """
    Génère une figure montrant les premiers échantillons du background 
    fourni à l'explainer SHAP pour vérification visuelle.
    """
    # Extraction du tenseur (N_bg, lags, lat, lon) en numpy
    bg_np = background_tensor.cpu().numpy()
    
    # On limite au nombre d'échantillons réellement disponibles
    num_to_plot = min(num_samples, bg_np.shape[0])
    
    # Création de la figure
    fig, axes = plt.subplots(1, num_to_plot, figsize=(4 * num_to_plot, 3), 
                             subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
    if num_to_plot == 1: axes = [axes]
    
    # On utilise une échelle globale pour voir les contrastes
    vmax = np.max(np.abs(bg_np[:num_to_plot, 0, :, :])) 
    vmin = -vmax if vmax > 0 else -1
    vmax = vmax if vmax > 0 else 1

    for i in range(num_to_plot):
        ax = axes[i]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='110m', color='black', linewidth=0.8)
        
        # On plot le premier lag (index 0) de l'échantillon i
        im = ax.imshow(
            bg_np[i, 0, :, :], 
            cmap='RdBu_r', 
            origin='lower',
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            extent=extent
        )
        ax.set_title(f"Sample {i+1}", fontsize=12)

    # Ajout d'une barre de couleur commune
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label=f'Anomalie {feature_name}')

    plt.suptitle(f"Vérification du Background SHAP ({num_to_plot} premiers échantillons)", fontsize=14, y=1.05)
    
    # Sauvegarde
    filename = f"Verification_Background_{feature_name}.png"
    plt.savefig(os.path.join(outdir, filename), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--machine', type=str, default='jean-zay-work', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--roll_sst', action='store_true')
    
    parser.add_argument('--method', type=str, default='gradient', choices=['gradient', 'deep'])
    parser.add_argument('--bg_type', type=str, default='zeros', choices=['zeros', 'data'])
    parser.add_argument('--n_background', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=300)
    
    # Nouveaux paramètres pour l'échantillonnage des Beeswarms
    parser.add_argument('--generate_beeswarms', action='store_true', help='Activer la génération des minimaps + beeswarms')
    parser.add_argument('--beeswarm_stride', type=int, default=20, help='Pas d\'échantillonnage des pixels (ex: 1 pixel tous les 20)')
    parser.add_argument('--nb_members_val', type=int, default=1, help='Nombre de membres de validation à utiliser pour le calcul de SHAP. Bon choix : 1')
    parser.add_argument('--member_shap', type=str, default=None, help='Nom du membre spécifique à utiliser pour le calcul de SHAP (ex: 1001.001). Si None, on prend les derniers membres de la liste.')

    parser.add_argument('--seed', type=int, default=42, help='Seed pour la reproductibilité du choix du / des membres')
    parser.add_argument('--early_fusion_sst', action='store_true', help='Indique si le modèle utilise une fusion précoce pour les données SST (affecte la construction du masque de fond pour SHAP)')
    
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)
    if args.member_shap is not None:
        if args.member_shap not in all_members:
            raise ValueError(f"Erreur : Le membre {args.member_shap} n'existe pas dans la liste all_members.")
        val_members = [args.member_shap]
    else:
        val_members = all_members[-args.nb_members_val:]

    # 1. INITIALISATION DU MODÈLE
    model = CNN_Classifier_Multimodal(
        num_classes=args.num_classes,
        in_chans_sst=len(args.sst_lags_days),
        in_chans_slp=len(args.slp_lags_days),
        n_feat=8, dr=0.0, early_fusion_sst=args.early_fusion_sst
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    # 2. CHARGEMENT DES DONNÉES DE TEST
    test_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.n_test, shuffle=True)
    
    test_sst, test_slp, _, _, dates, _ = next(iter(test_loader))
    test_sst = test_sst.to(device)
    test_data = [test_sst]
    
    if len(args.slp_lags_days) > 0:
        test_slp = test_slp.to(device)
        test_data.append(test_slp)

    extent_sst = [-180, 180, -15, 70] if args.roll_sst else [0, 359.9, -15, 70]
    extent_slp = [-100, 40, 20, 70]
    outdir = os.path.dirname(args.model_path)

    # ============================================================
    # CALCUL SHAP
    # ============================================================
    print(f"\nLancement de {args.method.upper()}")
    explain_dir = os.path.join(outdir, f"explain_{args.method}_background_type_{args.bg_type}_{args.n_test}_samples_stride{args.beeswarm_stride}_val_members_{'_'.join(val_members)}")
    os.makedirs(explain_dir, exist_ok=True)
    
    if args.generate_beeswarms:
        beeswarm_dir = os.path.join(explain_dir, "beeswarms")
        os.makedirs(beeswarm_dir, exist_ok=True)

    if args.bg_type == 'data':
        bg_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst)
        bg_loader = torch.utils.data.DataLoader(bg_set, batch_size=args.n_background, shuffle=True)
        bg_sst, bg_slp, _, _, _, _ = next(iter(bg_loader))
        background_data = [bg_sst.to(device)]
        if len(args.slp_lags_days) > 0: background_data.append(bg_slp.to(device))
    else:
        background_data = [torch.zeros_like(test_sst[0:1])]
        if len(args.slp_lags_days) > 0: background_data.append(torch.zeros_like(test_slp[0:1]))

    # Vérification visuelle du background
    print("Génération de l'image de vérification du background...")
    plot_background_samples(background_data[0], extent_sst, explain_dir, num_samples=5, feature_name="SST")
    if len(args.slp_lags_days) > 0:
        plot_background_samples(background_data[1], extent_slp, explain_dir, num_samples=5, feature_name="SLP")

    explainer = shap.GradientExplainer(model, background_data) if args.method == 'gradient' else shap.DeepExplainer(model, background_data)
    attributions_all_classes = explainer.shap_values(test_data)
    
    # ============================================================
    # SÉCURITÉ FORMAT SHAP (Bulletproof pour Binary ou Multi-class)
    # ============================================================
    if isinstance(attributions_all_classes, np.ndarray) or torch.is_tensor(attributions_all_classes):
        # Cas : 1 seule sortie (Classification binaire, num_classes=1) et 1 seule entrée (SST)
        attributions_all_classes = [[attributions_all_classes]]
    elif isinstance(attributions_all_classes, list):
        if len(attributions_all_classes) > 0 and not isinstance(attributions_all_classes[0], list):
            # Cas ambigus
            if args.num_classes == 1:
                # 1 Sortie / Multi-entrées (SST + SLP)
                attributions_all_classes = [attributions_all_classes]
            else:
                # Multi-sorties / 1 Entrée (SST seule)
                attributions_all_classes = [[arr] for arr in attributions_all_classes]

    sst_inputs_np = test_data[0].cpu().numpy() # Shape: (n_test, lags, H, W)

    # ============================================================
    # GÉNÉRATION DES GRAPHES
    # ============================================================
    for c in range(args.num_classes):
        print(f"  Traitement de la Classe {c} (Régime {c+1})...")
        attr_c = attributions_all_classes[c]
        shap_sst_class = attr_c[0] if isinstance(attr_c, list) else attr_c
        
        # Extraction en numpy (on le fait une seule fois en haut pour toutes les cartes)
        shap_sst_np = shap_sst_class.cpu().numpy() if torch.is_tensor(shap_sst_class) else shap_sst_class

        # 1. CARTE GLOBALE A : Moyenne de la VALEUR ABSOLUE (Importance)
        mean_abs_shap_sst = np.mean(np.abs(shap_sst_np), axis=0)
        plot_attribution_maps(mean_abs_shap_sst, args.sst_lags_days, extent_sst, f"Importance Absolue Regime {c+1} ({args.method.upper()})", explain_dir, "SST", negative_value=False)
        
        # 2. CARTE GLOBALE C : CARTE DE SENSIBILITÉ (Pente de régression brute, beta)
        slope_map_sst = compute_shap_regression_slope(shap_sst_np, sst_inputs_np)
        plot_attribution_maps(slope_map_sst, args.sst_lags_days, extent_sst, f"Sensibilité Regime {c+1} ({args.method.upper()})", explain_dir, "SST_Slope", negative_value=True)

        # 3. CARTE GLOBALE D : IMPACT TYPIQUE (Pente Standardisée)
        # Écart-type de l'entrée (SST)
        std_sst = np.std(sst_inputs_np, axis=0)
        standardized_slope = slope_map_sst * std_sst
        plot_attribution_maps(standardized_slope, args.sst_lags_days, extent_sst, f"Impact Typique Regime {c+1} ({args.method.upper()})", explain_dir, "SST_Typical_Impact", negative_value=True)

        # 4. CARTE GLOBALE B : CARTE DE CORRÉLATION (Déduite mathématiquement !)
        # Écart-type de la sortie (SHAP)
        std_shap = np.std(shap_sst_np, axis=0)
        
        # Calcul r = beta * (std_sst / std_shap) avec protection contre la division par zéro
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation_map_sst = slope_map_sst * (std_sst / std_shap)
            correlation_map_sst = np.nan_to_num(correlation_map_sst, nan=0.0)
            
        plot_attribution_maps(correlation_map_sst, args.sst_lags_days, extent_sst, f"Correlation Regime {c+1} ({args.method.upper()})", explain_dir, "SST_Corr", negative_value=True)

        # 2. CARTES LOCALES (BEESWARMS)
        if args.generate_beeswarms:
            n_test, n_lags, H, W = shap_sst_class.shape
            
            # On parcourt la grille avec le pas défini (stride)
            for lat_idx in range(0, H, args.beeswarm_stride):
                for lon_idx in range(0, W, args.beeswarm_stride):
                    
                    # On génère un plot pour chaque Lag de ce pixel
                    for lag_i, lag_days in enumerate(args.sst_lags_days):
                        # Extraction des 300 valeurs de SHAP et SST pour ce pixel précis
                        pixel_shap_values = shap_sst_class[:, lag_i, lat_idx, lon_idx]
                        pixel_sst_values = sst_inputs_np[:, lag_i, lat_idx, lon_idx]
                        
                        # Si le pixel est masqué (ex: terre), ses SHAP/SST seront toujours à 0
                        # On skip les pixels 100% vides pour gagner du temps
                        if np.max(np.abs(pixel_shap_values)) < 1e-7:
                            continue
                            
                        plot_pixel_beeswarm_with_locator(
                            pixel_shap=pixel_shap_values,
                            pixel_sst=pixel_sst_values,
                            lag=lag_days,
                            extent=extent_sst,
                            grid_shape=(H, W),
                            lat_idx=lat_idx,
                            lon_idx=lon_idx,
                            title_prefix=f"Regime {c+1}",
                            outdir=beeswarm_dir
                        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nToutes les générations sont terminées ! Temps écoulé : {elapsed_time:.2f} secondes ({elapsed_time/60:.2f} minutes)")
    print("\nToutes les générations statiques sont terminées !")