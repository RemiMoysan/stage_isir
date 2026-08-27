import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import torch
import torch.nn as nn
import shap
import random
import time 
import sys
import re
from pathlib import Path

# pas besoin des targets donc pas besoin des pondérations par la latitude

project_root = Path(__file__).resolve().parent.parent.parent

# Ajout des chemins
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# MODIFICATION : Ajout de Dataset_mensuel dans les imports
from tools.datasets import Dataset, Dataset_mensuel
from tools_cnn.models import CNN_Latent_SLP_Multimodal1
from tools.models import get_median_prediction

# ============================================================
# CLASSE WRAPPER POUR SHAP (Gestion des Quantiles)
# ============================================================
class SHAP_Embedding_Wrapper(nn.Module):
    """
    Coquille autour du CNN pour que SHAP ne voie qu'un vecteur classique de taille (latent_dim).
    Si on utilise la quantile loss, cette coquille extrait automatiquement la médiane
    pour calculer les gradients dessus, en ignorant les intervalles de confiance.
    """
    def __init__(self, base_model, loss_type, quantiles, latent_dim):
        super().__init__()
        self.base_model = base_model
        self.loss_type = loss_type
        self.quantiles = quantiles
        self.latent_dim = latent_dim

    def forward(self, *inputs):
        x_sst = inputs[0]
        x_slp = inputs[1] if len(inputs) > 1 else None
        
        out = self.base_model(x_sst, x_slp)
        
        if self.loss_type == 'quantile':
            return get_median_prediction(out, self.loss_type, self.quantiles, self.latent_dim)
        return out


# ============================================================
# FONCTIONS DE VISUALISATION (Adaptées pour jours / mois)
# ============================================================

def compute_shap_regression_slope(shap_array, input_array):
    shap_mean = np.mean(shap_array, axis=0)
    input_mean = np.mean(input_array, axis=0)
    shap_centered = shap_array - shap_mean
    input_centered = input_array - input_mean
    numerator = np.sum(shap_centered * input_centered, axis=0)
    denominator = np.sum(input_centered**2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope_map = numerator / denominator
        slope_map = np.nan_to_num(slope_map, nan=0.0)
    return slope_map

# MODIFICATION : Ajout de time_unit pour dynamiser l'affichage "d" ou "m"
def plot_attribution_maps(mean_attr_array, lags, extent, title_prefix, outdir, feature_name="SST", negative_value=False, time_unit="d"):
    num_lags = len(lags)
    fig, axes = plt.subplots(1, num_lags, figsize=(6 * num_lags, 4), subplot_kw={'projection': ccrs.PlateCarree()}, facecolor='white')
    if num_lags == 1: axes = [axes]

    if negative_value:
        vmax = np.max(np.abs(mean_attr_array))
        if vmax == 0: vmax = 1e-6
        vmin = -vmax
        cmap = 'RdBu_r'
    else:
        vmax = np.max(mean_attr_array)
        if vmax == 0: vmax = 1e-6
        vmin = 0
        cmap = 'Reds'

    for i, lag in enumerate(lags):
        ax = axes[i]
        ax.set_facecolor('white')
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='110m', color='black', linewidth=0.8)
        im = ax.imshow(mean_attr_array[i], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), extent=extent, interpolation='nearest')
        ax.set_title(f"{feature_name} Lag {lag}{time_unit}", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.08)

    plt.suptitle(title_prefix, fontsize=16, y=1.05)
    plt.tight_layout()
    filename = title_prefix.replace(" ", "_").replace("|", "").replace(":", "") + f"_{feature_name}.png"
    plt.savefig(os.path.join(outdir, filename), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()

# MODIFICATION : Ajout de time_unit
def plot_pixel_beeswarm_with_locator(pixel_shap, pixel_sst, lag, extent, grid_shape, lat_idx, lon_idx, title_prefix, outdir, time_unit="d"):
    H, W = grid_shape
    lon_val = extent[0] + (lon_idx / W) * (extent[1] - extent[0])
    lat_val = extent[2] + (lat_idx / H) * (extent[3] - extent[2])

    fig = plt.figure(figsize=(10, 4), facecolor='white')
    ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax_map.set_extent(extent, crs=ccrs.PlateCarree())
    ax_map.coastlines(resolution='110m', color='black', linewidth=0.5)
    ax_map.plot(lon_val, lat_val, marker='*', color='yellow', markersize=12, markeredgecolor='black', transform=ccrs.PlateCarree())
    ax_map.set_title(f"Localisation Pixel ({lat_idx}, {lon_idx})", fontsize=10)

    ax_scatter = fig.add_subplot(1, 2, 2)
    jitter = np.random.normal(0, 0.05, size=len(pixel_shap)) 
    sc = ax_scatter.scatter(pixel_shap, jitter, c=pixel_sst, cmap='RdBu_r', alpha=0.8, edgecolor='none', s=20)
    ax_scatter.axvline(x=0, color='grey', linestyle='--', linewidth=1)
    ax_scatter.set_yticks([]) 
    ax_scatter.set_xlabel("Valeur SHAP (Impact sur l'Embedding)")
    ax_scatter.set_title(f"Distribution des Impacts", fontsize=10)
    
    cbar = fig.colorbar(sc, ax=ax_scatter, orientation='vertical', pad=0.02)
    cbar.set_label('Anomalie SST (Entrée)')

    plt.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()
    filename = f"Beeswarm_{title_prefix.replace(' ', '_')}_Lag{lag}{time_unit}_Lat{lat_idx}_Lon{lon_idx}.png"
    plt.savefig(os.path.join(outdir, filename), dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--machine', type=str, default='jean-zay-work', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    
    # MODIFICATION : Ajout des arguments mensuels et spatiaux
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2, 3, 4])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données mensuelles (_1mo.nc)')
    parser.add_argument('--lat_weight', action='store_true', help='Pondération spatiale sqrt(cos(lat))')
    

    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--method', type=str, default='gradient', choices=['gradient', 'deep'])
    parser.add_argument('--bg_type', type=str, default='zeros', choices=['zeros', 'data'])
    parser.add_argument('--n_background', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=300)
    parser.add_argument('--generate_beeswarms', action='store_true')
    parser.add_argument('--beeswarm_stride', type=int, default=40)
    
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--top_k_components', type=int, default=3)
    parser.add_argument('--early_fusion_sst', action='store_true')
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile', 'correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    parser.add_argument('--nb_members_val', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)
    val_members = all_members[-args.nb_members_val:]

    # MODIFICATION : Sélection dynamique des lags actifs selon monthly_reduction
    active_sst_lags = args.sst_lags_months if args.monthly_reduction else args.sst_lags_days
    active_slp_lags = args.slp_lags_months if args.monthly_reduction else args.slp_lags_days
    time_unit = "m" if args.monthly_reduction else "d"

    # EXTRACTION AUTONOME DE SLP_STD DEPUIS LE NOM DU FICHIER / DOSSIER
    dynamic_slp_std = 596.0  # Valeur par défaut de sécurité
    match = re.search(r'slp_std([0-9.]+)', args.model_path)
    if match:
        dynamic_slp_std = float(match.group(1))
        print(f"✅ slp_std lu automatiquement depuis le nom du modèle : {dynamic_slp_std}")
    else:
        print(f"⚠️ 'slp_std' introuvable dans model_path. Utilisation de la valeur par défaut : {dynamic_slp_std}")

    # 1. INITIALISATION DU MODÈLE
    out_features = args.latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else args.latent_dim

    base_model = CNN_Latent_SLP_Multimodal1(
        dr=0.0, 
        nb_out=out_features, 
        in_chans_sst=len(active_sst_lags), 
        in_chans_slp=len(active_slp_lags), 
        n_feat=8, 
        early_fusion_sst=args.early_fusion_sst
    ).to(device)

    # Dummy forward
    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(active_sst_lags), 85, 360).to(device) if len(active_sst_lags) > 0 else None
        dummy_slp = torch.zeros(1, len(active_slp_lags), 53, 113).to(device) if len(active_slp_lags) > 0 else None
        _ = base_model(dummy_sst, dummy_slp)

    checkpoint = torch.load(args.model_path, map_location=device)
    base_model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    base_model.eval()

    model = SHAP_Embedding_Wrapper(base_model, args.loss_type, args.quantiles, args.latent_dim)
    model.eval()

    # 2. CHARGEMENT DYNAMIQUE DES DONNÉES DE TEST (Dataset vs Dataset_mensuel)
    if not args.monthly_reduction:
        test_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    else:
        test_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.n_test, shuffle=True)
    
    test_sst, test_slp, _, _, dates, _ = next(iter(test_loader))
    test_sst = test_sst.to(device)
    test_data = [test_sst]
    
    if len(active_slp_lags) > 0:
        test_slp = test_slp.to(device)
        test_data.append(test_slp)

    extent_sst = [-180, 180, -15, 70] if args.roll_sst else [0, 359.9, -15, 70]
    extent_slp = [-100, 40, 20, 70]
    outdir = os.path.dirname(args.model_path)

    # ============================================================
    # CALCUL SHAP
    # ============================================================
    print(f"\nLancement de {args.method.upper()} (Mode mensuel : {args.monthly_reduction})")
    explain_dir = os.path.join(outdir, f"explain_{args.method}_background_{args.bg_type}_{args.n_test}samples_top{args.top_k_components}comp_stride{args.beeswarm_stride}_val{args.nb_members_val}_monthly{args.monthly_reduction}")
    os.makedirs(explain_dir, exist_ok=True)
    
    if args.generate_beeswarms:
        beeswarm_dir = os.path.join(explain_dir, "beeswarms")
        os.makedirs(beeswarm_dir, exist_ok=True)

    if args.bg_type == 'data':
        if not args.monthly_reduction:
            bg_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        else:
            bg_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
            
        bg_loader = torch.utils.data.DataLoader(bg_set, batch_size=args.n_background, shuffle=True)
        bg_sst, bg_slp, _, _, _, _ = next(iter(bg_loader))
        background_data = [bg_sst.to(device)]
        if len(active_slp_lags) > 0: background_data.append(bg_slp.to(device))
    else:
        background_data = [torch.zeros_like(test_sst[0:1])]
        if len(active_slp_lags) > 0: background_data.append(torch.zeros_like(test_slp[0:1]))

    explainer = shap.GradientExplainer(model, background_data) if args.method == 'gradient' else shap.DeepExplainer(model, background_data)
    attributions_latent_dims = explainer.shap_values(test_data)

    # Sécurité format SHAP
    if isinstance(attributions_latent_dims, np.ndarray) or torch.is_tensor(attributions_latent_dims):
        attributions_latent_dims = [[attributions_latent_dims]]
    elif isinstance(attributions_latent_dims, list):
        if len(attributions_latent_dims) > 0 and not isinstance(attributions_latent_dims[0], list):
            if out_features == 1:
                attributions_latent_dims = [attributions_latent_dims]
            else:
                attributions_latent_dims = [[arr] for arr in attributions_latent_dims]             
    
    sst_inputs_np = test_data[0].cpu().numpy()

    # ============================================================
    # GÉNÉRATION DES GRAPHES
    # ============================================================
    for c in range(args.top_k_components):
        print(f"  Traitement de la Dimension Latente {c+1}/{args.latent_dim}...")
        
        attr_c = attributions_latent_dims[c]
        shap_sst_dim = attr_c[0] if isinstance(attr_c, list) else attr_c
        shap_sst_np = shap_sst_dim.cpu().numpy() if torch.is_tensor(shap_sst_dim) else shap_sst_dim

        # 1. Importance Absolue
        mean_abs_shap_sst = np.mean(np.abs(shap_sst_np), axis=0)
        plot_attribution_maps(mean_abs_shap_sst, active_sst_lags, extent_sst, f"Importance Absolue Regime {c+1} ({args.method.upper()})", explain_dir, "SST", negative_value=False, time_unit=time_unit)
        
        # 2. Sensibilité (Pente brute)
        slope_map_sst = compute_shap_regression_slope(shap_sst_np, sst_inputs_np)
        plot_attribution_maps(slope_map_sst, active_sst_lags, extent_sst, f"Sensibilité Regime {c+1} ({args.method.upper()})", explain_dir, "SST_Slope", negative_value=True, time_unit=time_unit)

        # 3. Impact Typique
        std_sst = np.std(sst_inputs_np, axis=0)
        standardized_slope = slope_map_sst * std_sst
        plot_attribution_maps(standardized_slope, active_sst_lags, extent_sst, f"Impact Typique Regime {c+1} ({args.method.upper()})", explain_dir, "SST_Typical_Impact", negative_value=True, time_unit=time_unit)

        # 4. Corrélation
        std_shap = np.std(shap_sst_np, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation_map_sst = slope_map_sst * (std_sst / std_shap)
            correlation_map_sst = np.nan_to_num(correlation_map_sst, nan=0.0)
            
        plot_attribution_maps(correlation_map_sst, active_sst_lags, extent_sst, f"Correlation Regime {c+1} ({args.method.upper()})", explain_dir, "SST_Corr", negative_value=True, time_unit=time_unit)
        
        if args.generate_beeswarms:
            n_test, n_lags, H, W = shap_sst_np.shape
            for lat_idx in range(0, H, args.beeswarm_stride):
                for lon_idx in range(0, W, args.beeswarm_stride):
                    for lag_i, lag_val in enumerate(active_sst_lags):
                        pixel_shap_values = shap_sst_np[:, lag_i, lat_idx, lon_idx]
                        pixel_sst_values = sst_inputs_np[:, lag_i, lat_idx, lon_idx]
                        
                        if np.max(np.abs(pixel_shap_values)) < 1e-7:
                            continue
                            
                        plot_pixel_beeswarm_with_locator(
                            pixel_shap=pixel_shap_values,
                            pixel_sst=pixel_sst_values,
                            lag=lag_val,
                            extent=extent_sst,
                            grid_shape=(H, W),
                            lat_idx=lat_idx,
                            lon_idx=lon_idx,
                            title_prefix=f"Latent Dim {c+1}",
                            outdir=beeswarm_dir,
                            time_unit=time_unit
                        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nToutes les générations sont terminées ! Temps écoulé : {elapsed_time:.2f} secondes ({elapsed_time/60:.2f} minutes)")