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
import joblib
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from shared_tools.datasets import Dataset, Dataset_mensuel
from shared_tools.models import get_median_prediction_full_slp
from shared_tools.evaluation_functions import generate_summary_plots, plot_individual_sample

# ============================================================
# ARCHITECTURE DU MODÈLE DE RÉGRESSION LINÉAIRE
# ============================================================
class LinearRegressionPredictor(nn.Module):
    def __init__(self, in_features, out_dim=128):
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim)

    def forward(self, x):
        return self.linear(x)

# ============================================================
# WRAPPER SHAP DIFFÉRENTIABLE (Pour gérer PCA & Raw)
# ============================================================
class SHAP_LinReg_Wrapper(nn.Module):
    """
    Encapsule la compression PCA (différentiable) et la prédiction linéaire.
    Cela permet à SHAP d'envoyer des pixels bruts et de récupérer les gradients spatiaux,
    même si le modèle a été entraîné sur un espace PCA réduit.
    """
    def __init__(self, base_model, loss_type, quantiles, latent_dim, input_format, 
                 sst_pca_dim, sst_pca_comps, sst_pca_mean, wgts_sst, 
                 slp_pca_comps, slp_pca_mean, wgts_slp, active_slp_lags):
        super().__init__()
        self.base_model = base_model
        self.loss_type = loss_type
        self.quantiles = quantiles
        self.latent_dim = latent_dim
        
        self.input_format = input_format
        self.sst_pca_dim = sst_pca_dim
        self.sst_pca_comps = sst_pca_comps
        self.sst_pca_mean = sst_pca_mean
        self.wgts_sst = wgts_sst
        
        self.slp_pca_comps = slp_pca_comps
        self.slp_pca_mean = slp_pca_mean
        self.wgts_slp = wgts_slp
        self.active_slp_lags = active_slp_lags

    def forward(self, *inputs):
        x_sst = inputs[0]
        x_slp = inputs[1] if len(inputs) > 1 else None
        
        B, L, H, W = x_sst.shape
        
        if self.input_format == 'pca':
            # -- Encodage différentiable SST --
            x_sst_flat = x_sst.view(B * L, -1)
            if self.wgts_sst is not None:
                x_sst_flat = x_sst_flat * self.wgts_sst
            x_sst_centered = x_sst_flat - self.sst_pca_mean
            sst_embed = torch.matmul(x_sst_centered, self.sst_pca_comps.T)
            X_sst_tensor = sst_embed.view(B, L * self.sst_pca_dim)
            
            # -- Encodage différentiable SLP --
            if x_slp is not None and len(self.active_slp_lags) > 0:
                x_slp_flat = x_slp.view(B * len(self.active_slp_lags), -1)
                if self.wgts_slp is not None:
                    x_slp_flat = x_slp_flat * self.wgts_slp
                x_slp_centered = x_slp_flat - self.slp_pca_mean
                slp_embed = torch.matmul(x_slp_centered, self.slp_pca_comps.T)
                X_slp_tensor = slp_embed.view(B, len(self.active_slp_lags) * 1)
                X_combined = torch.cat((X_sst_tensor, X_slp_tensor), dim=1)
            else:
                X_combined = X_sst_tensor
        else:
            X_sst_tensor = x_sst.view(B, -1)
            if x_slp is not None and len(self.active_slp_lags) > 0:
                X_slp_tensor = x_slp.view(B, -1)
                X_combined = torch.cat((X_sst_tensor, X_slp_tensor), dim=1)
            else:
                X_combined = X_sst_tensor
                
        out = self.base_model(X_combined)
        
        if self.loss_type == 'quantile':
            return get_median_prediction_full_slp(out, self.loss_type, self.quantiles, self.latent_dim)
        return out

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--linreg_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['best', 'final'], default='best')
    parser.add_argument('--machine', type=str, default='jean-zay-work', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])

    # --- PCA PATHS ---
    parser.add_argument('--embed_method', type=str, default='pca')
    parser.add_argument('--embed_path', type=str, required=True)
    parser.add_argument('--embed_path_sst', type=str, default=None)
    parser.add_argument('--top_k_components', type=int, default=1)

    # --- SPLIT ---
    parser.add_argument('--nb_members_train', type=int, default=10)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_members_test', type=int, default=1)
    parser.add_argument('--force_val_members', type=str, nargs='*', default=None)
    parser.add_argument('--force_test_members', type=str, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=1)

    # --- DATA PARAMS ---
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2, 3, 4])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--monthly_reduction', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--roll_sst', action='store_true')

    # --- HYPERPARAMS ---
    parser.add_argument('--method', type=str, default='gradient', choices=['gradient', 'deep'])
    parser.add_argument('--bg_type', type=str, default='zeros', choices=['zeros', 'data'])
    parser.add_argument('--n_background', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--quantiles', type=float, nargs='*', default=[])
    parser.add_argument('--input_format', type=str, choices=['raw', 'pca'], default='raw')
    parser.add_argument('--sst_pca_dim', type=int, default=0)
    parser.add_argument('--bs', type=int, default=64)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    # ============================================================
    # 1. SETUP DATASET & MEMBERS
    # ============================================================
    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    if args.force_val_members is not None or args.force_test_members is not None:
        val_members = args.force_val_members if args.force_val_members else []
        test_members = args.force_test_members if args.force_test_members else []
        remaining = [m for m in all_members if m not in val_members and m not in test_members]
        train_members = remaining[:args.nb_members_train]
    else:
        train_members = all_members[:args.nb_members_train]
        val_members = all_members[-args.nb_members_val:]
        test_members = all_members[args.nb_members_train:args.nb_members_train + args.nb_members_test] if args.nb_members_test > 0 else []

    if len(test_members) == 0:
        print("⚠️ Aucun membre de test. Le SHAP ne sera pas calculé.")
        sys.exit(0)

    active_sst_lags = sorted(args.sst_lags_months if args.monthly_reduction else args.sst_lags_days, reverse=True)
    active_slp_lags = sorted(args.slp_lags_months if args.monthly_reduction else args.slp_lags_days, reverse=True)
    time_unit = "m" if args.monthly_reduction else "d"

    dynamic_slp_std = 596.0 
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match: dynamic_slp_std = float(match.group(1))

    dynamic_sst_std = 0.707 
    if args.embed_path_sst:
        match = re.search(r'sst_std([0-9.]+)', args.embed_path_sst)
        if match: dynamic_sst_std = float(match.group(1))

    # ============================================================
    # 2. POIDS LATITUDE & MODELES PCA
    # ============================================================
    slp_pca_model = joblib.load(args.embed_path)
    slp_pca_mean_gpu = torch.tensor(slp_pca_model.mean_, dtype=torch.float32, device=device)
    slp_pca_components_gpu = torch.tensor(slp_pca_model.components_[:max(args.latent_dim, 1)], dtype=torch.float32, device=device)

    wgts_slp_gpu, safe_wgts_slp = None, None
    if args.lat_weight:
        sample_path = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-1001.001_1mo.nc"
        try:
            import xarray as xr
            with xr.open_dataset(sample_path) as ds_sample:
                coslat = np.cos(np.deg2rad(ds_sample['lat'].values)).clip(0., 1.)
                h, w = len(coslat), len(ds_sample['lon'].values)
                wgts_slp_flat = np.broadcast_to(np.sqrt(coslat).reshape(h, 1), (h, w)).flatten()
                wgts_slp_gpu = torch.tensor(wgts_slp_flat, dtype=torch.float32, device=device)
                safe_wgts_slp = np.maximum(wgts_slp_flat, 1e-5)
        except Exception as e:
            print(f"⚠️ Erreur latitude SLP : {e}")

    sst_pca_model = None
    sst_pca_mean_gpu, sst_pca_components_gpu = None, None
    if args.input_format == 'pca' and args.embed_path_sst:
        sst_pca_model = joblib.load(args.embed_path_sst)
        sst_pca_mean_gpu = torch.tensor(sst_pca_model.mean_, dtype=torch.float32, device=device)
        sst_pca_components_gpu = torch.tensor(sst_pca_model.components_[:args.sst_pca_dim], dtype=torch.float32, device=device)

    wgts_sst_gpu = None
    if args.lat_weight:
        sample_path_sst = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/SST_anom_LE2-1001.001_T_regrid_1mo.nc"
        try:
            with xr.open_dataset(sample_path_sst) as ds_sample_sst:
                ds_sample_sst = ds_sample_sst.sel(lat=slice(-15, 70))
                coslat_sst = np.cos(np.deg2rad(ds_sample_sst['lat'].values)).clip(0., 1.)
                h_sst, w_sst = len(coslat_sst), len(ds_sample_sst['lon'].values)
                wgts_sst_flat = np.broadcast_to(np.sqrt(coslat_sst).reshape(h_sst, 1), (h_sst, w_sst)).flatten()
                wgts_sst_gpu = torch.tensor(wgts_sst_flat, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"⚠️ Erreur latitude SST : {e}")

    # ============================================================
    # 3. INITIALISATION DU MODÈLE
    # ============================================================
    if args.input_format == 'pca':
        in_features_sst = len(active_sst_lags) * args.sst_pca_dim
        in_features_slp = len(active_slp_lags) * 1
    else:
        in_features_sst = len(active_sst_lags) * 85 * 360
        in_features_slp = len(active_slp_lags) * 53 * 113

    out_features = args.latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else args.latent_dim
    base_model = LinearRegressionPredictor(in_features=in_features_sst + in_features_slp, out_dim=out_features).to(device)

    if args.model_type == 'best':
        linreg_path = os.path.join(args.linreg_dir, "best_val_LinReg.pth")
        if not os.path.exists(linreg_path): linreg_path = os.path.join(args.linreg_dir, f"best_val_LinReg_bs{args.bs}.pth")
    else:
        linreg_path = os.path.join(args.linreg_dir, "final_model_LinReg.pth")
        if not os.path.exists(linreg_path): linreg_path = os.path.join(args.linreg_dir, f"final_model_LinReg_bs{args.bs}.pth")

    checkpoint = torch.load(linreg_path, map_location=device)
    base_model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)

    # Initialisation du Wrapper
    model = SHAP_LinReg_Wrapper(
        base_model=base_model, loss_type=args.loss_type, quantiles=args.quantiles, latent_dim=args.latent_dim,
        input_format=args.input_format, sst_pca_dim=args.sst_pca_dim, 
        sst_pca_comps=sst_pca_components_gpu, sst_pca_mean=sst_pca_mean_gpu, wgts_sst=wgts_sst_gpu,
        slp_pca_comps=slp_pca_components_gpu, slp_pca_mean=slp_pca_mean_gpu, wgts_slp=wgts_slp_gpu,
        active_slp_lags=active_slp_lags
    ).to(device)
    model.eval()

    # ============================================================
    # 4. DATASETS
    # ============================================================
    if not args.monthly_reduction:
        train_set = Dataset(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
        test_set = Dataset(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
    else:
        train_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
        test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    test_sst, test_slp, test_targets, test_maps, test_dates, test_members_list = next(iter(test_loader))

    test_data = []
    if len(active_sst_lags) > 0: test_data.append(test_sst.to(device))
    if len(active_slp_lags) > 0: test_data.append(test_slp.to(device))

    with torch.no_grad():
        test_preds = model(*test_data).cpu().numpy()

    # Reconstructions SLP
    slp_flat = test_targets.view(test_targets.size(0), -1).numpy()
    if args.lat_weight and safe_wgts_slp is not None:
        slp_flat *= safe_wgts_slp
    target_latent = slp_pca_model.transform(slp_flat)[:, :args.latent_dim]

    padded_pred = np.zeros((test_preds.shape[0], slp_pca_model.n_components_))
    padded_pred[:, :args.latent_dim] = test_preds
    pred_flat_recon = slp_pca_model.inverse_transform(padded_pred)

    padded_target = np.zeros((target_latent.shape[0], slp_pca_model.n_components_))
    padded_target[:, :args.latent_dim] = target_latent
    target_flat_recon = slp_pca_model.inverse_transform(padded_target)

    if args.lat_weight and safe_wgts_slp is not None:
        pred_flat_recon /= safe_wgts_slp
        target_flat_recon /= safe_wgts_slp

    pred_maps_recon = pred_flat_recon.reshape(-1, 53, 113)
    target_maps_recon = target_flat_recon.reshape(-1, 53, 113)

    extent_sst = [-180, 180, -15, 70] if args.roll_sst else [0, 359.9, -15, 70]
    extent_slp = [-100, 40, 20, 70]

    # ============================================================
    # CALCUL SHAP
    # ============================================================
    print(f"\nLancement SHAP ({args.method.upper()}) sur le Test Set...")
    explain_dir = os.path.join(args.linreg_dir, f"shap_eval_{args.model_type}_{args.method}_bg-{args.bg_type}_nbg-{args.n_background}")
    os.makedirs(explain_dir, exist_ok=True)

    if args.bg_type == 'data':
        bg_loader = torch.utils.data.DataLoader(train_set, batch_size=args.n_background, shuffle=True)
        bg_sst, bg_slp, _, _, _, _ = next(iter(bg_loader))
        background_data = []
        if len(active_sst_lags) > 0: background_data.append(bg_sst.to(device))
        if len(active_slp_lags) > 0: background_data.append(bg_slp.to(device))
    else:
        background_data = [torch.zeros_like(test_sst[0:1]).to(device)]
        if len(active_slp_lags) > 0: background_data.append(torch.zeros_like(test_slp[0:1]).to(device))

    explainer = shap.GradientExplainer(model, background_data) if args.method == 'gradient' else shap.DeepExplainer(model, background_data)
    attributions_latent_dims = explainer.shap_values(test_data)

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
    # GÉNÉRATION DES GRAPHES GLOBAUX & INDIVIDUELS
    # ============================================================
    for c in range(args.top_k_components):
        print(f"  Traitement de la Dimension Latente {c+1}/{args.latent_dim}...")

        attr_c = attributions_latent_dims[c]
        shap_sst_dim = attr_c[0] if isinstance(attr_c, list) else attr_c
        shap_sst_np = shap_sst_dim.cpu().numpy() if torch.is_tensor(shap_sst_dim) else shap_sst_dim

        pc1_std = np.std(target_latent[:, c]) if target_latent is not None else 1.0
        if pc1_std == 0: pc1_std = 1.0

        generate_summary_plots(
            shap_np=shap_sst_np, 
            inputs_np=sst_inputs_np, 
            lags=active_sst_lags, 
            extent=extent_sst, 
            outdir=explain_dir, 
            display_name=f"Component {c+1} (All test members)", 
            file_prefix=f"Dim_{c+1}_ALL", 
            time_unit=time_unit, 
            pc1_std=pc1_std
        )

        print("    Génération des résumés par membre...")
        unique_members = np.unique([m.item() if hasattr(m, 'item') else m for m in test_members_list])
        for mem in unique_members:
            if isinstance(mem, bytes): mem = mem.decode()
            mem_mask = np.array([(m.item() if hasattr(m, 'item') else m) == mem or 
                                 (m.item().decode() if hasattr(m, 'item') and isinstance(m.item(), bytes) else m) == mem 
                                 for m in test_members_list])

            if np.sum(mem_mask) > 1:
                mem_dir = os.path.join(explain_dir, str(mem))
                os.makedirs(mem_dir, exist_ok=True)
                generate_summary_plots(
                    shap_np=shap_sst_np[mem_mask], 
                    inputs_np=sst_inputs_np[mem_mask], 
                    lags=active_sst_lags, 
                    extent=extent_sst, 
                    outdir=mem_dir, 
                    display_name=f"Component {c+1} (Member {mem})", 
                    file_prefix=f"Dim_{c+1}", 
                    time_unit=time_unit, 
                    pc1_std=pc1_std
                )

        print("    Génération des plots individuels...")
        for i in range(len(test_sst)):
            mem = test_members_list[i].item() if hasattr(test_members_list[i], 'item') else test_members_list[i]
            if isinstance(mem, bytes): mem = mem.decode()
            date = test_dates[i].item() if hasattr(test_dates[i], 'item') else test_dates[i]

            p_val = test_preds[i, c]
            t_val = target_latent[i, c] if target_latent is not None else None

            plot_individual_sample(
                input_sst=sst_inputs_np[i],
                shap_sst=shap_sst_np[i],
                pred_val=p_val,
                target_val=t_val,
                pred_map=pred_maps_recon[i],
                target_map_recon=target_maps_recon[i],
                true_map=test_maps[i].numpy(),
                lags=active_sst_lags,
                extent_sst=extent_sst,
                extent_slp=extent_slp,
                member=mem,
                date=date,
                dim_c=c+1,
                outdir=explain_dir,
                time_unit=time_unit,
                pc1_std=pc1_std,
                dynamic_slp_std=dynamic_slp_std 
            )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n✅ SHAP terminé ! Graphes sauvegardés dans {explain_dir} (Temps écoulé : {elapsed_time/60:.2f} minutes)")