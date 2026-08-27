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

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from shared_tools.datasets import Dataset, Dataset_mensuel
from tools_cnn.models import CNN_Latent_SLP_Multimodal1_tunable
from shared_tools.models import get_median_prediction_full_slp, SHAP_Embedding_Wrapper
from shared_tools.evaluation_functions import generate_summary_plots, plot_individual_sample


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['best', 'final'], default='best')
    parser.add_argument('--machine', type=str, default='jean-zay-work', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    
    parser.add_argument('--embed_method', type=str, default='pca')
    parser.add_argument('--embed_path', type=str, default='')
    parser.add_argument('--top_k_components', type=int, default=1)
    
    parser.add_argument('--nb_members_train', type=int, default=10)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_members_test', type=int, default=1)
    parser.add_argument('--force_val_members', type=str, nargs='*', default=None)
    parser.add_argument('--force_test_members', type=str, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2, 3, 4])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--monthly_reduction', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--roll_sst', action='store_true')
    
    parser.add_argument('--method', type=str, default='gradient', choices=['gradient', 'deep'])
    parser.add_argument('--bg_type', type=str, default='zeros', choices=['zeros', 'data'])
    parser.add_argument('--n_background', type=int, default=100)
    
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--quantiles', type=float, nargs='*', default=[])
    parser.add_argument('--dr_conv', type=float, default=0.4)
    parser.add_argument('--dr_fc', type=float, default=0.1)
    parser.add_argument('--fc_dim', type=int, default=20)
    parser.add_argument('--n_feat', type=int, default=20)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--filter_mult', type=float, default=1.0)
    parser.add_argument('--pool_strategy', type=str, default='progressive')
    parser.add_argument('--pool_type', type=str, default='max')
    parser.add_argument('--sst_kx', type=int, default=3)
    parser.add_argument('--sst_ky', type=int, default=5)
    parser.add_argument('--sst_pool_x', type=int, default=2)
    parser.add_argument('--sst_pool_y', type=int, default=2)
    parser.add_argument('--use_gap', action='store_true')
    parser.add_argument('--early_fusion_sst', action='store_true')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

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

    active_sst_lags = args.sst_lags_months if args.monthly_reduction else args.sst_lags_days
    active_slp_lags = args.slp_lags_months if args.monthly_reduction else args.slp_lags_days
    time_unit = "m" if args.monthly_reduction else "d"

    dynamic_slp_std = 596.0 
    match = re.search(r'slp_std([0-9.]+)', args.cnn_dir)
    if match:
        dynamic_slp_std = float(match.group(1))
        print(f"✅ slp_std lu automatiquement depuis le cnn_dir : {dynamic_slp_std}")

    # ============================================================
    # PRÉPARATION POIDS LATITUDE & MODELE PCA
    # ============================================================
    safe_wgts = None
    pca_model = None
    if args.embed_method == 'pca':
        try:
            pca_model = joblib.load(args.embed_path)
        except Exception as e:
            print(f"⚠️ Impossible de charger le PCA : {e}")

        if args.lat_weight:
            import xarray as xr
            sample_path = os.path.join(parent_dir, f"data/SLP/PSL_anom_LE2-{train_members[0]}_1mo.nc")
            if os.path.exists(sample_path):
                ds_sample = xr.open_dataset(sample_path)
                lats = ds_sample['lat'].values
                coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
                wgts = np.sqrt(coslat).reshape(len(lats), 1)
                safe_wgts = np.broadcast_to(wgts, (len(lats), len(ds_sample['lon'].values))).flatten()
                safe_wgts = np.maximum(safe_wgts, 1e-5)
                ds_sample.close()

    # 1. INITIALISATION DU MODÈLE CNN
    out_features = args.latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else args.latent_dim
    base_model = CNN_Latent_SLP_Multimodal1_tunable(
        dr_conv=args.dr_conv, dr_fc=args.dr_fc, fc_dim=args.fc_dim, nb_out=out_features,
        in_chans_sst=len(active_sst_lags), in_chans_slp=len(active_slp_lags),
        n_feat=args.n_feat, early_fusion_sst=args.early_fusion_sst, depth=args.depth,
        filter_mult=args.filter_mult, sst_kx=args.sst_kx, sst_ky=args.sst_ky,
        sst_pool_x=args.sst_pool_x, sst_pool_y=args.sst_pool_y, pool_type=args.pool_type,
        pool_strategy=args.pool_strategy, activation=args.activation, use_gap=args.use_gap
    ).to(device)

    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(active_sst_lags), 85, 360).to(device) if len(active_sst_lags) > 0 else None
        dummy_slp = torch.zeros(1, len(active_slp_lags), 53, 113).to(device) if len(active_slp_lags) > 0 else None
        _ = base_model(dummy_sst, dummy_slp)

    model_path = os.path.join(args.cnn_dir, "best_val_CNN.pth" if args.model_type == 'best' else "final_model_CNN.pth")
    checkpoint = torch.load(model_path, map_location=device)
    base_model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    
    use_sst = len(active_sst_lags) > 0
    use_slp = len(active_slp_lags) > 0
    model = SHAP_Embedding_Wrapper(base_model, args.loss_type, args.quantiles, args.latent_dim)
    model.eval()

    # 2. DATASETS
    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1)
    
    if not args.monthly_reduction:
        train_set = Dataset(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        test_set = Dataset(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=active_sst_lags, slp_lags_days=active_slp_lags, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    else:
        train_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
        test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=active_slp_lags, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    test_sst, test_slp, test_targets, test_maps, test_dates, test_members_list = next(iter(test_loader))
    
    test_data = []
    if use_sst: test_data.append(test_sst.to(device))
    if use_slp: test_data.append(test_slp.to(device))

    # Obtenir les prédictions
    with torch.no_grad():
        test_preds = model(*test_data).cpu().numpy()

    # Obtenir les Targets et générer les Reconstructions
    target_latent = None
    pred_maps_recon = np.zeros((len(test_preds), 53, 113))
    target_maps_recon = np.zeros((len(test_preds), 53, 113))
    
    if pca_model is not None:
        slp_flat = test_targets.view(test_targets.size(0), -1).numpy()
        if args.lat_weight and safe_wgts is not None:
            slp_flat *= safe_wgts
        target_latent = pca_model.transform(slp_flat)[:, :args.latent_dim]
        
        # Reconstruction Pred
        padded_pred = np.zeros((test_preds.shape[0], pca_model.n_components_))
        padded_pred[:, :args.latent_dim] = test_preds
        pred_flat_recon = pca_model.inverse_transform(padded_pred)
        
        # Reconstruction Target
        padded_target = np.zeros((target_latent.shape[0], pca_model.n_components_))
        padded_target[:, :args.latent_dim] = target_latent
        target_flat_recon = pca_model.inverse_transform(padded_target)
        
        if args.lat_weight and safe_wgts is not None:
            pred_flat_recon /= safe_wgts
            target_flat_recon /= safe_wgts
            
        pred_maps_recon = pred_flat_recon.reshape(-1, 53, 113)
        target_maps_recon = target_flat_recon.reshape(-1, 53, 113)

    extent_sst = [-180, 180, -15, 70] if args.roll_sst else [0, 359.9, -15, 70]
    extent_slp = [-100, 40, 20, 70]

    # ============================================================
    # CALCUL SHAP
    # ============================================================
    print(f"\nLancement SHAP ({args.method.upper()}) sur le Test Set...")
    explain_dir = os.path.join(args.cnn_dir, f"shap_eval_{args.model_type}_{args.method}_bg-{args.bg_type}_nbg-{args.n_background}")
    os.makedirs(explain_dir, exist_ok=True)

    if args.bg_type == 'data':
        bg_loader = torch.utils.data.DataLoader(train_set, batch_size=args.n_background, shuffle=True)
        bg_sst, bg_slp, _, _, _, _ = next(iter(bg_loader))
        background_data = []
        if use_sst: background_data.append(bg_sst.to(device))
        if use_slp: background_data.append(bg_slp.to(device))
    else:
        background_data = [torch.zeros_like(test_sst[0:1]).to(device)]
        if use_slp: background_data.append(torch.zeros_like(test_slp[0:1]).to(device))

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

        # ---- Graphes Globaux (Tous membres) ----
        generate_summary_plots(
            shap_np=shap_sst_np, 
            inputs_np=sst_inputs_np, 
            lags=active_sst_lags, 
            extent=extent_sst, 
            outdir=explain_dir, 
            display_name=f"Component {c+1} (All test members)",  # Le beau titre 
            file_prefix=f"Dim_{c+1}_ALL",                          # Le nom pour le fichier
            time_unit=time_unit, 
            pc1_std=pc1_std
        )

        # ---- Graphes Globaux (Par Membre) ----
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
                    display_name=f"Component {c+1} (Member {mem})", # Le beau titre
                    file_prefix=f"Dim_{c+1}",                       # Le nom pour le fichier
                    time_unit=time_unit, 
                    pc1_std=pc1_std
                )

        # ---- Graphes Individuels (Par Sample) ----
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
                dynamic_slp_std=dynamic_slp_std  # <--- NOUVEL ARGUMENT
            )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n✅ SHAP terminé ! Graphes sauvegardés dans {explain_dir} (Temps écoulé : {elapsed_time/60:.2f} minutes)")