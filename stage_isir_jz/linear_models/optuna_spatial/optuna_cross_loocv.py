import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import random
import joblib
import re
import xarray as xr
import hashlib
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna
import copy

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from shared_tools.datasets import Dataset_mensuel
from shared_tools.models import ConvVAE, compute_loss, get_median_prediction, spatial_penalty_tikhonov, spatial_penalty_laplacian
from shared_tools.optuna_loop_helpers import encode_to_latent_gpu, decode_to_spatial_map_gpu, compute_targeted_spatial_metrics
from shared_tools.optuna_plots import generate_crossval_matrix, generate_1d_loocv_heatmap

# ============================================================
# ARCHITECTURE ADAPTÉE (Entrée 1D générique)
# ============================================================
class LinearRegressionPredictor(nn.Module):
    def __init__(self, in_features, out_dim=128):
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim)

    def forward(self, x):
        return self.linear(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

VAL_GRID_MEMBERS = ALL_MEMBERS[30:31]  
TEST_GRID_MEMBERS = ALL_MEMBERS[:]

def objective(trial):
    # 1. Sélection croisée par la Grid Optuna
    val_member = trial.suggest_categorical("val_member", ALL_MEMBERS)
    test_member = trial.suggest_categorical("test_member", ALL_MEMBERS)
    val_members = [val_member]
    test_members = [test_member]
        
    train_members = [m for m in ALL_MEMBERS if m != val_member and m != test_member]

    # 2. Hyperparamètres FIXES venant de argparse
    bs = args.bs
    lr = args.lr
    weight_decay = args.weight_decay
    alpha_l1 = args.alpha_l1
    alpha_tik = args.alpha_tik
    alpha_lap = args.alpha_lap
    noise_std = args.noise_std
    grad_clip = args.gradient_clip
    loss_type = args.loss_type
    input_format = args.input_format
    sst_lags_months = args.sst_lags_months
    slp_lags_months = args.slp_lags_months
    sst_pca_dim = args.sst_pca_dim
    latent_dim = args.latent_dim
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # ============================================================
    # 2. PRÉPARATION DES DONNÉES
    # ============================================================
    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 2)) - 1)

    train_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=True, noise_std=noise_std)
    val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=False)
    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=False)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # 3. INITIALISATION DU MODÈLE ET OPTIMIZER
    # ============================================================
    if input_format == 'pca':
        in_features_sst = len(sst_lags_months) * sst_pca_dim
        in_features_slp = len(slp_lags_months) * 1
    else:
        in_features_sst = len(sst_lags_months) * 85 * 360
        in_features_slp = len(slp_lags_months) * 53 * 113

    out_features = latent_dim * len(quantiles) if loss_type == 'quantile' else latent_dim

    model = LinearRegressionPredictor(in_features=in_features_sst + in_features_slp, out_dim=out_features).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ============================================================
    # 4. BOUCLE D'ENTRAÎNEMENT ET TRACKING SPATIAL
    # ============================================================
    best_trial_R2 = -float('inf')
    best_trial_L1 = -float('inf')
    best_trial_corr = -float('inf')
    best_target_metric = -float('inf')
    metrics_history = []
    patience_counter = 0
    best_model_state = None

    total_batches = len(trainloader)
    eval_steps_set = set(np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int)) | {0}
    eval_steps_epoch2_set = set(np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int)) | {0}

    for epoch in range(args.nb_epochs):
        model.train()
        for batch_idx, (X_sst, X_slp, y_target, _, _, _) in enumerate(trainloader):
            optimizer.zero_grad()
            
            X_sst = X_sst.to(device, non_blocking=True)
            if len(slp_lags_months) > 0:
                X_slp = X_slp.to(device, non_blocking=True)
            
            B, L, H, W = X_sst.shape
            
            if input_format == 'pca':
                X_sst_2d = X_sst.view(B * L, H, W)
                sst_embed = encode_to_latent_gpu(
                    X_sst_2d, 'pca', sst_pca_dim, 
                    sst_pca_components_gpu[:sst_pca_dim], sst_pca_mean_gpu, wgts_sst_gpu, None
                )
                X_sst_tensor = sst_embed.view(B, L * sst_pca_dim)
                if len(slp_lags_months) > 0:
                    X_slp_2d = X_slp.view(B * len(slp_lags_months), 53, 113)
                    slp_embed_entree = encode_to_latent_gpu(
                        X_slp_2d, 'pca', 1, 
                        slp_pca_components_gpu[:1], slp_pca_mean_gpu, wgts_slp_gpu, None
                    )
                    X_slp_tensor = slp_embed_entree.view(B, len(slp_lags_months) * 1)
                    X_combined = torch.cat((X_sst_tensor, X_slp_tensor), dim=1)
                else:
                    X_combined = X_sst_tensor
            else:
                X_sst_tensor = X_sst.view(B, -1).to(device, non_blocking=True)
                if len(slp_lags_months) > 0:
                    X_slp_tensor = X_slp.view(B, -1).to(device, non_blocking=True)
                    X_combined = torch.cat((X_sst_tensor, X_slp_tensor), dim=1)
                else:
                    X_combined = X_sst_tensor
            
            # Cible latente avec tronquature fixée (latent_dim)
            target_embed = encode_to_latent_gpu(y_target.to(device, non_blocking=True), args.embed_method, latent_dim, slp_pca_components_gpu[:latent_dim], slp_pca_mean_gpu, wgts_slp_gpu, vae_model)
                
            pred = model(X_combined)
            base_loss = compute_loss(pred, target_embed, loss_type=loss_type, quantiles=quantiles, reduction='mean')
            
            penalty = 0.0
            if alpha_l1 > 0: penalty += alpha_l1 * torch.norm(model.linear.weight, p=1)
            
            if input_format == 'raw':
                w_sst = model.linear.weight[:, :in_features_sst]
                if alpha_tik > 0: penalty += alpha_tik * spatial_penalty_tikhonov(w_sst, len(sst_lags_months), 85, 360)
                if alpha_lap > 0: penalty += alpha_lap * spatial_penalty_laplacian(w_sst, len(sst_lags_months), 85, 360)
                    
                if len(slp_lags_months) > 0:
                    w_slp = model.linear.weight[:, in_features_sst:]
                    if alpha_tik > 0: penalty += alpha_tik * spatial_penalty_tikhonov(w_slp, len(slp_lags_months), 113, 53)
                    if alpha_lap > 0: penalty += alpha_lap * spatial_penalty_laplacian(w_slp, len(slp_lags_months), 113, 53)
                
            loss = base_loss + penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            # --- INTRA-EPOCH EVALUATION SPATIALE ---
            if (epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set):
                model.eval()
                all_preds_latent, all_true_maps = [], []
                
                with torch.no_grad():
                    for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                        v_X_sst = v_X_sst.to(device, non_blocking=True)
                        if len(slp_lags_months) > 0:
                            v_X_slp = v_X_slp.to(device, non_blocking=True)
                        B, L, H, W = v_X_sst.shape

                        if input_format == 'pca':
                            v_X_sst_2d = v_X_sst.view(B * L, H, W)
                            v_sst_embed = encode_to_latent_gpu(
                                v_X_sst_2d, 'pca', sst_pca_dim, 
                                sst_pca_components_gpu[:sst_pca_dim], sst_pca_mean_gpu, wgts_sst_gpu, None
                            )
                            v_X_sst_tensor = v_sst_embed.view(B, L * sst_pca_dim)
                            if len(slp_lags_months) > 0:
                                v_X_slp_2d = v_X_slp.view(B * len(slp_lags_months), 53, 113)
                                v_slp_embed_entree = encode_to_latent_gpu(
                                    v_X_slp_2d, 'pca', 1, 
                                    slp_pca_components_gpu[:1], slp_pca_mean_gpu, wgts_slp_gpu, None
                                )
                                v_X_slp_tensor = v_slp_embed_entree.view(B, len(slp_lags_months) * 1)
                                v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                            else:
                                v_X_combined = v_X_sst_tensor
                        else:
                            v_X_sst_tensor = v_X_sst.view(B, -1).to(device, non_blocking=True)
                            if len(slp_lags_months) > 0:
                                v_X_slp_tensor = v_X_slp.view(B, -1).to(device, non_blocking=True)
                                v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                            else:
                                v_X_combined = v_X_sst_tensor
                        
                        v_pred = model(v_X_combined)
                        vp = get_median_prediction(v_pred, loss_type, quantiles, latent_dim) if loss_type == 'quantile' else v_pred
                        
                        all_preds_latent.append(vp)
                        all_true_maps.append(v_y_target)

                val_preds_latent = torch.cat(all_preds_latent, dim=0)
                val_true_maps = torch.cat(all_true_maps, dim=0).to(device, non_blocking=True)

                # DÉCODAGE SPATIAL 
                val_pred_maps = decode_to_spatial_map_gpu(val_preds_latent, args.embed_method, slp_pca_components_gpu[:latent_dim], slp_pca_mean_gpu, wgts_slp_gpu, vae_model)
                i_r2, i_l1, i_corr = compute_targeted_spatial_metrics(val_pred_maps, val_true_maps, wgts_slp_gpu)

                current_step = epoch + batch_idx / total_batches
                metrics_history.append((current_step, i_r2, i_l1, i_corr))

                metrics_dict = {'R2': i_r2, 'L1': i_l1, 'correlation': i_corr}
                current_metric = metrics_dict[args.optimize_metric]
                
                best_trial_R2 = max(best_trial_R2, i_r2)
                best_trial_L1 = max(best_trial_L1, i_l1)
                best_trial_corr = max(best_trial_corr, i_corr)
                
                if current_metric > best_target_metric:
                    best_target_metric = current_metric
                    best_model_state = copy.deepcopy(model.state_dict())

                model.train() 

        # --- END OF EPOCH EVALUATION SPATIALE ---
        model.eval()
        all_preds_latent, all_true_maps = [], []
        
        with torch.no_grad():
            for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader:
                v_X_sst = v_X_sst.to(device, non_blocking=True)
                if len(slp_lags_months) > 0:
                    v_X_slp = v_X_slp.to(device, non_blocking=True)
                B, L, H, W = v_X_sst.shape
                
                if input_format == 'pca':
                    v_X_sst_2d = v_X_sst.view(B * L, H, W)
                    v_sst_embed = encode_to_latent_gpu(
                        v_X_sst_2d, 'pca', sst_pca_dim, 
                        sst_pca_components_gpu[:sst_pca_dim], sst_pca_mean_gpu, wgts_sst_gpu, None
                    )
                    v_X_sst_tensor = v_sst_embed.view(B, L * sst_pca_dim)
                    if len(slp_lags_months) > 0:
                        v_X_slp_2d = v_X_slp.view(B * len(slp_lags_months), 53, 113)
                        v_slp_embed_entree = encode_to_latent_gpu(
                            v_X_slp_2d, 'pca', 1, 
                            slp_pca_components_gpu[:1], slp_pca_mean_gpu, wgts_slp_gpu, None
                        )
                        v_X_slp_tensor = v_slp_embed_entree.view(B, len(slp_lags_months) * 1)
                        v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                    else:
                        v_X_combined = v_X_sst_tensor
                else:
                    v_X_sst_tensor = v_X_sst.view(B, -1).to(device, non_blocking=True)
                    if len(slp_lags_months) > 0:
                        v_X_slp_tensor = v_X_slp.view(B, -1).to(device, non_blocking=True)
                        v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                    else:
                        v_X_combined = v_X_sst_tensor
                
                v_pred = model(v_X_combined)
                vp = get_median_prediction(v_pred, loss_type, quantiles, latent_dim) if loss_type == 'quantile' else v_pred
                all_preds_latent.append(vp)
                all_true_maps.append(v_y_target)

        val_preds_latent = torch.cat(all_preds_latent, dim=0)
        val_true_maps = torch.cat(all_true_maps, dim=0).to(device, non_blocking=True)

        val_pred_maps = decode_to_spatial_map_gpu(val_preds_latent, args.embed_method, slp_pca_components_gpu[:latent_dim], slp_pca_mean_gpu, wgts_slp_gpu, vae_model)
        e_r2, e_l1, e_corr = compute_targeted_spatial_metrics(val_pred_maps, val_true_maps, wgts_slp_gpu)

        metrics_history.append((epoch+1, e_r2, e_l1, e_corr))
        metrics_dict = {'R2': e_r2, 'L1': e_l1, 'correlation': e_corr}
        current_metric = metrics_dict[args.optimize_metric]

        best_trial_R2 = max(best_trial_R2, e_r2)
        best_trial_L1 = max(best_trial_L1, e_l1)
        best_trial_corr = max(best_trial_corr, e_corr)

        if current_metric > best_target_metric:
            best_target_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        trial.report(best_target_metric, epoch)
        
        if patience_counter >= args.patience:
            break
    
    # --- TEST SPATIAL AVEC LE MEILLEUR MODÈLE ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    model.eval()
    all_preds_latent, all_true_maps = [], []
    with torch.no_grad():
        for v_X_sst, v_X_slp, v_y_target, _, _, _ in testloader:
            v_X_sst = v_X_sst.to(device, non_blocking=True)
            if len(slp_lags_months) > 0:
                v_X_slp = v_X_slp.to(device, non_blocking=True)
            B, L, H, W = v_X_sst.shape
            
            if input_format == 'pca':
                v_X_sst_2d = v_X_sst.view(B * L, H, W)
                v_sst_embed = encode_to_latent_gpu(
                    v_X_sst_2d, 'pca', sst_pca_dim, 
                    sst_pca_components_gpu[:sst_pca_dim], sst_pca_mean_gpu, wgts_sst_gpu, None
                )
                v_X_sst_tensor = v_sst_embed.view(B, L * sst_pca_dim)
                if len(slp_lags_months) > 0:
                    v_X_slp_2d = v_X_slp.view(B * len(slp_lags_months), 53, 113)
                    v_slp_embed_entree = encode_to_latent_gpu(
                        v_X_slp_2d, 'pca', 1, 
                        slp_pca_components_gpu[:1], slp_pca_mean_gpu, wgts_slp_gpu, None
                    )
                    v_X_slp_tensor = v_slp_embed_entree.view(B, len(slp_lags_months) * 1)
                    v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                else:
                    v_X_combined = v_X_sst_tensor
            else:
                v_X_sst_tensor = v_X_sst.view(B, -1).to(device, non_blocking=True)
                if len(slp_lags_months) > 0:
                    v_X_slp_tensor = v_X_slp.view(B, -1).to(device, non_blocking=True)
                    v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                else:
                    v_X_combined = v_X_sst_tensor
            
            v_pred = model(v_X_combined)
            vp = get_median_prediction(v_pred, loss_type, quantiles, latent_dim) if loss_type == 'quantile' else v_pred
            all_preds_latent.append(vp)
            all_true_maps.append(v_y_target)

    test_preds_latent = torch.cat(all_preds_latent, dim=0)
    test_true_maps = torch.cat(all_true_maps, dim=0).to(device, non_blocking=True)

    test_pred_maps = decode_to_spatial_map_gpu(test_preds_latent, args.embed_method, slp_pca_components_gpu[:latent_dim], slp_pca_mean_gpu, wgts_slp_gpu, vae_model)
    t_r2, t_l1, t_corr = compute_targeted_spatial_metrics(test_pred_maps, test_true_maps, wgts_slp_gpu)

    trial.set_user_attr("best_test_R2", t_r2)
    trial.set_user_attr("best_test_L1", t_l1)
    trial.set_user_attr("best_test_corr", t_corr)
    trial.set_user_attr("R2_L1_corr_history", metrics_history)

    del model, optimizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return best_target_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize_metric', type=str, choices=['R2', 'L1', 'correlation'], default='R2')
    parser.add_argument('--embed_method', type=str, default='pca', choices=['pca', 'vae'])
    parser.add_argument('--embed_path', type=str, required=True)
    parser.add_argument('--embed_path_sst', type=str, default=None)
    parser.add_argument('--machine', type=str, default='jean-zay-work')
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--nb_intra_evals', type=int, default=5)
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--include_lag1', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')

    # HYPERPARAMÈTRES FIXÉS PAR L'UTILISATEUR (issus de la recherche HPO)
    parser.add_argument('--latent_dim', type=int, required=True, help="Dimension latente cible (Fixe)")
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--alpha_l1', type=float, default=0.0)
    parser.add_argument('--alpha_tik', type=float, default=0.0)
    parser.add_argument('--alpha_lap', type=float, default=0.0)
    parser.add_argument('--noise_std', type=float, required=True)
    parser.add_argument('--gradient_clip', type=float, required=True)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1','correlation','quantile'], required=True)
    parser.add_argument('--input_format', type=str, choices=['raw', 'pca'], required=True)
    parser.add_argument('--sst_pca_dim', type=int, default=0)
    parser.add_argument('--sst_lags_months', type=int, nargs='*', required=True)
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    
    args = parser.parse_args()

    # ============================================================
    # 1. RÉCUPÉRATION DU SLP_STD & SST_STD
    # ============================================================
    if args.machine == 'hacienda': base_home = "/home/moysan/stage_isir_jz/linear_models/optuna_spatial/loocv_spatial/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']: base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/optuna_spatial/loocv_spatial/"
    elif args.machine == 'mac_local': base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/linear_models/optuna_spatial/loocv_spatial/"

    dynamic_slp_std = 596.0 
    match = re.search(r'slp_std([0-9.]+)', args.embed_path)
    if match: dynamic_slp_std = float(match.group(1))

    dynamic_sst_std = 0.707
    if args.embed_path_sst:
        match = re.search(r'sst_std([0-9.]+)', args.embed_path_sst)
        if match: dynamic_sst_std = float(match.group(1))

    # ============================================================
    # 2. PRÉPARATION DU MODÈLE D'EMBEDDING (HORS BOUCLE)
    # ============================================================
    pca_model, vae_model = None, None
    slp_pca_mean_gpu, slp_pca_components_gpu = None, None

    if args.embed_method == 'pca':
        pca_model = joblib.load(args.embed_path) 
        slp_pca_mean_gpu = torch.tensor(pca_model.mean_, dtype=torch.float32, device=device)
        slp_pca_components_gpu = torch.tensor(pca_model.components_[:max(args.latent_dim, 1)], dtype=torch.float32, device=device)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=args.latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

    wgts_slp_gpu = None
    if args.lat_weight:
        sample_path = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-1001.001_1mo.nc"
        try:
            with xr.open_dataset(sample_path) as ds_sample:
                coslat = np.cos(np.deg2rad(ds_sample['lat'].values)).clip(0., 1.)
                h, w = len(coslat), len(ds_sample['lon'].values)
                wgts_flat = np.broadcast_to(np.sqrt(coslat).reshape(h, 1), (h, w)).flatten()
                wgts_slp_gpu = torch.tensor(wgts_flat, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"⚠️ Erreur chargement poids latitude SLP : {e}")

    sst_pca_model = None
    sst_pca_mean_gpu, sst_pca_components_gpu = None, None
    if args.embed_path_sst:
        sst_pca_model = joblib.load(args.embed_path_sst)
        sst_pca_mean_gpu = torch.tensor(sst_pca_model.mean_, dtype=torch.float32, device=device)
        sst_pca_components_gpu = torch.tensor(sst_pca_model.components_, dtype=torch.float32, device=device)

    wgts_sst_gpu = None
    if args.lat_weight:
        sample_path_sst = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/SST_anom_LE2-1001.001_T_regrid_1mo.nc"
        try:
            with xr.open_dataset(sample_path_sst) as ds_sample_sst:
                ds_sample_sst = ds_sample_sst.sel(lat=slice(-15,70))
                coslat_sst = np.cos(np.deg2rad(ds_sample_sst['lat'].values)).clip(0., 1.)
                h_sst, w_sst = len(coslat_sst), len(ds_sample_sst['lon'].values)
                if h_sst * w_sst != 30600:
                    raise ValueError(f"Le crop SST donne {h_sst*w_sst} au lieu de 30600")
                wgts_flat_sst = np.broadcast_to(np.sqrt(coslat_sst).reshape(h_sst, 1), (h_sst, w_sst)).flatten()
                wgts_sst_gpu = torch.tensor(wgts_flat_sst, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"⚠️ Erreur chargement poids latitude SST : {e}")

    # ============================================================
    # NOMMAGE SÉCURISÉ (Hash si trop long)
    # ============================================================
    base_name = f"LOOCV_LinReg_{args.embed_method}_{args.optimize_metric}latent_{args.latent_dim}_m{''.join(map(str, args.winter_months))}roll{args.roll_sst}latw{args.lat_weight}ep{args.nb_epochs}ie{args.nb_intra_evals}pat{args.patience}"
    short = {
        'bs': 'bs', 'lr': 'lr','loss_type': 'loss', 'input_format': 'input',
        'weight_decay': 'wd', 'alpha_l1': 'l1', 'alpha_tik': 'tik', 'alpha_lap': 'lap', 'noise_std': 'ns', 'gradient_clip': 'gc', 'sst_pca_dim': 'pcasst', 'sst_lags_months': 'sst','slp_lags_months': 'slp'
    }
    
    fixed = []
    for k, v in sorted(vars(args).items()):
        if k in short and v is not None:
            if isinstance(v, list):
                val_str = ''.join(map(str, v))
            elif isinstance(v, float):
                val_str = f"{v:.1e}" if v < 1e-3 else f"{v:.3f}"
            else:
                val_str = str(v)
            fixed.append(f"{short[k]}{val_str}")

    dynamic_name = f"{base_name}_{'_'.join(fixed)}"
    
    if len(dynamic_name) > 230:
        short_hash = hashlib.md5(dynamic_name.encode()).hexdigest()[:6]
        dynamic_name = dynamic_name[:220] + "_h" + short_hash

    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, "loocv_linreg_spatial.db")
    storage_name = f"sqlite:///{db_path}"

    # ============================================================
    # 4. LANCEMENT DE LA GRID SEARCH OPTUNA
    # ============================================================
    search_space = {
        "val_member": VAL_GRID_MEMBERS,
        "test_member": TEST_GRID_MEMBERS
    }
    
    sampler = optuna.samplers.GridSampler(search_space)
    
    study = optuna.create_study(
        study_name=dynamic_name, 
        storage=storage_name, 
        load_if_exists=True,
        direction="maximize", 
        sampler=sampler
    )
    
    print(f"\n🚀 Début du LOOCV GridSearch ({len(VAL_GRID_MEMBERS)}x{len(TEST_GRID_MEMBERS)} = {len(VAL_GRID_MEMBERS) * len(TEST_GRID_MEMBERS)} paires)...")
    print(f"📁 Dossier de sortie : {output_dir}")
    
    study.optimize(objective)

    # ============================================================
    # 5. GÉNÉRATION DES MATRICES FINALES
    # ============================================================
    for metric_key in ['best_test_R2', 'best_test_L1', 'best_test_corr']:
        generate_crossval_matrix(study, output_dir, metric_key)
        if len(VAL_GRID_MEMBERS) == 1:
            generate_1d_loocv_heatmap(study, output_dir, metric_key, VAL_GRID_MEMBERS[0])