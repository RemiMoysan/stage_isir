import os
import time
import argparse
import joblib
import numpy as np
import random 
import re
import xarray as xr
import hashlib
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys 
from pathlib import Path
import optuna
from optuna.samplers import TPESampler

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from shared_tools.datasets import Dataset_mensuel
from shared_tools.models import compute_loss, get_median_prediction, spatial_penalty_tikhonov, spatial_penalty_laplacian
from shared_tools.optuna_loop_helpers import encode_to_latent_gpu, compute_targeted_embedding_metrics

# ============================================================
# ARCHITECTURE ADAPTÉE (Entrée 1D générique)
# ============================================================
class LinearRegressionPredictor(nn.Module):
    def __init__(self, in_features, out_dim=128):
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim)

    def forward(self, x):
        # x est déjà aplati : [B, in_features]
        return self.linear(x)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

def objective(trial):
    # ============================================================
    # 1. DÉFINITION DE L'ESPACE DE RECHERCHE
    # ============================================================
    # Lags (Aligné sur le CNN)
    first_month = 1 if args.include_lag1 else 2
    if args.sst_lags_months is not None:
        sst_lags_months = args.sst_lags_months
    elif args.sequential_lags:
        n_sst = trial.suggest_int("num_sst_lags", 1, 12 - first_month + 1)
        sst_lags_months = list(range(first_month, first_month + n_sst))
    else:
        sst_lags_months = [m for m in range(first_month, 13) if trial.suggest_categorical(f"use_sst_lag_{m}", [True, False])]

    if args.slp_lags_months is not None:
        slp_lags_months = args.slp_lags_months
    elif args.sequential_lags:
        n_slp = trial.suggest_int("num_slp_lags", 0, 6 - first_month + 1)
        slp_lags_months = list(range(first_month, first_month + n_slp)) if n_slp > 0 else []
    else:
        slp_lags_months = [m for m in range(first_month, 6) if trial.suggest_categorical(f"use_slp_lag_{m}", [True, False])]

    if len(sst_lags_months) == 0 and len(slp_lags_months) == 0:
        return -float('inf')

    trial.set_user_attr("sst_lags_final", sst_lags_months)
    trial.set_user_attr("slp_lags_final", slp_lags_months)

    input_format = args.input_format if args.input_format is not None else trial.suggest_categorical("input_format", ["raw", "pca"])
    sst_pca_dim = trial.suggest_int("sst_pca_dim", 1, 128) if input_format == "pca" else 0
    
    loss_type = args.loss_type if args.loss_type is not None else trial.suggest_categorical("loss_type", ["mse", "l1", "quantile", "correlation"]) 
    if args.bs is not None:
        bs = args.bs
    else:
        exp = trial.suggest_int("bs_exp", 5, 7)
        bs = int(2**exp)
    lr = args.lr if args.lr is not None else trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = args.weight_decay if args.weight_decay is not None else trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    noise_std = args.noise_std if args.noise_std is not None else trial.suggest_float("noise_std", 1e-4, 1e-1, log=True)
    grad_clip = args.gradient_clip if args.gradient_clip is not None else trial.suggest_float("grad_clip", 0.1, 1000.0, log=True)
    
    alpha_l1 = args.alpha_l1 if args.alpha_l1 is not None else trial.suggest_float("alpha_l1", 1e-9, 10.0, log=True) 
    
    if input_format == "raw":
        alpha_tik = args.alpha_tik if args.alpha_tik is not None else trial.suggest_float("alpha_tik", 1e-9, 10.0, log=True)
        alpha_lap = args.alpha_lap if args.alpha_lap is not None else trial.suggest_float("alpha_lap", 1e-9, 10.0, log=True)
    else:
        alpha_tik, alpha_lap = 0.0, 0.0

    # ============================================================
    # 2. PRÉPARATION DES DONNÉES
    # ============================================================
    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 2)) - 1)

    training_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=True, noise_std=noise_std)
    val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=False)
    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=False)

    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # Dimensions du modèle
    if input_format == 'pca':
        in_features_sst = len(sst_lags_months) * sst_pca_dim
        in_features_slp = len(slp_lags_months) * 1 # PCA1 seulement en entrée
    else:
        in_features_sst = len(sst_lags_months) * 85 * 360
        in_features_slp = len(slp_lags_months) * 53 * 113

    out_features = args.latent_dim * len(args.quantiles) if loss_type == 'quantile' else args.latent_dim

    model = LinearRegressionPredictor(in_features=in_features_sst + in_features_slp, out_dim=out_features).to(device)
    trial.set_user_attr("num_params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimiseur AdamW unifié (remplace alpha_l2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # ============================================================
    # 3. BOUCLE D'ENTRAÎNEMENT & ÉVALUATION
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
            # Formattage Entrées (L étant le nombre de lags)
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
                        pca_components_gpu[:1], pca_mean_gpu, wgts_gpu, None
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
            
            # Cible unifiée GPU via Helper
            target_embed = encode_to_latent_gpu(y_target.to(device, non_blocking=True), 'pca', args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, None)
                
            pred = model(X_combined)
            base_loss = compute_loss(pred, target_embed, loss_type=loss_type, quantiles=args.quantiles, reduction='mean')
            
            # Pénalités Spatiales (si mode Raw)
            penalty = 0.0
            if alpha_l1 > 0: penalty += alpha_l1 * torch.norm(model.linear.weight, p=1)
            
            if input_format == 'raw':
                w_sst = model.linear.weight[:, :in_features_sst]
                if alpha_tik > 0:
                    penalty += alpha_tik * spatial_penalty_tikhonov(w_sst, len(sst_lags_months), 85, 360)
                if alpha_lap > 0:
                    penalty += alpha_lap * spatial_penalty_laplacian(w_sst, len(sst_lags_months), 85, 360)
                    
                if len(slp_lags_months) > 0:
                    w_slp = model.linear.weight[:, in_features_sst:]
                    if alpha_tik > 0:
                        penalty += alpha_tik * spatial_penalty_tikhonov(w_slp, len(slp_lags_months), 113, 53)
                    if alpha_lap > 0:
                        penalty += alpha_lap * spatial_penalty_laplacian(w_slp, len(slp_lags_months), 113, 53)
                
            loss = base_loss + penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        
            # --- INTRA-EPOCH EVALUATION ---
            if (epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set):
                model.eval()
                all_preds_latent, all_targets_latent = [], []
                
                with torch.no_grad():
                    for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                        B, L, H, W = v_X_sst.shape
                        v_X_sst = v_X_sst.to(device, non_blocking=True)
                        if len(slp_lags_months) > 0:
                            v_X_slp = v_X_slp.to(device, non_blocking=True)

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
                                    pca_components_gpu[:1], pca_mean_gpu, wgts_gpu, None
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
                        
                        v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), 'pca', args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, None)
                        v_pred = model(v_X_combined)
                        vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                        
                        all_preds_latent.append(vp)
                        all_targets_latent.append(v_target_embed)

                val_preds_latent = torch.cat(all_preds_latent, dim=0)
                val_targets_latent = torch.cat(all_targets_latent, dim=0)

                i_r2, i_l1, i_corr = compute_targeted_embedding_metrics(val_preds_latent, val_targets_latent)

                best_trial_R2 = max(best_trial_R2, i_r2)
                best_trial_L1 = max(best_trial_L1, i_l1)
                best_trial_corr = max(best_trial_corr, i_corr)

                metrics_dict = {'R2': i_r2, 'L1': i_l1, 'correlation': i_corr}
                current_metric = metrics_dict[args.optimize_metric]
                
                current_step = epoch + batch_idx / total_batches
                metrics_history.append((current_step, i_r2, i_l1, i_corr))

                if current_metric > best_target_metric:
                    best_target_metric = current_metric
                    best_model_state = copy.deepcopy(model.state_dict())

        # --- END OF EPOCH EVALUATION ---
        model.eval()
        all_preds_latent, all_targets_latent = [], []
        
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
                            pca_components_gpu[:1], pca_mean_gpu, wgts_gpu, None
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
                
                v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), 'pca', args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, None)
                v_pred = model(v_X_combined)
                vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                
                all_preds_latent.append(vp)
                all_targets_latent.append(v_target_embed)

        val_preds_latent = torch.cat(all_preds_latent, dim=0)
        val_targets_latent = torch.cat(all_targets_latent, dim=0)

        e_r2, e_l1, e_corr = compute_targeted_embedding_metrics(val_preds_latent, val_targets_latent)

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

        trial.report(current_metric, epoch)
        if trial.should_prune():
            trial.set_user_attr("best_trial_R2", best_trial_R2)
            trial.set_user_attr("best_trial_L1", best_trial_L1)
            trial.set_user_attr("best_trial_corr", best_trial_corr)
            trial.set_user_attr("R2_L1_corr_history", metrics_history)
            raise optuna.exceptions.TrialPruned()
            
        if patience_counter >= args.patience:
            break
    
    trial.set_user_attr("best_trial_R2", best_trial_R2)
    trial.set_user_attr("best_trial_L1", best_trial_L1)
    trial.set_user_attr("best_trial_corr", best_trial_corr)
    trial.set_user_attr("R2_L1_corr_history", metrics_history)

    # --- TEST AVEC LE MEILLEUR MODÈLE ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    model.eval()
    all_preds_latent, all_targets_latent = [], []
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
                        pca_components_gpu[:1], pca_mean_gpu, wgts_gpu, None
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
            
            v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), 'pca', args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, None)
            v_pred = model(v_X_combined)
            vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
            
            all_preds_latent.append(vp)
            all_targets_latent.append(v_target_embed)

    test_preds_latent = torch.cat(all_preds_latent, dim=0)
    test_targets_latent = torch.cat(all_targets_latent, dim=0)

    t_r2, t_l1, t_corr = compute_targeted_embedding_metrics(test_preds_latent, test_targets_latent)

    trial.set_user_attr("best_test_R2", t_r2)
    trial.set_user_attr("best_test_L1", t_l1)
    trial.set_user_attr("best_test_corr", t_corr)
    
    print(f"Trial terminé en {time.time() - start_time:.2f} s | Target Metric: {best_target_metric:.4f}")
    return best_target_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--n_startup_trials_tpe', type=int, default=20)
    parser.add_argument('--n_startup_trials_pruner', type=int, default=10)
    parser.add_argument('--n_warmup_steps', type=int, default=3)
    parser.add_argument('--interval_steps', type=int, default=1)

    parser.add_argument('--optimize_metric', type=str, choices=['R2', 'L1', 'correlation'], default='correlation', help="Métrique à maximiser")
    parser.add_argument('--embed_path', type=str, required=True, help="Chemin vers le PCA SLP (Target)")
    parser.add_argument('--embed_path_sst', type=str, default=None, help="Chemin vers le PCA SST (Optionnel)")
    parser.add_argument('--machine', type=str, default='jean-zay-work')
    parser.add_argument('--nb_members_val', type=int, default=5, help="Nombre de membres réservés à la validation")
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--fixed_seed', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=1, help="Nombre de composantes à prédire")
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    
    # Booléens structurés
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--include_lag1', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--sequential_lags', action='store_true')

    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--nb_intra_evals', type=int, default=5)
    
    # Paramètres dynamiques (peuvent être fixés)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--alpha_l1', type=float, default=None)
    parser.add_argument('--alpha_tik', type=float, default=None)
    parser.add_argument('--alpha_lap', type=float, default=None)
    parser.add_argument('--noise_std', type=float, default=None)
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1','correlation','quantile'], default=None)
    parser.add_argument('--input_format', type=str, choices=['raw', 'pca'], default=None)
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=None)
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=None)
    
    args = parser.parse_args()

    if args.machine == 'hacienda': base_home = "/home/moysan/stage_isir_jz/linear_models/optuna_embedding/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']: base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/optuna_embedding/"
    elif args.machine == 'mac_local': base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/linear_models/optuna_embedding/"

    dynamic_slp_std = 596.0 
    match = re.search(r'slp_std([0-9.]+)', args.embed_path)
    if match: dynamic_slp_std = float(match.group(1))

    dynamic_sst_std = 0.707
    if args.embed_path_sst:
        match = re.search(r'sst_std([0-9.]+)', args.embed_path_sst)
        if match: dynamic_sst_std = float(match.group(1))

    print(f"Dynamic SLP std: {dynamic_slp_std}, Dynamic SST std: {dynamic_sst_std}")    

    # Chargement Global de l'Embedder Target (SLP)
    slp_pca_model = joblib.load(args.embed_path) 
    pca_mean_gpu = torch.tensor(slp_pca_model.mean_, dtype=torch.float32, device=device)
    pca_components_gpu = torch.tensor(slp_pca_model.components_[:args.latent_dim], dtype=torch.float32, device=device)

    # Pondération Latitudes (si demandé)
    wgts_gpu = None
    if args.lat_weight:
        sample_path = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-1001.001_1mo.nc"
        try:
            with xr.open_dataset(sample_path) as ds_sample:
                coslat = np.cos(np.deg2rad(ds_sample['lat'].values)).clip(0., 1.)
                h, w = len(coslat), len(ds_sample['lon'].values)
                wgts_flat = np.broadcast_to(np.sqrt(coslat).reshape(h, 1), (h, w)).flatten()
                wgts_gpu = torch.tensor(wgts_flat, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"⚠️ Erreur chargement poids latitude : {e}")

    # Chargement du modèle PCA SST et transfert sur GPU
    sst_pca_model = None
    sst_pca_mean_gpu, sst_pca_components_gpu = None, None
    if args.embed_path_sst:
        sst_pca_model = joblib.load(args.embed_path_sst)
        sst_pca_mean_gpu = torch.tensor(sst_pca_model.mean_, dtype=torch.float32, device=device)
        sst_pca_components_gpu = torch.tensor(sst_pca_model.components_, dtype=torch.float32, device=device)

    # Pondération Latitudes de la SST 
    wgts_sst_gpu = None
    if args.lat_weight:
        sample_path_sst = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/SST_anom_LE2-1001.001_T_regrid_1mo.nc"
        try:
            with xr.open_dataset(sample_path_sst) as ds_sample_sst:
                ds_sample_sst = ds_sample_sst.sel(lat=slice(-15,70))
                coslat_sst = np.cos(np.deg2rad(ds_sample_sst['lat'].values)).clip(0., 1.)
                h_sst, w_sst = len(coslat_sst), len(ds_sample_sst['lon'].values)
                wgts_flat_sst = np.broadcast_to(np.sqrt(coslat_sst).reshape(h_sst, 1), (h_sst, w_sst)).flatten()
                wgts_sst_gpu = torch.tensor(wgts_flat_sst, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"⚠️ Erreur chargement poids latitude SST : {e}")

    # Splits fixes
    rng = random.Random(args.fixed_seed)
    members_shuffled = ALL_MEMBERS.copy()
    rng.shuffle(members_shuffled)
    train_members = members_shuffled[:-2*args.nb_members_val]
    val_members = members_shuffled[-args.nb_members_val:]
    test_members = members_shuffled[-2*args.nb_members_val:-args.nb_members_val] 

    # ============================================================
    # NOMMAGE SÉCURISÉ (Hash si trop long)
    # ============================================================
    base_name = f"study_{args.optimize_metric}latent_{args.latent_dim}_m{''.join(map(str, args.winter_months))}_ep{args.nb_epochs}ie{args.nb_intra_evals}pat{args.patience}val{args.nb_members_val}lag1{args.include_lag1}seq{args.sequential_lags}roll{args.roll_sst}latw{args.lat_weight}seed{args.fixed_seed}"
    short = {
        'bs': 'bs', 'lr': 'lr','loss_type': 'loss', 'input_format': 'input',
        'weight_decay': 'wd', 'alpha_l1': 'l1', 'alpha_tik': 'tik', 'alpha_lap': 'lap', 'noise_std': 'ns', 'gradient_clip': 'gc', 'sst_lags_months': 'sst','slp_lags_months': 'slp',
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

    dynamic_name = f"{base_name}_FIXED_{'_'.join(fixed)}" if fixed else f"{base_name}_full_search"
    dynamic_name += f"_opt_s{args.n_startup_trials_tpe}p{args.n_startup_trials_pruner}_{args.n_warmup_steps}i{args.interval_steps}"
    
    if len(dynamic_name) > 230:
        short_hash = hashlib.md5(dynamic_name.encode()).hexdigest()[:6]
        dynamic_name = dynamic_name[:220] + "_h" + short_hash

    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, "optuna.db")
    storage_name = f"sqlite:///{db_path}"

    sampler = TPESampler(seed=args.fixed_seed, n_startup_trials=args.n_startup_trials_tpe)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials_pruner, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps)

    study = optuna.create_study(
        study_name=dynamic_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True, 
        sampler=sampler,
        pruner=pruner
    )
    
    print(f"Début de l'optimisation LinReg EMBEDDING ({args.n_trials} essais)...")
    study.optimize(objective, n_trials=args.n_trials) 
    
    print("\n=== Bilan HPO EMBEDDING ===")
    trial = study.best_trial
    print(f"  Meilleur {args.optimize_metric.upper()} (Validation) : {trial.value:.4f}")
    
    print("\n  --- Performances sur le Set de TEST (Caché) ---")
    print(f"  Global R2 Score               : {trial.user_attrs.get('best_test_R2'):.4f}")
    print(f"  Global L1 Skill Score         : {trial.user_attrs.get('best_test_L1'):.4f}")
    print(f"  Global Correlation            : {trial.user_attrs.get('best_test_corr'):.4f}")
    
    print("\n  --- Meilleurs Hyperparamètres ---")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    study.trials_dataframe().to_csv(os.path.join(output_dir, "optuna_results.csv"), index=False)