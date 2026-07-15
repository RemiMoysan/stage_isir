import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import random 
import re
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import sys 
from pathlib import Path
import optuna
import copy
from optuna.samplers import TPESampler

project_root = Path(__file__).resolve().parent.parent.parent
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

from tools.datasets import Dataset_mensuel
from tools.models import compute_loss, get_median_prediction, spatial_penalty_tikhonov, spatial_penalty_laplacian


# ============================================================
# ARCHITECTURE ADAPTÉE (Entrée 1D générique car doit être éventuellement compatible avec pca)
# ============================================================
class LinearRegressionPredictor(nn.Module):
    def __init__(self, in_features, out_dim=128):
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim)

    def forward(self, x):
        # x est déjà aplati : [B, in_features]
        return self.linear(x)


ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

def objective(trial):
    # ============================================================
    # 1. DÉFINITION DE L'ESPACE DE RECHERCHE
    # ============================================================
    # Gestion des Lags
    if args.sst_lags_months is not None:
        sst_lags_months = args.sst_lags_months
    else:
        sst_lags_months = []
        first_month = 2 if not args.include_lag1 else 1
        for month in range(first_month, 13):
            if trial.suggest_categorical(f"use_sst_lag_{month}", [True, False]):
                sst_lags_months.append(month)

    if args.slp_lags_months is not None:
        slp_lags_months = args.slp_lags_months
    else:
        slp_lags_months = []
        first_month = 2 if not args.include_lag1 else 1
        for month in range(first_month, 6):
            if trial.suggest_categorical(f"use_slp_lag_{month}", [True, False]):
                slp_lags_months.append(month)

    if len(sst_lags_months) == 0 and len(slp_lags_months) == 0:
        return -float('inf') # Pire score possible 

    trial.set_user_attr("sst_lags_final", sst_lags_months)
    trial.set_user_attr("slp_lags_final", slp_lags_months)

    input_format = args.input_format if args.input_format is not None else trial.suggest_categorical("input_format", ["raw", "pca"])
    sst_pca_dim = trial.suggest_int("sst_pca_dim", 10, 128) if input_format == "pca" else 0
    
    loss_type = args.loss_type if args.loss_type is not None else trial.suggest_categorical("loss_type", ["mse", "l1", "quantile", "correlation"]) 
    bs = args.bs if args.bs is not None else trial.suggest_categorical("bs", [32, 64, 128])
    lr = args.lr if args.lr is not None else trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    
    # Pénalités (Tikhonov/Laplacien impossibles sur une PCA)
    # j'ai enlevé les if trial.suggest_categorical("use_l1", [True, False]) else 0.0 pour simplifier
    alpha_l1 = trial.suggest_float("alpha_l1", 1e-9, 10.0, log=True) 
    alpha_l2 = trial.suggest_float("alpha_l2", 1e-9, 10.0, log=True)
    
    if input_format == "raw":
        alpha_tik = trial.suggest_float("alpha_tik", 1e-9, 10.0, log=True)
        alpha_lap = trial.suggest_float("alpha_lap", 1e-9, 10.0, log=True)
    else:
        alpha_tik, alpha_lap = 0.0, 0.0

    # ============================================================
    # 2. PRÉPARATION DES DONNÉES
    # ============================================================
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)

    val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)
    training_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)

    # NOUVEAU : Création du test set
    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std)

    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)

    # NOUVEAU : Création du testloader
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # Dimensions du modèle
    # Dimensions du modèle
    if input_format == 'pca':
        in_features_sst = len(sst_lags_months) * sst_pca_dim
        in_features_slp = len(slp_lags_months) * 1 # on part sur 1 car de toute façon on veut predire la première compo au jour J 
    else:
        in_features_sst = len(sst_lags_months) * 85 * 360
        in_features_slp = len(slp_lags_months) * 53 * 113

    out_features = args.latent_dim * len(args.quantiles) if loss_type == 'quantile' else args.latent_dim

    model = LinearRegressionPredictor(in_features=in_features_sst + in_features_slp, out_dim=out_features).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("num_params", num_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # ============================================================
    # 3. BOUCLE D'ENTRAÎNEMENT & ÉVALUATION
    # ============================================================
    best_target_metric = -float('inf')
    best_trial_mse = float('inf')
    best_trial_corr = -float('inf')
    best_r2_score = -float('inf')
    history = []
    patience_counter = 0
    best_model_state = None

    # Calcul des steps intra-epoch pour la première époque
    total_batches = len(trainloader)
    eval_steps_set = set(np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int))
    eval_steps_set.add(0)

    eval_steps_epoch2 = np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int)
    eval_steps_epoch2 = np.insert(eval_steps_epoch2, 0, 0)
    eval_steps_epoch2_set = set(eval_steps_epoch2)

    for epoch in range(args.nb_epochs):
        model.train()
        for batch_idx, (X_sst, X_slp, y_target, _, _, _) in enumerate(trainloader):
            optimizer.zero_grad()
            
            # Formattage Entrée SST
            B, L, H, W = X_sst.shape
            B, L2, H2, W2 = X_slp.shape
            
            if input_format == 'pca':
                # On aplatit spatialement en fusionnant Batch et Lags : [B*L, 85*360]
                X_sst_reshaped = X_sst.view(B * L, -1).numpy()
                
                # Transform PCA puis slicing
                pca_trans = sst_pca_model.transform(X_sst_reshaped)[:, :sst_pca_dim]
                
                X_sst_tensor = torch.tensor(pca_trans, dtype=torch.float32).to(device, non_blocking=True)

                pca_trans_slp = slp_pca_model.transform(X_slp.view(B * len(slp_lags_months), -1).numpy())[:, :1]
                X_slp_tensor = torch.tensor(pca_trans_slp.reshape(B, len(slp_lags_months) * 1), dtype=torch.float32).to(device, non_blocking=True)
                X_combined = torch.cat((X_sst_tensor, X_slp_tensor), dim=1)
            else:
                # Si mode raw, on aplatit juste tout pour la couche linéaire
                X_sst_tensor = X_sst.view(B, -1).to(device, non_blocking=True)
                X_slp_tensor = X_slp.view(B, -1).to(device, non_blocking=True)
                X_combined = torch.cat((X_sst_tensor, X_slp_tensor), dim=1)
            
            # Formattage Cible
            slp_flat = y_target.view(y_target.size(0), -1).numpy()
            embed_np = slp_pca_model.transform(slp_flat)[:, :args.latent_dim]
            target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                
            pred = model(X_combined)
            
            # Calcul Loss & Pénalités
            base_loss = compute_loss(pred, target_embed, loss_type=loss_type, quantiles=args.quantiles, reduction='mean')
            
            penalty = 0.0
            if alpha_l1 > 0: penalty += alpha_l1 * torch.norm(model.linear.weight, p=1)
            if alpha_l2 > 0: penalty += alpha_l2 * torch.sum(model.linear.weight ** 2)
            if input_format == 'raw':
                w_sst = model.linear.weight[:, :in_features_sst]
                w_slp = model.linear.weight[:, in_features_sst:]

                penalty_tik = 0.0
                penalty_lap = 0.0
                if alpha_tik > 0:
                    penalty_tik = alpha_tik * (
                        spatial_penalty_tikhonov(w_sst, len(sst_lags_months), 85, 360)
                        + spatial_penalty_tikhonov(w_slp, len(slp_lags_months), 113, 53)
                    )
                if alpha_lap > 0:
                    penalty_lap = alpha_lap * (
                        spatial_penalty_laplacian(w_sst, len(sst_lags_months), 85, 360)
                        + spatial_penalty_laplacian(w_slp, len(slp_lags_months), 113, 53)
                    )
                penalty += penalty_tik + penalty_lap

                
            loss = base_loss + penalty
            loss.backward()
            optimizer.step()
        
        if (epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set):
            # Évaluation Intra epoch
            model.eval()
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                    # Formattage Entrée SST
                    B, L, H, W = v_X_sst.shape
                    
                    if input_format == 'pca':
                        # On aplatit spatialement en fusionnant Batch et Lags : [B*L, 85*360]
                        v_X_sst_reshaped = v_X_sst.view(B * L, -1).numpy()
                        
                        # Transform PCA puis slicing
                        pca_trans = sst_pca_model.transform(v_X_sst_reshaped)[:, :sst_pca_dim]
                        
                        v_X_sst_tensor = torch.tensor(pca_trans, dtype=torch.float32).to(device, non_blocking=True)
                        pca_trans_slp = slp_pca_model.transform(v_X_slp.view(B * len(slp_lags_months), -1).numpy())[:, :1]
                        v_X_slp_tensor = torch.tensor(pca_trans_slp.reshape(B, len(slp_lags_months) * 1), dtype=torch.float32).to(device, non_blocking=True)
                        v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                    else:
                        # Si mode raw, on aplatit juste tout pour la couche linéaire
                        v_X_sst_tensor = v_X_sst.view(B, -1).to(device, non_blocking=True)
                        v_X_slp_tensor = v_X_slp.view(B, -1).to(device, non_blocking=True)
                        v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                    
                    v_slp_flat = v_y_target.view(v_y_target.size(0), -1).cpu().numpy()
                    v_embed_np = slp_pca_model.transform(v_slp_flat)[:, :args.latent_dim]
                    v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                    
                    v_pred = model(v_X_combined)
                    vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                    
                    all_preds.append(vp)
                    all_targets.append(v_target_embed)

            val_preds_tensor = torch.cat(all_preds, dim=0)
            val_targets_tensor = torch.cat(all_targets, dim=0)

            # Calcul Métriques INTRA-ÉPOQUE
            intra_mse = F.mse_loss(val_preds_tensor, val_targets_tensor).item()
            intra_target_var = torch.var(val_targets_tensor, unbiased=False).item()
            intra_r2 = 1.0 - (intra_mse / intra_target_var) if intra_target_var > 0 else 0.0

            p, t = val_preds_tensor, val_targets_tensor
            p_mean, t_mean = p.mean(dim=0), t.mean(dim=0)
            p_var, t_var = ((p - p_mean)**2).mean(dim=0), ((t - t_mean)**2).mean(dim=0)
            cov = ((p - p_mean)*(t - t_mean)).mean(dim=0)
            intra_corr = (cov / torch.sqrt(p_var * t_var + 1e-8)).mean().item()

            # Mise à jour des meilleurs scores globaux
            if intra_r2 > best_r2_score: best_r2_score = intra_r2
            if intra_corr > best_trial_corr: best_trial_corr = intra_corr
            if intra_mse < best_trial_mse: best_trial_mse = intra_mse

            # Choix de la métrique d'optimisation
            current_metric = intra_r2 if args.optimize_metric == 'r2' else intra_corr
            
            # Sauvegarde avec le step fractionné (ex: 0.5 = milieu d'époque)
            current_step = epoch + batch_idx / total_batches
            history.append((current_step, intra_r2, intra_corr))

            if current_metric > best_target_metric:
                best_target_metric = current_metric
                best_model_state = copy.deepcopy(model.state_dict())

        # Évaluation Fin d'Époque
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader:
                    # Formattage Entrée SST
                B, L, H, W = v_X_sst.shape
                
                if input_format == 'pca':
                    # On aplatit spatialement en fusionnant Batch et Lags : [B*L, 85*360]
                    v_X_sst_reshaped = v_X_sst.view(B * L, -1).numpy()
                    
                    # Transform PCA puis slicing
                    pca_trans = sst_pca_model.transform(v_X_sst_reshaped)[:, :sst_pca_dim]
                    
                    # On redonne la forme [Batch, Lags * pca_dim]
                    pca_trans = pca_trans.reshape(B, L * sst_pca_dim)
                    v_X_sst_tensor = torch.tensor(pca_trans, dtype=torch.float32).to(device, non_blocking=True)
                    pca_trans_slp = slp_pca_model.transform(v_X_slp.view(B * len(slp_lags_months), -1).numpy())[:, :1]
                    v_X_slp_tensor = torch.tensor(pca_trans_slp.reshape(B, len(slp_lags_months) * 1), dtype=torch.float32).to(device, non_blocking=True)
                    v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                else:
                    # Si mode raw, on aplatit juste tout pour la couche linéaire
                    v_X_sst_tensor = v_X_sst.view(B, -1).to(device, non_blocking=True)
                    v_X_slp_tensor = v_X_slp.view(B, -1).to(device, non_blocking=True)
                    v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
                

                v_slp_flat = v_y_target.view(v_y_target.size(0), -1).cpu().numpy()
                v_embed_np = slp_pca_model.transform(v_slp_flat)[:, :args.latent_dim]
                v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                
                v_pred = model(v_X_combined)
                vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                
                all_preds.append(vp)
                all_targets.append(v_target_embed)

        val_preds_tensor = torch.cat(all_preds, dim=0)
        val_targets_tensor = torch.cat(all_targets, dim=0)

        # Calcul Métriques
        epoch_mse = F.mse_loss(val_preds_tensor, val_targets_tensor).item()
        val_target_variance = torch.var(val_targets_tensor, unbiased=False).item()
        epoch_r2 = 1.0 - (epoch_mse / val_target_variance) if val_target_variance > 0 else 0.0

        p, t = val_preds_tensor, val_targets_tensor
        p_mean, t_mean = p.mean(dim=0), t.mean(dim=0)
        p_var, t_var = ((p - p_mean)**2).mean(dim=0), ((t - t_mean)**2).mean(dim=0)
        cov = ((p - p_mean)*(t - t_mean)).mean(dim=0)
        epoch_corr = (cov / torch.sqrt(p_var * t_var + 1e-8)).mean().item()


        if epoch_r2 > best_r2_score: best_r2_score = epoch_r2
        if epoch_corr > best_trial_corr: best_trial_corr = epoch_corr
        if epoch_mse < best_trial_mse: best_trial_mse = epoch_mse
        # Choix de la métrique d'optimisation
        current_metric = epoch_r2 if args.optimize_metric == 'r2' else epoch_corr
        history.append((epoch, epoch_r2, epoch_corr))
        

        if current_metric > best_target_metric:
            best_target_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 0:
             trial.set_user_attr("val_target_variance", val_target_variance)

        trial.report(current_metric, epoch)
        if trial.should_prune():
            trial.set_user_attr("best_trial_mse", best_trial_mse)
            trial.set_user_attr("best_r2_score", best_r2_score)
            trial.set_user_attr("best_trial_corr", best_trial_corr)
            trial.set_user_attr("r2_corr_history", history)
            raise optuna.exceptions.TrialPruned()
            
        if patience_counter >= args.patience:
            break
    
    trial.set_user_attr("best_trial_mse", best_trial_mse)
    trial.set_user_attr("best_r2_score", best_r2_score)
    trial.set_user_attr("best_trial_corr", best_trial_corr)
    trial.set_user_attr("r2_corr_history", history)


    # test avec le meilleur modèle sauvegardé
    model.load_state_dict(best_model_state)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for v_X_sst, v_X_slp, v_y_target, _, _, _ in testloader:
            # Formattage Entrée SST
            B, L, H, W = v_X_sst.shape
            
            if input_format == 'pca':
                # On aplatit spatialement en fusionnant Batch et Lags : [B*L, 85*360]
                v_X_sst_reshaped = v_X_sst.view(B * L, -1).numpy()
                
                # Transform PCA puis slicing
                pca_trans = sst_pca_model.transform(v_X_sst_reshaped)[:, :sst_pca_dim]
                
                # On redonne la forme [Batch, Lags * pca_dim]
                pca_trans = pca_trans.reshape(B, L * sst_pca_dim)
                v_X_sst_tensor = torch.tensor(pca_trans, dtype=torch.float32).to(device, non_blocking=True)
                pca_trans_slp = slp_pca_model.transform(v_X_slp.view(B * len(slp_lags_months), -1).numpy())[:, :1]
                v_X_slp_tensor = torch.tensor(pca_trans_slp.reshape(B, len(slp_lags_months) * 1), dtype=torch.float32).to(device, non_blocking=True)
                v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
            else:
                # Si mode raw, on aplatit juste tout pour la couche linéaire
                v_X_sst_tensor = v_X_sst.view(B, -1).to(device, non_blocking=True)
                v_X_slp_tensor = v_X_slp.view(B, -1).to(device, non_blocking=True)
                v_X_combined = torch.cat((v_X_sst_tensor, v_X_slp_tensor), dim=1)
            
            v_slp_flat = v_y_target.view(v_y_target.size(0), -1).cpu().numpy()
            v_embed_np = slp_pca_model.transform(v_slp_flat)[:, :args.latent_dim]
            v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
        
            v_pred = model(v_X_combined)
            vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
            
            all_preds.append(vp)
            all_targets.append(v_target_embed)

    test_preds_tensor = torch.cat(all_preds, dim=0)
    test_targets_tensor = torch.cat(all_targets, dim=0)

    # Calcul Métriques
    test_mse = F.mse_loss(test_preds_tensor, test_targets_tensor).item()
    test_target_variance = torch.var(test_targets_tensor, unbiased=False).item()
    epoch_r2 = 1.0 - (test_mse / test_target_variance) if test_target_variance > 0 else 0.0

    p, t = test_preds_tensor, test_targets_tensor
    p_mean, t_mean = p.mean(dim=0), t.mean(dim=0)
    p_var, t_var = ((p - p_mean)**2).mean(dim=0), ((t - t_mean)**2).mean(dim=0)
    cov = ((p - p_mean)*(t - t_mean)).mean(dim=0)
    epoch_corr = (cov / torch.sqrt(p_var * t_var + 1e-8)).mean().item()

    trial.set_user_attr("test_target_variance", test_target_variance) 
    trial.set_user_attr("best_test_mse", test_mse)
    trial.set_user_attr("best_test_r2", epoch_r2)
    trial.set_user_attr("best_test_corr", epoch_corr)
    print(f"Trial terminé, temps écoulé: {time.time() - start_time:.2f} secondes")
    return best_target_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_startup_trials_tpe', type=int, default=10)
    parser.add_argument('--n_startup_trials_pruner', type=int, default=10)
    parser.add_argument('--n_warmup_steps', type=int, default=3)
    parser.add_argument('--interval_steps', type=int, default=1)

    parser.add_argument('--optimize_metric', type=str, choices=['r2', 'correlation'], default='correlation', help="Métrique à maximiser")
    parser.add_argument('--embed_path_slp', type=str, required=True)
    parser.add_argument('--embed_path_sst', type=str, required=True)
    parser.add_argument('--machine', type=str, default='jean-zay-work')
    parser.add_argument('--nb_members_val', type=int, default=1, help="Nombre de membres réservés à la validation")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_dim', type=int, default=1, help="pour ces métriques on ne prédit que la pc1")
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--include_lag1', action='store_true')
    parser.add_argument('--nb_intra_evals', type=int, default=15)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1','correlation','quantile'], default=None)
    parser.add_argument('--input_format', type=str, choices=['raw', 'pca'], default=None)
    parser.add_argument('--sst_lags_months', type=int, nargs='+', default=None, help='Lags SST à utiliser (optionnel)')
    parser.add_argument('--slp_lags_months', type=int, nargs='+', default=None, help='Lags SLP à utiliser (optionnel)')
    args = parser.parse_args()

    # Routage dynamique
    if args.machine == 'hacienda': base_home = "/home/moysan/stage_isir_jz/linear_models/optuna/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']: base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/optuna/"
    elif args.machine == 'mac_local': base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/linear_models/optuna/"

    dynamic_slp_std = 596.0 
    match = re.search(r'slp_std([0-9.]+)', args.embed_path_slp)
    if match: dynamic_slp_std = float(match.group(1))

    dynamic_sst_std = 0.707
    match = re.search(r'sst_std([0-9.]+)', args.embed_path_sst)
    if match: dynamic_sst_std = float(match.group(1))

    print(f"Dynamic SLP std: {dynamic_slp_std}, Dynamic SST std: {dynamic_sst_std}")    

    # Chargement Global de l'Embedder Target (SLP)
    slp_pca_model = joblib.load(args.embed_path_slp) 

    # Chargement du modèle PCA SST si nécessaire
    sst_pca_model = None
    if args.embed_path_sst:
        sst_pca_model = joblib.load(args.embed_path_sst)

    # Splits fixes
    rng = random.Random(args.seed)
    members_shuffled = ALL_MEMBERS.copy()
    rng.shuffle(members_shuffled)
    train_members = members_shuffled[:-2*args.nb_members_val]
    val_members = members_shuffled[-args.nb_members_val:]
    test_members = members_shuffled[-2*args.nb_members_val:-args.nb_members_val] 

    base_name = f"Optuna_LinReg_{args.optimize_metric}_months{''.join(map(str, args.winter_months))}_ep{args.nb_epochs}_ie{args.nb_intra_evals}_seed{args.seed}_pat{args.patience}_val{args.nb_members_val}_roll{args.roll_sst}_lag1{args.include_lag1}"
    short = {'bs': 'bs', 'lr': 'lr','loss_type': 'loss', 'input_format': 'input','sst_lags_months': 'lags'}
        # 3. Extraction et formatage compact (gère les listes, les notations scientifiques des floats et les str/int/bool)
    fixed = [f"{short[k]}{''.join(map(str, v)) if isinstance(v, list) else (f'{v:.1e}' if isinstance(v, float) and v < 1e-3 else str(v))}" for k, v in sorted(vars(args).items()) if k in short and v is not None]

    # 4. Assemblage final avec la configuration de l'échantillonneur Optuna
    dynamic_name = f"{base_name}_FIXED_{'_'.join(fixed)}" if fixed else f"{base_name}_full_search"
    dynamic_name += f"_optuna_s{args.n_startup_trials_tpe}_p{args.n_startup_trials_pruner}_{args.n_warmup_steps}_i{args.interval_steps}"
    study_name = dynamic_name
    output_dir = os.path.join(base_home, study_name)
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, "optuna.db")
    storage_name = f"sqlite:///{db_path}"

    # Définition de l'échantillonneur (TPE)
    sampler = TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials_tpe
    )
    
    # Définition de l'élagueur (Pruner) pour couper les mauvais essais
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials_pruner,
        n_warmup_steps=args.n_warmup_steps,
        interval_steps=args.interval_steps
    )

    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True, 
        sampler=sampler,
        pruner=pruner # NOUVEAU
    )
    
    print(f"Début de l'optimisation Optuna ({args.n_trials} essais)...")
    study.optimize(objective, n_trials=args.n_trials) 
    
    print("\nOptimisation Terminée !")
    print(f"Meilleur essai : {study.best_trial.value:.4f}")
    print("Meilleurs Hyperparamètres :")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
        
    study.trials_dataframe().to_csv(os.path.join(output_dir, "optuna_results.csv"), index=False)