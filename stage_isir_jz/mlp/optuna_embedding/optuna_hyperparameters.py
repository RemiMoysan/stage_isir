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
import warnings

warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from shared_tools.datasets import Dataset_mensuel
from shared_tools.models import compute_loss, get_median_prediction
from shared_tools.optuna_loop_helpers import encode_to_latent_gpu, compute_targeted_embedding_metrics

# ============================================================
# 1. ARCHITECTURES AVANCÉES SUR EMBEDDINGS
# ============================================================
class ResBlock1D(nn.Module):
    """Bloc résiduel 1D Pre-activation (He et al. 2016)."""
    def __init__(self, dim, dr, act_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            act_fn(),
            nn.Linear(dim, dim),
            nn.Dropout(dr),
            nn.LayerNorm(dim),
            act_fn(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class AdvancedEmbeddingPredictor(nn.Module):
    def __init__(self, sst_pca_dim, slp_pca_dim, n_sst_lags, n_slp_lags, out_dim=128, 
                 arch_type='mlp', hidden_dim=64, depth=2, dr=0.1, act='relu', 
                 use_time_attention=False, num_heads=4, use_global_skip=True):
        super().__init__()
        self.use_time_attention = use_time_attention
        self.use_global_skip = use_global_skip
        self.n_sst_lags = n_sst_lags
        self.n_slp_lags = n_slp_lags
        
        act_fn = nn.ReLU if act == 'relu' else nn.GELU
        in_features_flat = (n_sst_lags * sst_pca_dim) + (n_slp_lags * slp_pca_dim)
        
        # --- 1. GLOBAL LINEAR SKIP CONNECTION ---
        if self.use_global_skip:
            self.global_linear = nn.Linear(in_features_flat, out_dim)
        
        # --- 2. BLOC TEMPOREL (ATTENTION) ---
        if use_time_attention:
            self.sst_proj = nn.Linear(sst_pca_dim, hidden_dim)
            if n_slp_lags > 0:
                self.slp_proj = nn.Linear(slp_pca_dim, hidden_dim)
            
            self.time_embed = nn.Parameter(torch.zeros(1, n_sst_lags + n_slp_lags, hidden_dim))
            nn.init.trunc_normal_(self.time_embed, std=0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, 
                dropout=dr, activation=act, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            in_features_deep = hidden_dim * (n_sst_lags + n_slp_lags)
        else:
            in_features_deep = in_features_flat
            
        # --- 3. BLOC DE DÉCISION (MLP ou RESNET) ---
        self.input_layer = nn.Sequential(
            nn.Linear(in_features_deep, hidden_dim),
            act_fn()
        )
        
        blocks = []
        for _ in range(depth - 1):
            if arch_type == 'resnet':
                blocks.append(ResBlock1D(hidden_dim, dr, act_fn))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn(),
                    nn.Dropout(dr)
                ))
        self.hidden_layers = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_sst, x_slp=None):
        B = x_sst.size(0)
        
        x_sst_flat = x_sst.view(B, -1)
        if x_slp is not None and self.n_slp_lags > 0:
            x_slp_flat = x_slp.view(B, -1)
            x_flat = torch.cat([x_sst_flat, x_slp_flat], dim=1)
        else:
            x_flat = x_sst_flat
            
        # --- Voie 1 : Corrections Non-Linéaires ---
        if self.use_time_attention:
            tokens_sst = self.sst_proj(x_sst)
            if x_slp is not None and self.n_slp_lags > 0:
                tokens_slp = self.slp_proj(x_slp)
                tokens = torch.cat([tokens_sst, tokens_slp], dim=1)
            else:
                tokens = tokens_sst
            
            tokens = tokens + self.time_embed
            tokens = self.transformer(tokens)
            x_deep = tokens.view(B, -1) 
        else:
            x_deep = x_flat
                
        x_deep = self.input_layer(x_deep)
        x_deep = self.hidden_layers(x_deep)
        out_deep = self.output_layer(x_deep)
        
        # --- Fusion conditionnelle ---
        if self.use_global_skip:
            out_linear = self.global_linear(x_flat)
            return out_linear + out_deep
        else:
            return out_deep


# ============================================================
# 2. CONFIGURATION GLOBALE
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

def objective(trial):
    # ============================================================
    # 3. DÉFINITION DE L'ESPACE DE RECHERCHE
    # ============================================================
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

    sst_pca_dim = trial.suggest_int("sst_pca_dim", 1, 512)
    loss_type = args.loss_type if args.loss_type is not None else trial.suggest_categorical("loss_type", ["mse", "l1", "quantile", "correlation"]) 
    
    if args.bs is not None:
        bs = args.bs
    else:
        exp = trial.suggest_int("bs_exp", 5, 8)
        bs = int(2**exp)

    lr = args.lr if args.lr is not None else trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = args.weight_decay if args.weight_decay is not None else trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    noise_std = args.noise_std if args.noise_std is not None else trial.suggest_float("noise_std", 1e-4, 1e-1, log=True)
    grad_clip = args.gradient_clip if args.gradient_clip is not None else trial.suggest_float("grad_clip", 0.1, 1000.0, log=True)
    
    arch_type = trial.suggest_categorical("arch_type", ["mlp", "resnet"])
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64, 128])
    nn_depth = trial.suggest_int("nn_depth", 1, 4)
    dr = trial.suggest_float("dr", 0.0, 0.6)
    act = trial.suggest_categorical("act", ["relu", "gelu"])
    use_time_attention = trial.suggest_categorical("use_time_attention", [True, False])
    use_global_skip = trial.suggest_categorical("use_global_skip", [True, False])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8]) if use_time_attention else 4

    # ============================================================
    # 4. PRÉPARATION DES DONNÉES
    # ============================================================
    seed = args.fixed_seed if args.fixed_seed is not None else trial.suggest_int("seed", 0, 1000)
    rng = random.Random(seed)
    members_shuffled = ALL_MEMBERS.copy()
    rng.shuffle(members_shuffled)
    train_members = members_shuffled[:-2*args.nb_members_val]
    val_members = members_shuffled[-args.nb_members_val:]
    test_members = members_shuffled[-2*args.nb_members_val:-args.nb_members_val] 
    print(f"Validation members: {val_members}, Test members: {test_members}")

    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 2)) - 1)

    training_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=True, noise_std=noise_std)
    val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=False)
    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=False)

    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    out_features = args.latent_dim * len(args.quantiles) if loss_type == 'quantile' else args.latent_dim

    try:
        model = AdvancedEmbeddingPredictor(
            sst_pca_dim=sst_pca_dim, slp_pca_dim=1, n_sst_lags=len(sst_lags_months), n_slp_lags=len(slp_lags_months), 
            out_dim=out_features, arch_type=arch_type, hidden_dim=hidden_dim, depth=nn_depth, 
            dr=dr, act=act, use_time_attention=use_time_attention, num_heads=num_heads, use_global_skip=use_global_skip
        ).to(device)
    except RuntimeError as e:
        raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("num_params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "time_embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ], lr=lr)
    
    # ============================================================
    # 5. BOUCLE D'ENTRAÎNEMENT & ÉVALUATION
    # ============================================================
    best_target_metric = -float('inf')
    best_trial_R2, best_trial_L1, best_trial_corr = -float('inf'), -float('inf'), -float('inf')
    metrics_history = []
    patience_counter = 0
    best_model_state = None

    total_batches = len(trainloader)
    eval_steps_set = set(np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int)) | {0}
    eval_steps_epoch2_set = set(np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int)) | {0}

    try:
        for epoch in range(args.nb_epochs):
            model.train()
            for batch_idx, (X_sst, X_slp, y_target, _, _, _) in enumerate(trainloader):
                optimizer.zero_grad()
                X_sst = X_sst.to(device, non_blocking=True)
                X_slp = X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                B, L, H, W = X_sst.shape
                
                # Encodage PCA en gardant la dimension Lag [B, L, D]
                X_sst_2d = X_sst.view(B * L, H, W)
                sst_embed = encode_to_latent_gpu(X_sst_2d, 'pca', sst_pca_dim, sst_pca_components_gpu[:sst_pca_dim], sst_pca_mean_gpu, wgts_sst_gpu, None)
                X_sst_tensor = sst_embed.view(B, L, sst_pca_dim)
                
                if len(slp_lags_months) > 0:
                    X_slp_2d = X_slp.view(B * len(slp_lags_months), 53, 113)
                    slp_embed = encode_to_latent_gpu(X_slp_2d, 'pca', 1, pca_components_gpu[:1], pca_mean_gpu, wgts_gpu, None)
                    X_slp_tensor = slp_embed.view(B, len(slp_lags_months), 1)
                else:
                    X_slp_tensor = None

                target_embed = encode_to_latent_gpu(y_target.to(device, non_blocking=True), 'pca', args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, None)
                    
                pred = model(X_sst_tensor, X_slp_tensor)
                loss = compute_loss(pred, target_embed, loss_type=loss_type, quantiles=args.quantiles, reduction='mean')
                
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
                            v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None

                            v_X_sst_2d = v_X_sst.view(B * L, H, W)
                            v_sst_embed = encode_to_latent_gpu(v_X_sst_2d, 'pca', sst_pca_dim, sst_pca_components_gpu[:sst_pca_dim], sst_pca_mean_gpu, wgts_sst_gpu, None)
                            v_X_sst_tensor = v_sst_embed.view(B, L, sst_pca_dim)
                            
                            if len(slp_lags_months) > 0:
                                v_X_slp_2d = v_X_slp.view(B * len(slp_lags_months), 53, 113)
                                v_slp_embed = encode_to_latent_gpu(v_X_slp_2d, 'pca', 1, pca_components_gpu[:1], pca_mean_gpu, wgts_gpu, None)
                                v_X_slp_tensor = v_slp_embed.view(B, len(slp_lags_months), 1)
                            else:
                                v_X_slp_tensor = None
                            
                            v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), 'pca', args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, None)
                            v_pred = model(v_X_sst_tensor, v_X_slp_tensor)
                            vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                            
                            all_preds_latent.append(vp)
                            all_targets_latent.append(v_target_embed)

                    val_preds_latent = torch.cat(all_preds_latent, dim=0)
                    val_targets_latent = torch.cat(all_targets_latent, dim=0)

                    i_r2, i_l1, i_corr = compute_targeted_embedding_metrics(val_preds_latent, val_targets_latent)
                    metrics_dict = {'R2': i_r2, 'L1': i_l1, 'correlation': i_corr}
                    current_metric = metrics_dict[args.optimize_metric]
                    
                    best_trial_R2, best_trial_L1, best_trial_corr = max(best_trial_R2, i_r2), max(best_trial_L1, i_l1), max(best_trial_corr, i_corr)
                    metrics_history.append((epoch + batch_idx / total_batches, i_r2, i_l1, i_corr))

                    if current_metric > best_target_metric:
                        best_target_metric = current_metric
                        best_model_state = copy.deepcopy(model.state_dict())

                    model.train() 

            # --- END OF EPOCH EVALUATION ---
            model.eval()
            all_preds_latent, all_targets_latent = [], []
            with torch.no_grad():
                for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader:
                    B, L, H, W = v_X_sst.shape
                    v_X_sst = v_X_sst.to(device, non_blocking=True)
                    v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None

                    v_X_sst_2d = v_X_sst.view(B * L, H, W)
                    v_sst_embed = encode_to_latent_gpu(v_X_sst_2d, 'pca', sst_pca_dim, sst_pca_components_gpu[:sst_pca_dim], sst_pca_mean_gpu, wgts_sst_gpu, None)
                    v_X_sst_tensor = v_sst_embed.view(B, L, sst_pca_dim)
                    
                    if len(slp_lags_months) > 0:
                        v_X_slp_2d = v_X_slp.view(B * len(slp_lags_months), 53, 113)
                        v_slp_embed = encode_to_latent_gpu(v_X_slp_2d, 'pca', 1, pca_components_gpu[:1], pca_mean_gpu, wgts_gpu, None)
                        v_X_slp_tensor = v_slp_embed.view(B, len(slp_lags_months), 1)
                    else:
                        v_X_slp_tensor = None
                    
                    v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), 'pca', args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, None)
                    v_pred = model(v_X_sst_tensor, v_X_slp_tensor)
                    vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                    
                    all_preds_latent.append(vp)
                    all_targets_latent.append(v_target_embed)

            val_preds_latent = torch.cat(all_preds_latent, dim=0)
            val_targets_latent = torch.cat(all_targets_latent, dim=0)

            e_r2, e_l1, e_corr = compute_targeted_embedding_metrics(val_preds_latent, val_targets_latent)
            metrics_dict = {'R2': e_r2, 'L1': e_l1, 'correlation': e_corr}
            current_metric = metrics_dict[args.optimize_metric]

            metrics_history.append((epoch+1, e_r2, e_l1, e_corr))
            best_trial_R2, best_trial_L1, best_trial_corr = max(best_trial_R2, e_r2), max(best_trial_L1, e_l1), max(best_trial_corr, e_corr)

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
                
    except torch.OutOfMemoryError:
        print(f"⚠️ CUDA Out Of Memory détecté. Paramètres trop lourds. Trial élagué.")
        del model, optimizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            del model, optimizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        else:
            raise e
            
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
            B, L, H, W = v_X_sst.shape
            v_X_sst = v_X_sst.to(device, non_blocking=True)
            v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None

            v_X_sst_2d = v_X_sst.view(B * L, H, W)
            v_sst_embed = encode_to_latent_gpu(v_X_sst_2d, 'pca', sst_pca_dim, sst_pca_components_gpu[:sst_pca_dim], sst_pca_mean_gpu, wgts_sst_gpu, None)
            v_X_sst_tensor = v_sst_embed.view(B, L, sst_pca_dim)
            
            if len(slp_lags_months) > 0:
                v_X_slp_2d = v_X_slp.view(B * len(slp_lags_months), 53, 113)
                v_slp_embed = encode_to_latent_gpu(v_X_slp_2d, 'pca', 1, pca_components_gpu[:1], pca_mean_gpu, wgts_gpu, None)
                v_X_slp_tensor = v_slp_embed.view(B, len(slp_lags_months), 1)
            else:
                v_X_slp_tensor = None
            
            v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), 'pca', args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, None)
            v_pred = model(v_X_sst_tensor, v_X_slp_tensor)
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
    
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--include_lag1', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--sequential_lags', action='store_true')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--nb_intra_evals', type=int, default=5)
    
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1','correlation','quantile'], default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--noise_std', type=float, default=None)
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=None)
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=None)
    
    args = parser.parse_args()

    if args.machine == 'hacienda': base_home = "/home/moysan/stage_isir_jz/embedding_2_embedding/optuna_embedding/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']: base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/embedding_2_embedding/optuna_embedding/"
    elif args.machine == 'mac_local': base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/embedding_2_embedding/optuna_embedding/"

    dynamic_slp_std = 596.0 
    match = re.search(r'slp_std([0-9.]+)', args.embed_path)
    if match: dynamic_slp_std = float(match.group(1))

    dynamic_sst_std = 0.707
    if args.embed_path_sst:
        match = re.search(r'sst_std([0-9.]+)', args.embed_path_sst)
        if match: dynamic_sst_std = float(match.group(1))

    print(f"Dynamic SLP std: {dynamic_slp_std}, Dynamic SST std: {dynamic_sst_std}")    

    slp_pca_model = joblib.load(args.embed_path) 
    pca_mean_gpu = torch.tensor(slp_pca_model.mean_, dtype=torch.float32, device=device)
    pca_components_gpu = torch.tensor(slp_pca_model.components_[:args.latent_dim], dtype=torch.float32, device=device)

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
            pass

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
                wgts_flat_sst = np.broadcast_to(np.sqrt(coslat_sst).reshape(h_sst, 1), (h_sst, w_sst)).flatten()
                wgts_sst_gpu = torch.tensor(wgts_flat_sst, dtype=torch.float32, device=device)
        except Exception as e:
            pass

    rng = random.Random(args.fixed_seed)
    members_shuffled = ALL_MEMBERS.copy()
    rng.shuffle(members_shuffled)
    train_members = members_shuffled[:-2*args.nb_members_val]
    val_members = members_shuffled[-args.nb_members_val:]
    test_members = members_shuffled[-2*args.nb_members_val:-args.nb_members_val] 

    base_name = f"Optuna_MLP_{args.optimize_metric}_m{''.join(map(str, args.winter_months))}_ep{args.nb_epochs}ie{args.nb_intra_evals}pat{args.patience}val{args.nb_members_val}lag1{args.include_lag1}seq{args.sequential_lags}roll{args.roll_sst}latw{args.lat_weight}"
    short = {
        'bs': 'bs', 'lr': 'lr','loss_type': 'loss', 'sst_lags_months': 'lags',
        'weight_decay': 'wd', 'noise_std': 'ns', 'gradient_clip': 'gc', 'fixed_seed': 'seed'
    }
    
    fixed = []
    for k, v in sorted(vars(args).items()):
        if k in short and v is not None:
            val_str = ''.join(map(str, v)) if isinstance(v, list) else (f"{v:.1e}" if isinstance(v, float) and v < 1e-3 else str(v))
            fixed.append(f"{short[k]}{val_str}")

    dynamic_name = f"{base_name}_FIXED_{'_'.join(fixed)}" if fixed else f"{base_name}_full_search"
    dynamic_name += f"_optuna_s{args.n_startup_trials_tpe}p{args.n_startup_trials_pruner}_{args.n_warmup_steps}i{args.interval_steps}"
    
    if len(dynamic_name) > 230:
        short_hash = hashlib.md5(dynamic_name.encode()).hexdigest()[:6]
        dynamic_name = dynamic_name[:220] + "_h" + short_hash

    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, "optuna.db")
    storage_name = f"sqlite:///{db_path}"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials_pruner, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps)
    sampler = TPESampler(n_startup_trials=args.n_startup_trials_tpe, multivariate=True, group=True, seed=42)

    study = optuna.create_study(
        study_name=dynamic_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    print(f"Début de l'optimisation Advanced EMBEDDING ({args.n_trials} essais)...")
    study.optimize(objective, n_trials=args.n_trials) 
    
    print("\n=== Bilan HPO Advanced EMBEDDING ===")
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