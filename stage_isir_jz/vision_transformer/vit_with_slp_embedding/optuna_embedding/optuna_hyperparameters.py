import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import random
import joblib
import xarray as xr
import re
import hashlib
from datetime import datetime

import torch
import optuna
from optuna.trial import TrialState
import torch.nn.functional as F
import copy

project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

grand_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if grand_parent_dir not in sys.path:
    sys.path.append(grand_parent_dir)

from shared_tools.datasets import Dataset_mensuel
from tools.models import ViT_Latent_SLP_Multimodal_tunable 
from shared_tools.models import ConvVAE, compute_loss, get_median_prediction
from shared_tools.optuna_loop_helpers import encode_to_latent_gpu, compute_targeted_embedding_metrics

# Assure-toi que ce chemin d'import pointe bien vers ton fichier contenant le modèle ViT


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

def objective(trial):
    # ============================================================
    # 1. DÉFINITION DES HYPERPARAMÈTRES À TUNER (Spécifique ViT)
    # ============================================================
    nb_members_val = args.nb_members_val if args.nb_members_val is not None else trial.suggest_int("nb_members_val", 1, 20)
    seed = args.fixed_seed if args.fixed_seed is not None else trial.suggest_int("seed", 0, 1000)
    rng = random.Random(seed)
    members_shuffled = ALL_MEMBERS.copy()
    rng.shuffle(members_shuffled)
    train_members = members_shuffled[:-2*nb_members_val]
    val_members = members_shuffled[-nb_members_val:]
    test_members = members_shuffled[-2*nb_members_val:-nb_members_val] 
    print(f"Test members: {test_members}, Validation members: {val_members}")
    # --- ARCHITECTURE ---
    if args.bs is not None:
        bs = args.bs
    else:
        exp = trial.suggest_int("bs_exp", 5, 7)
        bs = int(2**exp)
    lr = args.lr if args.lr is not None else trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    dr = args.dr if args.dr is not None else trial.suggest_float("dr", 0.0, 0.5)

    # --- Paramètres de l'Architecture ViT ---
    depth = args.depth if args.depth is not None else trial.suggest_int("depth", 2, 6)
    num_heads = args.num_heads if args.num_heads is not None else trial.suggest_categorical("num_heads", [2, 4, 8])
    head_dim = args.head_dim if args.head_dim is not None else trial.suggest_categorical("head_dim", [16, 32, 64])
    
    # Dimension d'embedding calculée dynamiquement pour garantir la divisibilité
    embed_dim = num_heads * head_dim
    trial.set_user_attr("embed_dim", embed_dim) 
    
    mlp_ratio = args.mlp_ratio if args.mlp_ratio is not None else trial.suggest_float("mlp_ratio", 2.0, 6.0, step=1.0)
    transformer_act = args.transformer_act if args.transformer_act is not None else trial.suggest_categorical("transformer_act", ["gelu", "relu"])
    norm_first = args.norm_first if args.norm_first is not None else trial.suggest_categorical("norm_first", [True, False])

    use_lags_attention = args.use_lags_attention if args.use_lags_attention is not None else trial.suggest_categorical("use_lags_attention", [True, False])
    pool_strategy = args.pool_strategy if args.pool_strategy is not None else trial.suggest_categorical("pool_strategy", ['cls', 'gap'])
    head_act = args.head_act if args.head_act is not None else trial.suggest_categorical("head_act", ['tanh', 'relu'])

    # Optuna choisit un ratio de compression (1.0 = pas de compression, 0.5 = divisé par 2, etc.)
    bottleneck_ratio = args.bottleneck_ratio if args.bottleneck_ratio is not None else trial.suggest_categorical("bottleneck_ratio", [0.25, 0.5, 1.0])
    # On calcule la dimension cachée de la tête dynamiquement
    head_hidden_dim = int(embed_dim * bottleneck_ratio)
    # Sécurité au cas où la compression donne une dimension trop petite par rapport au latent final
    head_hidden_dim = max(head_hidden_dim, args.latent_dim)

    loss_type = args.loss_type if args.loss_type is not None else trial.suggest_categorical("loss_type", ['mse', 'l1','correlation','quantile'])
    
    sst_patch_y = args.sst_patch_y if args.sst_patch_y is not None else trial.suggest_categorical("sst_patch_y", [5, 17])
    sst_patch_x = args.sst_patch_x if args.sst_patch_x is not None else trial.suggest_categorical("sst_patch_x", [10, 15, 20,30,40]) # j'ai enlevé 4 et 6 et mis 30 et 40 car ça avait l'air de mieux marcher

    weight_decay = args.weight_decay if args.weight_decay is not None else trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    noise_std = args.noise_std if args.noise_std is not None else trial.suggest_float("noise_std", 1e-4, 1e-1, log=True)
    grad_clip = args.gradient_clip if args.gradient_clip is not None else trial.suggest_float("grad_clip", 0.1, 1000.0, log=True)

    # --- LAGS ---
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

    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trial.set_user_attr("sst_lags_final", sst_lags_months)
    trial.set_user_attr("slp_lags_final", slp_lags_months)

    # ============================================================
    # 2. PRÉPARATION DES DONNÉES
    # ============================================================
    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 2)) - 1)

    train_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=True, noise_std=noise_std)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=False)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)

    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std, augment=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # 3. INITIALISATION DU MODÈLE ViT
    # ============================================================
    out_feature = len(quantiles) * args.latent_dim if loss_type == 'quantile' else args.latent_dim

    model = ViT_Latent_SLP_Multimodal_tunable(
        sst_size=(85, 360), patch_size_sst=(sst_patch_y, sst_patch_x), in_chans_sst=len(sst_lags_months), in_chans_slp=len(slp_lags_months),
        slp_size=(53, 113), patch_size_slp=(5, 5),
        nb_out=out_feature, 
        embed_dim=embed_dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_ratio=mlp_ratio,              
        transformer_act=transformer_act,     
        dr=dr, 
        use_lags_attention=use_lags_attention,
        pool_strategy=pool_strategy,        
        head_hidden_dim=head_hidden_dim,       
        head_act=head_act,
        norm_first=norm_first
    ).to(device)

    # Vérification forward pass (Dry Run)
    try:
        with torch.no_grad():
            dummy_sst = torch.zeros(1, len(sst_lags_months), 85, 360).to(device) if len(sst_lags_months) > 0 else None
            dummy_slp = torch.zeros(1, len(slp_lags_months), 53, 113).to(device) if len(slp_lags_months) > 0 else None
            _ = model(dummy_sst, dummy_slp)
    except RuntimeError as e:
        print(f"⚠️ Architecture physiquement impossible ou erreur de patch size. Trial élagué. Erreur: {e}")
        raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("num_params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
            
        # On exclut : LayerNorm/BatchNorm (1D), Biais (1D), et les tables de position/tokens
        if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name or "time_embed" in name or "cls_token" in name:
            no_decay.append(param)
        else:
            decay.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ], lr=lr)

    # ============================================================
    # 4. BOUCLE D'ENTRAÎNEMENT ET D'ÉVALUATION UNIFIÉE
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
            X_slp = X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
            
            # Cible unifiée GPU via Helper
            target_embed = encode_to_latent_gpu(y_target.to(device, non_blocking=True), args.embed_method, args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
            
            pred = model(X_sst, X_slp)
            loss = compute_loss(pred, target_embed, loss_type=loss_type, quantiles=quantiles, reduction='mean')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            # --- INTRA-EPOCH EVALUATION ---
            if (epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set):
                model.eval()
                all_preds_latent, all_targets_latent = [], []   
                with torch.no_grad():
                    for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                        v_X_sst = v_X_sst.to(device, non_blocking=True)
                        v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                        
                        v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), args.embed_method, args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
                        v_pred = model(v_X_sst, v_X_slp)
                        
                        vp = get_median_prediction(v_pred, loss_type, quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                        
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

                model.train() 

        # --- END OF EPOCH EVALUATION ---
        model.eval()
        all_preds_latent, all_targets_latent = [], []
        
        with torch.no_grad():
            for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader:
                v_X_sst = v_X_sst.to(device, non_blocking=True)
                v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                
                v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), args.embed_method, args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
                v_pred = model(v_X_sst, v_X_slp)
                vp = get_median_prediction(v_pred, loss_type, quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                
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
            del model, optimizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
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
            v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
            
            v_target_embed = encode_to_latent_gpu(v_y_target.to(device, non_blocking=True), args.embed_method, args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
            v_pred = model(v_X_sst, v_X_slp)
            vp = get_median_prediction(v_pred, loss_type, quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
            
            all_preds_latent.append(vp)
            all_targets_latent.append(v_target_embed)

    test_preds_latent = torch.cat(all_preds_latent, dim=0)
    test_targets_latent = torch.cat(all_targets_latent, dim=0)

    t_r2, t_l1, t_corr = compute_targeted_embedding_metrics(test_preds_latent, test_targets_latent)

    trial.set_user_attr("best_test_R2", t_r2)
    trial.set_user_attr("best_test_L1", t_l1)
    trial.set_user_attr("best_test_corr", t_corr)

    del model, optimizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"Trial terminé en {time.time() - start_time:.2f} s | Target Metric: {best_target_metric:.4f}")
    return best_target_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize_metric', type=str, choices=['R2', 'L1', 'correlation'], default='correlation', help="Métrique d'embedding à maximiser")
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--nb_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_intra_evals', type=int, default=15)
    parser.add_argument('--n_startup_trials_tpe', type=int, default=10)
    parser.add_argument('--n_startup_trials_pruner', type=int, default=10)
    parser.add_argument('--n_warmup_steps', type=int, default=3)
    parser.add_argument('--interval_steps', type=int, default=1)
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca')
    parser.add_argument('--embed_path', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--machine', type=str, default='jean-zay-work')
    
    # Booléens via parser Helper unifié
    def stR2bool(v):
        if v is None or v.lower() == 'none': return None
        return v.lower() in ("yes", "true", "t", "1")

    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--include_lag1', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--sequential_lags', action='store_true')
    parser.add_argument('--norm_first', type=stR2bool, default=None)
    parser.add_argument('--use_lags_attention', type=stR2bool, default=None)

    # HYPERPARAMÈTRES DYNAMIQUES ViT
    parser.add_argument('--fixed_seed', type=int, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dr', type=float, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--head_dim', type=int, default=None)
    parser.add_argument('--mlp_ratio', type=float, default=None)
    parser.add_argument('--bottleneck_ratio', type=float, default=None)
    parser.add_argument('--transformer_act', type=str, choices=['gelu', 'relu'], default=None)
    parser.add_argument('--pool_strategy', type=str, choices=['cls', 'gap'], default=None)
    parser.add_argument('--head_act', type=str, choices=['tanh', 'relu'], default=None)
    parser.add_argument('--sst_patch_y', type=int, default=None)
    parser.add_argument('--sst_patch_x', type=int, default=None)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1','correlation','quantile'], default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--noise_std', type=float, default=None)
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=None)
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=None)

    args = parser.parse_args()

    # --- SETUP CHEMINS & STD ---
    if args.machine == 'hacienda': base_home = "/home/moysan/stage_isir_jz/vision_transformer/vit_with_slp_embedding/optuna_embedding/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']: base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/optuna_embedding/"
    elif args.machine == 'mac_local': base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/vision_transformer/vit_with_slp_embedding/optuna_embedding/"

    dynamic_slp_std = 466.93 # pour le vae ce n'est pas dans le nom du fichier donc je force ici pour retomber sur la bonne valeur
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))

    # Fallback générique pour SST STD. Effectivement, c'est bien la valeur par défaut pour la classe Dataset (mais c'est la seule utilisé ici)
    dynamic_sst_std = 0.707 

    # --- CHARGEMENT DU MODÈLE D'EMBEDDING TARGET (GPU) ---
    pca_model, vae_model = None, None
    pca_mean_gpu, pca_components_gpu = None, None
    latent_dim = args.latent_dim

    if args.embed_method == 'pca':
        pca_model = joblib.load(args.embed_path)
        pca_mean_gpu = torch.tensor(pca_model.mean_, dtype=torch.float32, device=device)
        pca_components_gpu = torch.tensor(pca_model.components_[:latent_dim], dtype=torch.float32, device=device)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

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
            
    # --- NOMMAGE SÉCURISÉ ---
    base_name = f"study_vit_{args.embed_method}{args.latent_dim}_{args.optimize_metric}_m{''.join(map(str, args.winter_months))}_ep{args.nb_epochs}ie{args.nb_intra_evals}pat{args.patience}val{args.nb_members_val}lag1{args.include_lag1}seq{args.sequential_lags}latw{args.lat_weight}"
    
    short = {
        'fixed_seed': 'seed', 'bs': 'bs', 'lr': 'lr', 'dr': 'dr', 'depth': 'dp', 'num_heads': 'nh', 'head_dim': 'hd', 
        'mlp_ratio': 'mlp', 'transformer_act': 'tact', 'norm_first': 'nf', 'use_lags_attention': 'latt', 
        'pool_strategy': 'pool', 'head_act': 'hact', 'use_bottleneck': 'bn', 'loss_type': 'loss', 
        'roll_sst': 'roll', 'sst_patch_y': 'py', 'sst_patch_x': 'px', 'weight_decay': 'wd', 
        'noise_std': 'ns', 'gradient_clip': 'gc', 'sst_lags_months': 'sstlags', 'slp_lags_months': 'slplags'
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
    dynamic_name += f"_optuna_s{args.n_startup_trials_tpe}p{args.n_startup_trials_pruner}_{args.n_warmup_steps}i{args.interval_steps}"

    if len(dynamic_name) > 230:
        short_hash = hashlib.md5(dynamic_name.encode()).hexdigest()[:6]
        dynamic_name = dynamic_name[:220] + "_h" + short_hash

    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "vit_optuna_embedding.db")
    csv_path = os.path.join(output_dir, "vit_optuna_embedding_results.csv")
    storage_name = f"sqlite:///{db_path}"
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials_pruner, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps)
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.n_startup_trials_tpe, multivariate=True, group=True, seed=42)

    study = optuna.create_study(
        study_name=dynamic_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    print(f"Début de l'optimisation EMBEDDING ViT (MAX {args.optimize_metric}) pour {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\n=== Bilan HPO EMBEDDING ViT ===")
    trial = study.best_trial
    print(f"  Meilleur {args.optimize_metric.upper()} (Validation) : {trial.value:.4f}")
    
    print("\n  --- Performances sur le Set de TEST (Caché) ---")
    print(f"  Global R2 Score               : {trial.user_attrs.get('best_test_R2'):.4f}")
    print(f"  Global L1 Skill Score         : {trial.user_attrs.get('best_test_L1'):.4f}")
    print(f"  Global Correlation            : {trial.user_attrs.get('best_test_corr'):.4f}")
    print("\n  --- Meilleurs Hyperparamètres ---")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés dans : {csv_path}")