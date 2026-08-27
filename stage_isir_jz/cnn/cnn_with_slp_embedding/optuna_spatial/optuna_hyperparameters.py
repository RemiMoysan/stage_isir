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
import glob
import xarray as xr
import torch
import torch.nn.functional as F
import optuna
import copy

project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

grand_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if grand_parent_dir not in sys.path:
    sys.path.append(grand_parent_dir)

from shared_tools.datasets import Dataset_mensuel
from tools_cnn.models import CNN_Latent_SLP_Multimodal1_tunable
from shared_tools.models import ConvVAE, compute_loss, get_median_prediction
from shared_tools.optuna_loop_helpers import encode_to_latent_gpu, decode_to_spatial_map_gpu, compute_targeted_spatial_metrics

# 2 usages : mode pca dossier avec deux pca typiquemet 128 pc, et une normalize l'autre non: au final la version non normalize est clairement meilleur. 
# mode vae avec un dossier avec des vae de dim différentes (latent16 par ex dans le nom) 

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def objective(trial):
    nb_members_val = args.nb_members_val if args.nb_members_val is not None else trial.suggest_int("nb_members_val", 1, 20)
    seed = args.fixed_seed if args.fixed_seed is not None else trial.suggest_int("seed", 0, 1000)
    rng = random.Random(seed)
    members_shuffled = ALL_MEMBERS.copy()
    rng.shuffle(members_shuffled)
    train_members = members_shuffled[:-2*nb_members_val]
    val_members = members_shuffled[-nb_members_val:]
    test_members = members_shuffled[-2*nb_members_val:-nb_members_val]

    # --- SÉLECTION DYNAMIQUE DE L'EMBEDDING ---
    if args.embed_dir is not None and os.path.isdir(args.embed_dir):
        if args.embed_method == 'vae':
            available_paths = glob.glob(os.path.join(args.embed_dir, "*.pth"))
            parsed_models = []
            
            # 1. On parse tous les fichiers du dossier
            for p in available_paths:
                fname = os.path.basename(p)
                match = re.search(r'beta([0-9.]+)latent([0-9]+)\.pth', fname)
                if match:
                    parsed_models.append((float(match.group(1)), int(match.group(2)), fname))
            
            if not parsed_models:
                raise ValueError("Aucun fichier VAE au format 'betaXlatentY.pth' trouvé.")
            
            # 2. Extraire TOUS les betas et TOUS les latents de manière globale et fixe
            all_betas = sorted(list(set(m[0] for m in parsed_models)))
            all_latents = sorted(list(set(m[1] for m in parsed_models)))
            
            # 3. Optuna suggère les index sur ces listes GLOBALES
            idx_beta = trial.suggest_int("idx_vae_beta", 0, len(all_betas) - 1)
            idx_lat = trial.suggest_int("idx_vae_latent", 0, len(all_latents) - 1)
            
            chosen_beta = all_betas[idx_beta]
            chosen_latent = all_latents[idx_lat]
            
            # 4. Vérifier si cette combinaison croisée (beta, latent) existe vraiment sur le disque
            matching_files = [m[2] for m in parsed_models if m[0] == chosen_beta and m[1] == chosen_latent]
            
            if not matching_files:
                # La combinaison n'a pas de fichier associé (trou dans la grille d'entraînement VAE)
                raise optuna.exceptions.TrialPruned()
            
            chosen_file = matching_files[0]
            embed_path = os.path.join(args.embed_dir, chosen_file)
            
            # On stocke les vraies valeurs pour que le Dashboard HTML soit lisible
            trial.set_user_attr("vae_beta_real", chosen_beta)
            trial.set_user_attr("vae_latent_real", chosen_latent)
            trial.set_user_attr("embed_file", chosen_file)
            
        elif args.embed_method == 'pca':
            available_paths = glob.glob(os.path.join(args.embed_dir, "*.joblib"))
            available_files = sorted([os.path.basename(p) for p in available_paths])
            chosen_file = trial.suggest_categorical("embed_file", available_files)
            embed_path = os.path.join(args.embed_dir, chosen_file)
            trial.set_user_attr("embed_file", chosen_file)
    else:
        embed_path = args.embed_path
        trial.set_user_attr("embed_file", os.path.basename(embed_path))

    pca_model, vae_model, max_latent_dim = None, None, 128
    pca_mean_gpu, pca_components_gpu = None, None

    if args.embed_method == 'pca':
        pca_model = joblib.load(embed_path)
        max_latent_dim = pca_model.n_components_
    elif args.embed_method == 'vae':
        match_dim = re.search(r'latent([0-9]+)', embed_path)
        max_latent_dim = int(match_dim.group(1))
        print(f"Détection de la dimension latente VAE : {max_latent_dim}")
        vae_model = ConvVAE(latent_dim=max_latent_dim).to(device)
        vae_model.load_state_dict(torch.load(embed_path, map_location=device))
        vae_model.eval()

    if args.embed_method == 'vae':
        current_latent_dim = max_latent_dim
    elif args.latent_dim is not None:
        current_latent_dim = args.latent_dim
    else:
        current_latent_dim = trial.suggest_int("latent_dim", 1, max_latent_dim)
        
    trial.set_user_attr("current_latent_dim", current_latent_dim)

    # Chargement PCA GPU avec Slicing natif
    if args.embed_method == 'pca':
        pca_mean_gpu = torch.tensor(pca_model.mean_, dtype=torch.float32, device=device)
        pca_components_gpu = torch.tensor(pca_model.components_[:current_latent_dim], dtype=torch.float32, device=device)

    # --- ARCHITECTURE ---
    if args.bs is not None:
        bs = args.bs
    else:
        exp = trial.suggest_int("bs_exp", 5, 7)
        bs = int(2**exp)

    lr = args.lr if args.lr is not None else trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    dr_conv = args.dr_conv if args.dr_conv is not None else trial.suggest_float("dr_conv", 0.0, 0.6)
    dr_fc = args.dr_fc if args.dr_fc is not None else trial.suggest_float("dr_fc", 0.0, 0.6)
    fc_dim = args.fc_dim if args.fc_dim is not None else trial.suggest_int("fc_dim", 8, 64)
    depth = args.depth if args.depth is not None else trial.suggest_int("depth", 2, 4)
    n_feat = args.n_feat if args.n_feat is not None else trial.suggest_int("n_feat", 4, 32)
    filter_mult = args.filter_mult if args.filter_mult is not None else trial.suggest_float("filter_mult", 1.0, 2.0, step=0.25)
    pool_type = args.pool_type if args.pool_type is not None else trial.suggest_categorical("pool_type", ['max', 'avg'])
    sst_pool_x = args.sst_pool_x if args.sst_pool_x is not None else trial.suggest_int("sst_pool_x", 2, 4)
    sst_pool_y = args.sst_pool_y if args.sst_pool_y is not None else trial.suggest_int("sst_pool_y", 2, 4)
    sst_kx = args.sst_kx if args.sst_kx is not None else trial.suggest_int("sst_kx", 3, 5)
    sst_ky = args.sst_ky if args.sst_ky is not None else trial.suggest_int("sst_ky", 3, 5)
    activation = args.activation if args.activation is not None else trial.suggest_categorical("activation", ['tanh', 'relu'])
    pool_strategy = args.pool_strategy if args.pool_strategy is not None else trial.suggest_categorical("pool_strategy", ['progressive', 'standard'])
    use_gap = args.use_gap if args.use_gap is not None else trial.suggest_categorical("use_gap", [True, False])
    early_fusion_sst = args.early_fusion_sst if args.early_fusion_sst is not None else trial.suggest_categorical("early_fusion_sst", [True, False])
    loss_type = args.loss_type if args.loss_type is not None else trial.suggest_categorical("loss_type", ['mse', 'l1','correlation','quantile'])
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
    # 2. PRÉPARATION DES DONNÉES ET POIDS SPATIAUX
    # ============================================================
    dynamic_slp_std = 466.93 # pour le vae ce n'est pas dans le nom du fichier donc je force ici pour retomber sur la bonne valeur
    match_std = re.search(r'slp_std([0-9.]+)', embed_path)
    if match_std: 
        dynamic_slp_std = float(match_std.group(1))
        print(f"slp std dynamique détecté dans le nom du fichier : {dynamic_slp_std}")
    print(f"std SLP dynamique utilisé : {dynamic_slp_std}")

    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 2)) - 1)

    train_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, augment=True, noise_std=noise_std)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, augment=False)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)

    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std, augment=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # 3. INITIALISATION DU MODÈLE ET OPTIMIZER (ADAMW)
    # ============================================================
    out_feature = len(quantiles) * current_latent_dim if loss_type == 'quantile' else current_latent_dim
    model = CNN_Latent_SLP_Multimodal1_tunable(
        dr_conv=dr_conv, dr_fc=dr_fc, fc_dim=fc_dim, nb_out=out_feature, in_chans_sst=len(sst_lags_months), in_chans_slp=len(slp_lags_months), 
        n_feat=n_feat, early_fusion_sst=early_fusion_sst, depth=depth, filter_mult=filter_mult,
        sst_kx=sst_kx, sst_ky=sst_ky, sst_pool_x=sst_pool_x, sst_pool_y=sst_pool_y,
        pool_type=pool_type, pool_strategy=pool_strategy, activation=activation, use_gap=use_gap
    ).to(device)

    try:
        with torch.no_grad():
            dummy_sst = torch.zeros(1, len(sst_lags_months), 85, 360).to(device) if len(sst_lags_months) > 0 else None
            dummy_slp = torch.zeros(1, len(slp_lags_months), 53, 113).to(device) if len(slp_lags_months) > 0 else None
            _ = model(dummy_sst, dummy_slp)
    except RuntimeError as e:
        if "Calculated output size" in str(e) or "Output size is too small" in str(e):
            print(f"⚠️ Architecture physiquement impossible (trop de pooling pour la depth). Trial élagué.")
            raise optuna.exceptions.TrialPruned()
        else:
            raise e # Si c'est une autre erreur (ex: Out Of Memory), on la laisse crasher

    trial.set_user_attr("num_params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ], lr=lr)

    # ============================================================
    # 4. BOUCLE D'ENTRAÎNEMENT ET TRACKING EMBEDDINGS
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
            
            # Encodage propre via Helper
            target_embed = encode_to_latent_gpu(y_target.to(device, non_blocking=True), args.embed_method, current_latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
            
            pred = model(X_sst, X_slp)
            loss = compute_loss(pred, target_embed, loss_type, quantiles=quantiles, reduction='mean')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            # --- INTRA-EPOCH EVALUATION ---
            if (epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set):
                model.eval()
                all_preds_latent, all_true_maps = [], []
                with torch.no_grad():
                    for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                        v_X_sst = v_X_sst.to(device, non_blocking=True)
                        v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                        
                        v_pred = model(v_X_sst, v_X_slp)
                        vp = get_median_prediction(v_pred, loss_type, quantiles, current_latent_dim) if loss_type == 'quantile' else v_pred
                        all_preds_latent.append(vp)
                        all_true_maps.append(v_y_target)

                val_preds_latent = torch.cat(all_preds_latent, dim=0)
                val_true_maps = torch.cat(all_true_maps, dim=0).to(device, non_blocking=True)

                val_pred_maps = decode_to_spatial_map_gpu(val_preds_latent, args.embed_method, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
                i_r2, i_l1, i_corr = compute_targeted_spatial_metrics(val_pred_maps, val_true_maps, wgts_gpu)
                
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

        # --- END OF EPOCH EVALUATION ---
        model.eval()
        all_preds_latent, all_true_maps = [], []
        with torch.no_grad():
            for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader:
                v_X_sst = v_X_sst.to(device, non_blocking=True)
                v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                
                v_pred = model(v_X_sst, v_X_slp)
                vp = get_median_prediction(v_pred, loss_type, quantiles, current_latent_dim) if loss_type == 'quantile' else v_pred
                all_preds_latent.append(vp)
                all_true_maps.append(v_y_target)

        val_preds_latent = torch.cat(all_preds_latent, dim=0)
        val_true_maps = torch.cat(all_true_maps, dim=0).to(device, non_blocking=True)

        val_pred_maps = decode_to_spatial_map_gpu(val_preds_latent, args.embed_method, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
        e_r2, e_l1, e_corr = compute_targeted_spatial_metrics(val_pred_maps, val_true_maps, wgts_gpu)

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
    all_preds_latent, all_true_maps = [], []
    with torch.no_grad():
        for v_X_sst, v_X_slp, v_y_target, _, _, _ in testloader:
            v_X_sst = v_X_sst.to(device, non_blocking=True)
            v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
            
            v_pred = model(v_X_sst, v_X_slp)
            vp = get_median_prediction(v_pred, loss_type, quantiles, current_latent_dim) if loss_type == 'quantile' else v_pred
            all_preds_latent.append(vp)
            all_true_maps.append(v_y_target)

    test_preds_latent = torch.cat(all_preds_latent, dim=0)
    test_true_maps = torch.cat(all_true_maps, dim=0).to(device, non_blocking=True)

    test_pred_maps = decode_to_spatial_map_gpu(test_preds_latent, args.embed_method, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
    t_r2, t_l1, t_corr = compute_targeted_spatial_metrics(test_pred_maps, test_true_maps, wgts_gpu)

    # Noms unifiés pour post-processing partagé !
    trial.set_user_attr("best_test_R2", t_r2)
    trial.set_user_attr("best_test_L1", t_l1)
    trial.set_user_attr("best_test_corr", t_corr)

    del model, optimizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"Trial terminé en {time.time() - start_time:.2f} s | Latent Dim: {current_latent_dim} | Target Metric: {best_target_metric:.4f}")
    return best_target_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize_metric', type=str, choices=['R2', 'L1', 'correlation'], default='R2', help="Métrique spatiale physique à maximiser par Optuna")
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--nb_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_intra_evals', type=int, default=5)
    parser.add_argument('--n_startup_trials_tpe', type=int, default=10)
    parser.add_argument('--n_startup_trials_pruner', type=int, default=10)
    parser.add_argument('--n_warmup_steps', type=int, default=3)
    parser.add_argument('--interval_steps', type=int, default=1)
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--embed_method', type=str, default='pca', choices=['pca', 'vae'])
    
    group_embed = parser.add_mutually_exclusive_group(required=True)
    group_embed.add_argument('--embed_path', type=str, default=None)
    group_embed.add_argument('--embed_dir', type=str, default=None)

    # Booléens
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--include_lag1', action='store_true')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--sequential_lags', action='store_true')

    # HYPERPARAMÈTRES DYNAMIQUES
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument('--fixed_seed', type=int, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dr_conv', type=float, default=None)
    parser.add_argument('--dr_fc', type=float, default=None)
    parser.add_argument('--fc_dim', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--n_feat', type=int, default=None)
    parser.add_argument('--filter_mult', type=float, default=None)
    parser.add_argument('--pool_type', type=str, choices=['max', 'avg'], default=None)
    parser.add_argument('--sst_pool_x', type=int, default=None)
    parser.add_argument('--sst_pool_y', type=int, default=None)
    parser.add_argument('--sst_kx', type=int, default=None)
    parser.add_argument('--sst_ky', type=int, default=None)
    parser.add_argument('--activation', type=str, choices=['tanh', 'relu'], default=None)
    parser.add_argument('--pool_strategy', type=str, choices=['progressive', 'standard'], default=None)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1','correlation','quantile'], default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--noise_std', type=float, default=None)
    parser.add_argument('--gradient_clip', type=float, default=None)
    
    def stR2bool(v):
        if v is None or v.lower() == 'none': return None
        return v.lower() in ("yes", "true", "t", "1")
    parser.add_argument('--use_gap', type=stR2bool, default=None)
    parser.add_argument('--early_fusion_sst', type=stR2bool, default=None)
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=None)
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=None)  

    args = parser.parse_args()

    base_name = f"study_{args.embed_method}_{args.optimize_metric}_m{''.join(map(str, args.winter_months))}_ep{args.nb_epochs}ie{args.nb_intra_evals}pat{args.patience}val{args.nb_members_val}lag1{args.include_lag1}seq{args.sequential_lags}roll{args.roll_sst}latw{args.lat_weight}"
    short = {
        'latent_dim': 'ldim','fixed_seed': 'seed', 'bs': 'bs', 'lr': 'lr', 'dr_conv': 'dr1', 'dr_fc': 'dr2', 'fc_dim': 'fc', 'depth': 'dp', 'n_feat': 'feat', 
        'filter_mult': 'mult', 'pool_type': 'pool', 'sst_pool_x': 'px', 
        'sst_pool_y': 'py', 'sst_kx': 'kx', 'sst_ky': 'ky', 'activation': 'act', 
        'pool_strategy': 'pstrat', 'loss_type': 'loss', 'weight_decay': 'wd', 'noise_std': 'ns', 'gradient_clip': 'gc',
        'use_gap': 'gap', 'early_fusion_sst': 'fus', 'sst_lags_months': 'sstlags', 
        'slp_lags_months': 'slplags'
    }
    
    fixed = [f"{short[k]}{''.join(map(str, v)) if isinstance(v, list) else (f'{v:.1e}' if isinstance(v, float) and v < 1e-3 else str(v))}" for k, v in sorted(vars(args).items()) if k in short and v is not None]

    dynamic_name = f"{base_name}_FIXED_{'_'.join(fixed)}" if fixed else f"{base_name}_full_search"
    dynamic_name += f"_optuna_s{args.n_startup_trials_tpe}p{args.n_startup_trials_pruner}_{args.n_warmup_steps}i{args.interval_steps}"

    wgts_gpu = None
    if args.lat_weight:  # <-- Retirer la restriction PCA ici
        sample_path = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-1001.001_1mo.nc"
        try:
            with xr.open_dataset(sample_path) as ds_sample:
                coslat = np.cos(np.deg2rad(ds_sample['lat'].values)).clip(0., 1.)
                h, w = len(coslat), len(ds_sample['lon'].values)
                wgts_flat = np.broadcast_to(np.sqrt(coslat).reshape(h, 1), (h, w)).flatten()
                wgts_gpu = torch.tensor(wgts_flat, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"⚠️ Erreur chargement poids latitude : {e}")
    
    # Chemins fixes et simplifiés
    base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_spatial/"
    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "cnn_optuna_spatial.db")
    csv_path = os.path.join(output_dir, "cnn_optuna_spatial_results.csv")
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
    
    print(f"Début de l'optimisation SPATIALE MAX {args.optimize_metric}) pour {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\n=== Bilan HPO SPATIAL ===")
    trial = study.best_trial
    print(f"  Meilleur {args.optimize_metric.upper()} (Validation Spatiale) : {trial.value:.6f}")
    
    print("\n  --- Performances SPATIALES sur le Set de TEST (Caché) ---")
    print(f"  Global R2 Score               : {trial.user_attrs.get('best_test_R2'):.6f}")
    print(f"  Global L1 Skill Score         : {trial.user_attrs.get('best_test_L1'):.4f}")
    print(f"  Global Spatio-Temporal Corr   : {trial.user_attrs.get('best_test_corr'):.4f}")
    print(f"  Fichier Embedder sélectionné  : {trial.user_attrs.get('embed_file')}")
    print(f"  Tronquature Latente optimale  : {trial.user_attrs.get('current_latent_dim')} composantes")
    print("\n  --- Meilleurs Hyperparamètres ---")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés dans : {csv_path}")