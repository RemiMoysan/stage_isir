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
import torch
import torch.nn.functional as F
import optuna
from optuna.trial import TrialState
import time
import copy

project_root = Path(__file__).resolve().parent.parent.parent.parent
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

grand_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if grand_parent_dir not in sys.path:
    sys.path.append(grand_parent_dir)

from tools.datasets import Dataset_mensuel
from tools_cnn.models import CNN_Latent_SLP_Multimodal1_tunable
from tools.models import ConvVAE, compute_loss, get_median_prediction

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

def objective(trial):
    # ============================================================
    # 1. DÉFINITION DES HYPERPARAMÈTRES (Fixes ou Tunés)
    # ============================================================
    # Logique : Si args.nom_param est fourni (non None), on l'utilise. Sinon, Optuna s'en charge.
    nb_members_val = args.nb_members_val if args.nb_members_val is not None else trial.suggest_int("nb_members_val", 1, 20)
    seed = args.fixed_seed if args.fixed_seed is not None else trial.suggest_int("seed", 0, 1000)
    rng = random.Random(seed)
    members_shuffled = ALL_MEMBERS.copy()
    rng.shuffle(members_shuffled)
    train_members = members_shuffled[:-2*nb_members_val]
    val_members = members_shuffled[-nb_members_val:]
    test_members = members_shuffled[-2*nb_members_val:-nb_members_val]

    bs = args.bs if args.bs is not None else trial.suggest_categorical("bs", [32, 64, 128])
    lr = args.lr if args.lr is not None else trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    dr = args.dr if args.dr is not None else trial.suggest_float("dr", 0.0, 0.6)
    
    depth = args.depth if args.depth is not None else trial.suggest_int("depth", 2, 4)
    n_feat = args.n_feat if args.n_feat is not None else trial.suggest_int("n_feat", 4, 32)
    filter_mult = args.filter_mult if args.filter_mult is not None else trial.suggest_categorical("filter_mult", [1, 2])
    
    pool_type = args.pool_type if args.pool_type is not None else trial.suggest_categorical("pool_type", ['max', 'avg'])
    sst_pool_x = args.sst_pool_x if args.sst_pool_x is not None else trial.suggest_categorical("sst_pool_x", [2, 3])
    sst_pool_y = args.sst_pool_y if args.sst_pool_y is not None else trial.suggest_categorical("sst_pool_y", [2, 3])
    sst_kx = args.sst_kx if args.sst_kx is not None else trial.suggest_categorical("sst_kx", [3, 5])
    sst_ky = args.sst_ky if args.sst_ky is not None else trial.suggest_categorical("sst_ky", [3, 5])
    
    activation = args.activation if args.activation is not None else trial.suggest_categorical("activation", ['tanh', 'relu'])
    pool_strategy = args.pool_strategy if args.pool_strategy is not None else trial.suggest_categorical("pool_strategy", ['progressive', 'standard'])
    use_gap = args.use_gap if args.use_gap is not None else trial.suggest_categorical("use_gap", [True, False])
    early_fusion_sst = args.early_fusion_sst if args.early_fusion_sst is not None else trial.suggest_categorical("early_fusion_sst", [True, False])
    
    loss_type = args.loss_type if args.loss_type is not None else trial.suggest_categorical("loss_type", ['mse', 'l1','correlation','quantile'])
    weight_decay = args.weight_decay if args.weight_decay is not None else trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

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

    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    latent_dim = args.latent_dim
    
    trial.set_user_attr("sst_lags_final", sst_lags_months)
    trial.set_user_attr("slp_lags_final", slp_lags_months)

    # ============================================================
    # 2. PRÉPARATION DES DONNÉES
    # ============================================================
    
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 2))
    n_workers = max(0, n_workers - 1)

    train_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)

    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    # ============================================================
    # 3. INITIALISATION DU MODÈLE
    # ============================================================
    pca_model = None
    vae_model = None
    if args.embed_method == 'pca':
        pca_path = args.embed_path
        pca_model = joblib.load(pca_path)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))

    out_feature = len(quantiles)*latent_dim if loss_type == 'quantile' else latent_dim
    model = CNN_Latent_SLP_Multimodal1_tunable(
        dr=dr, nb_out=out_feature, in_chans_sst=len(sst_lags_months), in_chans_slp=len(slp_lags_months), 
        n_feat=n_feat, early_fusion_sst=early_fusion_sst, depth=depth, filter_mult=filter_mult,
        sst_kx=sst_kx, sst_ky=sst_ky, sst_pool_x=sst_pool_x, sst_pool_y=sst_pool_y,
        pool_type=pool_type, pool_strategy=pool_strategy, activation=activation, use_gap=use_gap
    ).to(device)

    # ---> AJOUTE CE BLOC ICI <---
    with torch.no_grad():
        if len(sst_lags_months) >0:
            dummy_sst = torch.zeros(1, len(sst_lags_months), 85, 360).to(device)
        else:
            dummy_sst = None
        if len(slp_lags_months) > 0:
            dummy_slp = torch.zeros(1, len(slp_lags_months), 53, 113).to(device)
        else:
            dummy_slp = None
        _ = model(dummy_sst, dummy_slp)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("num_params", num_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ============================================================
    # 4. BOUCLE D'ENTRAÎNEMENT ET TRACKING MSE / VARIANCE
    # ============================================================
    best_trial_mse = float('inf')
    best_trial_corr = -float('inf')
    best_R2_score = -float('inf')
    best_target_metric = -float('inf')
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
            X_sst = X_sst.to(device, non_blocking=True)
            X_slp = X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
            
            slp_flat = y_target.view(y_target.size(0), -1).numpy()
            embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
            target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
            
            pred = model(X_sst, X_slp)
            loss = compute_loss(pred, target_embed, loss_type, quantiles=quantiles, reduction='mean')
            
            loss.backward()
            optimizer.step()

            # --- INTRA-EPOCH EVALUATION (Seulement Epoch 0 et 1 pour la vitesse) ---
            if (epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set):
                model.eval()
                all_preds_intra, all_targets_intra = [], []
                
                with torch.no_grad():
                    for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                        v_X_sst = v_X_sst.to(device, non_blocking=True)
                        v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                        
                        if args.embed_method == 'pca':
                            v_slp_flat = v_y_target.view(v_y_target.size(0), -1).numpy()
                            v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                            v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                        elif args.embed_method == 'vae':
                            v_target_embed, _ = vae_model.encode(v_y_target.to(device))
                        
                        v_pred = model(v_X_sst, v_X_slp)
                        vp = get_median_prediction(v_pred, loss_type, quantiles, latent_dim) if loss_type == 'quantile' else v_pred
                        all_preds_intra.append(vp)
                        all_targets_intra.append(v_target_embed)

                val_preds_intra = torch.cat(all_preds_intra, dim=0)
                val_targets_intra = torch.cat(all_targets_intra, dim=0)

                # Calcul des 3 métriques
                intra_mse = F.mse_loss(val_preds_intra, val_targets_intra).item()
                intra_target_var = torch.var(val_targets_intra, unbiased=False).item()
                intra_R2 = 1.0 - (intra_mse / intra_target_var) if intra_target_var > 0 else 0.0
                
                p, t = val_preds_intra, val_targets_intra
                p_mean, t_mean = p.mean(dim=0), t.mean(dim=0)
                p_var, t_var = ((p - p_mean)**2).mean(dim=0), ((t - t_mean)**2).mean(dim=0)
                cov = ((p - p_mean)*(t - t_mean)).mean(dim=0)
                intra_corr = (cov / torch.sqrt(p_var * t_var + 1e-8)).mean().item()

                # Ajout à l'historique avec le point flottant (ex: 0.5 = milieu de l'époque 0)
                current_step = epoch + batch_idx / total_batches
                history.append((current_step, intra_R2, intra_corr))

                # Update des meilleurs scores globaux (si pic trouvé ici)
                if intra_mse < best_trial_mse: best_trial_mse = intra_mse
                if intra_R2 > best_R2_score: best_R2_score = intra_R2
                if intra_corr > best_trial_corr: best_trial_corr = intra_corr
                
                current_metric = intra_R2 if args.optimize_metric == 'R2' else intra_corr
                if current_metric > best_target_metric:
                    best_target_metric = current_metric
                    best_model_state = copy.deepcopy(model.state_dict())

                model.train() 

        # --- END OF EPOCH EVALUATION ---
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader:
                v_X_sst = v_X_sst.to(device, non_blocking=True)
                v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                
                if args.embed_method == 'pca':
                    v_slp_flat = v_y_target.view(v_y_target.size(0), -1).numpy()
                    v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                    v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                elif args.embed_method == 'vae':
                    v_target_embed, _ = vae_model.encode(v_y_target.to(device))
                
                v_pred = model(v_X_sst, v_X_slp)
                vp = get_median_prediction(v_pred, loss_type, quantiles, latent_dim) if loss_type == 'quantile' else v_pred
                all_preds.append(vp)
                all_targets.append(v_target_embed)

        val_preds_tensor = torch.cat(all_preds, dim=0)
        val_targets_tensor = torch.cat(all_targets, dim=0)

        epoch_mse = F.mse_loss(val_preds_tensor, val_targets_tensor).item()
        val_target_variance = torch.var(val_targets_tensor, unbiased=False).item()
        epoch_R2 = 1.0 - (epoch_mse / val_target_variance) if val_target_variance > 0 else 0.0

        p, t = val_preds_tensor, val_targets_tensor
        p_mean, t_mean = p.mean(dim=0), t.mean(dim=0)
        p_var, t_var = ((p - p_mean)**2).mean(dim=0), ((t - t_mean)**2).mean(dim=0)
        cov = ((p - p_mean)*(t - t_mean)).mean(dim=0)
        epoch_corr = (cov / torch.sqrt(p_var * t_var + 1e-8)).mean().item()

        history.append((epoch, epoch_R2, epoch_corr))

        if epoch == 0:
             trial.set_user_attr("val_target_variance", val_target_variance)

        # Mise à jour des meilleurs scores absolus pour le rapport
        if epoch_mse < best_trial_mse: best_trial_mse = epoch_mse
        if epoch_R2 > best_R2_score: best_R2_score = epoch_R2
        if epoch_corr > best_trial_corr: best_trial_corr = epoch_corr

        # Early stopping et sauvegarde basés STRICTEMENT sur la métrique choisie
        current_metric = epoch_R2 if args.optimize_metric == 'R2' else epoch_corr
        if current_metric > best_target_metric:
            best_target_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Pruning sur la métrique cible
        trial.report(best_target_metric, epoch)
        if trial.should_prune():
            trial.set_user_attr("best_trial_mse", best_trial_mse) 
            trial.set_user_attr("best_R2_score", best_R2_score)
            trial.set_user_attr("best_trial_corr", best_trial_corr)
            trial.set_user_attr("R2_corr_history", history)
            raise optuna.exceptions.TrialPruned()
        
        if patience_counter >= args.patience:
            break

    # Enregistrement final pour ce trial
    trial.set_user_attr("best_trial_mse", best_trial_mse) 
    trial.set_user_attr("best_R2_score", best_R2_score)
    trial.set_user_attr("best_trial_corr", best_trial_corr)
    trial.set_user_attr("R2_corr_history", history)
    

    # --- TEST AVEC LE MEILLEUR MODÈLE ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for v_X_sst, v_X_slp, v_y_target, _, _, _ in testloader:
            v_X_sst = v_X_sst.to(device, non_blocking=True)
            v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
            
            if args.embed_method == 'pca':
                v_slp_flat = v_y_target.view(v_y_target.size(0), -1).numpy()
                v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
            elif args.embed_method == 'vae':
                v_target_embed, _ = vae_model.encode(v_y_target.to(device))
            
            v_pred = model(v_X_sst, v_X_slp)
            vp = get_median_prediction(v_pred, loss_type, quantiles, latent_dim) if loss_type == 'quantile' else v_pred
            all_preds.append(vp)
            all_targets.append(v_target_embed)

    val_preds_tensor = torch.cat(all_preds, dim=0)
    val_targets_tensor = torch.cat(all_targets, dim=0)

    test_mse = F.mse_loss(val_preds_tensor, val_targets_tensor).item()
    test_target_variance = torch.var(val_targets_tensor, unbiased=False).item()
    test_R2 = 1.0 - (test_mse / test_target_variance) if test_target_variance > 0 else 0.0

    p, t = val_preds_tensor, val_targets_tensor
    p_mean, t_mean = p.mean(dim=0), t.mean(dim=0)
    p_var, t_var = ((p - p_mean)**2).mean(dim=0), ((t - t_mean)**2).mean(dim=0)
    cov = ((p - p_mean)*(t - t_mean)).mean(dim=0)
    test_corr = (cov / torch.sqrt(p_var * t_var + 1e-8)).mean().item()

    trial.set_user_attr("test_target_variance", test_target_variance)
    trial.set_user_attr("best_test_mse", test_mse)
    trial.set_user_attr("best_test_R2", test_R2)
    trial.set_user_attr("best_test_corr", test_corr)

    print(f"Trial terminé, temps écoulé depuis le début de l'optimisation: {time.time() - start_time:.2f} secondes")
    return best_target_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Configuration Optuna de base
    parser.add_argument('--optimize_metric', type=str, choices=['R2', 'correlation'], default='correlation', help="Métrique à maximiser")
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
    parser.add_argument('--embed_method', type=str, default='pca')
    parser.add_argument('--embed_path', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=1, help="Dimension latente pour l'embedding (PCA ou VAE) target")
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--include_lag1', action='store_true')

    # HYPERPARAMÈTRES DYNAMIQUES (default=None => sera tuné par Optuna si non spécifié)
    parser.add_argument('--fixed_seed', type=int, default=None)
    # parser.add_argument('--nb_members_train', type=int, default=None) #obsolète car on prend tout sauf la val (grid search à part pour impact du nbr de membre)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dr', type=float, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--n_feat', type=int, default=None)
    parser.add_argument('--filter_mult', type=int, default=None)
    parser.add_argument('--pool_type', type=str, choices=['max', 'avg'], default=None)
    parser.add_argument('--sst_pool_x', type=int, default=None)
    parser.add_argument('--sst_pool_y', type=int, default=None)
    parser.add_argument('--sst_kx', type=int, default=None)
    parser.add_argument('--sst_ky', type=int, default=None)
    parser.add_argument('--activation', type=str, choices=['tanh', 'relu'], default=None)
    parser.add_argument('--pool_strategy', type=str, choices=['progressive', 'standard'], default=None)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1','correlation','quantile'], default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    
    # Pour les booléens dynamiques, on utilise un type spécifique car action='store_true' ne permet pas le None
    def stR2bool(v):
        if v is None or v.lower() == 'none': return None
        return v.lower() in ("yes", "true", "t", "1")
    parser.add_argument('--use_gap', type=stR2bool, default=None)
    parser.add_argument('--early_fusion_sst', type=stR2bool, default=None)
    
    # Lags spécifiques (si None, ils seront tunés)
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=None)
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=None)

    args = parser.parse_args()

    # Parsing dynamique pour slp_std...
    dynamic_slp_std = 596.0  
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))

    # Création du nom d'étude dynamique en fonction des paramètres fixés
    base_name = f"cnn_{args.embed_method}{args.latent_dim}_{args.optimize_metric}_m{''.join(map(str, args.winter_months))}_ep{args.nb_epochs}_ie{args.nb_intra_evals}_pat{args.patience}_val{args.nb_members_val}"
    
    # Raccourcis pour un nom compact
    short = {
        'bs': 'bs', 'lr': 'lr', 'dr': 'dr', 'depth': 'dp', 'n_feat': 'feat', 
        'filter_mult': 'mult', 'pool_type': 'pool', 'sst_pool_x': 'px', 
        'sst_pool_y': 'py', 'sst_kx': 'kx', 'sst_ky': 'ky', 'activation': 'act', 
        'pool_strategy': 'pstrat', 'loss_type': 'loss', 'weight_decay': 'wd', 
        'use_gap': 'gap', 'early_fusion_sst': 'fus', 'sst_lags_months': 'sstlags', 
        'slp_lags_months': 'slplags', 'fixed_seed': 'seed'
    }
    
    # Extraction et formatage
    fixed = [f"{short[k]}{''.join(map(str, v)) if isinstance(v, list) else (f'{v:.1e}' if isinstance(v, float) and v < 1e-3 else str(v))}" for k, v in sorted(vars(args).items()) if k in short and v is not None]

    dynamic_name = f"{base_name}_FIXED_{'_'.join(fixed)}" if fixed else f"{base_name}_full_search"
    dynamic_name += f"_optuna_s{args.n_startup_trials_tpe}_p{args.n_startup_trials_pruner}_{args.n_warmup_steps}_i{args.interval_steps}"
    
    base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna/"
    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, "cnn_optuna.db")
    csv_path = os.path.join(output_dir, "cnn_optuna_results.csv")
    storage_name = f"sqlite:///{db_path}"
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials_pruner, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps)
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.n_startup_trials_tpe, seed=42) 

    study = optuna.create_study(
        study_name=dynamic_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    print(f"Début de l'optimisation pour MAXIMISER {args.optimize_metric} ({args.n_trials} trials)...")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\n=== Bilan HPO CNN ===")
    trial = study.best_trial
    print(f"  Max {args.optimize_metric.upper()} atteint (Validation) : {trial.value:.4f}")
    
    print("\n  --- Performances sur le Set de TEST (Caché) ---")
    print(f"  R2 Score (Skill Score) : {trial.user_attrs.get('best_test_R2'):.4f}")
    print(f"  Corrélation            : {trial.user_attrs.get('best_test_corr'):.4f}")
    print(f"  MSE                    : {trial.user_attrs.get('best_test_mse'):.4f}")
    print(f"  Variance de la Target  : {trial.user_attrs.get('test_target_variance'):.4f}")
    
    print("\n  --- Meilleurs Hyperparamètres ---")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    print(f"\n  Lags SST finaux : {trial.user_attrs.get('sst_lags_final')}")
    print(f"  Lags SLP finaux : {trial.user_attrs.get('slp_lags_final')}")

    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés dans : {csv_path}")