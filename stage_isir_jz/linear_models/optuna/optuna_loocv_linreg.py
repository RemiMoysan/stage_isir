import os
import time
import argparse
import copy
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
from optuna.samplers import GridSampler
import time

project_root = Path(__file__).resolve().parent.parent.parent

vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)


from tools.datasets import Dataset, Dataset_mensuel
from tools.models import ConvVAE, vae_loss, compute_loss, get_median_prediction

# ============================================================
# MODÈLE DE RÉGRESSION LINÉAIRE
# ============================================================
class LinearRegressionPredictor(nn.Module):
    def __init__(self, sst_shape=(85, 360), slp_shape=(53, 113), in_chans_sst=3, in_chans_slp=0, out_dim=128):
        super().__init__()
        self.sst_size = in_chans_sst * sst_shape[0] * sst_shape[1]
        self.slp_size = in_chans_slp * slp_shape[0] * slp_shape[1]
        self.total_input_size = self.sst_size + self.slp_size
        
        self.linear = nn.Linear(self.total_input_size, out_dim)

    def forward(self, x_sst, x_slp):
        batch_size = x_sst.size(0)
        x_sst_flat = x_sst.view(batch_size, -1)
        
        if self.slp_size > 0:
            x_slp_flat = x_slp.view(batch_size, -1)
            x = torch.cat([x_sst_flat, x_slp_flat], dim=1)
        else:
            x = x_sst_flat
            
        return self.linear(x)


# ============================================================
# DEVICE CONFIGURATION
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

def objective(trial):
    # ============================================================
    # 1. PARAMÈTRES VARIABLES (Grille LOOCV)
    # ============================================================
    val_member = trial.suggest_categorical("val_member", train_val_members)
    val_members = [val_member]
    train_members = [m for m in train_val_members if m != val_member]

    trial.set_user_attr("val_member", val_member)
        
    # Paramètres fixés via args
    latent_dim = args.latent_dim
    bs = args.bs
    lr = args.lr
    
    # ============================================================
    # 2. PRÉPARATION DES DONNÉES
    # ============================================================
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 2))
    n_workers = max(0, n_workers - 1)

    training_set = Dataset_mensuel(members=train_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)

    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    

    # ============================================================
    # 3. INITIALISATION DE L'EMBEDDER ET DU MODÈLE
    # ============================================================
    pca_model = None
    vae_model = None
    if args.embed_method == 'pca':
        pca_model = joblib.load(args.embed_path)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

    model = LinearRegressionPredictor(
        sst_shape=(85, 360), 
        slp_shape=(53, 113), 
        in_chans_sst=len(args.sst_lags_months), 
        in_chans_slp=len(args.slp_lags_months), 
        out_dim=latent_dim
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("num_params", num_params)

    # ============================================================
    # 4. ENTRAÎNEMENT OU SOLUTION EXACTE
    # ============================================================
    best_trial_mse = float('inf')
    best_trial_corr = -float('inf')
    best_R2_score = -float('inf')
    best_target_metric = -float('inf')
    history = []
    patience_counter = 0
    best_model_state = None

    # --- CAS 1 : EXACT SOLVER ---
    if args.exact_solver:
        model.eval()
        X_list, Y_list = [], []
        current_samples = 0
        
        for X_sst, X_slp, y_target, _, _, _ in trainloader:
            batch_size = X_sst.size(0)
            X_sst_flat = X_sst.view(batch_size, -1).cpu()
            
            if X_slp.numel() > 0:
                X_slp_flat = X_slp.view(batch_size, -1).cpu()
                X_batch = torch.cat([X_sst_flat, X_slp_flat], dim=1)
            else:
                X_batch = X_sst_flat
                
            if args.embed_method == 'pca':
                slp_flat = y_target.view(batch_size, -1).numpy()
                embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                Y_batch = torch.tensor(embed_np, dtype=torch.float32)
            elif args.embed_method == 'vae':
                with torch.no_grad():
                    Y_batch, _ = vae_model.encode(y_target.to(device))
                    Y_batch = Y_batch.cpu()
                    
            X_list.append(X_batch)
            Y_list.append(Y_batch)
            current_samples += batch_size
            if current_samples >= args.max_samples_exact: break
                
        X = torch.cat(X_list, dim=0)[:args.max_samples_exact] 
        Y = torch.cat(Y_list, dim=0)[:args.max_samples_exact] 
        
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        X_c = X - X_mean
        Y_c = Y - Y_mean
        
        N, D = X.shape
        lambda_ridge = args.alpha_penalty * N 
        
        K_mat = X_c @ X_c.T + lambda_ridge * torch.eye(N) 
        dual_coef = torch.linalg.solve(K_mat, Y_c)
        W_exact = X_c.T @ dual_coef
        bias_exact = Y_mean.squeeze() - (X_mean @ W_exact).squeeze()
        
        with torch.no_grad():
            model.linear.weight.copy_(W_exact.T.to(device))
            model.linear.bias.copy_(bias_exact.to(device))

        # Évaluation unique sur le set de validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader:
                v_X_sst = v_X_sst.to(device, non_blocking=True)
                v_X_slp = v_X_slp.to(device, non_blocking=True)
                
                if args.embed_method == 'pca':
                    v_slp_flat = v_y_target.view(v_y_target.size(0), -1).cpu().numpy()
                    v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                    v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                elif args.embed_method == 'vae':
                    v_target_embed, _ = vae_model.encode(v_y_target.to(device))
                    
                v_pred = model(v_X_sst, v_X_slp)
                vp = get_median_prediction(v_pred, args.loss_type, args.quantiles, args.latent_dim) if args.loss_type == 'quantile' else v_pred
                all_preds.append(vp)
                all_targets.append(v_target_embed)

        val_preds_tensor = torch.cat(all_preds, dim=0)
        val_targets_tensor = torch.cat(all_targets, dim=0)
        
        # Calcul de toutes les métriques
        best_trial_mse = F.mse_loss(val_preds_tensor, val_targets_tensor).item()
        val_target_variance = torch.var(val_targets_tensor, unbiased=False).item()
        best_R2_score = 1.0 - (best_trial_mse / val_target_variance) if val_target_variance > 0 else 0.0
        
        p, t = val_preds_tensor, val_targets_tensor
        p_mean, t_mean = p.mean(dim=0), t.mean(dim=0)
        p_var, t_var = ((p - p_mean)**2).mean(dim=0), ((t - t_mean)**2).mean(dim=0)
        cov = ((p - p_mean)*(t - t_mean)).mean(dim=0)
        best_trial_corr = (cov / torch.sqrt(p_var * t_var + 1e-8)).mean().item()
        
        trial.set_user_attr("val_target_variance", val_target_variance)
        trial.set_user_attr("best_R2_score", best_R2_score)
        trial.set_user_attr("best_trial_mse", best_trial_mse)
        trial.set_user_attr("best_trial_corr", best_trial_corr)
        
        # Retourne la métrique ciblée par l'optimisation
        return best_R2_score if args.optimize_metric == 'R2' else best_trial_corr

    # --- CAS 2 : ENTRAÎNEMENT CLASSIQUE (Descente de gradient) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # --- SETUP INTRA-EVALUATION ---
    total_batches = len(trainloader)
    eval_steps_set = set(np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int)) if args.nb_intra_evals > 0 else set()
    eval_steps_set.add(0)

    eval_steps_epoch2 = np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int) if args.nb_intra_evals > 0 else []
    if len(eval_steps_epoch2) > 0:
        eval_steps_epoch2 = np.insert(eval_steps_epoch2, 0, 0)
    eval_steps_epoch2_set = set(eval_steps_epoch2)

    for epoch in range(args.nb_epochs):
        model.train()
        for batch_idx, (X_sst, X_slp, y_target, _, _, _) in enumerate(trainloader):
            optimizer.zero_grad()
            X_sst = X_sst.to(device, non_blocking=True)
            X_slp = X_slp.to(device, non_blocking=True) if len(args.slp_lags_months) > 0 else None
            
            if args.embed_method == 'pca':
                slp_flat = y_target.view(y_target.size(0), -1).numpy()
                embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
            elif args.embed_method == 'vae':
                target_embed, _ = vae_model.encode(y_target.to(device))
                
            pred = model(X_sst, X_slp)
            base_loss = compute_loss(pred, target_embed, loss_type=args.optimize_metric, reduction='mean')
            
            if args.penalty_type == 'l1':
                penalty = torch.norm(model.linear.weight, p=1)
            elif args.penalty_type == 'l2':
                penalty = torch.sum(model.linear.weight ** 2)
            else:
                penalty = 0.0
                
            loss = base_loss + args.alpha_penalty * penalty
            loss.backward()
            optimizer.step()

            # --- INTRA-EPOCH EVALUATION ---
            if args.nb_intra_evals > 0 and ((epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set)):
                model.eval()
                all_preds_intra, all_targets_intra = [], []
                
                with torch.no_grad():
                    for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                        v_X_sst = v_X_sst.to(device, non_blocking=True)
                        v_X_slp = v_X_slp.to(device, non_blocking=True) if len(args.slp_lags_months) > 0 else None
                        
                        if args.embed_method == 'pca':
                            v_slp_flat = v_y_target.view(v_y_target.size(0), -1).numpy()
                            v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                            v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                        elif args.embed_method == 'vae':
                            v_target_embed, _ = vae_model.encode(v_y_target.to(device))
                        
                        v_pred = model(v_X_sst, v_X_slp)
                        vp = get_median_prediction(v_pred, args.loss_type, args.quantiles, args.latent_dim) if args.loss_type == 'quantile' else v_pred
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

                model.train() # Retour à l'entraînement

        # Fin d'époque : Évaluation
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader:
                v_X_sst = v_X_sst.to(device, non_blocking=True)
                v_X_slp = v_X_slp.to(device, non_blocking=True) if len(args.slp_lags_months) > 0 else None
                
                if args.embed_method == 'pca':
                    v_slp_flat = v_y_target.view(v_y_target.size(0), -1).numpy()
                    v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                    v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                elif args.embed_method == 'vae':
                    v_target_embed, _ = vae_model.encode(v_y_target.to(device))
                
                v_pred = model(v_X_sst, v_X_slp)
                vp = get_median_prediction(v_pred, args.loss_type, args.quantiles, args.latent_dim) if args.loss_type == 'quantile' else v_pred
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
            v_X_slp = v_X_slp.to(device, non_blocking=True) if len(args.slp_lags_months) > 0 else None
            
            if args.embed_method == 'pca':
                v_slp_flat = v_y_target.view(v_y_target.size(0), -1).numpy()
                v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
            elif args.embed_method == 'vae':
                v_target_embed, _ = vae_model.encode(v_y_target.to(device))
            
            v_pred = model(v_X_sst, v_X_slp)
            vp = get_median_prediction(v_pred, args.loss_type, args.quantiles, args.latent_dim) if args.loss_type == 'quantile' else v_pred
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
    # On ajoute uniquement les arguments nécessaires à la config globale
    parser.add_argument('--optimize_metric', type=str, choices=['R2', 'correlation'], default='correlation', help="Métrique à maximiser via Optuna")
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca')
    parser.add_argument('--embed_path', type=str, required=True)
    parser.add_argument('--machine', type=str, default='jean-zay-work')
    parser.add_argument('--nb_intra_evals', type=int, default=15)
    parser.add_argument('--test_members', type=str, nargs='+', default=['1001.001'])
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--alpha_penalty', type=float, default=1e-5)
    parser.add_argument('--penalty_type', type=str, choices=['l1', 'l2'], default='l2')

    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    
    parser.add_argument('--exact_solver', action='store_true')
    parser.add_argument('--max_samples_exact', type=int, default=2000)
    
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4])
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--include_lag1', action='store_true')

    
    args = parser.parse_args()

    # Routage dynamique des dossiers
    if args.machine == 'hacienda': base_home = "/home/moysan/stage_isir_jz/linear_models/optuna/optuna_loocv/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']: base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/optuna/optuna_loocv/"
    elif args.machine == 'mac_local': base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/linear_models/optuna/optuna_loocv/"

    # Fallback slp_std
    dynamic_slp_std = 596.0 
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match: dynamic_slp_std = float(match.group(1))

    # Nom d'étude dynamique
    exact_str = "_exact" if args.exact_solver else ""
    study_name = f"LOOCV_LinReg_{args.optimize_metric}_{args.penalty_type}_{args.alpha_penalty}_{args.embed_method}{exact_str}_bs{args.bs}_lr{args.lr}_months{''.join(map(str, args.winter_months))}_sstlags{''.join(map(str, args.sst_lags_months))}_testmember{''.join(args.test_members)}"
    
    output_dir = os.path.join(base_home, study_name)
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, "linreg_optuna.db")
    csv_path = os.path.join(output_dir, "linreg_optuna_results.csv")
    storage_name = f"sqlite:///{db_path}"

    # ============================================================
    # LISTE DES MEMBRES
    # ============================================================
    ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    test_members = args.test_members
    train_val_members = [m for m in ALL_MEMBERS if m not in test_members]

    search_space = {
        "val_member": train_val_members
    }
    
    sampler = GridSampler(search_space)

    # Définition de la direction d'Optuna en fonction du choix utilisateur

    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction='maximize', 
        load_if_exists=True,
        sampler=sampler
    )
    
    total_trials = len(train_val_members)
    print(f"Début du LOOCV LinReg pour maximiser la {args.optimize_metric.upper()} ({total_trials} essais au total)...")
    study.optimize(objective, n_trials=total_trials) 
    
    print("\nLOOCV Terminé !")
    
    df = study.trials_dataframe()
    
    # Nettoyage de la colonne 'search_space' pour garder un CSV lisible
    cols_to_drop = [col for col in df.columns if 'search_space' in col]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    df.to_csv(csv_path, index=False)
    print(f"Résultats nettoyés et sauvegardés dans : {csv_path}")

    if not df.empty:
        mean_mse = df['user_attrs_best_trial_mse'].mean()
        std_mse = df['user_attrs_best_trial_mse'].std()
        mean_R2 = df['user_attrs_best_R2_score'].mean()
        mean_corr = df['user_attrs_best_trial_corr'].mean()
        print(f"\n=== Bilan LOOCV LinReg ({args.optimize_metric.upper()} Optimisé) ===")
        print(f"MSE Moyenne Finale  : {mean_mse:.4f} +/- {std_mse:.4f}")
        print(f"R2 Score Moyen      : {mean_R2:.4f}")
        print(f"Corrélation Moyenne : {mean_corr:.4f}")

        # On garde MSE de validation si tu ne calcules pas MSE de test à la fin,
        # mais on ajoute les métriques TEST
        mean_mse_val = df['user_attrs_best_trial_mse'].mean() 
        std_mse_val = df['user_attrs_best_trial_mse'].std()
        
        mean_test_R2 = df['user_attrs_best_test_R2'].mean()
        mean_test_corr = df['user_attrs_best_test_corr'].mean()
        
        print(f"\n=== Bilan LOOCV LinReg ({args.optimize_metric.upper()} Optimisé) ===")
        print(f"MSE Moyenne Finale (Validation) : {mean_mse_val:.4f} +/- {std_mse_val:.4f}")
        print(f"R2 Score Moyen (TEST SET)       : {mean_test_R2:.4f}")
        print(f"Corrélation Moyenne (TEST SET)  : {mean_test_corr:.4f}")