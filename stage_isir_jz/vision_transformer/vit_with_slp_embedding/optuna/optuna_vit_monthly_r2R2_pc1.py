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
import optuna
from optuna.trial import TrialState
import torch.nn.functional as F
import copy

# --- Chemins et imports ---
project_root = Path(__file__).resolve().parent.parent.parent.parent
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

grand_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if grand_parent_dir not in sys.path:
    sys.path.append(grand_parent_dir)

from tools.datasets import Dataset_mensuel
from tools.models import ConvVAE, compute_loss, get_median_prediction, ViT_Latent_SLP_Multimodal_tunable 

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

# Liste complète de tes membres
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
    train_members = members_shuffled[:-2*args.nb_members_val]
    val_members = members_shuffled[-args.nb_members_val:]
    test_members = members_shuffled[-2*args.nb_members_val:-args.nb_members_val] 
    bs = args.bs if args.bs is not None else trial.suggest_categorical("bs", [32, 64, 128])
    lr = args.lr if args.lr is not None else trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    dr = args.dr if args.dr is not None else trial.suggest_float("dr", 0.0, 0.5)

    # --- Paramètres de l'Architecture ViT ---
    depth = args.depth if args.depth is not None else trial.suggest_int("depth", 2, 6)
    num_heads = args.num_heads if args.num_heads is not None else trial.suggest_categorical("num_heads", [2, 4, 8])
    head_dim = args.head_dim if args.head_dim is not None else trial.suggest_categorical("head_dim", [16, 32, 64])
    
    # Astuce cruciale : calculer embed_dim dynamiquement pour garantir la divisibilité
    embed_dim = num_heads * head_dim
    trial.set_user_attr("embed_dim", embed_dim) # Pour garder une trace dans les logs
    
    mlp_ratio = args.mlp_ratio if args.mlp_ratio is not None else trial.suggest_float("mlp_ratio", 2.0, 6.0, step=1.0)
    transformer_act = args.transformer_act if args.transformer_act is not None else trial.suggest_categorical("transformer_act", ["gelu", "relu"])
    norm_first = args.norm_first if args.norm_first is not None else trial.suggest_categorical("norm_first", [True, False])

    use_lags_attention = args.use_lags_attention if args.use_lags_attention is not None else trial.suggest_categorical("use_lags_attention", [True, False])
    pool_strategy = args.pool_strategy if args.pool_strategy is not None else trial.suggest_categorical("pool_strategy", ['cls', 'gap'])
    head_act = args.head_act if args.head_act is not None else trial.suggest_categorical("head_act", ['tanh', 'relu'])

    # Goulot d'étranglement optionnel avant la prédiction finale (on pourrait aussi envisager de choisir directement head_hidden_dim, mais ici on le dérive de embed_dim)
    use_bottleneck = args.use_bottleneck if args.use_bottleneck is not None else trial.suggest_categorical("use_bottleneck", [True, False])
    head_hidden_dim = embed_dim // 2 if use_bottleneck else embed_dim

    loss_type = args.loss_type if args.loss_type is not None else trial.suggest_categorical("loss_type", ['mse', 'l1','correlation','quantile'])
    roll_sst = args.roll_sst if args.roll_sst is not None else trial.suggest_categorical("roll_sst", [True, False])

    sst_patch_y = args.sst_patch_y if args.sst_patch_y is not None else trial.suggest_categorical("sst_patch_y", [5, 17])
    sst_patch_x = args.sst_patch_x if args.sst_patch_x is not None else trial.suggest_categorical("sst_patch_x", [4, 6, 10, 15,20])

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

    winter_months = args.winter_months
    
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 2))
    n_workers = max(0, n_workers - 1)

    train_set = Dataset_mensuel(members=train_members, selected_months=winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=[], roll_sst=roll_sst, slp_std=dynamic_slp_std)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    val_set = Dataset_mensuel(members=val_members, selected_months=winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=[], roll_sst=roll_sst, slp_std=dynamic_slp_std)
    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=active_sst_lags, slp_lags_months=[], roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)

    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    pca_model = None
    vae_model = None
    if args.embed_method == 'pca':
        pca_path = args.embed_path
        pca_model = joblib.load(pca_path)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))

    # ============================================================
    # 3. INITIALISATION DU MODÈLE ViT
    # ============================================================
    out_feature = len(quantiles)*latent_dim if loss_type == 'quantile' else latent_dim

    model = ViT_Latent_SLP_Multimodal_tunable(
        # On garde tes tailles par défaut ici, mais on pourrait les rendre dynamiques
        sst_size=(85, 360), patch_size_sst=(sst_patch_y, sst_patch_x), in_chans_sst=len(sst_lags_months),in_chans_slp=len(slp_lags_months),
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
    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(sst_lags_months), 85, 360).to(device) if len(sst_lags_months) > 0 else None
        dummy_slp = torch.zeros(1, len(slp_lags_months), 53, 113).to(device) if len(slp_lags_months) > 0 else None
        _ = model(dummy_sst, dummy_slp)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("num_params", num_params)

    # L'optimizer AdamW est souvent préféré pour les ViT par rapport à Adam classique
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ============================================================
    # 4. BOUCLE D'ENTRAÎNEMENT ET DE TRACKING DU MAX CORR
    # ============================================================
    nb_epochs = args.nb_epochs 
    best_trial_mse = float('inf')
    best_trial_corr = -float('inf')
    best_r2_score = -float('inf')
    history = []
    patience_counter = 0
    best_model_state = None
    
    total_batches = len(trainloader)
    eval_steps_set = set(np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int))
    eval_steps_set.add(0)

    eval_steps_epoch2 = np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int)
    eval_steps_epoch2 = np.insert(eval_steps_epoch2, 0, 0)
    eval_steps_epoch2_set = set(eval_steps_epoch2)

    for epoch in range(nb_epochs):
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
            
            # Clipping des gradients (très recommandé pour la stabilité des ViT)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- INTRA-EPOCH EVALUATION ---
            if (epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set):
                model.eval()
                all_preds, all_targets = [], []
                
                with torch.no_grad():
                    for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                        v_X_sst = v_X_sst.to(device, non_blocking=True)
                        v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                        
                        v_slp_flat = v_y_target.view(v_y_target.size(0), -1).numpy()
                        if args.embed_method == 'pca':
                            v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                            v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                        elif args.embed_method == 'vae':
                            v_target_embed, _ = vae_model.encode(v_y_target.to(device, non_blocking=True))
                        
                        v_pred = model(v_X_sst, v_X_slp)
                        
                        p = get_median_prediction(v_pred, loss_type, quantiles, latent_dim) if loss_type == 'quantile' else v_pred
                        t = v_target_embed

                        p, t = p.detach(), t.detach()
                        all_preds.append(p)
                        all_targets.append(t)

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

                model.train() 

        # --- END OF EPOCH EVALUATION ---
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for X_sst, X_slp, y_target, _, _, _ in valloader:
                X_sst = X_sst.to(device, non_blocking=True)
                X_slp = X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                
                slp_flat = y_target.view(y_target.size(0), -1).numpy()
                if args.embed_method == 'pca':
                    embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                    target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                elif args.embed_method == 'vae':
                    target_embed, _ = vae_model.encode(y_target.to(device, non_blocking=True))
                
                pred = model(X_sst, X_slp)
                median_pred_latent = get_median_prediction(pred, loss_type, quantiles, latent_dim)
                
                p, t = median_pred_latent.detach(), target_embed.detach()
                all_preds.append(p)
                all_targets.append(t)

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
            for X_sst, X_slp, y_target, _, _, _ in testloader:
                X_sst = X_sst.to(device, non_blocking=True)
                X_slp = X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                
                slp_flat = y_target.view(y_target.size(0), -1).numpy()
                if args.embed_method == 'pca':
                    embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                    target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
                elif args.embed_method == 'vae':
                    target_embed, _ = vae_model.encode(y_target.to(device, non_blocking=True))
                
                pred = model(X_sst, X_slp)
                median_pred_latent = get_median_prediction(pred, loss_type, quantiles, latent_dim)
                
                p, t = median_pred_latent.detach(), target_embed.detach()
                all_preds.append(p)
                all_targets.append(t)

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
    parser.add_argument('--optimize_metric', type=str, choices=['r2', 'correlation'], default='correlation', help="Métrique à maximiser")
    parser.add_argument('--n_trials', type=int, default=100, help='Nombre de combinaisons à tester')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner.')
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca', help='Méthode pour l\'espace latent')
    parser.add_argument('--embed_path', type=str, default="/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/IPCA_latent1_NDJF_1members_normalizeTrue_monthly_reduction_wgtFalse_slp_std505.98/best_pca_model.joblib", help='Chemin vers le modèle PCA pour l\'embedding SLP')
    parser.add_argument('--nb_epochs', type=int, default=20, help='Nombre d\'époques pour chaque essai')
    parser.add_argument('--patience', type=int, default=5, help='Nombre d\'époques sans amélioration avant d\'arrêter l\'essai')
    parser.add_argument('--nb_members_val', type=int, default=5, help='Nombre de membres à utiliser pour la validation')
    parser.add_argument('--n_startup_trials_tpe', type=int, default=10, help='Nombre d\'essais avant d\'activer le pruner')
    parser.add_argument('--n_startup_trials_pruner', type=int, default=10, help='Nombre d\'essais avant d\'activer le pruner')
    parser.add_argument('--n_warmup_steps', type=int, default=3, help='Nombre d\'époques à attendre avant de commencer à évaluer pour le pruner')
    parser.add_argument('--interval_steps', type=int, default=1, help='Intervalle d\'évaluation pour le pruner')
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Nombre de points de validation intra-époque')
    parser.add_argument('--include_lag1', action='store_true', help='Inclure le lag 1 pour la target SST')
    parser.add_argument('--fixed_seed', type=int, default=None, help='Seed fixe pour le shuffle')
    parser.add_argument('--bs', type=int, default=None, help='Taille du batch (optionnel)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (optionnel)')
    parser.add_argument('--dr', type=float, default=None, help='Dropout rate (optionnel)')
    parser.add_argument('--depth', type=int, default=None, help='Profondeur du ViT (optionnel)')
    parser.add_argument('--num_heads', type=int, default=None, help='Nombre de têtes d\'attention (optionnel)')
    parser.add_argument('--head_dim', type=int, default=None, help='Dimension de chaque tête d\'attention (optionnel)')
    parser.add_argument('--mlp_ratio', type=float, default=None, help='Ratio MLP (optionnel)')
    parser.add_argument('--transformer_act', type=str, default=None, choices=['gelu', 'relu'], help='Fonction d\'activation du transformer (optionnel)')
    parser.add_argument('--norm_first', type=bool, default=None, help='Normalisation avant le transformer (optionnel)')
    parser.add_argument('--use_lags_attention', type=bool, default=None, help='Utiliser l\'attention sur les lags (optionnel)')
    parser.add_argument('--pool_strategy', type=str, default=None, choices=['cls', 'gap'], help='Stratégie de pooling (optionnel)')
    parser.add_argument('--head_act', type=str, default=None, choices=['tanh', 'relu'], help='Activation de la tête (optionnel)')
    parser.add_argument('--use_bottleneck', type=bool, default=None, help='Utiliser un goulot d\'étranglement avant la tête (optionnel)')
    parser.add_argument('--loss_type', type=str, default=None, choices=['mse', 'l1', 'correlation', 'quantile'], help='Type de loss (optionnel)')
    parser.add_argument('--roll_sst', type=bool, default=None, help='Appliquer un roll sur la SST (optionnel)')
    parser.add_argument('--sst_patch_y', type=int, default=None, help='Taille du patch SST en Y (optionnel)')
    parser.add_argument('--sst_patch_x', type=int, default=None, help='Taille du patch SST en X (optionnel)')
    parser.add_argument('--sst_lags_months', type=int, nargs='+', default=None, help='Lags SST à utiliser (optionnel)')
    parser.add_argument('--slp_lags_months', type=int, nargs='+', default=None, help='Lags SLP à utiliser (optionnel)')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay pour l\'optimizer (optionnel)')
    parser.add_argument('--latent_dim', type=int, default=1, help='Dimension de l\'espace latent pour l\'embedding SLP (optionnel)')
    args = parser.parse_args()

    dynamic_slp_std = 596.0 
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))
            print(f"\n✅ slp_std extrait avec succès du chemin PCA : {dynamic_slp_std}")
        else:
            print(f"\n⚠️ 'slp_std' introuvable dans le nom du dossier. Utilisation du fallback : {dynamic_slp_std}")
    else:
        print(f"\n⚠️ Aucun modèle pré-entraîné fourni. Utilisation du slp_std par défaut : {dynamic_slp_std}")

    # NOUVEAU: Modification du base_home pour le ViT
    base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_slp_embedding/optuna/"
    # 1. Tronc commun contenant les paramètres structuraux (toujours présents)
    base_name = f"vit_{args.embed_method}_{args.optimize_metric}_m{''.join(map(str, args.winter_months))}_ep{args.nb_epochs}_ie{args.nb_intra_evals}_seed{args.fixed_seed}_pat{args.patience}_val{args.nb_members_val}_lag1{args.include_lag1}"

    # 2. Raccourcis pour les hyperparamètres optionnels à tracker s'ils sont fixés (non None)
    short = {'bs': 'bs', 'lr': 'lr', 'dr': 'dr', 'weight_decay': 'wd', 'depth': 'dp', 'num_heads': 'nh', 'head_dim': 'hd', 'mlp_ratio': 'mlp', 'transformer_act': 'tact', 'norm_first': 'nf', 'use_lags_attention': 'latt', 'pool_strategy': 'pool', 'head_act': 'hact', 'use_bottleneck': 'bn', 'loss_type': 'loss', 'roll_sst': 'roll', 'sst_patch_y': 'py', 'sst_patch_x': 'px', 'sst_lags_months': 'lags'}

    # 3. Extraction et formatage compact (gère les listes, les notations scientifiques des floats et les str/int/bool)
    fixed = [f"{short[k]}{''.join(map(str, v)) if isinstance(v, list) else (f'{v:.1e}' if isinstance(v, float) and v < 1e-3 else str(v))}" for k, v in sorted(vars(args).items()) if k in short and v is not None]

    # 4. Assemblage final avec la configuration de l'échantillonneur Optuna
    dynamic_name = f"{base_name}_FIXED_{'_'.join(fixed)}" if fixed else f"{base_name}_full_search"
    dynamic_name += f"_optuna_s{args.n_startup_trials_tpe}_p{args.n_startup_trials_pruner}_{args.n_warmup_steps}_i{args.interval_steps}"
    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    study_name = dynamic_name
    
    # NOUVEAU: Changement des noms de fichiers
    db_path = os.path.join(output_dir, "vit_optuna.db")
    csv_path = os.path.join(output_dir, "vit_optuna_results.csv")
    storage_name = f"sqlite:///{db_path}"
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials_pruner, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps)
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.n_startup_trials_tpe, seed=42) 

    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    print(f"Début de l'optimisation ViT pour maximiser la {args.optimize_metric} ({args.n_trials} trials)...")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Max metric reached: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    print(f"  Final SST Lags (Months): {trial.user_attrs.get('sst_lags_final')}")
    print(f"  Taille de l'embedding calculée: {trial.user_attrs.get('embed_dim')}")

    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats complets sauvegardés dans : {csv_path}")