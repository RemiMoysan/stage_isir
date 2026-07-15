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
from optuna.samplers import GridSampler
import copy
import time

project_root = Path(__file__).resolve().parent.parent.parent.parent
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

grand_parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if grand_parent_dir not in sys.path:
    sys.path.append(grand_parent_dir)

from tools.datasets import Dataset_mensuel
from tools_cnn.models import CNN_Latent_SLP_Multimodal1, CNN_Latent_SLP_Multimodal1_tunable
from tools.models import ConvVAE, compute_loss, get_median_prediction

# Dans ce code, on fixe tous les hyperparamètres à des valeurs raisonables et on fait optuna pour voir l'impact du choix du membre de validation ie LOOCV (il faudra voir à part le choix du nombre de membre d'entraînement)

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

def objective(trial):
    # ============================================================
    # 1. PARAMÈTRES VARIABLES (Gérés par le GridSampler)
    # ============================================================
    # Optuna va piocher un membre de validation différent à chaque trial selon la grille
    val_member = trial.suggest_categorical("val_member", train_val_members)
    
    # Construction des sets (LOOCV : on garde 1 membre pour la val, tout le reste pour le train)
    val_members = [val_member]
    train_members = [m for m in train_val_members if m != val_member]

    # ============================================================
    # 2. HYPERPARAMÈTRES FIXÉS (Par l'utilisateur via argparse)
    # ============================================================
    bs = args.bs
    lr = args.lr
    dr = args.dr
    pool_type = args.pool_type
    sst_pool_x = args.sst_pool_x
    sst_pool_y = args.sst_pool_y
    pool_strategy = args.pool_strategy
    activation = args.activation
    use_gap = args.use_gap
    sst_kx = args.sst_kx
    sst_ky = args.sst_ky
    depth = args.depth
    filter_mult = args.filter_mult
    loss_type = args.loss_type
    roll_sst = args.roll_sst
    early_fusion_sst = args.early_fusion_sst
    n_feat = args.n_feat
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    latent_dim = args.latent_dim
    roll_sst = args.roll_sst
    early_fusion_sst = args.early_fusion_sst
    n_feat = args.n_feat

    # Pour les lags, on les fixe en dur ou via args
    sst_lags_months = args.sst_lags_months 
    slp_lags_months = args.slp_lags_months
    
    trial.set_user_attr("val_member", val_member)


    # ============================================================
    # 2. PRÉPARATION DES DONNÉES (Géré dans le trial car le seed change)
    # ============================================================

    winter_months = args.winter_months
    
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 2))
    n_workers = max(0, n_workers - 1)

    train_set = Dataset_mensuel(members=train_members, selected_months=winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=roll_sst, slp_std=dynamic_slp_std)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    val_set = Dataset_mensuel(members=val_members, selected_months=winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=roll_sst, slp_std=dynamic_slp_std)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)

    test_set = Dataset_mensuel(members=test_members, selected_months=args.winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst, slp_std=dynamic_slp_std)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # Chargement Embedder 1d normalize sans weight lat
    pca_model = None
    vae_model = None
    if args.embed_method == 'pca':
        pca_path = args.embed_path
        pca_model = joblib.load(pca_path)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))

    # ============================================================
    # 3. INITIALISATION DU MODÈLE
    # ============================================================

    out_feature = len(quantiles)*latent_dim if loss_type == 'quantile' else latent_dim

    model = CNN_Latent_SLP_Multimodal1_tunable(
        dr=dr, 
        nb_out=out_feature, 
        in_chans_sst=len(sst_lags_months), 
        in_chans_slp=len(slp_lags_months), 
        n_feat=n_feat, 
        early_fusion_sst=early_fusion_sst,
        depth = depth,
        filter_mult = filter_mult,
        sst_kx = sst_kx,
        sst_ky = sst_ky,
        sst_pool_x = sst_pool_x,
        sst_pool_y = sst_pool_y,
        pool_type = pool_type,
        pool_strategy = pool_strategy,
        activation = activation,
        use_gap = use_gap
    ).to(device)

    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(sst_lags_months), 85, 360).to(device)
        dummy_slp = None
        _ = model(dummy_sst, dummy_slp)

    # NOUVEAU : Calcul et sauvegarde du nombre de paramètres
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("num_params", num_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # ============================================================
    # 4. BOUCLE D'ENTRAÎNEMENT ET DE TRACKING DU MAX CORR
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
            if args.embed_method == 'pca':
                slp_flat = y_target.view(y_target.size(0), -1).numpy()
                embed_np = pca_model.transform(slp_flat)[:, :latent_dim]
                target_embed = torch.tensor(embed_np, dtype=torch.float32).to(device, non_blocking=True)
            elif args.embed_method == 'vae':
                target_embed, _ = vae_model.encode(y_target.to(device))
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
                        v_X_slp = v_X_slp.to(device, non_blocking=True) if len(args.slp_lags_months) > 0 else None
                        
                        if args.embed_method == 'pca':
                            v_slp_flat = v_y_target.view(v_y_target.size(0), -1).numpy()
                            v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                            v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                        elif args.embed_method == 'vae':
                            v_target_embed, _ = vae_model.encode(v_y_target.to(device))
                        
                        v_pred = model(v_X_sst, v_X_slp)
                        vp = get_median_prediction(v_pred, loss_type, args.quantiles, latent_dim) if loss_type == 'quantile' else v_pred
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

                model.train()  # Retour en mode train

        # --- END OF EPOCH EVALUATION ---
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
                vp = get_median_prediction(v_pred, loss_type, args.quantiles, latent_dim) if loss_type == 'quantile' else v_pred
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
            vp = get_median_prediction(v_pred, loss_type, args.quantiles, latent_dim) if loss_type == 'quantile' else v_pred
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
    parser.add_argument('--optimize_metric', type=str, choices=['R2', 'correlation'], default='correlation', help="Métrique à maximiser via Optuna")
    parser.add_argument('--test_members', type=str, nargs='+', default=['1001.001'])
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner.')
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca', help='Méthode pour l\'espace latent')
    parser.add_argument('--embed_path', type=str, default="/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/IPCA_latent1_NDJF_1members_normalizeTrue_monthly_reduction_wgtFalse_slp_std505.98/best_pca_model.joblib")
    parser.add_argument('--nb_epochs', type=int, default=20, help='Nombre d\'époques pour chaque essai')
    parser.add_argument('--patience', type=int, default=5, help='Nombre d\'époques sans amélioration avant d\'arrêter l\'essai')
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Nombre de points de validation intra-époque')
    parser.add_argument('--bs', type=int, default=64, help='Taille du batch')
    parser.add_argument('--dr', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--early_fusion_sst', action='store_true', help='Si activé, fusionne les données SST tôt dans le réseau')
    parser.add_argument('--lr', type=float, default=5e-5, help='Taux d\'apprentissage')
    parser.add_argument('--n_feat', type=int, default=20, help='Nombre de filtres dans la première couche convolutive')
    parser.add_argument('--roll_sst', action='store_true', help='Si activé, applique un décalage temporel aux données SST')
    parser.add_argument('--sst_lags_months', type=int, nargs='+', default=[2,3,4,5,6], help='Lags pour les données SST en mois')
    parser.add_argument('--slp_lags_months', type=int, nargs='+', default=[], help='Lags pour les données SLP en mois')
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh'], default='relu', help='Fonction d\'activation')
    parser.add_argument('--depth', type=int, default=3, help='Profondeur du réseau CNN')
    parser.add_argument('--filter_mult', type=int, default=1, help='Facteur de multiplication')
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile','correlation'], default='correlation', help='Type de fonction de perte')
    parser.add_argument('--pool_strategy', type=str, choices=['progressive','standart'], default='progressive', help='Stratégie de pooling')
    parser.add_argument('--pool_type', type=str, choices=['avg','max'], default='max', help='Type de pooling')
    parser.add_argument('--sst_kx', type=int, default=3, help='Taille du kernel pour le pooling SST en x')
    parser.add_argument('--sst_ky', type=int, default=5, help='Taille du kernel pour le pooling SST en y')
    parser.add_argument('--sst_pool_x', type=int, default=2, help='Facteur de pooling pour SST en x')
    parser.add_argument('--sst_pool_y', type=int, default=2, help='Facteur de pooling pour SST en y')
    parser.add_argument('--use_gap', action='store_true', help='Utilise Global Average Pooling')
    parser.add_argument('--latent_dim', type=int, default=1, help='Dimension de l\'espace latent target')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Poids de la régularisation L2')
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

    base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna/loocv/"
    
    # Liste complète de tes membres
    ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    test_members = args.test_members
    train_val_members = [m for m in ALL_MEMBERS if m not in test_members]

    dynamic_name = f"LOOCV_metric_{args.optimize_metric}_{args.embed_method}{args.latent_dim}_months{''.join(map(str, args.winter_months))}_epochs{args.nb_epochs}_sst{''.join(map(str, args.sst_lags_months))}slp{''.join(map(str, args.slp_lags_months))}_pat{args.patience}intra{args.nb_intra_evals}bs{args.bs}dr{args.dr}fusion{args.early_fusion_sst}lr{args.lr}feat{args.n_feat}roll{args.roll_sst}act{args.activation}test{''.join(test_members)}loss{args.loss_type}depth{args.depth}mult{args.filter_mult}pool{args.pool_type}{args.pool_strategy}sstpool{args.sst_pool_x}x{args.sst_pool_y}sstkernel{args.sst_kx}x{args.sst_ky}gap{args.use_gap}_decay{args.weight_decay}"
    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    
    study_name = dynamic_name
    db_path = os.path.join(output_dir, "cnn_optuna.db")
    csv_path = os.path.join(output_dir, "cnn_optuna_results.csv")
    storage_name = f"sqlite:///{db_path}"
    
    search_space = {
        "val_member": train_val_members
    }
    
    sampler = GridSampler(search_space)

    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True,
        sampler=sampler
    )
    
    total_trials = len(train_val_members)
    print(f"Début du LOOCV pour maximiser la corrélation ({total_trials} essais au total)...")
    study.optimize(objective, n_trials=total_trials) 
    
    print("\nLOOCV Terminé !")
    
    df = study.trials_dataframe()

    # --- NOUVEAU : Nettoyage du DataFrame ---
    # On cherche toutes les colonnes qui contiennent 'search_space' et on les supprime
    cols_to_drop = [col for col in df.columns if 'search_space' in col]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df.to_csv(csv_path, index=False)
    print(f"Résultats complets sauvegardés dans : {csv_path}")
    
    # Affichage rapide des statistiques de la baseline
    if not df.empty:
        mean_mse = df['user_attrs_best_trial_mse'].mean()
        std_mse = df['user_attrs_best_trial_mse'].std()
        mean_R2 = df['user_attrs_best_R2_score'].mean()
        mean_corr = df['user_attrs_best_trial_corr'].mean()
        print(f"\n=== Bilan LOOCV CNN ({args.optimize_metric.upper()} Optimisé) ===")
        print(f"MSE Moyenne Finale  : {mean_mse:.4f} +/- {std_mse:.4f}")
        print(f"R2 Score Moyen      : {mean_R2:.4f}")
        print(f"Corrélation Moyenne : {mean_corr:.4f}")

        # On garde MSE de validation si tu ne calcules pas MSE de test à la fin,
        # mais on ajoute les métriques TEST
        mean_mse_val = df['user_attrs_best_trial_mse'].mean() 
        std_mse_val = df['user_attrs_best_trial_mse'].std()
        
        mean_test_R2 = df['user_attrs_best_test_R2'].mean()
        mean_test_corr = df['user_attrs_best_test_corr'].mean()
        
        print(f"\n=== Bilan LOOCV CNN ({args.optimize_metric.upper()} Optimisé) ===")
        print(f"MSE Moyenne Finale (Validation) : {mean_mse_val:.4f} +/- {std_mse_val:.4f}")
        print(f"R2 Score Moyen (TEST SET)       : {mean_test_R2:.4f}")
        print(f"Corrélation Moyenne (TEST SET)  : {mean_test_corr:.4f}")