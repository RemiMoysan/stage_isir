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

#TO DO: ajouter lat_weight

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
    # 1. DÉFINITION DES HYPERPARAMÈTRES À TUNER
    # ============================================================
    nb_members_train = trial.suggest_int("nb_members_train", 10, 60)
    if args.fixed_seed is None:
        seed = trial.suggest_int("seed", 0, 1000)
    else: 
        seed = args.fixed_seed  
    bs = trial.suggest_categorical("bs", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    dr = trial.suggest_float("dr", 0.0, 0.6)

    pool_type = trial.suggest_categorical("pool_type", ['max', 'avg'])
    sst_pool_y = trial.suggest_categorical("sst_pool_y", [2, 3])
    sst_pool_x = trial.suggest_categorical("sst_pool_x", [2, 3])
    pool_strategy = trial.suggest_categorical("pool_strategy", ['progressive', 'standard'])
    activation = trial.suggest_categorical("activation", ['tanh', 'relu'])
    use_gap = trial.suggest_categorical("use_gap", [True, False])
    sst_ky = trial.suggest_categorical("sst_ky", [3, 5])
    sst_kx = trial.suggest_categorical("sst_kx", [3, 5])
    depth = trial.suggest_int("depth", 2, 4)
    filter_mult = trial.suggest_categorical("filter_mult", [1, 2])

    loss_type = trial.suggest_categorical("loss_type", ['mse', 'l1','correlation','quantile'])
    # on s'attend à ce que l1 et quantile soit plus ou moins équivalents
    # pour quantile on part sur les 10 quantiles de 0.1 à 0.9 
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    latent_dim = 1
    roll_sst = trial.suggest_categorical("roll_sst", [True, False])
    early_fusion_sst = trial.suggest_categorical("early_fusion_sst", [True, False])
    n_feat = trial.suggest_int("n_feat", 4, 32)
    
    sst_lags_months = []
    # On teste tous les mois de 2 à 12 (1 an d'antériorité max)
    first_month = 2 if not args.include_lag1 else 1
    for month in range(first_month, 13):
        # Optuna choisit True ou False pour chaque mois
        if trial.suggest_categorical(f"use_sst_lag_{month}", [True, False]):
            sst_lags_months.append(month)

    if len(sst_lags_months) == 0:
        return -float('inf')  # On rejette les essais sans lags SST
    slp_lags_months = [] # Forcé à vide comme demandé
    
    trial.set_user_attr("sst_lags_final", sst_lags_months)
    corr_history = [] # Liste pour stocker l'évolution de la corrélation intra-époque
    trial.set_user_attr("corr_history", corr_history)

    # ============================================================
    # 2. PRÉPARATION DES DONNÉES (Géré dans le trial car le seed change)
    # ============================================================
    rng = random.Random(seed)
    shuffled_members = ALL_MEMBERS.copy()
    rng.shuffle(shuffled_members)
    
    train_members = shuffled_members[:nb_members_train]
    val_members = shuffled_members[-args.nb_members_val:] 
    winter_months = args.winter_months
    
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 2))
    n_workers = max(0, n_workers - 1)

    train_set = Dataset_mensuel(members=train_members, selected_months=winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=roll_sst, slp_std=dynamic_slp_std)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    val_set = Dataset_mensuel(members=val_members, selected_months=winter_months, machine='jean-zay-work', target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=roll_sst, slp_std=dynamic_slp_std)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ============================================================
    # 4. BOUCLE D'ENTRAÎNEMENT ET DE TRACKING DU MAX CORR
    # ============================================================
    nb_epochs = args.nb_epochs # Suffisant pour voir le modèle overfitter ou converger
    best_trial_corr = -float('inf')
    patience_counter = 0

    
    # Calcul des steps intra-epoch pour la première époque
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
            optimizer.step()

            # --- INTRA-EPOCH EVALUATION (Seulement Epoch 0 et 1 pour la vitesse) ---
            if (epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set):
                model.eval()
                v_sum_p, v_sum_t, v_sum_p2, v_sum_t2, v_sum_pt, intra_n_samples = 0.0, 0.0, 0.0, 0.0, 0.0, 0
                
                with torch.no_grad():
                    for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                        v_X_sst = v_X_sst.to(device, non_blocking=True)
                        v_X_slp = v_X_slp.to(device, non_blocking=True) if len(slp_lags_months) > 0 else None
                        
                        v_slp_flat = v_y_target.view(v_y_target.size(0), -1).numpy()
                        v_embed_np = pca_model.transform(v_slp_flat)[:, :latent_dim]
                        v_target_embed = torch.tensor(v_embed_np, dtype=torch.float32).to(device, non_blocking=True)
                        
                        v_pred = model(v_X_sst, v_X_slp)
                        
                        # Calcul Corrélation
                        p = get_median_prediction(v_pred, loss_type, quantiles, latent_dim) if loss_type == 'quantile' else v_pred
                        t = v_target_embed

                        p, t = p.detach(), t.detach() # redondant avec .no_grad
                        v_sum_p += p.sum(dim=0)
                        v_sum_t += t.sum(dim=0)
                        v_sum_p2 += (p ** 2).sum(dim=0)
                        v_sum_t2 += (t ** 2).sum(dim=0)
                        v_sum_pt += (p * t).sum(dim=0)
                        intra_n_samples += p.size(0)

                v_mean_p, v_mean_t = v_sum_p / intra_n_samples, v_sum_t / intra_n_samples
                v_var_p, v_var_t = (v_sum_p2 / intra_n_samples) - v_mean_p**2, (v_sum_t2 / intra_n_samples) - v_mean_t**2
                v_cov_pt = (v_sum_pt / intra_n_samples) - (v_mean_p * v_mean_t)
                v_corr = (v_cov_pt / torch.sqrt(v_var_p * v_var_t + 1e-8)).mean().item()
                
                corr_history.append((epoch + batch_idx / total_batches, v_corr))
                # Mise à jour du meilleur pic de corrélation
                if v_corr > best_trial_corr:
                    best_trial_corr = v_corr

                model.train() # Retour en mode train

        # --- END OF EPOCH EVALUATION ---
        model.eval()
        sum_p, sum_t, sum_p2, sum_t2, sum_pt, total_val_samples = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        
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
                sum_p += p.sum(dim=0)
                sum_t += t.sum(dim=0)
                sum_p2 += (p ** 2).sum(dim=0)
                sum_t2 += (t ** 2).sum(dim=0)
                sum_pt += (p * t).sum(dim=0)
                total_val_samples += p.size(0)

        mean_p, mean_t = sum_p / total_val_samples, sum_t / total_val_samples
        var_p, var_t = (sum_p2 / total_val_samples) - mean_p**2, (sum_t2 / total_val_samples) - mean_t**2
        cov_pt = (sum_pt / total_val_samples) - (mean_p * mean_t)
        epoch_val_corr = (cov_pt / torch.sqrt(var_p * var_t + 1e-8)).mean().item()

        corr_history.append((epoch, epoch_val_corr))
        # Mise à jour de la meilleure corrélation globale
        if epoch_val_corr > best_trial_corr:
            best_trial_corr = epoch_val_corr
            patience_counter = 0
        else:
            patience_counter += 1

        # 3. L'Early Stopping Local 
        if patience_counter >= args.patience:
            print(f"Early stopping déclenché à l'époque {epoch}. Le modèle overfit.")
            break # On sort de la boucle d'époque, l'essai se termine proprement
            

        # PRUNING OPTUNA basé sur la corrélation de fin d'époque
        # trial.report(epoch_val_corr, epoch)
        # On reporte au pruner le PIC absolu de cet essai, pas sa valeur actuelle
        trial.report(best_trial_corr, epoch)
        
        if trial.should_prune():
            trial.set_user_attr("corr_history", corr_history)
            raise optuna.exceptions.TrialPruned()

    # On retourne la valeur MAXIMALE atteinte pendant tout l'essai
    trial.set_user_attr("corr_history", corr_history)
    print(f"Trial terminé, temps écoulé depuis le début de l'optimisation: {time.time() - start_time:.2f} secondes")
    return best_trial_corr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100, help='Nombre de combinaisons à tester')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner. Il vaut mieux en choisir un seul pour interpréter la corrélation')
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='pca', help='Méthode pour l\'espace latent')
    parser.add_argument('--embed_path', type=str, default="/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/IPCA_latent1_NDJF_1members_normalizeTrue_monthly_reduction_wgtFalse_slp_std505.98/best_pca_model.joblib", help='Chemin vers le modèle PCA pour l\'embedding SLP, pour l\'instant normalize et pas de weight lat')
    parser.add_argument('--nb_epochs', type=int, default=20, help='Nombre d\'époques pour chaque essai')
    parser.add_argument('--patience', type=int, default=5, help='Nombre d\'époques sans amélioration avant d\'arrêter l\'essai')
    parser.add_argument('--nb_members_val', type=int, default=5, help='Nombre de membres à utiliser pour la validation (fixé à 5 pour comparer équitablement les essais)')
    parser.add_argument('--n_startup_trials_tpe', type=int, default=10, help='Nombre d\'essais à exécuter avant d\'activer le pruner')
    parser.add_argument('--n_startup_trials_pruner', type=int, default=10, help='Nombre d\'essais à exécuter avant d\'activer le pruner')
    parser.add_argument('--n_warmup_steps', type=int, default=3, help='Nombre d\'époques à attendre avant de commencer à évaluer pour le pruner')
    parser.add_argument('--interval_steps', type=int, default=1, help='Intervalle d\'évaluation pour le pruner (en nombre d\'époques)')
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Nombre de points de validation intra-époque (espacement logarithmique epoch 1, espacement liénaire epoch 2)')
    parser.add_argument('--include_lag1', action='store_true', help='Inclure le lag 1 pour la target SST')
    parser.add_argument('--fixed_seed', type=int, default=None, help='Seed fixe pour le shuffle')
    args = parser.parse_args()

    dynamic_slp_std = 596.0  # Valeur de repli (fallback) par sécurité
    if args.embed_path:
        # On cherche le motif "slp_std" suivi de chiffres et d'un point
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))
            print(f"\n✅ slp_std extrait avec succès du chemin PCA : {dynamic_slp_std}")
        else:
            print(f"\n⚠️ 'slp_std' introuvable dans le nom du dossier. Utilisation du fallback : {dynamic_slp_std}")
    else:
        print(f"\n⚠️ Aucun modèle pré-entraîné fourni. Utilisation du slp_std par défaut : {dynamic_slp_std}")

    base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna/"
    
    dynamic_name = f"monthly_{args.embed_method}_correlation_max_months{''.join(map(str, args.winter_months))}_nb_epochs{args.nb_epochs}_intra_evals{args.nb_intra_evals}_patience{args.patience}_nb_members_val{args.nb_members_val}_sampler_pruner{args.n_startup_trials_tpe}_{args.n_startup_trials_pruner}_{args.n_warmup_steps}_{args.interval_steps}_fixed_seed{args.fixed_seed}_include_lag1{args.include_lag1}"
    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    study_name = dynamic_name
    # 3. On construit les chemins physiques pour les fichiers
    db_path = os.path.join(output_dir, "cnn_optuna.db")
    csv_path = os.path.join(output_dir, "cnn_optuna_results.csv")
    storage_name = f"sqlite:///{db_path}"
    
    
    # Attention au direction="maximize" car on cherche la meilleure corrélation
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials_pruner, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps)
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.n_startup_trials_tpe, seed=42) # Le seed ici fige l'exploration initiale (=/= fixed_seed qui fige le shuffle des membres)

    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    print(f"Début de l'optimisation pour maximiser la corrélation ({args.n_trials} trials)...")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Max Correlation reached: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    print(f"  Final SST Lags (Months): {trial.user_attrs.get('sst_lags_final')}")

    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats complets sauvegardés dans : {csv_path}")