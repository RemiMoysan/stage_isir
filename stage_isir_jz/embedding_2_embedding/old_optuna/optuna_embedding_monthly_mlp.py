import os
import time
import argparse
import joblib
import numpy as np
import random 
import re
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.samplers import TPESampler

# Setup path et imports personnalisés
project_root = Path(__file__).resolve().parent.parent.parent
import sys 
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

from tools.datasets import Dataset_mensuel
from tools.models import compute_loss, get_median_prediction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# ARCHITECTURE NON-LINÉAIRE : MLP RÉSIDUEL (DENSE RESNET)
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, activation=nn.GELU()):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = activation
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.dropout(self.act(self.ln1(self.fc1(x))))
        out = self.dropout(self.ln2(self.fc2(out)))
        return self.act(out + residual)

class NonLinearEmbeddingPredictor(nn.Module):
    def __init__(self, in_features, out_dim, hidden_dim=128, num_blocks=2, dropout=0.1, act_name="gelu"):
        super().__init__()
        
        act_dict = {
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "relu": nn.ReLU()
        }
        activation = act_dict.get(act_name, nn.GELU())

        # Projection initiale vers l'espace caché
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Dropout(dropout)
        )
        
        # Blocs résiduels non linéaires
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout, activation=activation)
            for _ in range(num_blocks)
        ])
        
        # Tête de prédiction finale
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# ============================================================
# OBJECTIF OPTUNA
# ============================================================
def objective(trial):
    start_time = time.time()
    
    # 1. Sélection des Lags
    sst_lags_months = args.sst_lags_months if args.sst_lags_months is not None else [
        m for m in range(2 if not args.include_lag1 else 1, 13)
        if trial.suggest_categorical(f"use_sst_lag_{m}", [True, False])
    ]
    slp_lags_months = args.slp_lags_months if args.slp_lags_months is not None else [
        m for m in range(2 if not args.include_lag1 else 1, 6)
        if trial.suggest_categorical(f"use_slp_lag_{m}", [True, False])
    ]

    if len(sst_lags_months) == 0 and len(slp_lags_months) == 0:
        return -float('inf')

    trial.set_user_attr("sst_lags_final", sst_lags_months)
    trial.set_user_attr("slp_lags_final", slp_lags_months)

    # 2. Hyperparamètres d'Embedding & Modèle Non-Linéaire
    sst_pca_dim = trial.suggest_int("sst_pca_dim", 5, 128)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    num_blocks = trial.suggest_int("num_blocks", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    act_name = trial.suggest_categorical("activation", ["gelu", "silu", "relu"])
    
    # Hyperparamètres d'entraînement
    loss_type = args.loss_type if args.loss_type is not None else trial.suggest_categorical("loss_type", ["mse", "l1", "quantile", "correlation"]) 
    bs = args.bs if args.bs is not None else trial.suggest_categorical("bs", [32, 64, 128])
    lr = args.lr if args.lr is not None else trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-1, log=True) # Remplace L2 manuel

    # 3. Chargement des données
    n_workers = max(0, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1)
    
    common_args = dict(
        selected_months=args.winter_months, machine=args.machine, target_type='map',
        sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months,
        roll_sst=args.roll_sst, slp_std=dynamic_slp_std, sst_std=dynamic_sst_std
    )
    
    training_set = Dataset_mensuel(members=train_members, **common_args)
    val_set = Dataset_mensuel(members=val_members, **common_args)
    test_set = Dataset_mensuel(members=test_members, **common_args)

    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=min(2, n_workers), pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # 4. Dimensions et initialisation du modèle
    in_features_sst = len(sst_lags_months) * sst_pca_dim
    in_features_slp = len(slp_lags_months) * 1 # On conserve PC1 du SLP historique
    out_features = args.latent_dim * len(args.quantiles) if loss_type == 'quantile' else args.latent_dim

    model = NonLinearEmbeddingPredictor(
        in_features=in_features_sst + in_features_slp,
        out_dim=out_features,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        act_name=act_name
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("num_params", num_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Helper de transformation PCA batchée pour alléger la boucle
    def prepare_batch(X_sst, X_slp, y_target):
        # On récupère la taille du batch B sur le premier tenseur non vide
        B = X_sst.shape[0] if X_sst.numel() > 0 else X_slp.shape[0]
        tensors_to_cat = []
        
        # 1. Traitement SST (uniquement si des lags SST ont été sélectionnés)
        if X_sst.shape[1] > 0:
            L_sst = X_sst.shape[1]
            sst_flat = X_sst.view(B * L_sst, -1).numpy()
            sst_pca = sst_pca_model.transform(sst_flat)[:, :sst_pca_dim]
            sst_tensor = torch.tensor(sst_pca.reshape(B, L_sst * sst_pca_dim), dtype=torch.float32, device=device)
            tensors_to_cat.append(sst_tensor)

        # 2. Traitement SLP (uniquement si des lags SLP ont été sélectionnés)
        if X_slp.shape[1] > 0:
            L_slp = X_slp.shape[1]
            slp_flat = X_slp.view(B * L_slp, -1).numpy()
            slp_pca = slp_pca_model.transform(slp_flat)[:, :1]
            slp_tensor = torch.tensor(slp_pca.reshape(B, L_slp * 1), dtype=torch.float32, device=device)
            tensors_to_cat.append(slp_tensor)

        # 3. Sécurité absolue : si les deux sont vides (ne devrait pas arriver avec la vérification amont)
        if not tensors_to_cat:
            raise ValueError("Erreur critique : X_sst et X_slp ont tous les deux 0 lag !")

        # 4. Assemblage : on concatène si on a les deux, sinon on prend l'unique tenseur disponible
        X_combined = torch.cat(tensors_to_cat, dim=1) if len(tensors_to_cat) > 1 else tensors_to_cat[0]

        # 5. Traitement de la Cible
        y_flat = y_target.view(B, -1).numpy()
        target_pca = slp_pca_model.transform(y_flat)[:, :args.latent_dim]
        target_tensor = torch.tensor(target_pca, dtype=torch.float32, device=device)

        return X_combined, target_tensor

    # 5. Boucle d'entraînement
    best_target_metric = -float('inf')
    best_trial_mse = float('inf')
    best_trial_corr = -float('inf')
    best_r2_score = -float('inf')
    history = []
    patience_counter = 0
    best_model_state = None

    total_batches = len(trainloader)
    eval_steps_set = set(np.geomspace(1, max(1, total_batches - 1), num=args.nb_intra_evals, dtype=int)) | {0}
    eval_steps_epoch2_set = set(np.linspace(0, max(1, total_batches - 1), num=args.nb_intra_evals, dtype=int)) | {0}

    for epoch in range(args.nb_epochs):
        model.train()
        for batch_idx, (X_sst, X_slp, y_target, _, _, _) in enumerate(trainloader):
            optimizer.zero_grad()
            X_combined, target_embed = prepare_batch(X_sst, X_slp, y_target)
            
            pred = model(X_combined)
            loss = compute_loss(pred, target_embed, loss_type=loss_type, quantiles=args.quantiles, reduction='mean')
            loss.backward()
            optimizer.step()
        
            # Évaluation intra-époque (Époques 0 et 1)
            if (epoch == 0 and batch_idx in eval_steps_set) or (epoch == 1 and batch_idx in eval_steps_epoch2_set):
                model.eval()
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader_intra:
                        v_X_comb, v_target = prepare_batch(v_X_sst, v_X_slp, v_y_target)
                        v_pred = model(v_X_comb)
                        vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                        all_preds.append(vp)
                        all_targets.append(v_target)

                val_preds_tensor = torch.cat(all_preds, dim=0)
                val_targets_tensor = torch.cat(all_targets, dim=0)

                intra_mse = F.mse_loss(val_preds_tensor, val_targets_tensor).item()
                intra_target_var = torch.var(val_targets_tensor, unbiased=False).item()
                intra_r2 = 1.0 - (intra_mse / intra_target_var) if intra_target_var > 0 else 0.0

                p, t = val_preds_tensor, val_targets_tensor
                cov = ((p - p.mean(0)) * (t - t.mean(0))).mean(0)
                intra_corr = (cov / torch.sqrt(((p - p.mean(0))**2).mean(0) * ((t - t.mean(0))**2).mean(0) + 1e-8)).mean().item()

                best_r2_score = max(best_r2_score, intra_r2)
                best_trial_corr = max(best_trial_corr, intra_corr)
                best_trial_mse = min(best_trial_mse, intra_mse)

                current_metric = intra_r2 if args.optimize_metric == 'r2' else intra_corr
                history.append((epoch + batch_idx / total_batches, intra_r2, intra_corr))

                if current_metric > best_target_metric:
                    best_target_metric = current_metric
                    best_model_state = copy.deepcopy(model.state_dict())
                model.train()

        # Évaluation Fin d'Époque
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for v_X_sst, v_X_slp, v_y_target, _, _, _ in valloader:
                v_X_comb, v_target = prepare_batch(v_X_sst, v_X_slp, v_y_target)
                v_pred = model(v_X_comb)
                vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                all_preds.append(vp)
                all_targets.append(v_target)

        val_preds_tensor = torch.cat(all_preds, dim=0)
        val_targets_tensor = torch.cat(all_targets, dim=0)

        epoch_mse = F.mse_loss(val_preds_tensor, val_targets_tensor).item()
        val_target_variance = torch.var(val_targets_tensor, unbiased=False).item()
        epoch_r2 = 1.0 - (epoch_mse / val_target_variance) if val_target_variance > 0 else 0.0

        p, t = val_preds_tensor, val_targets_tensor
        cov = ((p - p.mean(0)) * (t - t.mean(0))).mean(0)
        epoch_corr = (cov / torch.sqrt(((p - p.mean(0))**2).mean(0) * ((t - t.mean(0))**2).mean(0) + 1e-8)).mean().item()

        best_r2_score = max(best_r2_score, epoch_r2)
        best_trial_corr = max(best_trial_corr, epoch_corr)
        best_trial_mse = min(best_trial_mse, epoch_mse)

        current_metric = epoch_r2 if args.optimize_metric == 'r2' else epoch_corr
        history.append((epoch + 1.0, epoch_r2, epoch_corr))

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

    # 6. Évaluation finale sur le Test Set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for v_X_sst, v_X_slp, v_y_target, _, _, _ in testloader:
            v_X_comb, v_target = prepare_batch(v_X_sst, v_X_slp, v_y_target)
            v_pred = model(v_X_comb)
            vp = get_median_prediction(v_pred, loss_type, args.quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
            all_preds.append(vp)
            all_targets.append(v_target)

    test_preds_tensor = torch.cat(all_preds, dim=0)
    test_targets_tensor = torch.cat(all_targets, dim=0)

    test_mse = F.mse_loss(test_preds_tensor, test_targets_tensor).item()
    test_target_variance = torch.var(test_targets_tensor, unbiased=False).item()
    test_r2 = 1.0 - (test_mse / test_target_variance) if test_target_variance > 0 else 0.0

    p, t = test_preds_tensor, test_targets_tensor
    cov = ((p - p.mean(0)) * (t - t.mean(0))).mean(0)
    test_corr = (cov / torch.sqrt(((p - p.mean(0))**2).mean(0) * ((t - t.mean(0))**2).mean(0) + 1e-8)).mean().item()

    trial.set_user_attr("test_target_variance", test_target_variance) 
    trial.set_user_attr("best_test_mse", test_mse)
    trial.set_user_attr("best_test_r2", test_r2)
    trial.set_user_attr("best_test_corr", test_corr)
    
    print(f"Trial terminé - Score: {best_target_metric:.4f} - Temps: {time.time() - start_time:.2f}s")
    return best_target_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_startup_trials_tpe', type=int, default=10)
    parser.add_argument('--n_startup_trials_pruner', type=int, default=10)
    parser.add_argument('--n_warmup_steps', type=int, default=3)
    parser.add_argument('--interval_steps', type=int, default=1)
    parser.add_argument('--optimize_metric', type=str, choices=['r2', 'correlation'], default='correlation')
    parser.add_argument('--embed_path_slp', type=str, required=True)
    parser.add_argument('--embed_path_sst', type=str, required=True)
    parser.add_argument('--machine', type=str, default='jean-zay-work')
    parser.add_argument('--nb_members_val', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--include_lag1', action='store_true')
    parser.add_argument('--nb_intra_evals', type=int, default=15)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'correlation', 'quantile'], default=None)
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=None)
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=None)
    args = parser.parse_args()

    # Routage dynamique
    if args.machine == 'hacienda': base_home = "/home/moysan/stage_isir_jz/embedding_2_embedding/optuna/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']: base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/embedding_2_embedding/optuna/"
    elif args.machine == 'mac_local': base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/embedding_2_embedding/optuna/"
    else: base_home = "./optuna_output/"

    dynamic_slp_std = 596.0 
    match = re.search(r'slp_std([0-9.]+)', args.embed_path_slp)
    if match: dynamic_slp_std = float(match.group(1))

    dynamic_sst_std = 0.707
    match = re.search(r'sst_std([0-9.]+)', args.embed_path_sst)
    if match: dynamic_sst_std = float(match.group(1))

    slp_pca_model = joblib.load(args.embed_path_slp) 
    sst_pca_model = joblib.load(args.embed_path_sst)

    ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

    rng = random.Random(args.seed)
    members_shuffled = ALL_MEMBERS.copy()
    rng.shuffle(members_shuffled)
    train_members = members_shuffled[:-2*args.nb_members_val]
    val_members = members_shuffled[-2*args.nb_members_val:-args.nb_members_val]
    test_members = members_shuffled[-args.nb_members_val:] 

    # 1. Base du nom avec les paramètres structurels toujours présents
    base_name = f"Optuna_ResNet_{args.optimize_metric}_m{''.join(map(str, args.winter_months))}_ep{args.nb_epochs}_val{args.nb_members_val}_roll{args.roll_sst}_lag1{args.include_lag1}"
    
    # 2. Dictionnaire de raccourcis pour tous les hyperparamètres optionnels qu'Optuna peut chercher
    # Cela évite d'avoir des noms de dossiers de 300 caractères si tout est renseigné
    short = {
        'bs': 'bs', 
        'lr': 'lr', 
        'loss_type': 'loss', 
        'sst_lags_months': 'sstLags', 
        'slp_lags_months': 'slpLags'
    }

    # 3. Extraction dynamique des paramètres fixés par l'utilisateur (ceux qui ne sont pas None dans le dictionnaire 'short')
    # On formate joliment : si c'est une liste on accole les chiffres (ex: [1,2,3] -> "123"), si c'est un float on met en notation scientifique si très petit
    fixed_params = []
    for k, v in sorted(vars(args).items()):
        if k in short and v is not None:
            if isinstance(v, list):
                # Pour les lags: [1, 2, 3] devient "123". Si la liste est vide [] (fixée vide par l'user), ça devient "None" ou "0"
                val_str = ''.join(map(str, v)) if len(v) > 0 else 'empty'
            elif isinstance(v, float) and v < 1e-3:
                val_str = f"{v:.1e}"
            else:
                val_str = str(v)
            
            fixed_params.append(f"{short[k]}{val_str}")

    # 4. Construction du nom dynamique final
    if fixed_params:
        dynamic_name = f"{base_name}_FIXED_{'_'.join(fixed_params)}"
    else:
        dynamic_name = f"{base_name}_full_search"

    # Ajout des informations sur la configuration du pruner/sampler Optuna
    study_name = f"{dynamic_name}_optuna_s{args.n_startup_trials_tpe}_p{args.n_startup_trials_pruner}"
    
    output_dir = os.path.join(base_home, study_name)
    os.makedirs(output_dir, exist_ok=True)
    
    storage_name = f"sqlite:///{os.path.join(output_dir, 'optuna.db')}"

    sampler = TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials_tpe)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials_pruner,
        n_warmup_steps=args.n_warmup_steps,
        interval_steps=args.interval_steps
    )

    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True, sampler=sampler, pruner=pruner)
    print(f"Début de l'optimisation Optuna ({args.n_trials} essais)...")
    study.optimize(objective, n_trials=args.n_trials) 
    
    print("\nOptimisation Terminée !")
    print(f"Meilleur score ({args.optimize_metric}) : {study.best_trial.value:.4f}")
    study.trials_dataframe().to_csv(os.path.join(output_dir, "optuna_results.csv"), index=False)