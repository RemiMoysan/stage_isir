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
import xarray as xr
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
from shared_tools.optuna_plots import generate_crossval_matrix, generate_1d_loocv_heatmap 

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
print(f"Using device: {device}")

ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

# ============================================================
# CONFIGURATION DE LA GRILLE LOOCV (Slicing Indépendant)
# ============================================================
# Exemple pour une ligne 1D : 1 membre en validation, TOUS les membres en test
VAL_GRID_MEMBERS = ALL_MEMBERS[30:31]  
TEST_GRID_MEMBERS = ALL_MEMBERS[:]

def objective(trial):
    # 1. Sélection croisée par la Grid Optuna
    val_member = trial.suggest_categorical("val_member", ALL_MEMBERS)
    test_member = trial.suggest_categorical("test_member", ALL_MEMBERS)
    val_members = [val_member]
    test_members = [test_member]
        
    train_members = [m for m in ALL_MEMBERS if m != val_member and m != test_member]

    # 2. Hyperparamètres FIXES
    bs = args.bs
    lr = args.lr
    dr_conv = args.dr_conv
    dr_fc = args.dr_fc
    fc_dim = args.fc_dim
    depth = args.depth
    n_feat = args.n_feat
    filter_mult = args.filter_mult
    pool_type = args.pool_type
    sst_pool_x = args.sst_pool_x
    sst_pool_y = args.sst_pool_y
    sst_kx = args.sst_kx
    sst_ky = args.sst_ky
    activation = args.activation
    pool_strategy = args.pool_strategy
    use_gap = args.use_gap
    early_fusion_sst = args.early_fusion_sst
    loss_type = args.loss_type
    weight_decay = args.weight_decay
    noise_std = args.noise_std
    grad_clip = args.gradient_clip
    sst_lags_months = args.sst_lags_months
    slp_lags_months = args.slp_lags_months
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # ============================================================
    # 2. PRÉPARATION DES DONNÉES
    # ============================================================
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
    out_feature = len(quantiles) * args.latent_dim if loss_type == 'quantile' else args.latent_dim
    model = CNN_Latent_SLP_Multimodal1_tunable(
        dr_conv=dr_conv, dr_fc=dr_fc, fc_dim=fc_dim, nb_out=out_feature, in_chans_sst=len(sst_lags_months), in_chans_slp=len(slp_lags_months), 
        n_feat=n_feat, early_fusion_sst=early_fusion_sst, depth=depth, filter_mult=filter_mult,
        sst_kx=sst_kx, sst_ky=sst_ky, sst_pool_x=sst_pool_x, sst_pool_y=sst_pool_y,
        pool_type=pool_type, pool_strategy=pool_strategy, activation=activation, use_gap=use_gap
    ).to(device)

    # INITIALISATION LAZY 
    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(sst_lags_months), 85, 360).to(device) if len(sst_lags_months) > 0 else None
        dummy_slp = torch.zeros(1, len(slp_lags_months), 53, 113).to(device) if len(slp_lags_months) > 0 else None
        _ = model(dummy_sst, dummy_slp)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW([
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ], lr=lr)

    # ============================================================
    # 4. BOUCLE D'ENTRAÎNEMENT ET TRACKING DES MÉTRIQUES SPATIALES
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
            
            # Encodage pour le calcul de la loss
            target_embed = encode_to_latent_gpu(y_target.to(device, non_blocking=True), args.embed_method, args.latent_dim, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
            
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
                        vp = get_median_prediction(v_pred, loss_type, quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                        
                        all_preds_latent.append(vp)
                        all_true_maps.append(v_y_target)

                val_preds_latent = torch.cat(all_preds_latent, dim=0)
                val_true_maps = torch.cat(all_true_maps, dim=0).to(device, non_blocking=True)

                val_pred_maps = decode_to_spatial_map_gpu(val_preds_latent, args.embed_method, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
                i_r2, i_l1, i_corr = compute_targeted_spatial_metrics(val_pred_maps, val_true_maps, wgts_gpu)

                current_step = epoch + (batch_idx / total_batches)
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
                vp = get_median_prediction(v_pred, loss_type, quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
                
                all_preds_latent.append(vp)
                all_true_maps.append(v_y_target)

        val_preds_latent = torch.cat(all_preds_latent, dim=0)
        val_true_maps = torch.cat(all_true_maps, dim=0).to(device, non_blocking=True)

        val_pred_maps = decode_to_spatial_map_gpu(val_preds_latent, args.embed_method, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
        e_r2, e_l1, e_corr = compute_targeted_spatial_metrics(val_pred_maps, val_true_maps, wgts_gpu)

        metrics_history.append((epoch + 1, e_r2, e_l1, e_corr))

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
        
        if patience_counter >= args.patience:
            break
    
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
            vp = get_median_prediction(v_pred, loss_type, quantiles, args.latent_dim) if loss_type == 'quantile' else v_pred
            
            all_preds_latent.append(vp)
            all_true_maps.append(v_y_target)

    test_preds_latent = torch.cat(all_preds_latent, dim=0)
    test_true_maps = torch.cat(all_true_maps, dim=0).to(device, non_blocking=True)

    test_pred_maps = decode_to_spatial_map_gpu(test_preds_latent, args.embed_method, pca_components_gpu, pca_mean_gpu, wgts_gpu, vae_model)
    t_r2, t_l1, t_corr = compute_targeted_spatial_metrics(test_pred_maps, test_true_maps, wgts_gpu)

    trial.set_user_attr("best_test_R2", t_r2)
    trial.set_user_attr("best_test_L1", t_l1)
    trial.set_user_attr("best_test_corr", t_corr)
    trial.set_user_attr("R2_L1_corr_history", metrics_history)

    del model, optimizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"Trial completed: val_member={val_member}, test_member={test_member}, best_test_R2={t_r2:.4f}, best_test_L1={t_l1:.4f}, best_test_corr={t_corr:.4f}. Time elapsed: {time.time() - start_time:.2f} seconds.")
    return best_target_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize_metric', type=str, choices=['R2', 'L1', 'correlation'], default='correlation')
    parser.add_argument('--lat_weight', action='store_true')
    parser.add_argument('--nb_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--nb_intra_evals', type=int, default=5)
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--embed_method', type=str, default='pca')
    parser.add_argument('--embed_path', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--include_lag1', action='store_true')
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--dr_conv', type=float, required=True)
    parser.add_argument('--dr_fc', type=float, required=True)
    parser.add_argument('--fc_dim', type=int, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--n_feat', type=int, required=True)
    parser.add_argument('--filter_mult', type=float, required=True)
    parser.add_argument('--pool_type', type=str, choices=['max', 'avg'], required=True)
    parser.add_argument('--sst_pool_x', type=int, required=True)
    parser.add_argument('--sst_pool_y', type=int, required=True)
    parser.add_argument('--sst_kx', type=int, required=True)
    parser.add_argument('--sst_ky', type=int, required=True)
    parser.add_argument('--activation', type=str, choices=['tanh', 'relu'], required=True)
    parser.add_argument('--pool_strategy', type=str, choices=['progressive', 'standard'], required=True)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1','correlation','quantile'], required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--noise_std', type=float, required=True)
    parser.add_argument('--gradient_clip', type=float, required=True)
    parser.add_argument('--use_gap', action='store_true')
    parser.add_argument('--early_fusion_sst', action='store_true')
    parser.add_argument('--sst_lags_months', type=int, nargs='*', required=True)
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[])
    args = parser.parse_args()

    # ============================================================
    # 1. RÉCUPÉRATION DU SLP_STD 
    # ============================================================
    dynamic_slp_std = 466.93 # pour le vae ce n'est pas dans le nom du fichier donc je force ici pour retomber sur la bonne valeur 
    if args.embed_path:
        match = re.search(r'slp_std([0-9.]+)', args.embed_path)
        if match:
            dynamic_slp_std = float(match.group(1))
    print(f"slp std: {dynamic_slp_std}")
    
    # ============================================================
    # 2. PRÉPARATION DU MODÈLE D'EMBEDDING (HORS BOUCLE)
    # ============================================================
    pca_model, vae_model = None, None
    pca_mean_gpu, pca_components_gpu = None, None
    wgts_gpu = None

    if args.embed_method == 'pca':
        pca_model = joblib.load(args.embed_path)
        pca_mean_gpu = torch.tensor(pca_model.mean_, dtype=torch.float32, device=device)
        pca_components_gpu = torch.tensor(pca_model.components_[:args.latent_dim], dtype=torch.float32, device=device)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=args.latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

    if args.lat_weight:
        sample_member = ALL_MEMBERS[0]
        sample_path = f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-{sample_member}_1mo.nc"
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            h, w = len(lats), len(ds_sample['lon'].values)
            wgts = np.sqrt(coslat).reshape(h, 1)
            wgts_gpu = torch.tensor(np.broadcast_to(wgts, (h, w)).flatten(), dtype=torch.float32, device=device)
            ds_sample.close()
        except Exception as e:
            print(f"⚠️ Erreur chargement poids latitude : {e}")

    # ============================================================
    # 3. CRÉATION DU DOSSIER DYNAMIQUE ET CHEMINS
    # ============================================================
    base_name = f"LOOCV_SPATIAL_{args.embed_method}{args.latent_dim}_{args.optimize_metric}_m{''.join(map(str, args.winter_months))}"
    
    short = {
        'bs': 'bs', 'lr': 'lr', 'dr_conv': 'dr1', 'dr_fc': 'dr2', 'fc_dim': 'fc', 
        'depth': 'dp', 'n_feat': 'feat', 'filter_mult': 'mult', 'pool_type': 'pool', 'sst_pool_x': 'x','sst_pool_y': 'y', 'sst_kx': 'kx', 'sst_ky': 'ky','pool_strategy': 'strat','early_fusion_sst': 'fus','loss_type': 'loss', 'weight_decay': 'wd',
        'noise_std': 'noise', 'gradient_clip':'grad','sst_lags_months': 'sstlags'
    }
    
    fixed = []
    for k, v in sorted(vars(args).items()):
        if k in short and v is not None:
            if isinstance(v, list):
                val_str = ''.join(map(str, v))
            elif isinstance(v, float):
                # Arrondi les floats à 3 décimales pour raccourcir le nom du fichier
                val_str = f"{v:.1e}" if v < 1e-3 else f"{v:.3f}"
            else:
                val_str = str(v)
            fixed.append(f"{short[k]}{val_str}")

    dynamic_name = f"{base_name}_{'_'.join(fixed)}"
    
    base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_spatial/loocv_spatial/"
    output_dir = os.path.join(base_home, dynamic_name)
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, f"{dynamic_name}.db")
    storage_name = f"sqlite:///{db_path}"

    # ============================================================
    # 4. LANCEMENT DE LA GRID SEARCH OPTUNA
    # ============================================================
    search_space = {
        "val_member": VAL_GRID_MEMBERS,
        "test_member": TEST_GRID_MEMBERS
    }
    
    sampler = optuna.samplers.GridSampler(search_space)
    
    study = optuna.create_study(
        study_name=dynamic_name, 
        storage=storage_name, 
        load_if_exists=True,
        direction="maximize", 
        sampler=sampler
    )
    
    print(f"\n🚀 Début du LOOCV GridSearch ({len(VAL_GRID_MEMBERS)}x{len(TEST_GRID_MEMBERS)} = {len(VAL_GRID_MEMBERS) * len(TEST_GRID_MEMBERS)} paires)...")
    print(f"📁 Dossier de sortie : {output_dir}")
    
    study.optimize(objective)
    
    # ============================================================
    # 5. GÉNÉRATION DES MATRICES FINALES
    # ============================================================
    for best_test_metric in ['best_test_R2', 'best_test_L1', 'best_test_corr']:
        generate_crossval_matrix(study, output_dir, best_test_metric)
        if len(VAL_GRID_MEMBERS) == 1:
            generate_1d_loocv_heatmap(study, output_dir, best_test_metric, VAL_GRID_MEMBERS[0])