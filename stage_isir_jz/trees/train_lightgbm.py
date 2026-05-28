import torch
import numpy as np
import os
import argparse
import time
import random
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict
import joblib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy.ndimage as ndimage

import sys

# 1 epoch suffit pour Boosting trees...

project_root = Path(__file__).resolve().parent.parent

# Ajouter le dossier "tools" de vision_transformer au sys.path pour les imports
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

from tools.visualizations import plot_confusion_matrix
from tools.models import get_fast_labels
from tools.datasets import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    
    parser.add_argument('--nb_members_train', type=int, default=10, help='Nombre de membres à utiliser pour l\'entraînement')
    parser.add_argument('--nb_members_val', type=int, default=5, help='Nombre de membres à utiliser pour la validation')
    parser.add_argument('--seed', type=int, default=42, help='Seed pour le mélange inter membres')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours')
    parser.add_argument('--bs', type=int, default=256, help='Taille de batch pour le chargement des chunks')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate pour l\'entraînement de LightGBM')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95], help='Liste des lags pour SST')
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[], help='Liste des lags pour SLP')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner')

    parser.add_argument('--metric', type=str, default='mse', choices=['mse', 'correlation','pc1_quantiles','mse_latent'], help='Métrique pour le calcul des labels')    
    parser.add_argument('--master_ref_path', type=str, default="/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/composites_4_regimes/master_ref_generator_89_members_10d_embedding_method_pca/master_reference_global.npz", help='Chemin vers la référence maître')
    parser.add_argument('--projector_path', type=str, default="", help='Chemin vers le projector à utiliser (optionnel)')

    parser.add_argument('--smooth_sigma', type=float, default=0.0, help='Intensité du lissage Gaussien de l\'explicabilité (0.0 = brut, 1.5 = recommandé si lissage)')
    args = parser.parse_args()

    # --- ROUTAGE DES DOSSIERS ---
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/trees/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']:
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/trees/" 
    elif args.machine == "mac_local":
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/trees/"
    else:
        raise ValueError("Machine argument must be 'hacienda', 'jean-zay-work', 'jean-zay-scratch' or 'mac_local'.")

    # --- PARAMÈTRES & VARIABLES ---
    sst_lags_days = args.sst_lags_days
    slp_lags_days = args.slp_lags_days
    bs = args.bs
    lr = args.lr
    winter_months = args.winter_months
    duree_lissage = args.duree_lissage
    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val
    metric = args.metric
    smooth_sigma = args.smooth_sigma

    print("Arg Parameters:")
    print(f"  Metric: {metric}", f" SST Lags: {sst_lags_days}", f" SLP Lags: {slp_lags_days}", f" Batch Size (Chunking): {bs}", f" LR: {lr}", f" Months: {winter_months}", f" Smoothing: {duree_lissage}", f" Train Members: {nb_members_train}", f" Val Members: {nb_members_val}\n")

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[-nb_members_val:]

    outdir_name = f"LightGBM_classifier_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_lr{lr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_members_seed_{args.seed}_{duree_lissage}d_metric_{metric}"
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)
    print(f"Dossier de sauvegarde : {outdir}")

    # --- PRÉPARATION LABELS & CLASSES ---
    master_ref = dict(np.load(args.master_ref_path)) 
    print("Référence maître chargée !")

    if args.projector_path and os.path.exists(args.projector_path):
        projector = joblib.load(args.projector_path)
        print("Projector chargé !")
    else:
        projector = None

    # Déduction dynamique du nombre de classes
    if args.metric == 'pc1_quantiles':
        if 'pc1_bins' not in master_ref:
            raise ValueError(f"ERREUR : Le fichier {args.master_ref_path} ne contient pas 'pc1_bins'. Vérifiez le chemin !")
        num_classes = len(master_ref['pc1_bins']) - 1
    elif args.metric in ['correlation', 'mse']:
        regime_keys = [k for k in master_ref.keys() if k.endswith("_slp_0_mean") and not k.startswith("GLOBAL")]
        num_classes = len(regime_keys)
    elif args.metric == 'mse_latent':
        num_classes = master_ref['ref_centroids_latent'].shape[0]
    else:
        num_classes = 4

    print(f"--> Détection automatique : {num_classes} classes pour la métrique '{args.metric}'")


    # --- CHARGEMENT DATASETS PYTORCH ---
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")

    # shuffle=True pour l'apprentissage incrémental de lgbm
    training_set = Dataset(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage,roll_sst=True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers)

    val_set = Dataset(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage,roll_sst=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers)

    # --- 1. ENTRAÎNEMENT LAZY LOADING ---
    print("\n--- Début de l'entraînement LightGBM Incrémental (Lazy Loading) ---")
    
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'learning_rate': lr,
        'max_depth': 6,
        'colsample_bytree': 0.2, 
        'subsample': 0.8,
        'metric': 'multi_logloss',
        'random_state': args.seed,
        'n_jobs': -1,
        'verbosity': -1
    }

    booster = None 
    chunk_X, chunk_y = [], []
    chunk_size_target = 2048 
    arbres_par_chunk = 5     
    
    start_train = time.time()
    model_has_trained = False

    for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
        
        labels = get_fast_labels(y_target.numpy(), master_ref, metric=metric, projector=projector)
        B = X_sst.shape[0]
        
        x_sst_flat = X_sst.view(B, -1).numpy()
        x_slp_flat = X_slp.view(B, -1).numpy()
        x_flat = np.concatenate([x_sst_flat, x_slp_flat], axis=1)
        
        chunk_X.append(x_flat)
        chunk_y.append(labels.numpy())
        
        current_chunk_size = sum(len(x) for x in chunk_X)
        
        if current_chunk_size >= chunk_size_target or (batch_idx == len(trainloader) - 1):
            X_train_chunk = np.vstack(chunk_X)
            y_train_chunk = np.concatenate(chunk_y)
            
            lgb_train = lgb.Dataset(X_train_chunk, label=y_train_chunk, free_raw_data=True)
            
            booster = lgb.train(
                params,
                lgb_train,
                num_boost_round=arbres_par_chunk,
                init_model=booster,
                keep_training_booster=True
            )
            model_has_trained = True
            
            print(f"Train Batch {batch_idx+1}/{len(trainloader)} traité | Arbres totaux: {booster.current_iteration()}", end='\r')
            
            chunk_X, chunk_y = [], []
            del X_train_chunk, y_train_chunk, lgb_train

    print(f"\nEntraînement terminé en {time.time() - start_train:.2f}s")


    # --- 2. VALIDATION LAZY LOADING ---
    print("\n--- Début de la Validation ---")
    
    per_member_metrics = defaultdict(lambda: {'count': 0, 'preds': [], 'labels': []})
    all_preds = []
    all_labels = []
    
    start_val = time.time()
    
    if model_has_trained:
        with torch.no_grad():
            for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(valloader):
                
                labels = get_fast_labels(y_target.numpy(), master_ref, metric=metric, projector=projector)
                B = X_sst.shape[0]
                
                x_sst_flat = X_sst.view(B, -1).numpy()
                x_slp_flat = X_slp.view(B, -1).numpy()
                x_val_batch = np.concatenate([x_sst_flat, x_slp_flat], axis=1)
                
                y_pred_prob = booster.predict(x_val_batch)
                y_pred_batch = np.argmax(y_pred_prob, axis=1)
                labels_np = labels.numpy()
                
                members_list = [m if isinstance(m, str) else m.item().decode() if hasattr(m, 'item') else str(m) for m in members]
                
                for i, mem in enumerate(members_list):
                    per_member_metrics[mem]['count'] += 1
                    per_member_metrics[mem]['preds'].append(y_pred_batch[i])
                    per_member_metrics[mem]['labels'].append(labels_np[i])

                all_preds.extend(y_pred_batch)
                all_labels.extend(labels_np)
                
                print(f"Val Batch {batch_idx+1}/{len(valloader)} traité...", end='\r')
        
        print(f"\nValidation terminée en {time.time() - start_val:.2f}s")
        
        # --- CALCUL DES METRIQUES & MATRICES ---
        val_acc_globale = accuracy_score(all_labels, all_preds) * 100
        print(f"\nAccuracy Finale Globale : {val_acc_globale:.2f}%")
        plot_confusion_matrix(all_labels, all_preds, outdir, master_ref, filename='lgbm_confusion_matrix_global.png')
        
        for mem, d in per_member_metrics.items():
            if d['count'] > 0:
                mem_acc = accuracy_score(d['labels'], d['preds']) * 100
                member_outdir = os.path.join(outdir, "per_member", mem)
                os.makedirs(member_outdir, exist_ok=True)
                plot_confusion_matrix(d['labels'], d['preds'], member_outdir, master_ref, filename='lgbm_confusion_matrix.png')

        booster.save_model(os.path.join(outdir, 'lightgbm_lazy_model.txt'))
        
        # --- EXPLICABILITÉ (Dynamique : Lissage ou Brut) ---
        print("\n--- Génération des cartes d'explicabilité avec Coastlines ---")

        
        explain_dir = os.path.join(outdir, "explicabilite")
        os.makedirs(explain_dir, exist_ok=True)

        importances = booster.feature_importance(importance_type='gain')

        n_sst_features = len(sst_lags_days) * 85 * 360
        n_slp_features = len(slp_lags_days) * 53 * 113
        
        extent_sst = [-180, 180, -15, 70] 
        extent_slp = [-100, 40, 20, 70] 

        # --- CARTES SST ---
        if n_sst_features > 0:
            sst_importances = importances[:n_sst_features].reshape(len(sst_lags_days), 85, 360)
            
            # Bascule lissage / brut
            if smooth_sigma > 0.0:
                print(f"-> Application d'un lissage Gaussien (sigma={smooth_sigma}) sur la SST")
                sst_plot_data = [ndimage.gaussian_filter(sst_importances[i], sigma=smooth_sigma) for i in range(len(sst_lags_days))]
                title_suffix = f"(Lissé sigma={smooth_sigma})"
                file_suffix = f"_smoothed_{smooth_sigma}"
            else:
                print("-> Génération des cartes SST brutes (sans lissage)")
                sst_plot_data = sst_importances
                title_suffix = "(Brut)"
                file_suffix = "_raw"

            global_max_sst = np.max(sst_plot_data) if len(sst_plot_data) > 0 else 1.0

            for i, lag in enumerate(sst_lags_days):
                fig, ax = plt.subplots(figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})
                
                ax.coastlines(resolution='110m', color='black', linewidth=1)
                ax.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':', alpha=0.5)
                
                im = ax.imshow(sst_plot_data[i], cmap='Reds', origin='lower', 
                               vmin=0, vmax=global_max_sst, 
                               extent=extent_sst, transform=ccrs.PlateCarree(),
                               interpolation='nearest')
                
                gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False

                plt.colorbar(im, ax=ax, label="Importance (Gain LGBM)", orientation='vertical', pad=0.02)
                ax.set_title(f"Zones déterminantes - SST Lag {lag} jours {title_suffix}")
                
                plt.savefig(os.path.join(explain_dir, f"importance_sst_lag_{lag}{file_suffix}.png"), bbox_inches='tight', dpi=150, facecolor='white')
                plt.close()

        # --- CARTES SLP ---
        if n_slp_features > 0:
            slp_importances = importances[n_sst_features:].reshape(len(slp_lags_days), 53, 113)
            
            # Bascule lissage / brut
            if smooth_sigma > 0.0:
                print(f"-> Application d'un lissage Gaussien (sigma={smooth_sigma}) sur la SLP")
                slp_plot_data = [ndimage.gaussian_filter(slp_importances[i], sigma=smooth_sigma) for i in range(len(slp_lags_days))]
                title_suffix = f"(Lissé sigma={smooth_sigma})"
                file_suffix = f"_smoothed_{smooth_sigma}"
            else:
                print("-> Génération des cartes SLP brutes (sans lissage)")
                slp_plot_data = slp_importances
                title_suffix = "(Brut)"
                file_suffix = "_raw"

            global_max_slp = np.max(slp_plot_data) if len(slp_plot_data) > 0 else 1.0

            for i, lag in enumerate(slp_lags_days):
                fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.PlateCarree()})
                
                ax.coastlines(resolution='50m', color='black', linewidth=1)
                ax.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':', alpha=0.5)
                
                im = ax.imshow(slp_plot_data[i], cmap='Reds', origin='lower', 
                               vmin=0, vmax=global_max_slp, 
                               extent=extent_slp, transform=ccrs.PlateCarree(),
                               interpolation='nearest')
                
                gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False

                plt.colorbar(im, ax=ax, label="Importance (Gain LGBM)", orientation='vertical', pad=0.02)
                ax.set_title(f"Zones déterminantes - SLP Lag {lag} jours {title_suffix}")
                
                plt.savefig(os.path.join(explain_dir, f"importance_slp_lag_{lag}{file_suffix}.png"), bbox_inches='tight', dpi=150, facecolor='white')
                plt.close()
            
        print(f"Modèle, analyses et matrices sauvegardés dans :\n{outdir}")
    else:
        print("Erreur : L'entraînement a échoué ou les données étaient vides.")