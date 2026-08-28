import os
import argparse
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import random
import calendar
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
import re


import torch
import torch.nn as nn

# import des dossiers siblings
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from shared_tools.datasets import Dataset, Dataset_mensuel
from shared_tools.models import ConvVAE
from shared_tools.evaluation_functions import compute_latent_metrics_and_bootstraps, plot_combined_pcs_time_series, plot_latent_timeseries_raw_metrics 
from tools_cnn.models import CNN_Latent_SLP_Multimodal1, CNN_Latent_SLP_Multimodal1_tunable


# ============================================================
# MAIN EVALUATION SCRIPT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--embed_method', type=str, choices=['pca', 'vae'], default='vae')
    parser.add_argument('--cnn_dir', type=str, required=True, help='Chemin du dossier contenant le modèle entraîné (final_model_CNN...)')
    parser.add_argument('--model_type', type=str, choices=['best', 'final'], default='best', help='Évaluer le meilleur modèle (best) ou le dernier (final)')
    parser.add_argument('--embed_path', type=str, required=True, help='Chemin du dossier contenant le modèle d\'embedding (pca_model.joblib ou vae_model.pth)')
    parser.add_argument('--nb_members_train', type=int, default=10) # argument inutile
    parser.add_argument('--nb_members_val', type=int, default=5)
    parser.add_argument('--nb_members_test', type=int, default=5)
    parser.add_argument('--force_val_members', type=str, nargs='*', default=None, help='Forcer une liste spécifique de membres pour la val')
    parser.add_argument('--force_test_members', type=str, nargs='*', default=None, help='Forcer une liste spécifique de membres pour le test')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95])
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[])
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2,3,4], help='Liste des lags pour SST')
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[], help='Liste des lags pour SLP')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--duree_lissage', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    # parser.add_argument('--dr', type=float, default=0.2) Inutile car on ne fait que évaluer
    parser.add_argument('--roll_sst', action='store_true')
    parser.add_argument('--early_fusion_sst', action='store_true', help='Fusionner les lags SST dès les premières couches du CNN (au lieu de fusion tardive)')

    parser.add_argument('--monthly_mean', action='store_true', help='Appliquer une moyenne mensuelle. Si absent, garde la résolution journalière.')
    parser.add_argument('--n_bootstraps', type=int, default=1000, help='Itérations pour la p-value de corrélation')
    
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile','correlation'], default='mse')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données sous-échantillonnées mensuellement (_1mo.nc)')
    parser.add_argument('--lat_weight', action='store_true', help='Applique la pondération spatiale sqrt(cos(lat))')

    # Arguments pour le CNN tunable
    parser.add_argument('--dr_conv', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--dr_fc', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--fc_dim', type=int, default=20, help='Dimension de la couche fully connected')
    parser.add_argument('--n_feat', type=int, default=20, help='Nombre de filtres dans la première couche convolutive')
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh'], default='relu', help='Fonction d\'activation')
    parser.add_argument('--depth', type=int, default=3, help='Profondeur du réseau CNN')
    parser.add_argument('--filter_mult', type=float, default=1.0, help='Facteur de multiplication')
    parser.add_argument('--pool_strategy', type=str, choices=['progressive','standart'], default='progressive', help='Stratégie de pooling')
    parser.add_argument('--pool_type', type=str, choices=['avg','max'], default='max', help='Type de pooling')
    parser.add_argument('--sst_kx', type=int, default=3, help='Taille du kernel pour le pooling SST en x')
    parser.add_argument('--sst_ky', type=int, default=5, help='Taille du kernel pour le pooling SST en y')
    parser.add_argument('--sst_pool_x', type=int, default=2, help='Facteur de pooling pour SST en x')
    parser.add_argument('--sst_pool_y', type=int, default=2, help='Facteur de pooling pour SST en y')
    parser.add_argument('--use_gap', action='store_true', help='Utilise Global Average Pooling')
    
    args = parser.parse_args()

    if args.loss_type == 'quantile' and 0.5 not in args.quantiles:
        raise ValueError("Erreur: 0.5 doit être inclus dans la liste des quantiles pour extraire la médiane.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = time.time()

    # ============================================================
    # 1. SETUP DATASET & MEMBERS 
    # ============================================================
    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)
    if args.force_val_members is not None or args.force_test_members is not None:
        print("⚠️ OVERRIDE ACTIF : Utilisation des listes de membres forcées.")
        val_early_members = args.force_val_members if args.force_val_members else []
        test_members = args.force_test_members if args.force_test_members else []
        remaining = [m for m in all_members if m not in val_early_members and m not in test_members]
        # On coupe selon nb_members_train dans l'ordre de la seed !
        train_members = remaining[:args.nb_members_train]
        # Mise à jour des compteurs au cas où des parties du code les utilisent
        nb_members_train = len(train_members)
        nb_members_val = len(val_early_members)
        nb_members_test = len(test_members)
    else:
        nb_members_train = args.nb_members_train
        nb_members_val = args.nb_members_val
        nb_members_test = args.nb_members_test
        train_members = all_members[:nb_members_train]
        val_early_members = all_members[-nb_members_val:]
        test_members = all_members[nb_members_train:nb_members_train + nb_members_test] if nb_members_test > 0 else []
    # ---------------------------------

    val_members = val_early_members + test_members

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1) # On garde 1 CPU pour le process principal

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

    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_days=args.sst_lags_days, slp_lags_days=args.slp_lags_days, duree_lissage=args.duree_lissage, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=args.winter_months, machine=args.machine, target_type='map', sst_lags_months=args.sst_lags_months, slp_lags_months=args.slp_lags_months, roll_sst=args.roll_sst,slp_std=dynamic_slp_std)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=n_workers, pin_memory=True)


    # ============================================================
    # PRÉPARATION DES POIDS SPATIAUX (POUR DÉCODAGE PCA PONDÉRÉ) : à mettre avant le sanity check de l'embedder non??
    # ============================================================
    wgts_flat = None
    if args.lat_weight and args.embed_method == 'pca':
        # On lit un fichier SLP n'importe lequel juste pour extraire la grille de latitude
        sample_member = val_members[0]
        sample_path = os.path.join(f"/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/PSL_anom_LE2-{sample_member}_1mo.nc")        
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            h, w = len(lats), len(ds_sample['lon'].values)
            wgts = np.sqrt(coslat).reshape(h, 1)
            wgts_flat = np.broadcast_to(wgts, (h, w)).flatten()
            safe_wgts = np.maximum(wgts_flat, 1e-5) # Pour éviter division par zéro
            ds_sample.close()
            print("Grille de poids de latitude générée pour le décodage PCA.")
        except Exception as e:
            print(f"Erreur lors du chargement de la grille de latitude : {e}")

    # ============================================================
    # 2. LOAD MODELS
    # ============================================================
    pca_model, vae_model = None, None
    if args.embed_method == 'pca':
        pca_path = args.embed_path
        pca_model = joblib.load(pca_path)
    elif args.embed_method == 'vae':
        vae_model = ConvVAE(latent_dim=args.latent_dim).to(device)
        vae_model.load_state_dict(torch.load(args.embed_path, map_location=device))
        vae_model.eval()

    out_features = args.latent_dim * len(args.quantiles) if args.loss_type == 'quantile' else args.latent_dim

    if args.monthly_reduction:
        active_sst_lags = args.sst_lags_months
        active_slp_lags = args.slp_lags_months
    else:
        active_sst_lags = args.sst_lags_days
        active_slp_lags = args.slp_lags_days


    model = CNN_Latent_SLP_Multimodal1_tunable(
        dr_conv=args.dr_conv,
        dr_fc=args.dr_fc,
        fc_dim=args.fc_dim,
        nb_out=out_features,
        in_chans_sst=len(active_sst_lags),
        in_chans_slp=len(active_slp_lags),
        n_feat=args.n_feat,
        early_fusion_sst=args.early_fusion_sst,
        depth = args.depth,
        filter_mult = args.filter_mult,
        sst_kx = args.sst_kx,
        sst_ky = args.sst_ky,
        sst_pool_x = args.sst_pool_x,
        sst_pool_y = args.sst_pool_y,
        pool_type = args.pool_type,
        pool_strategy = args.pool_strategy,
        activation = args.activation,
        use_gap = args.use_gap
    ).to(device)

    # Dummy forward pour initialiser le LazyLinear
    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(active_sst_lags), 85, 360).to(device) if len(active_sst_lags) > 0 else None
        dummy_slp = torch.zeros(1, len(active_slp_lags), 53, 113).to(device) if len(active_slp_lags) > 0 else None
        _ = model(dummy_sst, dummy_slp)

    if args.model_type == 'best':
        cnn_path = os.path.join(args.cnn_dir, f"best_val_CNN.pth")
        if not os.path.exists(cnn_path):
            cnn_path = os.path.join(args.cnn_dir, f"best_val_CNN_bs{args.bs}.pth")
            print("⚠️ On utilise l'ancien nom")
    elif args.model_type == 'final':
        cnn_path = os.path.join(args.cnn_dir, f"final_model_CNN.pth")
        if not os.path.exists(cnn_path):
            cnn_path = os.path.join(args.cnn_dir, f"final_model_CNN_bs{args.bs}.pth")
            print("⚠️ On utilise l'ancien nom")
        
    checkpoint = torch.load(cnn_path, map_location=device)
    # Check si c'est un state_dict pur ou un dictionnaire avec 'state_dict' (cf. format de sauvegarde)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Models loaded successfully.")

    # ============================================================
    # 3. INFERENCE LOOP
    # ============================================================
    dates_list, members_list = [], []
    preds_list, trues_list = [], []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(valloader):
            X_sst = X_sst.to(device)
            X_slp = X_slp.to(device)
            
            # Récupération de la cible latente
            if args.embed_method == 'pca':
                slp_flat = y_target.view(y_target.size(0), -1).numpy()
                if wgts_flat is not None:
                    slp_flat *= wgts_flat
                target_embed = pca_model.transform(slp_flat)[:, :args.latent_dim]
                
            elif args.embed_method == 'vae':
                y_target = y_target.to(device)
                target_tensor, _ = vae_model.encode(y_target)
                target_embed = target_tensor.cpu().numpy()

            predicted_latent = model(X_sst, X_slp).cpu().numpy()

            preds_list.append(predicted_latent)
            trues_list.append(target_embed)
            dates_list.extend([str(d) for d in dates])
            
            for m in members:
                m_str = m if isinstance(m, str) else (m.item().decode() if isinstance(m.item(), bytes) else str(m.item()))
                members_list.append(m_str)

    preds_arr = np.concatenate(preds_list, axis=0)
    trues_arr = np.concatenate(trues_list, axis=0)

    if args.loss_type == 'quantile':
        # (N, latent_dim * n_quantiles) -> (N, latent_dim, n_quantiles)
        preds_arr = preds_arr.reshape(-1, args.latent_dim, len(args.quantiles))

    # Construction du DataFrame Pandas pour un traitement facile
    df = pd.DataFrame({'time': pd.to_datetime(dates_list), 'member': members_list})
    for i in range(args.latent_dim):
        # df[f'pred_pc{i+1}'] = preds_arr[:, i]
        df[f'true_pc{i+1}'] = trues_arr[:, i]
        if args.loss_type == 'quantile':
            # Extraction de la médiane et de tous les quantiles
            median_idx = args.quantiles.index(0.5)
            df[f'pred_pc{i+1}'] = preds_arr[:, i, median_idx]  # La médiane sert de PC principale pour les calculs d'erreurs
            for q_idx, q in enumerate(args.quantiles):
                df[f'pred_pc{i+1}_q{q}'] = preds_arr[:, i, q_idx]
        else:
            df[f'pred_pc{i+1}'] = preds_arr[:, i]

    # ============================================================
    # 4. STATISTIQUES & PLOTS PAR MEMBRE ET PAR MOIS (UNIFIÉ)
    # ============================================================
    freq_label = "Monthly" if args.monthly_mean else "Daily"
    print(f"\nTime Series Frequency Set To: {freq_label}")
    
    unique_members = df['member'].unique()
    max_pcs_to_plot = min(5, args.latent_dim) # Tracé groupé de PC1 à PC5

    for member in unique_members:
        print(f"\n{'='*40}\nEvaluating Latent Member: {member}\n{'='*40}")
        df_member = df[df['member'] == member].copy()
        
        split_name = 'val' if member in val_early_members else 'test'
        member_base_dir = os.path.join(args.cnn_dir, f"latent_eval_{args.model_type}", f"{member}_{split_name}", freq_label)

        # BOUCLE UNIFIÉE PAR MOIS D'HIVER
        for m in args.winter_months:
            month_name = calendar.month_name[m]
            print(f"  --- Month: {month_name} ---")
            
            df_month = df_member[df_member['time'].dt.month == m].copy().reset_index(drop=True)
            if len(df_month) < 2:
                print(f"    Not enough data for month {m}. Skipping.")
                continue

            month_outdir = os.path.join(member_base_dir, f"month_{m}_{calendar.month_abbr[m]}")
            os.makedirs(month_outdir, exist_ok=True)

            pred_cols = [f'pred_pc{i+1}' for i in range(args.latent_dim)]
            true_cols = [f'true_pc{i+1}' for i in range(args.latent_dim)]
            
            if args.monthly_mean:
                df_m_res = df_month.set_index('time')[pred_cols + true_cols].resample('1M').mean().dropna().reset_index()
            else:
                df_m_res = df_month[['time'] + pred_cols + true_cols].dropna().reset_index(drop=True)
                
            Z_p_m = df_m_res[pred_cols].values  # (T, latent_dim)
            Z_t_m = df_m_res[true_cols].values  # (T, latent_dim)
            
            print(f"    Computing all latent metrics & bootstraps for {month_name}...")
            df_lat_ts, stats_global_month, stats_per_pc = compute_latent_metrics_and_bootstraps(
                Z_t_m, Z_p_m, df_m_res['time'], n_bootstraps=min(300, args.n_bootstraps)
            )

            # 3. Tracé des séries brutes latentes (MSE, MAE, Corr vectorielle vs baselines 0)
            plot_latent_timeseries_raw_metrics(df_lat_ts, member, month_outdir, freq_label, stats_global_month)

            # 4. Préparation des quantiles pour le plot groupé des PC
            quantiles_dict = {}
            if args.loss_type == 'quantile':
                for pc_idx in range(1, max_pcs_to_plot + 1):
                    quantiles_dict[pc_idx] = {}
                    for q in args.quantiles:
                        if q == 0.5: continue
                        col_q = f'pred_pc{pc_idx}_q{q}'
                        if col_q in df_month:
                            quantiles_dict[pc_idx][q] = df_m_res[col_q] if col_q in df_m_res else df_month[col_q]

            # 5. Tracé du graphique groupé PC1..PCk pour CE mois
            plot_combined_pcs_time_series(
                df_m_res, stats_per_pc, stats_global_month, m, member, 
                month_outdir, max_pcs=max_pcs_to_plot, freq_label=freq_label, quantiles_dict=quantiles_dict
            )
            print(f"    -> Plots générés avec succès dans : {month_outdir}")



    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Evaluation Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("\nÉvaluation terminée avec succès !")