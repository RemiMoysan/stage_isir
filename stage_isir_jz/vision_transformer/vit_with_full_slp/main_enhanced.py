import os
import time
import argparse
import copy
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import random 
from datetime import datetime
from collections import defaultdict
import re
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import des dossiers siblings
import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.visualizations import (
    loss_figure, 
    loss_first_epoch, 
    plot_and_save_maps_light, 
    plot_correlation_evolution, 
    plot_r2_R2_evolution
)
from tools.datasets import Dataset, Dataset_mensuel
from tools.models import (
    ViT_Decoded_SLP_Multimodal, 
    compute_loss, 
    get_median_prediction_full_slp
)

# ============================================================
# DEVICE & ARGUMENTS CONFIGURATION & OUTPUT DIRECTORY SETUP
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', type=int, required=True, help='Loading of previous parameters (1) or start fresh (0)')
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'], help='Machine sur laquelle le code tourne')
    
    parser.add_argument('--nb_members_train', type=int, default=60, help='Nombre de membres à utiliser pour l\'entraînement')
    parser.add_argument('--nb_members_val', type=int, default=10, help='Nombre de membres à utiliser pour la validation')
    parser.add_argument('--nb_members_test', type=int, default=10, help='Nombre de membres à utiliser pour le test') 
    
    parser.add_argument('--seed', type=int, default=42, help='Seed pour le mélange inter membres')
    parser.add_argument('--nb_epochs', type=int, default=25, help='Nombre d\'époques pour l\'entraînement du ViT')
    parser.add_argument('--duree_lissage', type=int, default=0, help='Durée du lissage en jours pour les cibles')
    parser.add_argument('--bs', type=int, default=128, help='Taille de batch pour l\'entraînement (attention à la RAM en sortie carte complète)')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate pour l\'entraînement du ViT')
    parser.add_argument('--dr', type=float, default=0.20, help='Dropout rate pour le ViT')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[3], help='Liste des lags jours pour SST')
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[], help='Liste des lags jours pour SLP')
    parser.add_argument('--sst_lags_months', type=int, nargs='*', default=[2, 3, 4], help='Liste des lags mois pour SST')
    parser.add_argument('--slp_lags_months', type=int, nargs='*', default=[], help='Liste des lags mois pour SLP')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner pour l\'entraînement')
    
    parser.add_argument('--use_lags_attention', action='store_true', help='Activer l\'attention temporelle entre les lags')
    parser.add_argument('--roll_sst', action='store_true', help='Appliquer un roll sur les données SST pour centrer l\'océan Atlantique')
    
    # --- ARGUMENTS DE LOSS & REGULARISATION ---
    parser.add_argument('--loss_type', type=str, choices=['mse', 'l1', 'quantile', 'correlation'], default='mse', help='Fonction de coût pour l\'entraînement')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help='Quantiles à prédire (0.5 obligatoire)')
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Nombre de points de validation intra-époque (espacement logarithmique)')
    parser.add_argument('--monthly_reduction', action='store_true', help='Utiliser les données mensuelles (_1mo.nc)')
    parser.add_argument('--lat_weight', action='store_true', help='Pondération spatiale sqrt(cos(lat))')
    parser.add_argument('--weight_decay', type=float, default=0.03, help='Weight decay pour l\'optimiseur AdamW')
    args = parser.parse_args()

    # Vérification stricte des quantiles
    if args.loss_type == 'quantile':
        if 0.5 not in args.quantiles:
            raise ValueError("Erreur: Pour la quantile loss, la liste des quantiles (--quantiles) DOIT inclure la médiane (0.5).")

    # Routage dynamique des dossiers
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/vision_transformer/vit_with_full_slp/"
    elif args.machine in ['jean-zay-work', 'jean-zay-scratch']:
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_full_slp/"
    elif args.machine == 'mac_local':
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/vision_transformer/vit_with_full_slp/"

    # ============================================================
    # GLOBAL CONSTANTS & DATA SPLITTING
    # ============================================================
    sst_lags_days = args.sst_lags_days
    slp_lags_days = args.slp_lags_days
    sst_lags_months = args.sst_lags_months
    slp_lags_months = args.slp_lags_months
    bs = args.bs
    lr = args.lr
    dr = args.dr
    winter_months = args.winter_months
    duree_lissage = args.duree_lissage
    nb_epochs = args.nb_epochs
    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val
    nb_members_test = args.nb_members_test

    print("Arg Parameters:")
    print(f"  loss_type: {args.loss_type}", f" monthly_reduction: {args.monthly_reduction}", f" lat_weight: {args.lat_weight}", f" use_lags_attention: {args.use_lags_attention}")
    print(f"  SST Lags: {sst_lags_months if args.monthly_reduction else sst_lags_days}", f" SLP Lags: {slp_lags_months if args.monthly_reduction else slp_lags_days}")
    print(f"  Batch Size: {bs}", f" LR: {lr}", f" DR: {dr}", f" Winter Months: {winter_months}", f" Train/Val/Test: {nb_members_train}/{nb_members_val}/{nb_members_test}\n")

    patience = 10000
    target_indices = {100, 1000, 2000, 3000, 4000, 4500, 5000, 6000, 7000, 8000} if not args.monthly_reduction else {1, 10, 20, 30, 40, 45, 50, 60, 70, 80} 

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[nb_members_train:nb_members_train + nb_members_val]
    test_members = all_members[nb_members_train + nb_members_val : nb_members_train + nb_members_val + nb_members_test] if nb_members_test > 0 else []

    # Nom du dossier adapté
    loss_tag = args.loss_type
    if args.loss_type == 'quantile':
        loss_tag += "_" + "".join([str(q).replace('.', '') for q in args.quantiles])

    if not args.monthly_reduction:
        outdir_name = f"ViT_FullSLP_lags_att_{args.use_lags_attention}_lat_w_{args.lat_weight}_loss_{loss_tag}_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_{nb_members_test}_members_seed_{args.seed}_{duree_lissage}d_roll_{args.roll_sst}"
    else:
        outdir_name = f"ViT_FullSLP_lags_att_{args.use_lags_attention}_lat_w_{args.lat_weight}_loss_{loss_tag}_lags_{'_'.join(map(str, sst_lags_months))}_sst_{'_'.join(map(str, slp_lags_months))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_{nb_members_test}_members_seed_{args.seed}_monthly_roll_{args.roll_sst}"
            
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")

    # ============================================================
    # PRÉPARATION DES POIDS SPATIAUX (BARYCENTRE LATITUDE)
    # ============================================================
    wgts_map = None
    if args.lat_weight:
        sample_member = train_members[0]
        sample_path = os.path.join(base_home.replace("stage_isir_jz/vision_transformer/vit_with_full_slp/", ""), f"data/SLP/PSL_anom_LE2-{sample_member}_1mo.nc")
        try:
            ds_sample = xr.open_dataset(sample_path)
            lats = ds_sample['lat'].values
            
            # 1. Poids physiques réels : cos(lat)
            coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
            
            # 2. Normalisation barycentrique (somme des poids = nombre de pixels)
            # Dès lors, mean(coslat_norm) == 1.0 exactement.
            coslat_norm = coslat / np.mean(coslat)
            
            # 3. Adaptation algébrique selon la fonction de perte
            if args.loss_type == 'mse':
                # Pour MSE : on applique la racine carrée sur le poids NORMALISÉ
                # (wgts * y)^2 = coslat_norm * y^2
                wgts = np.sqrt(coslat_norm)
            else:
                # Pour L1, Quantile ou Correlation : le poids reste linéaire
                # |wgts * y| = coslat_norm * |y|
                wgts = coslat_norm

            h, w = len(lats), len(ds_sample['lon'].values)
            wgts = wgts.reshape(1, 1, h, 1)
            wgts_map = torch.tensor(np.broadcast_to(wgts, (1, 1, h, w)), dtype=torch.float32).to(device)
            ds_sample.close()
            print(f"✅ Grille de poids de latitude 2D (Barycentre exact pour loss '{args.loss_type}') générée.")
        except Exception as e:
            print(f"⚠️ Erreur chargement de la grille de latitude : {e}. Désactivation de lat_weight.")
            args.lat_weight = False

    # ============================================================
    # DATALOADERS
    # ============================================================
    intra_workers = min(2, n_workers)

    if not args.monthly_reduction:
        val_set = Dataset(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst)
        training_set = Dataset(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst)
    else:
        val_set = Dataset_mensuel(members=val_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst)
        training_set = Dataset_mensuel(members=train_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=intra_workers, pin_memory=True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    if nb_members_test > 0:
        if not args.monthly_reduction:
            test_set = Dataset(members=test_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst)
        else:
            test_set = Dataset_mensuel(members=test_members, selected_months=winter_months, machine=args.machine, target_type='map', sst_lags_months=sst_lags_months, slp_lags_months=slp_lags_months, roll_sst=args.roll_sst)
        
        testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)
        testloader_intra = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=intra_workers, pin_memory=True)
    else:
        testloader, testloader_intra = None, None

    # ============================================================
    # INITIALISATION DU VISION TRANSFORMER
    # ============================================================
    active_sst_lags = sst_lags_months if args.monthly_reduction else sst_lags_days
    active_slp_lags = slp_lags_months if args.monthly_reduction else slp_lags_days
    
    # Nombre de canaux de sortie en spatial : 1 par défaut, ou len(quantiles) si loss quantile
    out_chans_map = len(args.quantiles) if args.loss_type == 'quantile' else 1

    model = ViT_Decoded_SLP_Multimodal(
        sst_size=(85, 360),    
        slp_size=(53, 113),    
        patch_size_sst=(5, 10),    
        patch_size_slp=(5, 10),    
        in_chans_sst=len(active_sst_lags),  
        in_chans_slp=len(active_slp_lags),
        out_chans=out_chans_map,  # Paramètre à supporter dans ton ViT_Decoded_SLP_Multimodal
        embed_dim=128,         
        enc_depth=4,           
        dec_depth=4,           
        num_heads=4,           
        dr=dr,
        use_lags_attention=args.use_lags_attention # Paramètre à supporter dans ta classe
    ).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def get_vit_optimizer_groups(model, weight_decay=0.05):
        """
        Sépare les paramètres : weight decay pour les matrices 2D (Linear, Attention),
        0 weight decay pour les biais, LayerNorm et embeddings positionnels/temporels.
        """
        decay = []
        no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Si le paramètre est 1D (biais, LayerNorm) ou un embedding appris
            if len(param.shape) == 1 or "pos_embed" in name or "time_embed" in name or "cls_token" in name or "queries" in name:
                no_decay.append(param)
            else:
                decay.append(param)
                
        print(f"👉 Paramètres avec Weight Decay ({weight_decay}) : {sum(p.numel() for p in decay):,}")
        print(f"👉 Paramètres SANS Weight Decay (0.0) : {sum(p.numel() for p in no_decay):,}")
        
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]

    # ============================================================
    # INITIALISATION DE L'OPTIMISEUR (Remplacer torch.optim.Adam)
    # ============================================================
    # Un weight decay entre 0.01 et 0.05 est la norme pour un ViT de cette taille
    weight_decay = 0.03 if not hasattr(args, 'weight_decay') else args.weight_decay

    optimizer_groups = get_vit_optimizer_groups(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.95))

    print("Number of model parameters : ", sum(p.numel() for p in model.parameters()))

    if args.update == 1: 
        initial_params = torch.load(f"{outdir}/final_model_ViT_bs{bs}.pth")
        model.load_state_dict(initial_params['state_dict'])
        optimizer.load_state_dict(initial_params['optimizer'])
        train_losses = initial_params['train_losses']
        val_losses = initial_params['val_losses']
        best_val_loss = np.min(val_losses)
        train_corrs = initial_params.get('train_corrs', [])
        val_corrs = initial_params.get('val_corrs', [])
        train_R2 = initial_params.get('train_R2', [])
        val_R2 = initial_params.get('val_R2', [])
        test_losses = initial_params.get('test_losses', [])
        test_corrs = initial_params.get('test_corrs', [])
        test_R2 = initial_params.get('test_R2', [])
        train_ks = initial_params.get('train_ks', [])
        val_ks = initial_params.get('val_ks', [])
        test_ks = initial_params.get('test_ks', [])
        print("Model state updated")
    else: 
        train_losses, val_losses, train_corrs, val_corrs, train_R2, val_R2, test_losses, test_corrs, test_R2 = [], [], [], [], [], [], [], [], []
        best_val_loss = float('inf') 
        train_ks, val_ks, test_ks = [], [], []
        print("Initiated first ViT training")

    best_model_path = ""
    val_losses_per_member_history = defaultdict(list)
    epoch1_batch_losses, epoch1_baseline_losses = [], []
    
    # Suivi intra-époque 1 et 2
    intra_epoch1_steps, intra_epoch1_val_losses, intra_epoch1_val_corrs, intra_epoch1_val_R2, intra_epoch1_test_losses, intra_epoch1_test_corrs, intra_epoch1_test_R2, intra_epoch1_val_ks, intra_epoch1_test_ks = [], [], [], [], [], [], [], [], []
    intra_epoch2_steps, intra_epoch2_val_losses, intra_epoch2_val_corrs, intra_epoch2_val_R2, intra_epoch2_test_losses, intra_epoch2_test_corrs, intra_epoch2_test_R2, intra_epoch2_val_ks, intra_epoch2_test_ks = [], [], [], [], [], [], [], [], []

    # ============================================================
    # CALCUL DES STEPS DE VALIDATION INTRA-ÉPOQUE
    # ============================================================
    total_batches = len(trainloader)
    eval_steps = np.geomspace(1, total_batches - 1, num=args.nb_intra_evals, dtype=int)
    eval_steps = np.insert(eval_steps, 0, 0)
    eval_steps_set = set(eval_steps)
    print(f"Validation intra-époque 1 aux steps : {sorted(list(eval_steps_set))}")

    eval_steps_epoch2 = np.linspace(0, total_batches - 1, num=args.nb_intra_evals, dtype=int)
    eval_steps_epoch2 = np.insert(eval_steps_epoch2, 0, 0)
    eval_steps_epoch2_set = set(eval_steps_epoch2)

    # ============================================================
    # TRAINING & EVALUATION LOOP
    # ============================================================
    start_time = time.time() 
    best_model_state = None 
    patience_counter = 0
    epoch_times = []

    for epoch in range(nb_epochs):
        # ---------------- TRAINING ----------------
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0
        sum_p, sum_t, sum_p2, sum_t2, sum_pt, sum_res = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
            if batch_idx % 30 == 0:
                print(f" {100 * batch_idx / len(trainloader):.1f}% training complete", end='\r')
            optimizer.zero_grad()

            X_sst = X_sst.to(device, non_blocking=True)
            X_slp = X_slp.to(device, non_blocking=True)
            y_target = y_target.to(device, non_blocking=True)

            outputs = model(X_sst, X_slp)
            
            # Application de la pondération spatiale si activée (sur l'erreur ou les cartes)
            if args.lat_weight and wgts_map is not None:
                loss_value = compute_loss(outputs * wgts_map, y_target * wgts_map, args.loss_type, args.quantiles, reduction='mean')
            else:
                loss_value = compute_loss(outputs, y_target, args.loss_type, args.quantiles, reduction='mean')
                
            loss_value.backward()
            optimizer.step()
            
            running_train_loss += loss_value.item() * X_sst.size(0)
            total_train_samples += X_sst.size(0)

            # Statistiques pour corrélation & R2 (on aplatit l'espace spatial par batch)
            med_pred = get_median_prediction_full_slp(outputs, args.loss_type, args.quantiles, out_dim=1) if args.loss_type == 'quantile' else outputs
            p = med_pred.view(med_pred.size(0), -1).detach()
            t = y_target.view(y_target.size(0), -1).detach()
            
            sum_p += p.sum(dim=0)
            sum_t += t.sum(dim=0)
            sum_p2 += (p ** 2).sum(dim=0)
            sum_t2 += (t ** 2).sum(dim=0)
            sum_pt += (p * t).sum(dim=0)
            sum_res += ((p - t) ** 2).sum(dim=0)

            # Tracking époque 1
            if epoch == 0:
                epoch1_batch_losses.append(loss_value.item())
                with torch.no_grad():
                    zeros_pred = torch.zeros_like(outputs)
                    baseline_loss = compute_loss(zeros_pred, y_target, args.loss_type, args.quantiles, reduction='mean').item()
                    epoch1_baseline_losses.append(baseline_loss)

            # ----- INTRA-EPOCH EVALUATION -----
            if args.nb_intra_evals > 0 and (epoch == 0 or epoch == 1):
                current_eval_steps_set = eval_steps_set if epoch == 0 else eval_steps_epoch2_set
                if batch_idx in current_eval_steps_set or batch_idx == len(trainloader) - 1:
                    print(f"\n--- Intra-epoch validation at step {batch_idx}/{len(trainloader)} ---")
                    eval_phases = ['val', 'test'] if nb_members_test > 0 else ['val']
                    for key in eval_phases:
                        loader = valloader_intra if key == 'val' else testloader_intra
                        model.eval()
                        intra_val_loss, intra_n_samples = 0.0, 0
                        v_sum_p, v_sum_t, v_sum_p2, v_sum_t2, v_sum_pt, v_sum_res = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        
                        with torch.no_grad():
                            for v_X_sst, v_X_slp, v_y_target, _, _, _ in loader:
                                v_X_sst = v_X_sst.to(device, non_blocking=True)
                                v_X_slp = v_X_slp.to(device, non_blocking=True)
                                v_y_target = v_y_target.to(device, non_blocking=True)
                                
                                v_pred = model(v_X_sst, v_X_slp)
                                if args.lat_weight and wgts_map is not None:
                                    loss_val = compute_loss(v_pred * wgts_map, v_y_target * wgts_map, args.loss_type, args.quantiles, reduction='mean')
                                else:
                                    loss_val = compute_loss(v_pred, v_y_target, args.loss_type, args.quantiles, reduction='mean')
                                    
                                intra_val_loss += loss_val.item() * v_X_sst.size(0)
                                
                                p_val = get_median_prediction_full_slp(v_pred, args.loss_type, args.quantiles, out_dim=1) if args.loss_type == 'quantile' else v_pred
                                p_val = p_val.view(p_val.size(0), -1)
                                t_val = v_y_target.view(v_y_target.size(0), -1)
                                
                                v_sum_p += p_val.sum(dim=0)
                                v_sum_t += t_val.sum(dim=0)
                                v_sum_p2 += (p_val ** 2).sum(dim=0)
                                v_sum_t2 += (t_val ** 2).sum(dim=0)
                                v_sum_pt += (p_val * t_val).sum(dim=0)
                                v_sum_res += ((p_val - t_val) ** 2).sum(dim=0)
                                intra_n_samples += p_val.size(0)

                        v_mean_p, v_mean_t = v_sum_p / intra_n_samples, v_sum_t / intra_n_samples
                        v_var_p = (v_sum_p2 / intra_n_samples) - v_mean_p**2
                        v_var_t = (v_sum_t2 / intra_n_samples) - v_mean_t**2
                        v_cov_pt = (v_sum_pt / intra_n_samples) - (v_mean_p * v_mean_t)
                        
                        v_corr = (v_cov_pt / torch.sqrt(v_var_p * v_var_t + 1e-8)).mean().item()
                        v_r2 = (1 - (v_sum_res / (v_var_t * intra_n_samples + 1e-8))).mean().item()
                        v_k = torch.sqrt(v_var_p / (v_var_t + 1e-8)).mean().item()

                        # NOUVEAU CALCUL (R2 Global)
                        v_r2_global = 1 - (v_sum_res.sum() / (v_var_t.sum() * intra_n_samples + 1e-8))


                        
                        if epoch == 0:
                            if key == 'val': intra_epoch1_steps.append(batch_idx)  
                            intra_epoch1_val_losses.append(intra_val_loss / intra_n_samples) if key == 'val' else intra_epoch1_test_losses.append(intra_val_loss / intra_n_samples)
                            intra_epoch1_val_corrs.append(v_corr) if key == 'val' else intra_epoch1_test_corrs.append(v_corr)
                            intra_epoch1_val_R2.append(v_r2) if key == 'val' else intra_epoch1_test_R2.append(v_r2)
                            intra_epoch1_val_ks.append(v_k) if key == 'val' else intra_epoch1_test_ks.append(v_k)
                        elif epoch == 1:
                            if key == 'val': intra_epoch2_steps.append(batch_idx)
                            intra_epoch2_val_losses.append(intra_val_loss / intra_n_samples) if key == 'val' else intra_epoch2_test_losses.append(intra_val_loss / intra_n_samples)
                            intra_epoch2_val_corrs.append(v_corr) if key == 'val' else intra_epoch2_test_corrs.append(v_corr)
                            intra_epoch2_val_R2.append(v_r2) if key == 'val' else intra_epoch2_test_R2.append(v_r2)
                            intra_epoch2_val_ks.append(v_k) if key == 'val' else intra_epoch2_test_ks.append(v_k)
                        
                        current_intra_loss = intra_val_loss / intra_n_samples
                        print(f"-> Intra-{key} Loss: {current_intra_loss:.4f} | Corr: {v_corr:.4f} | R2: {v_r2:.4f} | R2 Global: {v_r2_global:.4f} | k: {v_k:.4f}")
                        
                        if key == 'val' and current_intra_loss < best_val_loss:
                            best_val_loss = current_intra_loss
                            best_model_state = copy.deepcopy(model.state_dict())
                            if best_model_path and os.path.exists(best_model_path):
                                os.remove(best_model_path)
                            best_model_path = os.path.join(outdir, f'best_val_ViT_bs{bs}_ep{epoch + 1}_step{batch_idx}_loss{best_val_loss:.4f}.pth')
                            torch.save(model.state_dict(), best_model_path)
                            print(f"   *** Nouveau Best Model (Intra) sauvegardé : {os.path.basename(best_model_path)} ***")

                        model.train()

        # Fin de l'époque d'entraînement
        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)
        
        mean_p, mean_t = sum_p / total_train_samples, sum_t / total_train_samples
        var_p = (sum_p2 / total_train_samples) - mean_p**2
        var_t = (sum_t2 / total_train_samples) - mean_t**2
        cov_pt = (sum_pt / total_train_samples) - (mean_p * mean_t)

        epoch_train_corr = (cov_pt / torch.sqrt(var_p * var_t + 1e-8)).mean().item()
        epoch_train_r2 = (1 - (sum_res / (var_t * total_train_samples + 1e-8))).mean().item()
        epoch_train_k = torch.sqrt(var_p / (var_t + 1e-8)).mean().item()

        epoch_train_r2_global = 1 - (sum_res.sum() / (var_t.sum() * total_train_samples + 1e-8))
        
        train_corrs.append(epoch_train_corr)
        train_R2.append(epoch_train_r2)
        train_ks.append(epoch_train_k)
        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f} | Corr: {epoch_train_corr:.4f} | R2: {epoch_train_r2:.4f} | R2 Global: {epoch_train_r2_global:.4f} | k: {epoch_train_k:.4f}')

        if epoch == 0 or epoch == 1:
            if epoch == 0:
                loss_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, outdir, label="Train")
            if args.nb_intra_evals > 0:
                c_losses = intra_epoch1_val_losses if epoch == 0 else intra_epoch2_val_losses
                c_steps = intra_epoch1_steps if epoch == 0 else intra_epoch2_steps
                c_losses_test = intra_epoch1_test_losses if epoch == 0 else intra_epoch2_test_losses
                c_corrs = intra_epoch1_val_corrs if epoch == 0 else intra_epoch2_val_corrs
                c_r2 = intra_epoch1_val_R2 if epoch == 0 else intra_epoch2_val_R2
                c_test_corrs = intra_epoch1_test_corrs if epoch == 0 else intra_epoch2_test_corrs
                c_test_r2 = intra_epoch1_test_R2 if epoch == 0 else intra_epoch2_test_R2
                c_ks = intra_epoch1_val_ks if epoch == 0 else intra_epoch2_val_ks
                c_test_ks = intra_epoch1_test_ks if epoch == 0 else intra_epoch2_test_ks

                loss_first_epoch(c_losses, [np.mean(epoch1_baseline_losses)]*len(c_losses), outdir, label="Intra-Val", batch_indexes=c_steps, epoch_num=epoch+1, batch_test_losses=c_losses_test)
                plot_correlation_evolution([], c_corrs, outdir, val_ks=c_ks, test_corrs=c_test_corrs, test_ks=c_test_ks, epoch_1=(epoch==0), epoch_2=(epoch==1), batch_indexes=c_steps)
                plot_r2_R2_evolution([], c_corrs, [], c_r2, outdir, epoch_1=(epoch==0), epoch_2=(epoch==1), batch_indexes=c_steps, test_R2=c_test_r2, test_corrs=c_test_corrs)

        # ---------------- VALIDATION & TEST ----------------
        per_member_loss = defaultdict(lambda: {'loss_sum': 0.0, 'count': 0})
        # Dictionnaire pour stocker uniquement les plots des indices cibles
        per_member_plots = defaultdict(
            lambda: {"time": [], "slp_true": [], "slp_pred": []}
        )
        
        eval_phases = ['val', 'test'] if nb_members_test > 0 else ['val']
        for key in eval_phases:
            loader = valloader if key == 'val' else testloader
            model.eval()
            running_val_loss, total_val_samples = 0.0, 0 
            sum_p, sum_t, sum_p2, sum_t2, sum_pt, sum_res = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            with torch.no_grad():
                for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(loader):
                    if batch_idx % 30 == 0:
                        print(f" {100 * batch_idx / len(loader):.1f}% {key} complete", end='\r')
                    
                    X_sst = X_sst.to(device, non_blocking=True)
                    X_slp = X_slp.to(device, non_blocking=True)
                    y_target = y_target.to(device, non_blocking=True)

                    outputs = model(X_sst, X_slp)
                    per_sample_losses = compute_loss(
                        outputs if not args.lat_weight or wgts_map is None else outputs * wgts_map, 
                        y_target if not args.lat_weight or wgts_map is None else y_target * wgts_map, 
                        args.loss_type, args.quantiles, reduction='none'
                    ).view(X_sst.size(0), -1).mean(dim=1).cpu().numpy()
                        
                    running_val_loss += float(per_sample_losses.sum())
                    total_val_samples += X_sst.size(0)

                    median_pred = get_median_prediction_full_slp(outputs, args.loss_type, args.quantiles, out_dim=1) if args.loss_type == 'quantile' else outputs
                    p = median_pred.view(median_pred.size(0), -1).detach()
                    t = y_target.view(y_target.size(0), -1).detach()
                    
                    sum_p += p.sum(dim=0)
                    sum_t += t.sum(dim=0)
                    sum_p2 += (p ** 2).sum(dim=0)
                    sum_t2 += (t ** 2).sum(dim=0)
                    sum_pt += (p * t).sum(dim=0)
                    sum_res += ((p - t) ** 2).sum(dim=0)

                    if key == 'val':
                        y_map_np = y_map.numpy()
                        pred_np = median_pred.cpu().numpy()
                        
                        members_list = [m if isinstance(m, str) else m.item().decode() if isinstance(m.item(), bytes) else str(m.item()) for m in members]
                        dates_list = [d if isinstance(d, str) else str(d) for d in dates]
                        for i, mem in enumerate(members_list):
                            current_idx = per_member_loss[mem]['count']
                            per_member_loss[mem]['loss_sum'] += float(per_sample_losses[i])
                            per_member_loss[mem]['count'] += 1
                            if current_idx in target_indices:
                                per_member_plots[mem]['time'].append(dates_list[i])
                                per_member_plots[mem]['slp_true'].append(y_map_np[i:i+1])
                                per_member_plots[mem]['slp_pred'].append(pred_np[i:i+1])

            if key == 'val':
                for mem, d in per_member_loss.items():
                    avg_loss = d['loss_sum'] / d['count'] if d['count'] > 0 else float('nan')
                    val_losses_per_member_history[mem].append(avg_loss)

            val_loss = running_val_loss / total_val_samples 
            val_losses.append(val_loss) if key == 'val' else test_losses.append(val_loss)
            
            mean_p, mean_t = sum_p / total_val_samples, sum_t / total_val_samples
            var_p = (sum_p2 / total_val_samples) - mean_p**2
            var_t = (sum_t2 / total_val_samples) - mean_t**2
            cov_pt = (sum_pt / total_val_samples) - (mean_p * mean_t)

            epoch_val_corr = (cov_pt / torch.sqrt(var_p * var_t + 1e-8)).mean().item()
            epoch_val_r2 = (1 - (sum_res / (var_t * total_val_samples + 1e-8))).mean().item()
            epoch_val_k = torch.sqrt(var_p / (var_t + 1e-8)).mean().item()

            # NOUVEAU CALCUL (R2 Global)
            epoch_val_r2_global = 1 - (sum_res.sum() / (var_t.sum() * total_val_samples + 1e-8))

            val_corrs.append(epoch_val_corr) if key == 'val' else test_corrs.append(epoch_val_corr)
            val_R2.append(epoch_val_r2) if key == 'val' else test_R2.append(epoch_val_r2)
            val_ks.append(epoch_val_k) if key == 'val' else test_ks.append(epoch_val_k)

            if key == 'val':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    if best_model_path and os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    best_model_path = os.path.join(outdir, f'best_val_ViT_bs{bs}_ep{epoch + 1}_end_loss{best_val_loss:.4f}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"   *** Nouveau Best Model (Fin d'époque) sauvegardé : {os.path.basename(best_model_path)} ***")
                else:
                    patience_counter += 1
            print(f'Epoch {epoch + 1} {key} Loss: {val_loss:.6f} | Corr: {epoch_val_corr:.4f} | R2: {epoch_val_r2:.4f} | R2_Global: {epoch_val_r2_global:.4f} | k: {epoch_val_k:.4f}')

        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        print(f'Epoch {epoch + 1} Elapsed Time: {current_time_min:.2f} minutes')

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
            break
        
        # ---------------- SAUVEGARDE RÉGULIÈRE ----------------
        if (epoch + 1) % 1 == 0:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses,
                'train_corrs': train_corrs, 'val_corrs': val_corrs, 'test_corrs': test_corrs,
                'train_R2': train_R2, 'val_R2': val_R2, 'test_R2': test_R2,
                'train_ks': train_ks, 'val_ks': val_ks, 'test_ks': test_ks
            }
            torch.save(state, f'{outdir}/final_model_ViT_bs{bs}.pth')
            loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times, per_member_val_losses=val_losses_per_member_history, test_losses=test_losses)
            plot_correlation_evolution(train_corrs, val_corrs, outdir, train_ks=train_ks, val_ks=val_ks, test_corrs=test_corrs, test_ks=test_ks)
            plot_r2_R2_evolution(train_corrs, val_corrs, train_R2, val_R2, outdir, test_R2=test_R2, test_corrs=test_corrs)
            # --- TRACÉ DES CARTES PAR MEMBRE DE VALIDATION ---
            for mem, d in per_member_plots.items():
                member_outdir = os.path.join(outdir, "per_member", mem)
                os.makedirs(member_outdir, exist_ok=True)
                plot_and_save_maps_light(
                    slp_true_list=d["slp_true"],
                    slp_pred_list=d["slp_pred"],
                    time_list=d["time"],
                    outdir=member_outdir,
                    epoch=(epoch + 1),
                )
            print(f"Saved checkpoint at epoch {epoch + 1}")
        
    print(f"Best Val Loss : {best_val_loss:.6f}")

    # Sauvegarde finale
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses,
        'train_corrs': train_corrs, 'val_corrs': val_corrs, 'test_corrs': test_corrs,
        'train_R2': train_R2, 'val_R2': val_R2, 'test_R2': test_R2,
        'train_ks': train_ks, 'val_ks': val_ks, 'test_ks': test_ks
    }
    torch.save(state, f'{outdir}/final_model_ViT_bs{bs}.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_ViT_bs{bs}.pth')

    end_time = time.time()
    print(f"Training complete, elapsed time: {(end_time - start_time) / 60:.2f} minutes")