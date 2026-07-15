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
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

# import des dossiers siblings

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

# Ajouter le dossier "tools" de vision_transformer au sys.path pour tes imports de modèles
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from tools.visualizations import loss_figure, accuracy_figure, plot_confusion_matrix, loss_acc_first_epoch
from tools.models import get_fast_labels
from tools.datasets import Dataset
from tools_cnn.models import CNN_Classifier_Multimodal

# ============================================================
# DEVICE & ARGUMENTS CONFIGURATION & OUTPUT DIRECTORY SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', type=int, required=True, help='Loading of previous parameters (1 for yes, 0 for no)')
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'], help='Machine sur laquelle le code tourne (adapte les chemins automatiquement)')
    
    parser.add_argument('--nb_members_train', type=int, default=10, help='Nombre de membres à utiliser pour l\'entraînement')
    parser.add_argument('--nb_members_val', type=int, default=5, help='Nombre de membres à utiliser pour la validation')
    parser.add_argument('--seed', type=int, default=42, help='Seed pour le mélange inter membres')
    parser.add_argument('--nb_epochs', type=int, default=30, help='Nombre d\'époques pour l\'entraînement du CNN')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours pour les cibles 10 ou 30')
    parser.add_argument('--bs', type=int, default=128, help='Taille de batch pour l\'entraînement')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate pour l\'entraînement du CNN')
    parser.add_argument('--dr', type=float, default=0.2, help='Dropout rate pour le CNN')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95], help='Liste des lags pour SST (ex: --sst_lags_days 30 60 90)')
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[], help='Liste des lags pour SLP (optionnel)')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner pour l\'entraînement')

    parser.add_argument('--metric', type=str, default='mse', choices=['mse', 'correlation','pc1_quantiles','mse_latent'], help='Métrique pour le calcul des labels')    
    parser.add_argument('--master_ref_path', type=str, default="/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/composites_4_regimes/master_ref_generator_89_members_10d_embedding_method_pca/master_reference_global.npz", help='Chemin vers la référence maître')
    parser.add_argument('--projector_path', type=str, default="", help='Chemin vers le projector à utiliser pour calculer les embeddings de référence (optionnel)')
    
    parser.add_argument('--roll_sst', action='store_true', help='Appliquer un roll sur les données SST pour centrer l\'océan Atlantique')
    parser.add_argument('--early_fusion_sst', action='store_true', help='Fusionner les lags SST dès les premières couches du CNN (au lieu de fusion tardive)')
    parser.add_argument('--nb_intra_evals', type=int, default=15, help='Nombre de points de validation intra-époque (espacement logarithmique epoch 1, linéaire epoch 2)')
    args = parser.parse_args()

    # Routage dynamique du dossier de sortie (Output Dir)
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/cnn/cnn_classifier/"
    elif args.machine == 'jean-zay-work' or args.machine == 'jean-zay-scratch':
        # WORK_uxg=/lustre/fswork/projects/rech/uxg/uca57ub
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/cnn/cnn_classifier/" 
    elif args.machine == "mac_local":
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/cnn/cnn_classifier/"
    else:
        raise ValueError("Machine argument must be 'hacienda', 'jean-zay-work', 'jean-zay-scratch' or 'mac_local'.")

    # ============================================================
    # GLOBAL CONSTANTS
    # ============================================================

    sst_lags_days = args.sst_lags_days
    slp_lags_days = args.slp_lags_days
    bs = args.bs
    lr = args.lr
    dr = args.dr
    winter_months = args.winter_months
    duree_lissage = args.duree_lissage
    nb_epochs = args.nb_epochs
    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val
    metric = args.metric
    patience = 10000             

    print("Arg Parameters:")
    print(f"  Metric: {metric}", f" SST Lags: {sst_lags_days}", f" SLP Lags: {slp_lags_days}", f" Batch Size: {bs}", f" LR: {lr}", f" DR: {dr}", f" Months: {winter_months}", f" Smoothing: {duree_lissage}", f" Epochs: {nb_epochs}", f" Train Members: {nb_members_train}", f" Val Members: {nb_members_val}\n")

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[-nb_members_val:]

    outdir_name = f"CNN_classifier_early_fusion_sst_{args.early_fusion_sst}_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_members_seed_{args.seed}_{duree_lissage}d_metric_{metric}_roll_sst_{args.roll_sst}"
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)
    print(f"Dossier de sauvegarde : {outdir}")

    # ============================================================
    # PRÉPARATION LABELS & CLASSES
    # ============================================================

    master_ref = dict(np.load(args.master_ref_path)) 
    print("Référence maître chargée !")

    if args.projector_path and os.path.exists(args.projector_path):
        projector = joblib.load(args.projector_path)
        print("Projector chargé !")
    else:
        projector = None

    # Déduction du nombre de classes selon la métrique choisie
    if args.metric == 'pc1_quantiles':
        num_classes = len(master_ref['pc1_bins']) - 1
    elif args.metric in ['correlation', 'mse']:
        regime_keys = [k for k in master_ref.keys() if k.endswith("_slp_0_mean") and not k.startswith("GLOBAL")]
        num_classes = len(regime_keys)
    elif args.metric == 'mse_latent':
        num_classes = master_ref['ref_centroids_latent'].shape[0]
    else:
        num_classes = 4

    print(f"--> Détection automatique : {num_classes} classes pour la métrique '{args.metric}'")

    # --- NOUVEAU : Détermination de la baseline (Priors et Classe Majoritaire) ---
    class_counts = np.ones(num_classes)
    if args.metric in ['correlation', 'mse', 'mse_latent']:
        for i in range(num_classes):
            for k, v in master_ref.items():
                if f"regime_{i+1}_" in k and k.endswith("_count"):
                    class_counts[i] = float(v)
                    break
    elif args.metric == 'pc1_quantiles':
        class_counts = np.ones(num_classes) 

    class_probs = class_counts / class_counts.sum()
    baseline_logits = torch.tensor(np.log(class_probs), dtype=torch.float32).unsqueeze(0).to(device)
    majority_class = int(np.argmax(class_counts))
    print(f"-> Baseline (Majority Class) : Classe {majority_class} avec une probabilité a priori de {class_probs[majority_class]*100:.1f}%")

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")

    val_set = Dataset(members=val_members, selected_months=winter_months, machine = args.machine,target_type ='map',sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    # 2. NOUVEAU : Dataloader allégé pour la validation intra-époque
    intra_workers = min(2, n_workers)
    valloader_intra = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=intra_workers, pin_memory=True)

    training_set = Dataset(members=train_members, selected_months=winter_months, machine = args.machine,target_type ='map',sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage, roll_sst=args.roll_sst)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # MODEL & OPTIMIZER SETUP
    # ============================================================

    model = CNN_Classifier_Multimodal(
        num_classes=num_classes,         # 4 régimes  
        in_chans_sst=len(sst_lags_days),  
        in_chans_slp=len(slp_lags_days),
        n_feat = 8,         # la même pour la slp et pour la sst          
        dr=dr, 
        early_fusion_sst=args.early_fusion_sst                  
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 2. Dummy forward (Le passage factice indispensable)
    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(sst_lags_days), 85, 360).to(device) if len(sst_lags_days) > 0 else None
        dummy_slp = torch.zeros(1, len(slp_lags_days), 53, 113).to(device) if len(slp_lags_days) > 0 else None
        _ = model(dummy_sst, dummy_slp)

    print("Number of model parameters : ", sum(p.numel() for p in model.parameters()))

    best_model_path = "" # Initialisation sécurisée

    # plus robuste que les autres codes je crois dans le cas update == 1
    if args.update == 1: 
        initial_params = torch.load(f"{outdir}/final_model_CNN_bs{bs}.pth")
        model.load_state_dict(initial_params['state_dict'])
        optimizer.load_state_dict(initial_params['optimizer'])
        train_losses = initial_params.get('train_losses', [])
        val_losses = initial_params.get('val_losses', [])
        train_acc_history = initial_params.get('train_acc', [])
        val_acc_history = initial_params.get('val_acc', [])
        best_val_loss = np.min(val_losses) if len(val_losses) > 0 else float('inf')
        
        # Rustine anti-crash
        epoch_times = [0.0] * len(train_losses) 
        val_losses_per_member_history = defaultdict(list, initial_params.get('val_losses_per_member_history', {}))
        val_acc_per_member_history = defaultdict(list, initial_params.get('val_acc_per_member_history', {}))
        print("Model state updated")
    else: 
        train_losses, val_losses, epoch_times = [], [], []
        train_acc_history, val_acc_history = [], []
        val_losses_per_member_history = defaultdict(list)
        val_acc_per_member_history = defaultdict(list)
        best_val_loss = float('inf') 
        print("Initiate first CNN training")

    # ============================================================
    # CALCUL DES STEPS DE VALIDATION INTRA-ÉPOQUE
    # ============================================================
    nb_intra_evals = args.nb_intra_evals
    total_batches = len(trainloader)
    
    # geomspace pour epoch 1 (espacement exponentiel)
    eval_steps = np.geomspace(1, total_batches - 1, num=nb_intra_evals, dtype=int)
    eval_steps = np.insert(eval_steps, 0, 0)
    eval_steps_set = set(eval_steps)
    
    # linspace pour epoch 2 (espacement linéaire)
    eval_steps_epoch2 = np.linspace(0, total_batches - 1, num=nb_intra_evals, dtype=int)
    eval_steps_epoch2 = np.insert(eval_steps_epoch2, 0, 0)
    eval_steps_epoch2_set = set(eval_steps_epoch2)

    print(f"Validation intra-époque aux steps : {sorted(list(eval_steps_set))}")

    # ============================================================
    # TRAINING & EVALUATION LOOP
    # ============================================================
    start_time = time.time() 
    best_model_state = None 
    patience_counter = 0

    # NOUVEAU : Suivi de la loss et accuracy par batch pour l'époque 1
    epoch1_batch_losses = []
    epoch1_baseline_losses = []
    epoch1_batch_accs = []
    epoch1_baseline_accs = []

    # Variables pour le suivi intra-époque 1
    intra_epoch1_steps = []
    intra_epoch1_val_losses = []
    intra_epoch1_val_accs = []

    # Variables pour le suivi intra-époque 2
    intra_epoch2_steps = []
    intra_epoch2_val_losses = []
    intra_epoch2_val_accs = []

    for epoch in range(nb_epochs):
        # ---------------- TRAINING ----------------
        model.train()
        running_train_loss = 0.0
        running_train_correct = 0 
        total_train_samples = 0

        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
            if batch_idx % 30 == 0:
                print(f" {100 * batch_idx / len(trainloader):.1f}% training complete", end='\r')
            optimizer.zero_grad()

            X_sst = X_sst.to(device, non_blocking=True)
            X_slp = X_slp.to(device, non_blocking=True)

        # --- Création des targets à la volée ---
            # y_map contient ta vraie SLP à t=0 (Numpy format)
            labels = get_fast_labels(y_target.numpy(), master_ref, metric=metric, projector=projector)
            labels = labels.to(device) 
            
            # --- Forward ---
            logits = model(X_sst, X_slp)

            # --- Calcul de l'accuracy batch ---
            preds = torch.argmax(logits, dim=1)
            running_train_correct += (preds == labels).sum().item()
            
            # --- Loss ---
            # CrossEntropy prend des logits (Batch, 4) et des vrais labels (Batch,)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            current_batch_size = X_sst.size(0)
            running_train_loss += loss.item() * current_batch_size
            total_train_samples += current_batch_size

            # ----- NOUVEAU : Enregistrer les métriques du batch et la baseline (Époque 1) -----
            if epoch == 0:
                epoch1_batch_losses.append(loss.item())
                batch_acc = (preds == labels).sum().item() / current_batch_size * 100.0
                epoch1_batch_accs.append(batch_acc)
                
                with torch.no_grad():
                    b_logits = baseline_logits.expand(current_batch_size, -1)
                    b_loss = criterion(b_logits, labels).item()
                    epoch1_baseline_losses.append(b_loss)
                    
                    b_preds = torch.full_like(labels, majority_class)
                    b_acc = (b_preds == labels).sum().item() / current_batch_size * 100.0
                    epoch1_baseline_accs.append(b_acc)

            # ----- NOUVEAU : Validation intra-époque -----
            if args.nb_intra_evals > 0 and (epoch == 0 or epoch == 1):
                current_eval_steps_set = eval_steps_set if epoch == 0 else eval_steps_epoch2_set
                if batch_idx in current_eval_steps_set or batch_idx == len(trainloader) - 1:
                    print(f"\n--- Intra-epoch validation at step {batch_idx}/{len(trainloader)} ---")
                    
                    model.eval()
                    intra_val_loss = 0.0
                    intra_correct = 0
                    intra_n_samples = 0
                    
                    with torch.no_grad():
                        for v_batch_idx, (v_X_sst, v_X_slp, v_y_target, _, _, _) in enumerate(valloader_intra):
                            v_X_sst = v_X_sst.to(device, non_blocking=True)
                            v_X_slp = v_X_slp.to(device, non_blocking=True)
                            
                            v_labels = get_fast_labels(v_y_target.numpy(), master_ref, metric=metric, projector=projector)
                            v_labels = v_labels.to(device)
                            
                            v_logits = model(v_X_sst, v_X_slp)
                            loss_val = criterion(v_logits, v_labels)
                            
                            v_preds = torch.argmax(v_logits, dim=1)
                            
                            batch_size_val = v_X_sst.size(0)
                            intra_val_loss += loss_val.item() * batch_size_val
                            intra_correct += (v_preds == v_labels).sum().item()
                            intra_n_samples += batch_size_val

                    current_intra_loss = intra_val_loss / intra_n_samples
                    current_intra_acc = (intra_correct / intra_n_samples) * 100.0
                    
                    if epoch == 0:
                        intra_epoch1_steps.append(batch_idx)
                        intra_epoch1_val_losses.append(current_intra_loss)
                        intra_epoch1_val_accs.append(current_intra_acc)
                    elif epoch == 1:
                        intra_epoch2_steps.append(batch_idx)
                        intra_epoch2_val_losses.append(current_intra_loss)
                        intra_epoch2_val_accs.append(current_intra_acc)
                    
                    print(f"-> Intra-Val Loss: {current_intra_loss:.4f} | Intra-Val Acc: {current_intra_acc:.2f}%")
                    
                    if current_intra_loss < best_val_loss:
                        best_val_loss = current_intra_loss
                        best_model_state = copy.deepcopy(model.state_dict())
                        
                        if best_model_path and os.path.exists(best_model_path):
                            os.remove(best_model_path)
                            
                        best_model_path = os.path.join(outdir, f'best_val_CNN_bs{bs}_ep{epoch + 1}_step{batch_idx}_loss{best_val_loss:.4f}.pth')
                        torch.save(model.state_dict(), best_model_path)
                        print(f"   *** Nouveau Best Model (Intra) sauvegardé : {os.path.basename(best_model_path)} ***")
                    
                    model.train()
            # ----------------------------------------------------------------------------------

        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)
        train_acc = (running_train_correct / total_train_samples) * 100
        train_acc_history.append(train_acc)

        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}, Training Accuracy: {train_acc:.2f}%')

        # ----- NOUVEAU : Appel de la fonction de visualisation avec routage dynamique -----
        if epoch == 0 or epoch == 1:
            if epoch == 0:
                loss_acc_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, 
                                     epoch1_batch_accs, epoch1_baseline_accs, 
                                     outdir, label="Train")
            
            if args.nb_intra_evals > 0:
                current_intra_losses = intra_epoch1_val_losses if epoch == 0 else intra_epoch2_val_losses
                current_intra_accs = intra_epoch1_val_accs if epoch == 0 else intra_epoch2_val_accs
                current_intra_steps = intra_epoch1_steps if epoch == 0 else intra_epoch2_steps
                
                loss_acc_first_epoch(current_intra_losses, [np.mean(epoch1_baseline_losses)] * len(current_intra_losses),
                                     current_intra_accs, [np.mean(epoch1_baseline_accs)] * len(current_intra_accs),
                                     outdir, label="Intra-Val", batch_indexes=current_intra_steps, epoch_num=epoch+1)
        # -----------------------------------------------------------

    # ---------------- VALIDATION ----------------
        model.eval()
        time_list = []
        running_val_loss = 0.0
        total_val_samples = 0
        
        per_member_metrics = defaultdict(lambda: {'loss_sum': 0.0, 'count': 0, 'preds': [], 'labels': []})
        # Listes pour stocker TOUTES les prédictions et les vrais labels de l'époque
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(valloader):
                if batch_idx % 30 == 0:
                    print(f" {100 * batch_idx / len(valloader):.1f}% val complete", end='\r')
                
                X_sst = X_sst.to(device, non_blocking=True)
                X_slp = X_slp.to(device, non_blocking=True)
                
                # --- Création des targets à la volée ---
                labels = get_fast_labels(y_target.numpy(), master_ref, metric=metric, projector=projector)
                labels = labels.to(device)
                
                # --- Forward ---
                logits = model(X_sst, X_slp)
                loss = criterion(logits, labels)
                current_batch_size = X_sst.size(0)
                running_val_loss += loss.item() * current_batch_size
                total_val_samples += current_batch_size

                # Loss INDIVIDUELLE
                per_sample_losses = F.cross_entropy(logits, labels, reduction='none').cpu().numpy()

                # --- Calcul des prédictions (on prend la classe avec le score le plus haut) ---
                preds = torch.argmax(logits, dim=1)
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                members_list = [m if isinstance(m, str) else m.item().decode() if hasattr(m, 'item') else str(m) for m in members]

                for i, mem in enumerate(members_list):
                    per_member_metrics[mem]['loss_sum'] += float(per_sample_losses[i])
                    per_member_metrics[mem]['count'] += 1
                    per_member_metrics[mem]['preds'].append(preds_np[i])
                    per_member_metrics[mem]['labels'].append(labels_np[i])
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy()) # ou utiliser directement labels_np, à optimiser dans le code initial déjà
                time_list.extend(dates)

        val_loss = running_val_loss / total_val_samples
        val_losses.append(val_loss)

        # Enregistrement par membre
        for mem, d in per_member_metrics.items():
            avg_loss = d['loss_sum'] / d['count'] if d['count'] > 0 else float('nan')
            val_losses_per_member_history[mem].append(avg_loss)

            mem_acc = accuracy_score(d['labels'], d['preds']) * 100 if d['count'] > 0 else float('nan')
            val_acc_per_member_history[mem].append(mem_acc)
        
        val_accuracy = accuracy_score(all_val_labels, all_val_preds) * 100
        val_acc_history.append(val_accuracy) 

        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        
        print(f'Epoch {epoch + 1} Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.2f}% | Elapsed Time: {current_time_min:.2f} min')

        # ---------------- EARLY STOPPING & SAVING ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                
            best_model_path = os.path.join(outdir, f'best_val_CNN_bs{bs}_ep{epoch + 1}_end_loss{best_val_loss:.4f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"   *** Nouveau Best Model (Fin d'époque) sauvegardé : {os.path.basename(best_model_path)} ***")
            
            # Optionnel : Sauvegarder la matrice de confusion du meilleur modèle en direct
            plot_confusion_matrix(all_val_labels, all_val_preds, outdir, master_ref, filename='best_confusion_matrix.png')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
                break
            
        if epoch % 1 == 0:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'train_losses': train_losses, 
                'val_losses': val_losses,
                'train_acc': train_acc_history, 
                'val_acc': val_acc_history,
                'val_losses_per_member_history': dict(val_losses_per_member_history),
                'val_acc_per_member_history': dict(val_acc_per_member_history)
            }
            torch.save(state, f'{outdir}/final_model_CNN_bs{bs}.pth')
            loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times, per_member_val_losses=val_losses_per_member_history)
            accuracy_figure(len(train_losses), train_acc_history, val_acc_history, outdir, epoch_times, per_member_val_accuracies=val_acc_per_member_history)

    # ---------------- SAUVEGARDE RÉGULIÈRE (Toutes les 2 époques) ----------------
        if (epoch + 1) % 2 == 0:
            plot_confusion_matrix(all_val_labels, all_val_preds, outdir, master_ref, filename=f'confusion_matrix_epoch_{epoch+1}.png')
            for mem, d in per_member_metrics.items():
                if d['count'] > 0:
                    member_outdir = os.path.join(outdir, "per_member", mem)
                    os.makedirs(member_outdir, exist_ok=True)
                    plot_confusion_matrix(d['labels'], d['preds'], member_outdir, master_ref, filename=f'confusion_matrix_epoch_{epoch+1}.png')

        

    print(f"Best Val Loss : {best_val_loss:.6f}")

    loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times, per_member_val_losses=val_losses_per_member_history)
    accuracy_figure(len(train_losses), train_acc_history, val_acc_history, outdir, epoch_times, per_member_val_accuracies=val_acc_per_member_history)


    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'train_losses': train_losses, 
        'val_losses': val_losses,
        'train_acc': train_acc_history, 
        'val_acc': val_acc_history,
        'val_losses_per_member_history': dict(val_losses_per_member_history),
        'val_acc_per_member_history': dict(val_acc_per_member_history)
    }
    torch.save(state, f'{outdir}/final_model_CNN_bs{bs}.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_CNN_bs{bs}.pth')

    # ============================================================
    # END OF TRAINING
    # ============================================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training complete, elapsed time: {elapsed_time / 60:.2f} minutes")