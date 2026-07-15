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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# import des dossiers siblings

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# Ajouter le dossier "tools" de vision_transformer au sys.path pour tes imports de dataset etc
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

from tools.visualizations import loss_figure, accuracy_figure, plot_confusion_matrix, loss_acc_first_epoch
from tools.models import get_fast_labels
from tools.datasets import Dataset


# ============================================================
# NOUVEAU : MODÈLE DE RÉGRESSION LOGISTIQUE
# ============================================================
class LogisticRegressionPredictor(nn.Module):
    def __init__(self, sst_shape=(85, 360), slp_shape=(53, 113), in_chans_sst=3, in_chans_slp=0, num_classes=4):
        super().__init__()
        self.sst_size = in_chans_sst * sst_shape[0] * sst_shape[1]
        self.slp_size = in_chans_slp * slp_shape[0] * slp_shape[1]
        self.total_input_size = self.sst_size + self.slp_size
        
        # Une seule couche linéaire projetant vers les logits des N classes
        self.linear = nn.Linear(self.total_input_size, num_classes)

    def forward(self, x_sst, x_slp):
        batch_size = x_sst.size(0)
        
        x_sst_flat = x_sst.view(batch_size, -1)
        
        if self.slp_size > 0:
            x_slp_flat = x_slp.view(batch_size, -1)
            x = torch.cat([x_sst_flat, x_slp_flat], dim=1)
        else:
            x = x_sst_flat
            
        # Renvoie les logits bruts (le Softmax est géré par nn.CrossEntropyLoss)
        return self.linear(x)

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
    parser.add_argument('--nb_epochs', type=int, default=30, help='Nombre d\'époques pour l\'entraînement du Logistic Regression')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours pour les cibles 10 ou 30')
    parser.add_argument('--bs', type=int, default=128, help='Taille de batch pour l\'entraînement')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate pour l\'entraînement du Logistic Regression')


    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95], help='Liste des lags pour SST (ex: --sst_lags_days 30 60 90)')
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[], help='Liste des lags pour SLP (optionnel)')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner pour l\'entraînement')

    parser.add_argument('--metric', type=str, default='mse', choices=['mse', 'correlation','pc1_quantiles','mse_latent'], help='Métrique pour le calcul des labels')    
    parser.add_argument('--master_ref_path', type=str, default="/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/composites_4_regimes/master_ref_generator_89_members_10d_embedding_method_pca/master_reference_global.npz", help='Chemin vers la référence maître')
    parser.add_argument('--projector_path', type=str, default="", help='Chemin vers le projector à utiliser pour calculer les embeddings de référence (optionnel)')
    

    parser.add_argument('--alpha_penalty', type=float, default=1e-5, help='Poids de la pénalité L1 ou L2')
    parser.add_argument('--penalty_type', type=str, choices=['l1', 'l2'], default='l2', help='Type de pénalité à utiliser')
    args = parser.parse_args()

    # Routage dynamique du dossier de sortie (Output Dir)
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/linear_models/"
    elif args.machine == 'jean-zay-work' or args.machine == 'jean-zay-scratch':
        # WORK_uxg=/lustre/fswork/projects/rech/uxg/uca57ub
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/linear_models/" 
    elif args.machine == "mac_local":
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/linear_models/"
    else:
        raise ValueError("Machine argument must be 'hacienda', 'jean-zay-work', 'jean-zay-scratch' or 'mac_local'.")

    # ============================================================
    # GLOBAL CONSTANTS
    # ============================================================


    # train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    # nb_members_train = 1
    # train_members = train_members_87[:nb_members_train]
    # print(f"{nb_members_train} members used for training")

    # val_members = ['1001.001'] # On peut aussi prendre ["1301.010"] ou les deux (attention à la ram)

    # winter_months = [1,2]   
    # months_label = "JF"      
    # sst_lags_days = [35,65,95]   
    # slp_lags_days = []  
    # bs = 128       # ATTENTION: Réduit à 128 (au lieu de 1024) pour éviter le Out Of Memory (OOM) avec des images 53x113 en sortie !
    # lr = 5e-5       # Learning rate
    # dr = 0.2        # Dropout rate
    # nb_epochs = 1               
    # duree_lissage = 30   


    sst_lags_days = args.sst_lags_days
    slp_lags_days = args.slp_lags_days
    bs = args.bs
    lr = args.lr
    winter_months = args.winter_months
    duree_lissage = args.duree_lissage
    nb_epochs = args.nb_epochs
    nb_members_train = args.nb_members_train
    nb_members_val = args.nb_members_val
    metric = args.metric
    patience = 10000       

    penalty_type = args.penalty_type
    alpha_penalty = args.alpha_penalty

    print("Arg Parameters:")
    print(f"  Metric: {metric}", f" SST Lags: {sst_lags_days}", f" SLP Lags: {slp_lags_days}", f" Batch Size: {bs}", f" LR: {lr}",f" Months: {winter_months}", f" Smoothing: {duree_lissage}", f" Epochs: {nb_epochs}", f" Train Members: {nb_members_train}", f" Val Members: {nb_members_val}\n")

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[-nb_members_val:]

    outdir_name = f"LogReg_{penalty_type}_{alpha_penalty}_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_bs{bs}_lr{lr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_members_seed_{args.seed}_{duree_lissage}d_metric_{metric}"
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

    val_set = Dataset(members=val_members, selected_months=winter_months, machine = args.machine,target_type ='map',sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage,roll_sst = True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    training_set = Dataset(members=train_members, selected_months=winter_months, machine = args.machine,target_type ='map',sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage,roll_sst = True)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    # ============================================================
    # MODEL & OPTIMIZER SETUP
    # ============================================================   

    model = LogisticRegressionPredictor(
        num_classes=num_classes,         # 4 régimes  
        in_chans_sst=len(sst_lags_days),  
        in_chans_slp=len(slp_lags_days),
        sst_shape=(85, 360),
        slp_shape=(53, 113)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 2. Dummy forward (Le passage factice indispensable)
    with torch.no_grad():
        dummy_sst = torch.zeros(1, len(sst_lags_days), 85, 360).to(device) if len(sst_lags_days) > 0 else None
        dummy_slp = torch.zeros(1, len(slp_lags_days), 53, 113).to(device) if len(slp_lags_days) > 0 else None
        _ = model(dummy_sst, dummy_slp)

    print("Number of model parameters : ", sum(p.numel() for p in model.parameters()))

    # plus robuste que les autres codes je crois dans le cas update == 1
    if args.update == 1: 
        initial_params = torch.load(f"{outdir}/final_model_LogReg_bs{bs}.pth")
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
        print("Initiate first Logistic Regression training")

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
            # --- Calcul de la Loss Principale ---
            ce_loss = criterion(logits, labels)
            
            # --- Ajout de la Pénalité (L1 ou L2) ---
            if penalty_type == 'l1':
                penalty = torch.norm(model.linear.weight, p=1)
            elif penalty_type == 'l2':
                penalty = torch.sum(model.linear.weight ** 2)
            else:
                penalty = 0.0
                
            loss = ce_loss + alpha_penalty * penalty
            
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
            # ----------------------------------------------------------------------------------

        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)
        train_acc = (running_train_correct / total_train_samples) * 100
        train_acc_history.append(train_acc)

        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}, Training Accuracy: {train_acc:.2f}%')

        # ----- NOUVEAU : Appel de la fonction de visualisation -----
        if epoch == 0:
            loss_acc_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, 
                                 epoch1_batch_accs, epoch1_baseline_accs, outdir)
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
                
                # --- Calcul de la Loss Principale ---
                ce_loss = criterion(logits, labels)
                
                # --- Ajout de la Pénalité (L1 ou L2) ---
                if penalty_type == 'l1':
                    penalty = torch.norm(model.linear.weight, p=1)
                elif penalty_type == 'l2':
                    penalty = torch.sum(model.linear.weight ** 2)
                else:
                    penalty = 0.0
                    
                loss = ce_loss + alpha_penalty * penalty
                
                current_batch_size = X_sst.size(0)
                running_val_loss += loss.item() * current_batch_size
                total_val_samples += current_batch_size

                # Loss INDIVIDUELLE (On ajoute la pénalité à chaque sample pour rester cohérent, 
                # bien que ça n'impacte que les graphes/logs individuels)
                per_sample_losses = (F.cross_entropy(logits, labels, reduction='none') + alpha_penalty * penalty).cpu().numpy()

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
            torch.save(model.state_dict(), f'{outdir}/best_val_LogReg_bs{bs}.pth')
            print(f"Saved best val model at epoch {epoch + 1}")
            
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
            torch.save(state, f'{outdir}/final_model_LogReg_bs{bs}.pth')
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
    torch.save(state, f'{outdir}/final_model_LogReg_bs{bs}.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_LogReg_bs{bs}.pth')

    # ============================================================
    # END OF TRAINING
    # ============================================================
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training complete, elapsed time: {elapsed_time / 60:.2f} minutes")

    # ============================================================
    # 7. EXPLICABILITÉ : VISUALISATION DES POIDS PAR CLASSE
    # ============================================================
    print("\n--- GÉNÉRATION DES CARTES D'EXPLICABILITÉ (LOG-ODDS) ---")
    
    def plot_explainability_weights_classification(model, outdir, sst_lags_days, slp_lags_days, sst_shape=(85, 360), slp_shape=(53, 113)):
        import cartopy.crs as ccrs
        
        extent_sst = [-180, 180, -15, 70] 
        extent_slp = [-100, 40, 20, 70] 
        
        # Récupération des poids : shape = [num_classes, total_input_size]
        weights = model.linear.weight.detach().cpu().numpy()
        num_classes = weights.shape[0]
        
        in_chans_sst = len(sst_lags_days)
        in_chans_slp = len(slp_lags_days)
        sst_size_total = in_chans_sst * sst_shape[0] * sst_shape[1]
        
        # On crée un dossier spécifique pour ranger ces cartes proprement
        exp_dir = os.path.join(outdir, "explainability_maps")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Boucle sur chaque régime (classe)
        for c in range(num_classes):
            print(f"Génération des cartes pour la Classe {c}...")
            
            # Extraction des poids spécifiques à cette classe
            class_weights_flat = weights[c, :]
            
            # Découpage SST
            sst_weights = class_weights_flat[:sst_size_total].reshape(in_chans_sst, sst_shape[0], sst_shape[1])

            # CORRECTION ICI : Calcul du vmax GLOBAL pour toute la SST de cette classe
            global_sst_vmax = np.max(np.abs(sst_weights))
            global_sst_vmin = -global_sst_vmax
            
            # --------------------------------------------------------
            # PLOT SST
            # --------------------------------------------------------
            fig, axes = plt.subplots(
                1, in_chans_sst, 
                figsize=(6 * in_chans_sst, 4), 
                subplot_kw={'projection': ccrs.PlateCarree()},
                facecolor='white'
            )
            if in_chans_sst == 1: axes = [axes]
                
            for i, lag in enumerate(sst_lags_days):
                ax = axes[i]
                ax.set_facecolor('white')
                

                im = ax.imshow(
                    sst_weights[i], 
                    cmap='RdBu_r', 
                    origin='lower', 
                    vmin=global_sst_vmin, 
                    vmax=global_sst_vmax, 
                    transform=ccrs.PlateCarree(), 
                    extent=extent_sst,
                    interpolation='nearest'
                )
                
                ax.set_title(f"Classe {c} | SST Influence - Lag {lag}d", fontsize=12)
                ax.coastlines(resolution='110m', color='black', linewidth=0.8)
                
                fig.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.08)

            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, f"explainability_Class_{c}_SST.png"), dpi=150, facecolor='white')
            plt.close()
            
            # --------------------------------------------------------
            # PLOT SLP (si utilisé)
            # --------------------------------------------------------
            if in_chans_slp > 0:
                slp_weights = class_weights_flat[sst_size_total:].reshape(in_chans_slp, slp_shape[0], slp_shape[1])
                fig, axes = plt.subplots(
                    1, in_chans_slp, 
                    figsize=(6 * in_chans_slp, 4), 
                    subplot_kw={'projection': ccrs.PlateCarree()},
                    facecolor='white'
                )
                if in_chans_slp == 1: axes = [axes]
                    
                # CORRECTION ICI : Calcul du vmax GLOBAL pour toute la SLP de cette classe
                global_slp_vmax = np.max(np.abs(slp_weights[i]))
                global_slp_vmin = -global_slp_vmax
                    
                for i, lag in enumerate(slp_lags_days):
                    ax = axes[i]
                    ax.set_facecolor('white')
                    
 

                    im = ax.imshow(
                        slp_weights[i], 
                        cmap='RdBu_r', 
                        origin='lower', 
                        vmin=global_slp_vmin, 
                        vmax=global_slp_vmax, 
                        transform=ccrs.PlateCarree(), 
                        extent=extent_slp,
                        interpolation='nearest'
                    )
                    
                    ax.set_title(f"Classe {c} | SLP Influence - Lag {lag}d", fontsize=12)
                    ax.coastlines(resolution='110m', color='black', linewidth=0.8)
                    
                    fig.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.08)

                plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f"explainability_Class_{c}_SLP.png"), dpi=150, facecolor='white')
                plt.close()
                
        print(f"-> Toutes les cartes d'explicabilité ont été sauvegardées dans {exp_dir}")

    # Appel de la fonction à la fin de l'entraînement
    plot_explainability_weights_classification(model, outdir, sst_lags_days, slp_lags_days)