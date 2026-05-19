import torch
import xarray as xr
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import time
import argparse
import os
import cftime
import random
from datetime import datetime
import copy
from sklearn.metrics import accuracy_score
from collections import defaultdict
import joblib

# import des dossiers siblings

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import os
import sys
from pathlib import Path



from tools.visualizations import loss_figure, accuracy_figure, plot_confusion_matrix, loss_acc_first_epoch
from tools.models import ViT_Classifier_Multimodal, get_fast_labels
from tools.datasets import Dataset


# ============================================================
# DEVICE & ARGUMENTS CONFIGURATION & OUTPUT DIRECTORY SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', type=int, required=True, help='Loading of previous parameters (1 for yes, 0 for no)')
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'], help='Machine sur laquelle le code tourne (adapte les chemins automatiquement)')
    
    parser.add_argument('--nb_members_train', type=int, default=10, help='Nombre de membres à utiliser pour l\'entraînement (sur les 89 disponibles)')
    parser.add_argument('--nb_members_val', type=int, default=5, help='Nombre de membres à utiliser pour la validation (sur les 89 disponibles)')
    parser.add_argument('--seed', type=int, default=42, help='Seed pour le mélange inter membres')
    parser.add_argument('--nb_epochs', type=int, default=30, help='Nombre d\'époques pour l\'entraînement du ViT')
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours pour les cibles 10 ou 30')
    parser.add_argument('--bs', type=int, default=128, help='Taille de batch pour l\'entraînement')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate pour l\'entraînement du ViT')
    parser.add_argument('--dr', type=float, default=0.22, help='Dropout rate pour le ViT')

    parser.add_argument('--sst_lags_days', type=int, nargs='*', default=[35, 65, 95], help='Liste des lags pour SST (ex: --sst_lags_days 30 60 90)')
    parser.add_argument('--slp_lags_days', type=int, nargs='*', default=[], help='Liste des lags pour SLP (optionnel)')
    parser.add_argument('--winter_months', type=int, nargs='+', default=[1, 2], help='Mois target à sélectionner pour l\'entraînement (ex: --winter_months 1 2 pour janvier et février)')

    parser.add_argument('--metric', type=str, default='mse', choices=['mse', 'correlation','pc1_quantiles','mse_latent'], help='Métrique pour le calcul des labels : mse, correlation marchent sur les pixels donc sans projector, pc1_quantiles et mse_latent marchent sur les embeddings donc avec projector.')    
    parser.add_argument('--master_ref_path', type=str, default="/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/composites_4_regimes/master_ref_generator_89_members_10d_embedding_method_pca/master_reference_global.npz", help='Chemin vers la référence maître pour le calcul des labels (fichier .npz). Contient les cartes de références mais aussi les embeddings de références (à condition de charger le bon projector)')
    parser.add_argument('--projector_path', type=str, default="/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/pca_slp/IPCA_latent128_NDJF_87members_normalizeFalse_duree_lissage10/best_pca_model.joblib", help='Chemin vers le projector à utiliser pour calculer les embeddings de référence, vae ou pca')

    args = parser.parse_args()


    # Routage dynamique du dossier de sortie (Output Dir)
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/vision_transformer/vit_classifier/"
    elif args.machine == 'jean-zay-work' or args.machine == 'jean-zay-scratch':
        # WORK_uxg=/lustre/fswork/projects/rech/uxg/uca57ub
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_classifier/" 
    elif args.machine == "mac_local":
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/vision_transformer/vit_classifier/"
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

    # train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    # nb_members_train = 2
    # train_members = train_members_87[:nb_members_train]
    # print(f"{nb_members_train} members used for training")
    # val_members = ['1001.001'] # On peut aussi prendre ["1301.010"] ou les deux (attention à la ram)
    # winter_months = [1,2]   
    # months_label = "J"      
    # sst_lags_days = [35,65,95]   
    # slp_lags_days = []  
    # bs = 128       # ATTENTION: Réduit à 128 (au lieu de 1024) pour éviter le Out Of Memory (OOM) avec des images 53x113 en sortie !
    # lr = 5e-5       # Learning rate
    # dr = 0.2        # Dropout rate
    # nb_epochs = 25     
    # duree_lissage = 30      
     
    patience = 10000   
           
    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    rng = random.Random(args.seed)
    rng.shuffle(all_members)

    train_members = all_members[:nb_members_train]
    val_members = all_members[-nb_members_val:]

    outdir_name = f"ViT_classifier_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_bs{bs}_lr{lr}_dr{dr}_months_{'_'.join(map(str, winter_months))}_train{nb_members_train}_val_{nb_members_val}_members_seed_{args.seed}_{duree_lissage}d_metric_{metric}"
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)
    print(f"Dossier de sauvegarde : {outdir}")

    # On charge directement le fichier .npz placé à côté
    master_ref_path = args.master_ref_path
    master_ref = dict(np.load(master_ref_path)) 
    print("Référence maître chargée depuis le dossier local !")

    # On charge le projector s'il est donné (sert si metric = 'mse_latent' ou 'pc1_quantiles')
    if args.projector_path:
        projector = joblib.load(args.projector_path)
        print("Projector chargé depuis le dossier local !")
    else:
        projector = None

# Déduction du nombre de classes selon la métrique choisie
    if args.metric == 'pc1_quantiles':
        # S'il y a 9 limites (bins), ça fait 8 quantiles (classes)
        num_classes = len(master_ref['pc1_bins']) - 1
    elif args.metric in ['correlation', 'mse']:
        # On compte les régimes physiques
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
            # On cherche la clé "_count" pour le cluster i+1
            for k, v in master_ref.items():
                if f"regime_{i+1}_" in k and k.endswith("_count"):
                    class_counts[i] = float(v)
                    break
    elif args.metric == 'pc1_quantiles':
        class_counts = np.ones(num_classes) # Distribution uniforme pour les quantiles

    class_probs = class_counts / class_counts.sum()
    # Logits naifs pour la CrossEntropy (log des probabilités)
    baseline_logits = torch.tensor(np.log(class_probs), dtype=torch.float32).unsqueeze(0).to(device)
    majority_class = int(np.argmax(class_counts))
    print(f"-> Baseline (Majority Class) : Classe {majority_class} avec une probabilité a priori de {class_probs[majority_class]*100:.1f}%")

    # ============================================================
    # MODEL & OPTIMIZER SETUP
    # ============================================================

    model = ViT_Classifier_Multimodal(
        num_classes=num_classes,         # Nombre de classes détecté automatiquement
        sst_size=(85, 360),    
        slp_size=(53, 113),    # Nouvelle taille de sortie !
        patch_size_sst=(5, 10),    
        patch_size_slp=(5, 10),    
        in_chans_sst=len(sst_lags_days),  
        in_chans_slp=len(slp_lags_days),
        embed_dim=128,         # la même pour la slp et pour la sst
        depth=4,                    
        num_heads=4,           
        dr=dr                  
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("Number of model parameters : ", sum(p.numel() for p in model.parameters()))

    if args.update == 1: 
        initial_params = torch.load(f"{outdir}/final_model_ViT_bs{bs}.pth")
        model.load_state_dict(initial_params['state_dict'])
        optimizer.load_state_dict(initial_params['optimizer'])
        train_losses = initial_params['train_losses']
        val_losses = initial_params['val_losses']
        best_val_loss = np.min(val_losses)
        print("Model state update")
    elif args.update == 0: 
        train_losses, val_losses = [], []
        best_val_loss = float('inf') 
        print("Initiate first model training")
    else :
        raise ValueError("Update parameter must be equal to 0 or 1.")

    # ============================================================
    # TRAINING & EVALUATION LOOP
    # ============================================================
    start_time = time.time() 
    best_model_state = None 
    patience_counter = 0
    epoch_times = []

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")

    val_set = Dataset(members=val_members, selected_months=winter_months, machine = args.machine,target_type ='map',sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True)

    training_set = Dataset(members=train_members, selected_months=winter_months, machine = args.machine,target_type ='map',sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days, duree_lissage=duree_lissage)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True)

    # On récupère le chemin du dossier où se trouve main.py
    # current_dir = Path(__file__).resolve().parent Inutile je crois



    val_losses_per_member_history = defaultdict(list)
    val_acc_per_member_history = defaultdict(list) # <-- NOUVEAU
    val_acc_history = []
    train_acc_history = [] # <--- NOUVEAU

    # NOUVEAU : Suivi de la loss et accuracy par batch pour l'époque 1
    epoch1_batch_losses = []
    epoch1_baseline_losses = []
    epoch1_batch_accs = []
    epoch1_baseline_accs = []

    for epoch in range(nb_epochs):
        # ---------------- TRAINING ----------------
        model.train()
        running_train_loss = 0.0
        running_train_correct = 0 # <--- NOUVEAU : compteur de bonnes réponses
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
            
            # --- NOUVEAU : Calcul de l'accuracy sur ce batch ---
            preds = torch.argmax(logits, dim=1)
            running_train_correct += (preds == labels).sum().item()

            # --- Loss ---
            # CrossEntropy prend des logits (Batch, 4) et des vrais labels (Batch,)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # CORRECTION : Vraie moyenne
            current_batch_size = X_sst.size(0)
            running_train_loss += loss.item() * current_batch_size
            total_train_samples += current_batch_size

            # ----- NOUVEAU : Enregistrer les métriques du batch et la baseline (Époque 1) -----
            if epoch == 0:
                epoch1_batch_losses.append(loss.item())
                batch_acc = (preds == labels).sum().item() / current_batch_size * 100.0
                epoch1_batch_accs.append(batch_acc)
                
                with torch.no_grad():
                    # 1. Baseline Loss (Prédiction constante de la distribution globale)
                    b_logits = baseline_logits.expand(current_batch_size, -1)
                    b_loss = criterion(b_logits, labels).item()
                    epoch1_baseline_losses.append(b_loss)
                    
                    # 2. Baseline Accuracy (Prédiction constante de la classe majoritaire)
                    b_preds = torch.full_like(labels, majority_class)
                    b_acc = (b_preds == labels).sum().item() / current_batch_size * 100.0
                    epoch1_baseline_accs.append(b_acc)
            # ----------------------------------------------------------------------------------

        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)

        # --- NOUVEAU : Enregistrement et affichage ---
        train_acc = (running_train_correct / total_train_samples) * 100
        train_acc_history.append(train_acc)

        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}, Training Accuracy: {train_acc:.2f}%')

        # ----- NOUVEAU : Appel de la fonction de visualisation -----
        if epoch == 0:
            loss_acc_first_epoch(epoch1_batch_losses, epoch1_baseline_losses, 
                                 epoch1_batch_accs, epoch1_baseline_accs, outdir)

    # ---------------- VALIDATION ----------------
        model.eval()
        time_list = []
        running_val_loss = 0.0
        total_val_samples = 0

        # --- NOUVEAU : On stocke aussi les preds et labels pour la matrice de confusion par membre ---
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
                # 1. Loss globale du batch
                loss = criterion(logits, labels)
                current_batch_size = X_sst.size(0)
                running_val_loss += loss.item() * current_batch_size
                total_val_samples += current_batch_size

                # 2. Loss INDIVIDUELLE pour la séparation par membre (reduction='none')
                per_sample_losses = F.cross_entropy(logits, labels, reduction='none').cpu().numpy()

                # --- Répartition par membre ---
                members_list = [m if isinstance(m, str) else m.item().decode() if hasattr(m, 'item') else str(m) for m in members]

                # --- Calcul des prédictions (on prend la classe avec le score le plus haut) ---
                preds = torch.argmax(logits, dim=1)
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # --- NOUVEAU : Remplissage des listes pour la matrice par membre ---
                for i, mem in enumerate(members_list):
                    per_member_metrics[mem]['loss_sum'] += float(per_sample_losses[i])
                    per_member_metrics[mem]['count'] += 1
                    per_member_metrics[mem]['preds'].append(preds_np[i])
                    per_member_metrics[mem]['labels'].append(labels_np[i])


                
                # Stockage pour les métriques
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                time_list.extend(dates) 

        val_loss = running_val_loss / total_val_samples
        val_losses.append(val_loss)

        # Enregistrement des loss moyennes de chaque membre
        for mem, d in per_member_metrics.items():
            avg_loss = d['loss_sum'] / d['count'] if d['count'] > 0 else float('nan')
            val_losses_per_member_history[mem].append(avg_loss)

            # 2. L'Accuracy (NOUVEAU)
            if d['count'] > 0:
                mem_acc = accuracy_score(d['labels'], d['preds']) * 100
            else:
                mem_acc = float('nan')
            val_acc_per_member_history[mem].append(mem_acc)
        
        val_accuracy = accuracy_score(all_val_labels, all_val_preds) * 100
        val_acc_history.append(val_accuracy) # <-- NOUVEAU : on la sauvegarde !

        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        
        print(f'Epoch {epoch + 1} Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.2f}% | Elapsed Time: {current_time_min:.2f} min')

# ---------------- EARLY STOPPING & SAVING ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"Saved best val model at epoch {epoch + 1}")
            
            # Matrice Globale
            plot_confusion_matrix(all_val_labels, all_val_preds, outdir, master_ref, filename='best_confusion_matrix.png')
            
            # --- NOUVEAU : Matrices par membre (Best Model) ---
            for mem, d in per_member_metrics.items():
                if d['count'] > 0:
                    member_outdir = os.path.join(outdir, "per_member", mem)
                    os.makedirs(member_outdir, exist_ok=True)
                    plot_confusion_matrix(d['labels'], d['preds'], member_outdir, master_ref, filename='best_confusion_matrix.png')

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
                break

        if epoch % 1 == 0:
            state = {'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    'train_losses': train_losses, 'val_losses': val_losses,
                    'train_acc': train_acc_history, 'val_acc': val_acc_history}
            torch.save(state, f'{outdir}/final_model_ViT_bs{bs}.pth')
            loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times,per_member_val_losses=val_losses_per_member_history)
            accuracy_figure(len(train_losses), train_acc_history, val_acc_history, outdir, epoch_times, per_member_val_accuracies=val_acc_per_member_history)     
            
        # ---------------- SAUVEGARDE RÉGULIÈRE (Toutes les 2 époques) ----------------
        if (epoch + 1) % 2 == 0:
            # Matrice Globale
            plot_confusion_matrix(all_val_labels, all_val_preds, outdir, master_ref, filename=f'confusion_matrix_epoch_{epoch+1}.png')
            
            # --- NOUVEAU : Matrices par membre (Toutes les 2 époques) ---
            for mem, d in per_member_metrics.items():
                if d['count'] > 0:
                    member_outdir = os.path.join(outdir, "per_member", mem)
                    os.makedirs(member_outdir, exist_ok=True)
                    plot_confusion_matrix(d['labels'], d['preds'], member_outdir, master_ref, filename=f'confusion_matrix_epoch_{epoch+1}.png')

    print(f"Best Val Loss : {best_val_loss:.6f}")

    loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times,per_member_val_losses=val_losses_per_member_history)
    accuracy_figure(len(train_losses), train_acc_history, val_acc_history, outdir, epoch_times, per_member_val_accuracies=val_acc_per_member_history) # <--- NOUVEAU

    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 
             'train_losses': train_losses, 'val_losses': val_losses,
             'train_acc': train_acc_history, 'val_acc': val_acc_history}
    torch.save(state, f'{outdir}/final_model_ViT_bs{bs}.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f'{outdir}/best_val_ViT_bs{bs}.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training complete, elapsed time: {elapsed_time / 60:.2f} minutes")