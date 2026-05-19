# customized lags pour la SLP inclus mais attention en input, c'est bien la carte entière de la SLP pas juste la PC (à voir si ce serait intéressant de mettre la PC mais bon)

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
from datetime import datetime
import copy


# import des dossiers siblings
import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from tools.visualizations import loss_figure, quantile_loss, plot_and_save_maps_3_columns
from tools.models import ViT_Multimodal # <-- Attention à bien importer le nouveau modèle
from tools.datasets import Dataset

# ============================================================
# DEVICE & OUTPUT DIRECTORY CONFIGURATION
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# GLOBAL CONSTANTS & LAGS
# ============================================================
bs = 1024       # Batch size
lr = 5e-5       # Learning rate for Adam optimizer
dr = 0.2        # Dropout rate for regularization

# --- DÉFINITION DES LAGS ABSOLUS ---
# Tu peux maintenant contrôler exactement ce que le modèle voit
sst_lags_days = [35, 65, 95] 
slp_lags_days = [15] 
# Note : on utilise plus "lag = 35", tout est géré par ces listes.
# Pour le nom du dossier de sortie, on peut utiliser le premier lag de SST comme référence
ref_lag = sst_lags_days[0] 

train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

nb_members_train = 1
train_members = train_members_87[:nb_members_train]
print(f"{nb_members_train} members used for training")

val_members = ['1001.001',"1301.010"]

winter_months = [1, 2]   
months_label = "JF"      

nb_epochs = 20           
patience = 10000         

file_path = "/data/moysan/data/"
path_sst = "/data/moysan/data/SST/"
path_slp = "/data/moysan/data/SLP/"

parser = argparse.ArgumentParser()
parser.add_argument('--update', type=int, required=True, help='Loading of previous parameters')
args = parser.parse_args()
update = args.update

base_home = "/home/moysan/stage_isir/vision_transformer/vit_with_pc1_customized_lags/"
outdir_name = f'ViT_Multi_lag{ref_lag}d_bs{bs}_lr{lr}_dr{dr}_{months_label}_train{nb_members_train}'
outdir = os.path.join(base_home, outdir_name)
os.makedirs(outdir, exist_ok=True)

# ============================================================
# MODEL & OPTIMIZER SETUP
# ============================================================
quantiles = torch.tensor([0.05 + i * 0.1 for i in range(10)]).to(device)

# --- INITIALISATION DU ViT MULTIMODAL ---
model = ViT_Multimodal(
    sst_size=(85, 360), patch_size_sst=(5, 10), in_chans_sst=len(sst_lags_days),
    slp_size=(53, 113), patch_size_slp=(5, 5),  in_chans_slp=len(slp_lags_days),
    nb_out=len(quantiles), 
    embed_dim=64, depth=4, num_heads=4, dr=dr
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = lambda preds, targets: quantile_loss(preds, targets, quantiles)

print("Number of model parameters : ", sum(p.numel() for p in model.parameters()))

if update == 1: 
    initial_params = torch.load(f"{outdir}/final_model_ViT_bs{bs}.pth")
    model.load_state_dict(initial_params['state_dict'])
    optimizer.load_state_dict(initial_params['optimizer'])
    train_losses = initial_params['train_losses']
    val_losses = initial_params['val_losses']
    best_val_loss = np.min(val_losses)
    print("Model state updated")
elif update == 0: 
    train_losses, val_losses = [], []
    best_val_loss = float('inf') 
    print("Initiate first model training")
else :
    raise ValueError(f" Update parameter must be equal to 0 or 1.")

# ============================================================
# DATALOADERS
# ============================================================
start_time = time.time() 
epoch_times = []
best_model_state = None 
patience_counter = 0

# Ajout des arguments sst_lags_days et slp_lags_days
val_set = Dataset(members=val_members, selected_months=winter_months, 
                  file_path_SST=path_sst, file_path_SLP=path_slp,
                  sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days)
valloader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)

training_set = Dataset(members=train_members, selected_months=winter_months, 
                       file_path_SST=path_sst, file_path_SLP=path_slp,
                       sst_lags_days=sst_lags_days, slp_lags_days=slp_lags_days)
trainloader = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)

# Charge le fichier contenant les EOFs
ds_eofs = xr.open_dataset("/home/moysan/EOFS/planche_eof1_20_membres_scaling_2.png")
# Calcule l'EOF moyen sur la dimension 'member' pour avoir le motif universel
eof1_universal = ds_eofs["EOF1_SLP"].mean(dim='member').values # Format NumPy Array

# ============================================================
# TRAINING & EVALUATION LOOP
# ============================================================
for epoch in range(nb_epochs):
    # ---------------- TRAINING ----------------
    model.train()
    running_train_loss = 0.0
    
    # On déballe maintenant 5 variables (dont inputs_slp et member_id)
    for batch_idx, (inputs_sst, inputs_slp, targets, _, _,_) in enumerate(trainloader):
        if batch_idx % 30 == 0:
            print(f" {100 * batch_idx / len(trainloader):.1f}% training complete", end='\r')
        
        optimizer.zero_grad()

        inputs_sst = inputs_sst.to(device, non_blocking=True)
        inputs_slp = inputs_slp.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Le modèle prend DEUX arguments
        outputs = model(inputs_sst, inputs_slp)
        
        loss_value = criterion(outputs, targets)
        loss_value.backward()
        optimizer.step()
        running_train_loss += loss_value.item()

    train_loss = running_train_loss / len(trainloader)
    train_losses.append(train_loss)
    print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}')

    # ---------------- VALIDATION ----------------
    model.eval()
    time_list, pcs_true_list, pcs_pred_list = [], [], []
    slp_true_maps_list = []  # <-- 1. AJOUT CRUCIAL : Initialisation de la liste
    running_val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs_sst, inputs_slp, target_pc1, target_map, dates, _) in enumerate(valloader):
            if batch_idx % 30 == 0:
                print(f" {100 * batch_idx / len(valloader):.1f}% val complete", end='\r')
            
            inputs_sst = inputs_sst.to(device, non_blocking=True)
            inputs_slp = inputs_slp.to(device, non_blocking=True)
            target_pc1 = target_pc1.to(device, non_blocking=True)
            
            outputs = model(inputs_sst, inputs_slp)
            loss_value = criterion(outputs, target_pc1)
            running_val_loss += loss_value.item()

            # <-- 2. CORRECTION : extend au lieu de append pour les dates
            time_list.extend(dates) 
            
            # <-- 3. AJOUT CRUCIAL : Sauvegarde des cartes complètes pour le plot
            slp_true_maps_list.append(target_map.cpu().numpy()) 
            
            pcs_true_list.append(target_pc1.cpu().numpy())
            pcs_pred_list.append(outputs.cpu().numpy())

    val_loss = running_val_loss / len(valloader)
    val_losses.append(val_loss)
    print(f'Epoch {epoch + 1} Val Loss: {val_loss:.2f}')

    # ---------------- EARLY STOPPING ----------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        #save_predictions(pcs_true_list, pcs_pred_list, time_list, quantiles, f"{outdir}/best_val_PCs.nc")
        print(f"Saved best val model at epoch {epoch}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
            break

    current_time_min = (time.time() - start_time) / 60.0
    epoch_times.append(current_time_min)
    if epoch % 5 == 0:
        state = {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 
                 'train_losses': train_losses, 'val_losses': val_losses}
        torch.save(state, f'{outdir}/final_model_ViT_bs{bs}.pth')
        #save_predictions(pcs_true_list, pcs_pred_list, time_list, quantiles, f"{outdir}/final_PCs.nc") # je ne sais pas exactement ce que fait cette fonction ni si c'est nécessaire, ni ça prend de la place. temporairement j'enlève. 
        loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times)
        print(f"Saved last model at epoch {epoch}")
    if epoch % 5 == 0:
        plot_and_save_maps_3_columns(slp_true_maps_list, pcs_true_list, pcs_pred_list, time_list, eof1_universal, outdir, epoch=(epoch + 1))


print(f"Best Val Loss : {best_val_loss:.2f}")

loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times)
#save_predictions(pcs_true_list, pcs_pred_list, time_list, quantiles, f"{outdir}/final_PCs.nc")

state = {'state_dict': model.state_dict(),
         'optimizer': optimizer.state_dict(), 
         'train_losses': train_losses, 'val_losses': val_losses}
torch.save(state, f'{outdir}/final_model_ViT_bs{bs}.pth')

if best_model_state:
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f'{outdir}/best_val_ViT_bs{bs}.pth')

# ============================================================
# END OF TRAINING
# ============================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training complete, elapsed time: {elapsed_time / 60:.2f} minutes")