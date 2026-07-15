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
from tools.visualizations import loss_figure, plot_and_save_maps
from tools.models import ViT_SST_to_SLP, ViT_Decoded_SLP_Multimodal
from tools.datasets import Dataset

# ============================================================
# DEVICE & ARGUMENTS CONFIGURATION & OUTPUT DIRECTORY SETUP
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', type=int, required=True, help='Loading of previous parameters (1 for yes, 0 for no)')
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'], help='Machine sur laquelle le code tourne (adapte les chemins automatiquement)')
    args = parser.parse_args()

    # Routage dynamique du dossier de sortie (Output Dir)
    if args.machine == 'hacienda':
        base_home = "/home/moysan/stage_isir_jz/vision_transformer/vit_with_full_slp/"
    elif args.machine == 'jean-zay-work' or args.machine == 'jean-zay-scratch':
        # WORK_uxg=/lustre/fswork/projects/rech/uxg/uca57ub
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/vision_transformer/vit_with_full_slp/" 
    elif args.machine == "mac_local":
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/vision_transformer/vit_with_full_slp/"
    else:
        raise ValueError("Machine argument must be 'hacienda', 'jean-zay-work', 'jean-zay-scratch' or 'mac_local'.")

    # ============================================================
    # GLOBAL CONSTANTS
    # ============================================================


    train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    nb_members_train = 60
    train_members = train_members_87[:nb_members_train]
    print(f"{nb_members_train} members used for training")

    val_members = ['1001.001'] # On peut aussi prendre ["1301.010"] ou les deux (attention à la ram)

    winter_months = [1, 2]   
    months_label = "JF"      
    sst_lags_days = [3]   
    slp_lags_days = []  
    bs = 128       # ATTENTION: Réduit à 128 (au lieu de 1024) pour éviter le Out Of Memory (OOM) avec des images 53x113 en sortie !
    lr = 5e-5       # Learning rate
    dr = 0.2        # Dropout rate
    nb_epochs = 25           
    patience = 10000
    duree_lissage = 0              

    outdir_name = f"ViT_EncDec_lags_{'_'.join(map(str, sst_lags_days))}_sst_{'_'.join(map(str, slp_lags_days))}_slp_bs{bs}_lr{lr}_dr{dr}_{months_label}_train{nb_members_train}members_duree_lissage{duree_lissage}"
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)
    print(f"Dossier de sauvegarde : {outdir}")

    # ============================================================
    # MODEL & OPTIMIZER SETUP
    # ============================================================

    model = ViT_Decoded_SLP_Multimodal(
        sst_size=(85, 360),    
        slp_size=(53, 113),    # Nouvelle taille de sortie !
        patch_size_sst=(5, 10),    
        patch_size_slp=(5, 10),    
        in_chans_sst=len(sst_lags_days),  
        in_chans_slp=len(slp_lags_days),
        embed_dim=128,         # la même pour la slp et pour la sst
        enc_depth=4,           
        dec_depth=4,           
        num_heads=4,           
        dr=dr                  
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

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

    for epoch in range(nb_epochs):
        # ---------------- TRAINING ----------------
        model.train()
        running_train_loss = 0.0
        for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(trainloader):
            if batch_idx % 30 == 0:
                print(f" {100 * batch_idx / len(trainloader):.1f}% training complete", end='\r')
            optimizer.zero_grad()

            X_sst = X_sst.to(device, non_blocking=True)
            y_target = y_target.to(device, non_blocking=True)
            X_slp = X_slp.to(device, non_blocking=True)

            outputs = model(X_sst, X_slp)
            loss_value = criterion(outputs, y_target)
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()

        train_loss = running_train_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f'Epoch {epoch + 1} Training Loss: {train_loss:.8f}')

        # ---------------- VALIDATION ----------------
        model.eval()
        time_list, slp_true_list, slp_pred_list = [], [], []
        running_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (X_sst, X_slp, y_target, y_map, dates, members) in enumerate(valloader):
                if batch_idx % 30 == 0:
                    print(f" {100 * batch_idx / len(valloader):.1f}% val complete", end='\r')
                
                X_sst = X_sst.to(device, non_blocking=True)
                X_slp = X_slp.to(device, non_blocking=True)
                y_target = y_target.to(device, non_blocking=True)
                
                outputs = model(X_sst, X_slp)
                loss_value = criterion(outputs, y_target)
                running_val_loss += loss_value.item()

                time_list.extend(dates) # extend car 'dates' est un tuple de strings
                slp_true_list.append(y_map.numpy())
                slp_pred_list.append(outputs.cpu().numpy()) #la fonction de plot gère le squeeze. 

        val_loss = running_val_loss / len(valloader)
        val_losses.append(val_loss)
        

        # CALCUL ET SAUVEGARDE DU TEMPS DE L'ÉPOQUE (en minutes)

        current_time_min = (time.time() - start_time) / 60.0
        epoch_times.append(current_time_min)
        print(f'Epoch {epoch + 1} Val Loss: {val_loss:.6f}', f'- Elapsed Time: {current_time_min:.2f} minutes')

        # ---------------- EARLY STOPPING ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"Saved best val model at epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (patience {patience} reached)")
                break

        if epoch % 1 == 0:
            state = {'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    'train_losses': train_losses, 'val_losses': val_losses}
            torch.save(state, f'{outdir}/final_model_ViT_bs{bs}.pth')
            loss_figure(len(train_losses), train_losses, val_losses, outdir,epoch_times)
            print(f"Saved checkpoint at epoch {epoch + 1}")
    # Tous les 5 epochs, on sort une image pour voir l'évolution de nos 3 dates fixes
        if (epoch + 1) % 1 == 0:
            plot_and_save_maps(slp_true_list, slp_pred_list, time_list, outdir, epoch=(epoch + 1),duree_moyennage=1)
        

    print(f"Best Val Loss : {best_val_loss:.6f}")

    loss_figure(len(train_losses), train_losses, val_losses, outdir, epoch_times)


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