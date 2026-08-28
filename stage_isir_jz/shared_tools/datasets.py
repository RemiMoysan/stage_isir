import os
import torch
import xarray as xr
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta


import torch
import xarray as xr
import numpy as np
import os
from datetime import timedelta

import torch
import xarray as xr
import pandas as pd
import numpy as np
import os
from datetime import timedelta

# ============================================================
# Version finale du dataset : 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, members, selected_months, 
                 machine='jean-zay-work', target_type='map', num_pcs=10,
                 sst_lags_days=[35, 65, 95], slp_lags_days=[15], 
                 augment=False, noise_std=0.05, custom_base_dir=None, duree_lissage=10, roll_sst=False,sst_std =0.707, slp_std=596):
        """
        Args:
            machine (str): 'jean-zay-work' ou 'jean-zay-scratch' ou 'hacienda', adapte les chemins automatiquement, ou 'custom' pour fournir un chemin personnalisé vers un dossier contennant les sous-dossiers 'SST' et 'SLP'.
            target_type (str): 'map' (renvoie la carte SLP) ou 'pc' (renvoie la PC1).
        """
        self.members = members
        self.augment = augment
        self.noise_std = noise_std
        self.sst_lags_days = sst_lags_days
        self.slp_lags_days = slp_lags_days
        self.num_pcs = num_pcs
        self.duree_lissage = duree_lissage
        self.roll_sst = roll_sst
        assert target_type in ['map', 'pc'], "target_type doit être 'map' ou 'pc'"
        self.target_type = target_type
        self.sst_std = sst_std
        self.slp_std = slp_std

        # ==========================================
        # 1. GESTION DES CHEMINS 
        # ==========================================
        if custom_base_dir is not None:
            base_dir = custom_base_dir
        elif machine == 'jean-zay-work':
            base_dir = '/lustre/fswork/projects/rech/uxg/uca57ub/data/'
        elif machine == 'jean-zay-scratch':
            base_dir = '/lustre/fsn1/projects/rech/uxg/uca57ub/data/'
        elif machine == "mac_local":
            base_dir = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/"
        elif machine == 'hacienda':
            base_dir = '/data/moysan/data/' 
        else:
            raise ValueError("Machine inconnue. Choisissez 'jean-zay-work', 'jean-zay-scratch', 'hacienda' ou fournissez un chemin personnalisé.")

        path_SST = os.path.join(base_dir, 'SST/')
        path_SLP = os.path.join(base_dir, 'SLP/')

        # ==========================================
        # 2. OUVERTURE UNIQUE DES FICHIERS (VITESSE ++)
        # ==========================================
        # On stocke les variables directement dans des dictionnaires indexés par les membres
        self.data_SST = {}
        self.data_SLP = {}
        self.data_PC = {}

        # print("Ouverture des fichiers NetCDF en cours...")
        for m in members:
            self.data_SST[m] = xr.open_dataset(f"{path_SST}SST_anom_LE2-{m}_T_regrid.nc")["SST"]
            if self.duree_lissage != 0:
                self.data_SLP[m] = xr.open_dataset(f"{path_SLP}PSL_anom_LE2-{m}_{self.duree_lissage}d.nc")["PSL"]
            else:
                self.data_SLP[m] = xr.open_dataset(f"{path_SLP}PSL_anom_LE2-{m}.nc")["PSL"]
            if self.target_type == 'pc':
                self.data_PC[m] = xr.open_dataset(f"{path_SLP}CESM_NDJF_10pcs_scaled-{m}_10d.nc")["pcs"]

        # ==========================================
        # 3. GESTION DES DATES CIBLES
        # ==========================================
        all_time = self.data_SLP[members[0]].time
        valid_dates = all_time.sel(time=all_time.dt.month.isin(selected_months))
        years = valid_dates.dt.year
        valid_mask = (years > years.min()) & (years < years.max() - 1)
        valid_dates = valid_dates.sel(time=valid_mask)
        # première date : 1er ou 2 janvier 1851
        # sécurité si on regarde des lags futurs, dernière date 1er janvier 2015
        self.list_dates = valid_dates.values.tolist()

        # print(f"Dataset initialisé sur {machine.upper()}.")
        # print(f"Target d'entraînement : {self.target_type.upper()} (Map 2D toujours incluse pour les plots)")
        # print(f"Membres : {len(members)} | Jours valides par membre : {len(self.list_dates)}")
    
    def augment_sst(self, X, noise_std=0.05):
        noise = noise_std * torch.randn_like(X)
        return X + noise

    def __len__(self):
        return len(self.members) * len(self.list_dates)

    def __getitem__(self, index):
        member_idx = index // len(self.list_dates)
        member_id = self.members[member_idx]
        t_target = self.list_dates[index % len(self.list_dates)]

        dates_sst = [t_target - timedelta(days=d) for d in self.sst_lags_days]
        dates_slp = [t_target - timedelta(days=d) for d in self.slp_lags_days]

        sst_std, slp_std = self.sst_std, self.slp_std


        # 1. Extraction SST Inputs (sans ouvrir le fichier, on tape dans le dict)
        sst = self.data_SST[member_id].sel(time=dates_sst, lat=slice(-15, 70))
        X_sst = torch.nan_to_num(torch.tensor(np.array(sst.data)), nan=0.0).float() / sst_std
        if self.roll_sst: # facultatif pour le ViT (pas de problème de bord)
            X_sst = torch.roll(X_sst, shifts=180, dims=-1)

        if self.augment:
            X_sst = self.augment_sst(X_sst,self.noise_std)
        
        # 2. Extraction SLP Inputs et MAP SLP cible
        slp_input = self.data_SLP[member_id].sel(time=dates_slp)
        X_slp = torch.nan_to_num(torch.tensor(np.array(slp_input.data)), nan=0.0).float() / slp_std
        
        slp_target_map = self.data_SLP[member_id].sel(time=t_target)
        y_map = torch.tensor(np.array(slp_target_map.data)).float() / slp_std

        # 3. Définition de y_target : map ou PC1
        if self.target_type == 'map':
            y_target = y_map 
        else:
            pc_target = self.data_PC[member_id].sel(time=t_target).isel(mode=slice(0, self.num_pcs))
            y_target = torch.tensor(np.array(pc_target.data)).float()

        # On renvoie 6 éléments : Inputs, Target pour la loss, Map pour les plots, Date, Membre
        return X_sst, X_slp, y_target.unsqueeze(0), y_map.unsqueeze(0), t_target.strftime('%Y-%m-%d'), member_id
    
#============================================================

# Dataset pour la réduction monthly

class Dataset_mensuel(torch.utils.data.Dataset):
    def __init__(self, members, selected_months, 
                 machine='jean-zay-work', target_type='map', num_pcs=10,
                 sst_lags_months=[1, 2, 3], slp_lags_months=[1], # Remplacé par "months"
                 augment=False, noise_std=0.05, custom_base_dir=None, roll_sst=False, 
                 sst_std=0.707, slp_std=596):
        """
        Args:
            machine (str): Machine utilisée pour les chemins.
            target_type (str): 'map' ou 'pc'.
            sst_lags_months (list): Lags en mois pour la SST.
            slp_lags_months (list): Lags en mois pour la SLP.
        """
        self.members = members
        self.augment = augment
        self.noise_std = noise_std
        self.sst_lags_months = sst_lags_months
        self.slp_lags_months = slp_lags_months
        self.num_pcs = num_pcs
        self.roll_sst = roll_sst
        assert target_type in ['map', 'pc'], "target_type doit être 'map' ou 'pc'"
        self.target_type = target_type
        self.sst_std = sst_std
        self.slp_std = slp_std

        # ==========================================
        # 1. GESTION DES CHEMINS 
        # ==========================================
        if custom_base_dir is not None:
            base_dir = custom_base_dir
        elif machine == 'jean-zay-work':
            base_dir = '/lustre/fswork/projects/rech/uxg/uca57ub/data/'
        elif machine == 'jean-zay-scratch':
            base_dir = '/lustre/fsn1/projects/rech/uxg/uca57ub/data/'
        elif machine == "mac_local":
            base_dir = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/"
        elif machine == 'hacienda':
            base_dir = '/data/moysan/data/' 
        else:
            raise ValueError("Machine inconnue.")

        path_SST = os.path.join(base_dir, 'SST/')
        path_SLP = os.path.join(base_dir, 'SLP/')

        # ==========================================
        # 2. OUVERTURE UNIQUE DES FICHIERS
        # ==========================================
        self.data_SST = {}
        self.data_SLP = {}
        self.data_PC = {}

        for m in members:
            # On charge explicitement les fichiers mensualisés
            self.data_SST[m] = xr.open_dataset(f"{path_SST}SST_anom_LE2-{m}_T_regrid_1mo.nc")["SST"]
            self.data_SLP[m] = xr.open_dataset(f"{path_SLP}PSL_anom_LE2-{m}_1mo.nc")["PSL"]
            
            if self.target_type == 'pc':
                # NON UTILISÉ (on repasse dans le modèle pca à chaque entraînement, plus modulable)
                self.data_PC[m] = xr.open_dataset(f"{path_SLP}CESM_NDJF_10pcs_scaled-{m}_1mo.nc")["pcs"]

        # ==========================================
        # 3. GESTION DES DATES CIBLES
        # ==========================================
        all_time = self.data_SLP[members[0]].time
        valid_dates = all_time.sel(time=all_time.dt.month.isin(selected_months))
        years = valid_dates.dt.year
        
        valid_mask = (years > years.min()) & (years < years.max() + 1) # au final on enlève rien pour la dernière année car on utilise que les codes dans le passé. 
        valid_dates = valid_dates.sel(time=valid_mask)
        
        self.list_dates = valid_dates.values.tolist()

        # print(f"Dataset_mensuel initialisé sur {machine.upper()}, Membres : {len(members)} | Mois valides par membre : {len(self.list_dates)}")
    
    def augment_sst(self, X, noise_std=0.05):
        noise = noise_std * torch.randn_like(X)
        return X + noise

    def __len__(self):
        return len(self.members) * len(self.list_dates)

    def __getitem__(self, index):
        member_idx = index // len(self.list_dates)
        member_id = self.members[member_idx]
        t_target = self.list_dates[index % len(self.list_dates)]

        # ==========================================
        # NOUVEAU : Décalage mathématique (Compatible cftime NoLeap)
        # ==========================================
        dates_sst = []
        for m in self.sst_lags_months:
            y_shift = (t_target.month - m - 1) // 12
            new_month = (t_target.month - m - 1) % 12 + 1
            dates_sst.append(t_target.replace(year=t_target.year + y_shift, month=new_month))
            
        dates_slp = []
        for m in self.slp_lags_months:
            y_shift = (t_target.month - m - 1) // 12
            new_month = (t_target.month - m - 1) % 12 + 1
            dates_slp.append(t_target.replace(year=t_target.year + y_shift, month=new_month))

        sst_std, slp_std = self.sst_std, self.slp_std

        # 1. Extraction SST Inputs
        sst = self.data_SST[member_id].sel(time=dates_sst, lat=slice(-15, 70))
        X_sst = torch.nan_to_num(torch.tensor(np.array(sst.data)), nan=0.0).float() / sst_std
        if self.roll_sst:
            X_sst = torch.roll(X_sst, shifts=180, dims=-1)

        if self.augment:
            X_sst = self.augment_sst(X_sst, noise_std=self.noise_std)
        
        # 2. Extraction SLP Inputs et MAP SLP cible
        slp_input = self.data_SLP[member_id].sel(time=dates_slp)
        X_slp = torch.nan_to_num(torch.tensor(np.array(slp_input.data)), nan=0.0).float() / slp_std
        
        slp_target_map = self.data_SLP[member_id].sel(time=t_target)
        y_map = torch.tensor(np.array(slp_target_map.data)).float() / slp_std

        # 3. Définition de y_target
        if self.target_type == 'map':
            y_target = y_map 
        else:
            pc_target = self.data_PC[member_id].sel(time=t_target).isel(mode=slice(0, self.num_pcs))
            y_target = torch.tensor(np.array(pc_target.data)).float()

        return X_sst, X_slp, y_target.unsqueeze(0), y_map.unsqueeze(0), t_target.strftime('%Y-%m-%d'), member_id


