import os
import torch
import xarray as xr
import pandas as pd
import numpy as np
from datetime import timedelta



## Version supposé plus rapide avce isel ...

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

# censé être plus rapide mais bof
#Eventuellement à utiliser à la place de Dataset normal mais je ne suis pas 100% convaincu. 

class Dataset_faster2(torch.utils.data.Dataset):
    def __init__(self, members, selected_months, 
                 machine='jean-zay-work', target_type='map', num_pcs=10,
                 sst_lags_days=[35, 65, 95], slp_lags_days=[15], 
                 augment=False, custom_base_dir=None, duree_lissage=10):
        
        self.members = members
        self.augment = augment
        self.sst_lags_days = sst_lags_days
        self.slp_lags_days = slp_lags_days
        self.num_pcs = num_pcs
        self.duree_lissage = duree_lissage
        self.target_type = target_type

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
            raise ValueError("Machine inconnue. Choisissez 'jean-zay-work', 'jean-zay-scratch', 'hacienda'.")

        path_SST = os.path.join(base_dir, 'SST/')
        path_SLP = os.path.join(base_dir, 'SLP/')

        self.data_SST = {}
        self.data_SLP = {}
        self.data_PC = {}

        self.date_to_idx_sst = {}
        self.date_to_idx_slp = {}
        self.date_to_idx_pc = {}

        print("Ouverture des pointeurs NetCDF (Lazy Loading) et création des index...")
        for m in members:
            ds_sst = xr.open_dataset(f"{path_SST}SST_anom_LE2-{m}_T_regrid.nc")
            ds_slp = xr.open_dataset(f"{path_SLP}PSL_anom_LE2-{m}_{self.duree_lissage}d.nc")
            
            # OPTIMISATION : On applique les slices spatiaux et PC dès l'ouverture !
            self.data_SST[m] = ds_sst["SST"].sel(lat=slice(-15, 70))
            self.data_SLP[m] = ds_slp["PSL"]

            if self.target_type == 'pc':
                ds_pc = xr.open_dataset(f"{path_SLP}CESM_NDJF_10pcs_scaled-{m}_10d.nc")
                self.data_PC[m] = ds_pc["pcs"].isel(mode=slice(0, self.num_pcs))

            # Création des dictionnaires sur le premier membre
            if m == members[0]:
                times_sst = ds_sst.time.values
                times_slp = ds_slp.time.values

                self.date_to_idx_slp = {t: i for i, t in enumerate(times_slp)}
                self.date_to_idx_sst = {t: i for i, t in enumerate(times_sst)}

                if self.target_type == 'pc':
                    times_pc = ds_pc.time.values
                    self.date_to_idx_pc = {t: i for i, t in enumerate(times_pc)}

        # Filtrage des dates valides
        all_time_da = self.data_SLP[members[0]].time
        valid_mask = all_time_da.dt.month.isin(selected_months) & (all_time_da.dt.year > all_time_da.dt.year.min() ) & (all_time_da.dt.year < all_time_da.dt.year.max() -1)
        self.list_dates = all_time_da.sel(time=valid_mask.values).values.tolist()

        print(f"Membres : {len(members)} | Jours valides par membre : {len(self.list_dates)}")
    
    def augment_sst(self, X, noise_std=0.05):
        noise = noise_std * torch.randn_like(X)
        return X + noise

    def __len__(self):
        return len(self.members) * len(self.list_dates)

    def __getitem__(self, index):
        member_idx = index // len(self.list_dates)
        member_id = self.members[member_idx]
        t_target = self.list_dates[index % len(self.list_dates)]

        
        sst_std, slp_std = 0.707, 596
        
        # 1. Extraction SST
        sst_arrays = []
        for lag in self.sst_lags_days:
            lag_date = t_target - timedelta(days=lag)
            idx = self.date_to_idx_sst.get(lag_date)
            
            # SÉCURITÉ : Si la date de lag est hors limites, on met des zéros
            if idx is not None:
                # Lecture ultra-pure : le slice a déjà été fait !
                arr = self.data_SST[member_id].isel(time=idx).values
            else:
                arr = np.zeros((85, 360)) # Taille de la SST après le slice latitudinal
            sst_arrays.append(arr)
            
        X_sst = torch.nan_to_num(torch.tensor(np.array(sst_arrays)), nan=0.0).float() / sst_std
        X_sst = torch.roll(X_sst, shifts=180, dims=-1)
        
        if self.augment:
            X_sst = self.augment_sst(X_sst)
        
        # 2. Extraction SLP
        slp_arrays = []
        for lag in self.slp_lags_days:
            lag_date = t_target - timedelta(days=lag)
            idx = self.date_to_idx_slp.get(lag_date)
            
            if idx is not None:
                arr = self.data_SLP[member_id].isel(time=idx).values
            else:
                arr = np.zeros((53, 113)) # Taille de la SLP
            slp_arrays.append(arr)
            
        if slp_arrays:
            X_slp = torch.nan_to_num(torch.tensor(np.array(slp_arrays)), nan=0.0).float() / slp_std
        else:
            X_slp = torch.empty(0, 53, 113)
        
        # 3. Extraction MAP SLP cible
        idx_target = self.date_to_idx_slp.get(t_target)
        # Ici on suppose que la target t=0 est toujours dans le dataset
        slp_target_map = self.data_SLP[member_id].isel(time=idx_target).values
        y_map = torch.tensor(np.array(slp_target_map)).float() / slp_std

        # 4. Target Finale
        if self.target_type == 'map':
            y_target = y_map 
        else:
            idx_pc = self.date_to_idx_pc.get(t_target)
            # Lecture pure : le slice num_pcs a déjà été fait !
            pc_target = self.data_PC[member_id].isel(time=idx_pc).values
            y_target = torch.tensor(np.array(pc_target)).float()

        return X_sst, X_slp, y_target.unsqueeze(0), y_map.unsqueeze(0), t_target.strftime('%Y-%m-%d'), member_id

# ============================================================
# Version finale du dataset : 
# - On n'utilise plus pd.Timedelta qui était plus lent que datetime.timedelta, de plus le test du 29 février était inutile (la soustraction comprend le caractère Noleap, mais il est vrai que la perte de temps liée à ce test semble négligeable).
# - Le rolling de la longitude de la SST semble facultatif pour les architectures transformers, mais si on souhaite le garder, la méthode torch.roll semble plus rapide que la méthode xarray (assign_coords + sortby). 
# - Attention aux unsqueeze et aux squeezes. 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, members, selected_months, 
                 machine='jean-zay-work', target_type='map', num_pcs=10,
                 sst_lags_days=[35, 65, 95], slp_lags_days=[15], 
                 augment=False, custom_base_dir=None, duree_lissage=10, roll_sst=False):
        """
        Args:
            machine (str): 'jean-zay-work' ou 'jean-zay-scratch' ou 'hacienda', adapte les chemins automatiquement, ou 'custom' pour fournir un chemin personnalisé vers un dossier contennant les sous-dossiers 'SST' et 'SLP'.
            target_type (str): 'map' (renvoie la carte SLP) ou 'pc' (renvoie la PC1).
        """
        self.members = members
        self.augment = augment
        self.sst_lags_days = sst_lags_days
        self.slp_lags_days = slp_lags_days
        self.num_pcs = num_pcs
        self.duree_lissage = duree_lissage
        self.roll_sst = roll_sst
        assert target_type in ['map', 'pc'], "target_type doit être 'map' ou 'pc'"
        self.target_type = target_type

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

        print("Ouverture des fichiers NetCDF en cours...")
        for m in members:
            self.data_SST[m] = xr.open_dataset(f"{path_SST}SST_anom_LE2-{m}_T_regrid.nc")["SST"]
            self.data_SLP[m] = xr.open_dataset(f"{path_SLP}PSL_anom_LE2-{m}_{self.duree_lissage}d.nc")["PSL"]
            if self.target_type == 'pc':
                self.data_PC[m] = xr.open_dataset(f"{path_SLP}CESM_NDJF_10pcs_scaled-{m}_10d.nc")["pcs"]

        # ==========================================
        # 3. GESTION DES DATES CIBLES
        # ==========================================
        # On utilise le dataset déjà ouvert du premier membre
        all_time = self.data_SLP[members[0]].time
        valid_dates = all_time.sel(time=all_time.dt.month.isin(selected_months))
        years = valid_dates.dt.year
        # CORRECTION : On combine les deux conditions (passé et futur) en une seule ligne
        valid_mask = (years > years.min()) & (years < years.max() - 1)
        valid_dates = valid_dates.sel(time=valid_mask)
         # première date : 1er ou 2 janvier 1851
        # sécurité si on regarde des lags futurs, dernière date 1er janvier 2015
        self.list_dates = valid_dates.values.tolist()

        print(f"Dataset initialisé sur {machine.upper()}.")
        print(f"Target d'entraînement : {self.target_type.upper()} (Map 2D toujours incluse pour les plots)")
        print(f"Membres : {len(members)} | Jours valides par membre : {len(self.list_dates)}")
    
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

        sst_std, slp_std = 0.707, 596


        # 1. Extraction SST Inputs (sans ouvrir le fichier, on tape dans le dict)
        sst = self.data_SST[member_id].sel(time=dates_sst, lat=slice(-15, 70))
        X_sst = torch.nan_to_num(torch.tensor(np.array(sst.data)), nan=0.0).float() / sst_std
        if self.roll_sst: # facultatif pour le ViT (pas de problème de bord)
            X_sst = torch.roll(X_sst, shifts=180, dims=-1)

        if self.augment:
            X_sst = self.augment_sst(X_sst)
        
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

class old_Dataset(torch.utils.data.Dataset):
    "awfully slow (metadata were opened for each item ?)"
    def __init__(self, members, selected_months, 
                 machine='jean-zay-work', target_type='map', num_pcs=10,
                 sst_lags_days=[35, 65, 95], slp_lags_days=[15], 
                 augment=False, custom_base_dir=None):
        """
        Args:
            machine (str): 'jean-zay-work' ou 'jean-zay-scratch' ou 'hacienda', adapte les chemins automatiquement, ou 'custom' pour fournir un chemin personnalisé vers un dossier contennant les sous-dossiers 'SST' et 'SLP'.
            target_type (str): 'map' (renvoie la carte SLP) ou 'pc' (renvoie la PC1).
        """
        self.members = members
        self.augment = augment
        self.sst_lags_days = sst_lags_days
        self.slp_lags_days = slp_lags_days
        self.num_pcs = num_pcs
        
        assert target_type in ['map', 'pc'], "target_type doit être 'map' ou 'pc'"
        self.target_type = target_type

        # ==========================================
        # 1. GESTION DES CHEMINS (Hacienda vs Jean Zay)
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

        self.files_SST = {m: f"{path_SST}SST_anom_LE2-{m}_T_regrid.nc" for m in members}
        self.files_SLP = {m: f"{path_SLP}PSL_anom_LE2-{m}_10d.nc" for m in members}
        
        if self.target_type == 'pc':
            self.files_PC = {m: f"{path_SLP}CESM_NDJF_10pcs_scaled-{m}_10d.nc" for m in members}

        # ==========================================
        # 2. GESTION DES DATES CIBLES
        # ==========================================
        with xr.open_dataset(self.files_SLP[members[0]]) as ds_temp:
            all_time = ds_temp.time
            valid_dates = all_time.sel(time=all_time.dt.month.isin(selected_months))
            years = valid_dates.dt.year
            valid_dates = valid_dates.sel(time=years > years.min())
            self.list_dates = valid_dates.values.tolist()

        print(f"Dataset initialisé sur {machine.upper()}.")
        print(f"Target d'entraînement : {self.target_type.upper()} (Map 2D toujours incluse pour les plots)")
        print(f"Membres : {len(members)} | Jours valides par membre : {len(self.list_dates)}")
    
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

        sst_std, slp_std = 0.707, 596

        try:
            # 1. Extraction SST Inputs
            with xr.open_dataset(self.files_SST[member_id]) as ds_sst:
                sst = ds_sst["SST"].sel(time=dates_sst, lat=slice(-15, 70))
                
                X_sst = torch.nan_to_num(torch.tensor(np.array(sst.data)), nan=0.0).float() / sst_std
                # X_sst = torch.roll(X_sst, shifts=180, dims=-1) # facultatif pour le ViT (pas de problème de bord)
                if self.augment:
                    X_sst = self.augment_sst(X_sst)
            
            # 2. Extraction SLP Inputs et MAP SLP cible (même si target PC, on veut quand même la vraie carte pour les graphiques)
            with xr.open_dataset(self.files_SLP[member_id]) as ds_slp:
                slp_input = ds_slp["PSL"].sel(time=dates_slp)
                X_slp = torch.nan_to_num(torch.tensor(np.array(slp_input.data)), nan=0.0).float() / slp_std
                
                # On extrait toujours la carte spatiale pour les graphes
                slp_target_map = ds_slp["PSL"].sel(time=t_target)
                y_map = torch.tensor(np.array(slp_target_map.data)).float() / slp_std

            # 3. Définition de y_target : map ou PC1 selon le mode choisi
            if self.target_type == 'map':
                y_target = y_map 
            else:
                with xr.open_dataset(self.files_PC[member_id]) as ds_pc:
                    pc_target = ds_pc["pcs"].sel(time=t_target).isel(mode=slice(0, self.num_pcs))
                    y_target = torch.tensor(np.array(pc_target.data)).float()

        except KeyError:
            # SÉCURITÉ : Tout est homogène. On met (1, 53, 113) pour les DEUX cartes.
            y_target_fallback = torch.zeros(1, 53, 113) if self.target_type == 'map' else torch.zeros(1, self.num_pcs)
            y_map_fallback = torch.zeros(1, 53, 113) # <-- Modifié ici
            
            return (torch.zeros(len(self.sst_lags_days), 85, 360), 
                    torch.zeros(len(self.slp_lags_days), 53, 113), 
                    y_target_fallback, 
                    y_map_fallback, 
                    str(t_target), 
                    member_id)

        # On renvoie 6 éléments : Inputs, Target pour la loss, Map pour les plots, Date, Membre
        # Attention les outputs ont toujours la dimension channel.
        return X_sst, X_slp, y_target.unsqueeze(0), y_map.unsqueeze(0), t_target.strftime('%Y-%m-%d'), member_id

# ============================================================
# Les anciennes versions de datasets.


# ============================================================
# Dataset : Cartes de SST et SLP, avec lag temporel (1, 2, 3 mois)
# ============================================================
class Dataset_SST_SLP(torch.utils.data.Dataset):
    """
    Dataset pour get la SST et la SLP totale, lag de ref puis 1 et 2 mois avant. Normalisation par l'écart type moyen des pixels.
    """
    def __init__(self, members, lag, selected_months, file_path_SST, file_path_SLP, augment=False):
        self.lag = lag
        self.members = members
        self.augment = augment

        # 1. Prepare the base datasets to get the structure
        self.SST = xr.open_dataset(file_path_SST + f'SST_anom_LE2-{members[0]}_T_regrid.nc')
        self.SLP = xr.open_dataset(file_path_SLP + f'PSL_anom_LE2-{members[0]}_10d.nc')
        
        self.SST = self.SST.drop_vars(["SST", "dayofyear"], errors="ignore")
        # Remplacer "PSL" par le vrai nom de la variable dans ton fichier si différent
        self.SLP = self.SLP.drop_vars(["PSL"], errors="ignore") 

        count = 0
        for i in members:
            # 2. Add a new variable per member
            self.SST[count] = xr.open_dataset(file_path_SST + f'SST_anom_LE2-{i}_T_regrid.nc')["SST"]
            # Plus de .sel(mode=0) car on veut la carte complète
            self.SLP[count] = xr.open_dataset(file_path_SLP + f'PSL_anom_LE2-{i}_10d.nc')["PSL"] 
            count += 1
        
        # 3. Shift the SST longitude (Vérifie si ta SLP a besoin de la même chose !) je pense qu'il n'y a pas besoin
        # self.SST = self.SST.assign_coords(lon=((((self.SST).lon + 180) % 360) - 180)).sortby("lon")
        
        # Crop dates made only on SLP data, since SST data are extracted to match the SLP dates
        self.SLP = self.SLP.sel(time=self.SLP['time'].dt.month.isin(selected_months))
        years = self.SLP.time.dt.year
        self.SLP = self.SLP.sel(time=years > years.min())
        self.list_dates = self.SLP.time.values.tolist()

        print("Dataset created for members", members, "and months ", selected_months, "for the period", 
              self.SLP.time.dt.year[0].values, "-", self.SLP.time.dt.year[-1].values)
        print(f"Target (SLP) shape per day: {self.SLP[0].isel(time=0).shape}") # Devrait afficher (53, 113)
        
    def augment_sst(self, X, noise_std=0.05):
        """Add small Gaussian noise to SST inputs for data augmentation."""
        noise = noise_std * torch.randn_like(X)
        return X + noise
    
    def subtract_days(self, date_atm, days):
        """Computes the date correponding to the given lag"""
        result = date_atm - pd.Timedelta(days, "d")
        if result.month == 2 and result.day == 29:
            result -= pd.Timedelta(days=1)
        return result
    
    def __len__(self):
        return len(self.members) * len(self.SLP.time)

    def dates(self):
        return self.SLP.time

    def __getitem__(self, index):
        """
        For a given index, extract SST images (lagged by 1, 2, 3 months)
        and the target SLP map for that date.
        """
        # For a given index, compute the corresponding member and date
        member = index // len(self.SLP.time)
        t_Atm = self.list_dates[index % len(self.SLP.time)]

        # Time deltas for 1, 2, and 3 months before target date
        tdSST_last = self.subtract_days(t_Atm, self.lag)
        tdSST_1 = self.subtract_days(t_Atm, self.lag + 30)
        tdSST_2 = self.subtract_days(t_Atm, self.lag + 60)
        sst_std, slp_std = 0.707, 596

        try:
            # Extract 3 lagged SST maps
            sst = self.SST[member].sel(
                time=[tdSST_last, tdSST_1, tdSST_2],
                lat=slice(-15, 70)  # Focus on mid-latitudes
            ) 
            
            # Extract the SLP target map
            slp = self.SLP[member].sel(time=t_Atm)
            
            # Transform to PyTorch tensors and handle NaNs
            X = torch.nan_to_num(torch.tensor(np.array(sst.data)), nan=0)/sst_std
            
            # y est maintenant une image 2D (53, 113)
            # On remplace aussi les NaNs par des zéros sur la SLP (important pour les continents/bords)
            y = torch.nan_to_num(torch.tensor(np.array(slp.data)), nan=0)/slp_std
            
        except KeyError as e:
            print(f"KeyError occurred for timestamp {t_Atm}: {e}")
            # En cas d'erreur, on renvoie des tenseurs vides de la bonne taille pour éviter le crash
            X = torch.zeros((3, 85, 360))
            y = torch.zeros((53, 113))

        return X.float(), y.unsqueeze(0).float(), t_Atm.strftime('%Y-%m-%d')
    
# ============================================================
# Dataset : Version avec les PCs quand la PC est une target 
# On choisit les lags qu'on veut y compris pour la SLP 
# ============================================================

class Dataset_SST_SLP_PC(torch.utils.data.Dataset):
    def __init__(self, members, selected_months, file_path_SST, file_path_SLP, 
                 sst_lags_days=[35, 65, 95], slp_lags_days=[15], augment=False):
        
        self.members = members
        self.augment = augment
        self.sst_lags_days = sst_lags_days
        self.slp_lags_days = slp_lags_days

        # ==========================================
        # 1. INITIALISATION DES DATASETS
        # ==========================================
        # On a maintenant 3 variables internes au lieu de 2 !
        self.SST = xr.open_dataset(f"{file_path_SST}SST_anom_LE2-{members[0]}_T_regrid.nc")
        
        # SLP_INPUT: Les cartes complètes pour l'encodeur
        self.SLP_INPUT = xr.open_dataset(f"{file_path_SLP}PSL_anom_LE2-{members[0]}_10d.nc")
        
        # SLP_TARGET: Le fichier avec les PCs pour la prédiction
        self.SLP_TARGET = xr.open_dataset(f"{file_path_SLP}CESM_NDJF_10pcs_scaled-{members[0]}_10d.nc").sel(mode=0)
        
        self.SST = self.SST.drop_vars(["SST", "dayofyear"], errors="ignore")
        self.SLP_INPUT = self.SLP_INPUT.drop_vars(["PSL", "dayofyear"], errors="ignore")
        self.SLP_TARGET = self.SLP_TARGET.drop_vars(["pcs"], errors="ignore")

        for count, i in enumerate(members):
            self.SST[count] = xr.open_dataset(f"{file_path_SST}SST_anom_LE2-{i}_T_regrid.nc")["SST"]
            self.SLP_INPUT[count] = xr.open_dataset(f"{file_path_SLP}PSL_anom_LE2-{i}_10d.nc")["PSL"]
            self.SLP_TARGET[count] = xr.open_dataset(f"{file_path_SLP}CESM_NDJF_10pcs_scaled-{i}_10d.nc")["pcs"].sel(mode=0)
        
        # Recadrage spatial de la SST
        self.SST = self.SST.assign_coords(lon=((((self.SST).lon + 180) % 360) - 180)).sortby("lon")
        
        # ==========================================
        # 2. GESTION DES DATES CIBLES (TARGET)
        # ==========================================
        # On se base sur le calendrier du fichier TARGET (qui ne contient déjà que NDJF en théorie)
        all_target_dates = self.SLP_TARGET.time
        
        # Double sécurité pour s'assurer de ne garder que les mois voulus
        valid_dates = all_target_dates.sel(time=all_target_dates.dt.month.isin(selected_months))
        years = valid_dates.dt.year
        valid_dates = valid_dates.sel(time=years > years.min())
        
        self.list_dates = valid_dates.values.tolist()

        print(f"Dataset multimodal créé pour {len(members)} membres.")
        print(f"Lags absolus SST : {self.sst_lags_days} jours.")
        print(f"Lags absolus SLP : {self.slp_lags_days} jours.")
        print(f"Période cible : {selected_months} ({len(self.list_dates)} jours valides par membre).")
        
    def subtract_days(self, date_atm, days):
        result = date_atm - pd.Timedelta(days, "d")
        if result.month == 2 and result.day == 29:
            result -= pd.Timedelta(days=1)
        return result
    
    def augment_sst(self, X, noise_std=0.05):
        noise = noise_std * torch.randn_like(X)
        return X + noise

    def __len__(self):
        return len(self.members) * len(self.list_dates)

    def dates(self):
        return self.list_dates

    def __getitem__(self, index):
        member_idx = index // len(self.list_dates)
        member_id = self.members[member_idx]
        t_target = self.list_dates[index % len(self.list_dates)]

        # --- Calcul exact des dates pour la SST et la SLP ---
        dates_sst = [self.subtract_days(t_target, d) for d in self.sst_lags_days]
        dates_slp = [self.subtract_days(t_target, d) for d in self.slp_lags_days]

        sst_std, slp_std = 0.707, 596

        try:
            # 1. Extraction SST Inputs (Cartes)
            sst = self.SST[member_idx].sel(time=dates_sst, lat=slice(-15, 70)) 
            X_sst = torch.nan_to_num(torch.tensor(np.array(sst.data)), nan=0).float() / sst_std
            if self.augment:
                X_sst = self.augment_sst(X_sst)
            
            # 2. Extraction SLP Inputs (Cartes complètes)
            slp_input = self.SLP_INPUT[member_idx].sel(time=dates_slp) 
            X_slp = torch.nan_to_num(torch.tensor(np.array(slp_input.data)), nan=0).float() / slp_std

            # 3. Extraction SLP Target (La PC1 unique !)
            slp_target_pc1 = self.SLP_TARGET[member_idx].sel(time=t_target)
            y_slp_pc1 = torch.tensor(np.array(slp_target_pc1.data)).float() 

            # 4. NOUVEAU : Extraction SLP Target MAP (La carte 2D pour les graphiques)
            slp_target_map = self.SLP_INPUT[member_idx].sel(time=t_target) 
            y_slp_map = torch.tensor(np.array(slp_target_map.data)).float() / slp_std

        except KeyError as e:
            # En cas de manque, on renvoie des zéros pour TOUTES les variables
            return (torch.zeros(len(self.sst_lags_days), 85, 360), 
                    torch.zeros(len(self.slp_lags_days), 53, 113), 
                    torch.zeros(1),        # 3. La cible PC1 vide
                    torch.zeros(53, 113),  # 4. La cible Carte vide
                    str(t_target), 
                    member_id)

        # On renvoie 6 éléments au lieu de 5 !
        return X_sst, X_slp, y_slp_pc1, y_slp_map, t_target.strftime('%Y-%m-%d'), member_id
    
