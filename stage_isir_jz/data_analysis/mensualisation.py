import os
import glob
import xarray as xr

# ==========================================
# CONFIGURATION
# ==========================================
# On cible le dossier SST
base_dir = '/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/'
suffixe_out = "_1mo.nc"

# ==========================================
# RÉCUPÉRATION DES FICHIERS
# ==========================================
# On cherche les fichiers qui commencent par SST_anom
tous_les_fichiers = glob.glob(os.path.join(base_dir, "SST_anom_LE2-*.nc"))

# On s'assure d'exclure les fichiers déjà mensualisés si tu relances le script
files = [f for f in tous_les_fichiers if not f.endswith("_1mo.nc")]

print(f"Fichiers SST bruts trouvés : {len(files)}")
print(f"Début du traitement dans {base_dir}")

# ==========================================
# BOUCLE DE SOUS-ÉCHANTILLONNAGE MENSUEL
# ==========================================
for f in files:
    print(f"\nTraitement de : {os.path.basename(f)}")
    
    # 1. Ouverture du fichier
    ds = xr.open_dataset(f)
    
    # 2. Moyenne mensuelle stricte
    ds_monthly = ds.resample(time='1MS').mean()
    
    # 3. Création du nouveau nom
    # Le replace(".nc", "_1mo.nc") transformera par exemple :
    # "SST_anom_LE2-1001.001_T_regrid.nc" en "SST_anom_LE2-1001.001_T_regrid_1mo.nc"
    f_out = f.replace(".nc", suffixe_out)
    
    # 4. Sauvegarde
    ds_monthly.to_netcdf(f_out)
    
    # 5. Libération de la mémoire
    ds.close()
    ds_monthly.close()
    
    print(f" -> Sauvegardé sous : {os.path.basename(f_out)}")

print("\n🎉 Tous les fichiers SST ont été convertis en moyennes mensuelles !")