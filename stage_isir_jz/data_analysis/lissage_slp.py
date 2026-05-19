import os
import glob
import xarray as xr

# ==========================================
# CONFIGURATION
# ==========================================
duree_moyennage = 30
# Remets ici le bon chemin si ce n'est pas le fswork (ex: "/data/home/cwetzel/with_warming/SLP/")
base_dir = '/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/'

# ==========================================
# RÉCUPÉRATION DES FICHIERS
# ==========================================
# Cette fois-ci, on cible EXPLICITEMENT les fichiers déjà lissés sur 10 jours
files = sorted(glob.glob(os.path.join(base_dir, "PSL_anom_LE2-*_10d.nc")))

print(f"Début du traitement. {len(files)} fichiers '_10d' trouvés dans {base_dir}")

# ==========================================
# BOUCLE DE LISSAGE ET SAUVEGARDE
# ==========================================
for f in files:
    print(f"\nTraitement de : {os.path.basename(f)}")
    
    # 1. Ouverture du fichier 10d
    ds = xr.open_dataset(f)
    
    # 2. Lissage supplémentaire
    ds_ma = ds.rolling(time=duree_moyennage, center=True, min_periods=1).mean()
    
    # 3. Création du nouveau nom (on remplace _10d.nc par _30d.nc)
    f_out = f.replace("_10d.nc", f"_{duree_moyennage}d.nc")
    
    # 4. Sauvegarde
    ds_ma.to_netcdf(f_out)
    
    # 5. Libération de la mémoire
    ds.close()
    ds_ma.close()
    
    print(f" -> Sauvegardé sous : {os.path.basename(f_out)}")

print("\n🎉 Tous les fichiers ont été retraités et sauvegardés !")