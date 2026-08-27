import os
import argparse
import time
from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image
import glob
import pandas as pd

# ==========================================
# 1. FONCTION UTILITAIRE DE CHEMINS
# ==========================================
def get_file_paths(path_slp, path_sst, mem):
    """
    Récupère les fichiers sans lissage (duree_lissage = 0) en données journalières.
    """
    file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}.nc')
    file_sst = os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid.nc')
    return file_slp, file_sst

# ==========================================
# 2. CARTES QUOTIDIENNES SUR 1 MOIS DONNÉ
# ==========================================
def plot_daily_maps_for_month(mem, year, month, path_slp, path_sst, output_dir):
    """
    Sauvegarde une image JPG par jour pour un mois et une année donnés.
    """
    print(f"\n=== Génération des cartes journalières : Membre {mem} | {year}-{month:02d} ===")
    file_slp, file_sst = get_file_paths(path_slp, path_sst, mem)
    
    if not os.path.exists(file_slp) or not os.path.exists(file_sst):
        print(f"❌ Fichiers introuvables pour le membre {mem}")
        return

    out_subfolder = os.path.join(output_dir, f"daily_maps_{mem}_{year}_{month:02d}")
    os.makedirs(out_subfolder, exist_ok=True)

    with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
        # Alignement longitude SST [-180, 180] et sélection latitude
        ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15, 70))
        
        # Filtrage sur l'année et le mois cibles
        time_cond_slp = (ds_slp['time'].dt.year == year) & (ds_slp['time'].dt.month == month)
        time_cond_sst = (ds_sst['time'].dt.year == year) & (ds_sst['time'].dt.month == month)
        
        slp_month = ds_slp['PSL'].sel(time=time_cond_slp) # On reste en Pa
        sst_month = ds_sst['SST'].sel(time=time_cond_sst) # En °C / K

        dates = slp_month.time.values
        print(f" -> {len(dates)} jours trouvés. Sauvegarde en cours...")

        for t in dates:
            date_str = t.strftime('%Y-%m-%d')
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={'width_ratios': [1, 1.5]})
            
            # --- Carte SLP ---
            ax_slp = axes[0]
            extent_slp = [-100, 40, 20, 70]
            im_slp = ax_slp.imshow(
                slp_month.sel(time=t).values, transform=ccrs.PlateCarree(),
                cmap='RdBu_r', origin='lower', extent=extent_slp, vmin=-3000, vmax=3000
            )
            ax_slp.set_extent(extent_slp, crs=ccrs.PlateCarree())
            ax_slp.coastlines(color='black', linewidth=0.8)
            ax_slp.set_title(f"SLP Anomaly - {date_str}", fontweight='bold')
            cbar_slp = fig.colorbar(im_slp, ax=ax_slp, fraction=0.046, pad=0.04, shrink=0.6)
            cbar_slp.set_label('Pa')

            # --- Carte SST ---
            ax_sst = axes[1]
            extent_sst = [-180, 180, -15, 70]
            im_sst = ax_sst.imshow(
                sst_month.sel(time=t).values, transform=ccrs.PlateCarree(),
                cmap='RdBu_r', origin='lower', extent=extent_sst, vmin=-3, vmax=3
            )
            ax_sst.set_extent(extent_sst, crs=ccrs.PlateCarree())
            ax_sst.coastlines(color='black', linewidth=0.8)
            ax_sst.set_title(f"SST Anomaly - {date_str}", fontweight='bold')
            cbar_sst = fig.colorbar(im_sst, ax=ax_sst, fraction=0.046, pad=0.04, shrink=0.6)
            cbar_sst.set_label('K')

            fig.tight_layout()
            
            # MODIFICATIONS ICI : Extension .jpg et réduction légère du DPI pour gagner de la place
            save_path = os.path.join(out_subfolder, f"map_{date_str}.jpg")
            # Ajout de format='jpg' et passage à dpi=120 (largement suffisant pour le PDF)
            fig.savefig(save_path, dpi=120, bbox_inches='tight', format='jpg')
            plt.close(fig)

    # Appel automatique du créateur de GIF avec le bon dossier cible
    gif_name = f"animation_{mem}_{year}_{month:02d}.gif"
    create_gif_from_folder(out_subfolder, os.path.join(output_dir, gif_name))
    print(f"✅ Cartes journalières sauvegardées dans : {out_subfolder}")

# ==========================================
# 3. ÉCART-TYPE EN HIVER (NDJF)
# ==========================================
def compute_winter_std(member_ids, winter_months, path_slp, path_sst, output_dir):
    """
    Calcule et affiche l'écart-type pixel par pixel en hiver en accumulant
    les sommes et sommes des carrés pour éviter la surcharge mémoire.
    """
    print(f"\n=== Calcul de l'écart-type hivernal sur {len(member_ids)} membres ===")
    
    sum_slp = 0.0; sum_sq_slp = 0.0; count_slp = 0
    sum_sst = 0.0; sum_sq_sst = 0.0; count_sst = 0
    
    start_time = time.time()
    for mem in member_ids:
        file_slp, file_sst = get_file_paths(path_slp, path_sst, mem)
        if not os.path.exists(file_slp) or not os.path.exists(file_sst):
            continue
            
        with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
            # 1. Traitement SLP
            slp_winter = ds_slp['PSL'].sel(time=ds_slp['time'].dt.month.isin(winter_months)).values
            slp_winter = np.nan_to_num(slp_winter, nan=0.0)  # Conserver les valeurs en Pa
            
            sum_slp += slp_winter.sum(axis=0)
            sum_sq_slp += (slp_winter**2).sum(axis=0)
            count_slp += slp_winter.shape[0]
            
            # 2. Traitement SST
            ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15, 70))
            sst_winter = ds_sst['SST'].sel(time=ds_sst['time'].dt.month.isin(winter_months)).values
            sst_winter = np.nan_to_num(sst_winter, nan=0.0)
            
            sum_sst += sst_winter.sum(axis=0)
            sum_sq_sst += (sst_winter**2).sum(axis=0)
            count_sst += sst_winter.shape[0]

        print(f" -> Membre {mem} accumulé ({count_slp} jours au total)...", end='\r')

    print(f"\nAccumulation terminée en {time.time() - start_time:.2f}s.")
    
    # Formule de la variance : E[X^2] - (E[X])^2
    var_slp = (sum_sq_slp / max(1, count_slp)) - (sum_slp / max(1, count_slp))**2
    std_slp = np.sqrt(np.maximum(0, var_slp))
    
    var_sst = (sum_sq_sst / max(1, count_sst)) - (sum_sst / max(1, count_sst))**2
    std_sst = np.sqrt(np.maximum(0, var_sst))

    # --- Tracé de la figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={'projection': ccrs.PlateCarree()},gridspec_kw={'width_ratios': [1, 1.5]})
    
    # Carte Std SLP
    ax_slp = axes[0]
    extent_slp = [-100, 40, 20, 70]
    im_slp = ax_slp.imshow(std_slp, transform=ccrs.PlateCarree(), cmap='Reds', origin='lower', extent=extent_slp)
    ax_slp.set_extent(extent_slp, crs=ccrs.PlateCarree())
    ax_slp.coastlines(color='black', linewidth=0.5)
    ax_slp.set_title(f"Écart-type SLP (Hiver NDJF)", fontweight='bold')
    cbar_slp = fig.colorbar(im_slp, ax=ax_slp, fraction=0.046, pad=0.04, shrink=0.6)
    cbar_slp.set_label('Pa')

    # Carte Std SST
    ax_sst = axes[1]
    extent_sst = [-180, 180, -15, 70]
    im_sst = ax_sst.imshow(std_sst, transform=ccrs.PlateCarree(), cmap='Reds', origin='lower', extent=extent_sst, vmax=2.0)
    ax_sst.set_extent(extent_sst, crs=ccrs.PlateCarree())
    ax_sst.coastlines(color='black', linewidth=0.5)
    ax_sst.set_title(f"Écart-type SST (Hiver NDJF)", fontweight='bold')
    cbar_sst = fig.colorbar(im_sst, ax=ax_sst, fraction=0.046, pad=0.04, shrink=0.6)
    cbar_sst.set_label('K')

    fig.tight_layout()
    save_path = os.path.join(output_dir, "Winter_Standard_Deviation_SLP_SST.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Carte d'écart-type sauvegardée : {save_path}")

def create_gif_from_folder(image_folder, output_gif_path, duration_ms=250):
    """
    Compile toutes les images PNG d'un dossier en un GIF animé.
    duration_ms : durée de chaque frame en millisecondes (250 ms = 4 images/seconde).
    """
    # Récupération et tri chronologique des fichiers PNG
    # À changer dans la fonction create_gif_from_folder :
    jpg_files = sorted(glob.glob(os.path.join(image_folder, "map_*.jpg")))
    
    if not jpg_files:
        print(f"⚠️ Aucune image trouvée dans {image_folder} pour créer le GIF.")
        return

    print(f"Création du GIF à partir de {len(jpg_files)} images...")
    
    # Chargement des images avec PIL
    frames = [Image.open(f) for f in jpg_files]
    
    # Sauvegarde en GIF
    frames[0].save(
        output_gif_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration_ms,
        loop=0  # 0 = boucle infinie
    )
    print(f"✅ GIF sauvegardé avec succès : {output_gif_path}")

def plot_chronological_panel(mem, year, month, path_slp, path_sst, output_dir, num_days=6):
    """
    Génère une planche statique de 'num_days' étapes temporelles espacées régulièrement
    pour observer l'évolution simultanée de SLP et SST. Idéal pour un rapport PDF/LaTeX.
    """
    print(f"\n=== Génération de la planche chronologique (LaTeX friendly) ===")
    file_slp, file_sst = get_file_paths(path_slp, path_sst, mem)
    
    if not os.path.exists(file_slp) or not os.path.exists(file_sst):
        print(f"❌ Fichiers introuvables pour le membre {mem}")
        return

    with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
        ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15, 70))
        
        time_cond_slp = (ds_slp['time'].dt.year == year) & (ds_slp['time'].dt.month == month)
        time_cond_sst = (ds_sst['time'].dt.year == year) & (ds_sst['time'].dt.month == month)
        
        slp_month = ds_slp['PSL'].sel(time=time_cond_slp)
        sst_month = ds_sst['SST'].sel(time=time_cond_sst)

        dates = slp_month.time.values
        total_days = len(dates)
        
        # Sélection de N indices répartis équitablement sur le mois
        indices = np.linspace(0, total_days - 1, num_days, dtype=int)
        selected_dates = dates[indices]

        # Initialisation de la grille de subplots (N lignes, 2 colonnes)
        fig, axes = plt.subplots(num_days, 2, figsize=(14, 3.5 * num_days), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        
        extent_slp = [-100, 40, 20, 70]
        extent_sst = [-180, 180, -15, 70]

        for i, t in enumerate(selected_dates):
            date_str = str(t).split('T')[0]
            
            # --- Colonne 1 : SLP ---
            ax_slp = axes[i, 0]
            im_slp = ax_slp.imshow(
                slp_month.sel(time=t).values, transform=ccrs.PlateCarree(),
                cmap='RdBu_r', origin='lower', extent=extent_slp, vmin=-3000, vmax=3000
            )
            ax_slp.set_extent(extent_slp, crs=ccrs.PlateCarree())
            ax_slp.coastlines(color='black', linewidth=0.6, alpha=0.7)
            ax_slp.set_title(f"SLP Anomaly — {date_str}", fontsize=11, fontweight='bold')
            
            # Barre de couleur uniquement pour la première ligne pour ne pas surcharger
            if i == 0:
                cbar_slp = fig.colorbar(im_slp, ax=ax_slp, orientation='vertical', fraction=0.046, pad=0.04)
                cbar_slp.set_label('Pa')
            else:
                fig.colorbar(im_slp, ax=ax_slp, fraction=0.046, pad=0.04)

            # --- Colonne 2 : SST ---
            ax_sst = axes[i, 1]
            im_sst = ax_sst.imshow(
                sst_month.sel(time=t).values, transform=ccrs.PlateCarree(),
                cmap='RdBu_r', origin='lower', extent=extent_sst, vmin=-3, vmax=3
            )
            ax_sst.set_extent(extent_sst, crs=ccrs.PlateCarree())
            ax_sst.coastlines(color='black', linewidth=0.6, alpha=0.7)
            ax_sst.set_title(f"SST Anomaly — {date_str}", fontsize=11, fontweight='bold')
            
            if i == 0:
                cbar_sst = fig.colorbar(im_sst, ax=ax_sst, orientation='vertical', fraction=0.046, pad=0.04)
                cbar_sst.set_label('K')
            else:
                fig.colorbar(im_sst, ax=ax_sst, fraction=0.046, pad=0.04)

        fig.suptitle(f"Chrono-Evolution — Member {mem} ({year}-{month:02d})", fontsize=16, fontweight='bold', y=1.01)
        fig.tight_layout()
        
        save_path = os.path.join(output_dir, f"Chrono_Panel_{mem}_{year}_{month:02d}.png")
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Planche chronologique sauvegardée : {save_path}")


# ==========================================
# 4. BLOC MAIN & CONFIGURATION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
    parser.add_argument('--number_of_members', type=int, default=10, help="Nombre de membres pour l'écart-type")
    args = parser.parse_args()

    # Configuration des chemins
    folder_name = "report_plots_winter_std_and_daily"
    if args.machine == 'hacienda':
        base_home = f"/home/moysan/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = "/data/moysan/data/"
    elif args.machine == 'jean-zay-work': 
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = "/lustre/fswork/projects/rech/uxg/uca57ub/data/"
    elif args.machine == 'jean-zay-scratch':
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/"

    os.makedirs(base_home, exist_ok=True)
    path_sst = os.path.join(data_dir, "SST/")
    path_slp = os.path.join(data_dir, "SLP/")

    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    members_used = all_members[:args.number_of_members]
    winter_months = [11, 12, 1, 2]

        # --- ACTION 1 : Cartes journalières (Génère le dossier de PNG individuels + le GIF) ---
    plot_daily_maps_for_month(
        mem=all_members[1], year=1999, month=11, 
        path_slp=path_slp, path_sst=path_sst, output_dir=base_home
    )

    plot_daily_maps_for_month(
        mem=all_members[1], year=1999, month=12, 
        path_slp=path_slp, path_sst=path_sst, output_dir=base_home
    )


    # --- ACTION 1 : Cartes journalières (Génère le dossier de PNG individuels + le GIF) ---
    plot_daily_maps_for_month(
        mem=all_members[1], year=2000, month=1, 
        path_slp=path_slp, path_sst=path_sst, output_dir=base_home
    )

        # --- ACTION 1 : Cartes journalières (Génère le dossier de PNG individuels + le GIF) ---
    plot_daily_maps_for_month(
        mem=all_members[1], year=2000, month=2, 
        path_slp=path_slp, path_sst=path_sst, output_dir=base_home
    )

    # --- ACTION 2 : Planche chronologique combinée (Idéal pour LaTeX) ---
    plot_chronological_panel(
        mem=all_members[1], year=1999, month=11, 
        path_slp=path_slp, path_sst=path_sst, output_dir=base_home, num_days=10
    )

    # --- ACTION 2 : Planche chronologique combinée (Idéal pour LaTeX) ---
    plot_chronological_panel(
        mem=all_members[1], year=1999, month=12, 
        path_slp=path_slp, path_sst=path_sst, output_dir=base_home, num_days=10
    )

    # --- ACTION 2 : Planche chronologique combinée (Idéal pour LaTeX) ---
    plot_chronological_panel(
        mem=all_members[1], year=2000, month=1, 
        path_slp=path_slp, path_sst=path_sst, output_dir=base_home, num_days=10
    )

    # --- ACTION 2 : Planche chronologique combinée (Idéal pour LaTeX) ---
    plot_chronological_panel(
        mem=all_members[1], year=2000, month=2, 
        path_slp=path_slp, path_sst=path_sst, output_dir=base_home, num_days=10
    )

    # # --- ACTION 3 : Écart-type par pixel en hiver ---
    # compute_winter_std(
    #     member_ids=members_used, winter_months=winter_months, 
    #     path_slp=path_slp, path_sst=path_sst, output_dir=base_home
    # )

    