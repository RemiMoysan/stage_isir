import os
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# ==========================================
# 1. UTILITIES
# ==========================================
def get_file_paths(path_slp, path_sst, mem):
    file_slp = os.path.join(path_slp, f'PSL_anom_LE2-{mem}_1mo.nc')
    file_sst = os.path.join(path_sst, f'SST_anom_LE2-{mem}_T_regrid_1mo.nc')
    return file_slp, file_sst

# ==========================================
# 2. MAIN PROCESSING & PLOTTING
# ==========================================
def generate_lag_frames(mem, path_slp, path_sst, output_dir):
    print(f"\n=======================================================")
    print(f" 🚀 GENERATING LATEX ANIMATION FRAMES (Member {mem})")
    print(f"=======================================================", flush=True)

    file_slp, file_sst = get_file_paths(path_slp, path_sst, mem)
    
    if not os.path.exists(file_slp) or not os.path.exists(file_sst):
        raise FileNotFoundError(f"❌ Missing NetCDF files for member {mem}.")

    out_anim_dir = os.path.join(output_dir, f"Animation_Frames_{mem}")
    os.makedirs(out_anim_dir, exist_ok=True)

    with xr.open_dataset(file_slp) as ds_slp, xr.open_dataset(file_sst) as ds_sst:
        # Alignement de la longitude SST
        ds_sst = ds_sst.assign_coords(lon=(((ds_sst.lon + 180) % 360) - 180)).sortby('lon').sel(lat=slice(-15, 70))
        
        # Identifier toutes les années disponibles pour le mois de février
        feb_times = ds_slp['time'].sel(time=ds_slp['time'].dt.month == 2)
        years = np.unique(feb_times.dt.year.values)
        
        print(f" -> Found {len(years)} potential February targets. Processing frames...", flush=True)
        
        extent_slp = [-100, 40, 20, 70]
        extent_sst = [-180, 180, -15, 70]
        valid_frames = 0

        for y in years:
            try:
                # Extraction stricte des 4 mois nécessaires
                sst_nov = ds_sst['SST'].sel(time=f"{y-1}-11").isel(time=0).values
                sst_dec = ds_sst['SST'].sel(time=f"{y-1}-12").isel(time=0).values
                sst_jan = ds_sst['SST'].sel(time=f"{y}-01").isel(time=0).values
                slp_feb = ds_slp['PSL'].sel(time=f"{y}-02").isel(time=0).values
            except Exception:
                continue

            # Création d'une seule ligne de 4 colonnes, très large (24 pouces)
            # Le width_ratios applique le fameux facteur 1.5 sur les cartes SST
            fig, axes = plt.subplots(
                1, 4, 
                figsize=(24, 4.5), 
                subplot_kw={'projection': ccrs.PlateCarree()},
                gridspec_kw={'width_ratios': [1.5, 1.5, 1.5, 1]}
            )
            
            # --- 1. SST Novembre (Lag -3) ---
            ax_nov = axes[0]
            im_nov = ax_nov.imshow(sst_nov, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent_sst, vmin=-3, vmax=3)
            ax_nov.set_extent(extent_sst, crs=ccrs.PlateCarree())
            ax_nov.coastlines(color='black', linewidth=0.8)
            ax_nov.set_title(f"SST Anomaly — November {y-1} (Lag -3)", fontweight='bold', fontsize=12)
            cbar_nov = fig.colorbar(im_nov, ax=ax_nov, fraction=0.03, pad=0.04, shrink=0.8)
            cbar_nov.set_label('K')

            # --- 2. SST Décembre (Lag -2) ---
            ax_dec = axes[1]
            im_dec = ax_dec.imshow(sst_dec, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent_sst, vmin=-3, vmax=3)
            ax_dec.set_extent(extent_sst, crs=ccrs.PlateCarree())
            ax_dec.coastlines(color='black', linewidth=0.8)
            ax_dec.set_title(f"SST Anomaly — December {y-1} (Lag -2)", fontweight='bold', fontsize=12)
            cbar_dec = fig.colorbar(im_dec, ax=ax_dec, fraction=0.03, pad=0.04, shrink=0.8)
            cbar_dec.set_label('K')

            # --- 3. SST Janvier (Lag -1) ---
            ax_jan = axes[2]
            im_jan = ax_jan.imshow(sst_jan, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent_sst, vmin=-3, vmax=3)
            ax_jan.set_extent(extent_sst, crs=ccrs.PlateCarree())
            ax_jan.coastlines(color='black', linewidth=0.8)
            ax_jan.set_title(f"SST Anomaly — January {y} (Lag -1)", fontweight='bold', fontsize=12)
            cbar_jan = fig.colorbar(im_jan, ax=ax_jan, fraction=0.03, pad=0.04, shrink=0.8)
            cbar_jan.set_label('K')

            # --- 4. SLP Février (Target) ---
            ax_feb = axes[3]
            im_feb = ax_feb.imshow(slp_feb, transform=ccrs.PlateCarree(), cmap='RdBu_r', origin='lower', extent=extent_slp, vmin=-3000, vmax=3000)
            ax_feb.set_extent(extent_slp, crs=ccrs.PlateCarree())
            ax_feb.coastlines(color='black', linewidth=0.8)
            ax_feb.set_title(f"SLP Anomaly — February {y}", fontweight='bold', fontsize=12)
            cbar_feb = fig.colorbar(im_feb, ax=ax_feb, fraction=0.046, pad=0.04, shrink=0.8)
            cbar_feb.set_label('Pa')

            # Titre global centré au-dessus de la ligne
            fig.suptitle(f"SST preceding February {y} SLP (Member {mem})", fontsize=16, fontweight='bold', y=1.05)
            fig.tight_layout()
            
            save_path = os.path.join(out_anim_dir, f"frame_{valid_frames:04d}_year_{y}.jpg")
            fig.savefig(save_path, dpi=150, pil_kwargs={'quality': 85}, bbox_inches='tight')
            plt.close(fig)
            
            valid_frames += 1
            print(f"    - Frame saved: Year {y} (Target)", end='\r', flush=True)

    print(f"\n ✅ Success ! {valid_frames} frames successfully generated in: {out_anim_dir}", flush=True)

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='jean-zay-work', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch'])
    parser.add_argument('--member', type=str, default='1081.005', help="Member ID to animate")
    args = parser.parse_args()

    folder_name = "report_animation_february"
    if args.machine == 'hacienda':
        base_home = f"/home/moysan/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = "/data/moysan/data/"
    else:
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/{folder_name}/"
        data_dir = f"/lustre/{'fswork' if args.machine == 'jean-zay-work' else 'fsn1'}/projects/rech/uxg/uca57ub/data/"

    os.makedirs(base_home, exist_ok=True)
    path_sst = os.path.join(data_dir, "SST/")
    path_slp = os.path.join(data_dir, "SLP/")

    generate_lag_frames(args.member, path_slp, path_sst, base_home)