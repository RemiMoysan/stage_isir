import xarray as xr
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_sst', type=str, required=True,
                        help="Chemin vers un fichier SST_anom_LE2-XXX_T_regrid.nc (n'importe quel membre)")
    args = parser.parse_args()

    # On charge juste un membre, un seul pas de temps suffit pour le masque
    ds = xr.open_dataset(args.file_sst)
    sst = ds['SST'].sel(lat=slice(-15, 70))

    # Masque océan (True = océan, False = terre/NaN), comme dans compute_variance_maps
    mask_ocean = ~np.isnan(sst.isel(time=0).values)   # shape (lat, lon)

    lats = sst['lat'].values
    coslat = np.cos(np.deg2rad(lats))                  # shape (lat,)

    # Poids 2D : cos(lat) répété sur toutes les longitudes
    coslat_2d = coslat[:, None] * np.ones((1, sst.sizes['lon']))

    sum_coslat_ocean = np.sum(coslat_2d * mask_ocean)
    sum_coslat_total = np.sum(coslat_2d)

    ratio_variance = sum_coslat_ocean / sum_coslat_total
    ratio_std = np.sqrt(ratio_variance)

    frac_ocean_pixels = mask_ocean.mean()  # proportion brute de pixels océan, pour comparaison

    print(f"Fraction brute de pixels océan (non pondérée)      : {frac_ocean_pixels:.4f}")
    print(f"Somme cos(lat) océan                                : {sum_coslat_ocean:.4f}")
    print(f"Somme cos(lat) totale (océan+terre)                 : {sum_coslat_total:.4f}")
    print(f"Ratio attendu des VARIANCES (biaisé/correct)        : {ratio_variance:.4f}")
    print(f"Ratio attendu des ECARTS-TYPES (biaisé/correct)     : {ratio_std:.4f}")