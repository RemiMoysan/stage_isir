import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import time


# ==========================================
# 1. GESTION DYNAMIQUE DES CHEMINS (ROOT)
# ==========================================
# __file__ correspond à main.py
# .parent 1 -> data_analysis 
# .parent 2 -> stage_isir_jz 

project_root = Path(__file__).resolve().parent.parent

# Ajouter le dossier "tools" de vision_transformer au sys.path pour tes imports de modèles
vision_transformer_dir = os.path.join(project_root, "vision_transformer")
if vision_transformer_dir not in sys.path:
    sys.path.append(vision_transformer_dir)

import argparse
from tools.datasets import Dataset
from tools.visualizations import plot_confusion_matrix
import torch

def create_master_reference(base_home_and_subfolder, member_ids, embedding_method='pca'):
    """
    Charge les fichiers .npz de tous les membres fournis et calcule la moyenne.
    """
    master_dict = {}
    n_members = len(member_ids)
    
    print(f"Création de la référence maître à partir de {n_members} membres...")
    
    for mem in member_ids:
        filepath = os.path.join(base_home_and_subfolder, f"Composites_Mem{mem}_{embedding_method}.npz")
        data = np.load(filepath)
        
        for key in data.files:
            # On ignore les comptages pour la moyenne des cartes
            if "count" in key:
                continue
                
            if key not in master_dict:
                master_dict[key] = data[key] / n_members
            else:
                master_dict[key] += data[key] / n_members
                
    print("Référence maître créée avec succès !")
    return master_dict

def calculate_predictability_accuracy(slp_samples, sst_samples, ref_dict, lag_sst, metric='mse', is_sanity_check=False):
    """
    Calcule l'accuracy de prédiction.
    metric : 'correlation' ou 'mse'
    is_sanity_check : Si True, compare sst_samples aux références SLP au lieu des références SST.
    """
    N = slp_samples.shape[0]
    
    # 1. Extraction dynamique des préfixes
    regime_prefixes = []
    for i in range(1, 5): 
        for key in ref_dict.keys():
            if key.startswith(f"regime_{i}_") and key.endswith("_slp_0_mean"):
                regime_prefixes.append(key.replace("_slp_0_mean", ""))
                break
                
    if len(regime_prefixes) != 4:
        raise ValueError("Erreur : Impossible de trouver les 4 régimes dans le dictionnaire.")

    # 2. Extraction des cartes de référence
    ref_slp = np.array([ref_dict[f"{prefix}_slp_0_mean"] for prefix in regime_prefixes])
    
    # --- LA CORRECTION EST ICI ---
    if is_sanity_check:
        ref_sst = ref_slp  # On force l'utilisation des centres SLP !
    else:
        ref_sst = np.array([ref_dict[f"{prefix}_sst_lag_{lag_sst}_mean"] for prefix in regime_prefixes])
    
    # 3. Aplatissement (Flatten) pour les calculs de distances
    slp_flat = slp_samples.reshape(N, -1)     
    sst_flat = sst_samples.reshape(N, -1)     
    ref_slp_flat = ref_slp.reshape(4, -1)     
    ref_sst_flat = ref_sst.reshape(4, -1)

    # --- MASQUAGE EXPLICITE DES NaN ---
    # On dérive le masque depuis la référence (même masque continental pour tous les régimes)
    valid_slp = ~np.isnan(ref_slp_flat[0])
    valid_sst = ~np.isnan(ref_sst_flat[0])

    slp_flat     = slp_flat[:, valid_slp]
    ref_slp_flat = ref_slp_flat[:, valid_slp]
    sst_flat     = sst_flat[:, valid_sst]
    ref_sst_flat = ref_sst_flat[:, valid_sst]
    # ----------------------------------

    # 4. Calcul des scores selon la métrique
    if metric == 'correlation':
        def get_corr(samples, refs):
            s_c = samples - samples.mean(axis=1, keepdims=True)
            r_c = refs - refs.mean(axis=1, keepdims=True)
            s_n = np.linalg.norm(s_c, axis=1, keepdims=True)
            r_n = np.linalg.norm(r_c, axis=1, keepdims=True)
            s_n[s_n == 0] = 1e-10 # Sécurité division par zéro
            r_n[r_n == 0] = 1e-10
            return np.dot(s_c, r_c.T) / (s_n * r_n.T)

        score_slp = get_corr(slp_flat, ref_slp_flat)
        score_sst = get_corr(sst_flat, ref_sst_flat)
        
        # Pour la corrélation, on cherche la valeur MAXIMALE
        true_labels = np.argmax(score_slp, axis=1)
        pred_labels = np.argmax(score_sst, axis=1)

    elif metric == 'mse':
        def get_mse(samples, refs):
            # Broadcasting Numpy pour calculer la distance entre chaque sample et chaque référence
            # Shape finale : (N, 4)
            diff = samples[:, np.newaxis, :] - refs[np.newaxis, :, :]
            return np.nanmean(diff**2, axis=2)
            
        score_slp = get_mse(slp_flat, ref_slp_flat)
        score_sst = get_mse(sst_flat, ref_sst_flat)
        
        # Pour la MSE, on cherche la valeur MINIMALE (l'erreur la plus faible)
        true_labels = np.argmin(score_slp, axis=1)
        pred_labels = np.argmin(score_sst, axis=1)
        
    else:
        raise ValueError("L'argument metric doit être 'correlation' ou 'mse'.")

    # 5. Bilan
    hits = np.sum(true_labels == pred_labels)
    accuracy = (hits / N) * 100
    
    print(f"--- RÉSULTATS (Métrique : {metric.upper()}) ---")
    print(f"Lag évalué : SST à -{lag_sst} jours")
    print(f"Accuracy de prédiction linéaire : {accuracy:.2f}% (Seuil de hasard = 25%)")
    
    return accuracy, true_labels, pred_labels

def plot_master_references(master_ref, outdir, lag_sst, vmax_plot_slp=2.5, vmax_plot_sst=0.6):
    """
    Trace les centres des clusters (matrices moyennes de SLP et SST).
    Ligne du haut : SLP (t=0)
    Ligne du bas : SST associée (t=-lag_sst)
    """
    print(f"\nGénération de la figure des références globales (SLP et SST lag {lag_sst})...")
    
    # 1. Identifier les clés correspondant à la SLP à t=0
    slp_keys = [k for k in master_ref.keys() if k.endswith("_slp_0_mean") and "global_mean" not in k]
    slp_keys.sort() # Pour garder l'ordre (Regime 1, 2, 3, 4)
    n_clusters = len(slp_keys)

    if n_clusters == 0:
        print("Erreur : Aucune clé SLP trouvée dans le dictionnaire maître.")
        return

    # 2. Création de la figure (2 lignes, N colonnes)
    fig, axes = plt.subplots(2, n_clusters, figsize=(5 * n_clusters, 8), 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    if n_clusters == 1: 
        axes = np.expand_dims(axes, axis=1)

    im_slp, im_sst = None, None

    # 3. Traçage de chaque référence
    for i, key in enumerate(slp_keys):
        # --- LIGNE 0 : SLP ---
        ax_slp = axes[0, i]
        map_data_slp = master_ref[key]
        regime_name = key.replace("_slp_0_mean", "").replace("regime_", "")
        
        # extent de la SLP inchangé par rapport à ta version d'origine
        im_slp = ax_slp.imshow(map_data_slp, transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                       origin='lower', extent=[-100, 40, 20, 70], 
                       vmin=-vmax_plot_slp, vmax=vmax_plot_slp)
        
        ax_slp.coastlines(alpha=0.5) 
        ax_slp.set_title(f"SLP : {regime_name}", fontweight='bold')

        # --- LIGNE 1 : SST ---
        ax_sst = axes[1, i]
        sst_key = key.replace("_slp_0_mean", f"_sst_lag_{lag_sst}_mean")
        
        if sst_key in master_ref:
            map_data_sst = master_ref[sst_key]
            
            # Utilisation du nouveau crop de l'extent pour la SST (-20 à 80)
            im_sst = ax_sst.imshow(map_data_sst, transform=ccrs.PlateCarree(), cmap='RdBu_r', 
                           origin='lower', extent=[-180, 180, -15, 70], 
                           vmin=-vmax_plot_sst, vmax=vmax_plot_sst)
            
            ax_sst.coastlines(alpha=0.5)
            ax_sst.set_title(f"SST (Lag {lag_sst}) : {regime_name}")
        else:
            ax_sst.set_title(f"Clé SST introuvable")

    # 4. Colorbars
    if im_slp:
        cbar_slp = fig.colorbar(im_slp, ax=axes[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        cbar_slp.set_label("Anomalie SLP (σ)")

    if im_sst:
        cbar_sst = fig.colorbar(im_sst, ax=axes[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        cbar_sst.set_label("Anomalie SST (σ)")

    # 5. Titre et Sauvegarde
    plt.suptitle(f"Centres des Clusters (SLP t=0 & SST t=-{lag_sst})", fontsize=16, y=1.02)
    save_path = os.path.join(outdir, f"SanityCheck_Master_References_lag{lag_sst}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Figure des références sauvegardée : {save_path}")

def plot_diagnostic_alignment(sample_ssts, ref_dict, lag_sst, true_labels,pred_labels, outdir, num_samples=24, vmax_sst=0.6):
    """
    Vérifie l'alignement géographique (côtes, longitudes) et la cohérence des prédictions.
    - Calcule la superposition mathématique des masques.
    - Affiche les continents (NaN ou 0) en vert fluo pour un contrôle visuel immédiat.
    - Affiche 4 références et jusqu'à 24 échantillons sur une grande grille.
    """
    print(f"\nGénération du diagnostic visuel d'alignement (Références + {num_samples} échantillons)...")
    
    # 1. Extraction des références
    regime_prefixes = []
    for i in range(1, 5): 
        for key in ref_dict.keys():
            if key.startswith(f"regime_{i}_") and key.endswith("_slp_0_mean"):
                regime_prefixes.append(key.replace("_slp_0_mean", ""))
                break
                
    ref_ssts = [ref_dict[f"{prefix}_sst_lag_{lag_sst}_mean"] for prefix in regime_prefixes]
    
    # 2. VÉRIFICATION DU MASQUE CONTINENTAL (En console)
    ref_0 = ref_ssts[0]
    sam_0 = sample_ssts[0]
    
    # On identifie ce qui sert de continent (soit des NaN, soit des 0 absolus)
    mask_ref = np.isnan(ref_0) if np.any(np.isnan(ref_0)) else (ref_0 == 0)
    mask_sam = np.isnan(sam_0) if np.any(np.isnan(sam_0)) else (sam_0 == 0)
    
    match_percentage = np.sum(mask_ref == mask_sam) / mask_ref.size * 100
    
    print(f"--- VÉRIFICATION SPATIALE ---")
    print(f"Superposition exacte du masque Océan/Continent : {match_percentage:.2f}%")
    if match_percentage < 99.0:
        print("🚨 ATTENTION : Les masques ne se superposent pas. Décalage de grille fortement probable !")
    else:
        print("✅ Masques parfaitement superposés.")
    print(f"-----------------------------")

    # 3. Préparation de la colormap (Continents en vert "Lime" pour le repérage visuel)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(color="white") # NaN en blanc
    # 4. Création de la figure dynamique (Ligne 0 = Références, Lignes suivantes = Échantillons)
    # Pour 24 échantillons, on aura 1 + 6 = 7 lignes.
    n_rows = 1 + int(np.ceil(num_samples / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 3.5 * n_rows), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # --- LIGNE 0 : Les 4 cartes de référence ---
    for i, ref_sst in enumerate(ref_ssts):
        ax_ref = axes[0, i]
        
        # Copie pour modifier l'affichage : On force les 0 en NaN pour que le vert s'applique
        plot_data = ref_sst.copy()
        if not np.any(np.isnan(plot_data)) and np.any(plot_data == 0):
            plot_data[plot_data == 0] = np.nan
            
        im_ref = ax_ref.imshow(plot_data, transform=ccrs.PlateCarree(), cmap=cmap, 
                               origin='lower', extent=[-180, 180, -15, 70], 
                               vmin=-vmax_sst, vmax=vmax_sst)
        ax_ref.coastlines(color='black', linewidth=1.5)
        ax_ref.set_title(f"RÉFÉRENCE : Régime {i+1}", fontweight='bold', fontsize=14)
        
    # --- LIGNES SUIVANTES : Les Échantillons ---
    # On choisit 'num_samples' indices répartis sur tout le dataset
    indices = np.linspace(0, len(sample_ssts)-1, num_samples, dtype=int)
    sample_axes = axes[1:].flatten() # On aplatit les axes restants pour boucler facilement
    
    for i, ax_sample in enumerate(sample_axes):
        if i < len(indices):
            idx = indices[i]
            sample = sample_ssts[idx]

            true_reg = true_labels[idx] + 1
            pred_reg = pred_labels[idx] + 1
            # Couleur : Vert si match, Rouge sinon
            title_color = 'green' if true_reg == pred_reg else 'red'
            
            # Application de l'astuce du vert fluo sur l'échantillon
            plot_data = sample.copy()
            if not np.any(np.isnan(plot_data)) and np.any(plot_data == 0):
                plot_data[plot_data == 0] = np.nan
                print("Il n'y avait pas de NaN dans cet échantillon, mais des 0 ont été convertis en NaN pour le diagnostic visuel.")
                
            im_sample = ax_sample.imshow(plot_data, transform=ccrs.PlateCarree(), cmap=cmap, 
                                         origin='lower', extent=[-180, 180, -15, 70], 
                                         vmin=-vmax_sst, vmax=vmax_sst)
            ax_sample.coastlines(color='black', linewidth=1.5)
            ax_sample.set_title(f"Échantillon {idx} | Vrai: Reg {true_reg} | Prédit: Reg {pred_reg}", color=title_color, fontweight='bold', fontsize=11)
        else:
            # On masque les subplots vides si le compte n'est pas un multiple de 4
            ax_sample.axis('off')

    # Colorbar globale
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im_ref, cax=cbar_ax, label="Anomalie SST (σ) [Blanc = Masque Continent]")

    plt.suptitle(f"Sanity Check : Masques, Alignement et Labels (Lag {lag_sst})", fontsize=20, y=0.92 + (0.005*n_rows))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    
    save_path = os.path.join(outdir, f"Diagnostic_Alignment_24_Samples_lag{lag_sst}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Figure de diagnostic (24 échantillons) sauvegardée : {save_path}\n")
# Ce code prédit à partir du clustering d'un certain lag de SST
# On pourrait essayer de refine pour mélanger plusieurs lags à termes mais du coup il faudrait réfléchir à comment les pondérer. 

if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch',"mac_local"])
    parser.add_argument('--target_lag', type=int, default=35, help="Le lag de SST à évaluer (en jours)")
    parser.add_argument('--relative_path', type=str, default="stage_isir_jz/data_analysis/composites_4_regimes/master_ref_generator_89members_normalizeFalse_duree_lissage10_embedding_method_pca_latent_dim_128_future/", help="Chemin relatif depuis le root vers la référence maître .npz")
    parser.add_argument('--relative_path_reference', type=str, default=None, help="Chemin relatif depuis le root vers la référence maître .npz (option alternative pour charger directement la référence au lieu de la créer à partir des membres)")
    parser.add_argument('--duree_lissage', type=int, default=10, help="Durée de lissage utilisée pour la référence maître (en jours)")
    parser.add_argument('--number_of_members', type=int, default=1, help="Nombre de membres à utiliser pour le test")
    args = parser.parse_args()

    start_time = time.time()

    if args.machine == 'hacienda':
        base_home = f"/home/moysan/"
    elif args.machine == 'jean-zay-work' or args.machine == 'jean-zay-scratch': 
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/" 
    elif args.machine == 'mac_local':
        base_home = f"/Users/remimoysan/Desktop/Jean_Zay/work_jz/"

    # Création du dossier pour sauvegarder les matrices de confusion
    outdir = os.path.join(base_home, args.relative_path, "Baseline_Evaluations")
    os.makedirs(outdir, exist_ok=True)

    # 1. On fabrique la référence maître

    #train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    #nb_members_train = 1
    #train_members = train_members_87[:nb_members_train]
    #val_members = ['1001.001'] finalement on va utiliser ça comme train
    # A ne pas pas prendre car un peu trop différent 1231.006; à ne pas prendre car ordre différent 1231.004, 1181.010
    # train_members = ["1001.001",'1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1231.001', '1231.002', '1231.003', '1231.005', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001']
    # val_members = ["1301.010"]
    # master_ref = create_master_reference(base_home, train_members, 'pca')

    # On utilise un seul lag SST pour le test (celui qu'on va évaluer)
    target_lag = args.target_lag
    relative_path_reference = os.path.join(args.relative_path, "master_reference_global.npz") 
    filepath = os.path.join(base_home, relative_path_reference)
    master_ref = np.load(filepath)


    # ---> Ajout du Sanity Check ici :
    plot_master_references(master_ref, outdir, lag_sst = target_lag)

    # 2. Chargement des données de validation via Dataset
    print("\nChargement des données de validation...")
    # On sélectionne les mois d'hiver (comme dans ton clustering : NDJF ou JF selon ta configuration habituelle)
    winter_months = [11, 12, 1, 2] 


    all_members = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    val_members = all_members[:args.number_of_members]
    # val_members = ["1301.010"] # du coup ce n'est pas vraiment du val puisque la ref maître a été entrainé sur tout. 

    val_set = Dataset(
        members=val_members, 
        selected_months=winter_months, 
        machine=args.machine, 
        target_type='map', 
        sst_lags_days=[target_lag], 
        slp_lags_days=[],
        roll_sst= True,
        duree_lissage = args.duree_lissage

    )
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    n_workers = max(0, n_workers - 1)
    print(f"Using {n_workers} workers for data loading")
    valloader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=n_workers)

    # 3. Accumulation des batchs pour construire l'échantillon complet
    all_slp_test = []
    all_sst_test = []

    print("Extraction des tenseurs...")
    with torch.no_grad():
        # Déballage des 6 variables renvoyées par Dataset : (X_sst, X_slp, y_target, y_map, dates, members)
        for batch_idx, (X_sst, _, _, y_map, _, _) in enumerate(valloader):
            # La vraie SLP (t=0) est dans y_map (Shape: Batch, 53, 113)
            all_slp_test.append(y_map.numpy())
            
            # La SST (à t-lag) est dans X_sst. (Shape: Batch, 1, 85, 360)
            # On supprime la dimension du canal "1" avec squeeze(1)
            all_sst_test.append(X_sst.squeeze(1).numpy())

    # Concaténation finale
    slp_samples_test = np.concatenate(all_slp_test, axis=0)
    sst_samples_test = np.concatenate(all_sst_test, axis=0)

    print(f"Forme finale SLP Test : {slp_samples_test.shape}")
    print(f"Forme finale SST Test : {sst_samples_test.shape}")

# 4. Évaluation et Génération des Matrices de Confusion

    # Test de sanité : Prédiction de la SLP par la SLP
    acc_mse_slp, true_mse_slp, pred_mse_slp = calculate_predictability_accuracy(
        slp_samples_test, slp_samples_test, # Remplacez sst_samples_test par slp_samples_test
        ref_dict=master_ref, lag_sst=target_lag, metric='mse',is_sanity_check=True
    )

    plot_confusion_matrix(
        y_true=true_mse_slp, 
        y_pred=pred_mse_slp, 
        outdir=outdir, 
        master_ref=master_ref, 
        filename=f'sanity_check_slp_slp.png'
    )

    print("\nLancement de l'évaluation MSE...")
    acc_mse, true_mse, pred_mse = calculate_predictability_accuracy(
        slp_samples_test, sst_samples_test, 
        ref_dict=master_ref, lag_sst=target_lag, metric='mse'
    )

# ==========================================
    # AJOUT : DIAGNOSTIC D'ALIGNEMENT ET LABELS
    # ==========================================
    plot_diagnostic_alignment(
        sample_ssts=sst_samples_test, 
        ref_dict=master_ref, 
        lag_sst=target_lag, 
        true_labels=true_mse,
        pred_labels=pred_mse, 
        outdir=outdir, 
        num_samples=24 # On affiche 24 échantillons
    )
    # ==========================================

    plot_confusion_matrix(
        y_true=true_mse, 
        y_pred=pred_mse, 
        outdir=outdir, 
        master_ref=master_ref, 
        filename=f'baseline_confusion_mse_lag{target_lag}.png'
    )

    print("\nLancement de l'évaluation Corrélation...")
    acc_corr, true_corr, pred_corr = calculate_predictability_accuracy(
        slp_samples_test, sst_samples_test, 
        ref_dict=master_ref, lag_sst=target_lag, metric='correlation'
    )
    plot_confusion_matrix(
        y_true=true_corr, 
        y_pred=pred_corr, 
        outdir=outdir, 
        master_ref=master_ref, 
        filename=f'baseline_confusion_corr_lag{target_lag}.png'
    )
    
    print(f"\nMatrices de confusion sauvegardées dans : {outdir}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training complete, elapsed time: {elapsed_time / 60:.2f} minutes")