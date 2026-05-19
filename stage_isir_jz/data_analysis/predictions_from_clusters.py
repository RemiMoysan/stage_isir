import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
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

def calculate_predictability_accuracy(slp_samples, sst_samples, ref_dict, lag_sst, metric='mse', outdir=None):
    """
    Calcule l'accuracy de prédiction et génère les histogrammes des distances.
    metric : 'correlation' ou 'mse'
    outdir : Dossier où sauvegarder les histogrammes (si None, ne sauvegarde rien).
    """
    N = slp_samples.shape[0]
    
    # 1. Extraction dynamique des préfixes
    regime_prefixes = []
    for i in range(1, 5): 
        for key in ref_dict.keys():
            if key.startswith(f"regime_{i}_") and key.endswith("_slp_0"):
                regime_prefixes.append(key.replace("_slp_0", ""))
                break
                
    if len(regime_prefixes) != 4:
        raise ValueError("Erreur : Impossible de trouver les 4 régimes dans le dictionnaire de référence.")

    # 2. Extraction des cartes de référence
    ref_slp = np.array([ref_dict[f"{prefix}_slp_0"] for prefix in regime_prefixes])
    ref_sst = np.array([ref_dict[f"{prefix}_sst_lag_{lag_sst}"] for prefix in regime_prefixes])
    
    # 3. Aplatissement (Flatten) pour les calculs de distances
    slp_flat = slp_samples.reshape(N, -1)     
    sst_flat = sst_samples.reshape(N, -1)     
    ref_slp_flat = ref_slp.reshape(4, -1)     
    ref_sst_flat = ref_sst.reshape(4, -1)     

    # --- AJOUT CRUCIAL : GESTION DES MASQUES TERRESTRES (NaN) ---
    # On identifie les pixels valides (ceux qui ne sont pas NaN dans la référence)
    valid_slp_pixels = ~np.isnan(ref_slp_flat[0])
    valid_sst_pixels = ~np.isnan(ref_sst_flat[0])

    # On ampute les tableaux pour ne garder QUE les pixels valides (l'océan pur)
    slp_flat = slp_flat[:, valid_slp_pixels]
    ref_slp_flat = ref_slp_flat[:, valid_slp_pixels]
    
    sst_flat = sst_flat[:, valid_sst_pixels]
    ref_sst_flat = ref_sst_flat[:, valid_sst_pixels]
    # ------------------------------------------------------------  

    # 4. Calcul des scores selon la métrique
    if metric == 'correlation':
        def get_corr(samples, refs):
            s_c = samples - samples.mean(axis=1, keepdims=True)
            r_c = refs - refs.mean(axis=1, keepdims=True)
            s_n = np.linalg.norm(s_c, axis=1, keepdims=True)
            r_n = np.linalg.norm(r_c, axis=1, keepdims=True)
            s_n[s_n == 0] = 1e-10 
            r_n[r_n == 0] = 1e-10
            return np.dot(s_c, r_c.T) / (s_n * r_n.T)

        score_slp = get_corr(slp_flat, ref_slp_flat)
        score_sst = get_corr(sst_flat, ref_sst_flat)
        
        true_labels = np.argmax(score_slp, axis=1)
        pred_labels = np.argmax(score_sst, axis=1)

    elif metric == 'mse':
        def get_mse(samples, refs):
            diff = samples[:, np.newaxis, :] - refs[np.newaxis, :, :]
            return np.mean(diff**2, axis=2)
            
        score_slp = get_mse(slp_flat, ref_slp_flat)
        score_sst = get_mse(sst_flat, ref_sst_flat)
        
        true_labels = np.argmin(score_slp, axis=1)
        pred_labels = np.argmin(score_sst, axis=1)
        
    else:
        raise ValueError("L'argument metric doit être 'correlation' ou 'mse'.")

    # --- DÉBUT DES AJOUTS POUR LE DEBUG (HISTOGRAMMES) ---
    if outdir is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Couleurs pour bien distinguer les 4 régimes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Histograme SLP
        for i, regime in enumerate(regime_prefixes):
            axes[0].hist(score_slp[:, i], bins=50, alpha=0.5, label=f"{i+1}: {regime}", color=colors[i])
        axes[0].set_title(f"Distribution des scores SLP ({metric.upper()})")
        axes[0].set_xlabel("Score (Plus bas = meilleur en MSE, Plus haut = meilleur en Corr)")
        axes[0].set_ylabel("Nombre de jours (N)")
        axes[0].legend()

        # Histograme SST
        for i, regime in enumerate(regime_prefixes):
            axes[1].hist(score_sst[:, i], bins=50, alpha=0.5, label=f"{i+1}: {regime}", color=colors[i])
        axes[1].set_title(f"Distribution des scores SST ({metric.upper()}) - Lag {lag_sst}")
        axes[1].set_xlabel("Score (Plus bas = meilleur en MSE, Plus haut = meilleur en Corr)")
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(outdir, f"histograms_{metric}_lag{lag_sst}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"-> Histogrammes des distributions sauvegardés dans : {save_path}")
    # --- FIN DES AJOUTS ---

    # 5. Bilan
    hits = np.sum(true_labels == pred_labels)
    accuracy = (hits / N) * 100
    
    print(f"--- RÉSULTATS (Métrique : {metric.upper()}) ---")
    print(f"Lag évalué : SST à -{lag_sst} jours")
    print(f"Accuracy de prédiction linéaire : {accuracy:.2f}% (Seuil de hasard = 25%)")
    
    return accuracy, true_labels, pred_labels


# Ce code prédit à partir du clustering d'un certain lag de SST
# On pourrait essayer de refine pour mélanger plusieurs lags à termes mais du coup il faudrait réfléchir à comment les pondérer. 

if __name__ == "__main__":  
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch',"mac_local"])
    args = parser.parse_args()

    number_of_members = 1
    sst_lags = [35, 65, 95, 140, 175, 210, 245, 280, 315, 350] 
    #sst_lags = [35, 65, 95, 140, 175] 
    slp_lags = [15, 30, 45, 60]     



    if args.machine == 'hacienda':
        base_home = f"/home/moysan/stage_isir_jz/data_analysis/four_regimes_result_{number_of_members}_members_lags_{'_'.join(map(str, sst_lags))}_sst_{'_'.join(map(str, slp_lags))}_slp/"
    elif args.machine == 'jean-zay-work' or args.machine == 'jean-zay-scratch': 
        base_home = f"/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/four_regimes_result_{number_of_members}_members_lags_{'_'.join(map(str, sst_lags))}_sst_{'_'.join(map(str, slp_lags))}_slp/" 
    elif args.machine == 'mac_local':
        base_home = f"/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/four_regimes_result_{number_of_members}_members_lags_{'_'.join(map(str, sst_lags))}_sst_{'_'.join(map(str, slp_lags))}_slp/"

    # Création du dossier pour sauvegarder les matrices de confusion
    outdir = os.path.join(base_home, "Baseline_Evaluations_bis")
    os.makedirs(outdir, exist_ok=True)

    # 1. On fabrique la référence maître

    #train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    #nb_members_train = 1
    #train_members = train_members_87[:nb_members_train]
    #val_members = ['1001.001'] finalement on va utiliser ça comme train
    train_members = ["1001.001"]
    val_members = ["1301.010"]



    # 1. On fabrique la référence maître
    master_ref = create_master_reference(base_home, train_members, 'pca')

    # 2. Chargement des données de validation via Dataset
    print("\nChargement des données de validation...")
    # On sélectionne les mois d'hiver (comme dans ton clustering : NDJF ou JF selon ta configuration habituelle)
    winter_months = [11, 12, 1, 2] 
    # On utilise un seul lag SST pour le test (celui qu'on va évaluer)
    target_lag = 35

    val_set = Dataset(
        members=val_members, 
        selected_months=winter_months, 
        machine=args.machine, 
        target_type='map', 
        sst_lags_days=[target_lag], 
        slp_lags_days=[]
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
    print("\nLancement de l'évaluation MSE...")
    acc_mse, true_mse, pred_mse = calculate_predictability_accuracy(
        slp_samples_test, sst_samples_test, 
        ref_dict=master_ref, lag_sst=target_lag, metric='mse', outdir=outdir
    )
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
        ref_dict=master_ref, lag_sst=target_lag, metric='correlation', outdir=outdir
    )
    plot_confusion_matrix(
        y_true=true_corr, 
        y_pred=pred_corr, 
        outdir=outdir, 
        master_ref=master_ref, 
        filename=f'baseline_confusion_corr_lag{target_lag}.png'
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTemps total d'exécution : {elapsed_time:.2f} secondes")

    print(f"\nMatrices de confusion sauvegardées dans : {outdir}")