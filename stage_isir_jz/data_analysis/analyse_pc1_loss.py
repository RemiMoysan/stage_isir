import xarray as xr
import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import joblib
import scipy.stats as stats

# ============================================================
# 1. FONCTIONS DE PRÉPARATION
# ============================================================

def load_member_data(member, file_path_SLP, selected_months, slp_std=596.0, duree_lissage=10):
    """
    Charge un membre, filtre sur les mois d'hiver, gère les NaNs 
    et aplatit les cartes pour être ingérées par la PCA.
    Retourne également les dates pour le groupement mensuel.
    """
    file_path = os.path.join(file_path_SLP, f'PSL_anom_LE2-{member}_{duree_lissage}d.nc')
    ds = xr.open_dataset(file_path)
    da = ds["PSL"]
        
    da = da.sel(time=da['time'].dt.month.isin(selected_months))
    dates = da.time.values
    
    data = da.values
    n_samples, h, w = data.shape
    
    # Remplacement des NaNs par 0 et normalisation
    data = np.nan_to_num(data, nan=0.0) / slp_std
    
    # Aplatissement spatial (Time, 53*113)
    data_flat = data.reshape(n_samples, h * w)
    
    return data_flat, dates

# ============================================================
# 2. CONFIGURATION GÉNÉRALE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda', choices=['hacienda', 'jean-zay-work', 'jean-zay-scratch', 'mac_local'])
    parser.add_argument('--duree_lissage', type=int, default=10, help='Durée du lissage en jours (10 ou 30)')
    parser.add_argument('--model_dir', type=str, required=True, help='Chemin vers le dossier contenant best_pca_model.joblib')
    parser.add_argument('--nb_members_train', type=int, default=87, help='Nombre de membres à analyser (doit être <= 87)')
    args = parser.parse_args()

    duree_lissage = args.duree_lissage
    model_dir = args.model_dir
    model_path = os.path.join(model_dir, 'best_pca_model.joblib')

    # Liste des membres à analyser (tu peux ajuster cette liste selon tes besoins, 
    # ici on reprend la liste d'entraînement pour l'exemple)
    train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']


    nb_members_train = args.nb_members_train
    members_to_analyze = train_members_87[:nb_members_train]
    
    winter_months = [11, 12, 1, 2] # NDJF

    # Définition des chemins selon la machine
    if args.machine == 'hacienda':
        path_slp = "/data/moysan/data/SLP/"
    elif args.machine == 'jean-zay-work':
        path_slp = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/"
    elif args.machine == 'jean-zay-scratch':
        path_slp = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/SLP/"
    else: # mac_local
        path_slp = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SLP/"

    outdir = os.path.join(model_dir, f'PC1_Analysis_{nb_members_train}members')
    os.makedirs(outdir, exist_ok=True)

    # ============================================================
    # 3. CHARGEMENT DU MODÈLE PCA
    # ============================================================
    print(f"Chargement du modèle PCA depuis : {model_path}")
    ipca = joblib.load(model_path)

    # ============================================================
    # 4. PROJECTION ET EXTRACTION DES PC1
    # ============================================================
    print("\n--- Début de l'extraction des PC1 ---")
    
    all_records = []

    for i, member in enumerate(members_to_analyze):
        print(f"Traitement du membre {member} ({i+1}/{len(members_to_analyze)})...")
        
        # Chargement des données
        X_data, dates = load_member_data(member, path_slp, winter_months, duree_lissage=duree_lissage)
        
        # Projection dans l'espace latent
        latent_vectors = ipca.transform(X_data)
        
        # Récupération de la PC1 (1ère composante, index 0)
        pc1_daily = latent_vectors[:, 0]
        
        # Création d'un DataFrame temporaire pour ce membre
        df_temp = pd.DataFrame({
            'member': member,
            'date': dates,
            'pc1_daily': pc1_daily
        })
        all_records.append(df_temp)

    # Concaténation de tous les membres
    df_all = pd.concat(all_records, ignore_index=True)

    # Extraction de l'année et du mois (gère datetime64 et cftime)
    df_all['year'] = [d.year for d in df_all['date']]
    df_all['month'] = [d.month for d in df_all['date']]

    # ============================================================
    # 4.5. ANALYSE DE LA QUANTILE LOSS (Avant moyennage)
    # ============================================================
    print("\n--- Analyse de la Quantile Loss Journalière ---")
    
    def compute_and_plot_quantile_analysis(df_subset, title_suffix, filename):
        """Fonction interne pour calculer les baselines et tracer le graphique"""
        pc1_vals = df_subset['pc1_daily'].values
        
        # 1. Calcul de l'espérance de la valeur absolue
        e_abs = np.mean(np.abs(pc1_vals))
        demi_e_abs = e_abs / 2.0
        
        # 2. Définition des quantiles
        quantiles_q = np.arange(0.1, 1.0, 0.1)
        
        def pinball_loss(y_true, y_pred, q):
            error = y_true - y_pred
            return np.mean(np.maximum(q * error, (q - 1) * error))

        loss_base_0 = []
        loss_base_q_empirique = []
        loss_theorique_normale = []
        
        std_pc1 = np.std(pc1_vals)

        for q in quantiles_q:
            # A. Baseline 1 : Prédire 0
            loss_base_0.append(pinball_loss(pc1_vals, 0.0, q))
            
            # B. Baseline 2 : Prédire le quantile empirique de ce subset
            q_val_empirique = np.quantile(pc1_vals, q)
            loss_base_q_empirique.append(pinball_loss(pc1_vals, q_val_empirique, q))
            
            # C. Théorique Gaussienne basée sur le STD de ce subset
            z_q = stats.norm.ppf(q)
            loss_theorique_normale.append(std_pc1 * stats.norm.pdf(z_q))

        mean_loss_base_q = np.mean(loss_base_q_empirique)
        
        # 3. Visualisation
        plt.figure(figsize=(10, 6))
        plt.plot(quantiles_q, loss_base_0, label="Baseline 1 (Prédire 0)", color='red', marker='o', linewidth=2)
        plt.plot(quantiles_q, loss_base_q_empirique, label="Baseline 2 (Prédire quantile empirique)", color='blue', marker='s', linewidth=2)
        plt.plot(quantiles_q, loss_theorique_normale, label="Théorique (Gaussienne)", color='black', linestyle='--', linewidth=2, alpha=0.8)
        plt.axhline(y=demi_e_abs, color='green', linestyle=':', linewidth=3, label="Empirique : 1/2 E(|PC1|)")
        plt.axhline(y=mean_loss_base_q, color='purple', linestyle='-.', linewidth=2, label=f"Moyenne Baseline 2 ({mean_loss_base_q:.3f})")

        plt.title(f"Vérification de la Quantile Loss - {title_suffix}")
        plt.xlabel("Quantile visé (q)")
        plt.ylabel("Quantile Loss Moyenne")
        plt.xticks(quantiles_q)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close()
        print(f"--> Graphique sauvegardé : {filename} (STD journalier du subset = {std_pc1:.4f})")

    # --- Éxécution pour les 3 cas demandés ---
    
    # 1. Cas Global (Tout l'hiver NDJF)
    compute_and_plot_quantile_analysis(df_all, "Globale (Hiver complet NDJF)", 'quantile_loss_baselines_global.png')
    
    # 2. Cas Novembre - Décembre (Début d'hiver)
    df_nov_dec = df_all[df_all['month'].isin([11, 12])]
    compute_and_plot_quantile_analysis(df_nov_dec, "Novembre & Décembre", 'quantile_loss_baselines_nov_dec.png')
    
    # 3. Cas Janvier - Février (Cœur d'hiver)
    df_jan_fev = df_all[df_all['month'].isin([1, 2])]
    compute_and_plot_quantile_analysis(df_jan_fev, "Janvier & Février", 'quantile_loss_baselines_jan_fev.png')

    # ============================================================
    # 5. CALCUL DES MOYENNES MENSUELLES ET STATISTIQUES
    # ============================================================
    print("\n--- Calcul des statistiques mensuelles ---")
    
    # 1. Écart type de la PC1 journalière (devrait être proche de 1 si whiten=True)
    std_daily_total = df_all['pc1_daily'].std()
    print(f"Écart-type de la PC1 journalière globale : {std_daily_total:.4f}")

    # 2. Calcul de la moyenne de la PC1 pour chaque (membre, année, mois)
    df_monthly_means = df_all.groupby(['member', 'year', 'month'])['pc1_daily'].mean().reset_index()
    df_monthly_means.rename(columns={'pc1_daily': 'pc1_monthly_mean'}, inplace=True)

    # 3. Calcul de l'écart-type de ces moyennes, classé par mois
    std_by_month = df_monthly_means.groupby('month')['pc1_monthly_mean'].std()
    
    print("\nÉcart-type des PC1 moyennes mensuelles par mois :")
    for month in winter_months:
        print(f"  Mois {month} : {std_by_month.get(month, np.nan):.4f}")

    # ============================================================
    # 6. VISUALISATIONS
    # ============================================================
    print("\n--- Génération des graphiques ---")
    
    # Paramètres de plot
    month_names = {11: 'Novembre', 12: 'Décembre', 1: 'Janvier', 2: 'Février'}
    
    # FIGURE 1 : Histogramme global des PC1 (Journalier vs Moyenne Mensuelle)
    plt.figure(figsize=(10, 6))
    plt.hist(df_all['pc1_daily'], bins=50, density=True, alpha=0.5, label=f'PC1 Journalière (std={std_daily_total:.2f})')
    plt.hist(df_monthly_means['pc1_monthly_mean'], bins=50, density=True, alpha=0.7, label=f"PC1 Moyenne Mensuelle (std={df_monthly_means['pc1_monthly_mean'].std():.2f})")
    
    plt.axvline(0, color='black', linestyle='dashed', linewidth=1)
    plt.title('Distribution de la PC1 : Journalière vs Moyenne Mensuelle')
    plt.xlabel('Valeur de la PC1')
    plt.ylabel('Densité')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'hist_pc1_daily_vs_monthly.png'), dpi=150)
    plt.close()

    # FIGURE 2 : Boxplot des moyennes mensuelles par mois (Montre la distribution et l'écart type)
    plt.figure(figsize=(8, 6))
    
    boxplot_data = [df_monthly_means[df_monthly_means['month'] == m]['pc1_monthly_mean'].values for m in winter_months]
    labels = [month_names[m] for m in winter_months]
    
    plt.boxplot(boxplot_data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    
    # Ajout du texte de l'écart-type au-dessus de chaque box
    for i, m in enumerate(winter_months):
        std_val = std_by_month.get(m, np.nan)
        plt.text(i + 1, plt.gca().get_ylim()[1] * 0.9, f"std = {std_val:.2f}", 
                 horizontalalignment='center', color='darkblue', fontweight='bold')

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Distribution des PC1 Moyennes Mensuelles par Mois (Hiver)")
    plt.ylabel("Valeur de la PC1 moyenne mensuelle")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'boxplot_pc1_monthly_by_month.png'), dpi=150)
    plt.close()

    print(f"Analyses et figures sauvegardées dans : {outdir}")