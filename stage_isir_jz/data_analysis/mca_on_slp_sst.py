import xarray as xr
import pandas as pd
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
from scipy.linalg import svd
from datetime import timedelta
import joblib



# ============================================================
# 1. FONCTIONS DE PRÉPARATION DE DONNÉES
# ============================================================

def load_member_mca_data(member, path_SST, path_SLP, selected_months, sst_lags_days=[35], duree_lissage=10, sst_std=0.707, slp_std=596.0):
    """
    Charge les paires (SST retardée, SLP actuelle) pour un membre.
    Les différents lags de SST sont concaténés comme des "features" supplémentaires.
    """
    # Chargement des datasets
    ds_slp = xr.open_dataset(os.path.join(path_SLP, f'PSL_anom_LE2-{member}_{duree_lissage}d.nc'))["PSL"]
    ds_sst = xr.open_dataset(os.path.join(path_SST, f'SST_anom_LE2-{member}_T_regrid.nc'))["SST"]
    
    # 1. Identifier les dates cibles valides pour la SLP (mois d'hiver)
    slp_winter = ds_slp.sel(time=ds_slp['time'].dt.month.isin(selected_months))
    years = slp_winter['time'].dt.year
    
    # Éviter les bornes extrêmes où les lags passés pourraient ne pas exister
    valid_mask = (years > years.min()) & (years < years.max() - 1)
    target_dates = slp_winter.sel(time=valid_mask).time.values
    
    # 2. Extraction vectorisée / itérative
    X_sst_list = []
    Y_slp_list = []
    valid_target_dates = []

    # L'extraction date par date avec xarray peut être lente. On boucle sur les dates.
    # Pour des perfs optimales, on pourrait utiliser des indexateurs numpy, mais xarray `.sel` garantit l'alignement.
    for t_target in target_dates:
        # Reconstitution de la date (gestion des types cftime)
        if isinstance(t_target, np.datetime64):
            t_obj = pd.Timestamp(t_target)
        else:
            t_obj = t_target # cftime object
            
        dates_sst = [t_obj - timedelta(days=d) for d in sst_lags_days]
        
        try:
            # Extraction SLP
            slp_t = ds_slp.sel(time=t_obj)
            
            # Extraction SST (on se restreint à la même zone que dans le Dataset PyTorch)
            sst_lags = ds_sst.sel(time=dates_sst, lat=slice(-15, 70))
            
            # Traitement NaNs et Normalisation
            slp_np = np.nan_to_num(slp_t.values, nan=0.0) / slp_std
            sst_np = np.nan_to_num(sst_lags.values, nan=0.0) / sst_std
            
            # Aplatissement. Si plusieurs lags, on les met à plat avec l'espace (features = lags * lat * lon)
            Y_slp_list.append(slp_np.flatten())
            X_sst_list.append(sst_np.flatten())
            valid_target_dates.append(t_target)
            
        except KeyError:
            # Si un lag déborde du calendrier disponible, on passe
            continue
            
    X_mat = np.array(X_sst_list) # Shape: (N, lags * lat_sst * lon_sst)
    Y_mat = np.array(Y_slp_list) # Shape: (N, lat_slp * lon_slp)
    
    # Récupération des dimensions spatiales pour la reconstruction future
    shape_sst = ds_sst.sel(lat=slice(-15, 70)).shape[1:] # (lat, lon)
    shape_slp = ds_slp.shape[1:]                         # (lat, lon)
    
    return X_mat, Y_mat, valid_target_dates, shape_sst, shape_slp


# ============================================================
# 2. FONCTIONS DE VISUALISATION (MCA)
# ============================================================

def plot_mca_modes(U, V, s, shape_sst, shape_slp, sst_lags_days, outdir, n_modes=5):
    """
    Affiche les premiers modes couplés de la MCA.
    S'adapte dynamiquement pour afficher tous les lags fournis pour la SST.
    """
    scf = (s**2) / np.sum(s**2) * 100 # Squared Covariance Fraction
    
    h_sst, w_sst = shape_sst
    h_slp, w_slp = shape_slp
    num_lags = len(sst_lags_days)
    
    # Création d'une grille adaptative : n_modes lignes, (num_lags + 1) colonnes
    fig, axes = plt.subplots(n_modes, num_lags + 1, figsize=(6 * (num_lags + 1), 4 * n_modes))
    fig.suptitle(f"Maximum Covariance Analysis (MCA) - Top {n_modes} Modes", fontsize=18)
    
    # Sécurité si un seul mode est demandé (axes devient 1D au lieu de 2D)
    if n_modes == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(n_modes):
        # Reconstruction 3D (lags, lat, lon) pour la SST
        mode_sst_full = U[:, i].reshape((num_lags, h_sst, w_sst))
        
        # Reconstruction 2D (lat, lon) pour la SLP
        mode_slp = V[i, :].reshape((h_slp, w_slp))
        
        # On calcule une limite de couleur globale pour TOUS les lags de la SST 
        # afin que l'échelle d'intensité soit comparable entre les différentes temporalités
        vlim_sst = np.max(np.abs(mode_sst_full))
        vlim_slp = np.max(np.abs(mode_slp))
        
        # 1. Plot de tous les Lags de SST
        for l in range(num_lags):
            ax_sst = axes[i, l]
            im_sst = ax_sst.imshow(mode_sst_full[l], cmap='RdBu_r', origin='lower', vmin=-vlim_sst, vmax=vlim_sst)
            ax_sst.set_title(f"Mode {i+1} SST (-{sst_lags_days[l]} jours)")
            
            # On n'ajoute la barre de couleur qu'au dernier lag SST pour ne pas surcharger la figure
            if l == num_lags - 1:
                fig.colorbar(im_sst, ax=ax_sst, fraction=0.046, pad=0.04)
        
        # 2. Plot de la SLP (dernière colonne)
        ax_slp = axes[i, num_lags]
        im_slp = ax_slp.imshow(mode_slp, cmap='RdBu_r', origin='lower', vmin=-vlim_slp, vmax=vlim_slp)
        ax_slp.set_title(f"Mode {i+1} SLP (SCF = {scf[i]:.2f}%)")
        fig.colorbar(im_slp, ax=ax_slp, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    # On descend légèrement le titre principal pour qu'il ne chevauche pas les sous-titres
    plt.subplots_adjust(top=1 - (0.05 / n_modes)) 
    plt.savefig(os.path.join(outdir, "mca_coupled_modes.png"), dpi=150)
    plt.close()

def plot_scf(s, outdir, n_modes_plot=50):
    """Plot de la fraction de covariance au carré."""
    scf = (s**2) / np.sum(s**2)
    scf_cum = np.cumsum(scf)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_modes_plot + 1), scf_cum[:n_modes_plot], marker='o', color='red', label='SCF cumulée')
    plt.bar(range(1, n_modes_plot + 1), scf[:n_modes_plot], alpha=0.5, label='SCF par mode')
    
    plt.xlabel('Numéro du mode')
    plt.ylabel('Fraction de covariance expliquée (SCF)')
    plt.title('Maximum Covariance Analysis : Spectre des valeurs singulières')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'mca_scf_variance.png'), dpi=150)
    plt.close()

def plot_mca_metrics_dashboard(RV_coeff, R_value, VF, HVF_X, HVF_Y, A1, B1, outdir):
    """
    Génère un dashboard synthétique pour les métriques globales et les projections temporelles.
    """
    fig = plt.figure(figsize=(15, 7)) # Légèrement agrandi pour les 5 barres
    fig.suptitle("Synthèse des Métriques de Couplage (SST vs SLP)", fontsize=16, fontweight='bold')
    
    # --- Sous-graphe 1 : Jauges / Barres des métriques ---
    ax1 = plt.subplot(1, 2, 1)
    metrics = [
        'Coefficient RV\n(Couplage spatial global)', 
        '|R| Temporel\n(Corrélation Mode 1)', 
        'Var. interne SST\nexpliquée par son Mode 1 (HVF)',
        'Var. interne SLP\nexpliquée par son Mode 1 (HVF)',
        'Var. SLP expliquée\npar la SST (VF croisée)'
    ]
    
    # On met tout en proportion (0-1) pour harmoniser l'échelle visuelle
    values = [RV_coeff, np.abs(R_value), HVF_X / 100, HVF_Y / 100, VF / 100] 
    actual_texts = [f"{RV_coeff:.4f}", f"{R_value:.3f}", f"{HVF_X:.2f} %", f"{HVF_Y:.2f} %", f"{VF:.2f} %"]
    
    y_pos = np.arange(len(metrics))
    # Nouvelles couleurs pour bien distinguer (Global, Temporel, Interne, Interne, Croisé)
    colors = ['#2ca02c', '#1f77b4', '#9467bd', '#e377c2', '#ff7f0e']
    
    bars = ax1.barh(y_pos, values, color=colors, alpha=0.8)
    ax1.set_xlim(0, 1.05) 
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics, fontsize=11)
    ax1.invert_yaxis() 
    ax1.set_title("Indicateurs d'intensité et de représentativité", fontsize=14)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Ajouter le texte exact sur/à côté des barres
    for i, bar in enumerate(bars):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                 actual_texts[i], va='center', fontsize=12, fontweight='bold', color='black')

    # Boîte de texte explicative mise à jour
    explications = (
        "• RV : Force de la superposition globale des deux grandeurs\n"
        "• HVF : Poids physique du mode au sein de son propre fluide\n"
        "• VF croisée : Capacité de l'océan (SST) à forcer/expliquer l'atmosphère (SLP)"
    )
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8)
    ax1.text(0.02, -0.25, explications, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=props, style='italic')

    # --- Sous-graphe 2 : Scatter plot A1 vs B1 ---
    ax2 = plt.subplot(1, 2, 2)
    
    A1_std = (A1 - np.mean(A1)) / np.std(A1)
    B1_std = (B1 - np.mean(B1)) / np.std(B1)
    
    alpha_val = max(0.05, min(0.4, 500 / len(A1_std))) 
    ax2.scatter(A1_std, B1_std, alpha=alpha_val, edgecolors='none', color='#1f77b4')
    
    m, b = np.polyfit(A1_std, B1_std, 1)
    x_line = np.array([np.min(A1_std), np.max(A1_std)])
    ax2.plot(x_line, m*x_line + b, color='#d62728', linewidth=2.5, 
             label=f'Régression (R = {R_value:.3f})')
    
    ax2.set_title("Diagramme de dispersion (Mode 1)", fontsize=14)
    ax2.set_xlabel("Amplitude normalisée SST ($A_1$)")
    ax2.set_ylabel("Amplitude normalisée SLP ($B_1$)")
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    plt.savefig(os.path.join(outdir, "mca_metrics_dashboard.png"), dpi=200, bbox_inches='tight')
    plt.close()
# ============================================================
# 3. SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='hacienda')
    parser.add_argument('--duree_lissage', type=int, default=10)
    # On peut rajouter des arguments pour gérer plusieurs lags
    parser.add_argument('--sst_lags', type=int, nargs='+', default=[35], help='Lags en jours pour la SST (ex: 35 65 95)')
    args = parser.parse_args()

    train_members_87 = ['1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']
    
    nb_members_train = len(train_members_87)
    train_members = train_members_87
    winter_months = [11, 12, 1, 2] # NDJF
    

    if args.machine == 'hacienda':
        path_SLP = "/data/moysan/data/SLP/"
        path_SST = "/data/moysan/data/SST/"
        base_home = "/home/moysan/stage_isir_jz/data_analysis/mca_slp_sst/"
    elif args.machine == 'jean-zay-work':
        path_SLP = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SLP/"
        path_SST = "/lustre/fswork/projects/rech/uxg/uca57ub/data/SST/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/mca_slp_sst/"
    elif args.machine == 'jean-zay-scratch':
        path_SLP = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/SLP/"
        path_SST = "/lustre/fsn1/projects/rech/uxg/uca57ub/data/SST/"
        base_home = "/lustre/fswork/projects/rech/uxg/uca57ub/stage_isir_jz/data_analysis/mca_slp_sst/"
    else: # mac_local
        path_SLP = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SLP/"
        path_SST = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/data/SST/"
        base_home = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/mca_slp_sst/"

    
    outdir_name = f'MCA_NDJF_{nb_members_train}membres_lags_{"_".join(map(str, args.sst_lags))}_lissage_{args.duree_lissage}d'
    outdir = os.path.join(base_home, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    print(f"Lancement de la MCA avec les lags de SST : {args.sst_lags} jours")

    # ============================================================
    # CALCUL INCRÉMENTAL DES MATRICES DE COVARIANCE
    # ============================================================
    start_time = time.time()
    
    C = None # Matrice de covariance croisée (X^T Y)
    Cxx = None # Auto-covariance de X
    Cyy = None # Auto-covariance de Y
    N_total = 0 # Nombre total d'échantillons
    
    for i, member in enumerate(train_members):
        print(f"Traitement du membre {member} ({i+1}/{nb_members_train})...")
        X, Y, dates, shape_sst, shape_slp = load_member_mca_data(
            member, path_SST, path_SLP, winter_months, 
            sst_lags_days=args.sst_lags, duree_lissage=args.duree_lissage
        )
        
        N_total += X.shape[0]
        
        if C is None:
            C = X.T @ Y
            Cxx = X.T @ X
            Cyy = Y.T @ Y
        else:
            C += X.T @ Y
            Cxx += X.T @ X
            Cyy += Y.T @ Y

    # Normalisation pour obtenir les vraies matrices de covariance
    C = C / (N_total - 1)
    Cxx = Cxx / (N_total - 1)
    Cyy = Cyy / (N_total - 1)
    
    print(f"Matrices de covariance calculées. Dimension croisée : {C.shape}. Calcul de la SVD...")

    # ============================================================
    # SVD ET EXTRACTION DES MODES
    # ============================================================
    
    # full_matrices=False permet de gagner drastiquement en temps et en mémoire
    U, s, Vt = svd(C, full_matrices=False)
    
    print("SVD terminée. Génération des plots...")
    
    # Tracer le spectre des valeurs singulières (SCF)
    plot_scf(s, outdir, n_modes_plot=50)
    
    # Tracer les 5 premiers modes couplés
    plot_mca_modes(U, Vt, s, shape_sst, shape_slp, args.sst_lags, outdir, n_modes=5)
    
    # Sauvegarde du modèle (les matrices) au cas où l'on veut calculer des projections (expansion coefficients) plus tard
    model_path = os.path.join(outdir, 'mca_model.joblib')
    joblib.dump({'U': U, 'V': Vt.T, 's': s, 'shape_sst': shape_sst, 'shape_slp': shape_slp}, model_path)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Exécution complète en {elapsed:.2f} minutes.")
    print(f"Résultats et figures sauvegardés dans : {outdir}")

    # ============================================================
    # 4. CALCUL ET SAUVEGARDE DES MÉTRIQUES GLOBALES ET PAR MODE
    # ============================================================
    print("\n--- Calcul des métriques de corrélation ---")
    
    # 1. Coefficient RV
    num_RV = np.sum(s**2) 
    den_RV = np.sqrt(np.sum(Cxx**2) * np.sum(Cyy**2))
    RV_coeff = num_RV / den_RV
    print(f"Coefficient RV global : {RV_coeff:.4f}")

    # 2. Homogeneous Variance Fraction (HVF) pour le Mode 1
    # Calcul matriciel direct de la variance expliquée (Traces)
    trace_Cxx = np.trace(Cxx)
    trace_Cyy = np.trace(Cyy)
    
    # U[:, 0] et Vt[0, :] sont des vecteurs 1D. La forme u.T @ M @ u donne la variance projetée.
    var_expl_X = U[:, 0] @ Cxx @ U[:, 0] 
    HVF_X = (var_expl_X / trace_Cxx) * 100
    
    var_expl_Y = Vt[0, :] @ Cyy @ Vt[0, :]
    HVF_Y = (var_expl_Y / trace_Cyy) * 100
    
    print(f"Variance interne SST expliquée par son Mode 1 (HVF) : {HVF_X:.2f}%")
    print(f"Variance interne SLP expliquée par son Mode 1 (HVF) : {HVF_Y:.2f}%")

    # 3. Séries temporelles et Fraction de Variance Croisée (VF)
    mode_idx = 0
    A1_list, B1_list, Y_all_list = [], [], []
    
    for member in train_members:
        X, Y, _, _, _ = load_member_mca_data(
            member, path_SST, path_SLP, winter_months, 
            sst_lags_days=args.sst_lags, duree_lissage=args.duree_lissage
        )
        A1_list.append(X @ U[:, mode_idx])
        B1_list.append(Y @ Vt[mode_idx, :].T)
        Y_all_list.append(Y)
        
    A1 = np.concatenate(A1_list)
    B1 = np.concatenate(B1_list)
    Y_mat_all = np.concatenate(Y_all_list)

    # Corrélation Temporelle (R)
    R_value = np.corrcoef(A1, B1)[0, 1]
    print(f"Corrélation temporelle R (Mode 1) : {R_value:.3f}")

    # Fraction de Variance (VF croisée)
    A1_norm = (A1 - np.mean(A1)) / np.std(A1)
    Y_predicted_by_X = np.outer(A1_norm, (A1_norm.T @ Y_mat_all) / len(A1_norm))
    var_explained = np.var(Y_predicted_by_X)
    var_total = np.var(Y_mat_all)
    VF = (var_explained / var_total) * 100
    print(f"Fraction de Variance croisée (VF SLP expliquée par SST) : {VF:.2f}%")

    # === GÉNÉRATION DU PLOT DE SYNTHÈSE ===
    print("Génération du dashboard des métriques...")
    plot_mca_metrics_dashboard(RV_coeff, R_value, VF, HVF_X, HVF_Y, A1, B1, outdir)
    print(f"Dashboard sauvegardé dans : {os.path.join(outdir, 'mca_metrics_dashboard.png')}")