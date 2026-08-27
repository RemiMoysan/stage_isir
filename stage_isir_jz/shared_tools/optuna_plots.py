import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import matplotlib.cm as cm
import matplotlib.colors as mcolors

ALL_MEMBERS = ['1001.001', '1041.003', '1061.004', '1081.005', '1101.006', '1121.007', '1141.008', '1161.009', '1181.010', '1231.001', '1231.002', '1231.003', '1231.004', '1231.005', '1231.006', '1231.007', '1231.008', '1231.009', '1231.010', '1231.011', '1231.012', '1231.013', '1231.014', '1231.015', '1231.016', '1231.017', '1231.018', '1231.019', '1231.020', '1251.001', '1251.002', '1251.003', '1251.004', '1251.005', '1251.006', '1251.007', '1251.008', '1251.009', '1251.010', '1251.011', '1251.012', '1251.013', '1251.014', '1251.015', '1251.016', '1251.017', '1251.018', '1251.019', '1251.020', '1281.001', '1281.002', '1281.003', '1281.004', '1281.005', '1281.006', '1281.007', '1281.008', '1281.009', '1281.010', '1281.011', '1281.012', '1281.013', '1281.014', '1281.015', '1281.016', '1281.017', '1281.018', '1281.019', '1281.020', '1301.001', '1301.002', '1301.003', '1301.004', '1301.005', '1301.006', '1301.007', '1301.008', '1301.009', '1301.010', '1301.011', '1301.012', '1301.013', '1301.014', '1301.015', '1301.016', '1301.017', '1301.018', '1301.019', '1301.020']

def get_smart_colormap(metric_name):
    """
    Définit la colormap pour que la 'meilleure' performance soit claire (blanc/jaune)
    et la 'pire' performance soit sombre (noir).
    """
    # L1, MSE, Loss : Plus c'est petit, mieux c'est. 
    # magma_r met les petites valeurs en clair, et les grandes en noir.
    if 'L1' in metric_name or 'mse' in metric_name.lower() or 'loss' in metric_name.lower():
        return 'magma_r'
    
    # R2, Correlation : Plus c'est grand, mieux c'est.
    # magma met les grandes valeurs en clair, et les petites en noir.
    else:
        return 'magma'

def get_proper_metric_name(metric_name):
    if "R2" in metric_name: return "Test R²"
    if "L1" in metric_name: return "Test SS-L1"
    if "corr" in metric_name.lower(): return "Test Correlation"
    return metric_name

def generate_crossval_matrix(study, output_dir, metric_name):
    """Génère la Heatmap 2D et le CSV globaux à partir de l'étude Optuna."""
    print(f"\n--- Génération de la Matrice croisée Globale ({metric_name}) ---", flush=True)
    
    M = len(ALL_MEMBERS)
    proper_metric_name = get_proper_metric_name(metric_name)
    mat_c = np.full((M, M), np.nan, dtype=np.float64)
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            v_mem = trial.params.get('val_member')
            t_mem = trial.params.get('test_member')
            
            if v_mem in ALL_MEMBERS and t_mem in ALL_MEMBERS:
                v_idx = ALL_MEMBERS.index(v_mem)
                t_idx = ALL_MEMBERS.index(t_mem)
                mat_c[v_idx, t_idx] = trial.user_attrs.get(metric_name, np.nan)

    global_mean = np.nanmean(mat_c)

    df_w = pd.DataFrame(mat_c, index=ALL_MEMBERS, columns=ALL_MEMBERS)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"LOOCV_Global_{metric_name}.csv")
    df_w.to_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.isnan(mat_c)
    cmap = get_smart_colormap(metric_name)
    
    sns.heatmap(mat_c, mask=mask, ax=ax, cmap=cmap, 
                xticklabels=ALL_MEMBERS, 
                yticklabels=ALL_MEMBERS, 
                cbar_kws={'label': f'{proper_metric_name} Score'})
    
    ax.tick_params(axis='x', rotation=90, labelsize=3, length=0)
    ax.tick_params(axis='y', labelsize=3, length=0)

    ax.set_title(f"Pairwise LOOCV Matrix — {proper_metric_name}\nMean {proper_metric_name}: {global_mean:.4f}", fontweight='bold', fontsize=14)
    ax.set_xlabel("Test Member ID")
    ax.set_ylabel("Validation Member ID")
        
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"LOOCV_Heatmap_Global_{metric_name}.jpg")
    fig.savefig(plot_path, dpi=300, pil_kwargs={'quality': 90, 'subsampling': 0})
    plt.close(fig)
    
    print(f"✅ Matrice globale générée : {plot_path}")


def generate_1d_loocv_heatmap(study, output_dir, metric_name, val_member_used):
    """
    Génère un Barplot 1D coloré calqué sur le full-grid ALL_MEMBERS.
    Les bugs d'affichage de Matplotlib sont corrigés via ax.text et bbox_inches.
    """
    print(f"\n--- Génération du Barplot 1D Global ({metric_name} - Val: {val_member_used}) ---", flush=True)
    
    M = len(ALL_MEMBERS)
    proper_metric_name = get_proper_metric_name(metric_name)
    
    scores = np.full(M, np.nan, dtype=np.float64)
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            v_mem = trial.params.get('val_member')
            t_mem = trial.params.get('test_member')
            
            if v_mem == val_member_used and t_mem in ALL_MEMBERS:
                t_idx = ALL_MEMBERS.index(t_mem)
                scores[t_idx] = trial.user_attrs.get(metric_name, np.nan)

    global_mean = np.nanmean(scores)
    os.makedirs(output_dir, exist_ok=True)
    
    if np.isnan(global_mean):
        print(f"⚠️ Pas assez de données pour plotter le 1D Barplot de {metric_name}.")
        return

    # On aère un tout petit peu en hauteur (3.5 au lieu de 3)
    fig, ax = plt.subplots(figsize=(15, 3.5))
    
    # --- GESTION DES COULEURS ---
    cmap_name = get_smart_colormap(metric_name)
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=np.nanmin(scores), vmax=np.nanmax(scores))
    bar_colors = [cmap(norm(val)) if not np.isnan(val) else (0,0,0,0) for val in scores]

    # --- TRACÉ DES BARRES ---
    x_positions = np.arange(M)
    ax.bar(x_positions, scores, color=bar_colors, edgecolor='black', linewidth=0.3, width=0.9)
    
    # --- ESTHÉTIQUE DE L'AXE X ---
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ALL_MEMBERS, rotation=90, fontsize=5)
    ax.set_xlim(-0.5, M - 0.5) 
    
    # --- DÉPLACEMENT DES GRADUATIONS À DROITE ---
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Score", fontsize=10, labelpad=10)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5) 
    
    # --- AJOUT DU TEXTE À GAUCHE (Méthode propre sans twinx) ---
    # transform=ax.transAxes permet de se placer par rapport à la boîte du graphe (0=gauche, 1=droite)
    # x=-0.01 décale le texte juste en dehors de la boîte, à gauche.
    ax.text(-0.01, 0.5, f"Val:\n{val_member_used}", transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='center', ha='right')

    # --- AJOUT DE LA COLORBAR ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # fraction : finesse de la barre, pad : marge forcée entre l'axe et la barre pour éviter l'overlap
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.08)
    cbar.set_label(f'{proper_metric_name}')
    
    ax.set_title(f"1D LOOCV per Test Member — {proper_metric_name}\nMean {proper_metric_name}: {global_mean:.4f}", fontweight='bold', fontsize=12)
    ax.set_xlabel("Test Member ID")
    
    fig.tight_layout()
    barplot_path = os.path.join(output_dir, f"LOOCV_1D_BarPlot_Global_{metric_name}.jpg")
    
    # bbox_inches='tight' est CRUCIAL ici : il garantit que le texte ajouté "hors cadre" ne sera pas rogné
    fig.savefig(barplot_path, dpi=200, bbox_inches='tight', pil_kwargs={'quality': 90})
    plt.close(fig)
    
    print(f"✅ Barplot 1D global généré : {barplot_path}")