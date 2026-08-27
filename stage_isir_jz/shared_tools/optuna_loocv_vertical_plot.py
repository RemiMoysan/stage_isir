import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from optuna_plot import get_proper_metric_name, get_smart_colormap

def process_and_plot_row_means(input_csv_path, metric_name):
    """
    Charge le CSV de cross-validation, calcule la moyenne ligne par ligne,
    sauvegarde un nouveau CSV et trace une Heatmap 1D assortie à la 2D.
    """
    print(f"\n--- Génération des moyennes par ligne ({metric_name}) ---", flush=True)
    
    # Déduction automatique du dossier de sortie à partir du chemin du CSV d'entrée
    output_dir = os.path.dirname(input_csv_path)
    
    # 1. Chargement du CSV
    df_matrix = pd.read_csv(input_csv_path, index_col=0)
    
    # 2. Calcul des moyennes par ligne
    row_means = df_matrix.mean(axis=1)
    df_means = row_means.to_frame(name=f'Mean_{metric_name}')
    
    # 3. Sauvegarde dans un nouveau fichier CSV
    csv_path = os.path.join(output_dir, f"LOOCV_RowMeans_{metric_name}.csv")
    df_means.to_csv(csv_path)
    
    # 4. Génération de la Heatmap 1D
    # Remplacement manuel pour matcher exactement la figure de droite si les fonctions manquent
    try:
        proper_metric_name = get_proper_metric_name(metric_name)
        cmap = get_smart_colormap(metric_name)
    except NameError:
        proper_metric_name = "Test R²" if metric_name == "R2" else metric_name
        cmap = "magma"  # <-- C'est la colormap exacte utilisée sur ton plot 2D

    vmin = df_matrix.min().min()
    vmax = df_matrix.max().max()
    global_mean = row_means.mean()
    
    # On garde la hauteur de 10, mais on réduit un peu la largeur
    fig, ax = plt.subplots(figsize=(3, 10))
    
    mask = df_means.isna()
    
    # Dessin de la heatmap avec une colorbar plus fine pour éviter le côté "géant"
    sns.heatmap(df_means, mask=mask, ax=ax, cmap=cmap, 
                vmin=vmin, vmax=vmax, 
                xticklabels=["Mean\nScore"], 
                yticklabels=df_means.index, 
                cbar_kws={'label': f'{proper_metric_name} Score',
                          'shrink': 0.8,     # Réduit un peu la hauteur de la barre
                          'aspect': 40})     # Rend la barre plus fine
    
    # Alignement rigoureux des ticks sur l'original
    ax.tick_params(axis='x', rotation=0, labelsize=8, length=0)
    ax.tick_params(axis='y', labelsize=3, length=0)

    # Titre calqué exactement sur la matrice Pairwise (même police, même formulation)
    ax.set_title(f"Row-wise Mean Matrix — {proper_metric_name}\nMean {proper_metric_name}: {global_mean:.4f}", 
                 fontweight='bold', fontsize=14)
    
    ax.set_xlabel(" ", fontsize=10) # Garde l'espace en bas pour s'aligner avec "Test Member ID"
    ax.set_ylabel("Validation Member ID", fontsize=10)
        
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"LOOCV_Heatmap_RowMeans_{metric_name}.jpg")
    fig.savefig(plot_path, dpi=300, pil_kwargs={'quality': 90, 'subsampling': 0})
    plt.close(fig)
    
    print(f"✅ Heatmap 1D des moyennes générée : {plot_path}")
    
    return df_means

if __name__ == "__main__":
    absolute_path = "/lustre/fswork/projects/rech/uxg/uca57ub"
    input_csv = os.path.join(absolute_path, "stage_isir_jz/cnn/cnn_with_slp_embedding/optuna_embedding/loocv_embedding/LOOCV_pca1_R2_m12_bs32_dp2_dr10.086_dr20.107_fusTrue_fc36_mult1.500_grad0.928_lossmse_lr5.3e-05_feat5_noise3.1e-04_stratprogressive_poolmax_kx3_ky5_sstlags12_x2_y3_wd5.1e-06/LOOCV_Global_best_test_R2.csv")
    metric = "R2"  
    
    process_and_plot_row_means(input_csv, metric)