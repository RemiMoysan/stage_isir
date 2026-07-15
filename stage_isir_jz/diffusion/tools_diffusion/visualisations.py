import matplotlib.pyplot as plt
import numpy as np
import os

def save_scatter_plot_1d(targets, ensembles, current_epoch, current_crps, outdir, phase_tag="checkpoint"):
    # Création d'une figure avec 2 subplots (2 lignes, 1 colonne)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    num_samples = targets.shape[0]
    x_indices = np.arange(num_samples)
    
    # ==========================================
    # 1. SUBPLOT HAUT : Scatter Plot Classique
    # ==========================================
    for m in range(ensembles.shape[1]):
        ax1.scatter(x_indices, ensembles[:, m], color='royalblue', alpha=0.30, edgecolors='none', s=15,
                    label="Membres de diffusion" if m == 0 else "")
    
    ax1.scatter(x_indices, targets, color='crimson', marker='X', s=30, label='Cible Réelle (Target)', zorder=3)
    ax1.set_title('Visualisation Brute : Nuage de points des Membres', fontsize=11)
    ax1.set_ylabel('Espace latent $z_0$', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # ==========================================
    # 2. SUBPLOT BAS : Fan Chart (Quantiles)
    # ==========================================
    # Calcul des percentiles sur l'axe des membres (axis=1)
    q10 = np.percentile(ensembles, 10, axis=1)
    q25 = np.percentile(ensembles, 25, axis=1)
    q50 = np.percentile(ensembles, 50, axis=1) # Médiane
    q75 = np.percentile(ensembles, 75, axis=1)
    q90 = np.percentile(ensembles, 90, axis=1)
    
    # Remplissage des zones de confiance
    ax2.fill_between(x_indices, q10, q90, color='royalblue', alpha=0.20, label='Intervalle de Confiance 80% (10e-90e)')
    ax2.fill_between(x_indices, q25, q75, color='royalblue', alpha=0.40, label='Intervalle de Confiance 50% (25e-75e)')
    
    # Tracé de la médiane et de la cible
    ax2.plot(x_indices, q50, color='midnightblue', linewidth=2, label='Médiane de l\'Ensemble')
    ax2.scatter(x_indices, targets, color='crimson', marker='X', s=30, label='Cible Réelle', zorder=3)
    
    ax2.set_title('Visualisation de la Densité : Fan Chart de la Distribution Conditionnelle', fontsize=11)
    ax2.set_xlabel('Index de l\'échantillon (Chronologie de Validation)', fontsize=12)
    ax2.set_ylabel('Espace latent $z_0$', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    
    # ==========================================
    # FINITIONS ET SAUVEGARDE
    # ==========================================
    # Application du zoom sur les deux axes X (grâce au sharex=True dans les subplots)
    plt.xlim(-5, 30) 
    
    # Titre global de la figure
    fig.suptitle(f'Incertitude de la Diffusion vs Réalité - Époque {current_epoch}\nValidation CRPS Latent: {current_crps:.5f}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    # Ajustement pour que le suptitle ne chevauche pas les sous-titres
    plt.subplots_adjust(top=0.90) 
    
    fig_name = f"ensemble_sequence_plot_epoch_{current_epoch}_{phase_tag}.png"
    plt.savefig(os.path.join(outdir, fig_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Graphique de dispersion et Fan Chart sauvegardés : {fig_name}")

def plot_val_metrics(crps_history, mse_latent_history, outdir):
    epochs = range(1, len(crps_history) + 1)
    plt.figure(figsize=(12, 5))
    
    # Plot CRPS
    plt.subplot(1, 2, 1)
    plt.plot(epochs, crps_history, marker='o', color='forestgreen', label='Model CRPS')
    plt.axhline(y=0.564, color='gray', linestyle='--', label='Baseline pour prédiction 0 pour N(0,1) (0.564)')
    plt.title('Validation CRPS (Latent Space)', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('CRPS')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Plot MSE Latent
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mse_latent_history, marker='s', color='purple', label='Model MSE (Ensemble Mean)')
    plt.title('Validation MSE (Latent Ensemble Mean)', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'validation_metrics_crps_mse.png'), dpi=150)
    plt.close()