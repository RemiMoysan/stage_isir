import optuna
import os
import argparse
import numpy as np
import plotly.graph_objects as go

parser = argparse.ArgumentParser()
parser.add_argument("--study-name", help="Nom de l'étude Optuna", default=None)
parser.add_argument("--storage-name", help="Nom du stockage Optuna (chemin vers le fichier .db)", required=True)
args = parser.parse_args()

# 1. Construire les chemins
db_absolute_path = os.path.join("/lustre/fswork/projects/rech/uxg/uca57ub/", args.storage_name)
storage_name = f"sqlite:///{db_absolute_path}"
output_dir = os.path.dirname(db_absolute_path)
print(f"Dossier cible pour les graphiques : {output_dir}")

# 2. Récupérer l'étude (Logique de Fallback)
summaries = optuna.get_all_study_summaries(storage=storage_name)
available_studies = [s.study_name for s in summaries]

study_name = args.study_name
if study_name not in available_studies:
    if not available_studies:
        raise ValueError(f"Le fichier .db est vide. Aucune étude trouvée dans : {storage_name}")
    fallback_name = available_studies[0]
    print(f"\n⚠️ L'étude '{study_name}' est introuvable. Chargement automatique de : '{fallback_name}'\n")
    study_name = fallback_name

study = optuna.load_study(study_name=study_name, storage=storage_name)
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if not complete_trials:
    print("Aucun essai terminé trouvé.")
    exit()

# =====================================================================
# GRAPHIQUE 1 : COURBES D'ENTRAÎNEMENT INTERACTIVES (r² et R²)
# =====================================================================
print(f"Génération du graphique interactif pour {len(complete_trials)} essais LOOCV...")

# On trie les essais du MEILLEUR au PIRE selon le R2 de validation
complete_trials.sort(key=lambda t: t.user_attrs.get("best_R2_score", -float('inf')), reverse=True)

fig_curves = go.Figure()

# Préparation des listes de visibilité pour le menu déroulant
vis_all, vis_top10, vis_worst10, vis_pos_r2, vis_neg_r2 = [], [], [], [], []

for i, trial in enumerate(complete_trials):
    history = trial.user_attrs.get("R2_corr_history")
    score_r2 = trial.user_attrs.get("best_R2_score", 0)
    score_r = trial.user_attrs.get("best_trial_corr", 0)
    val_member = trial.user_attrs.get("val_member", f"Trial {trial.number}")
    
    if history:
        epochs = [h[0] for h in history]
        R2_scores = [h[1] for h in history]
        r2_scores = [h[2]**2 for h in history] # r² = corrélation au carré
        
        nom_legende = f"{val_member} (R²={score_r2:.2f}, r²={score_r**2:.2f})"
        
        # Courbe r² (Corrélation au carré) en ligne continue
        fig_curves.add_trace(go.Scatter(
            x=epochs, y=r2_scores, mode='lines',
            name=f'r² - {nom_legende}',
            line=dict(width=1.5), opacity=0.8,
            legendgroup=val_member
        ))
        
        # Courbe R² (Skill score) en pointillé
        fig_curves.add_trace(go.Scatter(
            x=epochs, y=R2_scores, mode='lines',
            name=f'R² - {nom_legende}',
            line=dict(width=1.5, dash='dash'), opacity=0.8,
            legendgroup=val_member, showlegend=False # Masqué dans la légende pour ne pas la doubler
        ))

        # Remplissage des tableaux de visibilité (2 traces ajoutées par essai)
        vis_all.extend([True, True])
        vis_top10.extend([True, True] if i < 10 else [False, False])
        vis_worst10.extend([True, True] if i >= len(complete_trials) - 10 else [False, False])
        vis_pos_r2.extend([True, True] if score_r2 > 0 else [False, False])
        vis_neg_r2.extend([True, True] if score_r2 <= 0 else [False, False])

# Création du Menu Déroulant (Dropdown) pour filtrer interactivement
updatemenus = [dict(
    active=0,
    buttons=list([
        dict(label="Tous les membres", method="update", args=[{"visible": vis_all}]),
        dict(label="Top 10 (Meilleurs)", method="update", args=[{"visible": vis_top10}]),
        dict(label="Pires 10", method="update", args=[{"visible": vis_worst10}]),
        dict(label="Filtre : R² > 0", method="update", args=[{"visible": vis_pos_r2}]),
        dict(label="Filtre : R² < 0", method="update", args=[{"visible": vis_neg_r2}])
    ]),
    x=0.0, y=1.15, xanchor="left", yanchor="top"
)]

fig_curves.update_layout(
    title="Historique d'entraînement LOOCV : r² (continu) et R² (pointillé)",
    xaxis_title="Époques",
    yaxis_title="Score",
    template="plotly_white",
    hovermode="x unified",
    updatemenus=updatemenus,
    legend=dict(yanchor="top", y=1, xanchor="left", x=1.02) # Légende décalée à droite
)
fig_curves.write_html(os.path.join(output_dir, "loocv_history_curves.html"))


# =====================================================================
# GRAPHIQUES 2 & 3 : HISTOGRAMMES (SAUVEGARDÉS DIRECTEMENT EN PNG)
# =====================================================================
print("Génération des histogrammes (PNG)...")

# Extraction des métriques finales
val_corr = [t.user_attrs.get("best_trial_corr") for t in complete_trials if t.user_attrs.get("best_trial_corr") is not None]
val_R2 = [t.user_attrs.get("best_R2_score") for t in complete_trials if t.user_attrs.get("best_R2_score") is not None]
test_corr = [t.user_attrs.get("best_test_corr") for t in complete_trials if t.user_attrs.get("best_test_corr") is not None]
test_R2 = [t.user_attrs.get("best_test_R2") for t in complete_trials if t.user_attrs.get("best_test_R2") is not None]

mean_val_corr, mean_test_corr = np.mean(val_corr), np.mean(test_corr)
mean_val_R2, mean_test_R2 = np.mean(val_R2), np.mean(test_R2)

try:
    # --- Histogramme Corrélation ---
    fig_hist_corr = go.Figure()
    fig_hist_corr.add_trace(go.Histogram(x=val_corr, nbinsx=20, name="Dist. Validation", marker_color='royalblue', opacity=0.75))
    fig_hist_corr.add_vline(x=mean_val_corr, line_dash="dash", line_color="blue", annotation_text=f"Moy. Val: {mean_val_corr:.3f}")
    fig_hist_corr.add_vline(x=mean_test_corr, line_dash="solid", line_color="red", annotation_text=f"Moy. Test: {mean_test_corr:.3f}")
    
    fig_hist_corr.update_layout(title="Distribution de la Corrélation (r) sur les Folds LOOCV", xaxis_title="Corrélation de Pearson (r)", yaxis_title="Nombre de Folds", template="plotly_white")
    fig_hist_corr.write_image(os.path.join(output_dir, "loocv_histogram_corr.png"), width=1000, height=600)

    # --- Histogramme R² ---
    fig_hist_R2 = go.Figure()
    fig_hist_R2.add_trace(go.Histogram(x=val_R2, nbinsx=20, name="Dist. Validation", marker_color='seagreen', opacity=0.75))
    fig_hist_R2.add_vline(x=mean_val_R2, line_dash="dash", line_color="green", annotation_text=f"Moy. Val: {mean_val_R2:.3f}")
    fig_hist_R2.add_vline(x=mean_test_R2, line_dash="solid", line_color="red", annotation_text=f"Moy. Test: {mean_test_R2:.3f}")
    
    fig_hist_R2.update_layout(title="Distribution du R² (Skill Score) sur les Folds LOOCV", xaxis_title="Score R²", yaxis_title="Nombre de Folds", template="plotly_white")
    fig_hist_R2.write_image(os.path.join(output_dir, "loocv_histogram_R2.png"), width=1000, height=600)
    
    print("✅ Fichiers .png générés avec succès pour les histogrammes.")
except Exception as e:
    print(f"⚠️ Impossible de générer les PNG. Assure-toi d'avoir installé 'kaleido' (pip install -U kaleido). Erreur : {e}")

print(f"Terminé ! Retrouvez vos fichiers dans : {output_dir}")