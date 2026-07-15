import optuna
import os
import argparse
import plotly.graph_objects as go


parser = argparse.ArgumentParser()
parser.add_argument("--study-name", help="Nom de l'étude Optuna: en fait vu qu'on ne fait que 1 par storage on peut mettre n'importe quoi", default = None)
parser.add_argument("--storage-name", help="Nom du stockage Optuna (chemin vers le fichier .db", default = "stage_isir_jz/cnn/cnn_with_slp_embedding/optuna/cnn_monthly_pc1_correlation_max_n_trials500_months2_nb_epochs20_intra_evals15_patience5_nb_members_val5_sampler_pruner35_15_2_3.db")

args = parser.parse_args()

# 1. Construire les chemins
db_absolute_path = os.path.join("/lustre/fswork/projects/rech/uxg/uca57ub/", args.storage_name)
storage_name = f"sqlite:///{db_absolute_path}"

# --- LA CORRECTION EST ICI ---
# On extrait le dossier qui contient le fichier .db
output_dir = os.path.dirname(db_absolute_path)
print(f"Dossier cible pour les graphiques HTML : {output_dir}")

# 2. Récupérer toutes les études disponibles dans le fichier .db
summaries = optuna.get_all_study_summaries(storage=storage_name)
available_studies = [s.study_name for s in summaries]

print("Études disponibles dans la DB :", available_studies)

study_name = args.study_name

# --- LOGIQUE DE FALLBACK ---
if study_name not in available_studies:
    if len(available_studies) == 0:
        # Cas extrême : le fichier .db existe mais il est totalement vide
        raise ValueError(f"Le fichier .db est vide. Aucune étude trouvée dans : {storage_name}")
    else:
        # On prend la première étude de la liste
        fallback_name = available_studies[0]
        print(f"\n⚠️ ATTENTION : L'étude demandée '{study_name}' est introuvable !")
        print(f"🔄 FALLBACK ACTIVÉ : Chargement automatique de la première étude : '{fallback_name}'\n")
        study_name = fallback_name
# ---------------------------

print(optuna.get_all_study_names(storage_name))
study = optuna.load_study(study_name=study_name, storage=storage_name)

# --- VISUALISATIONS ---
print("Génération des graphiques en cours...")

# On ajoute os.path.join(output_dir, ...) à chaque sauvegarde
fig_history = optuna.visualization.plot_optimization_history(study)
fig_history.write_html(os.path.join(output_dir, "opt_history.html"))

fig_importances = optuna.visualization.plot_param_importances(study)
fig_importances.write_html(os.path.join(output_dir, "param_importances.html"))

try:
    fig_contour = optuna.visualization.plot_contour(study, params=["lr", "dr"])
    fig_contour.write_html(os.path.join(output_dir, "contour_lr_dr.html"))
except Exception as e:
    print(f"⚠️ Erreur lors de la génération du contour plot pour 'lr' et 'dr' : {e}")
    print("🔹 Vérifiez que ces paramètres existent dans l'étude. Si ce n'est pas le cas, vous pouvez modifier la liste des paramètres dans le code.")

fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
fig_parallel.write_html(os.path.join(output_dir, "parallel_coordinates.html"))

fig_slice = optuna.visualization.plot_slice(study)
fig_slice.write_html(os.path.join(output_dir, "slice_plot.html"))

# 3. Récupérer les essais terminés
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if not complete_trials:
    print("Aucun essai terminé trouvé.")
    exit()

# Créer la figure Plotly
fig = go.Figure()

# 4. Boucler sur les essais (on trace les 10 meilleurs par exemple pour ne pas surcharger)
# Trions les essais par leur valeur finale (le meilleur en premier)
complete_trials.sort(key=lambda t: t.value, reverse=True)
trials_to_plot = complete_trials[:10] 

for trial in trials_to_plot:
    history = trial.user_attrs.get("r2_corr_history")
    if history:
        # history est une liste de tuples : (epoch, r2, corr)
        epochs = [h[0] for h in history]
        r2_scores = [h[1] for h in history]
        correlations2 = [h[2]**2 for h in history]
        
        # Ajouter la courbe de Corrélation (ligne pleine)
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=correlations2,
            mode='lines',
            name=f'Trial {trial.number} - Corr^2', # <-- Ajout du numéro d'essai
            line=dict(width=2)
        ))
        
        # Ajouter la courbe de R2 (ligne pointillée) si tu veux la voir aussi
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=r2_scores,
            mode='lines',
            name=f'Trial {trial.number} - R2', # <-- Ajout du numéro d'essai
            line=dict(width=2, dash='dash')
        ))

# 5. Mise en page et sauvegarde
fig.update_layout(
    title=f"Évolution des Métriques (Intra-époque) - {study_name}",
    xaxis_title="Époques (incluant les steps intra-époques)",
    yaxis_title="Score (Corr^2 et R2)", # ✅ CORRECTION 2 : Titre de l'axe Y plus précis
    legend_title="Scores",
    template="plotly_white"
)

# Sauvegarde interactive (Idéal pour navigateur)
html_path = os.path.join(output_dir, "training_history_curves.html")
fig.write_html(html_path)

# Sauvegarde statique (Idéal pour VS Code)
png_path = os.path.join(output_dir, "training_history_curves.png")
fig.write_image(png_path, width=1200, height=800)

print(f"Toutes les visualisations interactives (.html) ont été générées avec succès dans le dossier de l'étude !")