import os
import argparse
import optuna
import pandas as pd
import plotly.graph_objects as go

parser = argparse.ArgumentParser()
parser.add_argument("--study-name", help="Nom de l'étude Optuna", default="default_study")
parser.add_argument("--storage-name", help="Chemin relatif vers le fichier .db", required=True)
parser.add_argument("--n-best", type=int, default=5, help="Nombre de meilleurs modèles dont on veut tracer l'historique")

args = parser.parse_args()

# ============================================================
# 1. GESTION DES CHEMINS ET DU FALLBACK
# ============================================================
base_path = "/lustre/fswork/projects/rech/uxg/uca57ub/"
db_absolute_path = os.path.join(base_path, args.storage_name)
storage_name = f"sqlite:///{db_absolute_path}"

# On extrait le dossier qui contient le fichier .db pour y ranger les plots
output_dir = os.path.dirname(db_absolute_path)
print(f"📁 Dossier cible pour les graphiques HTML : {output_dir}")

# Récupérer toutes les études disponibles dans le fichier .db
try:
    summaries = optuna.get_all_study_summaries(storage=storage_name)
except Exception as e:
    raise RuntimeError(f"Impossible de lire le fichier SQLite {db_absolute_path}. Le chemin est-il correct ?") from e

available_studies = [s.study_name for s in summaries]
print(f"🔍 Études disponibles dans la DB : {available_studies}")

study_name = args.study_name

# --- LOGIQUE DE FALLBACK ---
if study_name not in available_studies:
    if len(available_studies) == 0:
        raise ValueError(f"Le fichier .db est vide. Aucune étude trouvée dans : {storage_name}")
    else:
        fallback_name = available_studies[0]
        print(f"\n⚠️ ATTENTION : L'étude demandée '{study_name}' est introuvable !")
        print(f"🔄 FALLBACK ACTIVÉ : Chargement automatique de la première étude : '{fallback_name}'\n")
        study_name = fallback_name

# ============================================================
# 2. CHARGEMENT DE L'ÉTUDE
# ============================================================
study = optuna.load_study(study_name=study_name, storage=storage_name)
print(f"✅ Étude '{study_name}' chargée avec succès ({len(study.trials)} trials trouvés).")

# ============================================================
# 3. GÉNÉRATION DES PLOTS STANDARDS OPTUNA
# ============================================================
print("\n📊 Génération des graphiques Optuna standards...")
figs = {
    "optimization_history": optuna.visualization.plot_optimization_history(study),
    "param_importances": optuna.visualization.plot_param_importances(study),
    "slice_plot": optuna.visualization.plot_slice(study),
    "parallel_coordinate": optuna.visualization.plot_parallel_coordinate(study)
}

for name, fig in figs.items():
    out_path = os.path.join(output_dir, f"{name}.html")
    fig.write_html(out_path)
    print(f"  -> Sauvegardé : {out_path}")

# ============================================================
# 4. GÉNÉRATION DES COURBES D'APPRENTISSAGE (N MEILLEURS)
# ============================================================
print(f"\n📈 Génération des courbes d'apprentissage pour les {args.n_best} meilleurs trials...")

# Filtrer uniquement les trials terminés avec succès
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if not completed_trials:
    print("⚠️ Aucun trial complet trouvé dans l'étude.")
else:
    # Trier les trials selon leur performance (maximisation)
    is_maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=is_maximize)
    
    # Sélectionner le top N
    top_trials = sorted_trials[:args.n_best]

    for rank, trial in enumerate(top_trials, start=1):
        history = trial.user_attrs.get("R2_L1_corr_history", None)

        if history:
            df = pd.DataFrame(history, columns=["Step", "R2", "L1_Skill_Score", "Correlation"])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Step"], y=df["R2"], mode='lines+markers', name='R2 Global', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df["Step"], y=df["L1_Skill_Score"], mode='lines+markers', name='L1 Skill Score', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=df["Step"], y=df["Correlation"], mode='lines+markers', name='Corrélation', line=dict(color='red')))
            
            fig.update_layout(
                title=f"Learning Curves - TOP {rank} (Trial #{trial.number}) - Score final : {trial.value:.4f}",
                xaxis_title="Epoch (incluant évaluations intra-epoch)",
                yaxis_title="Valeur de la Métrique",
                hovermode="x unified",
                template="plotly_white"
            )
            
            # Nom de fichier dynamique selon le rang et le numéro de trial
            out_path = os.path.join(output_dir, f"top_{rank}_trial_{trial.number}_learning_curves.html")
            fig.write_html(out_path)
            print(f"  -> Sauvegardé : {out_path}")
        else:
            print(f"⚠️ Aucun historique trouvé pour le Top {rank} (Trial #{trial.number}).")

print("\n🎉 Tous les graphiques HTML interactifs ont été générés avec succès !")