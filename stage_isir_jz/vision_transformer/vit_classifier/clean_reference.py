import numpy as np

# 1. Charger l'ancien gros fichier
fichier_source = "four_clusters_reference.npz"
data = np.load(fichier_source)

# 2. Filtrer pour ne garder QUE les 4 cartes SLP de base
dico_propre = {}
for key in data.files:
    if key.startswith("regime_") and key.endswith("_slp_0"):
        dico_propre[key] = data[key]

# Vérification
print(f"Clés conservées ({len(dico_propre)}) :")
for k in dico_propre.keys():
    print(f" - {k}")

# 3. Sauvegarder le nouveau fichier allégé
fichier_dest = "master_reference_light.npz"
np.savez(fichier_dest, **dico_propre)
print(f"\nNouveau fichier sauvegardé : {fichier_dest}")