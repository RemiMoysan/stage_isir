import numpy as np
import os

# On run directement en local sur le mac. 

# Nom de ton fichier (à modifier si besoin)
filename_1 = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/composites_4_regimes/master_ref_generator_89members_normalizeFalse_duree_lissage10_embedding_method_pca_latent_dim_128/master_reference_global.npz"
filename_2 = "/Users/remimoysan/Desktop/Jean_Zay/work_jz/stage_isir_jz/data_analysis/composites_pc1_quantiles/pc1_8_quantiles_master_ref_generator_89members_normalizeFalse_duree_lissage10/master_reference_quantiles_PCA.npz"

for filename in [filename_2]:
    print(f"Recherche du fichier : {filename}")

    if not os.path.exists(filename):
        print(f"❌ Erreur : Le fichier '{filename}' est introuvable")
        print("📁 Fichier trouvé")
    else:
        # Chargement du fichier (allow_pickle=True par sécurité si le dico contient des objets Python)
        data = np.load(filename, allow_pickle=True)
        
        # La méthode .files liste les clés d'un objet NpzFile
        keys = data.files
        
        print("\n✅ Fichier chargé avec succès !\n")
        print("="*40)
        print("🔑 CLÉS CONTENUES DANS LE DICTIONNAIRE")
        print("="*40)
        
        for key in keys:
            print(f" -> {key}")
            
        print("\n" + "="*40)
        print("📊 DÉTAILS DES DONNÉES (SHAPES)")
        print("="*40)
        
        for key in keys:
            array_data = data[key]
            # On essaie d'afficher la shape (marche pour les arrays Numpy)
            if hasattr(array_data, 'shape'):
                print(f"{key:<20} : shape = {array_data.shape}")
            else:
                print(f"{key:<20} : type = {type(array_data)}")