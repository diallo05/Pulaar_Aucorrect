"""
Script d'entraînement du modèle de correction orthographique Pulaar.
Ce script entraîne le modèle sur le corpus et le sauvegarde en format pkl.
"""

import os
from spell_corrector import build_probabilities_from_corpus, save_model

# Chemins des fichiers
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus_pulaar.txt")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pulaar_spell_model.pkl")

def main():
    print("Début de l'entraînement du modèle...")
    print(f"Chargement du corpus depuis: {CORPUS_PATH}")
    
    # Construction du modèle de probabilités
    probabilities = build_probabilities_from_corpus(CORPUS_PATH)
    
    print(f"Modèle entraîné avec {len(probabilities)} mots uniques.")
    
    # Sauvegarde du modèle
    save_model(probabilities, MODEL_PATH)
    
    print("Entraînement terminé avec succès!")

if __name__ == "__main__":
    main()
