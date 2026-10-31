"""Entraîne les modèles, choisit le meilleur par validation croisée et le sauvegarde.

Lancement : python -m backend.train
"""

import joblib

from backend import config, data, model


def main():
    df = data.lire_analyse()
    X_train, X_test, y_train, y_test = model.separer(df)

    # Choix du modèle par validation croisée sur le train (le test n'intervient pas)
    modeles = model.construire_modeles()
    cv = model.comparer_cv(modeles, X_train, y_train)
    data.afficher("Validation croisée (train)", cv.round(4).reset_index())

    meilleur_nom = cv["ap_moyenne"].idxmax()
    meilleur_modele = modeles[meilleur_nom].fit(X_train, y_train)
    print(f"\nModèle retenu : {meilleur_nom}")

    # Évaluation finale, une seule fois, sur le test
    resultat = model.evaluer(meilleur_modele, X_test, y_test)
    print("\nMatrice de confusion :")
    print(resultat["matrice_confusion"])
    print("\nRapport de classification :")
    print(resultat["rapport"])

    config.DOSSIER_MODELES.mkdir(exist_ok=True)
    joblib.dump(meilleur_modele, config.CHEMIN_MODELE)
    print(f"Modèle sauvegardé : {config.CHEMIN_MODELE}")


if __name__ == "__main__":
    main()
