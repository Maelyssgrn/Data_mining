"""Entraîne les modèles, choisit le meilleur par validation croisée et le sauvegarde.

Lancement : python -m backend.train
"""

import joblib

from backend import config, data, model


def main():
    df = data.lire_analyse()
    train, test = model.separer(df)

    # Choix du modèle par validation croisée sur le train (le test n'intervient pas)
    modeles = model.construire_modeles()
    cv = model.comparer_cv(modeles, train)
    data.afficher("Validation croisée (train, 5 plis)", cv.round(4).reset_index())

    meilleur_nom = cv.loc[list(modeles), "ap_moyenne"].idxmax()
    print(f"\nModèle retenu (Average Precision en validation croisée) : {meilleur_nom}")

    # Les deux modèles sont entraînés pour le tableau du test. Le modèle retenu
    # reste celui de la validation croisée, même si l'autre fait mieux sur le test :
    # changer d'avis ici reviendrait à choisir le modèle sur le test.
    for modele in modeles.values():
        modele.fit(model.variables(train), train["Response"])

    # Évaluation finale, une seule fois, sur le test
    print(f"\nTest : {len(test)} clients, dont {int(test['Response'].sum())} répondants")
    data.afficher(f"Stratégies sur le test (contacter si p > {model.SEUIL_RENTABILITE:.1%})",
                  model.comparer_strategies(modeles, test).reset_index())

    config.DOSSIER_MODELES.mkdir(exist_ok=True)
    joblib.dump(modeles[meilleur_nom], config.CHEMIN_MODELE)
    print(f"\nModèle sauvegardé : {config.CHEMIN_MODELE}")


if __name__ == "__main__":
    main()
