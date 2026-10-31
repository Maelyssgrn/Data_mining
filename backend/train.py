"""Règle les modèles, choisit le meilleur par validation croisée et le sauvegarde.

Lancement : python -m backend.train
"""

import joblib

from backend import config, data, model


def main():
    df = data.lire_analyse()
    train, test = model.separer(df)

    # Réglage par GridSearchCV sur le train (critère : gain net moyen au seuil)
    recherches = model.regler(model.construire_modeles(), train)
    print("\n=== Meilleurs réglages (validation croisée, train) ===")
    for nom, recherche in recherches.items():
        reglages = {k.removeprefix("classification__"): v for k, v in recherche.best_params_.items()}
        print(f"{nom} : {reglages}")
    modeles = {nom: recherche.best_estimator_ for nom, recherche in recherches.items()}

    # Comparaison sur les mêmes plis, avec la baseline, puis choix du modèle
    cv = model.comparer_cv(modeles, train)
    data.afficher("Validation croisée (train, 5 plis)", cv.round(4).reset_index())
    retenu, raison = model.choisir(cv.loc[list(modeles)])
    print(f"\nModèle retenu : {retenu} ({raison})")

    # Évaluation finale, une seule fois, sur le test. Le modèle retenu ne change plus,
    # même si l'autre fait mieux ici : ce serait choisir le modèle sur le test.
    print(f"\nTest : {len(test)} clients, dont {int(test['Response'].sum())} répondants")
    data.afficher(f"Stratégies sur le test (contacter si p > {model.SEUIL_RENTABILITE:.1%})",
                  model.comparer_strategies(modeles, test).reset_index())
    data.afficher("Random Forest : OOB contre test",
                  model.oob_contre_test(modeles["Random Forest"], train, test).round(3).reset_index())
    # Importance = ce que le modèle utilise pour classer, pas une cause de la réponse
    data.afficher(f"Importance par permutation sur le test ({retenu}, baisse d'AP)",
                  model.importance_permutation(modeles[retenu], test).round(4))
    # Équité (variables sensibles) : comparer le taux de contact au taux de réponse
    # réel du groupe. Groupes petits : lecture descriptive, sans test statistique.
    data.afficher(f"Équité sur le test ({retenu})", model.equite(modeles[retenu], test))

    config.DOSSIER_MODELES.mkdir(exist_ok=True)
    joblib.dump(modeles[retenu], config.CHEMIN_MODELE)
    print(f"\nModèle sauvegardé : {config.CHEMIN_MODELE}")


if __name__ == "__main__":
    main()
