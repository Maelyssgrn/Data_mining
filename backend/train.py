"""Règle les modèles, choisit le meilleur par validation croisée et le sauvegarde,
avec les tableaux de résultats (relus par service.py, sans réentraîner).

Lancement : python -m backend.train
"""

import joblib

from backend import config, data, model


def main():
    df = data.lire_analyse()
    train, test = model.separer(df)

    # Réglage par GridSearchCV sur le train (critère : gain net moyen au seuil)
    recherches = model.regler(model.construire_modeles(), train)
    reglages = {nom: {k.removeprefix("classification__"): v for k, v in r.best_params_.items()}
                for nom, r in recherches.items()}
    print("\n=== Meilleurs réglages (validation croisée, train) ===")
    for nom, r in reglages.items():
        print(f"{nom} : {r}")
    modeles = {nom: recherche.best_estimator_ for nom, recherche in recherches.items()}

    # Comparaison sur les mêmes plis, avec la baseline, puis choix du modèle au meilleur
    # gain net moyen (la baseline n'entre pas dans le choix)
    cv = model.comparer_cv(modeles, train)
    data.afficher("Validation croisée (train, 5 plis)", cv.round(4).reset_index())
    retenu, raison = model.choisir(cv.loc[list(modeles)])
    print(f"\nModèle retenu : {retenu} ({raison})")

    # Évaluation finale, une seule fois, sur le test. Le modèle retenu ne change plus,
    # même si l'autre fait mieux ici : ce serait choisir le modèle sur le test.
    resultats = {
        "reglages": reglages, "cv": cv, "retenu": retenu, "raison": raison,
        "n_test": len(test), "n_repondants_test": int(test["Response"].sum()),
        "strategies": model.comparer_strategies(modeles, test),
        "oob": model.oob_contre_test(modeles["Random Forest"], train, test),
        # Importance = ce que le modèle utilise pour classer, pas une cause de la réponse
        "importance": model.importance_permutation(modeles[retenu], test),
        # Équité (variables sensibles) : comparer le taux de contact au taux de réponse
        # réel du groupe. Groupes petits : lecture descriptive, sans test statistique.
        "equite": model.equite(modeles[retenu], test),
    }
    print(f"\nTest : {resultats['n_test']} clients, dont {resultats['n_repondants_test']} répondants")
    data.afficher(f"Stratégies sur le test (contacter si p > {model.SEUIL_RENTABILITE:.1%})",
                  resultats["strategies"].reset_index())
    data.afficher("Random Forest : OOB contre test", resultats["oob"].round(3).reset_index())
    data.afficher(f"Importance par permutation sur le test ({retenu}, baisse d'AP)",
                  resultats["importance"].round(4))
    data.afficher(f"Équité sur le test ({retenu})", resultats["equite"])

    config.DOSSIER_MODELES.mkdir(exist_ok=True)
    joblib.dump(modeles[retenu], config.CHEMIN_MODELE)
    joblib.dump(resultats, config.CHEMIN_RESULTATS)
    print(f"\nModèle et résultats sauvegardés dans {config.DOSSIER_MODELES}")


if __name__ == "__main__":
    main()
