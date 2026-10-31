"""Modèles, seuil de décision et métriques métier."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend import config

# Variables explicatives (ni Response, ni ID, ni Z_, ni segment du k-means)
VARIABLES_NUMERIQUES = ["Age", "Income", "Enfants", "MntWines", "Recency",
                        "Nb_achats", "Anciennete_jours"]
VARIABLES_CATEGORIELLES = ["Education"]

# Contacter un client de probabilité de réponse p rapporte en moyenne
# p * REVENU_REPONSE - COUT_CONTACT : c'est positif si p > 3 / 11 = 27,3 %.
# (Coûts supposés : voir l'hypothèse sur les colonnes Z_ dans config.py.)
SEUIL_RENTABILITE = config.COUT_CONTACT / config.REVENU_REPONSE


def gain_net(n_contactes, n_reponses):
    """Gain net d'une campagne en UM : revenus des réponses moins coût des contacts."""
    return n_reponses * config.REVENU_REPONSE - n_contactes * config.COUT_CONTACT


def regle_deja_accepte(df):
    """Baseline : contacter les clients ayant déjà accepté au moins une campagne."""
    return df["Nb_campagnes_acceptees"] > 0


def bilan_campagne(contacter, response):
    """Bilan d'un ciblage. contacter : booléens (qui on contacte) ; response : la cible."""
    n_contactes = int(contacter.sum())
    n_reponses = int(response[contacter].sum())
    taux = 100 * n_reponses / n_contactes if n_contactes else 0.0
    return {"n_contactes": n_contactes, "n_reponses": n_reponses,
            "taux_pct": round(taux, 1), "gain_net": gain_net(n_contactes, n_reponses)}


def besoin_client(df):
    """Contacter tout le monde est-il rentable ? Comparaison avec la règle simple."""
    tous = pd.Series(True, index=df.index)
    # from_dict(orient="index") : une ligne par stratégie, chaque colonne garde son type
    return pd.DataFrame.from_dict({
        "Tout le monde": bilan_campagne(tous, df["Response"]),
        "Déjà accepté une campagne": bilan_campagne(regle_deja_accepte(df), df["Response"]),
    }, orient="index").rename_axis("strategie")


def separer(df):
    """Split 80/20 stratifié : le test est réservé à l'évaluation finale."""
    X = df[VARIABLES_NUMERIQUES + VARIABLES_CATEGORIELLES]
    return train_test_split(X, df["Response"], test_size=0.2,
                            random_state=config.SEED, stratify=df["Response"])


def _preprocesseur():
    """Imputation et encodage, appris sur le train seulement (dans le pipeline)."""
    numerique = Pipeline([("imputation", SimpleImputer(strategy="median")),
                          ("standardisation", StandardScaler())])
    categoriel = Pipeline([("imputation", SimpleImputer(strategy="most_frequent")),
                           ("encodage", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("numerique", numerique, VARIABLES_NUMERIQUES),
                              ("categoriel", categoriel, VARIABLES_CATEGORIELLES)])


def construire_modeles():
    """Les modèles comparés, chacun avec son propre préprocesseur."""
    return {
        "Régression logistique": Pipeline([
            ("preparation", _preprocesseur()),
            ("classification", LogisticRegression(max_iter=1000, random_state=config.SEED))]),
        "Random Forest": Pipeline([
            ("preparation", _preprocesseur()),
            ("classification", RandomForestClassifier(n_estimators=100, random_state=config.SEED,
                                                      n_jobs=-1))]),
    }


def comparer_cv(modeles, X_train, y_train):
    """Validation croisée stratifiée (5 plis) sur le train, score : Average Precision."""
    plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    resultats = {}
    for nom, modele in modeles.items():
        scores = cross_val_score(modele, X_train, y_train, cv=plis, scoring="average_precision")
        resultats[nom] = {"ap_moyenne": scores.mean(), "ap_ecart_type": scores.std()}
    return pd.DataFrame.from_dict(resultats, orient="index").rename_axis("modele")


def evaluer(modele, X_test, y_test):
    """Évaluation finale sur le test, faite une seule fois (seuil 0,5 de predict pour l'instant)."""
    y_pred = modele.predict(X_test)
    return {"matrice_confusion": confusion_matrix(y_test, y_pred),
            "rapport": classification_report(y_test, y_pred, zero_division=0)}


if __name__ == "__main__":
    from backend import data
    print(f"Seuil de rentabilité : {SEUIL_RENTABILITE:.1%}")
    data.afficher("Besoin client (table nettoyée)", besoin_client(data.lire_analyse()).reset_index())
