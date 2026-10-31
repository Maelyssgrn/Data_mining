"""Modèles, seuil de décision et métriques métier."""

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend import config

# Variables explicatives autorisées par la Data Quality Card
# (ni Response, ni ID, ni Z_, ni segment du k-means).
# Les totaux exacts (Enfants, Depense_totale, Nb_achats, Nb_campagnes_acceptees)
# et Year_Birth (= 2014 - Age) sont retirés : on garde le détail, sinon une même
# information est comptée deux fois (coefficients et importances faussés).
VARIABLES_NUMERIQUES = [
    "Age", "Income", "Kidhome", "Teenhome", "Recency", "Anciennete_jours", "Complain",
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth",
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
]
VARIABLES_CATEGORIELLES = ["Education", "Marital_Status"]

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
    """Split 80/20 stratifié : le test est réservé à l'évaluation finale.
    Renvoie les lignes complètes (train, test) : la baseline a besoin de colonnes
    qui ne sont pas des variables du modèle."""
    return train_test_split(df, test_size=0.2, random_state=config.SEED,
                            stratify=df["Response"])


def variables(df):
    """Les colonnes données au modèle."""
    return df[VARIABLES_NUMERIQUES + VARIABLES_CATEGORIELLES]


def _preprocesseur():
    """Imputation et encodage, appris sur le train seulement (dans le pipeline)."""
    numerique = Pipeline([("imputation", SimpleImputer(strategy="median")),
                          ("standardisation", StandardScaler())])
    # Sans drop="first" : la référence serait "Autre" (2 clients), et une modalité
    # inconnue serait confondue avec elle. La pénalité L2 gère la colinéarité.
    categoriel = Pipeline([("imputation", SimpleImputer(strategy="most_frequent")),
                           ("encodage", OneHotEncoder(handle_unknown="ignore"))])
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


def contacter_si_rentable(modele, df):
    """Décision : contacter si la probabilité de réponse dépasse le seuil de rentabilité."""
    p = modele.predict_proba(variables(df))[:, 1]
    return pd.Series(p > SEUIL_RENTABILITE, index=df.index)


def comparer_cv(modeles, train):
    """Validation croisée stratifiée (5 plis) sur le train seulement.
    Pour chaque pli : Average Precision des modèles, et gain net au seuil de
    rentabilité des modèles et de la baseline (même pli de validation)."""
    plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    lignes = []
    for i_app, i_val in plis.split(train, train["Response"]):
        app, val = train.iloc[i_app], train.iloc[i_val]
        for nom, modele in modeles.items():
            # clone : un modèle neuf par pli, prétraitements appris sur ce pli seulement
            m = clone(modele).fit(variables(app), app["Response"])
            p = m.predict_proba(variables(val))[:, 1]
            lignes.append({"strategie": nom,
                           "ap": average_precision_score(val["Response"], p),
                           "gain": bilan_campagne(contacter_si_rentable(m, val),
                                                  val["Response"])["gain_net"]})
        lignes.append({"strategie": "Déjà accepté une campagne", "ap": float("nan"),
                       "gain": bilan_campagne(regle_deja_accepte(val), val["Response"])["gain_net"]})
    # ddof=0 : même écart-type que cross_val_score à l'étape 1
    return (pd.DataFrame(lignes).groupby("strategie", sort=False)
            .agg(ap_moyenne=("ap", "mean"), ap_ecart_type=("ap", lambda s: s.std(ddof=0)),
                 gain_moyen=("gain", "mean"), gain_ecart_type=("gain", lambda s: s.std(ddof=0))))


def comparer_strategies(modeles, test):
    """Bilan sur le test (une seule fois) : modèles au seuil, baseline, tout le monde."""
    strategies = {nom: contacter_si_rentable(m, test) for nom, m in modeles.items()}
    strategies["Déjà accepté une campagne"] = regle_deja_accepte(test)
    strategies["Tout le monde"] = pd.Series(True, index=test.index)
    return pd.DataFrame.from_dict(
        {nom: bilan_campagne(c, test["Response"]) for nom, c in strategies.items()},
        orient="index").rename_axis("strategie")


if __name__ == "__main__":
    from backend import data
    print(f"Seuil de rentabilité : {SEUIL_RENTABILITE:.1%}")
    data.afficher("Besoin client (table nettoyée)", besoin_client(data.lire_analyse()).reset_index())
