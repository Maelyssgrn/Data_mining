"""Modèles, seuil de décision et métriques métier."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
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


def _gain_au_seuil(y, p):
    """Gain net (UM) si l'on contacte les clients de probabilité p > seuil de rentabilité."""
    contacter = np.asarray(p) > SEUIL_RENTABILITE
    return gain_net(int(contacter.sum()), int(np.asarray(y)[contacter].sum()))


# Critère de réglage : le gain net au seuil (scorer sklearn, sur les probabilités)
SCORER_GAIN = make_scorer(_gain_au_seuil, response_method="predict_proba")

# Grilles de réglage (préfixe "classification__" : étape du pipeline).
# n_estimators n'est pas réglé (fixé à 500) : plus d'arbres ne fait pas surapprendre,
# cela stabilise seulement les probabilités.
GRILLES = {
    "Régression logistique": {"classification__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "Random Forest": {"classification__max_depth": [3, 5, 8, None],
                      "classification__min_samples_leaf": [1, 5, 10, 20],
                      "classification__max_features": ["sqrt", 0.5]},
}


def _plis():
    """Les mêmes 5 plis stratifiés partout (réglage, comparaison, baseline)."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)


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
            # Pas de class_weight="balanced" : il déformerait les probabilités,
            # et la règle p > 3/11 ne serait plus justifiée.
            ("classification", RandomForestClassifier(n_estimators=500, oob_score=True,
                                                      random_state=config.SEED, n_jobs=-1))]),
    }


def regler(modeles, train):
    """GridSearchCV sur le train : meilleur gain net moyen (refit="gain"), AP à côté.
    Chaque recherche est ensuite réentraînée sur tout le train avec son meilleur réglage."""
    recherches = {}
    for nom, modele in modeles.items():
        recherche = GridSearchCV(modele, GRILLES[nom], cv=_plis(), n_jobs=-1, refit="gain",
                                 scoring={"gain": SCORER_GAIN, "ap": "average_precision"})
        recherches[nom] = recherche.fit(variables(train), train["Response"])
    return recherches


def choisir(cv, simple="Régression logistique"):
    """Règle de choix : le meilleur gain net moyen en validation croisée, sauf si son
    écart avec le modèle simple est inférieur à un écart-type : on garde alors le simple.
    cv : les lignes des modèles dans le tableau de comparer_cv."""
    meilleur = cv["gain_moyen"].idxmax()
    if meilleur == simple:
        return simple, "meilleur gain net moyen en validation croisée"
    ecart = cv.loc[meilleur, "gain_moyen"] - cv.loc[simple, "gain_moyen"]
    ecart_type = cv.loc[meilleur, "gain_ecart_type"]
    if ecart < ecart_type:
        return simple, (f"{meilleur} gagne {ecart:.1f} UM de plus, moins qu'un écart-type "
                        f"({ecart_type:.1f}) : on garde le modèle le plus simple")
    return meilleur, (f"{meilleur} gagne {ecart:.1f} UM de plus, plus qu'un écart-type "
                      f"({ecart_type:.1f})")


def contacter_si_rentable(modele, df):
    """Décision : contacter si la probabilité de réponse dépasse le seuil de rentabilité."""
    p = modele.predict_proba(variables(df))[:, 1]
    return pd.Series(p > SEUIL_RENTABILITE, index=df.index)


def comparer_cv(modeles, train):
    """Validation croisée stratifiée (5 plis) sur le train seulement.
    Pour chaque pli : Average Precision des modèles, et gain net au seuil de
    rentabilité des modèles et de la baseline (même pli de validation)."""
    lignes = []
    for i_app, i_val in _plis().split(train, train["Response"]):
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


def _ap_et_gain(y, p):
    """Average Precision et gain pour 100 clients (comparable entre jeux de tailles différentes)."""
    return {"ap": average_precision_score(y, p),
            "gain_pour_100": 100 * _gain_au_seuil(y, p) / len(y), "n_clients": len(y)}


def oob_contre_test(foret, train, test):
    """Score OOB (chaque client du train jugé par les arbres qui ne l'ont pas vu) contre test.
    oob_score_ de sklearn est une accuracy, inutile ici : on repart des probabilités OOB."""
    p_oob = foret[-1].oob_decision_function_[:, 1]
    p_test = foret.predict_proba(variables(test))[:, 1]
    return pd.DataFrame.from_dict({"OOB (train)": _ap_et_gain(train["Response"], p_oob),
                                   "Test": _ap_et_gain(test["Response"], p_test)},
                                  orient="index").rename_axis("jeu")


def importance_permutation(modele, test, n=10):
    """Baisse de l'Average Precision sur le test quand on mélange une variable d'origine.
    Une importance mesure ce que le modèle utilise, pas une cause."""
    r = permutation_importance(modele, variables(test), test["Response"],
                               scoring="average_precision", n_repeats=30,
                               random_state=config.SEED, n_jobs=-1)
    return (pd.DataFrame({"variable": variables(test).columns,
                          "baisse_ap": r.importances_mean, "ecart_type": r.importances_std})
            .sort_values("baisse_ap", ascending=False).head(n))


# Tranches d'âge de la vérification d'équité (55-64 et 65+ regroupés : 16 clients
# de 65 ans et plus dans le test, trop peu pour un taux)
TRANCHES_AGE = [0, 35, 45, 55, 200]
NOMS_TRANCHES = ["< 35 ans", "35-44 ans", "45-54 ans", "55 ans et +"]


def equite(modele, test):
    """Par situation familiale et tranche d'âge (sur le test) : taux de réponse réel du
    groupe, taux de contact par le modèle, et taux de réponse parmi les contactés.
    Un groupe plus contacté doit aussi répondre plus, sinon c'est un signal d'alerte."""
    contacter = contacter_si_rentable(modele, test)
    groupes = {"Situation familiale": test["Marital_Status"],
               "Tranche d'âge": pd.cut(test["Age"], TRANCHES_AGE, right=False,
                                       labels=NOMS_TRANCHES)}
    lignes = []
    for variable, groupe in groupes.items():
        for nom, index in groupe.groupby(groupe, observed=True).groups.items():
            bilan = bilan_campagne(contacter[index], test.loc[index, "Response"])
            lignes.append({"variable": variable, "groupe": nom, "n_clients": len(index),
                           "taux_reponse_groupe_pct": round(100 * test.loc[index, "Response"].mean(), 1),
                           "taux_contact_pct": round(100 * bilan["n_contactes"] / len(index), 1),
                           "n_contactes": bilan["n_contactes"],
                           "taux_reponse_contactes_pct": bilan["taux_pct"]})
    return pd.DataFrame(lignes)


if __name__ == "__main__":
    from backend import data
    print(f"Seuil de rentabilité : {SEUIL_RENTABILITE:.1%}")
    data.afficher("Besoin client (table nettoyée)", besoin_client(data.lire_analyse()).reset_index())
