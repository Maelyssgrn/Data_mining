"""Données brutes -> SQLite -> table d'analyse."""

import sqlite3

import numpy as np
import pandas as pd

from backend import config

# Libellés français des familles de produits et des canaux
PRODUITS = {"MntWines": "Vin", "MntMeatProducts": "Viande", "MntFishProducts": "Poisson",
            "MntGoldProds": "Or", "MntSweetProducts": "Sucreries", "MntFruits": "Fruits"}
CANAUX = {"NumStorePurchases": "Magasin", "NumWebPurchases": "Web",
          "NumCatalogPurchases": "Catalogue"}


def lire_sql(nom):
    """Renvoie le texte d'un fichier de backend/sql/."""
    return (config.DOSSIER_SQL / nom).read_text(encoding="utf-8")


def retirer_conflits_doublons(df):
    """Retire les conflits sur la cible, puis les doublons (ID mis à part)."""
    sans_id = [c for c in df.columns if c != "ID"]
    sans_id_rep = [c for c in sans_id if c != "Response"]

    # 1. Conflits : mêmes caractéristiques mais Response différente.
    # On ne sait pas quelle réponse est la bonne : tout le groupe est retiré.
    # dropna=False : sans cela, les clients sans Income seraient exclus du groupby.
    n_rep = df.groupby(sans_id_rep, dropna=False)["Response"].transform("nunique")
    conflits = df[n_rep > 1]
    print(f"Conflits retirés : {len(conflits)} lignes "
          f"({conflits.groupby(sans_id_rep, dropna=False).ngroups} groupes)")
    df = df[n_rep == 1]

    # 2. Doublons : on garde une seule copie (le plus petit ID).
    # Sinon une copie pourrait être dans le train et l'autre dans le test (fuite).
    avant = len(df)
    df = df.sort_values("ID").drop_duplicates(subset=sans_id, keep="first")
    print(f"Doublons retirés : {avant - len(df)} lignes")
    return df


def charger():
    """CSV -> table brute -> table sans_doublons -> table analyse."""
    config.CHEMIN_DB.unlink(missing_ok=True)  # on repart d'une base vide
    df = pd.read_csv(config.CHEMIN_CSV, sep="\t")
    with sqlite3.connect(config.CHEMIN_DB) as con:
        con.executescript(lire_sql("load.sql"))
        df.to_sql(config.TABLE_BRUTE, con, if_exists="append", index=False)
        retirer_conflits_doublons(df).to_sql(config.TABLE_SANS_DOUBLONS, con, index=False)
        con.executescript(lire_sql("analysis.sql"))
    print(f"Chargé : {df.shape[0]} lignes x {df.shape[1]} colonnes")


def preparer_base():
    """Crée la base depuis le CSV si elle n'existe pas (dépôt cloné à vide)."""
    if not config.CHEMIN_DB.exists():
        charger()


def lire_analyse():
    """Renvoie la table d'analyse, avec Dt_Customer en date."""
    preparer_base()
    df = sql(f"SELECT * FROM {config.TABLE_ANALYSE}")
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    return df


def sql(requete):
    """Exécute une requête sur la base clients et renvoie un DataFrame."""
    with sqlite3.connect(config.CHEMIN_DB) as con:
        return pd.read_sql(requete, con)


def afficher(titre, resultat):
    """Affiche un titre puis un DataFrame en entier."""
    print(f"\n=== {titre} ===")
    print(resultat.to_string(index=False))


def rapport_qualite():
    """Affiche tous les chiffres nécessaires à la Data Quality Card."""
    n = sql(f"SELECT COUNT(*) AS n FROM {config.TABLE_BRUTE}")["n"][0]
    cols = sql(f"PRAGMA table_info({config.TABLE_BRUTE})")["name"].tolist()
    sans_id = ", ".join(c for c in cols if c != "ID")
    sans_id_rep = ", ".join(c for c in cols if c not in ("ID", "Response"))

    # Identité
    afficher("Taille et identifiants",
             sql(f"SELECT COUNT(*) AS n_lignes, COUNT(DISTINCT ID) AS n_id FROM {config.TABLE_BRUTE}"))
    dates = pd.to_datetime(sql(f"SELECT Dt_Customer FROM {config.TABLE_BRUTE}")["Dt_Customer"],
                           format="mixed", dayfirst=True)
    print(f"\nPériode d'inscription : {dates.min().date()} -> {dates.max().date()}")

    # Cible
    afficher("Cible Response", sql(f"""
        SELECT Response, COUNT(*) AS n, ROUND(100.0 * COUNT(*) / {n}, 1) AS pct
        FROM {config.TABLE_BRUTE} GROUP BY Response"""))

    # Valeurs manquantes (toutes les colonnes)
    manquants = sql("SELECT " + ", ".join(f"SUM({c} IS NULL) AS {c}" for c in cols) + f" FROM {config.TABLE_BRUTE}").T
    manquants.columns = ["n"]
    manquants = manquants[manquants["n"] > 0]
    manquants["pct"] = (100 * manquants["n"] / n).round(1)
    print("\n=== Valeurs manquantes ===")
    print(manquants.to_string())

    # Doublons
    afficher("Doublons (ID exclu)", sql(f"""
        SELECT COALESCE(SUM(k - 1), 0) AS copies_a_retirer,
               ROUND(100.0 * COALESCE(SUM(k - 1), 0) / {n}, 1) AS pct
        FROM (SELECT COUNT(*) AS k FROM {config.TABLE_BRUTE} GROUP BY {sans_id} HAVING k > 1)"""))
    afficher("Mêmes clients, Response différente", sql(f"""
        SELECT COUNT(*) AS groupes, COALESCE(SUM(k), 0) AS lignes
        FROM (SELECT COUNT(*) AS k FROM {config.TABLE_BRUTE} GROUP BY {sans_id_rep}
              HAVING COUNT(DISTINCT Response) > 1)"""))

    # Valeurs impossibles ou aberrantes
    afficher("Naissances avant 1940",
             sql(f"SELECT ID, Year_Birth FROM {config.TABLE_BRUTE} WHERE Year_Birth < 1940"))
    afficher("5 plus hauts revenus",
             sql(f"SELECT ID, Income FROM {config.TABLE_BRUTE} ORDER BY Income DESC LIMIT 5"))
    afficher("Min / max des variables numériques", sql(f"""
        SELECT MIN(Income) AS income_min, MAX(Income) AS income_max,
               MIN(Recency) AS recency_min, MAX(Recency) AS recency_max,
               MIN(Year_Birth) AS naissance_min, MAX(Year_Birth) AS naissance_max
        FROM {config.TABLE_BRUTE}"""))
    afficher("Clients sans achat mais avec des montants", sql(f"""
        SELECT COUNT(*) AS n FROM {config.TABLE_BRUTE}
        WHERE NumWebPurchases + NumCatalogPurchases + NumStorePurchases = 0
          AND MntWines + MntFruits + MntMeatProducts + MntFishProducts
              + MntSweetProducts + MntGoldProds > 0"""))

    # Modalités
    afficher("Marital_Status",
             sql(f"SELECT Marital_Status, COUNT(*) AS n FROM {config.TABLE_BRUTE} GROUP BY 1 ORDER BY n DESC"))
    afficher("Education",
             sql(f"SELECT Education, COUNT(*) AS n FROM {config.TABLE_BRUTE} GROUP BY 1 ORDER BY n DESC"))

    # Colonnes constantes
    constantes = [c for c in cols
                  if sql(f"SELECT COUNT(DISTINCT {c}) AS k FROM {config.TABLE_BRUTE}")["k"][0] == 1]
    print(f"\n=== Colonnes constantes ===\n{constantes}")
    afficher("Valeurs des colonnes Z_",
             sql(f"SELECT DISTINCT Z_CostContact, Z_Revenue FROM {config.TABLE_BRUTE}"))


def resume_clients():
    """Qui sont nos clients ? Médiane, moyenne, min et max des variables clés.
    Avec_enfants vaut 0 ou 1 : sa moyenne est la part des clients avec enfants."""
    df = lire_analyse()
    df["Avec_enfants"] = (df["Enfants"] > 0).astype(int)
    variables = ["Age", "Income", "Enfants", "Avec_enfants", "Depense_totale", "Nb_achats"]
    return df[variables].agg(["median", "mean", "min", "max"]).T.round(2).rename_axis("variable")


def repartition(colonnes):
    """Part (en %) de chaque colonne dans le total de toutes les colonnes."""
    totaux = lire_analyse()[list(colonnes)].sum().rename(colonnes)
    res = pd.DataFrame({"total": totaux, "part_pct": (100 * totaux / totaux.sum()).round(1)})
    return res.sort_values("total", ascending=False)


def parts_produits():
    """Sur quels produits se fait le chiffre d'affaires ? Part de chaque famille
    dans la dépense totale de l'enseigne."""
    return repartition(PRODUITS).rename_axis("produit")


def parts_canaux():
    """Par quels canaux les clients achètent-ils ? Part de chaque canal dans les achats."""
    return repartition(CANAUX).rename_axis("canal")


def taux_reponse(variable, bornes, etiquettes):
    """Le taux de réponse varie-t-il selon cette variable ?
    Groupes [borne, borne suivante) ; les valeurs manquantes forment un groupe « Inconnu »."""
    df = lire_analyse()
    groupes = pd.cut(df[variable], bornes, right=False, labels=etiquettes)
    groupes = groupes.cat.add_categories("Inconnu").fillna("Inconnu")
    res = df.groupby(groupes, observed=True)["Response"].agg(n_clients="size", n_reponses="sum")
    res["taux_pct"] = (100 * res["n_reponses"] / res["n_clients"]).round(1)
    return res


def profils_reponse():
    """Quels profils répondent le plus à la campagne ? Taux de réponse par
    campagnes déjà acceptées, Recency, quartile de revenu et présence d'enfants."""
    # Quartiles calculés sur les revenus connus (les NaN sont ignorés)
    q = list(lire_analyse()["Income"].quantile([0, 0.25, 0.5, 0.75])) + [np.inf]
    return pd.concat({
        "Campagnes acceptées": taux_reponse("Nb_campagnes_acceptees", [0, 1, 2, np.inf],
                                            ["Aucune", "Une", "Deux ou plus"]),
        "Recency": taux_reponse("Recency", [0, 25, 50, 75, np.inf],
                                ["< 25 j", "25-49 j", "50-74 j", "75 j et +"]),
        "Revenu": taux_reponse("Income", q, ["Q1", "Q2", "Q3", "Q4"]),
        "Enfants": taux_reponse("Enfants", [0, 1, np.inf], ["Sans", "Avec"]),
    }, names=["profil", "groupe"])


if __name__ == "__main__":
    charger()
    rapport_qualite()
    n = sql(f"SELECT COUNT(*) AS n FROM {config.TABLE_ANALYSE}")["n"][0]
    print(f"\nTable d'analyse : {n} clients")

    # Exploration de la table d'analyse
    afficher("Résumé des clients", resume_clients().reset_index())
    afficher("Part des produits dans les dépenses", parts_produits().reset_index())
    afficher("Part des canaux dans les achats", parts_canaux().reset_index())
    afficher("Taux de réponse par profil", profils_reponse().reset_index())
