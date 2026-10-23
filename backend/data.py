"""Chargement du CSV brut dans SQLite"""

import sqlite3

import pandas as pd

CHEMIN_CSV = "data/raw/marketing_campaign.csv"
CHEMIN_DB = "data/clients.db"
TABLE = "brute"


def charger():
    """Lit le CSV brut (séparateur tabulation) et l'écrit dans SQLite."""
    df = pd.read_csv(CHEMIN_CSV, sep="\t")
    with sqlite3.connect(CHEMIN_DB) as con:
        df.to_sql(TABLE, con, if_exists="replace", index=False)
    print(f"Chargé : {df.shape[0]} lignes x {df.shape[1]} colonnes -> table '{TABLE}'")


def sql(requete):
    """Exécute une requête sur la base clients et renvoie un DataFrame."""
    with sqlite3.connect(CHEMIN_DB) as con:
        return pd.read_sql(requete, con)


def afficher(titre, resultat):
    """Affiche un titre puis un DataFrame en entier."""
    print(f"\n=== {titre} ===")
    print(resultat.to_string(index=False))


def rapport_qualite():
    """Affiche tous les chiffres nécessaires à la Data Quality Card."""
    n = sql(f"SELECT COUNT(*) AS n FROM {TABLE}")["n"][0]
    cols = sql(f"PRAGMA table_info({TABLE})")["name"].tolist()
    sans_id = ", ".join(c for c in cols if c != "ID")
    sans_id_rep = ", ".join(c for c in cols if c not in ("ID", "Response"))

    # Identité
    afficher("Taille et identifiants",
             sql(f"SELECT COUNT(*) AS n_lignes, COUNT(DISTINCT ID) AS n_id FROM {TABLE}"))
    dates = pd.to_datetime(sql(f"SELECT Dt_Customer FROM {TABLE}")["Dt_Customer"],
                           format="mixed", dayfirst=True)
    print(f"\nPériode d'inscription : {dates.min().date()} -> {dates.max().date()}")

    # Cible
    afficher("Cible Response", sql(f"""
        SELECT Response, COUNT(*) AS n, ROUND(100.0 * COUNT(*) / {n}, 1) AS pct
        FROM {TABLE} GROUP BY Response"""))

    # Valeurs manquantes (toutes les colonnes)
    manquants = sql("SELECT " + ", ".join(f"SUM({c} IS NULL) AS {c}" for c in cols) + f" FROM {TABLE}").T
    manquants.columns = ["n"]
    manquants = manquants[manquants["n"] > 0]
    manquants["pct"] = (100 * manquants["n"] / n).round(1)
    print("\n=== Valeurs manquantes ===")
    print(manquants.to_string())

    # Doublons
    afficher("Doublons (ID exclu)", sql(f"""
        SELECT COALESCE(SUM(k - 1), 0) AS copies_a_retirer,
               ROUND(100.0 * COALESCE(SUM(k - 1), 0) / {n}, 1) AS pct
        FROM (SELECT COUNT(*) AS k FROM {TABLE} GROUP BY {sans_id} HAVING k > 1)"""))
    afficher("Mêmes clients, Response différente", sql(f"""
        SELECT COUNT(*) AS groupes, COALESCE(SUM(k), 0) AS lignes
        FROM (SELECT COUNT(*) AS k FROM {TABLE} GROUP BY {sans_id_rep}
              HAVING COUNT(DISTINCT Response) > 1)"""))

    # Valeurs impossibles ou aberrantes
    afficher("Naissances avant 1940",
             sql(f"SELECT ID, Year_Birth FROM {TABLE} WHERE Year_Birth < 1940"))
    afficher("5 plus hauts revenus",
             sql(f"SELECT ID, Income FROM {TABLE} ORDER BY Income DESC LIMIT 5"))
    afficher("Min / max des variables numériques", sql(f"""
        SELECT MIN(Income) AS income_min, MAX(Income) AS income_max,
               MIN(Recency) AS recency_min, MAX(Recency) AS recency_max,
               MIN(Year_Birth) AS naissance_min, MAX(Year_Birth) AS naissance_max
        FROM {TABLE}"""))
    afficher("Clients sans achat mais avec des montants", sql(f"""
        SELECT COUNT(*) AS n FROM {TABLE}
        WHERE NumWebPurchases + NumCatalogPurchases + NumStorePurchases = 0
          AND MntWines + MntFruits + MntMeatProducts + MntFishProducts
              + MntSweetProducts + MntGoldProds > 0"""))

    # Modalités
    afficher("Marital_Status",
             sql(f"SELECT Marital_Status, COUNT(*) AS n FROM {TABLE} GROUP BY 1 ORDER BY n DESC"))
    afficher("Education",
             sql(f"SELECT Education, COUNT(*) AS n FROM {TABLE} GROUP BY 1 ORDER BY n DESC"))

    # Colonnes constantes
    constantes = [c for c in cols
                  if sql(f"SELECT COUNT(DISTINCT {c}) AS k FROM {TABLE}")["k"][0] == 1]
    print(f"\n=== Colonnes constantes ===\n{constantes}")
    afficher("Valeurs des colonnes Z_",
             sql(f"SELECT DISTINCT Z_CostContact, Z_Revenue FROM {TABLE}"))


if __name__ == "__main__":
    charger()
    rapport_qualite()