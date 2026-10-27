
import sqlite3

import pandas as pd

CHEMIN_CSV = "data/raw/marketing_campaign.csv"
CHEMIN_DB = "data/clients.db"
TABLE_PROPRE = "clients"

DATE_REFERENCE = pd.Timestamp("2014-06-29")  # dernière date d'inscription du fichier

PRODUITS = ["MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CANAUX = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CAMPAGNES = [f"AcceptedCmp{i}" for i in range(1, 6)]


def charger_brut():
    """Lit le CSV brut (séparateur tabulation)."""
    return pd.read_csv(CHEMIN_CSV, sep="\t")


def nettoyer(df):
    """Applique les règles de la Data Quality Card. Renvoie une copie."""
    df = df.copy()
    n_depart = len(df)

    # 1. Doublons (ID mis à part)
    #    a) mêmes clients mais Response différente : on ne sait pas laquelle croire -> on retire tout le groupe
    cols_sans_id_rep = [c for c in df.columns if c not in ("ID", "Response")]
    contradictoires = df.groupby(cols_sans_id_rep, dropna=False)["Response"].transform("nunique") > 1
    df = df[~contradictoires]
    n_contradictoires = int(contradictoires.sum())
    #    b) vrais doublons : on garde une seule ligne
    cols_sans_id = [c for c in df.columns if c != "ID"]
    n_avant = len(df)
    df = df.drop_duplicates(subset=cols_sans_id, keep="first")
    n_doublons = n_avant - len(df)

    # 2. Valeurs impossibles
    n_avant = len(df)
    df = df[(df["Year_Birth"] >= 1940) & ~(df["Income"] > 600_000)]
    n_impossibles = n_avant - len(df)

    # 3. Modalités douteuses
    df["Marital_Status"] = df["Marital_Status"].replace(
        {"Alone": "Single", "YOLO": "Autre", "Absurd": "Autre"})
    df["Education"] = df["Education"].replace({"2n Cycle": "Master"})

    # 4. Date d'inscription en vraie date
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="mixed", dayfirst=True)

    # 5. Colonnes constantes : retirées (gardées en tête pour le calcul du profit : coût 3, revenu 11)
    df = df.drop(columns=["Z_CostContact", "Z_Revenue"])

    # Income manquant : volontairement laissé vide ici.
    # L'imputation (médiane) se fait plus tard, sur le train pour le supervisé,
    # afin d'éviter toute fuite d'information.

    print(f"Départ : {n_depart} lignes")
    print(f"  - {n_contradictoires} lignes contradictoires (même client, Response différente)")
    print(f"  - {n_doublons} doublons")
    print(f"  - {n_impossibles} valeurs impossibles (naissance < 1940, revenu 666 666)")
    print(f"Après nettoyage : {len(df)} lignes")
    return df.reset_index(drop=True)


def creer_variables(df):
    """Ajoute les variables construites utiles pour la segmentation et la prédiction."""
    df = df.copy()
    df["Age"] = DATE_REFERENCE.year - df["Year_Birth"]
    df["Enfants"] = df["Kidhome"] + df["Teenhome"]
    df["Mnt_total"] = df[PRODUITS].sum(axis=1)
    df["Achats_total"] = df[CANAUX].sum(axis=1)
    df["Part_vin"] = (df["MntWines"] / df["Mnt_total"].where(df["Mnt_total"] > 0)).fillna(0)
    df["Anciennete_jours"] = (DATE_REFERENCE - df["Dt_Customer"]).dt.days
    df["Nb_campagnes_acceptees"] = df[CAMPAGNES].sum(axis=1)
    df["Sans_achat_signale"] = ((df["Achats_total"] == 0) & (df["Mnt_total"] > 0)).astype(int)
    return df


def sauvegarder(df):
    """Écrit la table propre dans SQLite."""
    with sqlite3.connect(CHEMIN_DB) as con:
        df.to_sql(TABLE_PROPRE, con, if_exists="replace", index=False)
    print(f"Table '{TABLE_PROPRE}' écrite dans {CHEMIN_DB} : {df.shape[0]} lignes x {df.shape[1]} colonnes")


def preparer():
    """Chaîne complète : CSV brut -> nettoyage -> variables -> SQLite."""
    df = creer_variables(nettoyer(charger_brut()))
    sauvegarder(df)
    return df


if __name__ == "__main__":
    clients = preparer()
    print(f"\nIncome manquants restants : {clients['Income'].isna().sum()}")
    print(f"Taux de réponse : {clients['Response'].mean():.1%}")
    print("\nMarital_Status :", clients["Marital_Status"].value_counts().to_dict())
    print("Education :", clients["Education"].value_counts().to_dict())
    print("\nAperçu des nouvelles variables :")
    print(clients[["Age", "Enfants", "Mnt_total", "Achats_total", "Part_vin",
                   "Anciennete_jours", "Nb_campagnes_acceptees"]].describe().round(1).T)
