"""Paramètres du projet : chemins, seed et paramètres métier."""

from pathlib import Path

# Chemins construits depuis la racine du dépôt : ils fonctionnent
# quel que soit le dossier d'où l'on lance pytest, streamlit ou train.
RACINE = Path(__file__).resolve().parent.parent
CHEMIN_CSV = RACINE / "data" / "raw" / "marketing_campaign.csv"  # brut, jamais modifié
CHEMIN_DB = RACINE / "data" / "clients.db"                      # régénérable depuis le CSV
DOSSIER_SQL = RACINE / "backend" / "sql"
DOSSIER_MODELES = RACINE / "modeles"

# Tables SQLite
TABLE_BRUTE = "brute"
TABLE_SANS_DOUBLONS = "sans_doublons"
TABLE_ANALYSE = "analyse"

# Reproductibilité (split, validation croisée, k-means)
SEED = 42

# Paramètres métier, en unités monétaires (UM, devise inconnue).
# HYPOTHÈSE : les colonnes Z_CostContact (= 3) et Z_Revenue (= 11) ne sont pas
# documentées. Nous les interprétons comme le coût d'un contact et le revenu
# d'une réponse positive. C'est cohérent avec les données :
# 2 240 x 3 = 6 720 UM de coût pour 334 x 11 = 3 674 UM de revenu.
COUT_CONTACT = 3
REVENU_REPONSE = 11
