
"""
Prédiction de la réponse à une campagne marketing.

Objectif :
Comparer une régression logistique et une Random Forest
pour prédire si un client accepte une campagne marketing.

Variable cible : Response (0 = refus, 1 = acceptation).
Indicateur principal : Average Precision.

"""

import sqlite3

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (StratifiedKFold,cross_val_score,train_test_split,)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Chargement des données

connexion = sqlite3.connect("data/clients.db")
df = pd.read_sql_query("SELECT * FROM analyse", connexion)
connexion.close()


# 2. Sélection des variables

# Caractéristiques utilisées pour prédire la réponse à la campagne.
variables_numeriques = [
    "Age",
    "Income",
    "Enfants",
    "MntWines",
    "Recency",
    "Nb_achats",
    "Anciennete_jours",
]

variables_categorielles = ["Education"]

X = df[variables_numeriques + variables_categorielles]
y = df["Response"]


# 3. Séparation entraînement / test

# Le jeu de test (20 %) est réservé à l'évaluation finale.
# La stratification conserve la proportion de réponses positives.
X_train, X_test, y_train, y_test = train_test_split(X,y,
    test_size=0.2,
    random_state=42,
    stratify=y,)


# 4. Prétraitement des données

# Variables numériques :
# - remplacement des valeurs manquantes par la médiane ;
# - standardisation des variables.
pipeline_numerique = Pipeline([
    ("imputation", SimpleImputer(strategy="median")),
    ("standardisation", StandardScaler()),])

# Variables catégorielles :
# - remplacement des valeurs manquantes par la modalité fréquente ;
# - transformation des catégories en colonnes numériques.
pipeline_categoriel = Pipeline([
    ("imputation", SimpleImputer(strategy="most_frequent")),
    ("encodage", OneHotEncoder(
        handle_unknown="ignore",
        drop="first",)),])

# Application des traitements selon le type de variable.
preprocesseur = ColumnTransformer([
    ("numerique", pipeline_numerique, variables_numeriques),
    ("categoriel", pipeline_categoriel, variables_categorielles),])


# 5. Définition des modèles

# Chaque modèle inclut le prétraitement pour éviter
# les fuites de données pendant la validation croisée.
modeles = {
    "Régression logistique": Pipeline([
        ("preparation", preprocesseur),
        ("classification", LogisticRegression(
            max_iter=1000,
            random_state=42,)),]),
    "Random Forest": Pipeline([
        ("preparation", preprocesseur),
        ("classification", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,)),]),
}


# 6. Comparaison par validation croisée

# Cinq groupes stratifiés pour conserver la proportion
# de clients ayant accepté la campagne.
validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,)

resultats = {}

for nom, modele in modeles.items():
    scores = cross_val_score(
        modele,
        X_train,
        y_train,
        cv=validation,
        scoring="average_precision",)

    resultats[nom] = scores.mean()

    print(f"\n{nom}")
    print(f"Average Precision moyenne : {scores.mean():.4f}")
    print(f"Écart-type : {scores.std():.4f}")


# 7. Sélection et entraînement du meilleur modèle

# Le modèle ayant la meilleure Average Precision moyenne
# est retenu pour l'évaluation finale.
meilleur_nom = max(resultats, key=resultats.get)
meilleur_modele = modeles[meilleur_nom]

meilleur_modele.fit(X_train, y_train)

print(f"\nModèle retenu : {meilleur_nom}")


# 8. Évaluation sur le jeu de test

# Le test est utilisé après la sélection du modèle.
y_pred = meilleur_modele.predict(X_test)

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, zero_division=0))