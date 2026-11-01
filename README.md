# 📣 Ciblage de campagne marketing 

![python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-blue)
![streamlit](https://img.shields.io/badge/streamlit-1.66.1-blue)
![sqlite](https://img.shields.io/badge/sqlite-3-blue)

## 🎯 La question business

Contacter tous les clients fait perdre de l'argent à l'enseigne : chaque contact coûte 3 UM et seuls 14,6 % des clients répondent.
**Quels profils de clients existent, et lesquels faut-il contacter lors de la prochaine campagne pour qu'elle devienne rentable ?**

## Présentation

Ce projet aide la direction marketing d'une enseigne de distribution (vin, viande, poisson, fruits, sucreries, or ; en magasin, sur le web et par catalogue) à préparer sa prochaine campagne.

Il combine deux approches :
- une **segmentation** des clients (k-means), qui décrit les profils et montre lesquels exclure 
- un **modèle de prédiction** (forêt aléatoire), qui donne à chaque client une probabilité de répondre, et une **règle de décision** : contacter seulement si cette probabilité dépasse 27,3 %, le seuil où le contact devient rentable.

Le tout est présenté dans une application **Streamlit**, pensée pour un utilisateur métier.

## 📊 Résultats

Sur le jeu de test (402 clients jamais vus par le modèle, dont 59 répondants) :

| Stratégie | Clients contactés | Réponses | Taux de réponse | Gain net |
|---|---|---|---|---|
| Contacter tout le monde | 402 | 59 | 14,7 % | **−557 UM** |
| Baseline : clients ayant déjà accepté une campagne | 88 | 35 | 39,8 % | +121 UM |
| **Forêt aléatoire, contacter si p > 27,3 %** | **81** | **43** | **53,1 %** | **+230 UM** |

- En contactant **20 % des clients**, le modèle capte **73 % des répondants** et fait **près de deux fois mieux que la baseline**.
- En validation croisée (5 plis du train), le gain moyen est de **162,2 ± 30,4 UM** par pli pour la forêt, contre **95,8 ± 14,7** pour la baseline.
- **Pourquoi 27,3 % ?** Un contact coûte 3 UM et une réponse rapporte 11 UM : contacter un client est rentable dès que sa probabilité de réponse dépasse 3/11.

**Les 3 profils de clients (segmentation) :**

| Persona | Clients | Taux de réponse | Action |
|---|---|---|---|
| 👨‍👩‍👧 Familles modestes au panier varié | 534 | 9,6 % | Ne pas contacter |
| 🍷 Familles amatrices de vin | 871 | 10,0 % | Ne pas contacter en masse |
| 💎 Aisés, peu d'enfants, vin et viande | 605 | 25,8 % | Le plus réceptif, mais pas rentable en bloc |

Aucun segment n'est rentable en bloc : c'est le modèle, client par client, qui rend la campagne rentable.

## ✨ L'application

- **🎯 Décision** : on choisit un client, on voit sa probabilité de réponse, la recommandation (contacter ou non), le gain attendu et son persona.
- **🔍 Exploration** : le besoin client, les 3 personas et les graphiques de la segmentation.
- **📊 Modèle** : les performances comparées à la baseline, les variables importantes, l'équité, les limites et les détails techniques.

## 🚀 Lancer le projet

**Prérequis :** Python 3.11 et Git.

```bash
git clone https://gitlab-mi.univ-reims.fr/alle0016/data_minning.git
cd data_minning
```

Puis, depuis le dossier du projet :

```bash
python -m pip install -r requirements.txt   # 1. installer les bibliothèques
python -m backend.train                     # 2. créer la base et entraîner le modèle (environ 3 min)
python -m pytest                            # 3. lancer les tests
python -m streamlit run frontend/app.py     # 4. ouvrir l'application
```

L'application s'ouvre dans le navigateur (en général http://localhost:8501). Si `python --version` n'affiche pas 3.11, remplacer `python` par `py -3.11`.

## 📁 Architecture

```
data_minning/
├── backend/                      # logique métier 
│   ├── config.py                 # chemins, seed, coût d'un contact et revenu d'une réponse
│   ├── sql/
│   │   ├── load.sql              # création de la table brute
│   │   └── analysis.sql          # table d'analyse nettoyée et variables créées
│   ├── data.py                   # CSV -> SQLite -> table d'analyse, rapport qualité, exploration
│   ├── segmentation.py           # ACP, k-means et personas
│   ├── model.py                  # modèles, seuil de décision, gain net, équité
│   ├── train.py                  # python -m backend.train : entraîne et sauvegarde
│   └── service.py                # seule interface appelée par le frontend
├── frontend/
│   └── app.py                    # application Streamlit (3 écrans)
├── .streamlit/
│   └── config.toml               # thème de l'application
├── tests/                        # 24 tests pytest (données, logique métier, segmentation, application)
├── docs/
│   ├── data_quality_card.md
│   ├── model_card.md
│   ├── interpretation_segmentation.md
│   ├── figures/                  # graphiques de la segmentation
│   ├── note_synthese.pdf
│   ├── soutenance.pdf
│   └── journal_ia.md
├── data/raw/
│   └── marketing_campaign.csv    # données brutes, jamais modifiées
├── modeles/                      # généré par python -m backend.train (non versionné)
├── requirements.txt              # versions fixées
└── README.md
```

## 🗂️ Données

Kaggle, [*Customer Personality Analysis*](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) (licence CC0) : 2 240 clients × 29 colonnes, **2 010 clients après nettoyage** (doublons, conflits sur la cible, valeurs impossibles et fiches erronées retirés). Détail dans [`docs/data_quality_card.md`](docs/data_quality_card.md).

| Groupe | Variables | Contenu |
|---|---|---|
| Profil | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` | âge, études, situation familiale, revenu, enfants |
| Relation client | `Dt_Customer`, `Recency`, `Complain` | date d'inscription, jours depuis le dernier achat, réclamation |
| Produits | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` | montants dépensés sur 2 ans |
| Canaux | `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumDealsPurchases`, `NumWebVisitsMonth` | achats par canal, achats en promotion, visites web |
| Campagnes | `AcceptedCmp1` à `AcceptedCmp5`, `Response` | offres acceptées ; `Response` = dernière campagne, la cible |
| Coûts | `Z_CostContact`, `Z_Revenue` | coût d'un contact (3) et revenu d'une réponse (11) |

## ⚠️ Limites

- **Coûts supposés** : `Z_CostContact` et `Z_Revenue` ne sont pas documentées ; nous les interprétons comme le coût d'un contact et le revenu d'une réponse.
- **Pas d'effet causal** : on suppose qu'un client non contacté ne rapporte rien.
- **Petit jeu de test** : 59 répondants ; un écart de quelques dizaines d'UM entre deux stratégies n'est pas significatif.
- **Une seule campagne observée** : une offre différente peut donner d'autres réponses.
- **Données anciennes et sans dimension temporelle** : clients inscrits entre 2012 et 2014, achats cumulés, pays inconnu.
- **Segmentation** : les clients forment un continuum plutôt que des groupes nets (silhouette ≈ 0,2).
- **Variables sensibles** : l'âge et la situation familiale sont utilisés ; l'équité est vérifiée, mais les petits groupes restent à surveiller.

Détail dans [`docs/model_card.md`](docs/model_card.md).


## 👥 Auteurs

Projet réalisé dans le cadre du cours de *Data Mining* du Master SEP, Université de Reims Champagne-Ardenne, promotion 2026-2027.

- Estelle ALLET
- Maëlyss GREIN
- Hugo ROLLAND
- Syrine SLIMAN