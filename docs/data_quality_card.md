# Data Quality Card : Customer Personality Analysis

## Identité

| Question | Réponse |
|---|---|
| Unité d'observation (une ligne = ?) | Un client de l'enseigne |
| Source, licence, lien | Kaggle, *Customer Personality Analysis* (imakash3011), CC0: Public Domain, https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis |
| Période et population couvertes | Clients inscrits du 30/07/2012 au 29/06/2014 ; achats cumulés sur les 2 dernières années ; pays et devise inconnus |
| Taille (lignes × colonnes) | Brut : 2 240 × 29 ; après nettoyage : 2 010 × 33 (table `analyse`) |
| Cible : définition, construction, répartition | `Response` = 1 si le client a accepté l'offre de la dernière campagne ; fournie telle quelle ; 294 / 2 010 (14,6 %) |

## Défauts trouvés et ce qu'on en fait

| Défaut | Combien (chiffre et %) | Comment on l'a vu | Ce qu'on en fait, et pourquoi |
|---|---|---|---|
| Valeurs manquantes | `Income` : 24 (1,1 %) | `SUM(col IS NULL)` en SQL | Gardées ; médiane apprise sur le train, pour éviter une fuite |
| Doublons | 182 (8,1 %), dont 40 lignes (19 groupes) en conflit sur `Response` | `GROUP BY ... HAVING COUNT(*) > 1` et `COUNT(DISTINCT Response) > 1` | Conflits retirés (vraie réponse inconnue), puis 180 doublons retirés : sinon un même client peut être dans le train et dans le test |
| Valeurs aberrantes ou impossibles | `Year_Birth` < 1901 : 3 ; `Income` = 666 666 : 1 (0,2 %) | Min / max et tri en SQL | Retirées : âge > 110 ans, revenu 4 fois supérieur au suivant |
| Modalités qui jouent le rôle d'un manquant | `Marital_Status` YOLO, Absurd : 4 ; Alone : 3 | `GROUP BY Marital_Status` | YOLO, Absurd → Autre ; Alone → Single |
| Classes déséquilibrées | 14,6 % de positifs | Répartition de `Response` | Split stratifié ; pas d'accuracy seule ; seuil fixé par le coût (3/11) |
| Incohérences | 6 clients sans achat avec des montants > 0 (0,3 %) | Requête SQL sur `Num*` et `Mnt*` | Retirés : fiches erronées (revenus extrêmes, dépense de 5 à 9) |
| Colonnes constantes | `Z_CostContact` = 3, `Z_Revenue` = 11 | `COUNT(DISTINCT col) = 1` | Exclues des variables ; interprétées comme coût d'un contact et revenu d'une réponse (hypothèse, en UM) |

## Les pièges pour le modèle

| Question | Réponse |
|---|---|
| Variables à risque de fuite (connues seulement après la décision) | `Response` : cible, exclue ; `Nb_campagnes_acceptees` calculée sans `Response` ; `AcceptedCmp1-5` : campagnes antérieures, gardées ; `Z_` : exclues ; `ID` : exclu ; `Mnt*` : gardées, risque non vérifiable (peuvent inclure des achats liés à la dernière campagne) |
| Variables sensibles (sexe, origine, santé, âge…) | `Age`, `Marital_Status` : gardées avec une vérification d'équité (taux de contact par tranche d'âge et par situation familiale) |
| Biais de sélection (qui manque dans les données ?) | Seuls les clients inscrits entre 2012 et 2014, sans prospects ; mode de sélection et pays inconnus |
| Données personnelles : base légale, anonymisation | Aucune donnée identifiante, `ID` pseudonyme ; données publiques CC0.