# Data Quality Card : Customer Personality Analysis

| Question | Réponse |
|---|---|
| Unité d'observation | Un client de l'enseigne |
| Source et licence | Kaggle, *Customer Personality Analysis* (imakash3011) ; licence : CC0: Public Domain |
| Période, population | Clients inscrits du 30/07/2012 au 29/06/2014 ; achats mesurés sur les 2 dernières années ; pays et mode de sélection non décrits |
| Taille | 2 240 lignes × 29 colonnes (dont `ID`) ; `ID` unique |
| Format | Séparateur tabulation ; `Dt_Customer` en texte jour-mois-année, converti en date |
| Valeurs manquantes | `Income` : 24 (1,1 %) ; imputation par la médiane apprise sur le train |
| Doublons | 182 lignes (8,1 %) identiques à un autre client, `ID` mis à part : retirées. 19 groupes (40 lignes) identiques mais avec une `Response` différente : retirés |
| Valeurs impossibles | `Year_Birth` = 1893, 1899, 1900 (3 lignes) ; `Income` = 666 666 (1 ligne, le suivant est 162 397) : retirées. 6 clients sans achat mais avec des montants positifs : gardés, signalés |
| Modalités douteuses | `Marital_Status` : Alone (3), YOLO (2), Absurd (2) ; Alone → Single, YOLO et Absurd → Autre. `Education` : « 2n Cycle » (203) et « Master » (370) regroupés |
| Colonnes inutiles | `Z_CostContact` (= 3) et `Z_Revenue` (= 11) constantes : exclues des variables explicatives |
| Variables sensibles | Âge (`Year_Birth`), situation familiale : à discuter dans la Model Card. Aucune donnée identifiante |
| Cible | À compléter selon la problématique |
| Risque de fuite | À compléter selon la problématique |
| Variables utiles attendues | À compléter selon la problématique |