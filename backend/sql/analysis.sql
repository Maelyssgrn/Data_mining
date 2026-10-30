-- Table d'analyse : clients nettoyés + variables créées.
-- Date de référence : 29/06/2014 (dernière inscription ; date de collecte inconnue).
-- Income n'est PAS imputé ici : la médiane sera apprise sur le train (pas de fuite).
-- Z_CostContact et Z_Revenue sont constantes : non reprises (valeurs dans config.py).
CREATE TABLE analyse AS
SELECT
    ID,
    Year_Birth,
    CASE WHEN Education = '2n Cycle' THEN 'Master' ELSE Education END AS Education,
    CASE WHEN Marital_Status = 'Alone' THEN 'Single'
         WHEN Marital_Status IN ('YOLO', 'Absurd') THEN 'Autre'
         ELSE Marital_Status END AS Marital_Status,
    Income,
    Kidhome,
    Teenhome,
    -- jj-mm-aaaa -> aaaa-mm-jj (format ISO, lu par SQLite et pandas)
    substr(Dt_Customer, 7, 4) || '-' || substr(Dt_Customer, 4, 2) || '-'
        || substr(Dt_Customer, 1, 2) AS Dt_Customer,
    Recency,
    MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds,
    NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases,
    NumWebVisitsMonth,
    AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5,
    Complain,
    Response,
    -- Variables créées
    2014 - Year_Birth AS Age,
    CAST(julianday('2014-06-29')
         - julianday(substr(Dt_Customer, 7, 4) || '-' || substr(Dt_Customer, 4, 2)
                     || '-' || substr(Dt_Customer, 1, 2)) AS INTEGER) AS Anciennete_jours,
    Kidhome + Teenhome AS Enfants,
    MntWines + MntFruits + MntMeatProducts + MntFishProducts
        + MntSweetProducts + MntGoldProds AS Depense_totale,
    -- Sans NumDealsPurchases : les achats en promotion sont déjà comptés dans les 3 canaux
    NumWebPurchases + NumCatalogPurchases + NumStorePurchases AS Nb_achats,
    -- Sans Response : ce serait mettre la cible dans une variable explicative (fuite)
    AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5
        AS Nb_campagnes_acceptees
FROM sans_doublons
-- Valeurs impossibles. "Income IS NULL OR" est indispensable : en SQL, NULL <> 666666
-- vaut NULL et non vrai, donc sans cette condition les 24 clients sans Income disparaîtraient.
WHERE Year_Birth >= 1901
  AND (Income IS NULL OR Income <> 666666)
  -- Fiches erronées : aucun achat mais des montants > 0 (6 clients)
  AND NumWebPurchases + NumCatalogPurchases + NumStorePurchases > 0;
