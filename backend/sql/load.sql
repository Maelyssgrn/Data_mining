-- Table brute : copie fidèle du CSV, avec des types déclarés explicitement.
-- ID est la clé primaire : SQLite refuse l'insertion si un ID est en double.
CREATE TABLE brute (
    ID                  INTEGER PRIMARY KEY,
    Year_Birth          INTEGER,
    Education           TEXT,
    Marital_Status      TEXT,
    Income              REAL,     -- 24 valeurs manquantes (NULL)
    Kidhome             INTEGER,
    Teenhome            INTEGER,
    Dt_Customer         TEXT,     -- jj-mm-aaaa, converti dans analysis.sql
    Recency             INTEGER,
    MntWines            INTEGER,
    MntFruits           INTEGER,
    MntMeatProducts     INTEGER,
    MntFishProducts     INTEGER,
    MntSweetProducts    INTEGER,
    MntGoldProds        INTEGER,
    NumDealsPurchases   INTEGER,
    NumWebPurchases     INTEGER,
    NumCatalogPurchases INTEGER,
    NumStorePurchases   INTEGER,
    NumWebVisitsMonth   INTEGER,
    AcceptedCmp3        INTEGER,
    AcceptedCmp4        INTEGER,
    AcceptedCmp5        INTEGER,
    AcceptedCmp1        INTEGER,
    AcceptedCmp2        INTEGER,
    Complain            INTEGER,
    Z_CostContact       INTEGER,
    Z_Revenue           INTEGER,
    Response            INTEGER
);
