"""Modèles, seuil de décision et métriques métier."""

import pandas as pd

from backend import config

# Contacter un client de probabilité de réponse p rapporte en moyenne
# p * REVENU_REPONSE - COUT_CONTACT : c'est positif si p > 3 / 11 = 27,3 %.
# (Coûts supposés : voir l'hypothèse sur les colonnes Z_ dans config.py.)
SEUIL_RENTABILITE = config.COUT_CONTACT / config.REVENU_REPONSE


def gain_net(n_contactes, n_reponses):
    """Gain net d'une campagne en UM : revenus des réponses moins coût des contacts."""
    return n_reponses * config.REVENU_REPONSE - n_contactes * config.COUT_CONTACT


def regle_deja_accepte(df):
    """Baseline : contacter les clients ayant déjà accepté au moins une campagne."""
    return df["Nb_campagnes_acceptees"] > 0


def bilan_campagne(contacter, response):
    """Bilan d'un ciblage. contacter : booléens (qui on contacte) ; response : la cible."""
    n_contactes = int(contacter.sum())
    n_reponses = int(response[contacter].sum())
    taux = 100 * n_reponses / n_contactes if n_contactes else 0.0
    return {"n_contactes": n_contactes, "n_reponses": n_reponses,
            "taux_pct": round(taux, 1), "gain_net": gain_net(n_contactes, n_reponses)}


def besoin_client(df):
    """Contacter tout le monde est-il rentable ? Comparaison avec la règle simple."""
    tous = pd.Series(True, index=df.index)
    # from_dict(orient="index") : une ligne par stratégie, chaque colonne garde son type
    return pd.DataFrame.from_dict({
        "Tout le monde": bilan_campagne(tous, df["Response"]),
        "Déjà accepté une campagne": bilan_campagne(regle_deja_accepte(df), df["Response"]),
    }, orient="index").rename_axis("strategie")


if __name__ == "__main__":
    from backend import data
    print(f"Seuil de rentabilité : {SEUIL_RENTABILITE:.1%}")
    data.afficher("Besoin client (table nettoyée)", besoin_client(data.lire_analyse()).reset_index())
