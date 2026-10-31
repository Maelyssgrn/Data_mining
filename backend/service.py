"""Seule interface appelée par le frontend (aucun import de streamlit ici).

Le modèle et les tableaux de résultats sont relus depuis config.DOSSIER_MODELES
(écrits par python -m backend.train) : rien n'est réentraîné. Chaque fonction
s'utilise aussi en ligne de commande, par exemple :
    python -c "from backend import service; print(service.decision_client(5524))"
"""

import re
from functools import lru_cache

import joblib

from backend import config, data, model, segmentation

MESSAGE_ABSENT = "Le modèle n'est pas encore entraîné. Lancer : python -m backend.train"

# Figures de docs/figures/ affichées dans l'écran Exploration (le plan de l'ACP est
# redessiné par le frontend aux couleurs des personas, son PNG n'est pas repris).
# Les chiffres des explications viennent de python -m backend.segmentation.
FIGURES = [
    ("choix_k.png", "Choix du nombre de segments",
     "L'inertie baisse nettement jusqu'à k = 3, puis se tasse (coude) ; la silhouette reste "
     "proche de son maximum. k = 3 donne trois profils lisibles pour le marketing."),
    ("acp_segmentation_eboulis.png", "Variance expliquée par les axes de l'ACP",
     "Les deux premiers axes résument 49 % de l'information des 11 variables de segmentation."),
    ("acp_segmentation_cercle.png", "Cercle des corrélations : variables de segmentation",
     "Axe 1 : clients aisés, vin et catalogue, contre petits paniers variés (poisson, fruits, "
     "sucreries, or). Axe 2 : familles avec enfants, vin et web, contre catalogue et viande."),
    ("acp_montants_cercle.png", "Cercle des corrélations : montants bruts",
     "Tous les montants vont dans le même sens sur l'axe 1 (56 % de la variance) : c'est un "
     "effet taille. Segmenter sur les montants trierait les clients par niveau de dépense ; "
     "c'est pourquoi la segmentation utilise les parts."),
]


# Chargements (une seule fois par processus)

@lru_cache(maxsize=1)
def _modele():
    if not config.CHEMIN_MODELE.exists():
        raise FileNotFoundError(MESSAGE_ABSENT)
    return joblib.load(config.CHEMIN_MODELE)


@lru_cache(maxsize=1)
def _test():
    """Les clients du test : le modèle n'a pas vu leurs réponses."""
    return model.separer(data.lire_analyse())[1]


@lru_cache(maxsize=1)
def _segmentation():
    """Variables de segmentation, segment et nom du persona de chaque client (index ID)."""
    X = segmentation.variables_segmentation()
    segments = segmentation.segmenter(X)
    noms = {s: nom for s, (_, nom, _) in segmentation.noms_segments(X, segments).items()}
    return X, segments, segments.map(noms)


# Fonctions appelées par le frontend

def modele_disponible():
    """Vrai si python -m backend.train a déjà été lancé (modèle et résultats présents)."""
    return config.CHEMIN_MODELE.exists() and config.CHEMIN_RESULTATS.exists()


def parametres_metier():
    return {"cout_contact": config.COUT_CONTACT, "revenu_reponse": config.REVENU_REPONSE,
            "seuil": model.SEUIL_RENTABILITE}


def clients_test():
    """Les 402 clients du test (index ID) : probabilité, décision, gain attendu, persona, profil."""
    test = _test()
    clients = test.set_index("ID")[["Age", "Income", "Enfants", "Marital_Status",
                                    "Nb_campagnes_acceptees", "Recency", "Depense_totale"]].copy()
    clients.insert(0, "probabilite", _modele().predict_proba(model.variables(test))[:, 1])
    clients.insert(1, "contacter", model.contacter_si_rentable(_modele(), test).to_numpy())
    clients.insert(2, "gain_attendu", model.gain_attendu(clients["probabilite"]))
    clients.insert(3, "persona", _segmentation()[2].loc[clients.index])
    clients["reponse_reelle"] = test["Response"].to_numpy()
    return clients


def decision_client(id_client):
    """Recommandation pour un client du test, sous forme de dictionnaire."""
    clients = clients_test()
    if id_client not in clients.index:
        raise KeyError(f"Le client {id_client} n'est pas dans le jeu de test")
    return {"id": id_client, **clients.loc[id_client].to_dict()}


def besoin_client():
    """Contacter tout le monde ou la baseline, sur les 2 010 clients."""
    return model.besoin_client(data.lire_analyse())


def personas():
    """Les 3 personas : portrait, taux de réponse, gain si on contacte tout le segment, action."""
    X, segments, _ = _segmentation()
    table = segmentation.personas(X, segments)
    return table[table["segment"] != "-"].reset_index(drop=True)


def plan_acp():
    """Coordonnées des clients sur les axes 1 et 2 de l'ACP, avec leur persona,
    et la part de variance (%) de ces deux axes (pour titrer le graphique)."""
    X, _, noms = _segmentation()
    vp, _, _, coords = segmentation.acp_segmentation(X)
    coords["persona"] = noms
    return coords, vp["variance_%"].head(2).tolist()


def figures():
    """(titre, chemin, explication) de chaque figure de docs/figures/ à afficher."""
    return [(titre, config.DOSSIER_FIGURES / fichier, explication)
            for fichier, titre, explication in FIGURES]


def resultats_modele():
    """Les tableaux sauvegardés par python -m backend.train."""
    if not config.CHEMIN_RESULTATS.exists():
        raise FileNotFoundError(MESSAGE_ABSENT)
    return joblib.load(config.CHEMIN_RESULTATS)


def limites():
    """Les limites, phrase par phrase, lues dans la ligne « Limites » de la Model Card
    (une seule source : si la card change, l'application suit)."""
    for ligne in config.CHEMIN_MODEL_CARD.read_text(encoding="utf-8").splitlines():
        if ligne.startswith("| Limites |"):
            return re.split(r"(?<=\.)\s+", ligne.split("|")[2].strip())
    raise ValueError("Ligne « Limites » absente de la Model Card")
