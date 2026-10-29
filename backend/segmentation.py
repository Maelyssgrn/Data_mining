"""Segmentation des clients (k-means) et ACP pour la visualiser.

    Variables (11) : revenu, enfants, part de chaque produit dans la dépense du client
    et part de chaque canal dans ses achats. Ni Response, ni ID, ni Z_.
    Pipeline : imputation médiane -> standardisation -> k-means.
    ACP (a) sur les variables de segmentation, pour voir les segments.
    ACP (b) sur les montants bruts, pour montrer l'effet taille qui justifie les parts.

    python -m backend.segmentation : affiche les résultats et enregistre les
    graphiques dans docs/figures/. Rien ne s'exécute à l'import.
"""

import matplotlib
matplotlib.use("Agg")  # enregistre les figures sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend import config, data, model

# k = 3 : coude net (dernière forte baisse d'inertie), silhouette proche du
# maximum (atteint à k = 2), et 3 profils lisibles pour le marketing
K_SEGMENTS = 3
N_AXES = 2  # axes de l'ACP interprétés

# Persona de chaque catégorie de revenu (du plus modeste au plus aisé) : nom et action.
# {taux} et {perte} sont remplis par le calcul (personas) ; le texte, lui, est
# rédigé à partir des résultats pour k = 3 : à relire si les segments changent.
PERSONAS = {
    "modeste": ("Familles modestes au panier varié",
                "Ne pas contacter : réponse faible ({taux})."),
    "moyen": ("Familles amatrices de vin",
              "Ne pas contacter en masse : réponse faible ({taux})."),
    "aisé": ("Aisés, peu d'enfants, vin et viande",
             "Segment le plus réceptif ({taux}), mais pas rentable en bloc : "
             "contacter tout le segment fait encore perdre {perte} UM."),
}

BLEU, ROUGE, VERT = "#4C72B0", "#C44E52", "#55A868"


# Calculs

def variables_segmentation():
    """Les 11 variables de segmentation, une ligne par client (index ID).
    Les parts sont calculées client par client : avec les montants bruts,
    les segments ne feraient que trier les clients par niveau de dépense."""
    df = data.lire_analyse().set_index("ID")
    X = df[["Income", "Enfants"]].copy()
    for col, nom in data.PRODUITS.items():
        X[f"Part_{nom}"] = df[col] / df["Depense_totale"]
    for col, nom in data.CANAUX.items():
        X[f"Part_{nom}"] = df[col] / df["Nb_achats"]  # Nb_achats >= 1 (analysis.sql)
    return X


def pipeline_segmentation(k):
    """Même pipeline pour choisir k et pour segmenter."""
    return Pipeline([
        ("imputation", SimpleImputer(strategy="median")),  # 24 Income manquants
        # Sans standardisation, Income (en UM) écraserait les parts (entre 0 et 1)
        ("standardisation", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=config.SEED)),
    ])


def choisir_k(X, ks=range(2, 7)):
    """Inertie (méthode du coude) et silhouette moyenne pour chaque k."""
    lignes = []
    for k in ks:
        pipe = pipeline_segmentation(k).fit(X)
        Z = pipe[:-1].transform(X)
        lignes.append({"k": k, "inertie": pipe[-1].inertia_,
                       "silhouette": silhouette_score(Z, pipe[-1].labels_)})
    res = pd.DataFrame(lignes)
    res["baisse_inertie_%"] = -100 * res["inertie"].pct_change()
    return res


def segmenter(X, k=K_SEGMENTS):
    """Segment de chaque client (Series indexée par ID)."""
    return pd.Series(pipeline_segmentation(k).fit_predict(X), index=X.index, name="Segment")


def noms_segments(X, segments):
    """Catégorie de revenu, nom et action de chaque segment. Les numéros du k-means
    sont arbitraires : on nomme selon l'ordre des revenus médians."""
    ordre = X["Income"].groupby(segments).median().sort_values().index
    return {s: (cat, *PERSONAS[cat]) for s, cat in zip(ordre, PERSONAS, strict=True)}


def principaux(parts, n=2):
    """Les n plus grandes parts moyennes, en texte : « Vin 64 %, Viande 20 % »."""
    top = parts.sort_values(ascending=False).head(n)
    return ", ".join(f"{nom.replace('Part_', '')} {v:.0f} %" for nom, v in top.items())


def portrait(X, df, masque):
    """Portrait chiffré d'un groupe de clients, et bilan si on le contacte en entier."""
    parts = 100 * X.loc[masque].filter(like="Part_").mean()
    bilan = model.bilan_campagne(masque, df["Response"])
    return {
        "n_clients": int(masque.sum()),
        "revenu_median": X.loc[masque, "Income"].median(),
        "avec_enfants_%": 100 * (X.loc[masque, "Enfants"] > 0).mean(),
        "depense_mediane": df.loc[masque, "Depense_totale"].median(),
        "produits_principaux": principaux(parts[[f"Part_{n}" for n in data.PRODUITS.values()]]),
        "canaux_principaux": principaux(parts[[f"Part_{n}" for n in data.CANAUX.values()]]),
        "taux_reponse_%": bilan["taux_pct"],
        "gain_net": bilan["gain_net"],
    }


def personas(X, segments):
    """Une ligne par segment (nom, portrait, action), puis l'ensemble des clients
    pour comparer. Response sert seulement à décrire, pas à construire les segments."""
    df = data.lire_analyse().set_index("ID").loc[X.index]
    lignes = []
    for s, (categorie, nom, action) in sorted(noms_segments(X, segments).items(),
                                               key=lambda item: list(PERSONAS).index(item[1][0])):
        p = portrait(X, df, segments == s)
        taux = f"{p['taux_reponse_%']:.1f} %".replace(".", ",")
        lignes.append({"segment": s, "nom": nom, "categorie_revenu": categorie, **p,
                       "action": action.format(taux=taux, perte=-p["gain_net"])})
    tous = pd.Series(True, index=X.index)
    lignes.append({"segment": "-", "nom": "Ensemble des clients", "categorie_revenu": "-",
                   **portrait(X, df, tous), "action": "-"})
    return pd.DataFrame(lignes)


def comparaison_ciblages(X, segments):
    """Contacter un segment fait-il mieux qu'un profil simple ou que la baseline ?
    Revenu Q4 et Sans enfants : mêmes définitions que data.profils_reponse()."""
    df = data.lire_analyse().set_index("ID").loc[X.index]
    ciblages = {nom: segments == s for s, (_, nom, _) in noms_segments(X, segments).items()}
    ciblages["Revenu Q4"] = df["Income"] >= df["Income"].quantile(0.75)
    ciblages["Sans enfants"] = df["Enfants"] == 0
    ciblages["Déjà accepté une campagne (baseline)"] = model.regle_deja_accepte(df)
    ciblages["Tout le monde"] = pd.Series(True, index=df.index)
    return pd.DataFrame.from_dict(
        {nom: model.bilan_campagne(masque, df["Response"]) for nom, masque in ciblages.items()},
        orient="index").rename_axis("ciblage")


def tableau_valeurs_propres(acp):
    """Valeur propre et part de variance de chaque axe."""
    return pd.DataFrame({
        "axe": range(1, len(acp.explained_variance_) + 1),
        "valeur_propre": acp.explained_variance_,
        "variance_%": 100 * acp.explained_variance_ratio_,
        "cumulee_%": 100 * acp.explained_variance_ratio_.cumsum(),
    })


def indicateurs_variables(acp, noms, n_axes):
    """Corrélations et contributions (%) des variables sur les n premiers axes."""
    colonnes = [f"Axe {i + 1}" for i in range(n_axes)]
    corr = pd.DataFrame(acp.components_[:n_axes].T * np.sqrt(acp.explained_variance_[:n_axes]),
                        index=noms, columns=colonnes)
    contrib = 100 * pd.DataFrame(acp.components_[:n_axes].T ** 2, index=noms, columns=colonnes)
    return corr, contrib


def analyser_acp(Z, noms, index):
    """ACP sur des données standardisées : valeurs propres, corrélations,
    contributions et coordonnées des clients sur les N_AXES premiers axes."""
    acp = PCA(random_state=config.SEED).fit(Z)
    corr, contrib = indicateurs_variables(acp, noms, N_AXES)
    coords = pd.DataFrame(acp.transform(Z)[:, :N_AXES], index=index, columns=corr.columns)
    return tableau_valeurs_propres(acp), corr, contrib, coords


def acp_segmentation(X):
    """ACP (a) sur les 11 variables, avec le même prétraitement que le k-means."""
    Z = pipeline_segmentation(K_SEGMENTS)[:-1].fit_transform(X)
    return analyser_acp(Z, list(X.columns), X.index)


def acp_montants():
    """ACP (b) sur les montants bruts par produit : valeurs propres et corrélations."""
    df = data.lire_analyse()
    montants = df[list(data.PRODUITS)].set_axis(list(data.PRODUITS.values()), axis=1)
    vp, corr, _, _ = analyser_acp(StandardScaler().fit_transform(montants),
                                  list(montants.columns), df["ID"])
    return vp, corr


# Graphiques (enregistrés dans config.DOSSIER_FIGURES)

def enregistrer(fig, fichier):
    """Crée le dossier si besoin, enregistre la figure puis la ferme."""
    config.DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.DOSSIER_FIGURES / fichier, dpi=120)
    plt.close(fig)


def graphique_choix_k(scores):
    """Coude (inertie) et silhouette selon k, le k retenu en pointillés."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(scores["k"], scores["inertie"], "o-", color=BLEU)
    ax1.set(xlabel="Nombre de segments k", ylabel="Inertie intra-segments",
            title="Méthode du coude")
    ax2.plot(scores["k"], scores["silhouette"], "o-", color=ROUGE)
    ax2.set(xlabel="Nombre de segments k", ylabel="Silhouette moyenne", title="Silhouette")
    for ax in (ax1, ax2):
        ax.axvline(K_SEGMENTS, color="grey", ls="--", lw=1)
    enregistrer(fig, "choix_k.png")


def graphique_eboulis(vp, nom_acp, fichier):
    axes = vp["axe"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(axes, vp["variance_%"], color=BLEU, label="par axe")
    ax.plot(axes, vp["cumulee_%"], "o-", color=ROUGE, label="cumulée")
    ax.set(xlabel="Axe", ylabel="% de variance", title=f"{nom_acp} : variance expliquée")
    ax.legend()
    enregistrer(fig, fichier)


def couleur(nom):
    """Couleur d'une variable dans le cercle : produits en bleu, canaux en vert,
    revenu et enfants en rouge."""
    if nom.replace("Part_", "") in data.CANAUX.values():
        return VERT
    if nom in ("Income", "Enfants"):
        return ROUGE
    return BLEU


def graphique_cercle(corr, vp, nom_acp, fichier):
    """Cercle des corrélations sur les axes 1 et 2."""
    var = vp["variance_%"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="grey"))
    for nom, (x, y) in corr[["Axe 1", "Axe 2"]].iterrows():
        ax.arrow(0, 0, x, y, head_width=0.03, color=couleur(nom), length_includes_head=True)
        ax.text(x * 1.12, y * 1.12, nom, ha="center", va="center", color=couleur(nom))
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal",
           xlabel=f"Axe 1 ({var[0]:.1f} %)", ylabel=f"Axe 2 ({var[1]:.1f} %)",
           title=f"{nom_acp} : cercle des corrélations")
    enregistrer(fig, fichier)


def graphique_plan(coords, segments, vp, fichier):
    """Clients sur le plan 1-2, colorés par segment."""
    var = vp["variance_%"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in sorted(segments.unique()):
        m = segments == s
        ax.scatter(coords.loc[m, "Axe 1"], coords.loc[m, "Axe 2"], s=6, alpha=0.6,
                   label=f"Segment {s}")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set(xlabel=f"Axe 1 ({var[0]:.1f} %)", ylabel=f"Axe 2 ({var[1]:.1f} %)",
           title="ACP des variables de segmentation : clients par segment")
    ax.legend(markerscale=3)
    enregistrer(fig, fichier)


# Programme principal

def titre(texte):
    print(f"\n{'=' * 70}\n{texte}\n{'=' * 70}")


def main():
    X = variables_segmentation()
    print(f"{len(X)} clients, {X.shape[1]} variables : {', '.join(X.columns)}")

    titre("CHOIX DE k : coude, puis silhouette")
    scores = choisir_k(X)
    print(scores.round({"inertie": 0, "silhouette": 3, "baisse_inertie_%": 1}).to_string(index=False))
    graphique_choix_k(scores)

    titre(f"PERSONAS DES {K_SEGMENTS} SEGMENTS")
    segments = segmenter(X)
    table = personas(X, segments)
    print(table.drop(columns="action").round(1).set_index("nom").T.to_string())
    print("\nActions :")
    for nom, action in zip(table["nom"], table["action"]):
        if action != "-":
            print(f"- {nom} : {action}")

    titre("CONTACTER UN SEGMENT FAIT-IL MIEUX QU'UN PROFIL SIMPLE ?")
    print(comparaison_ciblages(X, segments).to_string())

    titre("ACP (a) : VARIABLES DE SEGMENTATION")
    vp, corr, contrib, coords = acp_segmentation(X)
    print(vp.round(2).head(5).to_string(index=False))
    print(f"\nCorrélations variables / axes :\n{corr.round(2)}")
    print(f"\nContributions aux axes (%) :\n{contrib.round(1)}")
    graphique_eboulis(vp, "ACP des variables de segmentation", "acp_segmentation_eboulis.png")
    graphique_cercle(corr, vp, "ACP des variables de segmentation", "acp_segmentation_cercle.png")
    graphique_plan(coords, segments, vp, "acp_segmentation_plan.png")

    titre("ACP (b) : MONTANTS BRUTS PAR PRODUIT (effet taille)")
    vp_m, corr_m = acp_montants()
    print(vp_m.round(2).head(N_AXES).to_string(index=False))
    print(f"\nCorrélations variables / axes :\n{corr_m.round(2)}")
    graphique_cercle(corr_m, vp_m, "ACP des montants", "acp_montants_cercle.png")

    print(f"\nGraphiques enregistrés dans {config.DOSSIER_FIGURES}")


if __name__ == "__main__":
    main()


#ACP1:Le premier axe vaut 68% de l’information (valeur propre de 4,10). Le deuxième axe n’apporte que 11,2 %,
# conservé pour l’interprétation. Les deux premiers axes résument ainsi 79,6 % de l’info.
# Cette forte domination du premier axe s’explique par le cercle des corrélations : les six produits y sont tous fortement et positivement corrélés à l’axe 1, 
# avec des corrélations comprises entre 0,76 pour les produits gold et 0,92 pour la viande. 
# Le premier axe traduit donc un effet taille. Il ne distingue pas les clients selon ce qu’ils achètent, mais selon le volume global de leurs dépenses : 
# un client qui dépense beaucoup achète davantage de tous les produits.
#La projection des clients sur le premier plan  montre un résultat important pour notre problématique. 
# Les clients qui ont répondu à la dernière campagne sont concentrés à droite de l’axe 1
# alors que les deux groupes sont presque confondus sur l’axe 2 .
#  La réponse à la campagne est donc fortement liée au niveau global de dépenses du client, et très peu à la nature des produits achetés.
#

#acp2:Dans cette seconde analyse, chaque produit est exprimé en part des dépenses totales du client. 
# Les deux premiers axes résument 64,7 % de l’information (43,9 % et 20,8 %), 
# et nous avons conservé un troisième axe (12,8 %) pour mieux représenter les produits gold et les sucreries. 
# Le dernier axe a une variance nulle, ce qui est normal puisque les six parts somment toujours à 100 %.
#Le premier axe oppose le vin (corrn de 0,95) à l’ensemble des autres produits, 
# en particulier le poisson, les fruits et les sucreries (corrélations comprises entre −0,66 et −0,69). 
# Plus la part du vin dans le panier est élevée, plus celle des autres produits est faible : la part du vin est négativement corrélée à toutes les autres parts. 
# Le deuxième axe est presque entièrement construit par la viande (corrélation de 0,94, contribution de 70 %), qui s’oppose aux produits gold. 
# Le troisième axe est porté par les produits gold. La projection des clients prend la forme d’un triangle dont les trois sommets correspondent à trois façons d’acheter : 
# un panier dominé par le vin, 
# un panier dominé par la viande 
# et un panier diversifié en épicerie. 
# Contrairement à l’ACP 1, les clients ayant répondu à la campagne sont répartis sur l’ensemble du nuage, 
# avec seulement un léger décalage vers la viande et les produits gold.
#Un K-means sur les trois premiers axes fait apparaître quatre types de paniers.
# Le panier  vin et viande  (38 % de vin et 37 % de viande) est celui des clients qui dépensent le plus et obtient le meilleur taux de réponse (18 %).
# le  panier  vin et produits gold à petit budget (16 %). 
# le panier des Les clients qui achètent presque exclusivement du vin (71 % de leur panier) répondent peu (12 %). 
# Le panier diversifié des petits budgets a le taux de réponse le plus faible (10 %).
#


#conclusion: qui contacté ? c'est surtout le niveau de dépense plus que le panier
#Les clients à petit budget et au panier diversifié (10 % de réponse) et les petits dépensiers en général sont ceux à exclure en priorité. À l’inverse, les gros dépensiers concentrent l’essentiel des réponses : 
#Aucun groupe n’atteint le seuil de 27,3 % de réponses.
#au niveau des promo:Le vin s’achète avec la viande, et le panier vin et viande est à la fois celui qui dépense le plus et celui qui répond le mieux (18 %).
#Le cas des amateurs de vin exclusifs est différent. Ils représentent environ un tiers des clients et leur panier est à 71 % composé de vin, mais ils répondent peu (12 %).
#donc les campagnes passé n'ont pas été utile pour eux
# on a l'effet sur les campagnes passé par sur une nouvelle offre donc ne pas forcement faire une offre pour réglé le problème
#plutot faire un essaie d'une nouvelle offre et après l'étendre a tout ceux concerné et qui sont suceptible de répondre possitivement a l'offre.
