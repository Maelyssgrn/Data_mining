"""ACP des habitudes d'achat : avec quels produits le vin est-il acheté ?

    ACP 1 : montants dépensés par produit -> fait apparaître l'effet taille
    ACP 2 : composition du panier (part de chaque produit)
    Suite : types de paniers (K-means sur les axes de l'ACP 2) et lien avec la campagne
"""

import sqlite3

import matplotlib
matplotlib.use("Agg")  # enregistre les figures sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from backend import config, data

config.DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)

PRODUITS = {"MntWines": "Vin", "MntMeatProducts": "Viande", "MntFishProducts": "Poisson",
            "MntFruits": "Fruits", "MntSweetProducts": "Sucreries", "MntGoldProds": "Or"}
NOMS = list(PRODUITS.values())



# Fonctions communes aux deux ACP
def titre(texte):
    print(f"\n{'=' * 70}\n{texte}\n{'=' * 70}")


def tableau_valeurs_propres(acp):
    return pd.DataFrame({
        "axe": range(1, len(acp.explained_variance_) + 1),
        "valeur_propre": acp.explained_variance_.round(2),
        "variance_%": (100 * acp.explained_variance_ratio_).round(1),
        "cumulee_%": (100 * acp.explained_variance_ratio_.cumsum()).round(1),
    }).to_string(index=False)


def indicateurs_variables(acp, noms, n_axes):
    """Corrélations, cos2 et contributions des variables sur les n premiers axes."""
    colonnes = [f"Axe {i + 1}" for i in range(n_axes)]
    corr = pd.DataFrame(acp.components_[:n_axes].T * np.sqrt(acp.explained_variance_[:n_axes]),
                        index=noms, columns=colonnes)
    cos2 = corr ** 2
    contrib = 100 * pd.DataFrame(acp.components_[:n_axes].T ** 2, index=noms, columns=colonnes)
    return corr, cos2, contrib


def graphique_eboulis(acp, nom_acp, fichier):
    var = 100 * acp.explained_variance_ratio_
    axes = range(1, len(var) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(axes, var, color="#4C72B0", label="par axe")
    ax.plot(axes, var.cumsum(), "o-", color="#C44E52", label="cumulée")
    ax.set(xlabel="Axe", ylabel="% de variance", title=f"{nom_acp} : variance expliquée")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.DOSSIER_FIGURES / fichier, dpi=120)
    plt.close(fig)


def graphique_cercle(corr, acp, nom_acp, fichier):
    """Cercle des corrélations sur les axes 1 et 2 (le vin en rouge)."""
    var = 100 * acp.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="grey"))
    for nom, (x, y) in corr[["Axe 1", "Axe 2"]].iterrows():
        couleur = "#C44E52" if nom == "Vin" else "#4C72B0"
        ax.arrow(0, 0, x, y, head_width=0.03, color=couleur, length_includes_head=True)
        ax.text(x * 1.12, y * 1.12, nom, ha="center", va="center", color=couleur)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal",
           xlabel=f"Axe 1 ({var[0]:.1f} %)", ylabel=f"Axe 2 ({var[1]:.1f} %)",
           title=f"{nom_acp} : cercle des corrélations")
    fig.tight_layout()
    fig.savefig(config.DOSSIER_FIGURES / fichier, dpi=120)
    plt.close(fig)


def graphique_individus(coords, clients, acp, nom_acp, fichier):
    """Clients sur le plan 1-2, colorés par la réponse à la campagne."""
    var = 100 * acp.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 5))
    for valeur, couleur, label in [(0, "#BBBBBB", "N'a pas répondu"), (1, "#C44E52", "A répondu")]:
        m = (clients["Response"] == valeur).to_numpy()
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6, color=couleur, label=label)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set(xlabel=f"Axe 1 ({var[0]:.1f} %)", ylabel=f"Axe 2 ({var[1]:.1f} %)",
           title=f"{nom_acp} : projection des clients")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(config.DOSSIER_FIGURES / fichier, dpi=120)
    plt.close(fig)


def analyser_acp(donnees, clients, nom_acp, prefixe, n_axes):
    """Toute l'analyse d'une ACP, dans l'ordre. Renvoie l'ACP et les coordonnées des clients."""
    X = StandardScaler().fit_transform(donnees)
    acp = PCA(random_state=config.SEED).fit(X)
    coords = acp.transform(X)

    print("\n--- 1. Valeurs propres (critère de Kaiser : on garde > 1) ---")
    print(tableau_valeurs_propres(acp))
    graphique_eboulis(acp, nom_acp, f"{prefixe}_eboulis.png")

    corr, cos2, contrib = indicateurs_variables(acp, list(donnees.columns), n_axes)
    print(f"\n--- 2. Corrélations variables / axes ({n_axes} axes retenus) ---")
    print(corr.round(2))
    print("\n--- 3. Qualité de représentation (cos2) ---")
    print(cos2.round(2))
    print(f"cos2 cumulé sur {n_axes} axes : {cos2.sum(axis=1).round(2).to_dict()}")
    print("\n--- 4. Contributions aux axes (%) ---")
    print(contrib.round(1))
    graphique_cercle(corr, acp, nom_acp, f"{prefixe}_cercle.png")

    print("\n--- 5. Corrélations entre produits (vin en premier) ---")
    print(donnees.corr()["Vin"].drop("Vin").round(2).sort_values(ascending=False).to_string())

    print("\n--- 6. Position moyenne des clients : répondants vs non-répondants ---")
    positions = pd.DataFrame(coords[:, :n_axes], columns=[f"F{i + 1}" for i in range(n_axes)])
    print(positions.groupby(clients["Response"].to_numpy()).mean().round(2)
          .rename_axis("Response"))
    graphique_individus(coords, clients, acp, nom_acp, f"{prefixe}_individus.png")

    return acp, coords[:, :n_axes]



clients = data.lire_analyse()  # table d'analyse nettoyée (recréée si besoin)
clients = clients[clients["Depense_totale"] > 0].reset_index(drop=True)  # pas de panier vide
print(f"{len(clients)} clients analysés")


# ACP 1 : montants dépensés

titre("ACP 1 : MONTANTS DÉPENSÉS PAR PRODUIT (log)")
montants = np.log1p(clients[list(PRODUITS)]).set_axis(NOMS, axis=1)  # log : montants asymétriques
acp1, coords1 = analyser_acp(montants, clients, "ACP des montants", "acp1_montants", n_axes=2)
# Lecture : tous les produits du même côté de l'axe 1 = effet taille
# (un client qui dépense beaucoup achète beaucoup de tout). D'où l'ACP 2.



# ACP 2 : composition du panier
titre("ACP 2 : COMPOSITION DU PANIER (part de chaque produit)")
parts = clients[list(PRODUITS)].div(clients["Depense_totale"], axis=0).set_axis(NOMS, axis=1)
acp2, coords2 = analyser_acp(parts, clients, "ACP du panier", "acp2_panier", n_axes=3)
# Rq : le dernier axe vaut 0 % car les parts somment à 100 %


# --- Panier selon la place du vin ---
print("\n--- 7. Composition moyenne du panier (%) selon la place du vin ---")
clients["Profil_vin"] = pd.qcut(parts["Vin"], 3, labels=["Peu de vin", "Moyen", "Beaucoup de vin"])
profil_vin = (100 * parts.groupby(clients["Profil_vin"], observed=True).mean()).round(1)
profil_vin["revenu_med"] = clients.groupby("Profil_vin", observed=True)["Income"].median()
profil_vin["taux_reponse_%"] = (100 * clients.groupby("Profil_vin", observed=True)["Response"].mean()).round(1)
print(profil_vin.to_string())

# --- Types de paniers : K-means sur les 3 axes de l'ACP 2 ---
print("\n--- 8. Types de paniers : choix du nombre de groupes (silhouette) ---")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=config.SEED).fit(coords2)
    print(f"k = {k} : {silhouette_score(coords2, km.labels_):.3f}")

K_PANIERS = 4
clients["Panier"] = KMeans(n_clusters=K_PANIERS, n_init=20, random_state=config.SEED).fit_predict(coords2)

types = (100 * parts.groupby(clients["Panier"]).mean()).round(1)
types["clients"] = clients["Panier"].value_counts().sort_index()
types["depense_med"] = clients.groupby("Panier")["Depense_totale"].median()
types["revenu_med"] = clients.groupby("Panier")["Income"].median()
types["taux_reponse_%"] = (100 * clients.groupby("Panier")["Response"].mean()).round(1)
print(f"\n--- 9. Les {K_PANIERS} types de paniers ---")
print(types.to_string())

# Graphique : composition de chaque type de panier
fig, ax = plt.subplots(figsize=(8, 4))
types[NOMS].plot.barh(stacked=True, ax=ax, colormap="tab10", width=0.7)
for i, taux in enumerate(types["taux_reponse_%"]):
    ax.text(101, i, f"réponse {taux:.0f} %", va="center", fontsize=9)
ax.set(xlim=(0, 118), xlabel="% du panier", ylabel="Type de panier",
       title="Composition des types de paniers")
ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.18))
fig.tight_layout()
fig.savefig(config.DOSSIER_FIGURES / "acp2_types_paniers.png", dpi=120)
plt.close(fig)

# Sauvegarde du type de panier de chaque client
with sqlite3.connect(config.CHEMIN_DB) as con:
    clients[["ID", "Panier"]].to_sql("paniers", con, if_exists="replace", index=False)
print(f"\nTable 'paniers' (ID, Panier) écrite dans {config.CHEMIN_DB}")
print(f"Graphiques enregistrés dans {config.DOSSIER_FIGURES}")

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
#la segmentation montre que ne contacter que le segment des clients aisés sans enfants ramènerait la perte de −2 814 à −284
#tout en conservant près de la moitié des répondants. Cela ne suffit pas encore pour être rentable, puisqu’aucun groupe n’atteint le seuil de 27,3 % de réponses.
#au niveau des promo:Le vin s’achète avec la viande, et le panier vin et viande est à la fois celui qui dépense le plus et celui qui répond le mieux (18 %).
#Le cas des amateurs de vin exclusifs est différent. Ils représentent environ un tiers des clients et leur panier est à 71 % composé de vin, mais ils répondent peu (12 %).
#donc les campagnes passé n'ont pas été utile pour eux
# on a l'effet sur les campagnes passé par sur une nouvelle offre donc ne pas forcement faire une offre pour réglé le problème
#plutot faire un essaie d'une nouvelle offre et après l'étendre a tout ceux concerné et qui sont suceptible de répondre possitivement a l'offre.