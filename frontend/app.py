"""Démonstrateur Streamlit : affichage et saisie uniquement.

Toute la logique est dans le backend ; ce fichier n'importe que backend.service.
Lancement (depuis la racine du dépôt) : python -m streamlit run frontend/app.py
"""

import colorsys
import math
import re

import altair as alt
import pandas as pd
import streamlit as st

from backend import service

# Couleurs des accents (le thème général est dans .streamlit/config.toml),
# les mêmes dans tous les écrans et graphiques
CONTACTER = "#A8D5BA"          # vert sauge : contacter, gain positif
NE_PAS_CONTACTER = "#E2A58C"   # terracotta : ne pas contacter, gain négatif
SEUIL = "#2E2A35"              # ligne du seuil de rentabilité
TEXTE = "#2E2A35"
GRILLE = "#ECE8F3"
COULEURS_PERSONAS = {
    "Familles modestes au panier varié": "#A3C1D9",    # bleu
    "Familles amatrices de vin": "#F7C6D0",            # rose
    "Aisés, peu d'enfants, vin et viande": "#FBEFB5",  # jaune pâle
}
SITUATIONS = {"Married": "Marié(e)", "Together": "En couple", "Single": "Célibataire",
              "Divorced": "Divorcé(e)", "Widow": "Veuf ou veuve", "Autre": "Autre"}
# Équité : taux de contact par le modèle contre taux de réponse réel du groupe
COULEURS_MESURES = {"Contactés par le modèle": "#8E7CC3", "Réponse réelle du groupe": "#B8B4C0"}
BARRES = "#C9BEE8"  # lavande : barres sans signification de couleur (importances)
# Noms des variables du modèle en langage métier
LIBELLES = {
    "Age": "Âge", "Income": "Revenu", "Kidhome": "Jeunes enfants", "Teenhome": "Adolescents",
    "Recency": "Jours depuis le dernier achat", "Anciennete_jours": "Ancienneté du client",
    "Complain": "Réclamation récente", "MntWines": "Dépense en vin",
    "MntFruits": "Dépense en fruits", "MntMeatProducts": "Dépense en viande",
    "MntFishProducts": "Dépense en poisson", "MntSweetProducts": "Dépense en sucreries",
    "MntGoldProds": "Dépense en or", "NumDealsPurchases": "Achats en promotion",
    "NumWebPurchases": "Achats sur le web", "NumCatalogPurchases": "Achats par catalogue",
    "NumStorePurchases": "Achats en magasin", "NumWebVisitsMonth": "Visites web par mois",
    "AcceptedCmp1": "Campagne 1 acceptée", "AcceptedCmp2": "Campagne 2 acceptée",
    "AcceptedCmp3": "Campagne 3 acceptée", "AcceptedCmp4": "Campagne 4 acceptée",
    "AcceptedCmp5": "Campagne 5 acceptée", "Education": "Niveau d'études",
    "Marital_Status": "Situation familiale",
}

st.set_page_config(page_title="Ciblage de campagne", layout="wide")
st.markdown("""
<style>
.carte {border-radius: 18px; padding: 1.3rem 1.6rem; color: #2E2A35; margin-bottom: 0.8rem;}
.carte .verdict {font-size: 2.1rem; font-weight: 700; margin: 0 0 0.3rem 0;}
.carte p {margin: 0.2rem 0; font-size: 1.05rem;}
/* Cartes des personas côte à côte : une grille CSS donne la même hauteur aux trois */
.personas {display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 0.8rem;}
.personas .carte {margin-bottom: 0;}
.pastille {display: inline-block; padding: 0.35rem 0.95rem; border-radius: 999px;
           font-weight: 600; color: #2E2A35;}
</style>
""", unsafe_allow_html=True)


def contour(couleur, facteur=0.55):
    """Même teinte, plus foncée : contour des marques pâles sur le fond clair."""
    r, g, b = (int(couleur[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return "#" + "".join(f"{round(c * 255):02X}" for c in colorsys.hls_to_rgb(h, l * facteur, s))


# Mise en forme : nombres à la française, espaces insécables (pas de « 27,3 » en fin
# de ligne et « % » au début de la suivante, ni « : » en début de ligne)
INSECABLE = " "   # avant %, UM, :, ;, ? et !
FINE = " "        # séparateur des milliers


def nombre(x, decimales=0):
    return f"{x:,.{decimales}f}".replace(",", FINE).replace(".", ",")


def pourcent(p, decimales=1):
    return f"{nombre(100 * p, decimales)}{INSECABLE}%"


def um(x, decimales=2):
    signe = "+" if x > 0 else "−" if x < 0 else ""
    return f"{signe}{nombre(abs(x), decimales)}{INSECABLE}UM"


def insecable(texte):
    """Typographie française de tous les textes affichés : espace insécable avant % et UM
    (après un nombre) et avant : ; ? ! (aucun signe ne se retrouve en début de ligne)."""
    texte = re.sub(r"(\d) (%|UM)", rf"\1{INSECABLE}\2", texte)
    return re.sub(r" ([:;?!])", rf"{INSECABLE}\1", texte)


def style(graphique):
    """Police assez grande pour une démo projetée, texte foncé, axes titrés."""
    return (graphique.configure_axis(labelFontSize=16, titleFontSize=16, labelColor=TEXTE,
                                     titleColor=TEXTE, gridColor=GRILLE)
            # labelLimit=0 : noms des légendes jamais tronqués
            .configure_legend(labelFontSize=16, titleFontSize=16, labelColor=TEXTE, labelLimit=0)
            .configure_view(strokeWidth=0))


# Chargements mis en cache (le backend ne réentraîne rien)

@st.cache_data
def charger_clients():
    return service.clients_test()


@st.cache_data
def charger_personas():
    return service.personas()


@st.cache_data
def charger_plan_acp():
    return service.plan_acp()


@st.cache_data
def charger_figures():
    return service.figures()


@st.cache_data
def charger_resultats():
    return service.resultats_modele()


def echelle_personas(contours=False):
    """Couleurs des personas (ou leurs contours, plus foncés) pour les graphiques."""
    couleurs = COULEURS_PERSONAS.values()
    return alt.Scale(domain=list(COULEURS_PERSONAS),
                     range=[contour(c) for c in couleurs] if contours else list(couleurs))


def exiger_modele():
    """Arrête l'écran avec un message clair si le modèle n'a pas été entraîné."""
    if not service.modele_disponible():
        st.error(insecable("Le modèle n'est pas encore entraîné. Lancer, depuis la racine du dépôt :"))
        st.code("python -m backend.train", language="bash")
        st.stop()


# Écran Décision

def jauge(p, seuil, couleur):
    """Barre horizontale de la probabilité, avec le seuil de rentabilité en pointillés.
    Positions verticales relatives (de 0 à 1, axe masqué) : la jauge garde ses proportions
    quelle que soit la taille donnée par Streamlit, et l'axe reste sous la barre."""
    axe = alt.Axis(title="Probabilité de réponse", values=[0, 0.25, 0.5, 0.75, 1], grid=False,
                   ticks=True, labelExpr=f"format(datum.value * 100, '.0f') + '{INSECABLE}%'")
    x = alt.X("x:Q", scale=alt.Scale(domain=[0, 1]), axis=axe)
    y = alt.Y("y:Q", scale=alt.Scale(domain=[0, 1]), axis=None)
    # Barre de 0 à p (sans fond derrière), de y = 0,1 à 0,6
    barre = alt.Chart(pd.DataFrame({"x": [p], "y": [0.1], "y2": [0.6]})).mark_bar(
        color=couleur, stroke=contour(couleur), strokeWidth=1.5, cornerRadius=10).encode(
        x=x, x2=alt.datum(0), y=y, y2="y2:Q")
    # Seuil dessiné après la barre (donc par-dessus), sur toute sa hauteur et au-delà
    trait = alt.Chart(pd.DataFrame({"x": [seuil], "y": [0.02], "y2": [0.7]})).mark_rule(
        color=SEUIL, strokeDash=[6, 3], strokeWidth=3).encode(x=x, y=y, y2="y2:Q")
    texte = alt.Chart(pd.DataFrame({"x": [seuil], "y": [0.75], "t": [f"seuil {pourcent(seuil)}"]})
                      ).mark_text(align="left", baseline="bottom", dx=6, fontSize=15,
                                  color=SEUIL).encode(x=x, y=y, text="t:N")
    return style((barre + trait + texte).properties(height=110))


def ecran_decision(parametres):
    st.title(insecable("Décision : faut-il contacter ce client ?"))
    exiger_modele()
    seuil = parametres["seuil"]

    clients = charger_clients().sort_values("probabilite", ascending=False)
    filtre = st.radio("Afficher", ["Tous les clients", "À contacter", "À ne pas contacter"],
                      horizontal=True)
    if filtre == "À contacter":
        clients = clients[clients["contacter"]]
    elif filtre == "À ne pas contacter":
        clients = clients[~clients["contacter"]]
    id_client = st.selectbox(
        f"Client ({len(clients)} clients, du plus probable au moins probable)", clients.index,
        format_func=lambda i: f"Client {i}  ·  probabilité {pourcent(clients.loc[i, 'probabilite'])}")
    st.caption(insecable("Seuls les 402 clients du jeu de test sont proposés : le modèle n'a pas "
                         "vu leur réponse pendant l'entraînement, sa probabilité est donc honnête."))

    d = service.decision_client(id_client)
    p, gain = d["probabilite"], d["gain_attendu"]
    couleur = CONTACTER if d["contacter"] else NE_PAS_CONTACTER
    verdict = "Contacter" if d["contacter"] else "Ne pas contacter"
    position = "au-dessus" if d["contacter"] else "en dessous"

    gauche, droite = st.columns([3, 2], gap="large")
    with gauche:
        st.markdown(insecable(f"""
        <div class="carte" style="background:{couleur}">
          <div class="verdict">{verdict}</div>
          <p>Probabilité de réponse : <b>{pourcent(p)}</b>, {position} du seuil de
             rentabilité ({pourcent(seuil)}).</p>
          <p>Gain attendu d'un contact : {nombre(p, 3)} × {parametres['revenu_reponse']}
             − {parametres['cout_contact']} = <b>{um(gain)}</b></p>
        </div>"""), unsafe_allow_html=True)
        st.altair_chart(jauge(p, seuil, couleur), width="stretch")
    with droite:
        st.markdown("**Persona du client**")
        actions = charger_personas().set_index("nom")["action"]
        st.markdown(f'<span class="pastille" style="background:'
                    f'{COULEURS_PERSONAS.get(d["persona"], GRILLE)}">{d["persona"]}</span>',
                    unsafe_allow_html=True)
        st.caption(insecable(f"Pour l'ensemble du segment : {actions[d['persona']]}"))
        st.metric("Probabilité de réponse", pourcent(p), border=True)
        st.metric("Gain attendu d'un contact", um(gain), border=True)

    st.subheader("Profil du client")
    profil = [("Âge", f"{d['Age']} ans"),
              ("Revenu", "inconnu" if pd.isna(d["Income"]) else um(d["Income"], 0).lstrip("+")),
              ("Enfants au foyer", d["Enfants"]),
              ("Situation familiale", SITUATIONS.get(d["Marital_Status"], d["Marital_Status"])),
              ("Campagnes déjà acceptées", d["Nb_campagnes_acceptees"]),
              ("Dernier achat", f"il y a {d['Recency']} jours"),
              ("Dépense totale (2 ans)", um(d["Depense_totale"], 0).lstrip("+"))]
    colonnes = st.columns(4)
    for i, (libelle, valeur) in enumerate(profil):
        colonnes[i % 4].metric(libelle, valeur, border=True)

    with st.expander("Vérification (jeu de test)"):
        reponse = "a répondu" if d["reponse_reelle"] == 1 else "n'a pas répondu"
        st.write(f"À la dernière campagne, ce client **{reponse}**. Cette information sert "
                 "seulement à vérifier le modèle en démonstration : pour la prochaine "
                 "campagne, la réponse n'est pas connue à l'avance.")


# Écran Exploration

def carte_strategie(colonne, titre, bilan):
    colonne.metric(insecable(titre), um(bilan["gain_net"], 0), border=True)
    colonne.caption(f"{nombre(bilan['n_contactes'])} clients contactés · "
                    f"{pourcent(bilan['taux_pct'] / 100)} de réponse")


def carte_persona(persona):
    """HTML d'une carte : une information par ligne, toujours dans le même ordre."""
    couleur = COULEURS_PERSONAS.get(persona["nom"], GRILLE)
    return insecable(f"""
    <div class="carte" style="background:{couleur}">
      <div style="font-size:1.35rem; font-weight:700; margin-bottom:0.4rem">{persona['nom']}</div>
      <p><b>{nombre(persona['n_clients'])}</b> clients</p>
      <p>Revenu médian : <b>{um(persona['revenu_median'], 0).lstrip('+')}</b></p>
      <p>Dépense médiane : {um(persona['depense_mediane'], 0).lstrip('+')}</p>
      <p>Avec enfants : {pourcent(persona['avec_enfants_%'] / 100, 0)}</p>
      <p>Produits principaux : {persona['produits_principaux']}</p>
      <p>Canaux principaux : {persona['canaux_principaux']}</p>
      <p style="font-size:1.25rem; margin-top:0.5rem">Taux de réponse :
         <b>{pourcent(persona['taux_reponse_%'] / 100)}</b></p>
      <p><i>{persona['action']}</i></p>
    </div>""")


def graphique_reponse_personas(table, seuil):
    """Taux de réponse de chaque persona, comparé au seuil de rentabilité."""
    donnees = table.assign(taux=table["taux_reponse_%"] / 100)
    # Axe jusqu'à 50 % : de la place à droite du seuil pour son libellé (aucune barre ne
    # dépasse le seuil, le libellé ne chevauche donc rien)
    x = alt.X("taux:Q", title="Taux de réponse à la dernière campagne",
              scale=alt.Scale(domain=[0, 0.5]),
              axis=alt.Axis(values=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            labelExpr=f"format(datum.value * 100, '.0f') + '{INSECABLE}%'"))
    barres = alt.Chart(donnees).mark_bar(cornerRadiusEnd=8, size=46, strokeWidth=1.5).encode(
        y=alt.Y("nom:N", title=None, sort=None, axis=alt.Axis(labelLimit=320)), x=x,
        color=alt.Color("nom:N", scale=echelle_personas(), legend=None),
        stroke=alt.Stroke("nom:N", scale=echelle_personas(contours=True), legend=None),
        tooltip=[alt.Tooltip("nom:N", title="Persona"),
                 alt.Tooltip("taux_reponse_%:Q", title="Taux de réponse (%)")])
    seuil_df = pd.DataFrame({"taux": [seuil], "t": [f"seuil de rentabilité {pourcent(seuil)}"]})
    trait = alt.Chart(seuil_df).mark_rule(color=SEUIL, strokeDash=[6, 3], strokeWidth=2.5).encode(x=x)
    texte = alt.Chart(seuil_df).mark_text(align="left", baseline="top", dx=8, fontSize=15,
                                          color=SEUIL).encode(x=x, y=alt.value(6), text="t:N")
    return style((barres + trait + texte).properties(height=260))


def graphique_plan(coords, variance):
    """Clients sur les deux premiers axes de l'ACP, colorés par persona."""
    donnees = coords.reset_index()
    return style(alt.Chart(donnees).mark_circle(size=42, opacity=0.9, strokeWidth=0.8).encode(
        x=alt.X("Axe 1:Q", title=insecable(f"Axe 1 ({pourcent(variance[0] / 100)}) : aisés, vin, "
                                           "catalogue"),
                axis=alt.Axis(format=".0f", tickMinStep=1)),
        y=alt.Y("Axe 2:Q", title=insecable(f"Axe 2 ({pourcent(variance[1] / 100)}) : familles, web"),
                axis=alt.Axis(format=".0f", tickMinStep=1)),
        color=alt.Color("persona:N", scale=echelle_personas(), title="Persona",
                        legend=alt.Legend(orient="bottom", direction="vertical")),
        # Contour plus foncé de la même teinte : le jaune pâle reste visible sur le fond clair
        stroke=alt.Stroke("persona:N", scale=echelle_personas(contours=True), legend=None),
        tooltip=[alt.Tooltip("ID:Q", title="Client"), alt.Tooltip("persona:N", title="Persona")],
    ).properties(height=520))


def ecran_exploration(parametres):
    st.title(insecable("Exploration : quels profils de clients ?"))

    st.subheader(insecable("Le besoin : contacter tout le monde fait perdre de l'argent"))
    besoin = service.besoin_client()
    tous, baseline = besoin.loc["Tout le monde"], besoin.loc["Déjà accepté une campagne"]
    g, d = st.columns(2, gap="large")
    carte_strategie(g, "Contacter tout le monde", tous)
    carte_strategie(d, "Contacter ceux qui ont déjà accepté une campagne", baseline)
    # Chiffres calculés (besoin_client et seuil), pas écrits à la main
    st.markdown(insecable(
        f"Contacter tout le monde fait perdre {nombre(-tous['gain_net'])} UM : seuls "
        f"{pourcent(tous['taux_pct'] / 100)} des clients répondent, alors qu'il en faudrait au "
        f"moins {pourcent(parametres['seuil'])} pour couvrir le coût des contacts. En ne "
        "contactant que les clients qui ont déjà accepté une campagne, on gagne déjà "
        f"{nombre(baseline['gain_net'])} UM."))

    st.subheader("Trois personas")
    table = charger_personas()
    # Les trois cartes dans une même grille CSS : même hauteur, même si l'une est moins remplie
    st.markdown('<div class="personas">' + "".join(carte_persona(p) for _, p in table.iterrows())
                + "</div>", unsafe_allow_html=True)
    st.altair_chart(graphique_reponse_personas(table, parametres["seuil"]), width="stretch")
    st.caption(insecable("Aucun segment n'atteint le seuil : contacter un segment entier n'est "
                         "pas rentable. Il faut cibler client par client (écran Décision)."))

    st.subheader("Les clients sur le plan de l'ACP")
    coords, variance = charger_plan_acp()
    st.altair_chart(graphique_plan(coords, variance), width="stretch")
    st.caption(insecable("Sur ce graphique, les trois groupes se chevauchent : il n'y a pas de "
                         "frontière nette entre les profils, on passe progressivement de l'un à "
                         "l'autre. Les personas décrivent des tendances, pas des catégories "
                         "étanches."))

    with st.expander(insecable("Méthode : comment les segments ont été construits")):
        st.write(insecable("k-means sur 11 variables : revenu, enfants, part de chaque produit "
                           "dans la dépense du client et part de chaque canal dans ses achats. "
                           "La réponse à la campagne n'est pas utilisée pour construire les "
                           "segments."))
        figures = charger_figures()
        for i in range(0, len(figures), 2):
            for colonne, (titre, chemin, explication) in zip(st.columns(2, gap="large"),
                                                             figures[i:i + 2]):
                colonne.markdown(f"**{insecable(titre)}**")
                colonne.image(str(chemin), width="stretch")
                colonne.caption(insecable(explication))


# Écran Modèle

def graphique_gains(strategies):
    """Gain net de chaque stratégie sur le test : vert si positif, terracotta si négatif."""
    donnees = strategies.reset_index()
    donnees["signe"] = donnees["gain_net"].map(lambda g: "positif" if g > 0 else "négatif")
    # Marge de 20 % de chaque côté : les valeurs écrites au bout des barres restent visibles
    marge = 0.2 * (donnees["gain_net"].max() - donnees["gain_net"].min())
    base = alt.Chart(donnees).encode(
        y=alt.Y("strategie:N", title=None, sort=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("gain_net:Q", title="Gain net sur le test (UM)", scale=alt.Scale(
            domain=[min(0, donnees["gain_net"].min()) - marge, donnees["gain_net"].max() + marge])))
    signes = ["positif", "négatif"]
    barres = base.mark_bar(cornerRadius=6, size=40, strokeWidth=1.5).encode(
        color=alt.Color("signe:N", legend=None, scale=alt.Scale(
            domain=signes, range=[CONTACTER, NE_PAS_CONTACTER])),
        stroke=alt.Stroke("signe:N", legend=None, scale=alt.Scale(
            domain=signes, range=[contour(CONTACTER), contour(NE_PAS_CONTACTER)])),
        tooltip=[alt.Tooltip("strategie:N", title="Stratégie"),
                 alt.Tooltip("gain_net:Q", title="Gain net (UM)"),
                 alt.Tooltip("n_contactes:Q", title="Contactés")])
    textes = base.mark_text(align="left", dx=6, fontSize=15, color=TEXTE).encode(
        text=alt.Text("gain_net:Q", format="+d"))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TEXTE).encode(x="x:Q")
    return style((barres + textes + zero).properties(height=280))


def graphique_importance(importance):
    donnees = importance.assign(libelle=importance["variable"].map(lambda v: LIBELLES.get(v, v)))
    y = alt.Y("libelle:N", title=None, sort=None, axis=alt.Axis(labelLimit=320))
    x = alt.X("baisse_ap:Q", title="Baisse de l'Average Precision quand on mélange la variable")
    barres = alt.Chart(donnees).mark_bar(color=BARRES, cornerRadiusEnd=6).encode(
        y=y, x=x, tooltip=[alt.Tooltip("libelle:N", title="Variable"),
                           alt.Tooltip("baisse_ap:Q", title="Baisse d'AP", format=".3f"),
                           alt.Tooltip("ecart_type:Q", title="Écart-type", format=".3f")])
    erreurs = alt.Chart(donnees).mark_errorbar(color=TEXTE, ticks=True).encode(
        y=y, x=x, xError="ecart_type:Q")
    return style((barres + erreurs).properties(height=420))


def graphique_equite(equite, variable):
    """Pour chaque groupe : taux de contact par le modèle et taux de réponse réel."""
    lignes = equite[equite["variable"] == variable]
    ordre = [SITUATIONS.get(g, g) for g in lignes["groupe"]]
    donnees = lignes.assign(groupe=ordre).melt(
        id_vars=["groupe", "n_clients"], var_name="mesure", value_name="taux",
        value_vars=["taux_contact_pct", "taux_reponse_groupe_pct"])
    donnees["mesure"] = donnees["mesure"].map({"taux_contact_pct": "Contactés par le modèle",
                                               "taux_reponse_groupe_pct": "Réponse réelle du groupe"})
    return style(alt.Chart(donnees).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("groupe:N", title=variable, sort=ordre, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("mesure:N", sort=list(COULEURS_MESURES)),
        y=alt.Y("taux:Q", title="% des clients du groupe"),
        color=alt.Color("mesure:N", title=None, sort=list(COULEURS_MESURES),
                        scale=alt.Scale(domain=list(COULEURS_MESURES),
                                        range=list(COULEURS_MESURES.values())),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("groupe:N", title="Groupe"), alt.Tooltip("mesure:N", title="Mesure"),
                 alt.Tooltip("taux:Q", title="%"), alt.Tooltip("n_clients:Q", title="Clients")],
    ).properties(height=340))


def tableau(df, colonnes):
    """DataFrame avec des noms de colonnes en français, sans l'index."""
    st.dataframe(df.reset_index()[list(colonnes)].rename(columns=colonnes), hide_index=True)


# Hyperparamètres affichés pour chaque modèle, avec leur rôle en une phrase simple
# (les valeurs testées et retenues viennent du backend)
ROLES = {
    "Régression logistique": [
        ("C", "Force de la régularisation : plus C est petit, plus les coefficients sont "
              "tirés vers 0, donc plus le modèle est prudent."),
        ("l1_ratio", "Type de pénalité : L2 réduit tous les coefficients sans en annuler "
                     "aucun."),
        ("solver", "Algorithme qui calcule les coefficients (lbfgs, celui par défaut de "
                   "scikit-learn)."),
    ],
    "Random Forest": [
        ("n_estimators", "Nombre d'arbres. Fixé à 500 : plus d'arbres ne fait pas "
                         "surapprendre, cela stabilise seulement les probabilités ; il est "
                         "donc inutile de le régler."),
        ("max_depth", "Profondeur maximale d'un arbre : nombre de questions posées à la "
                      "suite sur un client."),
        ("min_samples_leaf", "Nombre minimal de clients dans une feuille : évite les règles "
                             "tirées de très peu de clients."),
        ("max_features", "Nombre de variables tirées au hasard à chaque coupure : rend les "
                         "arbres différents entre eux."),
        ("oob_score", "Chaque arbre est évalué sur les clients qu'il n'a pas tirés (hors du "
                      "sac) : une estimation de l'erreur sans toucher au test."),
        ("random_state", "Graine du hasard : les résultats sont reproductibles."),
    ],
}


def valeur(v, nom=""):
    """Valeur d'un hyperparamètre en français."""
    if nom == "l1_ratio":
        return "L2" if v == 0 else f"mélange L1/L2 ({v})"
    if v is None:
        return "aucune limite"
    if isinstance(v, bool):
        return "oui" if v else "non"
    if v == "sqrt":
        return "racine carrée du nombre de variables"
    if nom == "max_features":
        return f"{pourcent(v, 0)} des variables"
    return str(v).replace(".", ",")


def tableau_hyperparametres(nom, construits, grille, retenus):
    """Une ligne par hyperparamètre : rôle, valeurs testées, valeur retenue."""
    lignes = [{"Hyperparamètre": "pénalité (l1_ratio)" if p == "l1_ratio" else p,
               "Ce qu'il fait": insecable(role),
               # Séparateur « ; » : la virgule est déjà la virgule décimale (0,01 ; 0,03)
               "Valeurs testées": insecable(" ; ".join(valeur(v, p) for v in grille[p]))
               if p in grille
               else "fixé, non réglé",
               "Valeur retenue": valeur(retenus.get(p, construits[p]), p)}
              for p, role in ROLES[nom]]
    # st.table : le texte des cellules passe à la ligne au lieu d'être coupé
    st.table(pd.DataFrame(lignes).set_index("Hyperparamètre"))


def ecran_modele(parametres):
    st.title(insecable("Modèle : que vaut le ciblage ?"))
    exiger_modele()
    r = charger_resultats()
    strategies, retenu = r["strategies"], r["retenu"]

    st.subheader("Sur des clients jamais vus pendant l'entraînement")
    st.write(insecable(f"Jeu de test : {nombre(r['n_test'])} clients, dont "
                       f"{r['n_repondants_test']} répondants. Règle : contacter si la "
                       f"probabilité dépasse {pourcent(parametres['seuil'])}."))
    colonnes = st.columns(3, gap="large")
    for colonne, (titre, ligne) in zip(colonnes, [
            (f"Modèle retenu ({retenu})", retenu),
            ("Baseline : déjà accepté une campagne", "Déjà accepté une campagne"),
            ("Contacter tout le monde", "Tout le monde")]):
        carte_strategie(colonne, titre, strategies.loc[ligne])

    st.altair_chart(graphique_gains(strategies), width="stretch")
    tableau(strategies, {"strategie": "Stratégie", "n_contactes": "Contactés",
                         "n_reponses": "Réponses", "taux_pct": "Taux de réponse (%)",
                         "gain_net": "Gain net (UM)"})

    st.subheader(insecable("Choix du modèle : validation croisée sur l'entraînement"))
    tableau(r["cv"].round({"ap_moyenne": 3, "ap_ecart_type": 3, "gain_moyen": 1,
                           "gain_ecart_type": 1}),
            {"strategie": "Stratégie", "gain_moyen": "Gain net moyen par pli (UM)",
             "gain_ecart_type": "Écart-type du gain", "ap_moyenne": "Average Precision",
             "ap_ecart_type": "Écart-type de l'AP"})
    st.info(insecable(f"Modèle retenu : **{retenu}**. {r['raison']}."))

    st.subheader("Variables importantes")
    st.altair_chart(graphique_importance(r["importance"]), width="stretch")
    st.caption(insecable("Mesurée sur le test : plus la baisse est grande, plus le modèle "
                         "s'appuie sur la variable pour classer les clients. Une importance "
                         "n'est pas une cause, et des variables liées entre elles se partagent "
                         "l'importance."))

    st.subheader(insecable("Équité : âge et situation familiale"))
    equite = r["equite"]
    g, d = st.columns(2, gap="large")
    g.altair_chart(graphique_equite(equite, "Situation familiale"), width="stretch")
    d.altair_chart(graphique_equite(equite, "Tranche d'âge"), width="stretch")
    petits = equite[equite["n_contactes"] < 15]
    st.write(insecable(
        f"Chaque groupe est contacté à peu près en proportion de sa réponse réelle, et "
        f"parmi les contactés le taux de réponse est d'au moins "
        f"**{pourcent(equite['taux_reponse_contactes_pct'].min() / 100)}** dans tous les "
        f"groupes. Petits groupes, à lire avec prudence : "
        + ", ".join(f"{SITUATIONS.get(g, g)} ({n} contactés)"
                    for g, n in zip(petits["groupe"], petits["n_contactes"])) + "."))
    with st.expander("Tableau d'équité"):
        tableau(equite.assign(groupe=equite["groupe"].map(lambda g: SITUATIONS.get(g, g))),
                {"variable": "Variable", "groupe": "Groupe", "n_clients": "Clients",
                 "taux_reponse_groupe_pct": "Réponse réelle (%)",
                 "taux_contact_pct": "Contactés (%)", "n_contactes": "Contactés",
                 "taux_reponse_contactes_pct": "Réponse parmi les contactés (%)"})

    st.subheader("Limites")
    st.markdown("\n".join(f"- {insecable(limite)}" for limite in service.limites()))

    with st.expander("Détails techniques"):
        construits, grilles, n_plis = service.hyperparametres()
        for nom in ROLES:
            st.markdown(f"**{nom}**")
            tableau_hyperparametres(nom, construits[nom], grilles[nom], r["reglages"][nom])
        # Nombre de combinaisons d'une grille : produit des nombres de valeurs testées
        combinaisons = {nom: math.prod(len(v) for v in g.values()) for nom, g in grilles.items()}
        total = sum(combinaisons.values())
        st.markdown(insecable(
            f"- **Critère de réglage** : gain net moyen en validation croisée sur "
            f"l'entraînement ({n_plis} plis stratifiés), en contactant les clients dont la "
            f"probabilité dépasse {pourcent(parametres['seuil'])}. L'Average Precision est "
            "calculée à côté, pour information.\n"
            f"- **Combinaisons testées** : "
            + " ; ".join(f"{nom} : {n}" for nom, n in combinaisons.items())
            + f". Soit {total} combinaisons, chacune entraînée sur les {n_plis} plis.\n"
            "- **Règle de choix entre les deux modèles** : on prend le modèle au meilleur gain "
            "net moyen, sauf si son avance sur la régression logistique est plus petite qu'un "
            "écart-type du gain ; dans ce cas, on garde la régression logistique, plus simple "
            f"à expliquer. {r['raison']}."))

        st.write(insecable("Random Forest : score hors du sac (OOB, sur l'entraînement) "
                           "contre test."))
        # Nombres à la française, arrondis (0,612 ; 57,2)
        oob = r["oob"].assign(ap=r["oob"]["ap"].map(lambda x: nombre(x, 3)),
                              gain_pour_100=r["oob"]["gain_pour_100"].map(lambda x: nombre(x, 1)),
                              n_clients=r["oob"]["n_clients"].map(nombre))
        tableau(oob, {"jeu": "Jeu", "ap": "Average Precision",
                      "gain_pour_100": "Gain pour 100 clients (UM)", "n_clients": "Clients"})


# Navigation

parametres = service.parametres_metier()
st.sidebar.title("Campagne marketing")
ecran = st.sidebar.radio("Écran", ["Décision", "Exploration", "Modèle"])
st.sidebar.caption(insecable(f"Règle de décision : contacter si la probabilité de réponse "
                             f"dépasse {pourcent(parametres['seuil'])}. Un contact coûte "
                             f"{parametres['cout_contact']} UM, une réponse rapporte "
                             f"{parametres['revenu_reponse']} UM."))

{"Décision": ecran_decision, "Exploration": ecran_exploration,
 "Modèle": ecran_modele}[ecran](parametres)
