"""Démonstrateur Streamlit : affichage et saisie uniquement.

Toute la logique est dans le backend ; ce fichier n'importe que backend.service.
Lancement (depuis la racine du dépôt) : python -m streamlit run frontend/app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from backend import service

# Couleurs des accents (le thème général est dans .streamlit/config.toml)
CONTACTER = "#A8D5BA"          # vert sauge
NE_PAS_CONTACTER = "#F4B6B6"   # rose poudré
TEXTE = "#2E2A35"
FOND_JAUGE = "#E6E1F2"
GRILLE = "#ECE8F3"
# Une couleur fixe par persona, la même dans tous les écrans et graphiques
COULEURS_PERSONAS = {
    "Familles modestes au panier varié": "#F7C59F",    # pêche
    "Familles amatrices de vin": "#A7C7E7",            # bleu ciel
    "Aisés, peu d'enfants, vin et viande": "#B5E2D0",  # menthe
}
SITUATIONS = {"Married": "Marié(e)", "Together": "En couple", "Single": "Célibataire",
              "Divorced": "Divorcé(e)", "Widow": "Veuf ou veuve", "Autre": "Autre"}

st.set_page_config(page_title="Ciblage de campagne", layout="wide")
st.markdown("""
<style>
.carte {border-radius: 18px; padding: 1.3rem 1.6rem; color: #2E2A35; margin-bottom: 0.8rem;}
.carte .verdict {font-size: 2.1rem; font-weight: 700; margin: 0 0 0.3rem 0;}
.carte p {margin: 0.2rem 0; font-size: 1.05rem;}
.pastille {display: inline-block; padding: 0.35rem 0.95rem; border-radius: 999px;
           font-weight: 600; color: #2E2A35;}
</style>
""", unsafe_allow_html=True)


# Mise en forme (nombres à la française)

def nombre(x, decimales=0):
    return f"{x:,.{decimales}f}".replace(",", " ").replace(".", ",")


def pourcent(p, decimales=1):
    return f"{nombre(100 * p, decimales)} %"


def um(x, decimales=2):
    signe = "+" if x > 0 else "−" if x < 0 else ""
    return f"{signe}{nombre(abs(x), decimales)} UM"


def style(graphique):
    """Police assez grande pour une démo projetée, texte foncé, axes titrés."""
    return (graphique.configure_axis(labelFontSize=16, titleFontSize=16, labelColor=TEXTE,
                                     titleColor=TEXTE, gridColor=GRILLE)
            .configure_legend(labelFontSize=16, titleFontSize=16, labelColor=TEXTE)
            .configure_view(strokeWidth=0))


# Chargements mis en cache (le backend ne réentraîne rien)

@st.cache_data
def charger_clients():
    return service.clients_test()


@st.cache_data
def charger_personas():
    return service.personas()


def exiger_modele():
    """Arrête l'écran avec un message clair si le modèle n'a pas été entraîné."""
    if not service.modele_disponible():
        st.error("Le modèle n'est pas encore entraîné. Lancer, depuis la racine du dépôt :")
        st.code("python -m backend.train", language="bash")
        st.stop()


# Écran Décision

def jauge(p, seuil, couleur):
    """Barre horizontale de la probabilité, avec le seuil de rentabilité en pointillés."""
    axe = alt.Axis(title="Probabilité de réponse", values=[0, 0.25, 0.5, 0.75, 1],
                   labelExpr="format(datum.value * 100, '.0f') + ' %'")
    x = alt.X("x:Q", scale=alt.Scale(domain=[0, 1]), axis=axe)
    fond = alt.Chart(pd.DataFrame({"x": [1]})).mark_bar(color=FOND_JAUGE, size=36,
                                                        cornerRadius=10).encode(x=x)
    barre = alt.Chart(pd.DataFrame({"x": [p]})).mark_bar(color=couleur, size=36,
                                                         cornerRadius=10).encode(x=x)
    trait = alt.Chart(pd.DataFrame({"x": [seuil]})).mark_rule(
        color=TEXTE, strokeDash=[6, 4], strokeWidth=2).encode(x=x)
    texte = alt.Chart(pd.DataFrame({"x": [seuil], "t": [f"seuil {pourcent(seuil)}"]})).mark_text(
        dy=-30, fontSize=15, color=TEXTE).encode(x=x, text="t:N")
    return style((fond + barre + trait + texte).properties(height=120))


def ecran_decision(parametres):
    st.title("Décision : faut-il contacter ce client ?")
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
    st.caption("Seuls les 402 clients du jeu de test sont proposés : le modèle n'a pas vu leur "
               "réponse pendant l'entraînement, sa probabilité est donc honnête.")

    d = service.decision_client(id_client)
    p, gain = d["probabilite"], d["gain_attendu"]
    couleur = CONTACTER if d["contacter"] else NE_PAS_CONTACTER
    verdict = "Contacter" if d["contacter"] else "Ne pas contacter"
    position = "au-dessus" if d["contacter"] else "en dessous"

    gauche, droite = st.columns([3, 2], gap="large")
    with gauche:
        st.markdown(f"""
        <div class="carte" style="background:{couleur}">
          <div class="verdict">{verdict}</div>
          <p>Probabilité de réponse : <b>{pourcent(p)}</b>, {position} du seuil de
             rentabilité ({pourcent(seuil)}).</p>
          <p>Gain attendu d'un contact : {nombre(p, 3)} × {parametres['revenu_reponse']}
             − {parametres['cout_contact']} = <b>{um(gain)}</b></p>
        </div>""", unsafe_allow_html=True)
        st.altair_chart(jauge(p, seuil, couleur), width="stretch")
    with droite:
        st.markdown("**Persona du client**")
        actions = charger_personas().set_index("nom")["action"]
        st.markdown(f'<span class="pastille" style="background:'
                    f'{COULEURS_PERSONAS.get(d["persona"], GRILLE)}">{d["persona"]}</span>',
                    unsafe_allow_html=True)
        st.caption(f"Pour l'ensemble du segment : {actions[d['persona']]}")
        st.metric("Probabilité de réponse", pourcent(p), border=True)
        st.metric("Gain attendu d'un contact", um(gain), border=True)

    st.subheader("Profil du client")
    profil = [("Âge", f"{d['Age']} ans"),
              ("Revenu", "inconnu" if pd.isna(d["Income"]) else f"{nombre(d['Income'])} UM"),
              ("Enfants au foyer", d["Enfants"]),
              ("Situation familiale", SITUATIONS.get(d["Marital_Status"], d["Marital_Status"])),
              ("Campagnes déjà acceptées", d["Nb_campagnes_acceptees"]),
              ("Dernier achat", f"il y a {d['Recency']} jours"),
              ("Dépense totale (2 ans)", f"{nombre(d['Depense_totale'])} UM")]
    colonnes = st.columns(4)
    for i, (libelle, valeur) in enumerate(profil):
        colonnes[i % 4].metric(libelle, valeur, border=True)

    with st.expander("Vérification (jeu de test)"):
        reponse = "a répondu" if d["reponse_reelle"] == 1 else "n'a pas répondu"
        st.write(f"À la dernière campagne, ce client **{reponse}**. Cette information sert "
                 "seulement à vérifier le modèle en démonstration : pour la prochaine "
                 "campagne, la réponse n'est pas connue à l'avance.")


# Navigation

parametres = service.parametres_metier()
st.sidebar.title("Campagne marketing")
ecran = st.sidebar.radio("Écran", ["Décision", "Exploration", "Modèle"])
st.sidebar.caption(f"Règle de décision : contacter si la probabilité de réponse dépasse "
                   f"{pourcent(parametres['seuil'])}. Un contact coûte "
                   f"{parametres['cout_contact']} UM, une réponse rapporte "
                   f"{parametres['revenu_reponse']} UM.")

if ecran == "Décision":
    ecran_decision(parametres)
else:
    st.title(ecran)
    st.info("Écran en préparation.")
