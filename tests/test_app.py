"""Test de démarrage de l'application Streamlit (streamlit.testing.v1.AppTest) :
elle démarre, puis chacun des 3 écrans s'affiche sans erreur."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")
ECRANS = {"Décision": "Décision", "Exploration": "Exploration", "Modèle": "Modèle"}


@pytest.fixture(scope="module")
def app():
    # Premier lancement : segmentation et chargement du modèle (quelques secondes)
    return AppTest.from_file(APP, default_timeout=120).run()


def test_demarrage(app):
    assert not app.exception
    assert app.sidebar.radio[0].options == list(ECRANS)


@pytest.mark.parametrize("ecran", ECRANS)
def test_ecrans(app, ecran):
    app.sidebar.radio[0].set_value(ecran).run()
    assert not app.exception
    # Pas de message d'erreur : sinon l'écran s'est arrêté (modèle absent, par exemple)
    assert not app.error, "lancer d'abord python -m backend.train"
    assert app.title[0].value.startswith(ECRANS[ecran])
