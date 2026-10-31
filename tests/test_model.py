"""Tests de la logique métier : règle de décision de backend/model.py
(contacter si p > 3/11 : un contact coûte 3 UM, une réponse rapporte 11 UM)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from backend import config, data, model


def test_seuil_rentabilite():
    assert model.SEUIL_RENTABILITE == pytest.approx(3 / 11)


def test_gain_nul_au_seuil():
    # 3 réponses pour 11 contacts : exactement le seuil, ni gain ni perte
    assert model.gain_net(n_contactes=11, n_reponses=3) == 0


def test_bilan_exemple_a_la_main():
    # On contacte les 2 premiers clients ; seul le premier répond
    contacter = pd.Series([True, True, False, False])
    response = pd.Series([1, 0, 1, 0])
    assert model.bilan_campagne(contacter, response) == {
        "n_contactes": 2, "n_reponses": 1, "taux_pct": 50.0, "gain_net": 11 - 2 * 3}


def test_bilan_personne_contacte():
    # Aucun contact : pas de division par zéro, gain nul
    contacter = pd.Series([False, False, False])
    response = pd.Series([1, 0, 1])
    assert model.bilan_campagne(contacter, response) == {
        "n_contactes": 0, "n_reponses": 0, "taux_pct": 0.0, "gain_net": 0}


def test_campagne_de_masse_et_baseline():
    # Chiffres du tableau « Segments ou règles simples ? » de docs/interpretation_segmentation.md
    besoin = model.besoin_client(data.lire_analyse())
    assert besoin.loc["Tout le monde", "gain_net"] == -2796
    baseline = besoin.loc["Déjà accepté une campagne"]
    assert (baseline["n_contactes"], baseline["n_reponses"], baseline["gain_net"]) == (416, 168, 600)


@pytest.mark.parametrize("gain_foret, gain_logistique, ecart_type, attendu", [
    (162.2, 159.6, 30.4, "Régression logistique"),  # nos chiffres : écart < 1 écart-type
    (200.0, 150.0, 30.0, "Random Forest"),          # écart > 1 écart-type : la forêt l'emporte
])
def test_choisir_garde_le_modele_simple(gain_foret, gain_logistique, ecart_type, attendu):
    cv = pd.DataFrame({"gain_moyen": [gain_logistique, gain_foret],
                       "gain_ecart_type": [25.0, ecart_type]},
                      index=["Régression logistique", "Random Forest"])
    assert model.choisir(cv)[0] == attendu


class ProbasFixees:
    """Faux modèle : renvoie des probabilités choisies, pour tester la règle seule."""
    def __init__(self, p):
        self.p = np.array(p)

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])


@pytest.mark.parametrize("p, attendu", [
    ([0.10, 0.20, 0.27], [False, False, False]),       # tous sous 3/11 : personne
    ([0.10, 0.28, 3 / 11], [False, True, False]),      # p = 3/11 : gain nul, pas de contact
])
def test_contacter_si_rentable(p, attendu):
    clients = data.lire_analyse().head(3)
    assert model.contacter_si_rentable(ProbasFixees(p), clients).tolist() == attendu


def test_modele_sauvegarde():
    assert config.CHEMIN_MODELE.exists(), "lancer d'abord python -m backend.train"
    modele = joblib.load(config.CHEMIN_MODELE)
    _, test = model.separer(data.lire_analyse())
    # Échoue aussi si le modèle a été entraîné avec d'autres variables (modèle périmé)
    p = modele.predict_proba(model.variables(test))[:, 1]
    assert len(p) == len(test)
    assert ((p >= 0) & (p <= 1)).all()
