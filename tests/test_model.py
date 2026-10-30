"""Tests de la logique métier : règle de décision de backend/model.py
(contacter si p > 3/11 : un contact coûte 3 UM, une réponse rapporte 11 UM)."""

import pandas as pd
import pytest

from backend import data, model


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
