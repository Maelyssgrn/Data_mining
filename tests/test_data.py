"""Tests de la table d'analyse : elle respecte le nettoyage décrit dans
docs/data_quality_card.md. Depuis un clone vide, lire_analyse() recrée la base."""

import pytest

from backend import data


@pytest.fixture(scope="module")
def analyse():
    return data.lire_analyse()


def test_une_ligne_par_client(analyse):
    # Doublons retirés : sinon un même client pourrait être dans le train et dans le test
    assert analyse["ID"].is_unique


def test_nombre_de_clients(analyse):
    # 2 240 lignes brutes, moins conflits, doublons, valeurs impossibles et fiches erronées
    assert len(analyse) == 2010


def test_cible_binaire(analyse):
    assert analyse["Response"].notna().all()
    assert set(analyse["Response"].unique()) == {0, 1}


def test_pas_de_valeur_impossible(analyse):
    assert (analyse["Year_Birth"] >= 1901).all()
    assert not (analyse["Income"] == 666666).any()


def test_revenus_manquants_conserves(analyse):
    # Ni imputés ici (la médiane s'apprend sur le train), ni perdus par le filtre SQL
    assert analyse["Income"].isna().sum() == 24


def test_pas_de_colonne_z(analyse):
    # Colonnes constantes : utilisées comme coûts dans config.py, pas comme variables
    assert not [c for c in analyse.columns if c.startswith("Z_")]


def test_aucun_client_sans_achat(analyse):
    # Fiches erronées (aucun achat mais des montants) retirées
    assert (analyse["Nb_achats"] > 0).all()
