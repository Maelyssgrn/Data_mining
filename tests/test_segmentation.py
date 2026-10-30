"""Tests de la segmentation décrite dans docs/interpretation_segmentation.md :
11 variables (ni Response, ni ID, ni Z_), k = 3, rien d'exécuté à l'import."""

import importlib

import pytest

from backend import config, segmentation

VARIABLES = ["Income", "Enfants",
             "Part_Vin", "Part_Viande", "Part_Poisson", "Part_Or", "Part_Sucreries", "Part_Fruits",
             "Part_Magasin", "Part_Web", "Part_Catalogue"]


@pytest.fixture(scope="module")
def X():
    return segmentation.variables_segmentation()


def test_variables_autorisees(X):
    # Exactement les 11 variables prévues : la cible ne peut pas entrer dans les segments
    assert sorted(X.columns) == sorted(VARIABLES)
    assert not {"Response", "ID"} & set(X.columns)
    assert not [c for c in X.columns if c.startswith("Z_")]


def test_trois_segments_un_par_client(X):
    segments = segmentation.segmenter(X)
    assert segments.index.equals(X.index)  # aucun client perdu (Income manquants imputés)
    assert segments.notna().all()
    assert segments.nunique() == 3


def test_import_sans_effet(monkeypatch, tmp_path, capsys):
    # docs/figures/ est versionné : on pointe vers un dossier qui n'existe pas encore,
    # puis on réimporte le module. Il ne doit rien afficher ni rien créer.
    dossier = tmp_path / "figures"
    monkeypatch.setattr(config, "DOSSIER_FIGURES", dossier)
    importlib.reload(segmentation)
    assert capsys.readouterr().out == ""
    assert not dossier.exists()
