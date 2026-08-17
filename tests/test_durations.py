"""Tests des durées et de `keep_alive`.

@verifies docs/BACKLOG.md OC-013 « Durées et keep_alive »
@verifies docs/ollama.cpp-architecture.md §2.2 « Conventions comportementales », risques R3 et R5

Ces deux comportements échouent silencieusement quand ils sont faux : une durée en secondes au
lieu de nanosecondes produit un débit faux d'un facteur 10⁹ sans erreur, et un `keep_alive`
mal interprété laisse des modèles résidents pour toujours ou les décharge en boucle. D'où une
table de vérité exhaustive plutôt que quelques cas nominaux.
"""

from __future__ import annotations

import math

import pytest

from ollamacpp.durations import (
    DEFAULT_KEEP_ALIVE_SECONDS,
    DurationError,
    KeepAlive,
    format_go_duration,
    parse_go_duration,
    parse_keep_alive,
    seconds_to_nanoseconds,
)


class TestTableDeVeriteKeepAlive:
    """Table portée depuis `Duration.UnmarshalJSON` (`api/types.go` l. 1243-1271)."""

    def test_absent_vaut_cinq_minutes(self):
        assert parse_keep_alive(None).seconds == DEFAULT_KEEP_ALIVE_SECONDS == 300.0

    @pytest.mark.parametrize("valeur, attendu", [(0, 0.0), (1, 1.0), (300, 300.0), (42.5, 42.5)])
    def test_nombre_positif_est_en_secondes(self, valeur, attendu):
        """Piège n° 1 : un nombre est en SECONDES, pas en millisecondes ni en nanosecondes."""
        assert parse_keep_alive(valeur).seconds == attendu

    @pytest.mark.parametrize("valeur", [-1, -0.5, -3600])
    def test_nombre_negatif_vaut_residence_illimitee(self, valeur):
        assert parse_keep_alive(valeur).is_infinite

    @pytest.mark.parametrize(
        "texte, attendu",
        [
            ("5m", 300.0),
            ("10m", 600.0),
            ("1h", 3600.0),
            ("1h30m", 5400.0),
            ("2h45m30s", 9930.0),
            ("300ms", 0.3),
            ("1.5h", 5400.0),
            ("0", 0.0),
            ("30s", 30.0),
            ("500us", 0.0005),
            ("500µs", 0.0005),
            ("100ns", 1e-7),
        ],
    )
    def test_chaine_est_une_duree_go(self, texte, attendu):
        assert parse_keep_alive(texte).seconds == pytest.approx(attendu)

    @pytest.mark.parametrize("texte", ["-1s", "-5m", "-1h"])
    def test_chaine_negative_vaut_residence_illimitee(self, texte):
        assert parse_keep_alive(texte).is_infinite

    @pytest.mark.parametrize("valeur", [True, False])
    def test_booleen_refuse(self, valeur):
        """`bool` est un `int` en Python : sans garde, `true` deviendrait « une seconde »."""
        with pytest.raises(DurationError):
            parse_keep_alive(valeur)

    @pytest.mark.parametrize("valeur", [[], {}, object()])
    def test_type_non_supporte_refuse(self, valeur):
        with pytest.raises(DurationError):
            parse_keep_alive(valeur)

    @pytest.mark.parametrize("texte", ["", "   ", "abc", "5x", "5m3", "m5"])
    def test_chaine_invalide_refuse(self, texte):
        """Aucun repli silencieux sur la valeur par défaut : une durée illisible est une erreur."""
        with pytest.raises(DurationError):
            parse_keep_alive(texte)

    def test_nan_refuse(self):
        with pytest.raises(DurationError):
            parse_keep_alive(float("nan"))


class TestSemantiqueKeepAlive:
    def test_zero_decharge_immediatement(self):
        """`keep_alive: 0` signifie décharger dès la fin de la requête (`docs/api.md`)."""
        assert parse_keep_alive(0).unloads_immediately

    def test_valeur_positive_ne_decharge_pas_immediatement(self):
        assert not parse_keep_alive("5m").unloads_immediately

    def test_illimite_nest_pas_un_dechargement(self):
        illimite = parse_keep_alive(-1)
        assert illimite.is_infinite
        assert not illimite.unloads_immediately

    def test_representation_lisible(self):
        assert str(KeepAlive(math.inf)) == "infinite"
        assert str(KeepAlive(300)) == "300s"


class TestNanosecondes:
    """Risque R3 : toutes les durées des réponses Ollama sont en nanosecondes."""

    @pytest.mark.parametrize(
        "secondes, attendu",
        [(1.0, 1_000_000_000), (0.001, 1_000_000), (10.706818083, 10_706_818_083), (0.0, 0)],
    )
    def test_conversion(self, secondes, attendu):
        assert seconds_to_nanoseconds(secondes) == attendu

    def test_resultat_entier(self):
        """Un flottant en sortie casserait les clients qui attendent un entier JSON."""
        assert isinstance(seconds_to_nanoseconds(1.5), int)

    def test_ordre_de_grandeur_realiste(self):
        """Contrôle du facteur 10⁹ sur un cas issu de la documentation Ollama."""
        assert seconds_to_nanoseconds(10.706818083) == 10_706_818_083


class TestFormatage:
    @pytest.mark.parametrize(
        "secondes, attendu",
        [(0, "0s"), (30, "30s"), (300, "5m0s"), (5400, "1h30m0s"), (3600, "1h0m0s")],
    )
    def test_format_go(self, secondes, attendu):
        assert format_go_duration(secondes) == attendu

    def test_illimite(self):
        assert format_go_duration(math.inf) == "infinite"

    @pytest.mark.parametrize("secondes", [30, 300, 5400, 0.3, 9930])
    def test_aller_retour(self, secondes):
        """Ce qui est formaté doit pouvoir être relu : c'est ce que fait un client Ollama."""
        assert parse_go_duration(format_go_duration(secondes)) == pytest.approx(secondes)
