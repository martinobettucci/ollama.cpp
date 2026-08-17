"""Tests du nommage des modèles.

@verifies docs/BACKLOG.md OC-012 « Nommage des modèles »
@verifies docs/ollama.cpp-architecture.md §2.5 « Nommage »

Les cas couverts sont ceux qui font la différence entre « mêmes URLs » et « même comportement » :
défauts implicites, découpage tag/hôte-avec-port, parties promises mais vides, et refus de dériver
un chemin depuis un nom invalide (risque R8).
"""

from __future__ import annotations

import pytest

from ollamacpp.names import (
    DEFAULT_HOST,
    DEFAULT_NAMESPACE,
    DEFAULT_TAG,
    MISSING_PART,
    ModelRef,
    parse,
    parse_bare,
    parse_from_filepath,
)


class TestParseDefaults:
    """Les valeurs par défaut d'Ollama sont appliquées à l'analyse complète."""

    def test_nom_nu_recoit_les_trois_defauts(self):
        ref = parse("llama3")
        assert (ref.host, ref.namespace, ref.model, ref.tag) == (
            DEFAULT_HOST,
            DEFAULT_NAMESPACE,
            "llama3",
            DEFAULT_TAG,
        )

    def test_tag_explicite_est_conserve(self):
        assert parse("llama3:8b").tag == "8b"

    def test_namespace_explicite_est_conserve(self):
        ref = parse("acme/llama3")
        assert ref.namespace == "acme"
        assert ref.host == DEFAULT_HOST

    def test_nom_pleinement_qualifie(self):
        ref = parse("registre.local/acme/llama3:8b")
        assert (ref.host, ref.namespace, ref.model, ref.tag) == (
            "registre.local",
            "acme",
            "llama3",
            "8b",
        )


class TestParseBare:
    """L'analyse nue n'invente aucune valeur : c'est ce qui permet de détecter un nom incomplet."""

    def test_aucun_defaut_applique(self):
        ref = parse_bare("llama3")
        assert (ref.host, ref.namespace, ref.tag) == ("", "", "")
        assert ref.model == "llama3"

    def test_schema_de_protocole_retire(self):
        assert parse_bare("https://registre.local/acme/llama3:8b").host == "registre.local"


class TestDecoupageTagVsHote:
    """Un « : » d'hôte avec port ne doit jamais être pris pour un tag."""

    def test_hote_avec_port_sans_tag(self):
        ref = parse_bare("registre.local:8443/acme/llama3")
        assert ref.host == "registre.local:8443"
        assert ref.tag == ""
        assert ref.model == "llama3"

    def test_hote_avec_port_et_tag(self):
        ref = parse_bare("registre.local:8443/acme/llama3:8b")
        assert ref.host == "registre.local:8443"
        assert ref.tag == "8b"


class TestPartiesPromises:
    """Un séparateur présent « promet » une partie non vide ; sinon le nom devient invalide."""

    @pytest.mark.parametrize(
        "raw, champ",
        [
            ("llama3:", "tag"),
            ("/llama3", "model"),
            ("acme//llama3", "namespace"),
        ],
    )
    def test_partie_vide_devient_missing(self, raw: str, champ: str):
        ref = parse_bare(raw)
        assert MISSING_PART in (ref.host, ref.namespace, ref.model, ref.tag)
        assert not parse(raw).is_valid, f"{raw!r} ne doit pas être valide ({champ} vide)"


class TestValidite:
    @pytest.mark.parametrize(
        "raw",
        ["llama3", "llama3:8b", "acme/llama3:8b", "registre.local/acme/llama3:8b", "_x/_y:_z"],
    )
    def test_noms_valides(self, raw: str):
        assert parse(raw).is_valid

    @pytest.mark.parametrize(
        "raw, motif",
        [
            ("-llama3", "premier caractère non alphanumérique"),
            ("acme.corp/llama3", "point interdit dans un namespace"),
            ("llama3:8b:extra", "deux-points interdits dans un tag"),
            ("", "nom vide"),
        ],
    )
    def test_noms_invalides(self, raw: str, motif: str):
        assert not parse(raw).is_valid, motif

    def test_longueur_maximale_du_modele(self):
        assert parse("a" * 80).is_valid
        assert not parse("a" * 81).is_valid

    def test_longueur_maximale_de_lhote(self):
        assert parse(f"{'h' * 350}/acme/m:t").is_valid
        assert not parse(f"{'h' * 351}/acme/m:t").is_valid


class TestRendu:
    def test_str_conserve_toutes_les_parties(self):
        assert str(parse("llama3")) == f"{DEFAULT_HOST}/{DEFAULT_NAMESPACE}/llama3:{DEFAULT_TAG}"

    def test_display_shortest_omet_les_defauts(self):
        assert parse("llama3").display_shortest() == "llama3:latest"

    def test_display_shortest_conserve_un_namespace_non_defaut(self):
        assert parse("acme/llama3:8b").display_shortest() == "acme/llama3:8b"

    def test_display_shortest_conserve_un_hote_non_defaut(self):
        ref = parse("registre.local/acme/llama3:8b")
        assert ref.display_shortest() == "registre.local/acme/llama3:8b"

    def test_display_shortest_force_toujours_le_tag(self):
        """Même implicite, le tag est affiché : c'est le comportement d'`ollama list`."""
        assert parse("llama3").display_shortest().endswith(":latest")

    def test_hote_non_defaut_force_aussi_le_namespace(self):
        """Quand l'hôte diffère du défaut, le namespace est affiché même s'il vaut le défaut.

        Le nom doit être écrit en trois parties pour porter un hôte : Ollama analyse de droite à
        gauche, donc `a/b` est `namespace/model`, jamais `host/model` (cf. test suivant).
        """
        ref = parse(f"registre.local/{DEFAULT_NAMESPACE}/llama3")
        assert ref.host == "registre.local"
        assert ref.display_shortest() == f"registre.local/{DEFAULT_NAMESPACE}/llama3:latest"

    def test_deux_parties_designent_un_namespace_pas_un_hote(self):
        """`a/b` est `namespace/model` : c'est l'analyse de droite à gauche d'Ollama.

        Piège réel : un opérateur qui écrit `registre.local/llama3` en croyant désigner un hôte
        obtient un namespace nommé `registre.local` sur l'hôte par défaut.
        """
        ref = parse("registre.local/llama3")
        assert ref.host == DEFAULT_HOST
        assert ref.namespace == "registre.local"
        assert ref.display_shortest() == "registre.local/llama3:latest"


class TestFilepath:
    def test_chemin_canonique(self):
        assert str(parse("acme/llama3:8b").filepath()) == f"{DEFAULT_HOST}/acme/llama3/8b"

    def test_refus_dun_nom_invalide(self):
        """Aucun chemin ne peut être dérivé d'un nom invalide (risque R8)."""
        with pytest.raises(ValueError):
            parse("-invalide").filepath()

    @pytest.mark.parametrize("attaque", ["../../etc/passwd", "acme/../../../etc/x:latest"])
    def test_refus_des_tentatives_de_traversee(self, attaque: str):
        ref = parse(attaque)
        assert not ref.is_valid
        with pytest.raises(ValueError):
            ref.filepath()

    def test_aller_retour_chemin(self):
        ref = parse("acme/llama3:8b")
        assert parse_from_filepath(str(ref.filepath())) == ref

    def test_chemin_a_mauvais_nombre_de_segments(self):
        assert parse_from_filepath("acme/llama3") == ModelRef()

    def test_chemin_invalide_ignore(self):
        assert parse_from_filepath("h/-ns/m/t") == ModelRef()


class TestComparaison:
    def test_equal_fold_ignore_la_casse(self):
        assert parse("ACME/Llama3:8B").equal_fold(parse("acme/llama3:8b"))

    def test_reference_utilisable_comme_cle(self):
        """`ModelRef` sert de clé de registre et de cache : elle doit être hachable."""
        assert len({parse("llama3"), parse("llama3:latest")}) == 1
