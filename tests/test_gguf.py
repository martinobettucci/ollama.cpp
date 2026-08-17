"""Tests de la lecture des métadonnées GGUF.

@verifies docs/BACKLOG.md OC-023 « Métadonnées GGUF »
@verifies docs/ollama.cpp-architecture.md §5.6 « Ordre de précédence de la configuration »
"""

from __future__ import annotations

import struct

import pytest

from ollamacpp.gguf import GGUFError, file_type_name, human_number, read_metadata

from .ggufbuild import DEFAULT_KV, build_gguf, build_mmproj, write_gguf


@pytest.fixture
def modele(tmp_path):
    return write_gguf(tmp_path / "modele.gguf")


class TestLectureDenTete:
    def test_version_et_compteurs(self, modele):
        meta = read_metadata(modele)
        assert meta.version == 3
        assert meta.tensor_count == 0
        assert meta.file_size == modele.stat().st_size

    def test_architecture(self, modele):
        assert read_metadata(modele).architecture == "qwen3"

    def test_toutes_les_cles_lues(self, modele):
        assert set(read_metadata(modele).kv) == set(DEFAULT_KV)


class TestChampsDerivés:
    def test_quantisation(self, modele):
        """`general.file_type` 15 = Q4_K_M, nom aligné sur `fs/ggml/type.go` d'Ollama."""
        assert read_metadata(modele).quantization_level == "Q4_K_M"

    def test_taille_de_parametres(self, modele):
        """Format `HumanNumber` d'Ollama : 7,6 milliards → `7.6B`."""
        assert read_metadata(modele).parameter_size == "7.6B"

    def test_contexte_natif_prefixe_par_larchitecture(self, modele):
        """La clé est `<architecture>.context_length` : il faut lire l'architecture d'abord."""
        assert read_metadata(modele).context_length == 32768

    def test_longueur_dembedding(self, modele):
        assert read_metadata(modele).embedding_length == 4096

    def test_chat_template(self, modele):
        assert "{% for m in messages %}" in read_metadata(modele).chat_template

    def test_architecture_absente_ne_casse_pas(self, tmp_path):
        chemin = write_gguf(tmp_path / "nu.gguf", kv={"general.name": "sans architecture"})
        meta = read_metadata(chemin)
        assert meta.architecture == ""
        assert meta.context_length == 0

    def test_projecteur_multimodal_detecte(self, tmp_path):
        chemin = tmp_path / "mmproj.gguf"
        chemin.write_bytes(build_mmproj())
        assert read_metadata(chemin).is_multimodal_projector

    def test_modele_normal_nest_pas_un_projecteur(self, modele):
        assert not read_metadata(modele).is_multimodal_projector


class TestModelInfoPublic:
    def test_tableaux_volumineux_remplaces_par_leur_longueur(self, modele):
        """Transporter un vocabulaire entier dans chaque `/api/show` serait absurde."""
        info = read_metadata(modele).public_model_info()
        assert "tokenizer.ggml.tokens" not in info
        assert info["tokenizer.ggml.tokens.length"] == 5

    def test_scalaires_conserves(self, modele):
        info = read_metadata(modele).public_model_info()
        assert info["qwen3.context_length"] == 32768
        assert info["general.architecture"] == "qwen3"

    def test_serialisable_en_json(self, modele):
        import json

        json.dumps(read_metadata(modele).public_model_info())


class TestTypesDeValeurs:
    @pytest.mark.parametrize(
        "valeur, attendu",
        [(True, True), (False, False), (42, 42), ("texte", "texte"), (["a", "b"], ["a", "b"])],
    )
    def test_aller_retour_des_types(self, tmp_path, valeur, attendu):
        chemin = tmp_path / "types.gguf"
        chemin.write_bytes(build_gguf({"general.architecture": "test", "x": valeur}))
        assert read_metadata(chemin).kv["x"] == attendu

    def test_tableau_vide(self, tmp_path):
        chemin = tmp_path / "vide.gguf"
        chemin.write_bytes(build_gguf({"vide": []}))
        assert read_metadata(chemin).kv["vide"] == []


class TestRobustesse:
    """Un GGUF hostile ou corrompu ne doit ni planter le service ni allouer sans borne."""

    def test_magie_absente(self, tmp_path):
        chemin = tmp_path / "faux.gguf"
        chemin.write_bytes(b"XXXX" + b"\x00" * 32)
        with pytest.raises(GGUFError, match="magie"):
            read_metadata(chemin)

    def test_fichier_tronque(self, tmp_path):
        chemin = tmp_path / "tronque.gguf"
        chemin.write_bytes(build_gguf(DEFAULT_KV)[:40])
        with pytest.raises(GGUFError):
            read_metadata(chemin)

    def test_nombre_de_cles_demesure(self, tmp_path):
        """Défense contre un en-tête annonçant des milliards de paires."""
        chemin = tmp_path / "hostile.gguf"
        chemin.write_bytes(
            b"GGUF" + struct.pack("<I", 3) + struct.pack("<q", 0) + struct.pack("<q", 2**40)
        )
        with pytest.raises(GGUFError, match="clé/valeur"):
            read_metadata(chemin)

    def test_nombre_de_tenseurs_negatif(self, tmp_path):
        chemin = tmp_path / "negatif.gguf"
        chemin.write_bytes(
            b"GGUF" + struct.pack("<I", 3) + struct.pack("<q", -1) + struct.pack("<q", 0)
        )
        with pytest.raises(GGUFError, match="tenseurs"):
            read_metadata(chemin)

    def test_fichier_vide(self, tmp_path):
        chemin = tmp_path / "vide.gguf"
        chemin.write_bytes(b"")
        with pytest.raises(GGUFError):
            read_metadata(chemin)


class TestFileTypeName:
    @pytest.mark.parametrize(
        "valeur, attendu",
        [(0, "F32"), (1, "F16"), (2, "Q4_0"), (15, "Q4_K_M"), (18, "Q6_K"), (32, "BF16"),
         (25, "IQ4_NL"), (38, "MXFP4_MOE")],
    )
    def test_noms_alignes_sur_ollama(self, valeur, attendu):
        assert file_type_name(valeur) == attendu

    def test_ftype_inconnu_nempeche_pas_le_listing(self, tmp_path):
        """Un ftype d'une version future de `llama.cpp` ne doit pas casser `/api/tags`."""
        assert file_type_name(9999) == "unknown"
        assert file_type_name(None) == "unknown"


class TestHumanNumber:
    @pytest.mark.parametrize(
        "valeur, attendu",
        [
            (7_600_000_000, "7.6B"),
            (8_000_000_000, "8B"),
            (109_000_000, "109M"),
            (109_500_000, "109.50M"),
            (560_000, "560K"),
            (42, "42"),
        ],
    )
    def test_formats_dollama(self, valeur, attendu):
        """Seuils et décimales repris de `format.HumanNumber` (`format/format.go`)."""
        assert human_number(valeur) == attendu
