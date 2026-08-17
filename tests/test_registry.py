"""Tests du registre de modèles.

@verifies docs/BACKLOG.md OC-022 « ModelRegistry »
@verifies docs/ollama.cpp-architecture.md §4.2 « Fonctionnalités manquantes », risque R4
          « size / digest bouchons hérités de llama-server »
@verifies docs/DAT.md §3.2 « Chargement d'un modèle »
"""

from __future__ import annotations

import pytest

from ollamacpp.errors import ModelNotFound
from ollamacpp.registry import ModelRegistry
from ollamacpp.storage import Artifacts, BlobStore, Manifest, ManifestStore
from ollamacpp.names import parse

from .ggufbuild import build_gguf, build_mmproj, DEFAULT_KV


@pytest.fixture
def registre(tmp_path) -> ModelRegistry:
    blobs = BlobStore(tmp_path / "blobs", tmp_path / "tmp")
    blobs.ensure_layout()
    return ModelRegistry(blobs, ManifestStore(tmp_path / "manifests"))


def installer(registre: ModelRegistry, nom: str = "qwen3:8b", *, avec_mmproj: bool = False,
              mmproj_present: bool = True) -> Manifest:
    """Installe un modèle de test avec de vrais artefacts GGUF dans le magasin."""
    modele = registre.blobs.ingest_bytes(build_gguf(DEFAULT_KV))
    mmproj_digest = None
    if avec_mmproj:
        contenu = build_mmproj()
        if mmproj_present:
            mmproj_digest = registre.blobs.ingest_bytes(contenu).digest
        else:
            from ollamacpp.storage import compute_digest
            import io

            mmproj_digest = compute_digest(io.BytesIO(contenu))

    manifest = Manifest(
        name=parse(nom),
        artifacts=Artifacts(model=modele.digest, mmproj=mmproj_digest),
    )
    registre.install(manifest)
    return manifest


class TestIdentiteReelle:
    """Risque R4 : `size` et `digest` doivent être réels, pas des bouchons."""

    def test_digest_reel(self, registre):
        manifest = installer(registre)
        assert registre.get("qwen3:8b").digest == manifest.artifacts.model
        assert registre.get("qwen3:8b").digest.startswith("sha256:")

    def test_taille_reelle(self, registre):
        installer(registre)
        modele = registre.get("qwen3:8b")
        assert modele.size == len(build_gguf(DEFAULT_KV))
        assert isinstance(modele.size, int)

    def test_taille_somme_les_artefacts(self, registre):
        installer(registre, avec_mmproj=True)
        attendu = len(build_gguf(DEFAULT_KV)) + len(build_mmproj())
        assert registre.get("qwen3:8b").size == attendu

    def test_date_de_modification_reelle(self, registre):
        installer(registre)
        import datetime as dt

        modele = registre.get("qwen3:8b")
        assert isinstance(modele.modified_at, dt.datetime)
        assert modele.modified_at.tzinfo is not None

    def test_taille_nannonce_pas_un_artefact_absent(self, registre):
        """Annoncer la taille d'un artefact manquant tromperait le scheduler."""
        installer(registre, avec_mmproj=True, mmproj_present=False)
        assert registre.get("qwen3:8b").size == len(build_gguf(DEFAULT_KV))


class TestDetailsOllama:
    def test_details_derives_du_gguf(self, registre):
        installer(registre)
        details = registre.get("qwen3:8b").details()
        assert details["family"] == "qwen3"
        assert details["families"] == ["qwen3"]
        assert details["quantization_level"] == "Q4_K_M"
        assert details["parameter_size"] == "7.6B"
        assert details["format"] == "gguf"

    def test_tous_les_champs_presents(self, registre):
        """Ollama ne marque pas `details` en `omitempty` : les clients lisent sans vérifier."""
        installer(registre)
        details = registre.get("qwen3:8b").details()
        for champ in ("parent_model", "format", "family", "families",
                      "parameter_size", "quantization_level"):
            assert champ in details

    def test_artefact_non_gguf_ne_casse_pas(self, registre):
        """Un artefact illisible laisse les champs dérivés vides, sans faire échouer le listing."""
        blob = registre.blobs.ingest_bytes(b"ceci n'est pas un gguf")
        registre.install(Manifest(name=parse("bizarre"), artifacts=Artifacts(model=blob.digest)))
        details = registre.get("bizarre").details()
        assert details["family"] == ""
        assert registre.get("bizarre").size == len(b"ceci n'est pas un gguf")


class TestResolution:
    def test_nom_court_resolu(self, registre):
        installer(registre, "qwen3:8b")
        assert registre.get("qwen3:8b").name == "qwen3:8b"

    def test_tag_implicite(self, registre):
        installer(registre, "qwen3")
        assert registre.get("qwen3").name == "qwen3:latest"
        assert registre.get("qwen3:latest").name == "qwen3:latest"

    def test_modele_absent(self, registre):
        with pytest.raises(ModelNotFound) as info:
            registre.get("fantome")
        assert "model 'fantome' not found" == info.value.message

    def test_nom_invalide_leve_model_not_found(self, registre):
        """Message contenant « model » : préserve la sonde d'`ollama-gateway` (risque R1)."""
        with pytest.raises(ModelNotFound) as info:
            registre.get("-invalide")
        assert "model" in info.value.message

    def test_nom_vide(self, registre):
        """Cas réel de la sonde d'`ollama-gateway`, qui envoie un corps `{}`."""
        with pytest.raises(ModelNotFound) as info:
            registre.get("")
        assert "model" in info.value.message

    def test_try_get_renvoie_none(self, registre):
        assert registre.try_get("fantome") is None

    def test_exists(self, registre):
        installer(registre)
        assert registre.exists("qwen3:8b")
        assert not registre.exists("fantome")
        assert not registre.exists("-invalide")


class TestListing:
    def test_listing_trie(self, registre):
        installer(registre, "zeta")
        installer(registre, "alpha")
        assert [m.name for m in registre.list()] == ["alpha:latest", "zeta:latest"]

    def test_listing_vide(self, registre):
        assert registre.list() == []

    def test_manifest_corrompu_ignore_sans_casser_le_listing(self, registre):
        """`/api/tags` est la sonde de disponibilité : un fichier corrompu ne doit pas
        faire passer le serveur pour hors ligne (criticité P0 de la matrice)."""
        installer(registre, "bon")
        chemin = registre.manifests.path(parse("casse"))
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text("{pas du json")
        assert [m.name for m in registre.list()] == ["bon:latest"]


class TestCompletude:
    def test_modele_complet(self, registre):
        installer(registre, avec_mmproj=True)
        modele = registre.get("qwen3:8b")
        assert modele.is_complete
        assert modele.missing_artifacts == ()

    def test_artefact_obligatoire_manquant_detecte(self, registre):
        """Mission §14 : un modèle multimodal sans `mmproj` ne doit jamais devenir READY."""
        installer(registre, avec_mmproj=True, mmproj_present=False)
        modele = registre.get("qwen3:8b")
        assert not modele.is_complete
        assert len(modele.missing_artifacts) == 1


class TestCopie:
    def test_copie_partage_les_artefacts(self, registre):
        """`ollama cp` ne duplique aucun octet : le stockage est adressé par contenu."""
        installer(registre, "source:latest")
        copie = registre.copy("source:latest", "destination:v2")
        assert copie.name == "destination:v2"
        assert copie.digest == registre.get("source:latest").digest
        assert len(list(registre.blobs.list())) == 1

    def test_copie_conserve_la_configuration(self, registre):
        from ollamacpp.storage import LifecycleConfig, RuntimeConfig

        blob = registre.blobs.ingest_bytes(build_gguf(DEFAULT_KV))
        registre.install(
            Manifest(
                name=parse("source"),
                artifacts=Artifacts(model=blob.digest),
                runtime=RuntimeConfig(context=8192, cache_type_k="iq4_nl"),
                lifecycle=LifecycleConfig(keep_alive="15m", priority=50),
                system="prompt système",
            )
        )
        copie = registre.copy("source", "cible")
        assert copie.manifest.runtime.context == 8192
        assert copie.manifest.runtime.cache_type_k == "iq4_nl"
        assert copie.manifest.lifecycle.priority == 50
        assert copie.manifest.system == "prompt système"

    def test_copie_dun_modele_absent(self, registre):
        with pytest.raises(ModelNotFound):
            registre.copy("fantome", "cible")


class TestSuppression:
    def test_suppression_retire_le_modele(self, registre):
        installer(registre)
        assert registre.delete("qwen3:8b") is True
        assert not registre.exists("qwen3:8b")

    def test_suppression_idempotente(self, registre):
        assert registre.delete("fantome") is False

    def test_suppression_libere_les_blobs_orphelins(self, registre):
        installer(registre)
        registre.delete("qwen3:8b")
        assert list(registre.blobs.list()) == []

    def test_suppression_conserve_les_blobs_encore_references(self, registre):
        """Un alias créé par `copy` doit survivre à la suppression de son original."""
        installer(registre, "source")
        registre.copy("source", "alias")
        registre.delete("source")
        assert registre.exists("alias")
        assert registre.get("alias").size > 0
        assert len(list(registre.blobs.list())) == 1

    def test_digests_references(self, registre):
        manifest = installer(registre)
        assert registre.referenced_digests() == {manifest.artifacts.model}


class TestInstallation:
    def test_date_posee_a_linstallation(self, registre):
        blob = registre.blobs.ingest_bytes(build_gguf(DEFAULT_KV))
        modele = registre.install(
            Manifest(name=parse("neuf"), artifacts=Artifacts(model=blob.digest))
        )
        assert modele.manifest.modified_at != ""

    def test_reinstallation_remplace(self, registre):
        installer(registre, "x")
        installer(registre, "x")
        assert len([m for m in registre.list() if m.name == "x:latest"]) == 1
