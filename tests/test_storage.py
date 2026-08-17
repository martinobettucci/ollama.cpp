"""Tests du stockage : blobs adressés par contenu et manifests.

@verifies docs/BACKLOG.md OC-020 « BlobStore adressé par contenu », OC-021 « Manifests de modèles »
@verifies docs/ollama.cpp-architecture.md §5.5 « Manifest de modèle », §5.10 « Arborescence »,
          risque R8 « chemins issus de manifests distants »
@verifies docs/DAT.md §7 « Sécurité »
"""

from __future__ import annotations

import hashlib
import json

import pytest

from ollamacpp.names import parse
from ollamacpp.storage import (
    Artifacts,
    BlobError,
    BlobStore,
    LifecycleConfig,
    Manifest,
    ManifestError,
    ManifestStore,
    ModelSource,
    RuntimeConfig,
    compute_digest,
    is_valid_digest,
)
from ollamacpp.storage.blobs import digest_to_filename, filename_to_digest

CONTENU = b"des octets de modele"
DIGEST = "sha256:" + hashlib.sha256(CONTENU).hexdigest()


@pytest.fixture
def blobs(tmp_path) -> BlobStore:
    store = BlobStore(tmp_path / "blobs", tmp_path / "tmp")
    store.ensure_layout()
    return store


@pytest.fixture
def manifests(tmp_path) -> ManifestStore:
    return ManifestStore(tmp_path / "manifests")


# --- BlobStore ------------------------------------------------------------------------------------


class TestValidationDesDigests:
    """Risque R8 : un digest est la seule chose qui devienne un nom de fichier."""

    @pytest.mark.parametrize("digest", [DIGEST, "sha256:" + "0" * 64])
    def test_digests_valides(self, digest):
        assert is_valid_digest(digest)

    @pytest.mark.parametrize(
        "digest",
        [
            "sha256:xyz",                       # trop court
            "sha256:" + "A" * 64,               # majuscules refusées
            "md5:" + "0" * 32,                  # algorithme non supporté
            "sha256:../../etc/passwd",          # traversée de chemin
            "../../etc/passwd",
            "sha256:" + "0" * 63 + "/",         # séparateur de chemin
            "",
        ],
    )
    def test_digests_invalides(self, digest):
        assert not is_valid_digest(digest)

    def test_conversion_refuse_un_digest_invalide(self):
        with pytest.raises(BlobError):
            digest_to_filename("sha256:../../etc/passwd")

    def test_aller_retour_nom_de_fichier(self):
        assert filename_to_digest(digest_to_filename(DIGEST)) == DIGEST

    def test_chemin_reste_dans_le_repertoire(self, blobs):
        """Aucun digest valide ne peut produire un chemin hors du répertoire des blobs."""
        chemin = blobs.path(DIGEST)
        assert chemin.parent == blobs.blobs_dir


class TestIngestion:
    def test_ingestion_calcule_le_digest(self, blobs):
        info = blobs.ingest_bytes(CONTENU)
        assert info.digest == DIGEST
        assert info.size == len(CONTENU)

    def test_contenu_relisible(self, blobs):
        blobs.ingest_bytes(CONTENU)
        assert blobs.path(DIGEST).read_bytes() == CONTENU

    def test_checksum_attendu_verifie(self, blobs):
        assert blobs.ingest_bytes(CONTENU, expected_digest=DIGEST).digest == DIGEST

    def test_checksum_non_conforme_refuse_et_ne_laisse_rien(self, blobs):
        """Un artefact substitué ou corrompu ne doit jamais entrer dans le magasin."""
        faux = "sha256:" + "1" * 64
        with pytest.raises(BlobError, match="checksum"):
            blobs.ingest_bytes(CONTENU, expected_digest=faux)
        assert not blobs.has(faux)
        assert not blobs.has(DIGEST)
        assert list(blobs.blobs_dir.iterdir()) == []

    def test_aucun_fichier_partiel_ne_survit_a_un_refus(self, blobs, tmp_path):
        with pytest.raises(BlobError):
            blobs.ingest_bytes(CONTENU, expected_digest="sha256:" + "2" * 64)
        assert list((tmp_path / "tmp").iterdir()) == []

    def test_deduplication(self, blobs):
        """Deux ingestions du même contenu produisent un seul fichier."""
        blobs.ingest_bytes(CONTENU)
        blobs.ingest_bytes(CONTENU)
        assert len(list(blobs.list())) == 1

    def test_ingestion_par_blocs(self, blobs):
        info = blobs.ingest_chunks([b"des octets ", b"de modele"])
        assert info.digest == compute_digest_of(b"des octets de modele")

    def test_ingestion_dun_fichier(self, blobs, tmp_path):
        source = tmp_path / "source.bin"
        source.write_bytes(CONTENU)
        info = blobs.ingest_file(source)
        assert info.digest == DIGEST
        assert source.exists(), "le fichier source ne doit pas être consommé"

    def test_digest_attendu_malforme_refuse(self, blobs):
        with pytest.raises(BlobError):
            blobs.ingest_bytes(CONTENU, expected_digest="pas-un-digest")


def compute_digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class TestLectureEtEntretien:
    def test_absence(self, blobs):
        assert not blobs.has(DIGEST)

    def test_taille_dun_blob_absent(self, blobs):
        with pytest.raises(BlobError):
            blobs.size(DIGEST)

    def test_suppression_idempotente(self, blobs):
        blobs.ingest_bytes(CONTENU)
        assert blobs.delete(DIGEST) is True
        assert blobs.delete(DIGEST) is False

    def test_fichiers_etrangers_ignores(self, blobs):
        (blobs.blobs_dir / "README.txt").write_text("intrus")
        blobs.ingest_bytes(CONTENU)
        assert [b.digest for b in blobs.list()] == [DIGEST]

    def test_collecte_des_telechargements_interrompus(self, blobs, tmp_path):
        (tmp_path / "tmp" / "abc.partial").write_bytes(b"incomplet")
        assert blobs.sweep_tmp() == 1
        assert list((tmp_path / "tmp").iterdir()) == []

    def test_ramasse_miettes(self, blobs):
        blobs.ingest_bytes(CONTENU)
        autre = blobs.ingest_bytes(b"autre contenu")
        supprimes = blobs.collect_garbage(referenced={DIGEST})
        assert supprimes == [autre.digest]
        assert blobs.has(DIGEST)

    def test_ramasse_miettes_ne_touche_a_rien_si_tout_est_reference(self, blobs):
        blobs.ingest_bytes(CONTENU)
        assert blobs.collect_garbage(referenced={DIGEST}) == []


class TestCalculDeDigest:
    def test_sur_fichier(self, tmp_path):
        chemin = tmp_path / "f.bin"
        chemin.write_bytes(CONTENU)
        assert compute_digest(chemin) == DIGEST


# --- Manifests -------------------------------------------------------------------------------------


def manifest_minimal(nom: str = "test:latest", **kwargs) -> Manifest:
    return Manifest(name=parse(nom), artifacts=Artifacts(model=DIGEST), **kwargs)


class TestValidationDesManifests:
    def test_manifest_minimal_valide(self):
        manifest_minimal().validate()

    def test_artefact_modele_obligatoire(self):
        with pytest.raises(ManifestError, match="model"):
            Manifest(name=parse("test"), artifacts=Artifacts()).validate()

    @pytest.mark.parametrize(
        "artefact",
        ["../../etc/passwd", "/etc/passwd", "modele.gguf", "sha256:court"],
    )
    def test_artefact_doit_etre_un_digest(self, artefact):
        """Risque R8 : un manifest distant ne peut pas désigner un chemin arbitraire."""
        with pytest.raises(ManifestError, match="digest"):
            Manifest(name=parse("test"), artifacts=Artifacts(model=artefact)).validate()

    def test_mmproj_doit_aussi_etre_un_digest(self):
        with pytest.raises(ManifestError, match="digest"):
            Manifest(
                name=parse("test"),
                artifacts=Artifacts(model=DIGEST, mmproj="../../evil.gguf"),
            ).validate()

    def test_nom_invalide_refuse(self):
        with pytest.raises(ManifestError, match="nom"):
            Manifest(name=parse("-invalide"), artifacts=Artifacts(model=DIGEST)).validate()

    def test_version_de_schema_inconnue_refusee(self):
        with pytest.raises(ManifestError, match="schéma"):
            manifest_minimal(schema_version=99).validate()

    def test_keep_alive_illisible_refuse(self):
        from ollamacpp.durations import DurationError

        with pytest.raises(DurationError):
            manifest_minimal(lifecycle=LifecycleConfig(keep_alive="toujours")).validate()


class TestCapabilitiesOverride:
    """Le manifest peut restreindre une capacité, jamais l'accorder."""

    def test_desactivation_autorisee(self):
        manifest_minimal(capabilities_override={"vision": False}).validate()

    def test_activation_refusee(self):
        """Forcer `vision: true` produirait exactement le mensonge interdit par la mission §13."""
        with pytest.raises(ManifestError, match="désactiver"):
            manifest_minimal(capabilities_override={"vision": True}).validate()

    def test_capacite_inconnue_refusee(self):
        with pytest.raises(ManifestError, match="capacité inconnue"):
            manifest_minimal(capabilities_override={"telepathie": False}).validate()


class TestSerialisation:
    def test_aller_retour_json(self):
        original = manifest_minimal(
            "acme/qwen3:8b",
            source=ModelSource(type="huggingface", reference="acme/qwen3-gguf"),
            runtime=RuntimeConfig(context=262144, flash_attention=True,
                                  cache_type_k="iq4_nl", cache_type_v="q8_0", parallel=4),
            lifecycle=LifecycleConfig(keep_alive="15m", priority=100),
            system="tu es utile",
        )
        recharge = Manifest.from_json(json.loads(json.dumps(original.to_json())))
        assert recharge.name.display_shortest() == "acme/qwen3:8b"
        assert recharge.runtime.cache_type_k == "iq4_nl"
        assert recharge.runtime.context == 262144
        assert recharge.lifecycle.priority == 100
        assert recharge.system == "tu es utile"

    def test_valeurs_runtime_nulles_omises(self):
        """Un manifest lisible ne doit pas être noyé sous des dizaines de `null`."""
        payload = manifest_minimal(runtime=RuntimeConfig(context=4096)).to_json()
        assert payload["runtime"] == {"context": 4096}

    def test_json_invalide_refuse(self):
        with pytest.raises(ManifestError):
            Manifest.from_json({"artifacts": "pas un objet"})

    def test_champs_runtime_inconnus_ignores(self):
        """Un manifest écrit par une version ultérieure ne doit pas faire planter la lecture."""
        payload = manifest_minimal().to_json()
        payload["runtime"]["option_du_futur"] = 42
        assert Manifest.from_json(payload).runtime.context is None


class TestRuntimeConfig:
    def test_fusion_priorise_la_surcharge(self):
        socle = RuntimeConfig(context=4096, flash_attention=True)
        fusion = socle.merged_with(RuntimeConfig(context=8192))
        assert fusion.context == 8192
        assert fusion.flash_attention is True

    def test_extra_args_concatenes(self):
        fusion = RuntimeConfig(extra_args=("--a",)).merged_with(RuntimeConfig(extra_args=("--b",)))
        assert fusion.extra_args == ("--a", "--b")

    def test_false_explicite_surcharge(self):
        """`flash_attention=False` est une décision, pas une absence."""
        fusion = RuntimeConfig(flash_attention=True).merged_with(
            RuntimeConfig(flash_attention=False)
        )
        assert fusion.flash_attention is False


class TestManifestStore:
    def test_ecriture_et_lecture(self, manifests):
        manifests.write(manifest_minimal("acme/qwen3:8b"))
        relu = manifests.read(parse("acme/qwen3:8b"))
        assert relu.artifacts.model == DIGEST

    def test_chemin_canonique(self, manifests):
        chemin = manifests.path(parse("acme/qwen3:8b"))
        assert chemin.relative_to(manifests.root).as_posix() == (
            "registry.ollama.ai/acme/qwen3/8b"
        )

    def test_lecture_dun_manifest_absent(self, manifests):
        with pytest.raises(ManifestError, match="absent"):
            manifests.read(parse("fantome"))

    def test_manifest_corrompu(self, manifests):
        chemin = manifests.path(parse("casse"))
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text("{ceci n'est pas du json")
        with pytest.raises(ManifestError, match="illisible"):
            manifests.read(parse("casse"))

    def test_suppression_nettoie_les_repertoires_vides(self, manifests):
        manifests.write(manifest_minimal("acme/qwen3:8b"))
        assert manifests.delete(parse("acme/qwen3:8b")) is True
        assert not (manifests.root / "registry.ollama.ai" / "acme").exists()

    def test_suppression_idempotente(self, manifests):
        assert manifests.delete(parse("fantome")) is False

    def test_listing(self, manifests):
        manifests.write(manifest_minimal("a:latest"))
        manifests.write(manifest_minimal("acme/b:8b"))
        noms = {ref.display_shortest() for ref in manifests.list_refs()}
        assert noms == {"a:latest", "acme/b:8b"}

    def test_listing_sur_repertoire_absent(self, tmp_path):
        assert ManifestStore(tmp_path / "jamais-cree").list_refs() == []

    def test_fichiers_temporaires_ignores_par_le_listing(self, manifests):
        manifests.write(manifest_minimal("a:latest"))
        (manifests.root / "parasite.tmp").write_text("{}")
        assert len(manifests.list_refs()) == 1

    def test_ecriture_atomique_ne_laisse_pas_de_tmp(self, manifests):
        manifests.write(manifest_minimal("a:latest"))
        assert not list(manifests.root.rglob("*.tmp"))

    def test_manifest_invalide_refuse_a_lecriture(self, manifests):
        with pytest.raises(ManifestError):
            manifests.write(Manifest(name=parse("x"), artifacts=Artifacts(model="pas-un-digest")))
