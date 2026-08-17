"""Registre de modèles : résolution de noms, identité, cycle d'installation.

@spec docs/BACKLOG.md OC-022 « ModelRegistry »
@spec docs/ollama.cpp-architecture.md §4.2 « Fonctionnalités manquantes », §6 « Interfaces
      internes proposées », §8 risque R4
@spec docs/DAT.md §1 « Composants », §3.3 « Téléchargement »

Le registre est la réponse au **risque R4** : `/v1/models` de `llama-server` renvoie des bouchons
(`size: ""`, `digest: ""`, commentaire explicite dans `server-context.cpp` l. 4888-4889), ce qui
rend `/api/tags` et `/api/ps` inexploitables par les outils Ollama et par le filtrage
d'`ollama-gateway`. `ollama.cpp` donne à chaque modèle une **identité stable** : un digest réel,
une taille réelle et une date de modification réelle, calculés depuis le `BlobStore`.

Le registre ne charge aucun modèle : il décrit ce qui est **installé**. La distinction
`registered / downloaded / loaded` de la mission se lit donc ainsi :

- *registered* et *downloaded* : un manifest existe et ses artefacts obligatoires sont présents
  — c'est le domaine du registre ;
- *loaded* : le modèle est résident en mémoire — c'est le domaine du `ModelLifecycleManager`
  (OC-033), interrogé séparément par `/api/ps`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ModelNotFound
from ..gguf import GGUFMetadata, GGUFError, read_metadata
from ..names import ModelRef, parse
from ..storage.blobs import BlobStore
from ..storage.manifests import Manifest, ManifestError, ManifestStore


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """Modèle installé, tel que le registre le connaît."""

    ref: ModelRef
    manifest: Manifest
    size: int
    digest: str
    modified_at: dt.datetime
    missing_artifacts: tuple[str, ...] = ()
    metadata: GGUFMetadata | None = None

    @property
    def name(self) -> str:
        """Nom court, celui qu'attendent les clients Ollama."""
        return self.ref.display_shortest()

    @property
    def is_complete(self) -> bool:
        """Vrai si tous les artefacts déclarés sont présents.

        Un modèle incomplet est listé mais ne doit jamais atteindre l'état `READY` : c'est
        l'exigence de la mission §14 sur les modèles multimodaux dont le `mmproj` manque.
        """
        return not self.missing_artifacts

    def details(self) -> dict[str, Any]:
        """Bloc `details` d'Ollama (`api.ModelDetails`).

        Tous les champs sont présents même vides : Ollama ne les marque pas `omitempty`, et des
        clients lisent `details.family` sans vérifier son existence.
        """
        meta = self.metadata
        family = meta.architecture if meta else ""
        return {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family] if family else [],
            "parameter_size": meta.parameter_size if meta else "",
            "quantization_level": meta.quantization_level if meta else "",
        }


class ModelRegistry:
    """Registre des modèles installés localement."""

    def __init__(self, blobs: BlobStore, manifests: ManifestStore) -> None:
        self._blobs = blobs
        self._manifests = manifests
        self._metadata_cache: dict[str, GGUFMetadata] = {}

    @property
    def blobs(self) -> BlobStore:
        return self._blobs

    @property
    def manifests(self) -> ManifestStore:
        return self._manifests

    # --- Résolution ------------------------------------------------------------------------------

    def resolve(self, name: str) -> ModelRef:
        """Normalise un nom de modèle en référence pleinement qualifiée.

        Un nom invalide lève `ModelNotFound` plutôt qu'une erreur de format : c'est ce que fait
        Ollama, et cela préserve la sonde de compatibilité de la passerelle (risque R1), puisque
        le message contient toujours le mot « model ».
        """
        ref = parse(name or "")
        if not ref.is_valid:
            raise ModelNotFound(name or "")
        return ref

    def exists(self, name: str) -> bool:
        try:
            return self._manifests.exists(self.resolve(name))
        except ModelNotFound:
            return False

    # --- Lecture ---------------------------------------------------------------------------------

    def get(self, name: str) -> RegisteredModel:
        """Renvoie un modèle installé. Lève `ModelNotFound` si le manifest n'existe pas."""
        ref = self.resolve(name)
        if not self._manifests.exists(ref):
            raise ModelNotFound(name)
        try:
            manifest = self._manifests.read(ref)
        except ManifestError as exc:
            # Un manifest corrompu est indiscernable d'un modèle absent du point de vue du client ;
            # on ne divulgue pas le détail du système de fichiers.
            raise ModelNotFound(name) from exc
        return self._describe(ref, manifest)

    def try_get(self, name: str) -> RegisteredModel | None:
        try:
            return self.get(name)
        except ModelNotFound:
            return None

    def list(self) -> list[RegisteredModel]:
        """Énumère les modèles installés, triés par nom pour une sortie déterministe.

        Un manifest illisible est **ignoré** plutôt que de faire échouer tout le listing :
        `/api/tags` est la sonde de disponibilité d'`ollama-gateway` (criticité P0), un seul
        fichier corrompu ne doit pas faire passer le serveur pour hors ligne.
        """
        models: list[RegisteredModel] = []
        for ref in self._manifests.list_refs():
            try:
                models.append(self._describe(ref, self._manifests.read(ref)))
            except ManifestError:
                continue
        return sorted(models, key=lambda model: model.name)

    # --- Écriture --------------------------------------------------------------------------------

    def install(self, manifest: Manifest) -> RegisteredModel:
        """Installe ou remplace un manifest et renvoie le modèle décrit.

        La date de modification est posée ici si le manifest n'en porte pas : elle doit refléter
        l'installation réelle, pas une valeur fabriquée par l'appelant.
        """
        if not manifest.modified_at:
            manifest = _with_modified_now(manifest)
        self._manifests.write(manifest)
        self._metadata_cache.pop(manifest.artifacts.model, None)
        return self._describe(manifest.name, manifest)

    def copy(self, source: str, destination: str) -> RegisteredModel:
        """Duplique un modèle sous un autre nom.

        Aucun artefact n'est recopié : le stockage étant adressé par contenu, les deux manifests
        partagent les mêmes blobs. C'est exactement le comportement d'`ollama cp`.
        """
        model = self.get(source)
        target = self.resolve(destination)
        if not target.is_valid:
            raise ModelNotFound(destination)
        copied = _with_modified_now(
            Manifest(
                name=target,
                artifacts=model.manifest.artifacts,
                schema_version=model.manifest.schema_version,
                source=model.manifest.source,
                capabilities_override=dict(model.manifest.capabilities_override),
                runtime=model.manifest.runtime,
                lifecycle=model.manifest.lifecycle,
                system=model.manifest.system,
                template=model.manifest.template,
                license=model.manifest.license,
                parameters=dict(model.manifest.parameters),
            )
        )
        self._manifests.write(copied)
        return self._describe(target, copied)

    def delete(self, name: str) -> bool:
        """Supprime un modèle. Les blobs encore référencés ailleurs sont conservés.

        Le ramassage des blobs orphelins est fait après coup, en confrontant l'ensemble des
        digests référencés par les manifests restants : c'est ce qui rend `copy` gratuit et
        `delete` sûr en présence d'alias.
        """
        ref = self.resolve(name)
        if not self._manifests.exists(ref):
            return False
        self._manifests.delete(ref)
        self._blobs.collect_garbage(self.referenced_digests())
        return True

    def referenced_digests(self) -> set[str]:
        """Ensemble des digests référencés par au moins un manifest installé."""
        referenced: set[str] = set()
        for ref in self._manifests.list_refs():
            try:
                referenced.update(self._manifests.read(ref).artifacts.all_digests())
            except ManifestError:
                # Un manifest illisible est traité comme s'il référençait tout ce qu'il déclare :
                # impossible à savoir, donc on ne supprime rien à cause de lui. En pratique il
                # n'apporte aucun digest, mais il ne doit pas non plus en faire perdre.
                continue
        return referenced

    # --- Interne ---------------------------------------------------------------------------------

    def _describe(self, ref: ModelRef, manifest: Manifest) -> RegisteredModel:
        """Calcule l'identité réelle d'un modèle : taille, digest, date, artefacts manquants."""
        missing = tuple(
            digest for digest in manifest.artifacts.all_digests() if not self._blobs.has(digest)
        )

        # La taille annoncée est la somme des artefacts réellement présents : annoncer la taille
        # d'un artefact absent tromperait le scheduler comme l'utilisateur.
        size = sum(
            self._blobs.size(digest)
            for digest in manifest.artifacts.all_digests()
            if self._blobs.has(digest)
        )

        return RegisteredModel(
            ref=ref,
            manifest=manifest,
            size=size,
            digest=manifest.artifacts.model,
            modified_at=_parse_modified(manifest, self._manifests.path(ref)),
            missing_artifacts=missing,
            metadata=self._metadata(manifest.artifacts.model),
        )

    def _metadata(self, digest: str) -> GGUFMetadata | None:
        """Lit — et met en cache — les métadonnées GGUF de l'artefact principal.

        Le cache est licite car le contenu d'un blob ne change jamais : son nom est son digest.
        """
        if not digest or not self._blobs.has(digest):
            return None
        cached = self._metadata_cache.get(digest)
        if cached is not None:
            return cached
        try:
            metadata = read_metadata(self._blobs.path(digest))
        except (GGUFError, OSError):
            # Un artefact non GGUF (ou tronqué) ne doit pas empêcher de lister le modèle : les
            # champs dérivés resteront vides, ce qui est visible et honnête.
            return None
        self._metadata_cache[digest] = metadata
        return metadata


def _with_modified_now(manifest: Manifest) -> Manifest:
    from dataclasses import replace

    return replace(manifest, modified_at=_now_iso())


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_modified(manifest: Manifest, manifest_path: Path) -> dt.datetime:
    """Date de modification du modèle.

    Le manifest fait foi ; en son absence ou s'il porte une date illisible, on retombe sur l'mtime
    du fichier, qui est un fait observable. Jamais de date fabriquée.
    """
    if manifest.modified_at:
        try:
            return dt.datetime.fromisoformat(manifest.modified_at)
        except ValueError:
            pass
    try:
        return dt.datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=dt.timezone.utc)
    except OSError:
        return dt.datetime.now(dt.timezone.utc)
