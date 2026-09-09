"""Manifests de modèles : schéma, validation, lecture et écriture.

@spec docs/BACKLOG.md OC-021 « Manifests de modèles »
@spec docs/ollama.cpp-architecture.md §5.5 « Manifest de modèle », §5.6 « Précédence »,
      §5.10 « Arborescence de données », §8 risque R8
@spec docs/DAT.md §4.2 « Manifest », §4.3 « Précédence de configuration »

Un manifest décrit un **modèle logique** : d'où il vient, de quels artefacts il est composé,
comment il doit être lancé, et comment il doit résider en mémoire.

Deux règles structurent ce schéma.

**Les artefacts sont des digests, jamais des chemins.** Un manifest ne peut donc pas désigner un
fichier arbitraire du système, même s'il provient d'un registre distant hostile (risque R8). La
provenance est conservée séparément, dans `source`, à titre informatif.

**`capabilities_override` restreint, il n'élève jamais.** Une capacité ne peut être forcée qu'à
`false`. Forcer `vision: true` sur un modèle sans projecteur multimodal produirait exactement le
mensonge que la mission interdit (§13) : le modèle accepterait des images qu'il ne peut pas
traiter. La détection réelle est faite par OC-032 à partir de faits observables.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..durations import KeepAlive, parse_keep_alive
from ..names import ModelRef, parse, parse_from_filepath
from .blobs import is_valid_digest

SCHEMA_VERSION = 1

#: Capacités qu'un manifest peut restreindre. Miroir de l'énumération d'Ollama
#: (`types/model/capability.go`), à laquelle s'ajoute `reranking` que sert `llama-server`.
OVERRIDABLE_CAPABILITIES = frozenset(
    {"completion", "tools", "insert", "vision", "embedding", "thinking", "reranking"}
)


class ManifestError(ValueError):
    """Manifest malformé ou incohérent."""


@dataclass(frozen=True, slots=True)
class ModelSource:
    """Provenance du modèle. Informatif : ne sert jamais à construire un chemin d'écriture."""

    type: str = "file"  # file | huggingface | private
    reference: str = ""  # dépôt HF, nom dans le registre privé, ou chemin d'origine

    def to_json(self) -> dict[str, Any]:
        return {"type": self.type, "reference": self.reference}


@dataclass(frozen=True, slots=True)
class Artifacts:
    """Artefacts composant le modèle logique, désignés par digest."""

    model: str = ""
    mmproj: str | None = None
    draft: str | None = None
    adapters: tuple[str, ...] = ()
    template: str | None = None

    def all_digests(self) -> tuple[str, ...]:
        values = [self.model, self.mmproj, self.draft, self.template, *self.adapters]
        return tuple(value for value in values if value)

    def validate(self) -> None:
        if not self.model:
            raise ManifestError("artefact `model` obligatoire")
        for digest in self.all_digests():
            if not is_valid_digest(digest):
                raise ManifestError(f"artefact non désigné par un digest valide : {digest!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "mmproj": self.mmproj,
            "draft": self.draft,
            "adapters": list(self.adapters),
            "template": self.template,
        }


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Paramètres `llama-server` propres au modèle.

    C'est **la raison d'être du projet** : ne pas masquer les capacités de `llama.cpp` comme le
    fait Ollama (mission §16). Chaque champ correspond à un drapeau réel de `common/arg.cpp`,
    vérifié à l'audit ; la traduction en ligne de commande appartient à OC-031.

    `extra_args` est une soupape explicite pour les drapeaux non modélisés : elle évite qu'un
    besoin ponctuel impose une modification du schéma, tout en restant visible dans le manifest.
    """

    context: int | None = None            # --ctx-size
    batch: int | None = None              # --batch-size
    ubatch: int | None = None             # --ubatch-size
    parallel: int | None = None           # --parallel
    threads: int | None = None            # --threads
    threads_batch: int | None = None      # --threads-batch
    gpu_layers: int | None = None         # --n-gpu-layers
    tensor_split: str | None = None       # --tensor-split
    main_gpu: int | None = None           # --main-gpu
    flash_attention: bool | None = None   # --flash-attn
    cache_type_k: str | None = None       # --cache-type-k
    cache_type_v: str | None = None       # --cache-type-v
    kv_unified: bool | None = None        # --kv-unified
    kv_offload: bool | None = None        # --no-kv-offload quand False
    mmap: bool | None = None              # --no-mmap quand False
    mlock: bool | None = None             # --mlock
    numa: str | None = None               # --numa
    draft_max: int | None = None          # --draft-max
    draft_min: int | None = None          # --draft-min
    draft_p_min: float | None = None      # --draft-p-min
    embedding: bool | None = None         # --embedding
    reranking: bool | None = None         # --reranking
    pooling: str | None = None            # --pooling
    chat_template: str | None = None      # --chat-template
    reasoning_format: str | None = None   # --reasoning-format
    reasoning_budget: int | None = None   # --reasoning-budget
    extra_args: tuple[str, ...] = ()

    def merged_with(self, other: "RuntimeConfig") -> "RuntimeConfig":
        """Fusionne deux configurations, `other` prioritaire quand il précise une valeur."""
        changes: dict[str, Any] = {}
        for name in self.__slots__:
            if name == "extra_args":
                continue
            value = getattr(other, name)
            if value is not None:
                changes[name] = value
        if other.extra_args:
            changes["extra_args"] = (*self.extra_args, *other.extra_args)
        return replace(self, **changes)

    def to_json(self) -> dict[str, Any]:
        out = {name: getattr(self, name) for name in self.__slots__ if name != "extra_args"}
        out["extra_args"] = list(self.extra_args)
        return {key: value for key, value in out.items() if value not in (None, [])}


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    """Politique de résidence du modèle."""

    keep_alive: str | None = None
    priority: int = 0

    def resolved_keep_alive(self) -> KeepAlive | None:
        """Interprète `keep_alive` selon la sémantique Ollama, ou `None` si non précisé."""
        return None if self.keep_alive is None else parse_keep_alive(self.keep_alive)

    def to_json(self) -> dict[str, Any]:
        return {"keep_alive": self.keep_alive, "priority": self.priority}


@dataclass(frozen=True, slots=True)
class Manifest:
    """Manifest complet d'un modèle logique."""

    name: ModelRef
    artifacts: Artifacts
    schema_version: int = SCHEMA_VERSION
    source: ModelSource = field(default_factory=ModelSource)
    capabilities_override: dict[str, bool] = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)

    # Champs de présentation exposés par `/api/show`, alimentés par `/api/create`.
    system: str = ""
    template: str = ""
    license: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    modified_at: str = ""

    def validate(self) -> None:
        """Valide la cohérence structurelle. Appelée à l'écriture comme à la lecture."""
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"version de schéma non supportée : {self.schema_version} "
                f"(attendu {SCHEMA_VERSION})"
            )
        if not self.name.is_valid:
            raise ManifestError(f"nom de modèle invalide : {self.name!s}")
        self.artifacts.validate()

        for capability, value in self.capabilities_override.items():
            if capability not in OVERRIDABLE_CAPABILITIES:
                raise ManifestError(f"capacité inconnue : {capability!r}")
            if value is not False:
                # Restriction seulement : une capacité ne s'obtient que par observation.
                raise ManifestError(
                    f"capabilities_override ne peut que désactiver une capacité "
                    f"({capability!r} = {value!r})"
                )

        if self.lifecycle.keep_alive is not None:
            self.lifecycle.resolved_keep_alive()  # lève si la durée est illisible

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name.display_shortest(),
            "source": self.source.to_json(),
            "artifacts": self.artifacts.to_json(),
            "capabilities_override": dict(self.capabilities_override),
            "runtime": self.runtime.to_json(),
            "lifecycle": self.lifecycle.to_json(),
            "system": self.system,
            "template": self.template,
            "license": self.license,
            "parameters": dict(self.parameters),
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Manifest":
        """Reconstruit un manifest depuis sa forme JSON, en refusant les formes inattendues."""
        if not isinstance(payload, dict):
            raise ManifestError("manifest : objet JSON attendu")

        artifacts_raw = payload.get("artifacts") or {}
        if not isinstance(artifacts_raw, dict):
            raise ManifestError("manifest : `artifacts` doit être un objet")

        adapters_raw = artifacts_raw.get("adapters") or []
        if not isinstance(adapters_raw, list):
            raise ManifestError("manifest : `artifacts.adapters` doit être une liste")

        source_raw = payload.get("source") or {}
        runtime_raw = payload.get("runtime") or {}
        lifecycle_raw = payload.get("lifecycle") or {}
        if not all(isinstance(value, dict) for value in (source_raw, runtime_raw, lifecycle_raw)):
            raise ManifestError("manifest : `source`, `runtime` et `lifecycle` doivent être des objets")

        runtime_fields = set(RuntimeConfig.__slots__)
        runtime_kwargs = {
            key: value for key, value in runtime_raw.items()
            if key in runtime_fields and key != "extra_args"
        }
        extra_args = runtime_raw.get("extra_args") or []
        if not isinstance(extra_args, list):
            raise ManifestError("manifest : `runtime.extra_args` doit être une liste")

        manifest = cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            name=parse(str(payload.get("name", ""))),
            source=ModelSource(
                type=str(source_raw.get("type", "file")),
                reference=str(source_raw.get("reference", "")),
            ),
            artifacts=Artifacts(
                model=str(artifacts_raw.get("model") or ""),
                mmproj=artifacts_raw.get("mmproj") or None,
                draft=artifacts_raw.get("draft") or None,
                adapters=tuple(str(item) for item in adapters_raw),
                template=artifacts_raw.get("template") or None,
            ),
            capabilities_override=dict(payload.get("capabilities_override") or {}),
            runtime=RuntimeConfig(**runtime_kwargs, extra_args=tuple(str(a) for a in extra_args)),
            lifecycle=LifecycleConfig(
                keep_alive=lifecycle_raw.get("keep_alive"),
                priority=int(lifecycle_raw.get("priority", 0)),
            ),
            system=str(payload.get("system", "")),
            template=str(payload.get("template", "")),
            license=str(payload.get("license", "")),
            parameters=dict(payload.get("parameters") or {}),
            modified_at=str(payload.get("modified_at", "")),
        )
        manifest.validate()
        return manifest


class ManifestStore:
    """Persistance des manifests sous `manifests/<host>/<namespace>/<model>/<tag>`."""

    def __init__(self, manifests_dir: Path) -> None:
        self._root = Path(manifests_dir)

    @property
    def root(self) -> Path:
        return self._root

    def path(self, ref: ModelRef) -> Path:
        """Chemin du manifest. `ModelRef.filepath()` refuse toute référence non qualifiée (R8)."""
        return self._root / ref.filepath()

    def exists(self, ref: ModelRef) -> bool:
        try:
            return self.path(ref).is_file()
        except ValueError:
            return False

    def read(self, ref: ModelRef) -> Manifest:
        path = self.path(ref)
        if not path.is_file():
            raise ManifestError(f"manifest absent : {ref.display_shortest()}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ManifestError(f"manifest illisible : {ref.display_shortest()}") from exc
        return Manifest.from_json(payload)

    def write(self, manifest: Manifest) -> Path:
        """Écrit un manifest de façon atomique, après validation.

        Le remplacement par `os.replace` évite qu'une interruption laisse un manifest tronqué,
        qui rendrait le modèle définitivement illisible au redémarrage.
        """
        import os

        manifest.validate()
        path = self.path(manifest.name)
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = path.with_suffix(".tmp")
        payload = json.dumps(manifest.to_json(), ensure_ascii=False, indent=2, sort_keys=True)
        temp_path.write_text(payload + "\n", encoding="utf-8")
        os.replace(temp_path, path)
        return path

    def delete(self, ref: ModelRef) -> bool:
        """Supprime un manifest et les répertoires devenus vides. Idempotent."""
        path = self.path(ref)
        if not path.is_file():
            return False
        path.unlink()

        # Remonte tant que les répertoires parents sont vides, sans jamais dépasser la racine.
        parent = path.parent
        while parent != self._root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        return True

    def list_refs(self) -> list[ModelRef]:
        """Énumère les modèles installés, en ignorant les chemins qui ne sont pas des manifests."""
        if not self._root.is_dir():
            return []
        refs: list[ModelRef] = []
        for path in sorted(self._root.rglob("*")):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            ref = parse_from_filepath(str(path.relative_to(self._root).as_posix()))
            if ref.is_valid:
                refs.append(ref)
        return refs


def manifest_as_dict(manifest: Manifest) -> dict[str, Any]:
    """Vue brute du manifest, utile aux tests et au diagnostic."""
    return asdict(manifest)
