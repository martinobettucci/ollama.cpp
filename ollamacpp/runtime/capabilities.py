"""Détection de capacités à partir de faits observables.

@spec docs/BACKLOG.md OC-032 « Détection de capacités observables »
@spec docs/ollama.cpp-architecture.md §1.3 « /props : la source de vérité runtime »,
      §5.5 « Manifest de modèle »
@spec docs/DAT.md §5.2 « Interfaces consommées »

Exigence centrale de la mission (§13) : **ne jamais annoncer `vision=true` si le modèle ne peut
pas effectivement traiter une image.** Une capacité n'est donc jamais déclarée — elle est
*constatée*, à partir de trois sources de faits :

1. **`GET /props` de l'instance `llama-server`** — la source la plus fiable, car elle décrit le
   modèle réellement chargé :
   - `modalities.vision` / `modalities.audio` : le projecteur multimodal est chargé et
     opérationnel ;
   - `chat_template_caps.supports_tools` / `supports_tool_calls` : le template sait rendre des
     outils et relire des appels (`common/jinja/caps.h`) ;
   - `chat_template_caps.supports_reasoning_effort` / `supports_preserve_reasoning` : le
     template gère une trace de raisonnement.
2. **Les artefacts présents** — sans `mmproj` installé, la vision est impossible quoi qu'annonce
   le reste.
3. **Le manifest** — uniquement pour **retirer** une capacité (`capabilities_override`), jamais
   pour en ajouter une.

Le vocabulaire de sortie est celui d'Ollama (`types/model/capability.go`), pour que
`/api/show.capabilities` et `/api/tags` soient directement lisibles par les clients existants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..gguf import GGUFMetadata
from ..storage.manifests import Manifest

#: Vocabulaire d'Ollama (`types/model/capability.go`), plus `reranking` que sert `llama-server`
#: et qu'Ollama ne connaît pas. `reranking` n'est émis que si le modèle est lancé en mode rerank.
CAPABILITY_COMPLETION = "completion"
CAPABILITY_TOOLS = "tools"
CAPABILITY_INSERT = "insert"
CAPABILITY_VISION = "vision"
CAPABILITY_EMBEDDING = "embedding"
CAPABILITY_THINKING = "thinking"
CAPABILITY_RERANKING = "reranking"

#: Ordre d'émission stable, pour que deux réponses identiques soient identiques octet pour octet.
CAPABILITY_ORDER = (
    CAPABILITY_COMPLETION,
    CAPABILITY_TOOLS,
    CAPABILITY_INSERT,
    CAPABILITY_VISION,
    CAPABILITY_EMBEDDING,
    CAPABILITY_THINKING,
    CAPABILITY_RERANKING,
)


@dataclass(frozen=True, slots=True)
class DetectedCapabilities:
    """Capacités constatées d'un modèle, avec la trace de ce qui les a établies."""

    capabilities: frozenset[str] = frozenset()
    evidence: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.evidence is None:
            object.__setattr__(self, "evidence", {})

    def __contains__(self, capability: str) -> bool:
        return capability in self.capabilities

    def as_list(self) -> list[str]:
        """Liste ordonnée, format attendu par `/api/show.capabilities`."""
        return [name for name in CAPABILITY_ORDER if name in self.capabilities]


def detect(
    manifest: Manifest,
    *,
    props: dict[str, Any] | None = None,
    metadata: GGUFMetadata | None = None,
    has_mmproj: bool = False,
) -> DetectedCapabilities:
    """Constate les capacités d'un modèle.

    `props` est facultatif : avant tout chargement, `/api/tags` doit déjà pouvoir annoncer des
    capacités. On se rabat alors sur les artefacts et les métadonnées GGUF, qui sont des faits
    observables eux aussi — simplement moins précis que le modèle réellement chargé.
    """
    found: set[str] = set()
    evidence: dict[str, str] = {}
    props = props or {}

    runtime = manifest.runtime
    is_embedding_only = bool(runtime.embedding) or bool(runtime.reranking)

    # --- completion --------------------------------------------------------------------------
    # Un modèle lancé en mode embedding ou rerank ne génère pas de texte : `llama-server` refuse
    # les requêtes de complétion dans ce mode. Annoncer `completion` serait donc faux.
    if not is_embedding_only:
        found.add(CAPABILITY_COMPLETION)
        evidence[CAPABILITY_COMPLETION] = "mode de génération"

    # --- embedding / reranking ----------------------------------------------------------------
    if runtime.embedding:
        found.add(CAPABILITY_EMBEDDING)
        evidence[CAPABILITY_EMBEDDING] = "runtime.embedding"
    if runtime.reranking:
        found.add(CAPABILITY_RERANKING)
        evidence[CAPABILITY_RERANKING] = "runtime.reranking"

    # --- vision -------------------------------------------------------------------------------
    # Deux conditions cumulatives : un projecteur installé ET, si l'instance tourne, une modalité
    # image effectivement active. `/props` prime : c'est le modèle réellement chargé qui parle.
    modalities = props.get("modalities") or {}
    if isinstance(modalities, dict) and "vision" in modalities:
        if modalities.get("vision") and has_mmproj:
            found.add(CAPABILITY_VISION)
            evidence[CAPABILITY_VISION] = "props.modalities.vision"
    elif has_mmproj:
        found.add(CAPABILITY_VISION)
        evidence[CAPABILITY_VISION] = "artefact mmproj installé"

    # --- tools --------------------------------------------------------------------------------
    caps = props.get("chat_template_caps") or {}
    if isinstance(caps, dict) and caps:
        if caps.get("supports_tools") and caps.get("supports_tool_calls"):
            found.add(CAPABILITY_TOOLS)
            evidence[CAPABILITY_TOOLS] = "props.chat_template_caps.supports_tool_calls"
        if caps.get("supports_reasoning_effort") or caps.get("supports_preserve_reasoning"):
            found.add(CAPABILITY_THINKING)
            evidence[CAPABILITY_THINKING] = "props.chat_template_caps (raisonnement)"
    elif metadata is not None and metadata.chat_template:
        # Sans instance chargée, le chat template du GGUF reste un fait observable : un template
        # qui ne mentionne jamais les outils ne peut pas les rendre.
        template = metadata.chat_template
        if "tools" in template or "tool_calls" in template:
            found.add(CAPABILITY_TOOLS)
            evidence[CAPABILITY_TOOLS] = "chat template GGUF (mention des outils)"
        if "thinking" in template or "reasoning" in template:
            found.add(CAPABILITY_THINKING)
            evidence[CAPABILITY_THINKING] = "chat template GGUF (mention du raisonnement)"

    # --- insert (FIM) --------------------------------------------------------------------------
    # `llama-server` sert `/infill` si le modèle possède les tokens FIM ; ils sont visibles dans
    # les métadonnées du tokenizer.
    if metadata is not None and any(
        key.startswith("tokenizer.ggml.") and "fim" in key for key in metadata.kv
    ):
        found.add(CAPABILITY_INSERT)
        evidence[CAPABILITY_INSERT] = "tokens FIM présents dans le GGUF"

    # --- Restriction par le manifest -----------------------------------------------------------
    # `capabilities_override` ne peut que retirer (validé par `Manifest.validate`).
    for capability, enabled in manifest.capabilities_override.items():
        if enabled is False and capability in found:
            found.discard(capability)
            evidence[capability] = "désactivée par le manifest"

    return DetectedCapabilities(capabilities=frozenset(found), evidence=evidence)


def context_length(props: dict[str, Any] | None, metadata: GGUFMetadata | None,
                   fallback: int) -> int:
    """Contexte effectif d'un modèle.

    Priorité : le contexte **réellement alloué** par l'instance (`/props`), puis le contexte natif
    du GGUF, puis le défaut. Un `/api/ps` doit refléter ce qui est configuré, pas ce qui était
    souhaité.
    """
    if props:
        settings = props.get("default_generation_settings") or {}
        if isinstance(settings, dict):
            value = settings.get("n_ctx")
            if isinstance(value, int) and value > 0:
                return value
    if metadata is not None and metadata.context_length > 0:
        return metadata.context_length
    return fallback
