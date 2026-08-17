"""Requête canonique : cible unique de conversion des quatre façades.

@spec docs/BACKLOG.md OC-014 « Représentation conversationnelle canonique »
@spec docs/ollama.cpp-architecture.md §5.4 « Modèle conversationnel canonique »
@spec docs/DAT.md §3.1 « Requête d'inférence »

Une `CanonicalRequest` décrit **ce qui est demandé**, indépendamment de la façade qui l'a exprimé.
Le champ `source_api` conserve l'origine — utile aux traces et à la sérialisation de la réponse —
mais il est explicitement exclu de la comparaison d'équivalence entre façades (OC-082).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from ..durations import KeepAlive
from ..names import ModelRef
from .messages import CanonicalMessage


class SourceAPI(str, Enum):
    """Façade dont provient la requête."""

    OLLAMA_CHAT = "ollama-chat"
    OLLAMA_GENERATE = "ollama-generate"
    OPENAI_CHAT = "openai-chat"
    OPENAI_COMPLETIONS = "openai-completions"
    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC_MESSAGES = "anthropic-messages"


class ResponseFormatKind(str, Enum):
    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"


class ToolChoiceMode(str, Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    NAMED = "named"


class ThinkingLevel(str, Enum):
    """Niveaux d'effort de raisonnement acceptés par Ollama (`ThinkValue.IsValid`)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolChoice:
    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    name: str = ""


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    kind: ResponseFormatKind = ResponseFormatKind.TEXT
    schema: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ThinkingRequest:
    """Demande de raisonnement.

    `enabled is None` signifie « non précisé » : le comportement par défaut du modèle et de son
    template s'applique. C'est la distinction qu'Ollama matérialise par un pointeur `*ThinkValue`
    et qu'il faut conserver — `false` explicite et absence n'ont pas le même sens.
    """

    enabled: bool | None = None
    level: ThinkingLevel | None = None

    @property
    def is_requested(self) -> bool:
        return bool(self.enabled) or self.level is not None


@dataclass(frozen=True, slots=True)
class SamplingOptions:
    """Options de génération, normalisées sous les noms Ollama.

    Les noms Ollama servent de vocabulaire pivot parce que la façade Ollama est la plus riche des
    quatre en paramètres de runtime (`num_ctx`, `num_batch`, `num_gpu`…) et qu'elle est la seule
    dont `ollama-gateway` réécrit le corps.

    `extra` conserve les options non reconnues plutôt que de les jeter : une option ajoutée par
    une version ultérieure d'Ollama traverse ainsi le middleware sans le casser, tout en restant
    inspectable.
    """

    num_ctx: int | None = None
    num_predict: int | None = None
    num_keep: int | None = None
    num_batch: int | None = None
    num_gpu: int | None = None
    num_thread: int | None = None
    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    typical_p: float | None = None
    repeat_last_n: int | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def merged_with(self, other: "SamplingOptions") -> "SamplingOptions":
        """Fusionne deux jeux d'options, `other` étant prioritaire quand il précise une valeur.

        Sert la règle de précédence `requête > manifest > GGUF > défauts` : le manifest fournit un
        socle, la requête le surcharge.
        """
        changes: dict[str, Any] = {}
        for name in self.__slots__:
            if name in {"stop", "extra"}:
                continue
            value = getattr(other, name)
            if value is not None:
                changes[name] = value
        if other.stop:
            changes["stop"] = other.stop
        if other.extra:
            changes["extra"] = {**self.extra, **other.extra}
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class CanonicalRequest:
    """Requête d'inférence, indépendante de la façade d'origine."""

    model: ModelRef
    messages: tuple[CanonicalMessage, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice = field(default_factory=ToolChoice)
    options: SamplingOptions = field(default_factory=SamplingOptions)
    response_format: ResponseFormat = field(default_factory=ResponseFormat)
    thinking: ThinkingRequest = field(default_factory=ThinkingRequest)
    stream: bool = False
    keep_alive: KeepAlive | None = None
    source_api: SourceAPI = SourceAPI.OLLAMA_CHAT

    #: Prompt brut de `/api/generate` avec `raw: true` : aucun template n'est appliqué.
    raw_prompt: str | None = None

    def equivalence_key(self) -> tuple:
        """Projection utilisée par les tests d'équivalence inter-façades (OC-082).

        `source_api`, `stream` et `keep_alive` en sont exclus : ce sont des propriétés du
        transport, pas de l'échange conceptuel.
        """
        return (
            str(self.model),
            self.messages,
            self.tools,
            self.tool_choice,
            self.options,
            self.response_format,
            self.thinking,
            self.raw_prompt,
        )
