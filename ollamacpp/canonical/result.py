"""Résultat canonique : source unique de toutes les sérialisations de réponse.

@spec docs/BACKLOG.md OC-014 « Représentation conversationnelle canonique »
@spec docs/ollama.cpp-architecture.md §5.4 « Modèle conversationnel canonique », §8 risque R3
@spec docs/DAT.md §3.1 « Requête d'inférence »

Le backend produit un `CanonicalResult` (réponse complète) ou une suite de `CanonicalDelta`
(streaming). Chaque façade sérialise ensuite ce résultat à son format : NDJSON pour Ollama, SSE
pour OpenAI et Anthropic.

Les durées sont portées **en secondes** dans `Timings` et converties en nanosecondes au moment de
la seule sérialisation qui l'exige, celle d'Ollama. Garder des secondes en interne et convertir au
bord évite d'avoir deux unités qui circulent (risque R3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..durations import seconds_to_nanoseconds
from .messages import ReasoningBlock, TextBlock, ToolCall


class FinishReason(str, Enum):
    """Raison d'arrêt, canonique.

    Chaque façade a son vocabulaire — `done_reason` chez Ollama, `finish_reason` chez OpenAI,
    `stop_reason` chez Anthropic. La correspondance est faite une seule fois, à la sérialisation.
    """

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    STOP_SEQUENCE = "stop_sequence"
    LOAD = "load"
    UNLOAD = "unload"
    ERROR = "error"

    def to_ollama(self) -> str:
        """Vocabulaire Ollama : les appels d'outils ne sont pas une raison d'arrêt distincte."""
        if self in (FinishReason.TOOL_CALLS, FinishReason.STOP_SEQUENCE):
            return "stop"
        return self.value

    def to_openai(self) -> str:
        if self in (FinishReason.STOP_SEQUENCE, FinishReason.LOAD, FinishReason.UNLOAD):
            return "stop"
        if self is FinishReason.ERROR:
            return "stop"
        return self.value

    def to_anthropic(self) -> str:
        mapping = {
            FinishReason.STOP: "end_turn",
            FinishReason.LENGTH: "max_tokens",
            FinishReason.TOOL_CALLS: "tool_use",
            FinishReason.STOP_SEQUENCE: "stop_sequence",
        }
        return mapping.get(self, "end_turn")


@dataclass(frozen=True, slots=True)
class Usage:
    """Comptage de tokens, vocabulaire pivot Ollama."""

    prompt_eval_count: int = 0
    eval_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_eval_count + self.eval_count


@dataclass(frozen=True, slots=True)
class Timings:
    """Durées mesurées, **en secondes**. Converties en nanosecondes à la sérialisation Ollama."""

    total_s: float = 0.0
    load_s: float = 0.0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0

    def to_ollama_metrics(self, usage: Usage) -> dict[str, int]:
        """Produit les six champs `Metrics` d'Ollama, en nanosecondes (risque R3).

        Les champs sont `omitempty` côté Ollama : une valeur nulle est omise plutôt qu'émise à
        zéro, ce que reproduit ce filtrage.
        """
        raw = {
            "total_duration": seconds_to_nanoseconds(self.total_s),
            "load_duration": seconds_to_nanoseconds(self.load_s),
            "prompt_eval_count": usage.prompt_eval_count,
            "prompt_eval_duration": seconds_to_nanoseconds(self.prompt_eval_s),
            "eval_count": usage.eval_count,
            "eval_duration": seconds_to_nanoseconds(self.eval_s),
        }
        return {key: value for key, value in raw.items() if value}


@dataclass(frozen=True, slots=True)
class CanonicalResult:
    """Réponse complète d'une inférence."""

    model: str
    content: tuple[TextBlock | ReasoningBlock, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = field(default_factory=Usage)
    timings: Timings = field(default_factory=Timings)

    @property
    def text(self) -> str:
        """Texte visible seul : le raisonnement en est exclu (invariant n° 3 d'OC-014)."""
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))

    @property
    def reasoning(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, ReasoningBlock))


@dataclass(frozen=True, slots=True)
class CanonicalDelta:
    """Incrément de streaming.

    Le texte et le raisonnement sont deux champs distincts jusqu'au bout de la chaîne : les
    fusionner au niveau du delta rendrait impossible leur séparation par la façade, puisque le
    découpage en tokens ne respecte aucune frontière sémantique.
    """

    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    timings: Timings | None = None

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None
