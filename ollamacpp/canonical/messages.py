"""Messages de la représentation conversationnelle canonique.

@spec docs/BACKLOG.md OC-014 « Représentation conversationnelle canonique »
@spec docs/ollama.cpp-architecture.md §5.4 « Modèle conversationnel canonique », §8 risque R7
@spec docs/DAT.md §1 « Composants » (paquet `ollamacpp/canonical/`)

Les quatre façades — Ollama `/api/chat`, OpenAI `/v1/chat/completions`, OpenAI `/v1/responses` et
Anthropic `/v1/messages` — convertissent leur requête **vers ces types**, et uniquement vers eux.
Aucune façade ne convertit vers une autre façade : c'est la règle qui interdit les chaînes de
conversion fragiles dénoncées par la mission (§22) et qui rend la cohérence inter-façades
testable (§35).

Invariants garantis par ces types et vérifiés par les tests d'OC-082 et OC-083 :

1. `ToolCall.id` traverse les conversions **inchangé** ;
2. un `function_call_output` (Responses) ou un `tool_result` (Messages) devient un
   `ToolResultMessage`, **jamais** un `UserMessage` — c'est la perte sémantique n° 1 des
   passerelles naïves, et elle casse les boucles d'agents ;
3. un `ReasoningBlock` n'est jamais fusionné dans le texte visible ;
4. l'ordre des clés de `ToolCall.arguments` est préservé (les `dict` Python conservent l'ordre
   d'insertion, comme l'`orderedmap` d'Ollama).
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field

from ..errors import BadRequest

_DATA_URI = re.compile(r"^data:(?P<media_type>[\w.+-]+/[\w.+-]+)?(?:;charset=[\w-]+)?;base64,(?P<data>.*)$",
                       re.DOTALL)

#: Signatures magiques utilisées pour déduire le type d'une image dont l'appelant ne le dit pas.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


# --- Blocs de contenu ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextBlock:
    """Texte visible par l'utilisateur."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningBlock:
    """Trace de raisonnement, distincte du texte visible.

    `signature` et `redacted` existent pour l'API Messages d'Anthropic, qui peut renvoyer un bloc
    de raisonnement signé ou expurgé. Les conserver évite de les perdre en aller-retour, même
    quand la façade d'origine ne les exploite pas.
    """

    text: str
    signature: str | None = None
    redacted: bool = False


@dataclass(frozen=True, slots=True)
class ImageInput:
    """Image d'entrée, normalisée en octets bruts + type MIME.

    Les trois façades expriment les images différemment — base64 nu (Ollama), data URI ou URL
    (OpenAI), bloc `source` typé (Anthropic). La normalisation a lieu à l'entrée, une fois, pour
    que le reste du système ne connaisse qu'une seule forme.
    """

    data: bytes
    media_type: str = "image/png"

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def to_data_uri(self) -> str:
        return f"data:{self.media_type};base64,{self.to_base64()}"

    @staticmethod
    def _detect_media_type(data: bytes, fallback: str = "image/png") -> str:
        for magic, media_type in _MAGIC:
            if data.startswith(magic):
                return media_type
        return fallback

    @classmethod
    def from_base64(cls, encoded: str, media_type: str | None = None) -> "ImageInput":
        """Décode une image base64, avec ou sans préfixe data URI.

        Une entrée non décodable lève `BadRequest` : une image illisible doit produire une erreur
        de requête explicite, pas une image vide passée silencieusement au modèle.
        """
        payload = encoded.strip()
        declared = media_type

        match = _DATA_URI.match(payload)
        if match:
            payload = match.group("data")
            declared = declared or match.group("media_type")

        # Le rembourrage `=` est parfois omis par les clients ; on le restitue avant décodage.
        padding = (-len(payload)) % 4
        try:
            data = base64.b64decode(payload + "=" * padding, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BadRequest("invalid image data") from exc

        if not data:
            raise BadRequest("invalid image data")

        return cls(data=data, media_type=declared or cls._detect_media_type(data))

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str | None = None) -> "ImageInput":
        if not data:
            raise BadRequest("invalid image data")
        return cls(data=data, media_type=media_type or cls._detect_media_type(data))


# --- Appels d'outils ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Appel d'outil émis par le modèle.

    `id` est l'identifiant de corrélation avec le `ToolResultMessage` correspondant. Il est
    **préservé tel quel** sur les quatre façades : `tool_call_id` côté Ollama et OpenAI Chat,
    `call_id` côté Responses, `id` du bloc `tool_use` côté Anthropic. Le perdre ou le régénérer
    casse toute boucle d'agent de plus d'un tour (risque R7).
    """

    id: str
    name: str
    arguments: dict[str, object] = field(default_factory=dict)
    index: int = 0


# --- Messages -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SystemMessage:
    role: str = field(default="system", init=False)
    text: str = ""


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: tuple[TextBlock | ImageInput, ...] = ()
    role: str = field(default="user", init=False)

    @property
    def text(self) -> str:
        """Concaténation des blocs textuels, pour les backends qui n'acceptent qu'une chaîne."""
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))

    @property
    def images(self) -> tuple[ImageInput, ...]:
        return tuple(block for block in self.content if isinstance(block, ImageInput))


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: tuple[TextBlock | ReasoningBlock, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    role: str = field(default="assistant", init=False)

    @property
    def text(self) -> str:
        """Texte visible **seul** : les blocs de raisonnement en sont exclus (invariant n° 3)."""
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))

    @property
    def reasoning(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, ReasoningBlock))


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """Résultat d'exécution d'un outil, renvoyé au modèle.

    Type distinct de `UserMessage` par construction : il est structurellement impossible de
    dégrader un résultat d'outil en message utilisateur (invariant n° 2).
    """

    call_id: str
    content: str = ""
    name: str = ""
    is_error: bool = False
    role: str = field(default="tool", init=False)


CanonicalMessage = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage
ContentBlock = TextBlock | ImageInput | ReasoningBlock


def link_tool_results(
    messages: tuple[CanonicalMessage, ...]
) -> tuple[CanonicalMessage, ...]:
    """Complète le nom d'outil d'un `ToolResultMessage` depuis l'appel qu'il référence.

    Les protocoles divergent sur ce point : Ollama transmet `tool_name` dans le message de
    résultat, OpenAI un `name` facultatif, et Anthropic **rien du tout** — son bloc `tool_result`
    ne porte que `tool_use_id`. L'information n'est pourtant pas perdue : elle est portée par
    l'appel corrélé, dans la même conversation.

    Reconstituer le nom ici rend la représentation canonique complète quelle que soit la façade
    d'entrée, ce qui est la condition pour que le même échange conceptuel produise la même
    représentation (mission §35). Un résultat sans appel correspondant est laissé tel quel plutôt
    que deviné.
    """
    names: dict[str, str] = {}
    for message in messages:
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if call.id:
                    names[call.id] = call.name

    linked: list[CanonicalMessage] = []
    for message in messages:
        if (
            isinstance(message, ToolResultMessage)
            and not message.name
            and message.call_id in names
        ):
            linked.append(
                ToolResultMessage(
                    call_id=message.call_id,
                    content=message.content,
                    name=names[message.call_id],
                    is_error=message.is_error,
                )
            )
        else:
            linked.append(message)
    return tuple(linked)
