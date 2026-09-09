"""Analyse des requêtes Ollama vers la représentation canonique.

@spec docs/BACKLOG.md OC-044 « /api/chat », OC-045 « /api/generate », OC-047 « options.num_ctx »
@spec docs/ollama.cpp-architecture.md §2.4 « Schémas de requêtes et de réponses »,
      §5.4 « Modèle conversationnel canonique », §8 risque R2
@spec docs/DAT.md §3.1 « Requête d'inférence »

Les schémas reproduits ici viennent de `api/types.go` d'Ollama (révision auditée `d67ad83`) :
`GenerateRequest` l. 62-129, `ChatRequest` l. 133-179, `Message` l. 197-207.

**Validation à la main, comme Ollama.** Aucun modèle Pydantic n'est utilisé pour le corps des
requêtes : Ollama valide manuellement et répond `400 {"error": ...}`, jamais le `422
{"detail": ...}` de FastAPI. Accepter un `dict` brut permet aussi de laisser passer les champs
inconnus au lieu de les rejeter, ce qui évite qu'un client d'une version ultérieure d'Ollama soit
cassé par le middleware (risque R1).

**`options.num_ctx` est un contrat dur** : `ollama-gateway` réécrit le corps des requêtes pour y
forcer le plafond de contexte de la clé (`app/context.py` l. 198-222). Le refuser casserait
toutes les clés à plafond (risque R2).
"""

from __future__ import annotations

from typing import Any

from ..canonical import (
    AssistantMessage,
    CanonicalMessage,
    CanonicalRequest,
    ImageInput,
    ReasoningBlock,
    ResponseFormat,
    ResponseFormatKind,
    SamplingOptions,
    SourceAPI,
    SystemMessage,
    TextBlock,
    ThinkingLevel,
    ThinkingRequest,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
    link_tool_results,
)
from ..durations import DurationError, KeepAlive, parse_keep_alive
from ..errors import BadRequest
from ..names import ModelRef

#: Champs d'`api.Options` (`api/types.go` l. 568-598) portés explicitement par `SamplingOptions`.
_KNOWN_OPTIONS = {
    "num_ctx", "num_predict", "num_keep", "num_batch", "num_gpu", "num_thread",
    "seed", "temperature", "top_k", "top_p", "min_p", "typical_p",
    "repeat_last_n", "repeat_penalty", "presence_penalty", "frequency_penalty", "stop",
}

#: Options qui configurent le **runtime** de l'instance, pas l'échantillonnage. Elles ne sont pas
#: transmises comme paramètres de génération : `num_ctx` en particulier doit devenir `--ctx-size`.
RUNTIME_OPTIONS = {"num_ctx", "num_batch", "num_gpu", "num_thread"}


def require_model(body: dict[str, Any]) -> str:
    """Extrait le nom du modèle, en acceptant l'alias déprécié `name`.

    Ollama accepte encore `name` sur plusieurs endpoints (`DeleteRequest.Name`,
    `ShowRequest.Name`, `PullRequest.Name`), marqué déprécié mais toujours honoré.
    """
    for key in ("model", "name"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_options(raw: Any) -> SamplingOptions:
    """Convertit le champ `options` d'Ollama en options canoniques.

    Les clés inconnues sont conservées dans `extra` plutôt que rejetées : une option ajoutée par
    une version ultérieure d'Ollama doit traverser le middleware sans le casser.
    """
    if raw is None:
        return SamplingOptions()
    if not isinstance(raw, dict):
        raise BadRequest("options must be an object")

    known: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "stop":
            if isinstance(value, str):
                known["stop"] = (value,)
            elif isinstance(value, list):
                known["stop"] = tuple(str(item) for item in value)
            continue
        if key in _KNOWN_OPTIONS:
            known[key] = value
        elif key not in RUNTIME_OPTIONS:
            extra[key] = value

    return SamplingOptions(**known, extra=extra)


def parse_format(raw: Any) -> ResponseFormat:
    """Convertit le champ `format` d'Ollama.

    Ollama accepte `"json"` ou un schéma JSON complet (`docs/api.md`, « Structured outputs »).
    """
    if raw is None or raw == "":
        return ResponseFormat()
    if isinstance(raw, str):
        if raw == "json":
            return ResponseFormat(kind=ResponseFormatKind.JSON)
        raise BadRequest(f'invalid format: "{raw}"')
    if isinstance(raw, dict):
        return ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA, schema=raw)
    raise BadRequest("format must be a string or a JSON schema object")


def parse_think(raw: Any) -> ThinkingRequest:
    """Convertit le champ `think` d'Ollama.

    `ThinkValue` accepte un booléen ou l'un de `low`, `medium`, `high`, `max`
    (`api/types.go` l. 1126-1145). L'absence du champ n'est **pas** équivalente à `false` : elle
    laisse le comportement par défaut du modèle, distinction qu'Ollama matérialise par un
    pointeur.
    """
    if raw is None:
        return ThinkingRequest()
    if isinstance(raw, bool):
        return ThinkingRequest(enabled=raw)
    if isinstance(raw, str):
        try:
            return ThinkingRequest(enabled=True, level=ThinkingLevel(raw))
        except ValueError:
            raise BadRequest(f'invalid think value: "{raw}"') from None
    raise BadRequest("think must be a boolean or one of: low, medium, high, max")


def parse_keep_alive_field(raw: Any) -> KeepAlive | None:
    """Interprète `keep_alive`, en convertissant une erreur de durée en 400."""
    if raw is None:
        return None
    try:
        return parse_keep_alive(raw)
    except DurationError as exc:
        raise BadRequest(str(exc)) from exc


def parse_images(raw: Any) -> tuple[ImageInput, ...]:
    """Convertit la liste d'images base64 d'Ollama."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise BadRequest("images must be an array of base64-encoded strings")
    return tuple(
        ImageInput.from_base64(item) if isinstance(item, str) else _reject_image()
        for item in raw
    )


def _reject_image() -> ImageInput:
    raise BadRequest("images must be an array of base64-encoded strings")


def parse_tools(raw: Any) -> tuple[ToolDefinition, ...]:
    """Convertit la liste d'outils d'Ollama (même forme que celle d'OpenAI)."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise BadRequest("tools must be an array")

    tools: list[ToolDefinition] = []
    for item in raw:
        if not isinstance(item, dict):
            raise BadRequest("each tool must be an object")
        function = item.get("function")
        if not isinstance(function, dict):
            raise BadRequest("each tool must have a function object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise BadRequest("each tool function must have a name")
        parameters = function.get("parameters")
        tools.append(
            ToolDefinition(
                name=name,
                description=str(function.get("description") or ""),
                parameters=parameters if isinstance(parameters, dict) else {},
            )
        )
    return tuple(tools)


def parse_message(raw: Any) -> CanonicalMessage:
    """Convertit un `api.Message` d'Ollama en message canonique.

    Le rôle est normalisé en minuscules, comme le fait `Message.UnmarshalJSON`
    (`api/types.go` l. 209-219). Un message de rôle `tool` devient un `ToolResultMessage` — le
    type qui rend structurellement impossible sa dégradation en message utilisateur (risque R7).
    """
    if not isinstance(raw, dict):
        raise BadRequest("each message must be an object")

    role = str(raw.get("role") or "").lower()
    content = raw.get("content")
    if content is not None and not isinstance(content, str):
        raise BadRequest("message content must be a string")
    text = content or ""

    if role == "system":
        return SystemMessage(text=text)

    if role == "tool":
        return ToolResultMessage(
            call_id=str(raw.get("tool_call_id") or ""),
            name=str(raw.get("tool_name") or ""),
            content=text,
        )

    if role == "assistant":
        blocks: list[TextBlock | ReasoningBlock] = []
        thinking = raw.get("thinking")
        if isinstance(thinking, str) and thinking:
            blocks.append(ReasoningBlock(text=thinking))
        if text:
            blocks.append(TextBlock(text=text))
        return AssistantMessage(
            content=tuple(blocks), tool_calls=_parse_tool_calls(raw.get("tool_calls"))
        )

    if role in ("user", ""):
        parts: list[TextBlock | ImageInput] = []
        if text:
            parts.append(TextBlock(text=text))
        parts.extend(parse_images(raw.get("images")))
        return UserMessage(content=tuple(parts))

    raise BadRequest(f'invalid role: "{role}"')


def _parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        arguments = function.get("arguments")
        calls.append(
            ToolCall(
                id=str(item.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
                index=int(function.get("index", index)),
            )
        )
    return tuple(calls)


def parse_chat_request(body: dict[str, Any], ref: ModelRef) -> CanonicalRequest:
    """Convertit `POST /api/chat` en requête canonique."""
    raw_messages = body.get("messages")
    if raw_messages is None:
        raw_messages = []
    if not isinstance(raw_messages, list):
        raise BadRequest("messages must be an array")

    return CanonicalRequest(
        model=ref,
        messages=link_tool_results(tuple(parse_message(item) for item in raw_messages)),
        tools=parse_tools(body.get("tools")),
        options=parse_options(body.get("options")),
        response_format=parse_format(body.get("format")),
        thinking=parse_think(body.get("think")),
        # Ollama streame par défaut : `Stream *bool` non renseigné vaut `true`.
        stream=_parse_stream(body.get("stream")),
        keep_alive=parse_keep_alive_field(body.get("keep_alive")),
        source_api=SourceAPI.OLLAMA_CHAT,
    )


def parse_generate_request(body: dict[str, Any], ref: ModelRef) -> CanonicalRequest:
    """Convertit `POST /api/generate` en requête canonique.

    `raw: true` court-circuite le template : le prompt part tel quel. Ollama refuse alors
    `system`, `template` et `context` (`server/routes.go` l. 428) — refus reproduit ici.
    """
    prompt = body.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise BadRequest("prompt must be a string")

    system = body.get("system")
    template = body.get("template")
    raw_mode = bool(body.get("raw"))

    if raw_mode and (system or template or body.get("context")):
        raise BadRequest("raw mode does not support template, system, or context")

    messages: list[CanonicalMessage] = []
    if isinstance(system, str) and system:
        messages.append(SystemMessage(text=system))

    parts: list[TextBlock | ImageInput] = []
    if prompt:
        parts.append(TextBlock(text=prompt))
    parts.extend(parse_images(body.get("images")))
    if parts:
        messages.append(UserMessage(content=tuple(parts)))

    return CanonicalRequest(
        model=ref,
        messages=tuple(messages),
        options=parse_options(body.get("options")),
        response_format=parse_format(body.get("format")),
        thinking=parse_think(body.get("think")),
        stream=_parse_stream(body.get("stream")),
        keep_alive=parse_keep_alive_field(body.get("keep_alive")),
        source_api=SourceAPI.OLLAMA_GENERATE,
        raw_prompt=prompt if raw_mode else None,
    )


def _parse_stream(raw: Any) -> bool:
    """`stream` vaut `true` par défaut chez Ollama (`Stream *bool`, absent = streaming)."""
    if raw is None:
        return True
    if not isinstance(raw, bool):
        raise BadRequest("stream must be a boolean")
    return raw


def runtime_overrides(raw_options: Any) -> dict[str, Any]:
    """Extrait les options qui pilotent le **runtime** de l'instance plutôt que la génération.

    `num_ctx` est le cas critique : `ollama-gateway` l'injecte pour plafonner le contexte d'une
    clé (risque R2). Il doit devenir `--ctx-size` sur l'instance, pas un paramètre
    d'échantillonnage — sinon le plafond serait silencieusement sans effet.
    """
    if not isinstance(raw_options, dict):
        return {}
    out: dict[str, Any] = {}
    mapping = {
        "num_ctx": "context",
        "num_batch": "batch",
        "num_gpu": "gpu_layers",
        "num_thread": "threads",
    }
    for source, target in mapping.items():
        value = raw_options.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            out[target] = value
    return out
