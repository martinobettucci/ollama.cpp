"""Façade Anthropic Messages (`/v1/messages`).

@spec docs/BACKLOG.md OC-075 « /v1/messages », OC-076 « /v1/messages/count_tokens »
@spec docs/ollama.cpp-architecture.md §5.2, §5.4 « Modèle conversationnel canonique »,
      §8 risque R7
@spec docs/DAT.md §5.1 « Interfaces exposées »

Cette façade convertit **directement** vers la représentation canonique : jamais
`Messages → OpenAI → Ollama`, la chaîne fragile que la mission interdit explicitement (§22).
Elle emprunte donc exactement un aller et un retour, comme les trois autres.

Points de fidélité au protocole Anthropic :

- le `system` est un champ de premier niveau, pas un message ;
- le contenu est une liste de **blocs typés** (`text`, `image`, `tool_use`, `tool_result`,
  `thinking`) ;
- un bloc `tool_result` porte `tool_use_id` — il devient un `ToolResultMessage`, jamais un
  message `user` (risque R7) ;
- `max_tokens` est **obligatoire** ;
- le streaming est une séquence d'événements nommés (`message_start`, `content_block_delta`…),
  et non une suite de chunks homogènes.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .. import backend
from ..canonical import (
    AssistantMessage,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResult,
    FinishReason,
    ImageInput,
    ReasoningBlock,
    SamplingOptions,
    SourceAPI,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
    link_tool_results,
)
from ..errors import BadRequest
from .common import get_service, read_body, reject_unsupported

router = APIRouter()

SSE = "text/event-stream"


# --- Analyse -------------------------------------------------------------------------------------


def parse_system(raw: Any) -> tuple[SystemMessage, ...]:
    """Le `system` d'Anthropic accepte une chaîne ou une liste de blocs `text`."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (SystemMessage(text=raw),) if raw else ()
    if isinstance(raw, list):
        text = "".join(
            str(block.get("text") or "")
            for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        )
        return (SystemMessage(text=text),) if text else ()
    raise BadRequest("system must be a string or an array of text blocks")


def _parse_image_block(block: dict[str, Any]) -> ImageInput:
    source = block.get("source")
    if not isinstance(source, dict):
        raise BadRequest("image block must have a source")
    kind = source.get("type")
    if kind != "base64":
        # Le type `url` obligerait `ollama.cpp` à émettre une requête sortante pilotée par le
        # client : refus explicite (surface SSRF), pas un échec silencieux.
        raise BadRequest("only base64 image sources are supported")
    data = source.get("data")
    if not isinstance(data, str) or not data:
        raise BadRequest("image source must contain base64 data")
    return ImageInput.from_base64(data, media_type=str(source.get("media_type") or "image/png"))


def parse_messages(raw: Any) -> tuple[CanonicalMessage, ...]:
    """Convertit les messages Anthropic en messages canoniques.

    Un message `user` peut contenir des blocs `tool_result` : ceux-ci sont **extraits** en
    `ToolResultMessage` distincts, dans l'ordre, avant le reste du contenu. Les laisser dans le
    message utilisateur perdrait la corrélation d'appel (risque R7).
    """
    if not isinstance(raw, list):
        raise BadRequest("messages must be an array")

    messages: list[CanonicalMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            raise BadRequest("each message must be an object")
        role = str(item.get("role") or "").lower()
        content = item.get("content")

        if isinstance(content, str):
            blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            blocks = [block for block in content if isinstance(block, dict)]
        else:
            raise BadRequest("message content must be a string or an array of blocks")

        if role == "user":
            messages.extend(_parse_user_blocks(blocks))
        elif role == "assistant":
            messages.append(_parse_assistant_blocks(blocks))
        else:
            raise BadRequest(f"invalid role: {role}")

    return tuple(messages)


def _parse_user_blocks(blocks: list[dict[str, Any]]) -> list[CanonicalMessage]:
    results: list[CanonicalMessage] = []
    parts: list[TextBlock | ImageInput] = []

    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(TextBlock(text=text))
        elif kind == "image":
            parts.append(_parse_image_block(block))
        elif kind == "tool_result":
            content = block.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            results.append(
                ToolResultMessage(
                    call_id=str(block.get("tool_use_id") or ""),
                    content=content if isinstance(content, str) else json.dumps(content),
                    is_error=bool(block.get("is_error")),
                )
            )

    if parts:
        results.append(UserMessage(content=tuple(parts)))
    return results


def _parse_assistant_blocks(blocks: list[dict[str, Any]]) -> AssistantMessage:
    content: list[TextBlock | ReasoningBlock] = []
    tool_calls: list[ToolCall] = []

    for index, block in enumerate(blocks):
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                content.append(TextBlock(text=text))
        elif kind == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str) and thinking:
                content.append(
                    ReasoningBlock(text=thinking, signature=block.get("signature"))
                )
        elif kind == "redacted_thinking":
            content.append(ReasoningBlock(text="", redacted=True))
        elif kind == "tool_use":
            arguments = block.get("input")
            tool_calls.append(
                ToolCall(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    index=index,
                )
            )

    return AssistantMessage(content=tuple(content), tool_calls=tuple(tool_calls))


def parse_tools(raw: Any) -> tuple[ToolDefinition, ...]:
    """Les outils Anthropic ont leurs champs à plat, avec `input_schema`."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise BadRequest("tools must be an array")

    tools: list[ToolDefinition] = []
    for item in raw:
        if not isinstance(item, dict):
            raise BadRequest("each tool must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise BadRequest("each tool must have a name")
        schema = item.get("input_schema")
        tools.append(
            ToolDefinition(
                name=name,
                description=str(item.get("description") or ""),
                parameters=schema if isinstance(schema, dict) else {},
            )
        )
    return tuple(tools)


def parse_tool_choice(raw: Any) -> ToolChoice:
    if not isinstance(raw, dict):
        return ToolChoice()
    kind = raw.get("type")
    if kind == "auto":
        return ToolChoice(mode=ToolChoiceMode.AUTO)
    if kind == "any":
        return ToolChoice(mode=ToolChoiceMode.REQUIRED)
    if kind == "none":
        return ToolChoice(mode=ToolChoiceMode.NONE)
    if kind == "tool":
        name = raw.get("name")
        if isinstance(name, str) and name:
            return ToolChoice(mode=ToolChoiceMode.NAMED, name=name)
    return ToolChoice()


def build_request(body: dict[str, Any], ref) -> CanonicalRequest:
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        # Contrairement à OpenAI, `max_tokens` est requis par l'API Messages.
        raise BadRequest("max_tokens is required and must be a positive integer")

    stop = body.get("stop_sequences")
    stop_tuple = tuple(str(item) for item in stop) if isinstance(stop, list) else ()

    thinking_raw = body.get("thinking")
    thinking = None
    if isinstance(thinking_raw, dict):
        from ..canonical import ThinkingRequest

        thinking = ThinkingRequest(enabled=thinking_raw.get("type") == "enabled")

    from ..canonical import ThinkingRequest as _TR

    return CanonicalRequest(
        model=ref,
        messages=link_tool_results(
            parse_system(body.get("system")) + parse_messages(body.get("messages"))
        ),
        tools=parse_tools(body.get("tools")),
        tool_choice=parse_tool_choice(body.get("tool_choice")),
        options=SamplingOptions(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            num_predict=max_tokens,
            stop=stop_tuple,
        ),
        thinking=thinking or _TR(),
        stream=bool(body.get("stream")),
        source_api=SourceAPI.ANTHROPIC_MESSAGES,
    )


# --- Sérialisation ---------------------------------------------------------------------------------


def message_payload(result: CanonicalResult, *, model: str, message_id: str) -> dict[str, Any]:
    """Réponse `/v1/messages` : blocs typés, dans l'ordre raisonnement → texte → outils."""
    blocks: list[dict[str, Any]] = []
    if result.reasoning:
        blocks.append({"type": "thinking", "thinking": result.reasoning})
    if result.text:
        blocks.append({"type": "text", "text": result.text})
    for call in result.tool_calls:
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": result.finish_reason.to_anthropic(),
        "stop_sequence": None,
        "usage": {
            "input_tokens": result.usage.prompt_eval_count,
            "output_tokens": result.usage.eval_count,
        },
    }


# --- Routes -----------------------------------------------------------------------------------------


@router.post("/v1/messages")
async def messages(request: Request):
    service = get_service(request)
    body = await read_body(request)
    model = service.registry.get(str(body.get("model") or ""))
    canonical = build_request(body, model.ref)

    message_id = f"msg_{int(time.time() * 1000)}"
    async with service.lifecycle.acquire(model.name) as resident:
        reject_unsupported(resident, canonical)
        upstream = backend.serialize_request(canonical, model_id=resident.name)
        client = resident.instance.client

        if not canonical.stream:
            result = await backend.chat(client, upstream, model=resident.name)
            return message_payload(result, model=resident.name, message_id=message_id)

        return StreamingResponse(
            _messages_sse(client, upstream, resident.name, message_id), media_type=SSE
        )


async def _messages_sse(client, upstream, model: str, message_id: str) -> AsyncIterator[bytes]:
    """Flux d'événements nommés de l'API Messages.

    Le protocole impose une structure stricte : `message_start`, puis des blocs de contenu
    délimités par `content_block_start` / `content_block_stop`, puis `message_delta` et
    `message_stop`. Un client Anthropic qui ne reçoit pas ces bornes ne peut pas assembler la
    réponse — d'où la gestion explicite de l'ouverture et de la fermeture des blocs.
    """
    def frame(event: str, payload: dict[str, Any]) -> bytes:
        return (
            f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        ).encode("utf-8")

    yield frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    block_index = -1
    open_kind: str | None = None
    finish = FinishReason.STOP
    usage = None

    def close_block() -> bytes | None:
        nonlocal open_kind
        if open_kind is None:
            return None
        open_kind = None
        return frame("content_block_stop", {"type": "content_block_stop", "index": block_index})

    async for delta in backend.chat_stream(client, upstream):
        if delta.usage is not None:
            usage = delta.usage
        if delta.finish_reason is not None:
            finish = delta.finish_reason

        if delta.reasoning:
            if open_kind != "thinking":
                closing = close_block()
                if closing:
                    yield closing
                block_index += 1
                open_kind = "thinking"
                yield frame("content_block_start", {
                    "type": "content_block_start", "index": block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                })
            yield frame("content_block_delta", {
                "type": "content_block_delta", "index": block_index,
                "delta": {"type": "thinking_delta", "thinking": delta.reasoning},
            })

        if delta.text:
            if open_kind != "text":
                closing = close_block()
                if closing:
                    yield closing
                block_index += 1
                open_kind = "text"
                yield frame("content_block_start", {
                    "type": "content_block_start", "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
            yield frame("content_block_delta", {
                "type": "content_block_delta", "index": block_index,
                "delta": {"type": "text_delta", "text": delta.text},
            })

        for call in delta.tool_calls:
            closing = close_block()
            if closing:
                yield closing
            block_index += 1
            open_kind = "tool_use"
            yield frame("content_block_start", {
                "type": "content_block_start", "index": block_index,
                "content_block": {"type": "tool_use", "id": call.id, "name": call.name,
                                  "input": {}},
            })
            yield frame("content_block_delta", {
                "type": "content_block_delta", "index": block_index,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(call.arguments, ensure_ascii=False)},
            })

    closing = close_block()
    if closing:
        yield closing

    yield frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": finish.to_anthropic(), "stop_sequence": None},
        "usage": {"output_tokens": usage.eval_count if usage else 0},
    })
    yield frame("message_stop", {"type": "message_stop"})


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> dict[str, Any]:
    """Comptage de tokens d'entrée, via `/tokenize` de l'instance.

    Le comptage utilise le **vrai tokenizer du modèle**, pas une estimation : c'est la seule
    façon d'obtenir un chiffre qu'un client puisse utiliser pour dimensionner ses requêtes.
    """
    service = get_service(request)
    body = await read_body(request)
    model = service.registry.get(str(body.get("model") or ""))

    canonical = CanonicalRequest(
        model=model.ref,
        messages=link_tool_results(
            parse_system(body.get("system")) + parse_messages(body.get("messages"))
        ),
        tools=parse_tools(body.get("tools")),
        source_api=SourceAPI.ANTHROPIC_MESSAGES,
    )

    rendered = "\n".join(_render_for_count(message) for message in canonical.messages)

    async with service.lifecycle.acquire(model.name) as resident:
        response = await resident.instance.client.post("/tokenize", json={"content": rendered})
        payload = response.json() if response.status_code < 400 else {}

    tokens = payload.get("tokens")
    return {"input_tokens": len(tokens) if isinstance(tokens, list) else 0}


def _render_for_count(message: CanonicalMessage) -> str:
    """Rendu textuel approché d'un message pour le comptage.

    Le template exact n'est pas appliqué : `/tokenize` compte des tokens de texte brut. Le
    résultat est donc une borne inférieure fidèle au contenu, pas au gabarit — limite documentée
    plutôt que masquée.
    """
    if isinstance(message, SystemMessage):
        return message.text
    if isinstance(message, UserMessage):
        return message.text
    if isinstance(message, AssistantMessage):
        return message.text
    if isinstance(message, ToolResultMessage):
        return message.content
    return ""
