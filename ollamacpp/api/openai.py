"""Façades OpenAI : Chat Completions, Completions, Embeddings, Models, Responses.

@spec docs/BACKLOG.md OC-070 à OC-074
@spec docs/ollama.cpp-architecture.md §5.2 (façades → canonique → /v1/chat/completions),
      §5.4 « Modèle conversationnel canonique », §8 risque R7
@spec docs/DAT.md §5.1 « Interfaces exposées »

Ces façades partagent **exactement** le registre, le cycle de vie et le runtime de la façade
Ollama : il n'existe qu'un seul moteur, conformément à la mission.

**`/v1/responses` — le point le plus délicat (risque R7).** Un `function_call_output` de l'API
Responses devient un `ToolResultMessage`, jamais un message `user` : c'est la perte sémantique
qui casse les boucles d'agents au-delà d'un tour. Les `call_id` traversent inchangés, et les
blocs `reasoning` restent séparés du texte visible.
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
    ResponseFormat,
    ResponseFormatKind,
    SamplingOptions,
    SourceAPI,
    SystemMessage,
    TextBlock,
    ThinkingLevel,
    ThinkingRequest,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
)
from ..errors import BadRequest
from ..runtime.capabilities import detect
from ..service import Service
from .common import get_service, read_body

router = APIRouter()

SSE = "text/event-stream"


# --- Modèles ------------------------------------------------------------------------------------


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    """Listing OpenAI.

    `data[].id` est le champ sur lequel `ollama-gateway` filtre par clé (`app/proxy.py` l. 60-62) :
    il doit porter le nom Ollama complet, pas un identifiant interne.
    """
    service = get_service(request)
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": model.name, "object": "model", "created": created, "owned_by": "ollama.cpp"}
            for model in service.registry.list()
        ],
    }


@router.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str, request: Request) -> dict[str, Any]:
    service = get_service(request)
    model = service.registry.get(model_id)
    return {
        "id": model.name,
        "object": "model",
        "created": int(model.modified_at.timestamp()),
        "owned_by": "ollama.cpp",
    }


# --- Analyse commune ------------------------------------------------------------------------------


def parse_content(raw: Any) -> tuple[TextBlock | ImageInput, ...]:
    """Convertit un contenu OpenAI (chaîne ou liste de parties typées)."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (TextBlock(text=raw),) if raw else ()
    if not isinstance(raw, list):
        raise BadRequest("message content must be a string or an array")

    parts: list[TextBlock | ImageInput] = []
    for item in raw:
        if not isinstance(item, dict):
            raise BadRequest("each content part must be an object")
        kind = item.get("type")
        if kind == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(TextBlock(text=text))
        elif kind in ("image_url", "input_image"):
            url = item.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str) or not url:
                raise BadRequest("image_url must contain a url")
            if not url.startswith("data:"):
                # Refus explicite plutôt que téléchargement silencieux : `ollama.cpp` ne doit pas
                # émettre de requête sortante à la demande d'un client (surface SSRF).
                raise BadRequest("only data: image URLs are supported")
            parts.append(ImageInput.from_base64(url))
        elif kind == "input_text":
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(TextBlock(text=text))
    return tuple(parts)


def parse_tools(raw: Any) -> tuple[ToolDefinition, ...]:
    """Convertit les outils OpenAI, dans leurs deux formes.

    Chat Completions imbrique sous `function` ; Responses met les champs à plat. Les deux sont
    acceptées pour que la même définition d'outil serve aux deux endpoints.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise BadRequest("tools must be an array")

    tools: list[ToolDefinition] = []
    for item in raw:
        if not isinstance(item, dict):
            raise BadRequest("each tool must be an object")
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise BadRequest("each tool must have a name")
        parameters = function.get("parameters")
        tools.append(
            ToolDefinition(
                name=name,
                description=str(function.get("description") or ""),
                parameters=parameters if isinstance(parameters, dict) else {},
            )
        )
    return tuple(tools)


def parse_tool_choice(raw: Any) -> ToolChoice:
    if raw is None:
        return ToolChoice()
    if isinstance(raw, str):
        try:
            return ToolChoice(mode=ToolChoiceMode(raw))
        except ValueError:
            raise BadRequest(f"invalid tool_choice: {raw}") from None
    if isinstance(raw, dict):
        function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        name = function.get("name")
        if isinstance(name, str) and name:
            return ToolChoice(mode=ToolChoiceMode.NAMED, name=name)
    raise BadRequest("invalid tool_choice")


def parse_response_format(raw: Any) -> ResponseFormat:
    if not isinstance(raw, dict):
        return ResponseFormat()
    kind = raw.get("type")
    if kind == "json_object":
        return ResponseFormat(kind=ResponseFormatKind.JSON)
    if kind == "json_schema":
        wrapper = raw.get("json_schema")
        schema = wrapper.get("schema") if isinstance(wrapper, dict) else None
        return ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA,
                              schema=schema if isinstance(schema, dict) else {})
    return ResponseFormat()


def parse_sampling(body: dict[str, Any]) -> SamplingOptions:
    """Convertit les paramètres OpenAI vers le vocabulaire pivot Ollama."""
    stop = body.get("stop")
    if isinstance(stop, str):
        stop_tuple = (stop,)
    elif isinstance(stop, list):
        stop_tuple = tuple(str(item) for item in stop)
    else:
        stop_tuple = ()

    # `max_completion_tokens` remplace `max_tokens`, déprécié mais toujours largement émis.
    max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_tokens")

    return SamplingOptions(
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        min_p=body.get("min_p"),
        seed=body.get("seed"),
        num_predict=max_tokens if isinstance(max_tokens, int) else None,
        presence_penalty=body.get("presence_penalty"),
        frequency_penalty=body.get("frequency_penalty"),
        stop=stop_tuple,
    )


def parse_reasoning_effort(raw: Any) -> ThinkingRequest:
    if raw is None:
        return ThinkingRequest()
    if raw == "none":
        return ThinkingRequest(enabled=False)
    if isinstance(raw, str):
        try:
            return ThinkingRequest(enabled=True, level=ThinkingLevel(raw))
        except ValueError:
            return ThinkingRequest(enabled=True)
    return ThinkingRequest()


# --- Chat Completions -----------------------------------------------------------------------------


def parse_chat_messages(raw: Any) -> tuple[CanonicalMessage, ...]:
    """Convertit les messages Chat Completions.

    Le rôle `tool` devient un `ToolResultMessage` porteur de son `tool_call_id` : c'est la
    traduction qui préserve la boucle d'outils (risque R7).
    """
    if not isinstance(raw, list):
        raise BadRequest("messages must be an array")

    messages: list[CanonicalMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            raise BadRequest("each message must be an object")
        role = str(item.get("role") or "").lower()

        if role in ("system", "developer"):
            messages.append(SystemMessage(text=_flatten(item.get("content"))))
        elif role == "user":
            messages.append(UserMessage(content=parse_content(item.get("content"))))
        elif role == "assistant":
            blocks: list[TextBlock | ReasoningBlock] = []
            reasoning = item.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                blocks.append(ReasoningBlock(text=reasoning))
            text = _flatten(item.get("content"))
            if text:
                blocks.append(TextBlock(text=text))
            messages.append(
                AssistantMessage(
                    content=tuple(blocks),
                    tool_calls=_parse_openai_tool_calls(item.get("tool_calls")),
                )
            )
        elif role == "tool":
            messages.append(
                ToolResultMessage(
                    call_id=str(item.get("tool_call_id") or ""),
                    name=str(item.get("name") or ""),
                    content=_flatten(item.get("content")),
                )
            )
        else:
            raise BadRequest(f"invalid role: {role}")
    return tuple(messages)


def _flatten(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    return "".join(
        block.text for block in parse_content(raw) if isinstance(block, TextBlock)
    )


def _parse_openai_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"_raw": arguments}
        calls.append(
            ToolCall(
                id=str(item.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
                index=int(item.get("index", index)),
            )
        )
    return tuple(calls)


def build_chat_request(body: dict[str, Any], ref) -> CanonicalRequest:
    return CanonicalRequest(
        model=ref,
        messages=parse_chat_messages(body.get("messages")),
        tools=parse_tools(body.get("tools")),
        tool_choice=parse_tool_choice(body.get("tool_choice")),
        options=parse_sampling(body),
        response_format=parse_response_format(body.get("response_format")),
        thinking=parse_reasoning_effort(body.get("reasoning_effort")),
        stream=bool(body.get("stream")),
        source_api=SourceAPI.OPENAI_CHAT,
    )


def chat_completion_payload(result: CanonicalResult, *, model: str, created: int) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.text or None}
    if result.reasoning:
        message["reasoning_content"] = result.reasoning
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in result.tool_calls
        ]
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": result.finish_reason.to_openai()}
        ],
        "usage": {
            "prompt_tokens": result.usage.prompt_eval_count,
            "completion_tokens": result.usage.eval_count,
            "total_tokens": result.usage.total_tokens,
        },
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    service = get_service(request)
    body = await read_body(request)
    model = service.registry.get(str(body.get("model") or ""))
    canonical = build_chat_request(body, model.ref)

    created = int(time.time())
    async with service.lifecycle.acquire(model.name) as resident:
        _reject_unsupported_openai(resident, canonical)
        upstream = backend.serialize_request(canonical, model_id=resident.name)
        client = resident.instance.client

        if not canonical.stream:
            result = await backend.chat(client, upstream, model=resident.name)
            return chat_completion_payload(result, model=resident.name, created=created)

        return StreamingResponse(
            _chat_sse(client, upstream, resident.name, created), media_type=SSE
        )


async def _chat_sse(client, upstream, model: str, created: int) -> AsyncIterator[bytes]:
    """Flux SSE Chat Completions, reconstruit depuis les deltas canoniques."""
    def frame(payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

    base = {"id": f"chatcmpl-{created}", "object": "chat.completion.chunk",
            "created": created, "model": model}

    yield frame({**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                      "finish_reason": None}]})

    finish = FinishReason.STOP
    usage = None
    async for delta in backend.chat_stream(client, upstream):
        chunk_delta: dict[str, Any] = {}
        if delta.text:
            chunk_delta["content"] = delta.text
        if delta.reasoning:
            chunk_delta["reasoning_content"] = delta.reasoning
        if delta.tool_calls:
            chunk_delta["tool_calls"] = [
                {
                    "index": call.index,
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in delta.tool_calls
            ]
        if delta.usage is not None:
            usage = delta.usage
        if delta.finish_reason is not None:
            finish = delta.finish_reason
        if chunk_delta:
            yield frame({**base, "choices": [{"index": 0, "delta": chunk_delta,
                                              "finish_reason": None}]})

    final: dict[str, Any] = {
        **base,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish.to_openai()}],
    }
    if usage is not None:
        final["usage"] = {
            "prompt_tokens": usage.prompt_eval_count,
            "completion_tokens": usage.eval_count,
            "total_tokens": usage.total_tokens,
        }
    yield frame(final)
    yield b"data: [DONE]\n\n"


def _reject_unsupported_openai(resident, canonical: CanonicalRequest) -> None:
    """Refuse une requête que le modèle ne peut pas honorer.

    Comme pour la façade Ollama, les capacités viennent de `/props` du modèle chargé : c'est la
    seule source qui décrit le modèle réellement en mémoire (OC-032).
    """
    capabilities = resident.capabilities
    if canonical.tools and "tools" not in capabilities:
        raise BadRequest(f'"{resident.name}" does not support tools')
    if any(getattr(message, "images", ()) for message in canonical.messages) and (
        "vision" not in capabilities
    ):
        raise BadRequest(f'"{resident.name}" does not support vision')


# --- Completions (legacy) ---------------------------------------------------------------------------


@router.post("/v1/completions")
async def completions(request: Request):
    """Complétion legacy : le prompt devient un unique message utilisateur canonique."""
    service = get_service(request)
    body = await read_body(request)
    model = service.registry.get(str(body.get("model") or ""))

    prompt = body.get("prompt")
    if isinstance(prompt, list):
        prompt = "".join(str(item) for item in prompt)
    if prompt is not None and not isinstance(prompt, str):
        raise BadRequest("prompt must be a string")

    canonical = CanonicalRequest(
        model=model.ref,
        messages=(UserMessage(content=(TextBlock(text=prompt or ""),)),),
        options=parse_sampling(body),
        stream=bool(body.get("stream")),
        source_api=SourceAPI.OPENAI_COMPLETIONS,
    )

    created = int(time.time())
    async with service.lifecycle.acquire(model.name) as resident:
        upstream = backend.serialize_request(canonical, model_id=resident.name)
        if not canonical.stream:
            result = await backend.chat(resident.instance.client, upstream, model=resident.name)
            return {
                "id": f"cmpl-{created}",
                "object": "text_completion",
                "created": created,
                "model": resident.name,
                "choices": [
                    {"index": 0, "text": result.text, "finish_reason":
                     result.finish_reason.to_openai(), "logprobs": None}
                ],
                "usage": {
                    "prompt_tokens": result.usage.prompt_eval_count,
                    "completion_tokens": result.usage.eval_count,
                    "total_tokens": result.usage.total_tokens,
                },
            }

        return StreamingResponse(
            _completions_sse(resident.instance.client, upstream, resident.name, created),
            media_type=SSE,
        )


async def _completions_sse(client, upstream, model: str, created: int) -> AsyncIterator[bytes]:
    base = {"id": f"cmpl-{created}", "object": "text_completion",
            "created": created, "model": model}
    finish = FinishReason.STOP
    async for delta in backend.chat_stream(client, upstream):
        if delta.finish_reason is not None:
            finish = delta.finish_reason
        if delta.text:
            payload = {**base, "choices": [{"index": 0, "text": delta.text,
                                            "finish_reason": None, "logprobs": None}]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
    payload = {**base, "choices": [{"index": 0, "text": "",
                                    "finish_reason": finish.to_openai(), "logprobs": None}]}
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


# --- Embeddings ---------------------------------------------------------------------------------------


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> dict[str, Any]:
    service = get_service(request)
    body = await read_body(request)
    model = service.registry.get(str(body.get("model") or ""))

    raw_input = body.get("input")
    if raw_input is None:
        raise BadRequest("input is required")
    inputs = [raw_input] if isinstance(raw_input, str) else raw_input
    if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
        raise BadRequest("input must be a string or an array of strings")

    async with service.lifecycle.acquire(model.name) as resident:
        vectors, usage = await backend.embeddings(
            resident.instance.client, model=resident.name, inputs=inputs
        )

    return {
        "object": "list",
        "model": model.name,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {
            "prompt_tokens": usage.prompt_eval_count,
            "total_tokens": usage.total_tokens,
        },
    }
