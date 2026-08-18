"""Façade OpenAI Responses (`/v1/responses`).

@spec docs/BACKLOG.md OC-074 « /v1/responses »
@spec docs/ollama.cpp-architecture.md §5.4 « Modèle conversationnel canonique », §8 risque R7
@spec docs/DAT.md §5.1 « Interfaces exposées »

C'est l'API que la mission demande de traiter « avec beaucoup de rigueur » (§21), pour une raison
précise : son format d'entrée est une **liste d'items hétérogènes** dans laquelle un résultat
d'outil est un item de premier niveau — `{"type": "function_call_output", "call_id": ...}` — et
non un message. Une passerelle naïve le transforme en message `user`, ce qui détruit la
corrélation d'appel et casse toute boucle d'agent au-delà d'un tour.

Ici, un `function_call_output` devient un `ToolResultMessage`, un type qui rend cette dégradation
structurellement impossible, et son `call_id` traverse inchangé dans les deux sens.

Sont préservés : `input`, `output`, messages, `reasoning`, `function_call`,
`function_call_output`, `call_id`, `tools`, événements de streaming et `usage`.
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
    ReasoningBlock,
    SourceAPI,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    link_tool_results,
)
from ..errors import BadRequest
from .common import get_service, read_body, reject_unsupported
from .openai import parse_content, parse_sampling, parse_tool_choice, parse_tools

router = APIRouter()

SSE = "text/event-stream"


def parse_input(raw: Any, instructions: str | None) -> tuple[CanonicalMessage, ...]:
    """Convertit le champ `input` de l'API Responses.

    `input` accepte une chaîne simple ou une liste d'items typés. Les types traités :

    - `message` (ou un objet portant `role`) → message du rôle correspondant ;
    - `function_call` → appel d'outil porté par un message assistant ;
    - `function_call_output` → **`ToolResultMessage`**, jamais un message `user` (risque R7) ;
    - `reasoning` → bloc de raisonnement, conservé séparé du texte visible.
    """
    messages: list[CanonicalMessage] = []
    if instructions:
        # `instructions` est le `system` de l'API Responses.
        messages.append(SystemMessage(text=instructions))

    if raw is None:
        return tuple(messages)
    if isinstance(raw, str):
        messages.append(UserMessage(content=(TextBlock(text=raw),)))
        return tuple(messages)
    if not isinstance(raw, list):
        raise BadRequest("input must be a string or an array of items")

    #: Les `function_call` et les blocs `reasoning` d'un même tour assistant sont regroupés :
    #: l'API les émet comme des items séparés, mais ils appartiennent au même message.
    pending_calls: list[ToolCall] = []
    pending_blocks: list[TextBlock | ReasoningBlock] = []

    def flush() -> None:
        if pending_calls or pending_blocks:
            messages.append(
                AssistantMessage(content=tuple(pending_blocks), tool_calls=tuple(pending_calls))
            )
            pending_calls.clear()
            pending_blocks.clear()

    for item in raw:
        if not isinstance(item, dict):
            raise BadRequest("each input item must be an object")
        kind = item.get("type") or ("message" if item.get("role") else None)

        if kind == "function_call_output":
            flush()
            output = item.get("output")
            messages.append(
                ToolResultMessage(
                    call_id=str(item.get("call_id") or ""),
                    content=output if isinstance(output, str) else json.dumps(output),
                    name=str(item.get("name") or ""),
                )
            )

        elif kind == "function_call":
            arguments = item.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"_raw": arguments}
            pending_calls.append(
                ToolCall(
                    id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    index=len(pending_calls),
                )
            )

        elif kind == "reasoning":
            summary = item.get("summary") or item.get("content") or []
            text = "".join(
                str(part.get("text") or "")
                for part in summary
                if isinstance(part, dict)
            ) if isinstance(summary, list) else str(summary)
            if text:
                pending_blocks.append(ReasoningBlock(text=text))

        elif kind == "message":
            role = str(item.get("role") or "user").lower()
            content = item.get("content")
            if role == "assistant":
                text = "".join(
                    block.text for block in parse_content(content) if isinstance(block, TextBlock)
                )
                if text:
                    pending_blocks.append(TextBlock(text=text))
            else:
                flush()
                if role in ("system", "developer"):
                    text = "".join(
                        block.text for block in parse_content(content)
                        if isinstance(block, TextBlock)
                    )
                    messages.append(SystemMessage(text=text))
                else:
                    messages.append(UserMessage(content=parse_content(content)))
        else:
            raise BadRequest(f"unsupported input item type: {kind}")

    flush()
    return link_tool_results(tuple(messages))


def build_request(body: dict[str, Any], ref) -> CanonicalRequest:
    instructions = body.get("instructions")
    max_output = body.get("max_output_tokens")

    options = parse_sampling(body)
    if isinstance(max_output, int) and not isinstance(max_output, bool):
        from dataclasses import replace

        options = replace(options, num_predict=max_output)

    thinking = _parse_reasoning(body.get("reasoning"))

    return CanonicalRequest(
        model=ref,
        messages=parse_input(body.get("input"), instructions if isinstance(instructions, str) else None),
        tools=parse_tools(body.get("tools")),
        tool_choice=parse_tool_choice(body.get("tool_choice")),
        options=options,
        thinking=thinking,
        stream=bool(body.get("stream")),
        source_api=SourceAPI.OPENAI_RESPONSES,
    )


def _parse_reasoning(raw: Any):
    from ..canonical import ThinkingLevel, ThinkingRequest

    if not isinstance(raw, dict):
        return ThinkingRequest()
    effort = raw.get("effort")
    if effort == "none":
        return ThinkingRequest(enabled=False)
    if isinstance(effort, str):
        try:
            return ThinkingRequest(enabled=True, level=ThinkingLevel(effort))
        except ValueError:
            return ThinkingRequest(enabled=True)
    return ThinkingRequest(enabled=True)


def response_payload(result: CanonicalResult, *, model: str, response_id: str,
                     created: int) -> dict[str, Any]:
    """Réponse `/v1/responses` : liste d'items `output`, ordonnée et typée.

    Les appels d'outils sortent en items `function_call` portant `call_id` — c'est ce
    `call_id` que le client renverra dans un `function_call_output`, et qui doit donc être
    exactement celui produit par le modèle.
    """
    output: list[dict[str, Any]] = []

    if result.reasoning:
        output.append(
            {
                "type": "reasoning",
                "id": f"rs_{created}",
                "summary": [{"type": "summary_text", "text": result.reasoning}],
            }
        )

    if result.text:
        output.append(
            {
                "type": "message",
                "id": f"msg_{created}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result.text, "annotations": []}],
            }
        )

    for call in result.tool_calls:
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{call.id or created}",
                "call_id": call.id,
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                "status": "completed",
            }
        )

    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": result.text,
        "usage": {
            "input_tokens": result.usage.prompt_eval_count,
            "output_tokens": result.usage.eval_count,
            "total_tokens": result.usage.total_tokens,
        },
    }


@router.post("/v1/responses")
async def responses(request: Request):
    service = get_service(request)
    body = await read_body(request)
    model = service.registry.get(str(body.get("model") or ""))
    canonical = build_request(body, model.ref)

    created = int(time.time())
    response_id = f"resp_{created}"

    async with service.lifecycle.acquire(model.name) as resident:
        reject_unsupported(resident, canonical)
        upstream = backend.serialize_request(canonical, model_id=resident.name)
        client = resident.instance.client

        if not canonical.stream:
            result = await backend.chat(client, upstream, model=resident.name)
            return response_payload(
                result, model=resident.name, response_id=response_id, created=created
            )

        return StreamingResponse(
            _responses_sse(client, upstream, resident.name, response_id, created),
            media_type=SSE,
        )


async def _responses_sse(client, upstream, model: str, response_id: str,
                         created: int) -> AsyncIterator[bytes]:
    """Flux d'événements de l'API Responses.

    Le protocole est événementiel et nommé : `response.created`, puis des deltas de texte, puis
    `response.completed` portant la réponse complète. Le dernier événement contient l'intégralité
    de `output`, ce qui permet à un client de tout reconstruire même s'il a ignoré les deltas.
    """
    def frame(event: str, payload: dict[str, Any]) -> bytes:
        return (
            f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        ).encode("utf-8")

    skeleton = {
        "id": response_id, "object": "response", "created_at": created,
        "status": "in_progress", "model": model, "output": [],
    }
    yield frame("response.created", {"type": "response.created", "response": skeleton})

    content: list[TextBlock | ReasoningBlock] = []
    tool_calls: tuple[ToolCall, ...] = ()
    finish = FinishReason.STOP
    usage = None

    async for delta in backend.chat_stream(client, upstream):
        if delta.usage is not None:
            usage = delta.usage
        if delta.finish_reason is not None:
            finish = delta.finish_reason
        if delta.tool_calls:
            tool_calls = delta.tool_calls
        if delta.reasoning:
            content.append(ReasoningBlock(text=delta.reasoning))
            yield frame("response.reasoning_summary_text.delta", {
                "type": "response.reasoning_summary_text.delta",
                "item_id": f"rs_{created}", "output_index": 0,
                "delta": delta.reasoning,
            })
        if delta.text:
            content.append(TextBlock(text=delta.text))
            yield frame("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": f"msg_{created}", "output_index": 0,
                "delta": delta.text,
            })

    result = CanonicalResult(
        model=model,
        content=tuple(content),
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=usage or CanonicalResult(model=model).usage,
    )
    final = response_payload(result, model=model, response_id=response_id, created=created)
    yield frame("response.completed", {"type": "response.completed", "response": final})
