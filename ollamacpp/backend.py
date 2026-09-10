"""Pont entre la représentation canonique et `llama-server`.

@spec docs/BACKLOG.md OC-014 « Représentation conversationnelle canonique »,
      OC-044 « /api/chat », OC-071 « /v1/chat/completions »
@spec docs/ollama.cpp-architecture.md §5.2 « Choix technologiques » (façades → canonique →
      /v1/chat/completions), §5.4 « Modèle conversationnel canonique »
@spec docs/DAT.md §3.1 « Requête d'inférence », §5.2 « Interfaces consommées »

**Sérialiseur backend unique.** Les quatre façades convergent vers la représentation canonique,
et c'est ce module — et lui seul — qui la traduit en requête `llama-server`. Il n'existe donc
jamais de conversion façade → façade : chaque API fait exactement un aller et un retour par la
forme canonique, ce qui est précisément ce que la mission demande (§22, §23).

La cible backend est `POST /v1/chat/completions` : c'est la surface la plus complète et la plus
stable de `llama-server` (outils, images, streaming, raisonnement), et n'en maintenir qu'une
seule rend la cohérence inter-façades testable (OC-082).

Les correspondances non évidentes ont été vérifiées dans les sources de `llama.cpp` à la révision
`39be55c` :

- `reasoning_effort: "none"` **désactive** le raisonnement ; toute autre valeur alimente le
  template (`tools/server/server-common.cpp` l. 1295-1304) ;
- `chat_template_kwargs.enable_thinking` (booléen) force l'activation ou la désactivation
  (l. 1285-1293) ;
- `response_format` accepte `json_object` (avec `schema` facultatif) et `json_schema`
  (l. 1145-1157) ;
- le raisonnement revient dans `message.reasoning_content` (`server-task.cpp` l. 605) ;
- les mesures reviennent dans `timings` : `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`
  (`server-common.cpp` l. 67-79) — des **millisecondes**, converties ici en secondes, la
  conversion en nanosecondes n'ayant lieu qu'à la sérialisation Ollama (risque R3).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import httpx

from .canonical import (
    AssistantMessage,
    CanonicalDelta,
    CanonicalRequest,
    CanonicalResult,
    FinishReason,
    ImageInput,
    ReasoningBlock,
    ResponseFormatKind,
    SamplingOptions,
    SystemMessage,
    TextBlock,
    Timings,
    ToolCall,
    ToolChoiceMode,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .errors import UpstreamError

#: Correspondance `finish_reason` OpenAI → raison canonique.
_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.STOP,
}


# --- Sérialisation : canonique → llama-server -----------------------------------------------------


def _serialize_user_content(message: UserMessage) -> Any:
    """Contenu d'un message utilisateur.

    Une chaîne simple est préférée quand il n'y a que du texte : certains templates Jinja ne
    gèrent que le contenu textuel (`supports_typed_content` vaut `false` par défaut dans
    `common/jinja/caps.h`), et leur envoyer une liste typée casserait le rendu.
    """
    images = message.images
    if not images:
        return message.text

    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text:
                parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageInput):
            parts.append({"type": "image_url", "image_url": {"url": block.to_data_uri()}})
    return parts


def serialize_messages(request: CanonicalRequest) -> list[dict[str, Any]]:
    """Traduit les messages canoniques en messages OpenAI.

    Le point sensible est `ToolResultMessage` → `{"role": "tool", "tool_call_id": ...}` : c'est
    la traduction qui préserve la boucle d'outils. La dégrader en message `user` — erreur
    classique des passerelles naïves — ferait perdre la corrélation d'appel (risque R7).
    """
    out: list[dict[str, Any]] = []
    for message in request.messages:
        if isinstance(message, SystemMessage):
            out.append({"role": "system", "content": message.text})

        elif isinstance(message, UserMessage):
            out.append({"role": "user", "content": _serialize_user_content(message)})

        elif isinstance(message, AssistantMessage):
            payload: dict[str, Any] = {"role": "assistant", "content": message.text or None}
            if message.reasoning:
                payload["reasoning_content"] = message.reasoning
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            out.append(payload)

        elif isinstance(message, ToolResultMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.call_id,
                    "content": message.content,
                }
            )

    return out


def serialize_options(options: SamplingOptions) -> dict[str, Any]:
    """Traduit les options canoniques (vocabulaire Ollama) en paramètres `llama-server`.

    Les paramètres hors standard OpenAI (`top_k`, `min_p`, `repeat_penalty`…) sont acceptés par
    `llama-server` sur son endpoint compatible : les émettre est ce qui permet de ne pas perdre
    les réglages fins d'Ollama en passant par la façade OpenAI.
    """
    body: dict[str, Any] = {}
    mapping = {
        "temperature": options.temperature,
        "top_p": options.top_p,
        "top_k": options.top_k,
        "min_p": options.min_p,
        "typical_p": options.typical_p,
        "seed": options.seed,
        "presence_penalty": options.presence_penalty,
        "frequency_penalty": options.frequency_penalty,
        "repeat_penalty": options.repeat_penalty,
        "repeat_last_n": options.repeat_last_n,
    }
    for key, value in mapping.items():
        if value is not None:
            body[key] = value

    if options.num_predict is not None:
        # `num_predict` d'Ollama est le nombre de tokens à générer : `max_tokens` côté OpenAI.
        # La valeur -1 signifie « sans limite » chez Ollama ; on l'omet plutôt que de l'émettre,
        # `llama-server` interprétant l'absence comme la même chose.
        if options.num_predict >= 0:
            body["max_tokens"] = options.num_predict
    if options.stop:
        body["stop"] = list(options.stop)
    if options.extra:
        # Les options non modélisées traversent telles quelles : une option ajoutée par une
        # version ultérieure d'Ollama ne doit pas être perdue silencieusement.
        body.update(options.extra)
    return body


def serialize_request(request: CanonicalRequest, *, model_id: str) -> dict[str, Any]:
    """Construit le corps `POST /v1/chat/completions` à envoyer à `llama-server`."""
    body: dict[str, Any] = {
        "model": model_id,
        "messages": serialize_messages(request),
        "stream": request.stream,
    }

    body.update(serialize_options(request.options))

    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters or {"type": "object", "properties": {}},
                },
            }
            for tool in request.tools
        ]
        if request.tool_choice.mode is ToolChoiceMode.NAMED:
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": request.tool_choice.name},
            }
        elif request.tool_choice.mode is not ToolChoiceMode.AUTO:
            body["tool_choice"] = request.tool_choice.mode.value

    fmt = request.response_format
    if fmt.kind is ResponseFormatKind.JSON:
        body["response_format"] = {"type": "json_object"}
    elif fmt.kind is ResponseFormatKind.JSON_SCHEMA and fmt.schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": fmt.schema},
        }

    thinking = request.thinking
    if thinking.enabled is False:
        # « none » est la valeur qui désactive réellement le raisonnement côté llama.cpp.
        body["reasoning_effort"] = "none"
    elif thinking.enabled is True or thinking.level is not None:
        body["chat_template_kwargs"] = {"enable_thinking": True}
        if thinking.level is not None:
            body["reasoning_effort"] = thinking.level.value

    if request.stream:
        # Sans cette option, le dernier chunk ne porte pas `usage` et les compteurs de tokens
        # d'Ollama seraient absents de la réponse finale.
        body["stream_options"] = {"include_usage": True}

    return body


# --- Désérialisation : llama-server → canonique ----------------------------------------------------


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Décode les arguments d'un appel d'outil en préservant l'ordre des clés.

    Une chaîne non JSON n'est pas une erreur fatale : on la conserve sous une clé dédiée plutôt
    que de perdre l'information ou de faire échouer toute la réponse.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        return {"_raw": raw}
    return decoded if isinstance(decoded, dict) else {"_value": decoded}


def _parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        calls.append(
            ToolCall(
                id=str(item.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=_parse_arguments(function.get("arguments")),
                index=int(item.get("index", index)),
            )
        )
    return tuple(calls)


def parse_timings(payload: dict[str, Any]) -> Timings:
    """Convertit le bloc `timings` de `llama-server` (millisecondes) en secondes."""
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        return Timings()
    prompt_s = float(timings.get("prompt_ms") or 0.0) / 1000.0
    eval_s = float(timings.get("predicted_ms") or 0.0) / 1000.0
    return Timings(total_s=prompt_s + eval_s, prompt_eval_s=prompt_s, eval_s=eval_s)


def parse_usage(payload: dict[str, Any]) -> Usage:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    return Usage(
        prompt_eval_count=int(usage.get("prompt_tokens") or 0),
        eval_count=int(usage.get("completion_tokens") or 0),
    )


def parse_response(payload: dict[str, Any], *, model: str) -> CanonicalResult:
    """Convertit une réponse `/v1/chat/completions` complète en résultat canonique."""
    choices = payload.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}

    content: list[TextBlock | ReasoningBlock] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        content.append(ReasoningBlock(text=reasoning))
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append(TextBlock(text=text))

    tool_calls = _parse_tool_calls(message.get("tool_calls"))
    finish = _FINISH_REASONS.get(str(choice.get("finish_reason") or "stop"), FinishReason.STOP)
    if tool_calls and finish is FinishReason.STOP:
        # `llama-server` peut renvoyer `stop` tout en produisant des appels d'outils ; la raison
        # canonique doit refléter ce qui s'est réellement passé, sinon les façades OpenAI et
        # Anthropic annonceraient une fin de tour alors qu'un outil est attendu.
        finish = FinishReason.TOOL_CALLS

    return CanonicalResult(
        model=model,
        content=tuple(content),
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=parse_usage(payload),
        timings=parse_timings(payload),
    )


def parse_chunk(payload: dict[str, Any]) -> CanonicalDelta:
    """Convertit un chunk SSE `/v1/chat/completions` en delta canonique."""
    choices = payload.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    delta = choice.get("delta") or {}

    raw_finish = choice.get("finish_reason")
    finish = _FINISH_REASONS.get(str(raw_finish), FinishReason.STOP) if raw_finish else None

    tool_calls = _parse_tool_calls(delta.get("tool_calls"))
    if tool_calls and finish is FinishReason.STOP:
        finish = FinishReason.TOOL_CALLS

    text = delta.get("content")
    reasoning = delta.get("reasoning_content")

    return CanonicalDelta(
        text=text if isinstance(text, str) else "",
        reasoning=reasoning if isinstance(reasoning, str) else "",
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=parse_usage(payload) if "usage" in payload else None,
        timings=parse_timings(payload) if "timings" in payload else None,
    )


# --- Appels -----------------------------------------------------------------------------------------


async def chat(client: httpx.AsyncClient, body: dict[str, Any], *, model: str) -> CanonicalResult:
    """Exécute une complétion non streamée et renvoie le résultat canonique."""
    payload = await _post_json(client, "/v1/chat/completions", body)
    return parse_response(payload, model=model)


class _ToolCallAssembler:
    """Réassemble les appels d'outils fragmentés par le streaming.

    En flux, `llama-server` découpe `function.arguments` en fragments de tokens, répartis sur
    plusieurs chunks et corrélés par `index` — `{`, puis `"id":"`, puis `document`, puis `-word`…
    Chaque fragment pris isolément n'est pas du JSON valide.

    Les traiter chunk par chunk produit **un appel d'outil par fragment**, aux arguments
    inexploitables (`{"_raw": "{"}`), là où le modèle n'en a émis qu'un seul. Un agent qui reçoit
    cela rappelle l'outil, reçoit à nouveau des fragments, et **boucle** : constaté en production,
    5 422 appels à `ask_user` et 4 167 à `view_skill` pour une seule question, sans jamais aboutir.

    On accumule donc par `index` et on ne décode qu'à la fin du flux.
    """

    __slots__ = ("_calls",)

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    @property
    def pending(self) -> bool:
        return bool(self._calls)

    def feed(self, raw: Any) -> None:
        """Absorbe les `tool_calls` d'un chunk. Les champs vides ne remplacent jamais un acquis."""
        if not isinstance(raw, list):
            return
        for position, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", position))
            slot = self._calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if item.get("id"):
                slot["id"] = str(item["id"])
            function = item.get("function") or {}
            if function.get("name"):
                slot["name"] = str(function["name"])
            fragment = function.get("arguments")
            if isinstance(fragment, str):
                slot["arguments"] += fragment
            elif isinstance(fragment, dict):
                # Un amont non fragmenté peut livrer les arguments déjà décodés.
                slot["arguments"] = json.dumps(fragment, ensure_ascii=False)

    def drain(self) -> tuple[ToolCall, ...]:
        """Rend les appels complets et se vide. Décodage seulement ici, sur la chaîne entière."""
        calls = tuple(
            ToolCall(
                id=slot["id"],
                name=slot["name"],
                arguments=_parse_arguments(slot["arguments"]),
                index=index,
            )
            for index, slot in sorted(self._calls.items())
            if slot["name"]
        )
        self._calls.clear()
        return calls


async def chat_stream(
    client: httpx.AsyncClient, body: dict[str, Any]
) -> AsyncIterator[CanonicalDelta]:
    """Exécute une complétion streamée et produit des deltas canoniques.

    Le flux SSE de `llama-server` se termine par `data: [DONE]`, qui n'est pas du JSON : il est
    reconnu explicitement plutôt que d'être traité comme une erreur de parsage.

    Les appels d'outils sont **réassemblés** avant d'être émis (cf. `_ToolCallAssembler`) : ils
    sortent en un seul delta, complets, au moment où le flux les clôt. Les façades en aval les
    reçoivent donc tels que le modèle les a formés, et non en miettes.
    """
    assembler = _ToolCallAssembler()
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as response:
            if response.status_code >= 400:
                await response.aread()
                raise UpstreamError(_upstream_message(response))
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(payload, dict):
                    continue

                choice = (payload.get("choices") or [{}])[0]
                choice = choice if isinstance(choice, dict) else {}
                assembler.feed((choice.get("delta") or {}).get("tool_calls"))

                delta = parse_chunk(payload)
                # Les fragments viennent d'être absorbés : ce que `parse_chunk` en a tiré est
                # partiel par construction, on ne le propage pas.
                calls = assembler.drain() if delta.finish_reason is not None else ()
                if delta.text or delta.reasoning or calls or delta.finish_reason is not None \
                        or delta.usage is not None or delta.timings is not None:
                    yield replace(delta, tool_calls=calls)

            # Flux clos sans `finish_reason` explicite : ne pas perdre les appels en attente.
            if assembler.pending:
                yield CanonicalDelta(tool_calls=assembler.drain(),
                                     finish_reason=FinishReason.TOOL_CALLS)
    except httpx.HTTPError as exc:
        raise UpstreamError("upstream inference server is unreachable") from exc


async def embeddings(
    client: httpx.AsyncClient, *, model: str, inputs: list[str]
) -> tuple[list[list[float]], Usage]:
    """Exécute une requête d'embeddings et renvoie les vecteurs et le comptage."""
    payload = await _post_json(
        client, "/v1/embeddings", {"model": model, "input": inputs}
    )
    data = payload.get("data")
    vectors: list[list[float]] = []
    if isinstance(data, list):
        # Les entrées sont réordonnées par `index` : l'ordre du tableau de sortie doit
        # correspondre à l'ordre des entrées, ce dont dépendent tous les clients.
        for item in sorted(
            (entry for entry in data if isinstance(entry, dict)),
            key=lambda entry: entry.get("index", 0),
        ):
            vector = item.get("embedding")
            if isinstance(vector, list):
                vectors.append([float(value) for value in vector])
    return vectors, parse_usage(payload)


async def _post_json(
    client: httpx.AsyncClient, path: str, body: dict[str, Any]
) -> dict[str, Any]:
    try:
        response = await client.post(path, json=body)
    except httpx.HTTPError as exc:
        raise UpstreamError("upstream inference server is unreachable") from exc

    if response.status_code >= 400:
        raise UpstreamError(_upstream_message(response))

    try:
        payload = response.json()
    except ValueError as exc:
        raise UpstreamError("invalid response from upstream inference server") from exc
    return payload if isinstance(payload, dict) else {}


def _upstream_message(response: httpx.Response) -> str:
    """Extrait un message d'erreur amont exploitable, sans divulguer l'infrastructure.

    L'URL, le port et la trace interne de l'instance ne sont jamais renvoyés au client : seul le
    message applicatif l'est (règle §20 sur les erreurs qui ne doivent pas exposer
    l'infrastructure).
    """
    try:
        payload = response.json()
    except ValueError:
        return f"upstream inference server returned HTTP {response.status_code}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    return f"upstream inference server returned HTTP {response.status_code}"
