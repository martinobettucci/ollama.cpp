"""Sérialisation des réponses au format Ollama.

@spec docs/BACKLOG.md OC-041 « /api/tags », OC-042 « /api/show », OC-043 « /api/ps »,
      OC-044 « /api/chat », OC-045 « /api/generate », OC-046 « /api/embed »
@spec docs/ollama.cpp-architecture.md §2.2 « Conventions comportementales »,
      §2.4 « Schémas de requêtes et de réponses », §3.2 (matrice), §8 risques R3 et R4
@spec docs/DAT.md §5.1 « Interfaces exposées »

Schémas reproduits depuis `api/types.go` d'Ollama (révision auditée `d67ad83`) :
`ChatResponse` l. 520-550, `GenerateResponse` l. 891-930, `ListModelResponse` l. 822-833,
`ProcessModelResponse` l. 835-844, `ShowResponse` l. 737-756, `EmbedResponse` l. 621-629.

Trois exigences dictent la forme exacte des sorties.

**Durées en nanosecondes (risque R3).** Les six champs de `Metrics` sont des entiers en
nanosecondes ; un client calcule des tokens/s en divisant `eval_count` par `eval_duration` puis
en multipliant par 10⁹.

**`name` ET `model` dans `/api/tags` (risque R4).** `ollama-gateway` filtre son listing sur
`m.get("name") or m.get("model")` (`app/proxy.py` l. 57-59) : les deux champs doivent être
présents et porter le nom complet `modèle:tag`.

**Horodatages RFC 3339.** `created_at`, `modified_at` et `expires_at` sont des dates Go
sérialisées en RFC 3339 ; les clients Ollama les analysent comme telles.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from ..canonical import CanonicalDelta, CanonicalResult, ToolCall
from ..durations import seconds_to_nanoseconds


def now_rfc3339() -> str:
    """Horodatage courant au format attendu par les clients Ollama."""
    return to_rfc3339(dt.datetime.now(dt.timezone.utc))


def to_rfc3339(moment: dt.datetime) -> str:
    """Sérialise une date en RFC 3339, en forçant un fuseau explicite.

    Une date naïve serait ambiguë pour un client ; on la considère alors comme UTC plutôt que de
    produire un horodatage sans fuseau que Go refuserait d'analyser.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.isoformat()


def tool_calls_json(calls: tuple[ToolCall, ...]) -> list[dict[str, Any]]:
    """Sérialise des appels d'outils au format Ollama.

    `function.index` est présent parce qu'Ollama le sérialise sans `omitempty`
    (`ToolCallFunction.Index`) ; `id` est omis s'il est vide, comme chez Ollama.
    """
    out: list[dict[str, Any]] = []
    for call in calls:
        payload: dict[str, Any] = {
            "function": {
                "index": call.index,
                "name": call.name,
                "arguments": call.arguments,
            }
        }
        if call.id:
            payload["id"] = call.id
        out.append(payload)
    return out


# --- /api/chat ----------------------------------------------------------------------------------


def chat_chunk(model: str, delta: CanonicalDelta) -> dict[str, Any]:
    """Chunk NDJSON intermédiaire de `/api/chat`."""
    message: dict[str, Any] = {"role": "assistant", "content": delta.text}
    if delta.reasoning:
        message["thinking"] = delta.reasoning
    if delta.tool_calls:
        message["tool_calls"] = tool_calls_json(delta.tool_calls)
    return {
        "model": model,
        "created_at": now_rfc3339(),
        "message": message,
        "done": False,
    }


def chat_final(model: str, result: CanonicalResult, *, load_s: float = 0.0) -> dict[str, Any]:
    """Réponse finale de `/api/chat`, avec message complet et métriques."""
    message: dict[str, Any] = {"role": "assistant", "content": result.text}
    if result.reasoning:
        message["thinking"] = result.reasoning
    if result.tool_calls:
        message["tool_calls"] = tool_calls_json(result.tool_calls)

    payload: dict[str, Any] = {
        "model": model,
        "created_at": now_rfc3339(),
        "message": message,
        "done": True,
        "done_reason": result.finish_reason.to_ollama(),
    }
    payload.update(_metrics(result, load_s))
    return payload


def chat_stream_final(
    model: str, result: CanonicalResult, *, load_s: float = 0.0
) -> dict[str, Any]:
    """Dernier chunk NDJSON d'un `/api/chat` streamé.

    Le message y est **vide** : le contenu a déjà été envoyé morceau par morceau, et le répéter
    ferait doubler la réponse chez tout client qui concatène les chunks.
    """
    payload = chat_final(model, result, load_s=load_s)
    payload["message"] = {"role": "assistant", "content": ""}
    if result.tool_calls:
        # Les appels d'outils font exception : Ollama les émet dans le chunk final, car un client
        # doit pouvoir les lire sans réassembler des fragments d'arguments JSON.
        payload["message"]["tool_calls"] = tool_calls_json(result.tool_calls)
    return payload


# --- /api/generate -------------------------------------------------------------------------------


def generate_chunk(model: str, delta: CanonicalDelta) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "created_at": now_rfc3339(),
        "response": delta.text,
        "done": False,
    }
    if delta.reasoning:
        payload["thinking"] = delta.reasoning
    return payload


def generate_final(
    model: str, result: CanonicalResult, *, load_s: float = 0.0, streamed: bool = False
) -> dict[str, Any]:
    """Réponse finale de `/api/generate`.

    En streaming, `response` est vide dans le chunk final — comportement documenté par Ollama :
    « `response`: empty if the response was streamed » (`docs/api.md`).
    """
    payload: dict[str, Any] = {
        "model": model,
        "created_at": now_rfc3339(),
        "response": "" if streamed else result.text,
        "done": True,
        "done_reason": result.finish_reason.to_ollama(),
    }
    if result.reasoning and not streamed:
        payload["thinking"] = result.reasoning
    if result.tool_calls:
        payload["tool_calls"] = tool_calls_json(result.tool_calls)
    payload.update(_metrics(result, load_s))
    return payload


def _metrics(result: CanonicalResult, load_s: float) -> dict[str, int]:
    """Six champs de `api.Metrics`, en nanosecondes, champs nuls omis (risque R3)."""
    timings = result.timings
    raw = {
        "total_duration": seconds_to_nanoseconds(timings.total_s + load_s),
        "load_duration": seconds_to_nanoseconds(load_s),
        "prompt_eval_count": result.usage.prompt_eval_count,
        "prompt_eval_duration": seconds_to_nanoseconds(timings.prompt_eval_s),
        "eval_count": result.usage.eval_count,
        "eval_duration": seconds_to_nanoseconds(timings.eval_s),
    }
    return {key: value for key, value in raw.items() if value}


# --- /api/tags, /api/ps, /api/show ------------------------------------------------------------------


def list_model_entry(model, capabilities: list[str] | None = None) -> dict[str, Any]:
    """Entrée de `/api/tags` (`api.ListModelResponse`).

    `name` **et** `model` portent tous deux le nom complet : `ollama-gateway` filtre son listing
    sur l'un ou l'autre (risque R4).
    """
    payload: dict[str, Any] = {
        "name": model.name,
        "model": model.name,
        "modified_at": to_rfc3339(model.modified_at),
        "size": model.size,
        "digest": _bare_digest(model.digest),
        "details": model.details(),
    }
    if capabilities:
        payload["capabilities"] = capabilities
    return payload


def vram_bytes(*, size: int, gpu_layers: int | None, block_count: int) -> int:
    """Estime la part de l'empreinte réellement placée en VRAM.

    Ce champ n'est pas décoratif : le CLI Ollama en déduit la colonne « PROCESSOR » de
    `ollama ps` (`cmd/cmd.go` l. 1141-1150) — `0` affiche « 100% CPU », une valeur égale à `size`
    affiche « 100% GPU », et toute valeur intermédiaire affiche la répartition en pourcentage.

    Renvoyer systématiquement `size` ferait donc annoncer « 100% GPU » à un serveur qui calcule
    intégralement sur CPU. La valeur est déduite du nombre de couches réellement déportées :
    aucune couche sur GPU signifie aucune VRAM.

    Approximation assumée : `llama-server` n'expose pas l'occupation VRAM réelle, et les couches
    n'ont pas toutes le même poids. La répartition est donc proportionnelle au nombre de couches,
    ce qui donne un ordre de grandeur juste et une colonne « PROCESSOR » honnête.
    """
    if not gpu_layers:  # `None` (non configuré) ou `0` (CPU explicite)
        return 0
    if block_count > 0 and gpu_layers < block_count:
        return int(size * gpu_layers / block_count)
    return size


def process_model_entry(
    resident, *, size: int, size_vram: int, digest: str, details: dict[str, Any],
    now: dt.datetime
) -> dict[str, Any]:
    """Entrée de `/api/ps` (`api.ProcessModelResponse`).

    `expires_at` reflète l'état **réel** du cycle de vie : un `keep_alive` illimité produit une
    date très lointaine, comme Ollama, plutôt qu'un champ absent que les clients ne sauraient pas
    interpréter.
    """
    remaining = resident.keep_alive.seconds
    if resident.keep_alive.is_infinite:
        expires = now + dt.timedelta(days=3650)
    else:
        expires = now + dt.timedelta(seconds=max(0.0, remaining))

    return {
        "name": resident.name,
        "model": resident.name,
        "size": size,
        "digest": _bare_digest(digest),
        "details": details,
        "expires_at": to_rfc3339(expires),
        "size_vram": size_vram,
        "context_length": resident.context_length,
    }


def _bare_digest(digest: str) -> str:
    """Digest sans son préfixe d'algorithme.

    Ollama expose des digests hexadécimaux nus (`docs/api.md`, exemples de `/api/tags`) ; le
    préfixe `sha256:` est interne au magasin.
    """
    return digest.split(":", 1)[1] if ":" in digest else digest


def show_response(
    model,
    *,
    capabilities: list[str],
    template: str,
    system: str,
    parameters: str,
    model_info: dict[str, Any],
    projector_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Réponse de `/api/show` (`api.ShowResponse`).

    `model_info` n'est pas `omitempty` chez Ollama : il est toujours présent, même vide.
    """
    payload: dict[str, Any] = {
        "details": model.details(),
        "model_info": model_info,
        "capabilities": capabilities,
        "modified_at": to_rfc3339(model.modified_at),
    }
    if template:
        payload["template"] = template
    if system:
        payload["system"] = system
    if parameters:
        payload["parameters"] = parameters
    if model.manifest.license:
        payload["license"] = model.manifest.license
    if projector_info:
        payload["projector_info"] = projector_info
    return payload


def parameters_block(parameters: dict[str, Any]) -> str:
    """Rend le bloc `parameters` de `/api/show` au format Modelfile.

    Ollama renvoie une **chaîne** de lignes `clé valeur`, pas un objet : un client qui l'affiche
    telle quelle doit y retrouver la forme d'un Modelfile.
    """
    lines: list[str] = []
    for key, value in parameters.items():
        if isinstance(value, list):
            lines.extend(f"{key} {_quote(item)}" for item in value)
        else:
            lines.append(f"{key} {_quote(value)}")
    return "\n".join(lines)


def _quote(value: Any) -> str:
    text = str(value)
    return f'"{text}"' if " " in text else text


# --- /api/embed ------------------------------------------------------------------------------------


def embed_response(
    model: str, vectors: list[list[float]], *, prompt_eval_count: int, total_s: float,
    load_s: float = 0.0
) -> dict[str, Any]:
    """Réponse de `/api/embed` (`api.EmbedResponse`, forme plurielle)."""
    payload: dict[str, Any] = {"model": model, "embeddings": vectors}
    raw = {
        "total_duration": seconds_to_nanoseconds(total_s),
        "load_duration": seconds_to_nanoseconds(load_s),
        "prompt_eval_count": prompt_eval_count,
    }
    payload.update({key: value for key, value in raw.items() if value})
    return payload


def embeddings_response(vector: list[float]) -> dict[str, Any]:
    """Réponse de `/api/embeddings` (forme legacy, singulière et non imbriquée).

    Schéma volontairement différent de `/api/embed` : `api.EmbeddingResponse` ne porte qu'un
    champ `embedding`, sans nom de modèle ni métriques.
    """
    return {"embedding": vector}
