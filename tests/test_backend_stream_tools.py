"""Réassemblage des appels d'outils fragmentés par le streaming.

@verifies docs/BACKLOG.md OC-030 « LlamaServerSupervisor », façades OC-04x
@verifies docs/ollama.cpp-architecture.md §1.2 « Le mode routeur de llama-server »

En flux, `llama-server` découpe `function.arguments` en fragments de tokens répartis sur
plusieurs chunks et corrélés par `index`. Pris isolément, un fragment n'est pas du JSON valide.

Les traiter chunk par chunk produisait **un appel d'outil par fragment**, aux arguments
inexploitables (`{"_raw": "{"}`). Un agent qui reçoit cela rappelle l'outil, reçoit à nouveau des
miettes, et boucle : constaté en production, 5 422 appels à `ask_user` et 4 167 à `view_skill`
pour une seule question, sans jamais aboutir.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ollamacpp import backend
from ollamacpp.canonical import FinishReason


def _sse(chunks: list[dict]) -> bytes:
    corps = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
    return (corps + "data: [DONE]\n\n").encode()


def _fragment(index: int, *, call_id: str = "", name: str = "", args: str = "") -> dict:
    fonction: dict = {}
    if name:
        fonction["name"] = name
    if args:
        fonction["arguments"] = args
    appel: dict = {"index": index, "function": fonction}
    if call_id:
        appel["id"] = call_id
    return {"choices": [{"index": 0, "delta": {"tool_calls": [appel]}}]}


def _client(chunks: list[dict]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(chunks),
                              headers={"content-type": "text/event-stream"})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                             base_url="http://amont.invalide")


async def _collecte(chunks: list[dict]) -> list:
    async with _client(chunks) as client:
        return [d async for d in backend.chat_stream(client, {})]


class TestReassemblage:
    async def test_les_fragments_forment_un_seul_appel(self):
        """Le cas de la panne : `{`, `"id":"`, `document`, `-word`, `"`, `}` en six chunks."""
        deltas = await _collecte([
            _fragment(0, call_id="call_1", name="view_skill"),
            _fragment(0, args="{"),
            _fragment(0, args='"id":"'),
            _fragment(0, args="document"),
            _fragment(0, args="-word"),
            _fragment(0, args='"'),
            _fragment(0, args="}"),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ])
        appels = [c for d in deltas for c in d.tool_calls]
        assert len(appels) == 1, "un fragment ne doit JAMAIS devenir un appel"
        assert appels[0].name == "view_skill"
        assert appels[0].id == "call_1"
        # Le décodage n'a lieu qu'une fois la chaîne complète : plus de `_raw`.
        assert appels[0].arguments == {"id": "document-word"}

    async def test_plusieurs_appels_sont_distingues_par_index(self):
        """Deux outils dans le même tour : leurs fragments ne doivent pas se mélanger."""
        deltas = await _collecte([
            _fragment(0, call_id="a", name="search_knowledge_bases"),
            _fragment(1, call_id="b", name="view_skill"),
            _fragment(0, args='{"query":"ZRD'),
            _fragment(1, args='{"id":"document'),
            _fragment(0, args=' Marne"}'),
            _fragment(1, args='-word"}'),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ])
        appels = sorted((c for d in deltas for c in d.tool_calls), key=lambda c: c.index)
        assert [c.name for c in appels] == ["search_knowledge_bases", "view_skill"]
        assert appels[0].arguments == {"query": "ZRD Marne"}
        assert appels[1].arguments == {"id": "document-word"}

    async def test_les_appels_sortent_une_seule_fois(self):
        """Aucun delta intermédiaire ne porte d'appel : sinon l'aval en verrait plusieurs."""
        deltas = await _collecte([
            _fragment(0, call_id="c", name="ask_user"),
            _fragment(0, args='{"question":"?"}'),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ])
        porteurs = [d for d in deltas if d.tool_calls]
        assert len(porteurs) == 1
        assert porteurs[0].finish_reason is FinishReason.TOOL_CALLS

    async def test_flux_clos_sans_finish_reason(self):
        """Un amont qui coupe sans clôturer ne doit pas faire perdre l'appel accumulé."""
        deltas = await _collecte([
            _fragment(0, call_id="d", name="get_current_timestamp"),
            _fragment(0, args="{}"),
        ])
        appels = [c for d in deltas for c in d.tool_calls]
        assert len(appels) == 1 and appels[0].name == "get_current_timestamp"
        assert appels[0].arguments == {}

    async def test_le_texte_reste_streame_token_par_token(self):
        """Le réassemblage ne concerne QUE les outils : le texte doit rester progressif."""
        deltas = await _collecte([
            {"choices": [{"index": 0, "delta": {"content": "Bon"}}]},
            {"choices": [{"index": 0, "delta": {"content": "jour"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ])
        assert [d.text for d in deltas if d.text] == ["Bon", "jour"]

    async def test_arguments_deja_decodes_par_l_amont(self):
        """Un amont non fragmenté peut livrer un objet : il doit passer sans dégradation."""
        deltas = await _collecte([
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "e", "function": {"name": "timer",
                                                     "arguments": {"seconds": 5}}}]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ])
        appels = [c for d in deltas for c in d.tool_calls]
        assert len(appels) == 1 and appels[0].arguments == {"seconds": 5}
