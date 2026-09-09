"""Tests de bout en bout des façades OpenAI et Anthropic.

@verifies docs/BACKLOG.md OC-070 à OC-076
@verifies docs/ollama.cpp-architecture.md §3.2 (matrice de compatibilité), §5.4, risque R7
@verifies docs/DAT.md §5.1 « Interfaces exposées »
"""

from __future__ import annotations

import json

import pytest

from ollamacpp.storage import RuntimeConfig

from .conftest import install_model


def sse_events(response) -> list[tuple[str | None, dict]]:
    """Décode un flux SSE en couples (événement, données). `[DONE]` est ignoré."""
    events: list[tuple[str | None, dict]] = []
    current_event: str | None = None
    for line in response.text.splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            events.append((current_event, json.loads(payload)))
            current_event = None
    return events


# --- OpenAI : modèles ------------------------------------------------------------------------------


class TestModelsOpenAI:
    def test_listing_expose_id(self, client, installed):
        """`ollama-gateway` filtre `/v1/models` sur `data[].id` (`app/proxy.py` l. 60-62)."""
        install_model(installed, "qwen3:8b")
        body = client.get("/v1/models").json()
        assert body["object"] == "list"
        assert [m["id"] for m in body["data"]] == ["qwen3:8b"]

    def test_listing_vide(self, client):
        assert client.get("/v1/models").json() == {"object": "list", "data": []}

    def test_recuperation_dun_modele(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.get("/v1/models/qwen3:8b").json()
        assert body["id"] == "qwen3:8b"
        assert body["object"] == "model"

    def test_modele_absent(self, client):
        assert client.get("/v1/models/fantome").status_code == 404


# --- OpenAI : Chat Completions ------------------------------------------------------------------------


class TestChatCompletions:
    def test_reponse_complete(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/v1/chat/completions", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "bonjour"}],
        }).json()

        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "echo: bonjour"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 11
        assert body["usage"]["total_tokens"] == 18

    def test_contenu_typé_accepte(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "salut"}]}],
        })
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "echo: salut"

    def test_appels_doutils(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/v1/chat/completions", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "météo"}],
            "tools": [{"type": "function", "function": {"name": "meteo", "parameters": {}}}],
        }).json()

        call = body["choices"][0]["message"]["tool_calls"][0]
        assert call["id"] == "call_fake_1"
        assert call["function"]["name"] == "meteo"
        # Les arguments sont une CHAÎNE JSON côté OpenAI, contrairement à Ollama.
        assert json.loads(call["function"]["arguments"]) == {"ok": True}
        assert body["choices"][0]["finish_reason"] == "tool_calls"

    def test_resultat_doutil_accepte(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3:8b",
            "messages": [
                {"role": "user", "content": "météo ?"},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "meteo", "arguments": '{"ville":"Paris"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "18°C"},
            ],
        })
        assert response.status_code == 200

    def test_streaming_sse(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "bonjour le monde"}],
            "stream": True,
        })
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text.rstrip().endswith("data: [DONE]")

        events = [payload for _event, payload in sse_events(response)]
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        texte = "".join(
            e["choices"][0]["delta"].get("content", "") for e in events
        )
        assert texte == "echo: bonjour le monde"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_usage_dans_le_dernier_chunk(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "x"}], "stream": True,
        })
        _event, dernier = sse_events(response)[-1]
        assert dernier["usage"]["total_tokens"] == 18

    def test_parametres_dechantillonnage(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.2, "top_p": 0.9, "max_tokens": 64, "stop": ["\n"], "seed": 7,
        })
        assert response.status_code == 200

    def test_max_completion_tokens_accepte(self, client, installed):
        """`max_completion_tokens` remplace `max_tokens`, déprécié mais toujours émis."""
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 64,
        })
        assert response.status_code == 200

    def test_image_url_distante_refusee(self, client, installed):
        """Télécharger une URL fournie par le client ouvrirait une surface SSRF."""
        install_model(installed, "qwen3:8b", with_mmproj=True)
        response = client.post("/v1/chat/completions", json={
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://exemple.test/a.png"}}]}],
        })
        assert response.status_code == 400
        assert "data:" in response.json()["error"]


class TestCompletionsLegacy:
    def test_reponse_complete(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/v1/completions", json={
            "model": "qwen3:8b", "prompt": "bonjour",
        }).json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "echo: bonjour"

    def test_streaming(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/completions", json={
            "model": "qwen3:8b", "prompt": "salut", "stream": True,
        })
        texte = "".join(p["choices"][0]["text"] for _e, p in sse_events(response))
        assert texte == "echo: salut"


class TestEmbeddingsOpenAI:
    def test_forme_openai(self, client, installed):
        install_model(installed, "embed", runtime=RuntimeConfig(embedding=True))
        body = client.post("/v1/embeddings", json={"model": "embed", "input": ["a", "bb"]}).json()
        assert body["object"] == "list"
        assert len(body["data"]) == 2
        assert body["data"][0]["object"] == "embedding"
        assert body["data"][1]["index"] == 1


# --- Anthropic Messages -------------------------------------------------------------------------------


class TestMessages:
    def test_reponse_en_blocs(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/v1/messages", json={
            "model": "qwen3:8b", "max_tokens": 64,
            "messages": [{"role": "user", "content": "bonjour"}],
        }).json()

        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["content"] == [{"type": "text", "text": "echo: bonjour"}]
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["input_tokens"] == 11
        assert body["usage"]["output_tokens"] == 7

    def test_max_tokens_obligatoire(self, client, installed):
        """Contrairement à OpenAI, l'API Messages exige `max_tokens`."""
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/messages", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "x"}],
        })
        assert response.status_code == 400
        assert "max_tokens" in response.json()["error"]

    def test_system_en_champ_de_premier_niveau(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/messages", json={
            "model": "qwen3:8b", "max_tokens": 64, "system": "tu es utile",
            "messages": [{"role": "user", "content": "x"}],
        })
        assert response.status_code == 200

    def test_system_en_blocs(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/messages", json={
            "model": "qwen3:8b", "max_tokens": 64,
            "system": [{"type": "text", "text": "tu es utile"}],
            "messages": [{"role": "user", "content": "x"}],
        })
        assert response.status_code == 200

    def test_tool_use_en_sortie(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/v1/messages", json={
            "model": "qwen3:8b", "max_tokens": 64,
            "messages": [{"role": "user", "content": "météo"}],
            "tools": [{"name": "meteo", "description": "météo",
                       "input_schema": {"type": "object", "properties": {}}}],
        }).json()

        bloc = body["content"][0]
        assert bloc["type"] == "tool_use"
        assert bloc["id"] == "call_fake_1"
        assert bloc["name"] == "meteo"
        # `input` est un OBJET côté Anthropic, pas une chaîne JSON.
        assert bloc["input"] == {"ok": True}
        assert body["stop_reason"] == "tool_use"

    def test_tool_result_accepte(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/messages", json={
            "model": "qwen3:8b", "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "météo ?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "meteo",
                     "input": {"ville": "Paris"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18°C"}]},
            ],
        })
        assert response.status_code == 200

    def test_streaming_evenements_nommes(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/messages", json={
            "model": "qwen3:8b", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "bonjour le monde"}],
        })
        assert response.headers["content-type"].startswith("text/event-stream")

        events = sse_events(response)
        noms = [event for event, _payload in events]
        assert noms[0] == "message_start"
        assert "content_block_start" in noms
        assert "content_block_stop" in noms
        assert noms[-1] == "message_stop"
        assert noms[-2] == "message_delta"

        texte = "".join(
            payload["delta"]["text"]
            for event, payload in events
            if event == "content_block_delta" and payload["delta"]["type"] == "text_delta"
        )
        assert texte == "echo: bonjour le monde"

    def test_streaming_stop_reason_final(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/messages", json={
            "model": "qwen3:8b", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "x"}],
        })
        delta = [p for e, p in sse_events(response) if e == "message_delta"][0]
        assert delta["delta"]["stop_reason"] == "end_turn"

    def test_source_image_url_refusee(self, client, installed):
        install_model(installed, "qwen3:8b", with_mmproj=True)
        response = client.post("/v1/messages", json={
            "model": "qwen3:8b", "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://exemple.test/a.png"}}]}],
        })
        assert response.status_code == 400

    def test_count_tokens(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/v1/messages/count_tokens", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "un deux trois"}],
        }).json()
        assert "input_tokens" in body
        assert body["input_tokens"] > 0


# --- OpenAI Responses -----------------------------------------------------------------------------------


class TestResponses:
    def test_entree_chaine_simple(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/v1/responses", json={
            "model": "qwen3:8b", "input": "bonjour",
        }).json()

        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert body["output_text"] == "echo: bonjour"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["type"] == "output_text"

    def test_instructions_font_office_de_system(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/responses", json={
            "model": "qwen3:8b", "instructions": "tu es utile", "input": "x",
        })
        assert response.status_code == 200

    def test_function_call_en_sortie(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/v1/responses", json={
            "model": "qwen3:8b", "input": "météo",
            "tools": [{"type": "function", "name": "meteo",
                       "parameters": {"type": "object", "properties": {}}}],
        }).json()

        appel = [item for item in body["output"] if item["type"] == "function_call"][0]
        assert appel["call_id"] == "call_fake_1"
        assert appel["name"] == "meteo"
        assert json.loads(appel["arguments"]) == {"ok": True}

    def test_function_call_output_nest_pas_un_message_user(self, client, installed):
        """Risque R7 : c'est la perte sémantique qui casse les boucles d'agents."""
        from ollamacpp.api.responses import parse_input
        from ollamacpp.canonical import ToolResultMessage, UserMessage

        messages = parse_input(
            [
                {"type": "message", "role": "user", "content": "météo ?"},
                {"type": "function_call", "call_id": "fc_1", "name": "meteo",
                 "arguments": '{"ville":"Paris"}'},
                {"type": "function_call_output", "call_id": "fc_1", "output": "18°C"},
            ],
            None,
        )
        resultat = messages[-1]
        assert isinstance(resultat, ToolResultMessage)
        assert not isinstance(resultat, UserMessage)
        assert resultat.call_id == "fc_1"

    def test_aller_retour_complet_dun_tour_doutil(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/responses", json={
            "model": "qwen3:8b",
            "input": [
                {"type": "message", "role": "user", "content": "météo ?"},
                {"type": "function_call", "call_id": "fc_1", "name": "meteo",
                 "arguments": '{"ville":"Paris"}'},
                {"type": "function_call_output", "call_id": "fc_1", "output": "18°C"},
            ],
        })
        assert response.status_code == 200

    def test_reasoning_conserve_separement(self, client, installed, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_THINKING", "1")
        install_model(installed, "qwen3:8b")
        body = client.post("/v1/responses", json={
            "model": "qwen3:8b", "input": "x", "reasoning": {"effort": "high"},
        }).json()

        raisonnement = [item for item in body["output"] if item["type"] == "reasoning"]
        assert raisonnement, "le bloc de raisonnement doit être un item distinct"
        assert raisonnement[0]["summary"][0]["text"] == "réflexion simulée"
        # Le raisonnement ne doit pas polluer le texte visible.
        assert "réflexion" not in body["output_text"]

    def test_item_inconnu_refuse(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/responses", json={
            "model": "qwen3:8b", "input": [{"type": "type_inconnu"}],
        })
        assert response.status_code == 400

    def test_streaming_evenements(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/responses", json={
            "model": "qwen3:8b", "input": "bonjour le monde", "stream": True,
        })
        events = sse_events(response)
        assert events[0][0] == "response.created"
        assert events[-1][0] == "response.completed"

        texte = "".join(
            payload["delta"] for event, payload in events
            if event == "response.output_text.delta"
        )
        assert texte == "echo: bonjour le monde"
        assert events[-1][1]["response"]["output_text"] == "echo: bonjour le monde"

    def test_max_output_tokens(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/responses", json={
            "model": "qwen3:8b", "input": "x", "max_output_tokens": 32,
        })
        assert response.status_code == 200


# --- Capacités : refus uniforme sur les quatre façades ---------------------------------------------


class TestRefusDeVisionSurToutesLesFacades:
    """Un modèle sans projecteur doit refuser une image **de la même façon** partout.

    Défaut trouvé sur un vrai modèle de vision : le garde-fou n'existait que sur les façades
    Ollama et OpenAI. Sur Responses et Anthropic, l'image atteignait `llama-server`, qui la
    refusait — le client recevait alors un `502` portant un message d'amont
    (« you may need to provide the mmproj… ») au lieu d'un `400` disant que le modèle ne sait pas
    voir. Deux défauts en un : un code de statut faux, qui suggère une panne serveur là où la
    requête est simplement invalide, et une fuite de détail d'implémentation vers le client.

    Le résultat attendu est celui d'Ollama : `400`, et le message `does not support vision`.
    """

    IMAGE = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
             "AAAABJRU5ErkJggg==")

    @pytest.fixture
    def sans_vision(self, client, installed):
        """Modèle sans `mmproj` : ses capacités observées ne contiennent pas `vision`."""
        install_model(installed, "texte-seul:v1", with_mmproj=False)
        return "texte-seul:v1"

    def test_ollama_refuse(self, client, sans_vision):
        reponse = client.post("/api/chat", json={
            "model": sans_vision, "stream": False,
            "messages": [{"role": "user", "content": "Décris.", "images": [self.IMAGE]}]})
        assert reponse.status_code == 400
        assert "does not support vision" in reponse.json()["error"]

    def test_openai_refuse(self, client, sans_vision):
        reponse = client.post("/v1/chat/completions", json={
            "model": sans_vision,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Décris."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{self.IMAGE}"}}]}]})
        assert reponse.status_code == 400
        assert "does not support vision" in reponse.json()["error"]

    def test_responses_refuse(self, client, sans_vision):
        reponse = client.post("/v1/responses", json={
            "model": sans_vision,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Décris."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{self.IMAGE}"}]}]})
        assert reponse.status_code == 400, reponse.text
        assert "does not support vision" in reponse.json()["error"]

    def test_anthropic_refuse(self, client, sans_vision):
        reponse = client.post("/v1/messages", json={
            "model": sans_vision, "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Décris."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": self.IMAGE}}]}]})
        assert reponse.status_code == 400, reponse.text
        assert "does not support vision" in reponse.json()["error"]

    def test_aucun_message_damont_ne_fuit(self, client, sans_vision):
        """Le client ne doit jamais voir un conseil destiné à l'exploitant du serveur."""
        for chemin, corps in (
            ("/v1/responses", {"model": sans_vision, "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{self.IMAGE}"}]}]}),
            ("/v1/messages", {"model": sans_vision, "max_tokens": 16,
                              "messages": [{"role": "user", "content": [
                                  {"type": "image", "source": {"type": "base64",
                                                               "media_type": "image/png",
                                                               "data": self.IMAGE}}]}]}),
        ):
            texte = client.post(chemin, json=corps).text
            assert "mmproj" not in texte, f"{chemin} divulgue un détail d'amont : {texte[:120]}"

    def test_avec_projecteur_la_meme_requete_passe(self, client, installed):
        """Contre-épreuve : le refus doit venir de la capacité, pas du chemin de l'image."""
        install_model(installed, "avec-vision:v1", with_mmproj=True)
        for chemin, corps in (
            ("/v1/responses", {"model": "avec-vision:v1", "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Décris."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{self.IMAGE}"}]}]}),
            ("/v1/messages", {"model": "avec-vision:v1", "max_tokens": 16,
                              "messages": [{"role": "user", "content": [
                                  {"type": "text", "text": "Décris."},
                                  {"type": "image", "source": {"type": "base64",
                                                               "media_type": "image/png",
                                                               "data": self.IMAGE}}]}]}),
        ):
            reponse = client.post(chemin, json=corps)
            assert reponse.status_code == 200, f"{chemin} : {reponse.text[:150]}"


class TestRefusDoutilsSurToutesLesFacades:
    """Même exigence pour les outils : un modèle sans `tools` refuse partout de la même façon."""

    OUTIL_OPENAI = {"type": "function", "function": {
        "name": "f", "description": "", "parameters": {"type": "object", "properties": {}}}}

    @pytest.fixture
    def sans_outils(self, client, installed, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_TOOLS", "0")
        install_model(installed, "sans-outils:v1")
        return "sans-outils:v1"

    def test_ollama_refuse(self, client, sans_outils):
        reponse = client.post("/api/chat", json={
            "model": sans_outils, "stream": False, "tools": [self.OUTIL_OPENAI],
            "messages": [{"role": "user", "content": "Bonjour"}]})
        assert reponse.status_code == 400
        assert "does not support tools" in reponse.json()["error"]

    def test_openai_refuse(self, client, sans_outils):
        reponse = client.post("/v1/chat/completions", json={
            "model": sans_outils, "tools": [self.OUTIL_OPENAI],
            "messages": [{"role": "user", "content": "Bonjour"}]})
        assert reponse.status_code == 400
        assert "does not support tools" in reponse.json()["error"]

    def test_responses_refuse(self, client, sans_outils):
        reponse = client.post("/v1/responses", json={
            "model": sans_outils,
            "tools": [{"type": "function", "name": "f", "description": "",
                       "parameters": {"type": "object", "properties": {}}}],
            "input": [{"role": "user", "content": "Bonjour"}]})
        assert reponse.status_code == 400, reponse.text
        assert "does not support tools" in reponse.json()["error"]

    def test_anthropic_refuse(self, client, sans_outils):
        reponse = client.post("/v1/messages", json={
            "model": sans_outils, "max_tokens": 16,
            "tools": [{"name": "f", "description": "",
                       "input_schema": {"type": "object", "properties": {}}}],
            "messages": [{"role": "user", "content": "Bonjour"}]})
        assert reponse.status_code == 400, reponse.text
        assert "does not support tools" in reponse.json()["error"]
