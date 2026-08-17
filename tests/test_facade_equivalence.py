"""Équivalence des quatre façades et boucles multi-tours d'appels d'outils.

@verifies docs/BACKLOG.md OC-082 « Équivalence des quatre façades »,
          OC-083 « Multi-tours ≥ 10 appels d'outils »
@verifies docs/ollama.cpp-architecture.md §5.4 « Modèle conversationnel canonique », risque R7
@verifies docs/DAT.md §3.1 « Requête d'inférence »

Ce fichier vérifie les exigences §35 et §36 de la mission.

**§35 — équivalence.** Le même échange conceptuel

    user → tool call → tool result → assistant

exprimé dans les quatre formats (Ollama `/api/chat`, OpenAI `/v1/chat/completions`, OpenAI
`/v1/responses`, Anthropic `/v1/messages`) doit produire un `CanonicalRequest` **structurellement
égal**, hors champs de transport.

**§36 — multi-tours.** Une boucle d'au moins dix appels d'outils successifs doit préserver les
rôles, les identifiants d'appels, les résultats, le raisonnement et l'intention initiale.
"""

from __future__ import annotations

import json

import pytest

from ollamacpp.api import anthropic as anthropic_api
from ollamacpp.api import ollama_parse
from ollamacpp.api import openai as openai_api
from ollamacpp.api import responses as responses_api
from ollamacpp.canonical import (
    AssistantMessage,
    CanonicalRequest,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from ollamacpp.names import parse as parse_name

from .conftest import install_model

REF = parse_name("qwen3:8b")

SYSTEME = "Tu es un assistant météo."
QUESTION = "Quel temps fait-il à Paris ?"
ARGUMENTS = {"ville": "Paris", "unite": "celsius"}
RESULTAT = "18°C, ensoleillé"
REPONSE = "Il fait 18°C et ensoleillé à Paris."
CALL_ID = "call_meteo_001"

#: Même plafond de génération exprimé dans les quatre protocoles : `options.num_predict` chez
#: Ollama, `max_tokens` en Chat Completions, `max_output_tokens` en Responses, `max_tokens` chez
#: Anthropic — où il est d'ailleurs obligatoire. Sans cet alignement, la comparaison porterait sur
#: des échanges qui ne sont pas les mêmes.
MAX_TOKENS = 1024

OUTIL = {
    "name": "obtenir_meteo",
    "description": "Renvoie la météo d'une ville",
    "parameters": {
        "type": "object",
        "properties": {"ville": {"type": "string"}, "unite": {"type": "string"}},
        "required": ["ville"],
    },
}


# --- Construction du même échange dans les quatre formats ------------------------------------------


def requete_ollama() -> CanonicalRequest:
    return ollama_parse.parse_chat_request(
        {
            "model": "qwen3:8b",
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEME},
                {"role": "user", "content": QUESTION},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": CALL_ID,
                         "function": {"name": OUTIL["name"], "arguments": ARGUMENTS}}
                    ],
                },
                {"role": "tool", "tool_call_id": CALL_ID, "tool_name": OUTIL["name"],
                 "content": RESULTAT},
                {"role": "assistant", "content": REPONSE},
            ],
            "tools": [{"type": "function", "function": OUTIL}],
            "options": {"num_predict": MAX_TOKENS},
        },
        REF,
    )


def requete_openai_chat() -> CanonicalRequest:
    return openai_api.build_chat_request(
        {
            "model": "qwen3:8b",
            "messages": [
                {"role": "system", "content": SYSTEME},
                {"role": "user", "content": QUESTION},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": CALL_ID, "type": "function",
                         "function": {"name": OUTIL["name"],
                                      "arguments": json.dumps(ARGUMENTS)}}
                    ],
                },
                {"role": "tool", "tool_call_id": CALL_ID, "content": RESULTAT},
                {"role": "assistant", "content": REPONSE},
            ],
            "tools": [{"type": "function", "function": OUTIL}],
            "max_tokens": MAX_TOKENS,
        },
        REF,
    )


def requete_responses() -> CanonicalRequest:
    return responses_api.build_request(
        {
            "model": "qwen3:8b",
            "instructions": SYSTEME,
            "input": [
                {"type": "message", "role": "user", "content": QUESTION},
                {"type": "function_call", "call_id": CALL_ID, "name": OUTIL["name"],
                 "arguments": json.dumps(ARGUMENTS)},
                {"type": "function_call_output", "call_id": CALL_ID, "output": RESULTAT},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": REPONSE}]},
            ],
            "tools": [{"type": "function", **OUTIL}],
            "max_output_tokens": MAX_TOKENS,
        },
        REF,
    )


def requete_anthropic() -> CanonicalRequest:
    return anthropic_api.build_request(
        {
            "model": "qwen3:8b",
            "max_tokens": MAX_TOKENS,
            "system": SYSTEME,
            "messages": [
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": CALL_ID, "name": OUTIL["name"],
                     "input": ARGUMENTS}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": CALL_ID, "content": RESULTAT}]},
                {"role": "assistant", "content": [{"type": "text", "text": REPONSE}]},
            ],
            "tools": [{"name": OUTIL["name"], "description": OUTIL["description"],
                       "input_schema": OUTIL["parameters"]}],
        },
        REF,
    )


TOUTES = {
    "ollama": requete_ollama,
    "openai-chat": requete_openai_chat,
    "openai-responses": requete_responses,
    "anthropic": requete_anthropic,
}


# --- §35 : équivalence ------------------------------------------------------------------------------


class TestEquivalenceDesQuatreFacades:
    @pytest.mark.parametrize("nom", sorted(TOUTES))
    def test_meme_sequence_de_roles(self, nom):
        """Système, utilisateur, assistant+outil, résultat d'outil, assistant."""
        messages = TOUTES[nom]().messages
        types = [type(message).__name__ for message in messages]
        assert types == [
            "SystemMessage", "UserMessage", "AssistantMessage",
            "ToolResultMessage", "AssistantMessage",
        ], f"la façade {nom} ne produit pas la même structure"

    @pytest.mark.parametrize("nom", sorted(TOUTES))
    def test_identifiant_dappel_preserve(self, nom):
        """Invariant n° 1 : `call_id` traverse inchangé sur les quatre façades (risque R7)."""
        messages = TOUTES[nom]().messages
        appel = messages[2].tool_calls[0]
        resultat = messages[3]
        assert appel.id == CALL_ID
        assert resultat.call_id == CALL_ID

    @pytest.mark.parametrize("nom", sorted(TOUTES))
    def test_resultat_doutil_nest_jamais_un_message_utilisateur(self, nom):
        """Invariant n° 2 : la perte sémantique qui casse les boucles d'agents."""
        resultat = TOUTES[nom]().messages[3]
        assert isinstance(resultat, ToolResultMessage)
        assert not isinstance(resultat, UserMessage)
        assert resultat.content == RESULTAT

    @pytest.mark.parametrize("nom", sorted(TOUTES))
    def test_arguments_identiques_et_ordonnes(self, nom):
        """Invariant n° 4 : l'ordre des clés d'arguments est préservé."""
        appel = TOUTES[nom]().messages[2].tool_calls[0]
        assert appel.arguments == ARGUMENTS
        assert list(appel.arguments.keys()) == ["ville", "unite"]

    @pytest.mark.parametrize("nom", sorted(TOUTES))
    def test_outil_declare_identiquement(self, nom):
        outils = TOUTES[nom]().tools
        assert len(outils) == 1
        assert outils[0].name == OUTIL["name"]
        assert outils[0].description == OUTIL["description"]
        assert outils[0].parameters == OUTIL["parameters"]

    @pytest.mark.parametrize("nom", sorted(TOUTES))
    def test_systeme_et_question_identiques(self, nom):
        messages = TOUTES[nom]().messages
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].text == SYSTEME
        assert messages[1].text == QUESTION

    def test_les_quatre_facades_produisent_la_meme_representation(self):
        """Exigence §35 de la mission, vérifiée sur la représentation entière.

        La comparaison porte sur `equivalence_key()`, qui exclut les champs de transport
        (`source_api`, `stream`, `keep_alive`) : ce sont des propriétés de l'appel, pas de
        l'échange conceptuel.
        """
        cles = {nom: fabrique().equivalence_key() for nom, fabrique in TOUTES.items()}
        reference = cles["ollama"]
        for nom, cle in cles.items():
            assert cle == reference, f"la façade {nom} diverge de la représentation canonique"

    def test_le_champ_source_est_bien_le_seul_a_differer(self):
        """Contrôle de cohérence : les façades sont distinguables, mais seulement par `source_api`."""
        sources = {fabrique().source_api for fabrique in TOUTES.values()}
        assert len(sources) == 4


# --- §36 : multi-tours ---------------------------------------------------------------------------------

TOURS = 12


def construire_boucle_ollama(tours: int) -> dict:
    """Construit une conversation Ollama de `tours` appels d'outils successifs."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEME},
        {"role": "user", "content": QUESTION},
    ]
    for index in range(tours):
        messages.append({
            "role": "assistant",
            "content": "",
            "thinking": f"étape {index} : il me faut la météo",
            "tool_calls": [{
                "id": f"call_{index:03d}",
                "function": {"name": OUTIL["name"], "arguments": {"ville": f"ville-{index}"}},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{index:03d}",
            "tool_name": OUTIL["name"],
            "content": f"résultat {index}",
        })
    messages.append({"role": "assistant", "content": REPONSE})
    return {"model": "qwen3:8b", "stream": False, "messages": messages,
            "tools": [{"type": "function", "function": OUTIL}]}


def construire_boucle_responses(tours: int) -> dict:
    items: list[dict] = [{"type": "message", "role": "user", "content": QUESTION}]
    for index in range(tours):
        items.append({"type": "reasoning",
                      "summary": [{"type": "summary_text",
                                   "text": f"étape {index} : il me faut la météo"}]})
        items.append({"type": "function_call", "call_id": f"call_{index:03d}",
                      "name": OUTIL["name"],
                      "arguments": json.dumps({"ville": f"ville-{index}"})})
        items.append({"type": "function_call_output", "call_id": f"call_{index:03d}",
                      "output": f"résultat {index}"})
    items.append({"type": "message", "role": "assistant",
                  "content": [{"type": "output_text", "text": REPONSE}]})
    return {"model": "qwen3:8b", "instructions": SYSTEME, "input": items,
            "tools": [{"type": "function", **OUTIL}]}


def construire_boucle_anthropic(tours: int) -> dict:
    messages: list[dict] = [{"role": "user", "content": QUESTION}]
    for index in range(tours):
        messages.append({"role": "assistant", "content": [
            {"type": "thinking", "thinking": f"étape {index} : il me faut la météo"},
            {"type": "tool_use", "id": f"call_{index:03d}", "name": OUTIL["name"],
             "input": {"ville": f"ville-{index}"}},
        ]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"call_{index:03d}",
             "content": f"résultat {index}"},
        ]})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": REPONSE}]})
    return {"model": "qwen3:8b", "max_tokens": 1024, "system": SYSTEME, "messages": messages,
            "tools": [{"name": OUTIL["name"], "description": OUTIL["description"],
                       "input_schema": OUTIL["parameters"]}]}


class TestMultiTours:
    """Exigence §36 : au moins dix appels d'outils successifs."""

    def test_nombre_de_tours_suffisant(self):
        assert TOURS >= 10

    @pytest.mark.parametrize("nom", ["ollama", "responses", "anthropic"])
    def test_tous_les_identifiants_preserves(self, nom):
        """Aucun `call_id` ne doit être perdu, régénéré ni confondu sur douze tours."""
        requete = {
            "ollama": lambda: ollama_parse.parse_chat_request(
                construire_boucle_ollama(TOURS), REF),
            "responses": lambda: responses_api.build_request(
                construire_boucle_responses(TOURS), REF),
            "anthropic": lambda: anthropic_api.build_request(
                construire_boucle_anthropic(TOURS), REF),
        }[nom]()

        appels = [call.id for message in requete.messages
                  if isinstance(message, AssistantMessage) for call in message.tool_calls]
        resultats = [message.call_id for message in requete.messages
                     if isinstance(message, ToolResultMessage)]

        attendus = [f"call_{index:03d}" for index in range(TOURS)]
        assert appels == attendus
        assert resultats == attendus

    @pytest.mark.parametrize("nom", ["ollama", "responses", "anthropic"])
    def test_appariement_appel_resultat(self, nom):
        """Chaque résultat doit suivre son appel, dans l'ordre : c'est l'ordre qui porte le sens."""
        requete = {
            "ollama": lambda: ollama_parse.parse_chat_request(
                construire_boucle_ollama(TOURS), REF),
            "responses": lambda: responses_api.build_request(
                construire_boucle_responses(TOURS), REF),
            "anthropic": lambda: anthropic_api.build_request(
                construire_boucle_anthropic(TOURS), REF),
        }[nom]()

        for index, message in enumerate(requete.messages):
            if isinstance(message, AssistantMessage) and message.tool_calls:
                suivant = requete.messages[index + 1]
                assert isinstance(suivant, ToolResultMessage)
                assert suivant.call_id == message.tool_calls[0].id

    @pytest.mark.parametrize("nom", ["ollama", "responses", "anthropic"])
    def test_raisonnement_conserve_et_separe(self, nom):
        """Le raisonnement de chaque tour survit et ne contamine jamais le texte visible."""
        requete = {
            "ollama": lambda: ollama_parse.parse_chat_request(
                construire_boucle_ollama(TOURS), REF),
            "responses": lambda: responses_api.build_request(
                construire_boucle_responses(TOURS), REF),
            "anthropic": lambda: anthropic_api.build_request(
                construire_boucle_anthropic(TOURS), REF),
        }[nom]()

        raisonnements = [message.reasoning for message in requete.messages
                         if isinstance(message, AssistantMessage) and message.reasoning]
        assert len(raisonnements) == TOURS
        assert raisonnements[0].startswith("étape 0")

        for message in requete.messages:
            if isinstance(message, AssistantMessage):
                assert "étape" not in message.text

    @pytest.mark.parametrize("nom", ["ollama", "responses", "anthropic"])
    def test_intention_initiale_preservee(self, nom):
        """Après douze tours, la question d'origine doit rester intacte et à sa place."""
        requete = {
            "ollama": lambda: ollama_parse.parse_chat_request(
                construire_boucle_ollama(TOURS), REF),
            "responses": lambda: responses_api.build_request(
                construire_boucle_responses(TOURS), REF),
            "anthropic": lambda: anthropic_api.build_request(
                construire_boucle_anthropic(TOURS), REF),
        }[nom]()

        assert isinstance(requete.messages[0], SystemMessage)
        assert requete.messages[0].text == SYSTEME
        utilisateurs = [m for m in requete.messages if isinstance(m, UserMessage)]
        assert len(utilisateurs) == 1, "aucun résultat d'outil ne doit être devenu un message user"
        assert utilisateurs[0].text == QUESTION

    def test_boucle_complete_acceptee_par_lapi(self, client, installed):
        """Vérification de bout en bout : douze tours acceptés par l'API réelle."""
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json=construire_boucle_ollama(TOURS))
        assert response.status_code == 200
        assert response.json()["done"] is True

    def test_boucle_responses_acceptee_par_lapi(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/responses", json=construire_boucle_responses(TOURS))
        assert response.status_code == 200

    def test_boucle_anthropic_acceptee_par_lapi(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/v1/messages", json=construire_boucle_anthropic(TOURS))
        assert response.status_code == 200

    def test_les_trois_boucles_produisent_la_meme_structure(self):
        """Douze tours, trois façades : la représentation canonique doit rester identique."""
        cles = {
            "ollama": ollama_parse.parse_chat_request(
                construire_boucle_ollama(TOURS), REF),
            "responses": responses_api.build_request(
                construire_boucle_responses(TOURS), REF),
            "anthropic": anthropic_api.build_request(
                construire_boucle_anthropic(TOURS), REF),
        }
        structures = {
            nom: [type(message).__name__ for message in requete.messages]
            for nom, requete in cles.items()
        }
        reference = structures["ollama"]
        for nom, structure in structures.items():
            assert structure == reference, f"la boucle {nom} diverge"
