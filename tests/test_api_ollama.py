"""Tests de bout en bout de la façade Ollama native.

@verifies docs/BACKLOG.md OC-040 à OC-047, OC-051 à OC-055
@verifies docs/ollama.cpp-architecture.md §2.1 « Routes », §2.2 « Conventions »,
          §3.2 (matrice de compatibilité), risques R1, R2, R3, R4
@verifies docs/DAT.md §5.1 « Interfaces exposées »

L'application complète est démarrée, adossée au faux `llama-server` réellement lancé : ce sont
des tests d'API, pas des appels de fonctions.
"""

from __future__ import annotations

import json

import pytest

from ollamacpp.storage import LifecycleConfig, RuntimeConfig

from .conftest import install_model


def ndjson(response) -> list[dict]:
    """Décode un flux NDJSON en liste d'objets."""
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


# --- Découverte ------------------------------------------------------------------------------------


class TestDecouverte:
    def test_racine_texte_brut(self, client):
        """Sonde de vie d'Ollama : texte brut, pas du JSON."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == "Ollama is running"

    def test_version(self, client):
        response = client.get("/api/version")
        assert response.status_code == 200
        assert "version" in response.json()

    def test_version_en_head(self, client):
        assert client.head("/api/version").status_code == 200


class TestTags:
    """Criticité P0 : sonde de disponibilité et source du filtrage d'`ollama-gateway`."""

    def test_listing_vide_repond_200(self, client):
        """Un serveur sans modèle est **en ligne** : renvoyer une erreur le ferait passer pour mort."""
        response = client.get("/api/tags")
        assert response.status_code == 200
        assert response.json() == {"models": []}

    def test_name_et_model_presents(self, client, installed):
        """`ollama-gateway` filtre sur `name` ou `model` (risque R4)."""
        install_model(installed, "qwen3:8b")
        entry = client.get("/api/tags").json()["models"][0]
        assert entry["name"] == "qwen3:8b"
        assert entry["model"] == "qwen3:8b"

    def test_taille_et_digest_reels(self, client, installed):
        """Le trou que comble le registre : `llama-server` renvoie des bouchons ici."""
        install_model(installed, "qwen3:8b")
        entry = client.get("/api/tags").json()["models"][0]
        assert isinstance(entry["size"], int) and entry["size"] > 0
        assert len(entry["digest"]) == 64, "digest hexadécimal nu, sans préfixe sha256:"

    def test_details_complets(self, client, installed):
        install_model(installed, "qwen3:8b")
        details = client.get("/api/tags").json()["models"][0]["details"]
        assert details["family"] == "qwen3"
        assert details["quantization_level"] == "Q4_K_M"
        assert details["parameter_size"] == "7.6B"
        assert details["format"] == "gguf"

    def test_date_de_modification_presente(self, client, installed):
        install_model(installed, "qwen3:8b")
        assert client.get("/api/tags").json()["models"][0]["modified_at"]

    def test_head_supporte(self, client):
        """Ollama enregistre `/api/tags` en GET **et** HEAD."""
        assert client.head("/api/tags").status_code == 200

    def test_sonde_de_disponibilite_de_la_passerelle(self, client, installed):
        """Reproduit `app/servers.py::probe` : elle lit `name` ou `model` de chaque entrée."""
        install_model(installed, "qwen3:8b")
        data = client.get("/api/tags").json()
        modeles = [m.get("name") or m.get("model") for m in data.get("models", [])]
        assert modeles == ["qwen3:8b"]


class TestShow:
    def test_modele_absent(self, client):
        response = client.post("/api/show", json={"model": "fantome"})
        assert response.status_code == 404
        assert response.json() == {"error": "model 'fantome' not found"}

    def test_details_et_capacites(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/api/show", json={"model": "qwen3:8b"}).json()
        assert body["details"]["family"] == "qwen3"
        assert "capabilities" in body
        assert "completion" in body["capabilities"]

    def test_model_info_toujours_present(self, client, installed):
        """`ShowResponse.ModelInfo` n'est pas `omitempty` chez Ollama."""
        install_model(installed, "qwen3:8b")
        body = client.post("/api/show", json={"model": "qwen3:8b"}).json()
        assert "model_info" in body
        assert body["model_info"]["general.architecture"] == "qwen3"

    def test_template_issu_du_gguf(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/api/show", json={"model": "qwen3:8b"}).json()
        assert "{% for m in messages %}" in body["template"]

    def test_alias_name_accepte(self, client, installed):
        """`ShowRequest.Name` est déprécié mais toujours honoré par Ollama."""
        install_model(installed, "qwen3:8b")
        assert client.post("/api/show", json={"name": "qwen3:8b"}).status_code == 200

    def test_vocabulaire_non_transporte_en_entier(self, client, installed):
        install_model(installed, "qwen3:8b")
        info = client.post("/api/show", json={"model": "qwen3:8b"}).json()["model_info"]
        assert "tokenizer.ggml.tokens" not in info
        assert info["tokenizer.ggml.tokens.length"] == 5


class TestPs:
    def test_vide_avant_tout_chargement(self, client, installed):
        install_model(installed, "qwen3:8b")
        assert client.get("/api/ps").json() == {"models": []}

    def test_reflete_letat_reel(self, client, installed):
        install_model(installed, "qwen3:8b")
        client.post("/api/chat", json={"model": "qwen3:8b", "stream": False,
                                       "messages": [{"role": "user", "content": "bonjour"}]})
        models = client.get("/api/ps").json()["models"]
        assert len(models) == 1
        entry = models[0]
        assert entry["name"] == "qwen3:8b"
        assert entry["size_vram"] > 0
        assert entry["context_length"] > 0
        assert entry["expires_at"]

    def test_contexte_reflete_la_configuration(self, client, installed):
        install_model(installed, "qwen3:8b", runtime=RuntimeConfig(context=16384))
        client.post("/api/chat", json={"model": "qwen3:8b", "stream": False,
                                       "messages": [{"role": "user", "content": "x"}]})
        assert client.get("/api/ps").json()["models"][0]["context_length"] == 16384


# --- Inférence -----------------------------------------------------------------------------------


class TestChat:
    def test_reponse_non_streamee(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "user", "content": "bonjour"}],
        }).json()

        assert body["model"] == "qwen3:8b"
        assert body["done"] is True
        assert body["done_reason"] == "stop"
        assert body["message"]["role"] == "assistant"
        assert body["message"]["content"] == "echo: bonjour"
        assert body["created_at"]

    def test_metriques_en_nanosecondes(self, client, installed):
        """Risque R3 : un client divise `eval_count` par `eval_duration` puis multiplie par 10⁹."""
        install_model(installed, "qwen3:8b")
        body = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "user", "content": "x"}],
        }).json()

        assert body["prompt_eval_count"] == 11
        assert body["eval_count"] == 7
        # 42 ms côté faux serveur → 42 000 000 ns. Un chiffre en secondes vaudrait 0.042.
        assert body["eval_duration"] == 42_000_000
        assert body["prompt_eval_duration"] == 12_500_000
        assert all(isinstance(body[k], int) for k in
                   ("total_duration", "prompt_eval_duration", "eval_duration"))

    def test_streaming_ndjson(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "bonjour le monde"}],
        })
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")

        chunks = ndjson(response)
        assert len(chunks) > 1
        assert all(c["done"] is False for c in chunks[:-1])
        assert chunks[-1]["done"] is True
        assemble = "".join(c["message"]["content"] for c in chunks[:-1])
        assert assemble == "echo: bonjour le monde"

    def test_dernier_chunk_ne_repete_pas_le_contenu(self, client, installed):
        """Répéter le contenu ferait doubler la réponse chez tout client qui concatène."""
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "salut"}],
        })
        assert ndjson(response)[-1]["message"]["content"] == ""

    def test_streaming_par_defaut(self, client, installed):
        """`Stream *bool` absent vaut `true` chez Ollama."""
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "x"}],
        })
        assert response.headers["content-type"].startswith("application/x-ndjson")

    def test_appels_doutils(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "user", "content": "météo ?"}],
            "tools": [{"type": "function", "function": {
                "name": "meteo", "description": "météo",
                "parameters": {"type": "object", "properties": {}}}}],
        }).json()

        calls = body["message"]["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "meteo"
        assert calls[0]["function"]["arguments"] == {"ok": True}
        assert calls[0]["id"] == "call_fake_1"

    def test_done_reason_reste_stop_avec_outils(self, client, installed):
        """Ollama ne connaît pas `tool_calls` comme raison d'arrêt : il signale par `tool_calls`."""
        install_model(installed, "qwen3:8b")
        body = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        }).json()
        assert body["done_reason"] == "stop"

    def test_resultat_doutil_accepte(self, client, installed):
        """Un message de rôle `tool` doit être accepté et corrélé, pas dégradé."""
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [
                {"role": "user", "content": "météo ?"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_1",
                                 "function": {"name": "meteo", "arguments": {"ville": "Paris"}}}]},
                {"role": "tool", "tool_call_id": "call_1", "tool_name": "meteo",
                 "content": "18°C"},
            ],
        })
        assert response.status_code == 200

    def test_role_normalise_en_minuscules(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "USER", "content": "x"}],
        })
        assert response.status_code == 200

    def test_rejet_dun_role_inconnu(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "messages": [{"role": "wizard", "content": "x"}],
        })
        assert response.status_code == 400
        assert "error" in response.json()

    def test_dechargement_par_keep_alive_zero(self, client, installed):
        """`messages: []` + `keep_alive: 0` décharge le modèle (`docs/api.md` l. 1141-1149)."""
        install_model(installed, "qwen3:8b")
        client.post("/api/chat", json={"model": "qwen3:8b", "stream": False,
                                       "messages": [{"role": "user", "content": "x"}]})
        assert len(client.get("/api/ps").json()["models"]) == 1

        body = client.post("/api/chat", json={"model": "qwen3:8b", "messages": [],
                                              "keep_alive": 0}).json()
        assert body["done_reason"] == "unload"
        assert client.get("/api/ps").json()["models"] == []

    def test_chargement_sans_generation(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/api/chat", json={"model": "qwen3:8b", "messages": [],
                                              "keep_alive": "5m"}).json()
        assert body["done_reason"] == "load"
        assert len(client.get("/api/ps").json()["models"]) == 1


class TestGenerate:
    def test_reponse_non_streamee(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/api/generate", json={
            "model": "qwen3:8b", "prompt": "bonjour", "stream": False,
        }).json()
        assert body["response"] == "echo: bonjour"
        assert body["done"] is True

    def test_streaming_reponse_finale_vide(self, client, installed):
        """« `response`: empty if the response was streamed » (`docs/api.md`)."""
        install_model(installed, "qwen3:8b")
        chunks = ndjson(client.post("/api/generate", json={
            "model": "qwen3:8b", "prompt": "bonjour",
        }))
        assert chunks[-1]["response"] == ""
        assert "".join(c["response"] for c in chunks[:-1]) == "echo: bonjour"

    def test_system_pris_en_compte(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/api/generate", json={
            "model": "qwen3:8b", "prompt": "x", "system": "tu es utile", "stream": False,
        })
        assert response.status_code == 200

    def test_raw_incompatible_avec_system(self, client, installed):
        """Message repris tel quel d'Ollama (`server/routes.go` l. 428)."""
        install_model(installed, "qwen3:8b")
        response = client.post("/api/generate", json={
            "model": "qwen3:8b", "prompt": "x", "raw": True, "system": "s",
        })
        assert response.status_code == 400
        assert response.json()["error"] == "raw mode does not support template, system, or context"


class TestOptions:
    """Risque R2 : contrat dur avec `ollama-gateway`."""

    def test_num_ctx_accepte(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "user", "content": "x"}],
            "options": {"num_ctx": 2048},
        })
        assert response.status_code == 200, "un 400 casserait toutes les clés à plafond"

    def test_num_ctx_atteint_reellement_linstance(self, client, installed):
        """Le plafond doit devenir `--ctx-size`, sinon il serait silencieusement sans effet."""
        install_model(installed, "qwen3:8b", runtime=RuntimeConfig(context=16384))
        client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "user", "content": "x"}],
            "options": {"num_ctx": 2048},
        })
        assert client.get("/api/ps").json()["models"][0]["context_length"] == 2048

    def test_options_dechantillonnage_acceptees(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "user", "content": "x"}],
            "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 40,
                        "seed": 42, "stop": ["\n\n"], "num_predict": 128},
        })
        assert response.status_code == 200

    def test_option_inconnue_ne_casse_pas(self, client, installed):
        """Une option d'une version ultérieure d'Ollama doit traverser sans erreur."""
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "user", "content": "x"}],
            "options": {"option_du_futur": 1},
        })
        assert response.status_code == 200


class TestEmbeddings:
    def test_embed_forme_plurielle(self, client, installed):
        install_model(installed, "embed", runtime=RuntimeConfig(embedding=True))
        body = client.post("/api/embed", json={"model": "embed", "input": "bonjour"}).json()
        assert body["model"] == "embed:latest"
        assert len(body["embeddings"]) == 1
        assert isinstance(body["embeddings"][0], list)

    def test_embed_entrees_multiples(self, client, installed):
        install_model(installed, "embed", runtime=RuntimeConfig(embedding=True))
        body = client.post("/api/embed", json={"model": "embed", "input": ["a", "bb"]}).json()
        assert len(body["embeddings"]) == 2

    def test_embeddings_legacy_singulier(self, client, installed):
        """Schéma volontairement différent : `{"embedding": [...]}`, sans modèle ni métriques."""
        install_model(installed, "embed", runtime=RuntimeConfig(embedding=True))
        body = client.post("/api/embeddings", json={"model": "embed", "prompt": "x"}).json()
        assert set(body) == {"embedding"}
        assert isinstance(body["embedding"], list)

    def test_input_manquant(self, client, installed):
        install_model(installed, "embed", runtime=RuntimeConfig(embedding=True))
        response = client.post("/api/embed", json={"model": "embed"})
        assert response.status_code == 400


# --- Plan de contrôle --------------------------------------------------------------------------------


class TestPlanDeControle:
    def test_copy(self, client, installed):
        install_model(installed, "source:latest")
        response = client.post("/api/copy", json={"source": "source:latest",
                                                  "destination": "copie:v2"})
        assert response.status_code == 200
        noms = [m["name"] for m in client.get("/api/tags").json()["models"]]
        assert "copie:v2" in noms

    def test_copy_source_absente(self, client):
        response = client.post("/api/copy", json={"source": "fantome", "destination": "x"})
        assert response.status_code == 404

    def test_delete(self, client, installed):
        install_model(installed, "jetable")
        assert client.request("DELETE", "/api/delete",
                              json={"model": "jetable"}).status_code == 200
        assert client.get("/api/tags").json()["models"] == []

    def test_delete_modele_absent_renvoie_404(self, client):
        """`ollama-gateway` interprète 404 comme « modèle déjà absent » (`app/servers.py` l. 465)."""
        response = client.request("DELETE", "/api/delete", json={"model": "fantome"})
        assert response.status_code == 404

    def test_delete_decharge_avant_suppression(self, client, installed):
        install_model(installed, "jetable")
        client.post("/api/chat", json={"model": "jetable", "stream": False,
                                       "messages": [{"role": "user", "content": "x"}]})
        client.request("DELETE", "/api/delete", json={"model": "jetable"})
        assert client.get("/api/ps").json()["models"] == []

    def test_blobs_tete_et_creation(self, client):
        import hashlib

        contenu = b"contenu de blob"
        digest = "sha256:" + hashlib.sha256(contenu).hexdigest()

        assert client.head(f"/api/blobs/{digest}").status_code == 404
        assert client.post(f"/api/blobs/{digest}", content=contenu).status_code == 201
        assert client.head(f"/api/blobs/{digest}").status_code == 200

    def test_blob_checksum_non_conforme_refuse(self, client):
        faux = "sha256:" + "0" * 64
        response = client.post(f"/api/blobs/{faux}", content=b"autre chose")
        assert response.status_code == 400
        assert client.head(f"/api/blobs/{faux}").status_code == 404

    def test_blob_digest_malforme_refuse(self, client):
        assert client.post("/api/blobs/pas-un-digest", content=b"x").status_code == 400

    def test_create_depuis_un_modele_existant(self, client, installed):
        install_model(installed, "base")
        response = client.post("/api/create", json={
            "model": "derive", "from": "base", "system": "tu es utile", "stream": False,
        })
        assert response.status_code == 200
        assert response.json() == {"status": "success"}
        body = client.post("/api/show", json={"model": "derive"}).json()
        assert body["system"] == "tu es utile"

    def test_create_depuis_des_blobs(self, client):
        import hashlib

        from .ggufbuild import DEFAULT_KV, build_gguf

        contenu = build_gguf(DEFAULT_KV)
        digest = "sha256:" + hashlib.sha256(contenu).hexdigest()
        client.post(f"/api/blobs/{digest}", content=contenu)

        response = client.post("/api/create", json={
            "model": "depuis-blob", "files": {"modele.gguf": digest}, "stream": False,
        })
        assert response.status_code == 200
        assert client.post("/api/show", json={"model": "depuis-blob"}).status_code == 200

    def test_create_blob_absent(self, client):
        digest = "sha256:" + "0" * 64
        response = client.post("/api/create", json={
            "model": "x", "files": {"m.gguf": digest}, "stream": False,
        })
        assert response.status_code == 400
        assert "upload it first" in response.json()["error"]

    def test_create_quantisation_refusee_explicitement(self, client, installed):
        """Capacité absente : refus explicite plutôt qu'ignorance silencieuse."""
        install_model(installed, "base")
        response = client.post("/api/create", json={
            "model": "q", "from": "base", "quantize": "q4_K_M", "stream": False,
        })
        assert response.status_code == 501
        assert "quantization" in response.json()["error"]

    def test_push_hors_perimetre(self, client):
        response = client.post("/api/push", json={"model": "x"})
        assert response.status_code == 501
        assert "not supported" in response.json()["error"]

    def test_gestion_desactivable(self, config, installed):
        """Le verrou est appliqué côté serveur, pas seulement en amont (DAT §6)."""
        import dataclasses
        import warnings

        from fastapi.testclient import TestClient

        from ollamacpp.app import create_app

        verrouille = dataclasses.replace(config, management_enabled=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with TestClient(create_app(verrouille)) as c:
                assert c.post("/api/pull", json={"model": "x"}).status_code == 403
                assert c.request("DELETE", "/api/delete",
                                 json={"model": "x"}).status_code == 403
                assert c.post("/api/copy", json={"source": "a",
                                                 "destination": "b"}).status_code == 403
                # L'inférence et la lecture restent servies.
                assert c.get("/api/tags").status_code == 200


class TestKeepAlivePlanDeControle:
    def test_keep_alive_du_manifest_visible_dans_ps(self, client, installed):
        install_model(installed, "qwen3:8b", lifecycle_config=LifecycleConfig(keep_alive="1h"))
        client.post("/api/chat", json={"model": "qwen3:8b", "stream": False,
                                       "messages": [{"role": "user", "content": "x"}]})
        assert client.get("/api/ps").json()["models"][0]["expires_at"]

    @pytest.mark.parametrize("valeur", [0, 30, "5m", "1h30m", -1])
    def test_formats_de_keep_alive_acceptes(self, client, installed, valeur):
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False, "keep_alive": valeur,
            "messages": [{"role": "user", "content": "x"}],
        })
        assert response.status_code == 200

    def test_keep_alive_invalide_refuse(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "keep_alive": "toujours",
            "messages": [{"role": "user", "content": "x"}],
        })
        assert response.status_code == 400
