"""Conformance : contrat comportemental d'`ollama-gateway`.

@verifies docs/BACKLOG.md OC-080 « Tests de conformité Ollama », OC-081 « sonde _is_served »
@verifies docs/ollama.cpp-architecture.md §3.1 « Ce que la passerelle attend de l'amont »,
          §3.2 « Matrice de compatibilité », risques R1, R2, R3, R4
@verifies docs/DAT.md §5.1 « Interfaces exposées »

Ce fichier rejoue **la logique exacte** d'`ollama-gateway` (révision auditée `e26fe13`) contre
`ollama.cpp`. C'est la vérification qui donne son sens au projet : si elle passe, remplacer
`target = Ollama` par `target = ollama.cpp` ne demande aucune modification de la passerelle.

Chaque classe reproduit une fonction précise de la passerelle, citée en docstring, plutôt que de
reformuler son intention — c'est ce qui rend le test réellement probant.
"""

from __future__ import annotations

import json

import pytest

from ollamacpp.storage import RuntimeConfig

from .conftest import install_model

pytestmark = pytest.mark.conformance


# --- Reproductions littérales de la passerelle ------------------------------------------------------


def is_served(status_code: int, body: str) -> bool:
    """Copie de `app/servers.py::_is_served` (l. 301-312).

    « Un chemin est "servi" sauf s'il renvoie un 404 de ROUTEUR (chemin réellement absent). […]
    On le reconnaît au mot "model" dans le corps. »
    """
    if status_code != 404:
        return True
    return "model" in (body or "").lower()


def filter_models(content: bytes, allowed: set[str]) -> bytes:
    """Copie de `app/proxy.py::_filter_models` (l. 47-63)."""
    obj = json.loads(content)
    if isinstance(obj.get("models"), list):  # Ollama /api/tags
        obj["models"] = [m for m in obj["models"] if isinstance(m, dict)
                         and (m.get("name") in allowed or m.get("model") in allowed)]
    if isinstance(obj.get("data"), list):  # OpenAI/Anthropic /v1/models
        obj["data"] = [m for m in obj["data"]
                       if isinstance(m, dict) and m.get("id") in allowed]
    return json.dumps(obj).encode("utf-8")


def inject_num_ctx(body: bytes, limit: int) -> bytes:
    """Copie de `app/context.py::inject_num_ctx` (l. 198-222)."""
    obj = json.loads(body)
    opts = obj.get("options")
    if not isinstance(opts, dict):
        opts = {}
    current = opts.get("num_ctx")
    if isinstance(current, int) and not isinstance(current, bool) and current > 0:
        opts["num_ctx"] = min(current, limit)
    else:
        opts["num_ctx"] = limit
    obj["options"] = opts
    return json.dumps(obj).encode("utf-8")


#: Catalogue sondé par la passerelle : `app/apis.py::CATALOG` (l. 45-77).
#: Les familles image en sont exclues, `llama.cpp` ne générant pas d'images (architecture §5.3).
CATALOG = {
    "ollama": [
        ("GET", "/api/version"),
        ("GET", "/api/tags"),
        ("GET", "/api/ps"),
        ("POST", "/api/show"),
        ("POST", "/api/generate"),
        ("POST", "/api/chat"),
        ("POST", "/api/embed"),
        ("POST", "/api/embeddings"),
    ],
    "openai": [
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/responses"),
    ],
    "anthropic": [
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
    ],
}

TOUS_LES_ENDPOINTS = [(f, m, p) for f, entries in CATALOG.items() for m, p in entries]


# --- OC-081 : matrice de compatibilité --------------------------------------------------------------


class TestSondeDeCompatibilite:
    """Risque R1 : un 404 sans le mot « model » ferait passer l'endpoint pour absent."""

    @pytest.mark.parametrize("famille, methode, chemin", TOUS_LES_ENDPOINTS,
                             ids=[f"{m}:{p}" for _f, m, p in TOUS_LES_ENDPOINTS])
    def test_endpoint_vu_comme_servi(self, client, famille, methode, chemin):
        """Reproduit `_probe_endpoint` : corps `{}` pour les POST, puis `_is_served`."""
        if methode == "GET":
            response = client.get(chemin)
        else:
            response = client.post(chemin, json={})

        assert is_served(response.status_code, response.text), (
            f"{methode} {chemin} apparaîtrait comme ABSENT dans la matrice de la passerelle "
            f"(HTTP {response.status_code} : {response.text[:120]})"
        )

    @pytest.mark.parametrize("famille", sorted(CATALOG))
    def test_famille_entierement_servie(self, client, famille):
        """Une famille n'est compatible que si TOUS ses endpoints sont servis."""
        for methode, chemin in CATALOG[famille]:
            response = client.get(chemin) if methode == "GET" else client.post(chemin, json={})
            assert is_served(response.status_code, response.text), f"{chemin} manquant"

    def test_aucune_erreur_de_validation_fastapi(self, client):
        """Un 422 `{"detail": ...}` ne casse pas la sonde mais n'est pas la forme d'Ollama."""
        for _famille, methode, chemin in TOUS_LES_ENDPOINTS:
            if methode != "POST":
                continue
            response = client.post(chemin, json={})
            assert response.status_code != 422, f"{chemin} répond en 422 au lieu du format Ollama"
            assert "detail" not in response.json(), f"{chemin} expose la forme FastAPI"

    def test_chemin_reellement_absent_reste_distinguable(self, client):
        """Contre-épreuve : un chemin inexistant DOIT apparaître comme absent."""
        response = client.post("/api/chemin-qui-nexiste-pas", json={})
        assert response.status_code == 404
        assert not is_served(response.status_code, response.text)


# --- Sonde de disponibilité d'un serveur d'exécution -------------------------------------------------


class TestSondeDeDisponibilite:
    """Reproduit `app/servers.py::probe` (l. 251-266), criticité P0 de la matrice."""

    def test_serveur_vide_declare_en_ligne(self, client):
        response = client.get("/api/tags")
        assert response.status_code == 200
        data = response.json()
        modeles = [m.get("name") or m.get("model")
                   for m in data.get("models", []) if isinstance(m, dict)]
        assert modeles == []

    def test_modeles_remontes(self, client, installed):
        install_model(installed, "qwen3:8b")
        install_model(installed, "acme/autre:v1")
        data = client.get("/api/tags").json()
        modeles = sorted(m.get("name") or m.get("model") for m in data["models"])
        assert modeles == ["acme/autre:v1", "qwen3:8b"]


class TestFiltrageDesListings:
    """Reproduit `app/proxy.py::_filter_models` (l. 47-63)."""

    def test_filtrage_ollama(self, client, installed):
        install_model(installed, "autorise")
        install_model(installed, "interdit")
        filtre = filter_models(client.get("/api/tags").content, {"autorise:latest"})
        noms = [m["name"] for m in json.loads(filtre)["models"]]
        assert noms == ["autorise:latest"]

    def test_filtrage_openai(self, client, installed):
        install_model(installed, "autorise")
        install_model(installed, "interdit")
        filtre = filter_models(client.get("/v1/models").content, {"autorise:latest"})
        ids = [m["id"] for m in json.loads(filtre)["data"]]
        assert ids == ["autorise:latest"]

    def test_les_deux_champs_sont_exploitables(self, client, installed):
        """La passerelle lit `name` OU `model` : les deux doivent porter le nom complet."""
        install_model(installed, "qwen3:8b")
        entree = client.get("/api/tags").json()["models"][0]
        assert entree["name"] == entree["model"] == "qwen3:8b"


# --- OC-047 / risque R2 : injection de num_ctx ---------------------------------------------------------


class TestInjectionDuPlafondDeContexte:
    """Reproduit `app/context.py` : la passerelle RÉÉCRIT le corps des requêtes."""

    CHEMINS = ["/api/chat", "/api/generate", "/api/embed", "/api/embeddings"]

    def corps_pour(self, chemin: str) -> dict:
        if chemin == "/api/chat":
            return {"model": "qwen3:8b", "stream": False,
                    "messages": [{"role": "user", "content": "x"}]}
        if chemin == "/api/generate":
            return {"model": "qwen3:8b", "stream": False, "prompt": "x"}
        if chemin == "/api/embed":
            return {"model": "qwen3:8b", "input": "x"}
        return {"model": "qwen3:8b", "prompt": "x"}

    @pytest.mark.parametrize("chemin", CHEMINS)
    def test_corps_reecrit_accepte(self, client, installed, chemin):
        """Un refus casserait TOUTES les clés à plafond de contexte de la passerelle."""
        install_model(installed, "qwen3:8b", runtime=RuntimeConfig(embedding=True))
        corps = inject_num_ctx(json.dumps(self.corps_pour(chemin)).encode(), 2048)
        response = client.post(chemin, content=corps,
                               headers={"content-type": "application/json"})
        assert response.status_code == 200, f"{chemin} : {response.text[:150]}"

    def test_plafond_effectivement_applique(self, client, installed):
        """Un plafond accepté mais ignoré serait pire qu'un refus : silencieusement faux."""
        install_model(installed, "qwen3:8b", runtime=RuntimeConfig(context=32768))
        corps = inject_num_ctx(json.dumps(self.corps_pour("/api/chat")).encode(), 2048)
        client.post("/api/chat", content=corps, headers={"content-type": "application/json"})
        assert client.get("/api/ps").json()["models"][0]["context_length"] == 2048

    def test_le_plus_petit_des_deux_est_conserve(self, client, installed):
        """La passerelle garde `min(demandé, plafond)` : le comportement doit suivre."""
        install_model(installed, "qwen3:8b")
        corps = json.dumps({**self.corps_pour("/api/chat"),
                            "options": {"num_ctx": 1024}}).encode()
        client.post("/api/chat", content=inject_num_ctx(corps, 8192),
                    headers={"content-type": "application/json"})
        assert client.get("/api/ps").json()["models"][0]["context_length"] == 1024


# --- Gestion du catalogue depuis la console d'administration ---------------------------------------------


class TestGestionDuCatalogue:
    """Reproduit `app/servers.py::pull_model` (l. 402-433) et `delete_model` (l. 436-470)."""

    def test_delete_succes(self, client, installed):
        install_model(installed, "jetable")
        response = client.request("DELETE", "/api/delete", json={"model": "jetable"})
        assert response.status_code in (200, 204), "codes acceptés par la passerelle"

    def test_delete_absent_vaut_deja_supprime(self, client):
        """La passerelle traduit 404 en « modèle introuvable sur le serveur »."""
        assert client.request("DELETE", "/api/delete",
                              json={"model": "fantome"}).status_code == 404

    def test_pull_sans_source_configuree_echoue_lisiblement(self, client):
        """Sans registre privé ni référence `hf.co/…`, `pull` ne peut rien résoudre.

        La passerelle traite tout code ≠ 200 comme un échec et affiche le message : celui-ci doit
        donc expliquer quoi faire, pas seulement constater l'échec.
        """
        response = client.post("/api/pull", json={"model": "inconnu", "stream": False})
        assert response.status_code == 400
        message = response.json()["error"]
        assert "hf.co" in message and "OLLAMACPP_REGISTRY_URL" in message

    def test_installation_locale_par_blobs_et_create(self, client):
        """Chemin natif d'Ollama pour un GGUF local : `/api/blobs` puis `/api/create`.

        C'est aussi ce que fait la console d'administration quand elle n'utilise pas `pull`.
        """
        import hashlib

        from .ggufbuild import DEFAULT_KV, build_gguf

        contenu = build_gguf(DEFAULT_KV)
        digest = "sha256:" + hashlib.sha256(contenu).hexdigest()

        assert client.post(f"/api/blobs/{digest}", content=contenu).status_code == 201
        reponse = client.post("/api/create", json={
            "model": "local:v1", "files": {"modele.gguf": digest}, "stream": False,
        })
        assert reponse.status_code == 200
        assert reponse.json() == {"status": "success"}

        noms = [m.get("name") or m.get("model") for m in client.get("/api/tags").json()["models"]]
        assert "local:v1" in noms


# --- Conventions de réponse d'Ollama ----------------------------------------------------------------------


class TestConventionsOllama:
    """Conventions de `docs/api.md` : nanosecondes, NDJSON, schéma d'erreur."""

    def test_durees_en_nanosecondes(self, client, installed):
        install_model(installed, "qwen3:8b")
        body = client.post("/api/chat", json={
            "model": "qwen3:8b", "stream": False,
            "messages": [{"role": "user", "content": "x"}],
        }).json()
        for champ in ("total_duration", "prompt_eval_duration", "eval_duration"):
            assert isinstance(body[champ], int)
        # Un débit calculé par un client doit être plausible : eval_count / eval_duration * 1e9.
        debit = body["eval_count"] / body["eval_duration"] * 1e9
        assert 1 < debit < 100_000, f"débit implausible : {debit:g} tokens/s"

    def test_flux_ndjson_une_ligne_par_objet(self, client, installed):
        install_model(installed, "qwen3:8b")
        response = client.post("/api/chat", json={
            "model": "qwen3:8b", "messages": [{"role": "user", "content": "a b c"}],
        })
        assert response.headers["content-type"].startswith("application/x-ndjson")
        lignes = [l for l in response.text.splitlines() if l.strip()]
        for ligne in lignes:
            json.loads(ligne)  # chaque ligne doit être un objet JSON complet
        assert sum(1 for l in lignes if json.loads(l)["done"]) == 1

    def test_schema_derreur_uniforme(self, client):
        for chemin in ("/api/show", "/api/chat", "/api/generate", "/api/embed"):
            body = client.post(chemin, json={"model": "fantome"}).json()
            assert set(body) == {"error"}
            assert body["error"] == "model 'fantome' not found"

    def test_modele_inconnu_partout_en_404(self, client):
        for chemin in ("/api/show", "/api/chat", "/api/generate", "/api/embed",
                       "/api/embeddings", "/v1/chat/completions", "/v1/completions",
                       "/v1/embeddings", "/v1/responses", "/v1/messages"):
            response = client.post(chemin, json={"model": "fantome", "max_tokens": 8})
            assert response.status_code == 404, f"{chemin} → {response.status_code}"
