"""Tests du schéma d'erreur compatible Ollama.

@verifies docs/BACKLOG.md OC-011 « Schéma d'erreur compatible Ollama »
@verifies docs/ollama.cpp-architecture.md §2.3 « Schéma d'erreur », §3.1, risque R1

Le test central de ce fichier rejoue **la logique exacte** de `_is_served` d'`ollama-gateway`
(`app/servers.py` l. 301-312). C'est la garantie qu'un endpoint servi par `ollama.cpp` ne sera
jamais compté comme absent dans la matrice de compatibilité de la passerelle.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from ollamacpp.errors import (
    BadRequest,
    Forbidden,
    ModelNotFound,
    NotImplementedByDesign,
    OllamaError,
    UpstreamError,
    install_error_handlers,
    invalid_model_name,
    missing_body,
    unsupported_capability,
)


def is_served(status_code: int, body: str) -> bool:
    """Copie fidèle de `_is_served` d'`ollama-gateway` (`app/servers.py` l. 301-312).

    Reproduite ici volontairement plutôt qu'importée : `ollama-gateway` n'est pas une dépendance
    de `ollama.cpp`, et c'est justement le comportement figé de la passerelle qu'on veut tester.
    """
    if status_code != 404:
        return True
    return "model" in (body or "").lower()


class TestSchemaDerreur:
    def test_forme_ollama(self):
        assert BadRequest("boum").to_response().body == b'{"error":"boum"}'

    @pytest.mark.parametrize(
        "erreur, code",
        [
            (BadRequest("x"), 400),
            (Forbidden("x"), 403),
            (ModelNotFound("m"), 404),
            (NotImplementedByDesign("x"), 501),
            (UpstreamError("x"), 502),
            (OllamaError("x"), 500),
        ],
    )
    def test_codes_http(self, erreur: OllamaError, code: int):
        assert erreur.status_code == code


class TestMessagesNormalises:
    def test_modele_introuvable_reproduit_ollama(self):
        """Message repris tel quel de `server/routes.go` : `model '<nom>' not found`."""
        assert ModelNotFound("llama3:8b").message == "model 'llama3:8b' not found"

    def test_corps_manquant(self):
        assert missing_body().message == "missing request body"

    def test_nom_de_modele_invalide(self):
        assert invalid_model_name().status_code == 400

    def test_capacite_absente(self):
        """Format repris d'Ollama, guillemets compris."""
        erreur = unsupported_capability("llama3:8b", "thinking")
        assert erreur.message == '"llama3:8b" does not support thinking'


class TestSondeDeCompatibilite:
    """Risque R1 : tout 404 émis par `ollama.cpp` doit mentionner le modèle."""

    def test_modele_introuvable_reste_vu_comme_servi(self):
        erreur = ModelNotFound("")
        assert is_served(erreur.status_code, json.dumps({"error": erreur.message}))

    def test_modele_introuvable_avec_nom_vide(self):
        """Cas réel de la sonde : elle envoie `{}`, donc un nom de modèle vide."""
        corps = json.dumps({"error": ModelNotFound("").message})
        assert "model" in corps
        assert is_served(404, corps)

    def test_404_generique_serait_vu_comme_absent(self):
        """Contre-exemple qui justifie la règle : la forme Starlette casse la sonde."""
        assert not is_served(404, json.dumps({"detail": "Not Found"}))

    def test_404_de_routeur_gin_serait_vu_comme_absent(self):
        assert not is_served(404, "404 page not found")

    @pytest.mark.parametrize("code", [200, 400, 403, 422, 500, 501, 502])
    def test_tout_code_non_404_est_vu_comme_servi(self, code: int):
        assert is_served(code, "{}")


class CorpsRequis(BaseModel):
    """Modèle de corps déclaré au niveau module.

    Il ne peut pas être local à la fixture : ce fichier active `from __future__ import
    annotations`, donc FastAPI résout les annotations par `get_type_hints` sur les globales du
    module. Une classe locale y est introuvable et le paramètre serait pris pour un paramètre de
    requête au lieu d'un corps.
    """

    model: str


@pytest.fixture
def app_de_test() -> FastAPI:
    """Application minimale reproduisant les cas d'erreur d'une façade réelle."""
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/api/faux-endpoint")
    async def _endpoint(corps: CorpsRequis) -> dict:
        return {"model": corps.model}

    @app.post("/api/modele-inconnu")
    async def _inconnu() -> dict:
        raise ModelNotFound("fantome")

    @app.post("/api/hors-perimetre")
    async def _hors_perimetre() -> dict:
        raise NotImplementedByDesign("push is not supported by ollama.cpp")

    return app


class TestGestionnairesGlobaux:
    def test_erreur_de_validation_devient_400_au_format_ollama(self, app_de_test: FastAPI):
        """FastAPI répondrait 422 `{"detail": ...}` : Ollama répond 400 `{"error": ...}`."""
        with TestClient(app_de_test) as client:
            reponse = client.post("/api/faux-endpoint", json={})
        assert reponse.status_code == 400
        assert "error" in reponse.json()
        assert "detail" not in reponse.json()

    def test_message_de_validation_lisible(self, app_de_test: FastAPI):
        """Le message nomme le champ fautif sans exposer la structure interne de FastAPI."""
        with TestClient(app_de_test) as client:
            corps = client.post("/api/faux-endpoint", json={}).json()
        assert "model" in corps["error"]

    def test_erreur_metier_serialisee(self, app_de_test: FastAPI):
        with TestClient(app_de_test) as client:
            reponse = client.post("/api/modele-inconnu")
        assert reponse.status_code == 404
        assert reponse.json() == {"error": "model 'fantome' not found"}
        assert is_served(reponse.status_code, reponse.text)

    def test_endpoint_hors_perimetre_repond_501_explicite(self, app_de_test: FastAPI):
        """Un endpoint volontairement absent ne doit jamais ressembler à un chemin inexistant."""
        with TestClient(app_de_test) as client:
            reponse = client.post("/api/hors-perimetre")
        assert reponse.status_code == 501
        assert reponse.json()["error"] == "push is not supported by ollama.cpp"
        assert is_served(reponse.status_code, reponse.text)

    def test_chemin_reellement_inconnu_reste_au_format_ollama(self, app_de_test: FastAPI):
        """Même un vrai 404 doit être lisible par un client Ollama."""
        with TestClient(app_de_test) as client:
            reponse = client.post("/api/nexiste-pas")
        assert reponse.status_code == 404
        assert "error" in reponse.json()
