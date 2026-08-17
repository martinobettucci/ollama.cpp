"""Schéma d'erreur unique, compatible Ollama, pour toutes les façades.

@spec docs/BACKLOG.md OC-011 « Schéma d'erreur compatible Ollama »
@spec docs/ollama.cpp-architecture.md §2.3 « Schéma d'erreur », §3.1 « Sonde de compatibilité »,
      §8 risque R1
@spec docs/DAT.md §1 « Composants » (module `ollamacpp/errors.py`)

Ollama répond `{"error": "<message>"}` sur l'intégralité de sa surface
(`server/routes.go`, révision auditée `d67ad83`). Reproduire ce schéma n'est pas cosmétique :
`ollama-gateway` s'en sert pour décider si un endpoint **existe**.

**Risque R1 — la sonde `_is_served` d'`ollama-gateway`.** La passerelle sonde chaque endpoint
avec un corps `{}` et applique la règle suivante (`app/servers.py` l. 301-312) :

    un chemin est « servi » sauf s'il répond 404 sans le mot « model » dans le corps

Conséquences pour `ollama.cpp` :

1. le `422` de validation par défaut de FastAPI ne casse pas la sonde (≠ 404), mais il ne
   correspond pas au comportement d'Ollama, qui répond `400 {"error": ...}` ;
2. un 404 générique — `{"detail":"Not Found"}` de Starlette — ferait apparaître l'endpoint comme
   **absent** dans la matrice de compatibilité de la passerelle, alors qu'il est bien servi ;
3. tout 404 émis par `ollama.cpp` doit donc mentionner le modèle, ce que fait naturellement le
   message d'Ollama `model '<nom>' not found`.

C'est pourquoi ce module centralise à la fois les erreurs métier et les gestionnaires globaux qui
réécrivent les erreurs de Starlette et de FastAPI au format Ollama.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class OllamaError(Exception):
    """Erreur applicative sérialisée au format Ollama `{"error": ...}`."""

    status_code = 500

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code

    def to_response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status_code, content={"error": self.message})


class BadRequest(OllamaError):
    status_code = 400


class ModelNotFound(OllamaError):
    """404 dont le message contient toujours le mot « model » (contrainte de la sonde, R1)."""

    status_code = 404

    def __init__(self, model: str) -> None:
        super().__init__(f"model '{model}' not found")
        self.model = model


class Forbidden(OllamaError):
    status_code = 403


class NotImplementedByDesign(OllamaError):
    """Endpoint hors périmètre assumé : répond un message explicite, jamais un 404 de routeur.

    Voir `docs/ollama.cpp-architecture.md` §5.3 pour la liste et les justifications.
    """

    status_code = 501


class UpstreamError(OllamaError):
    """Échec de l'instance `llama-server` amont, exposé sans divulguer l'infrastructure."""

    status_code = 502


# --- Messages normalisés ----------------------------------------------------------------------

MISSING_REQUEST_BODY = "missing request body"
INVALID_MODEL_NAME = "invalid model name"


def missing_body() -> BadRequest:
    return BadRequest(MISSING_REQUEST_BODY)


def invalid_model_name() -> BadRequest:
    return BadRequest(INVALID_MODEL_NAME)


def unsupported_capability(model: str, capability: str) -> BadRequest:
    """Reproduit `"<modèle>" does not support <capacité>` (`server/routes.go`)."""
    return BadRequest(f'"{model}" does not support {capability}')


# --- Gestionnaires globaux --------------------------------------------------------------------


def install_error_handlers(app: FastAPI) -> None:
    """Impose le schéma Ollama à toutes les erreurs de l'application.

    Sans cela, FastAPI renverrait `{"detail": ...}` en 422 sur un corps invalide et Starlette un
    404 générique sur un chemin inconnu — deux formes qu'aucun client Ollama ne sait lire, et
    dont la seconde fausse la matrice de compatibilité d'`ollama-gateway` (R1).
    """

    @app.exception_handler(OllamaError)
    async def _ollama_error(_: Request, exc: OllamaError) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Ollama valide à la main et répond 400, jamais 422. On aplatit la première erreur en un
        # message lisible plutôt que d'exposer la structure interne de FastAPI.
        errors = exc.errors()
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
            detail = first.get("msg", "invalid request")
            message = f"{location}: {detail}" if location else detail
        else:
            message = "invalid request"
        return JSONResponse(status_code=400, content={"error": message})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "error"
        return JSONResponse(status_code=exc.status_code, content={"error": detail})
