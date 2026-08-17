"""Application ASGI de `ollama.cpp`.

@spec docs/BACKLOG.md OC-010 « Configuration centralisée », OC-011 « Schéma d'erreur »,
      OC-040 à OC-055, OC-070 à OC-076
@spec docs/ollama.cpp-architecture.md §5.1 « Vue d'ensemble », §8 risque R1
@spec docs/DAT.md §1 « Composants », §6 « Authentification et autorisation »

Assemble les quatre façades sur un `Service` unique et impose le schéma d'erreur Ollama à
l'ensemble de l'application.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api import anthropic, ollama, openai, responses
from .config import Config
from .errors import install_error_handlers
from .service import Service

#: Chemins publics, servis sans clé API : ce sont les sondes de disponibilité. Les exiger
#: authentifiées empêcherait `ollama-gateway` de détecter qu'un serveur est en ligne.
PUBLIC_PATHS = frozenset({"/", "/api/version"})


def create_app(config: Config | None = None) -> FastAPI:
    """Construit l'application. `config` permet aux tests d'injecter un environnement isolé."""
    resolved = config or Config.from_env()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = Service(resolved)
        app.state.service = service
        await service.start()
        try:
            yield
        finally:
            # L'arrêt doit avoir lieu même si le démarrage a partiellement échoué : sinon des
            # processus `llama-server` survivraient au service.
            await service.stop()

    app = FastAPI(
        title="ollama.cpp",
        lifespan=lifespan,
        # Aucune documentation OpenAPI exposée : `ollama.cpp` imite Ollama, qui n'en sert pas, et
        # ces routes ajouteraient une surface sans usage pour ses clients.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.config = resolved

    install_error_handlers(app)
    _install_auth(app, resolved)

    app.include_router(ollama.router)
    app.include_router(openai.router)
    app.include_router(responses.router)
    app.include_router(anthropic.router)

    return app


def _install_auth(app: FastAPI, config: Config) -> None:
    """Contrôle d'accès par jeton, appliqué **côté serveur** (DAT §6).

    Inactif par défaut : `ollama.cpp` est conçu pour tourner derrière `ollama-gateway`, qui porte
    les clés API. Quand `OLLAMACPP_API_KEY` est défini, tout appel non public doit porter
    `Authorization: Bearer <clé>`.

    L'erreur renvoyée suit le schéma Ollama et mentionne l'autorisation, jamais la clé attendue.
    """
    if not config.api_key:
        return

    expected = f"Bearer {config.api_key}"

    @app.middleware("http")
    async def _require_api_key(request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        # Comparaison à temps constant : une comparaison naïve laisse fuir la clé octet par
        # octet à travers le temps de réponse.
        import hmac

        provided = request.headers.get("authorization", "")
        if not hmac.compare_digest(provided, expected):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return await call_next(request)


#: Application par défaut, utilisée par `uvicorn ollamacpp.app:app`.
app = create_app
