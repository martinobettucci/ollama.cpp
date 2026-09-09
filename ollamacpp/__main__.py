"""Point d'entrée : `python -m ollamacpp`.

@spec docs/BACKLOG.md OC-010 « Configuration centralisée », OC-091 « runDev / runStaging / runProd »
@spec docs/ollama.cpp-architecture.md §5.1 « Vue d'ensemble »
@spec docs/DAT.md §12 « Commandes »
@spec README.md « Commandes principales »
"""

from __future__ import annotations

import sys

from .config import Config, ConfigError
from .observability import configure, emit


def main() -> int:
    """Démarre le service. Une configuration invalide empêche le démarrage, avec un diagnostic."""
    try:
        config = Config.from_env()
    except (ConfigError, ValueError) as exc:
        print(f"configuration invalide : {exc}", file=sys.stderr)
        return 2

    configure(config.log_level)

    try:
        import uvicorn
    except ImportError:  # pragma: no cover - dépendance déclarée dans requirements.txt
        print("uvicorn est requis : pip install -r requirements.txt", file=sys.stderr)
        return 2

    from .app import create_app

    emit("starting", host=config.host, port=config.port)
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        # `ollama.cpp` supervise lui-même des processus enfants : un rechargement automatique
        # les laisserait orphelins.
        reload=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
