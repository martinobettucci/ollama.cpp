"""Configuration centralisée de `ollama.cpp`.

@spec docs/BACKLOG.md OC-010 « Configuration centralisée »
@spec docs/ollama.cpp-architecture.md §5.10 « Arborescence de données », §8 risques R6 et R9
@spec docs/DAT.md §1 « Composants », §6 « Authentification et autorisation », §7 « Sécurité »

Toute la configuration globale vit ici. Les paramètres **propres à un modèle** n'y sont pas :
ils appartiennent à son manifest (`docs/ollama.cpp-architecture.md` §5.5), conformément à la
règle de précédence `requête > manifest > GGUF > défauts`.

Aucune valeur métier n'est codée en dur ailleurs dans le projet, et aucun secret n'est écrit dans
le dépôt : les jetons ne sont lus que depuis l'environnement, et `redacted()` garantit qu'ils ne
peuvent pas être journalisés par accident (risque R9).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from .durations import DEFAULT_KEEP_ALIVE, KeepAlive, parse_keep_alive

#: Version annoncée par `GET /api/version`. Les clients Ollama comparent parfois cette valeur
#: pour activer des fonctionnalités ; elle reste donc configurable.
DEFAULT_REPORTED_VERSION = "0.12.0"

#: Noms des champs contenant un secret. Jamais journalisés, jamais sérialisés en clair.
SECRET_FIELDS = frozenset({"api_key", "registry_token", "hf_token"})


class ConfigError(ValueError):
    """Configuration invalide : le service refuse de démarrer plutôt que de deviner."""


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} doit être un entier, reçu {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} doit être >= {minimum}, reçu {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} doit être un nombre, reçu {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} doit être >= {minimum}, reçu {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} doit être un booléen, reçu {raw!r}")


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration effective du service. Immuable une fois construite."""

    # --- Service HTTP ---
    host: str = "0.0.0.0"
    port: int = 11434
    reported_version: str = DEFAULT_REPORTED_VERSION
    log_level: str = "INFO"

    # --- Stockage ---
    models_dir: Path = field(default_factory=lambda: Path.home() / ".ollama.cpp" / "models")

    # --- Backend d'inférence ---
    llama_server_bin: str = "llama-server"
    llama_server_host: str = "127.0.0.1"
    llama_server_port_min: int = 18000
    llama_server_port_max: int = 18999
    load_timeout_s: float = 300.0
    request_timeout_s: float = 600.0

    # --- Cycle de vie et ordonnancement ---
    default_keep_alive: KeepAlive = DEFAULT_KEEP_ALIVE
    max_loaded_models: int = 3
    memory_limit_bytes: int = 0  # 0 = déduit de la mémoire de l'hôte
    memory_safety_margin: float = 0.10
    default_context: int = 4096

    # --- Téléchargement ---
    download_concurrency: int = 2
    registry_url: str = ""
    registry_token: str = ""
    hf_endpoint: str = "https://huggingface.co"
    hf_token: str = ""

    # --- Garde-fous ---
    api_key: str = ""
    management_enabled: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """Construit la configuration depuis l'environnement, en validant chaque valeur.

        Une valeur invalide lève `ConfigError` au démarrage plutôt que de produire un
        comportement dégradé silencieux en production.
        """
        port_min = _env_int("OLLAMACPP_LLAMA_SERVER_PORT_MIN", 18000, minimum=1)
        port_max = _env_int("OLLAMACPP_LLAMA_SERVER_PORT_MAX", 18999, minimum=1)
        if port_max < port_min:
            raise ConfigError(
                "OLLAMACPP_LLAMA_SERVER_PORT_MAX doit être >= OLLAMACPP_LLAMA_SERVER_PORT_MIN"
            )

        margin = _env_float("OLLAMACPP_MEMORY_SAFETY_MARGIN", 0.10, minimum=0.0)
        if margin >= 1.0:
            raise ConfigError("OLLAMACPP_MEMORY_SAFETY_MARGIN doit être < 1.0")

        keep_alive_raw = os.environ.get("OLLAMACPP_KEEP_ALIVE")
        default_keep_alive = (
            DEFAULT_KEEP_ALIVE if not keep_alive_raw else parse_keep_alive(keep_alive_raw)
        )

        return cls(
            host=_env_str("OLLAMACPP_HOST", "0.0.0.0"),
            port=_env_int("OLLAMACPP_PORT", 11434, minimum=1),
            reported_version=_env_str("OLLAMACPP_VERSION", DEFAULT_REPORTED_VERSION),
            log_level=_env_str("OLLAMACPP_LOG_LEVEL", "INFO").upper(),
            models_dir=Path(
                _env_str("OLLAMACPP_MODELS", str(Path.home() / ".ollama.cpp" / "models"))
            ).expanduser(),
            llama_server_bin=_env_str("OLLAMACPP_LLAMA_SERVER_BIN", "llama-server"),
            llama_server_host=_env_str("OLLAMACPP_LLAMA_SERVER_HOST", "127.0.0.1"),
            llama_server_port_min=port_min,
            llama_server_port_max=port_max,
            load_timeout_s=_env_float("OLLAMACPP_LOAD_TIMEOUT_S", 300.0, minimum=1.0),
            request_timeout_s=_env_float("OLLAMACPP_REQUEST_TIMEOUT_S", 600.0, minimum=1.0),
            default_keep_alive=default_keep_alive,
            max_loaded_models=_env_int("OLLAMACPP_MAX_LOADED_MODELS", 3, minimum=1),
            memory_limit_bytes=_env_int("OLLAMACPP_MEMORY_LIMIT_BYTES", 0, minimum=0),
            memory_safety_margin=margin,
            default_context=_env_int("OLLAMACPP_DEFAULT_CONTEXT", 4096, minimum=1),
            download_concurrency=_env_int("OLLAMACPP_DOWNLOAD_CONCURRENCY", 2, minimum=1),
            registry_url=_env_str("OLLAMACPP_REGISTRY_URL", ""),
            registry_token=os.environ.get("OLLAMACPP_REGISTRY_TOKEN", ""),
            hf_endpoint=_env_str("OLLAMACPP_HF_ENDPOINT", "https://huggingface.co"),
            hf_token=os.environ.get("OLLAMACPP_HF_TOKEN", ""),
            api_key=os.environ.get("OLLAMACPP_API_KEY", ""),
            management_enabled=_env_bool("OLLAMACPP_MANAGEMENT_ENABLED", True),
        )

    # --- Chemins dérivés -----------------------------------------------------------------------

    @property
    def blobs_dir(self) -> Path:
        return self.models_dir / "blobs"

    @property
    def manifests_dir(self) -> Path:
        return self.models_dir / "manifests"

    @property
    def tmp_dir(self) -> Path:
        return self.models_dir / "tmp"

    def ensure_layout(self) -> None:
        """Crée l'arborescence de stockage si elle manque (idempotent)."""
        for path in (self.blobs_dir, self.manifests_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)

    # --- Journalisation sûre --------------------------------------------------------------------

    def redacted(self) -> dict[str, object]:
        """Vue journalisable de la configuration : les secrets sont masqués (risque R9).

        Le masque distingue « non défini » de « défini » — utile au diagnostic — sans jamais
        révéler ni la valeur ni sa longueur.
        """
        out: dict[str, object] = {}
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.name in SECRET_FIELDS:
                out[spec.name] = "<set>" if value else "<unset>"
            elif isinstance(value, Path):
                out[spec.name] = str(value)
            elif isinstance(value, KeepAlive):
                out[spec.name] = str(value)
            else:
                out[spec.name] = value
        return out
