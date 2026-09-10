"""Assemblage des composants du service.

@spec docs/BACKLOG.md OC-010 « Configuration centralisée », OC-033 « ModelLifecycleManager »
@spec docs/ollama.cpp-architecture.md §5.1 « Vue d'ensemble »
@spec docs/DAT.md §1 « Composants », §2 « Services et processus », §9 « Reprise »

Point unique de câblage entre le registre, le superviseur, l'ordonnanceur et le cycle de vie.
Les façades ne construisent jamais ces composants elles-mêmes : elles reçoivent le `Service`, ce
qui garantit que les quatre APIs partagent bien **le même** registre et le même runtime — exigence
centrale de la mission (§« il ne faut pas créer trois runtimes différents »).
"""

from __future__ import annotations

import asyncio
import contextlib

from .config import Config
from .observability import configure, emit
from .registry import ModelRegistry
from .runtime import (
    HostMemoryProbe,
    LlamaServerSupervisor,
    ModelLifecycleManager,
    ModelScheduler,
    resolve_memory_budget,
)
from .storage import BlobStore, ManifestStore

#: Période du balayage des `keep_alive` expirés. Assez fine pour que `/api/ps` reflète la réalité
#: à quelques secondes près, assez large pour rester négligeable en coût.
SWEEP_INTERVAL_S = 5.0


class Service:
    """État partagé du service : registre, cycle de vie, configuration."""

    def __init__(self, config: Config) -> None:
        self.config = config

        config.ensure_layout()
        blobs = BlobStore(config.blobs_dir, config.tmp_dir)
        blobs.ensure_layout()
        # Les téléchargements interrompus par un arrêt précédent sont collectés au démarrage
        # (DAT §9 « Reprise ») : ils ne seraient jamais repris et occuperaient le disque.
        removed = blobs.sweep_tmp()
        if removed:
            emit("startup_sweep", partial_downloads_removed=removed)

        self.registry = ModelRegistry(blobs, ManifestStore(config.manifests_dir))

        self.scheduler = ModelScheduler(
            memory_budget_bytes=resolve_memory_budget(
                HostMemoryProbe(),
                configured_limit=config.memory_limit_bytes,
                safety_margin=config.memory_safety_margin,
            ),
            max_loaded_models=config.max_loaded_models,
            eviction_grace_s=config.eviction_grace_s,
        )

        self.supervisor = LlamaServerSupervisor(
            binary=config.llama_server_bin,
            host=config.llama_server_host,
            port_min=config.llama_server_port_min,
            port_max=config.llama_server_port_max,
            load_timeout_s=config.load_timeout_s,
            request_timeout_s=config.request_timeout_s,
            default_context=config.default_context,
        )

        self.lifecycle = ModelLifecycleManager(
            registry=self.registry,
            supervisor=self.supervisor,
            scheduler=self.scheduler,
            config=config,
        )

        self._sweeper: asyncio.Task | None = None

    async def start(self) -> None:
        """Démarre les tâches de fond."""
        configure(self.config.log_level)
        emit("service_started", **{
            key: value for key, value in self.config.redacted().items()
            if key in {"port", "models_dir", "max_loaded_models", "default_keep_alive",
                       "llama_server_bin", "management_enabled"}
        })
        self._sweeper = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        """Arrête les tâches de fond et toutes les instances `llama-server`.

        Aucun processus enfant ne doit survivre à l'arrêt du service : sur GPU, un orphelin
        retiendrait plusieurs gigaoctets de VRAM jusqu'au redémarrage de la machine.
        """
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None
        await self.lifecycle.shutdown()

    async def _sweep_loop(self) -> None:
        """Décharge périodiquement les modèles dont le `keep_alive` a expiré.

        Sans cette boucle, `keep_alive` ne serait effectif qu'au moment où une autre requête
        réclame de la mémoire — ce qui n'est pas la sémantique d'Ollama, où un modèle inactif
        libère sa mémoire de lui-même.
        """
        while True:
            try:
                await asyncio.sleep(SWEEP_INTERVAL_S)
                await self.lifecycle.sweep()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - filet de sécurité
                # Une erreur de balayage ne doit jamais arrêter la boucle : le service
                # continuerait à tourner sans jamais plus décharger un modèle.
                emit("sweep_failed", detail=type(exc).__name__)
