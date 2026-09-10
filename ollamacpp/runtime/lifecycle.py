"""Cycle de vie des modèles : états, chargement single-flight, `keep_alive`.

@spec docs/BACKLOG.md OC-033 « ModelLifecycleManager », OC-035 « Observabilité des décisions »
@spec docs/ollama.cpp-architecture.md §5.7 « États du cycle de vie », §5.8 « Scheduler »,
      §8 risques R5 et R6
@spec docs/DAT.md §3.2 « Chargement d'un modèle »

Machine à états (architecture §5.7) :

    NOT_PRESENT ──pull──► DOWNLOADING ──► UNLOADED ──load──► LOADING ──► READY
                               │                                │          │ ▲
                               └──────échec───────► FAILED ◄─────┘   requête│ │fin
                                                                            ▼ │
                                                            IDLE ◄──────── BUSY

Trois garanties structurent ce module.

**Single-flight (mission §19).** Deux requêtes concurrentes sur le même modèle partagent un
chargement unique. L'implémentation partage la *tâche* de chargement et l'attend sous
`asyncio.shield`, de sorte que l'annulation d'un client n'annule pas le chargement des autres —
un simple verrou ne l'aurait pas garanti.

**Un modèle incomplet n'atteint jamais `READY` (mission §14).** Un modèle multimodal dont le
`mmproj` manque échoue au chargement avec un message explicite, plutôt que d'accepter des images
qu'il ne peut pas traiter.

**La mémoire est réservée à l'admission, pas à la fin du chargement.** Un modèle en `LOADING`
compte déjà dans le budget et est marqué inévinçable : sans cela, deux chargements simultanés
pourraient chacun se croire admissibles et faire dépasser le budget.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..config import Config
from ..durations import KeepAlive
from ..errors import ModelNotFound, OllamaError, UpstreamError
from ..names import ModelRef
from ..observability import (
    EVENT_KEEP_ALIVE_EXPIRED,
    EVENT_LOAD_REQUESTED,
    EVENT_MODEL_IDLE,
    EVENT_REQUEST_ASSIGNED,
    EVENT_UNLOAD,
    EVENT_FAILURE,
    emit,
)
from ..registry import ModelRegistry, RegisteredModel
from ..storage.manifests import RuntimeConfig
from .capabilities import DetectedCapabilities, context_length, detect
from .memory import MemoryEstimate, estimate_model_memory
from .scheduler import ModelScheduler, ResidentInfo, REASON_ALL_BUSY
from .supervisor import LlamaServerInstance, LlamaServerSupervisor


class ModelState(str, Enum):
    """États du cycle de vie, tels que définis par la mission §17."""

    NOT_PRESENT = "not_present"
    DOWNLOADING = "downloading"
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    UNLOADING = "unloading"
    FAILED = "failed"


@dataclass(slots=True)
class Resident:
    """Modèle résident en mémoire, ou en cours de le devenir."""

    ref: ModelRef
    state: ModelState
    estimate: MemoryEstimate
    keep_alive: KeepAlive
    priority: int = 0
    last_used: float = 0.0
    active_requests: int = 0
    instance: LlamaServerInstance | None = None
    capabilities: DetectedCapabilities = field(default_factory=DetectedCapabilities)
    context_length: int = 0
    loaded_at: float = 0.0

    #: Configuration runtime effective ayant servi à lancer l'instance. Conservée pour détecter
    #: qu'une requête ultérieure demande une configuration différente (`options.num_ctx`), ce qui
    #: impose un rechargement — c'est la sémantique d'Ollama, où `num_ctx` est une option de
    #: *runner*, appliquée au chargement et non à la génération.
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def name(self) -> str:
        return self.ref.display_shortest()

    @property
    def is_serving(self) -> bool:
        return self.state in (ModelState.READY, ModelState.BUSY, ModelState.IDLE)

    def expires_at(self, now: float) -> float | None:
        """Instant d'expiration du `keep_alive`, ou `None` s'il est illimité."""
        if self.keep_alive.is_infinite:
            return None
        return self.last_used + self.keep_alive.seconds

    def is_expired(self, now: float) -> bool:
        """Vrai si le `keep_alive` a expiré et qu'aucune requête n'est en cours."""
        if self.active_requests > 0 or self.keep_alive.is_infinite:
            return False
        return now >= self.last_used + self.keep_alive.seconds


class ModelLifecycleManager:
    """Charge, maintient et décharge les modèles."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        supervisor: LlamaServerSupervisor,
        scheduler: ModelScheduler,
        config: Config,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._supervisor = supervisor
        self._scheduler = scheduler
        self._config = config
        self._clock = clock
        self._residents: dict[str, Resident] = {}
        self._loading: dict[str, asyncio.Task[Resident]] = {}
        self._admission_lock = asyncio.Lock()

    # --- Consultation ------------------------------------------------------------------------

    def residents(self) -> list[Resident]:
        """Modèles réellement résidents, triés par nom. Jamais un état simulé (mission §6)."""
        return sorted(
            (r for r in self._residents.values() if r.is_serving),
            key=lambda resident: resident.name,
        )

    def state(self, name: str) -> ModelState:
        """État d'un modèle, du point de vue du cycle de vie."""
        try:
            ref = self._registry.resolve(name)
        except ModelNotFound:
            return ModelState.NOT_PRESENT
        key = ref.display_shortest()
        resident = self._residents.get(key)
        if resident is not None:
            return resident.state
        if not self._registry.exists(name):
            return ModelState.NOT_PRESENT
        return ModelState.UNLOADED

    def get_resident(self, name: str) -> Resident | None:
        try:
            key = self._registry.resolve(name).display_shortest()
        except ModelNotFound:
            return None
        resident = self._residents.get(key)
        return resident if resident is not None and resident.is_serving else None

    # --- Chargement --------------------------------------------------------------------------

    async def ensure_ready(
        self,
        name: str,
        keep_alive: KeepAlive | None = None,
        runtime_override: RuntimeConfig | None = None,
    ) -> Resident:
        """Garantit qu'un modèle est chargé et prêt, en un seul chargement partagé.

        `runtime_override` porte les options qui configurent l'**instance** et non la génération
        — au premier chef `options.num_ctx`, qu'`ollama-gateway` injecte pour plafonner le
        contexte d'une clé (risque R2). Si le modèle est déjà résident avec une configuration
        différente, il est **rechargé** : c'est la sémantique d'Ollama, où ces options sont des
        options de *runner*. Sans ce rechargement, le plafond de la passerelle serait
        silencieusement sans effet.

        Lève `ModelNotFound` si le modèle n'est pas installé, `UpstreamError` s'il ne peut pas
        être chargé. Ne renvoie jamais un modèle en cours de chargement.
        """
        model = self._registry.get(name)  # lève ModelNotFound si absent
        key = model.ref.display_shortest()
        wanted = self._effective_runtime(model, runtime_override)

        while True:
            resident = self._residents.get(key)
            if resident is not None and resident.is_serving:
                if resident.runtime == wanted:
                    self._touch(resident, keep_alive)
                    return resident
                if resident.active_requests > 0:
                    # Recharger sous une requête en cours la tuerait. On sert la requête avec la
                    # configuration en place et on le signale : c'est visible, jamais silencieux.
                    emit(
                        EVENT_FAILURE,
                        model=key,
                        reason="runtime_override_ignored_while_busy",
                        active=resident.active_requests,
                    )
                    self._touch(resident, keep_alive)
                    return resident
                emit(EVENT_UNLOAD, model=key, reason="runtime_override_changed")
                await self._unload_key(key)
                continue

            task = self._loading.get(key)
            if task is None:
                emit(EVENT_LOAD_REQUESTED, model=key, keep_alive=str(
                    keep_alive or self._effective_keep_alive(model, None)))
                task = asyncio.create_task(self._load(model, keep_alive, wanted))
                self._loading[key] = task
                task.add_done_callback(lambda _t, k=key: self._loading.pop(k, None))

            # `shield` : l'annulation d'un client ne doit pas annuler le chargement partagé par
            # les autres. Sans lui, un client qui abandonne casserait toutes les requêtes en
            # attente du même modèle.
            resident = await asyncio.shield(task)
            self._touch(resident, keep_alive)
            return resident

    def _effective_runtime(
        self, model: RegisteredModel, override: RuntimeConfig | None
    ) -> RuntimeConfig:
        """Applique la précédence `requête > manifest` sur la configuration runtime (§5.6)."""
        return build_runtime_config(model.manifest.runtime, override)

    async def _load(
        self,
        model: RegisteredModel,
        keep_alive: KeepAlive | None,
        runtime: RuntimeConfig | None = None,
    ) -> Resident:
        """Charge effectivement un modèle : admission, éviction, démarrage, détection."""
        key = model.ref.display_shortest()

        if not model.is_complete:
            # Mission §14 : un artefact obligatoire manquant interdit l'état READY.
            missing = ", ".join(model.missing_artifacts)
            emit(EVENT_FAILURE, model=key, reason="missing_artifacts", artifacts=missing)
            raise UpstreamError(f"model '{key}' is missing required artifacts")

        runtime = runtime if runtime is not None else model.manifest.runtime
        effective_context = context_length(None, model.metadata, self._config.default_context)
        if runtime.context:
            effective_context = runtime.context

        estimate = estimate_model_memory(
            artifacts_bytes=model.size,
            metadata=model.metadata,
            context=effective_context,
            cache_type_k=runtime.cache_type_k,
            cache_type_v=runtime.cache_type_v,
            parallel=runtime.parallel or 1,
        )
        resolved_keep_alive = self._effective_keep_alive(model, keep_alive)
        priority = model.manifest.lifecycle.priority

        # --- Admission : sous verrou, pour que deux chargements concurrents ne se croient pas
        # tous deux admissibles avec la même mémoire libre.
        async with self._admission_lock:
            plan = self._scheduler.plan(
                name=key, required_bytes=estimate.total_bytes, residents=self._resident_infos()
            )
            if not plan.admitted:
                # Deux échecs très différents, deux messages : « réessaie » quand la place est
                # prise par un modèle qui travaille, « revois la configuration » quand le modèle
                # ne tient pas même seul. Un message unique ferait chercher au mauvais endroit.
                if plan.reason == REASON_ALL_BUSY:
                    raise UpstreamError(
                        f"cannot load model '{key}': another model is currently serving requests "
                        f"and cannot be evicted — retry in a moment"
                    )
                raise UpstreamError(
                    f"cannot load model '{key}': not enough memory "
                    f"({estimate.total_bytes} bytes required)"
                )
            for decision in plan.evictions:
                ModelScheduler.log(decision)
                await self._unload_key(decision.model)

            # Réservation : le modèle compte dans le budget dès maintenant, et il est épinglé le
            # temps du chargement.
            placeholder = Resident(
                ref=model.ref,
                state=ModelState.LOADING,
                estimate=estimate,
                keep_alive=resolved_keep_alive,
                priority=priority,
                last_used=self._clock(),
                runtime=runtime,
            )
            self._residents[key] = placeholder

        try:
            instance = await self._supervisor.spawn(
                name=key,
                model_path=self._registry.blobs.path(model.manifest.artifacts.model),
                runtime=runtime,
                mmproj_path=self._artifact_path(model.manifest.artifacts.mmproj),
                draft_path=self._artifact_path(model.manifest.artifacts.draft),
                adapter_paths=tuple(
                    self._registry.blobs.path(digest)
                    for digest in model.manifest.artifacts.adapters
                ),
            )
        except BaseException as exc:
            # La réservation doit disparaître quoi qu'il arrive, sinon le budget mémoire fuit
            # définitivement à chaque échec de chargement.
            self._residents.pop(key, None)
            if isinstance(exc, OllamaError):
                emit(EVENT_FAILURE, model=key, reason="spawn_failed", detail=exc.message)
            raise

        now = self._clock()
        placeholder.instance = instance
        placeholder.state = ModelState.READY
        placeholder.loaded_at = now
        placeholder.last_used = now
        placeholder.context_length = context_length(
            instance.props, model.metadata, effective_context
        )
        placeholder.capabilities = detect(
            model.manifest,
            props=instance.props,
            metadata=model.metadata,
            has_mmproj=bool(model.manifest.artifacts.mmproj),
        )
        return placeholder

    def _artifact_path(self, digest: str | None):
        return self._registry.blobs.path(digest) if digest else None

    def _effective_keep_alive(
        self, model: RegisteredModel, requested: KeepAlive | None
    ) -> KeepAlive:
        """Applique la précédence `requête > manifest > configuration globale` (architecture §5.6)."""
        if requested is not None:
            return requested
        from_manifest = model.manifest.lifecycle.resolved_keep_alive()
        return from_manifest if from_manifest is not None else self._config.default_keep_alive

    # --- Usage --------------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def acquire(
        self,
        name: str,
        keep_alive: KeepAlive | None = None,
        runtime_override: RuntimeConfig | None = None,
    ) -> AsyncIterator[Resident]:
        """Réserve un modèle pour la durée d'une requête.

        Le modèle passe `BUSY` pendant l'exécution — donc inévinçable — puis `IDLE`. Le
        `try/finally` est essentiel : sans lui, une requête interrompue laisserait le modèle
        éternellement BUSY, donc jamais déchargeable.
        """
        resident = await self.ensure_ready(name, keep_alive, runtime_override)
        resident.active_requests += 1
        resident.state = ModelState.BUSY
        emit(EVENT_REQUEST_ASSIGNED, model=resident.name, active=resident.active_requests)
        try:
            yield resident
        finally:
            resident.active_requests = max(0, resident.active_requests - 1)
            resident.last_used = self._clock()
            if resident.active_requests == 0 and resident.state is ModelState.BUSY:
                resident.state = ModelState.IDLE
                emit(EVENT_MODEL_IDLE, model=resident.name)

            # `keep_alive: 0` décharge dès la fin de la requête (sémantique Ollama, risque R5).
            if resident.active_requests == 0 and resident.keep_alive.unloads_immediately:
                await self._unload_key(resident.name)

    def _touch(self, resident: Resident, keep_alive: KeepAlive | None) -> None:
        """Rafraîchit `last_used` et applique un `keep_alive` fourni par la requête.

        Une requête peut **changer** la politique de résidence d'un modèle déjà chargé : c'est le
        comportement d'Ollama, où `keep_alive` accompagne chaque appel d'inférence.
        """
        resident.last_used = self._clock()
        if keep_alive is not None:
            resident.keep_alive = keep_alive
        if resident.state is ModelState.IDLE:
            resident.state = ModelState.READY

    # --- Déchargement ---------------------------------------------------------------------------

    async def unload(self, name: str) -> bool:
        """Décharge un modèle. Renvoie `False` s'il n'était pas résident."""
        try:
            key = self._registry.resolve(name).display_shortest()
        except ModelNotFound:
            return False
        return await self._unload_key(key)

    async def _unload_key(self, key: str) -> bool:
        resident = self._residents.get(key)
        if resident is None:
            return False

        resident.state = ModelState.UNLOADING
        instance = resident.instance
        # Le retrait de la table précède l'arrêt du processus : une requête arrivant pendant
        # l'arrêt doit repartir sur un chargement neuf, jamais réutiliser une instance mourante.
        self._residents.pop(key, None)

        if instance is not None:
            await self._supervisor.terminate(instance)
        emit(EVENT_UNLOAD, model=key)
        return True

    async def sweep(self) -> list[str]:
        """Décharge les modèles dont le `keep_alive` a expiré. Renvoie les noms déchargés.

        Appelée périodiquement : sans elle, `keep_alive` ne serait effectif qu'au moment où une
        autre requête réclame de la mémoire, ce qui n'est pas la sémantique d'Ollama.
        """
        now = self._clock()
        expired = [
            resident.name
            for resident in list(self._residents.values())
            if resident.is_serving and resident.is_expired(now)
        ]
        for name in expired:
            emit(EVENT_KEEP_ALIVE_EXPIRED, model=name)
            await self._unload_key(name)
        return expired

    async def shutdown(self) -> None:
        """Arrête toutes les instances. Aucun processus `llama-server` ne doit survivre."""
        for task in list(self._loading.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        for key in list(self._residents):
            await self._unload_key(key)

    # --- Vue scheduler ----------------------------------------------------------------------------

    def _resident_infos(self) -> list[ResidentInfo]:
        now = self._clock()
        return [
            ResidentInfo(
                name=resident.name,
                estimated_bytes=resident.estimate.total_bytes,
                active_requests=resident.active_requests,
                idle_seconds=max(0.0, now - resident.last_used),
                priority=resident.priority,
                keep_alive_expired=resident.is_expired(now),
                # Un modèle en cours de chargement ou d'arrêt a déjà réservé sa mémoire : il ne
                # doit jamais être choisi comme candidat à l'éviction.
                pinned=resident.state in (ModelState.LOADING, ModelState.UNLOADING),
            )
            for resident in self._residents.values()
        ]

    def resident_infos(self) -> list[ResidentInfo]:
        """Vue publique de l'état des résidents, pour le diagnostic et les tests."""
        return self._resident_infos()


def build_runtime_config(base: RuntimeConfig, override: RuntimeConfig | None) -> RuntimeConfig:
    """Fusionne une configuration runtime de manifest avec une surcharge de requête.

    Point unique de la précédence `requête > manifest`, exposé ici pour que les façades n'aient
    pas à connaître les détails de fusion.
    """
    return base if override is None else base.merged_with(override)
