"""Supervision des processus `llama-server`.

@spec docs/BACKLOG.md OC-030 « LlamaServerSupervisor »
@spec docs/ollama.cpp-architecture.md §1.2 « Le mode routeur de llama-server »,
      §5.2 « Choix technologiques », §8 risque R10
@spec docs/DAT.md §2 « Services et processus », §5.2 « Interfaces consommées »

Une instance `llama-server` par modèle logique, sur un port de boucle locale. Le pattern est
repris de `tools/server/server-models.cpp`, qui fait déjà exactement cela en amont : ce n'est pas
une invention de `ollama.cpp`, mais l'adoption d'une architecture éprouvée par le projet
`llama.cpp` lui-même.

Le contrôle direct de la ligne de commande est **la raison d'être du projet** : c'est lui qui
permet `--cache-type-k iq4_nl` ou `--tensor-split` par modèle, là où Ollama impose ses propres
choix (mission §16).

**Risque R10 — divergence de l'upstream.** La dépendance à `llama.cpp` est volontairement réduite
à sa surface publique : le binaire `llama-server`, ses drapeaux CLI documentés, et les endpoints
HTTP `/health`, `/props`, `/v1/*`. Aucun en-tête, aucune structure interne, aucun endpoint privé.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..errors import UpstreamError
from ..observability import EVENT_LOAD_COMPLETE, EVENT_LOAD_STARTED, emit
from ..storage.manifests import RuntimeConfig
from .args import build_args, describe_args

#: Intervalle de sondage de `/health` pendant le chargement. Assez court pour ne pas ajouter de
#: latence perceptible, assez long pour ne pas marteler une instance qui charge un gros modèle.
_HEALTH_POLL_INTERVAL_S = 0.1

#: Délai laissé à une instance pour s'arrêter proprement avant `SIGKILL`.
_TERMINATE_GRACE_S = 10.0

#: Nombre de lignes de journal conservées par instance, pour le diagnostic d'un échec de
#: chargement. Borne mémoire : les lignes plus anciennes sont oubliées au fil de l'eau.
_LOG_TAIL_LINES = 40


@dataclass(slots=True)
class LlamaServerInstance:
    """Instance `llama-server` en cours d'exécution."""

    name: str
    port: int
    process: asyncio.subprocess.Process
    base_url: str
    client: httpx.AsyncClient
    started_at: float
    args: list[str] = field(default_factory=list)
    props: dict = field(default_factory=dict)
    #: Fin du journal de l'instance (stdout + stderr confondus), alimentée en continu par les
    #: tâches de drainage. Sert au diagnostic d'un échec de chargement.
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=_LOG_TAIL_LINES))
    #: Tâches qui vident les tubes du fils. Annulées à l'arrêt de l'instance.
    drains: list[asyncio.Task] = field(default_factory=list)

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def is_running(self) -> bool:
        return self.process.returncode is None

    @property
    def exit_code(self) -> int | None:
        return self.process.returncode


def allocate_port(host: str, port_min: int, port_max: int) -> int:
    """Réserve un port libre dans la plage configurée.

    La réservation est faite en ouvrant réellement une socket puis en la refermant : c'est la
    seule façon fiable de savoir qu'un port est libre. Une fenêtre de course subsiste entre la
    fermeture et le `bind` de `llama-server` ; elle est acceptée parce que la plage est privée au
    service et qu'un échec de `bind` est détecté au démarrage, pas silencieux.
    """
    for port in range(port_min, port_max + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise UpstreamError(f"no free port available in range {port_min}-{port_max}")


class LlamaServerSupervisor:
    """Lance, surveille et arrête les instances `llama-server`."""

    def __init__(
        self,
        *,
        binary: str,
        host: str,
        port_min: int,
        port_max: int,
        load_timeout_s: float,
        request_timeout_s: float,
        default_context: int = 4096,
        env: dict[str, str] | None = None,
    ) -> None:
        self._binary = binary
        self._host = host
        self._port_min = port_min
        self._port_max = port_max
        self._load_timeout_s = load_timeout_s
        self._request_timeout_s = request_timeout_s
        self._default_context = default_context
        self._env = env

    async def spawn(
        self,
        *,
        name: str,
        model_path: Path,
        runtime: RuntimeConfig,
        mmproj_path: Path | None = None,
        draft_path: Path | None = None,
        adapter_paths: tuple[Path, ...] = (),
    ) -> LlamaServerInstance:
        """Démarre une instance et attend qu'elle soit réellement prête.

        « Prête » signifie que `/health` répond 200 **et** que `/props` a pu être lu : une
        instance qui accepte les connexions mais dont le modèle n'est pas chargé ne doit jamais
        être présentée comme `READY`.
        """
        port = allocate_port(self._host, self._port_min, self._port_max)
        args = build_args(
            model_path=model_path,
            alias=name,
            host=self._host,
            port=port,
            runtime=runtime,
            mmproj_path=mmproj_path,
            draft_path=draft_path,
            adapter_paths=adapter_paths,
            default_context=self._default_context,
        )

        emit(EVENT_LOAD_STARTED, model=name, port=port, args=describe_args(args))
        started_at = time.monotonic()

        process = await asyncio.create_subprocess_exec(
            self._binary,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(self._env or {})},
            # Groupe de processus dédié : `llama-server` peut engendrer des enfants, et on veut
            # pouvoir tout arrêter d'un coup sans laisser d'orphelins.
            start_new_session=True,
        )

        base_url = f"http://{self._host}:{port}"
        client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(self._request_timeout_s, connect=10.0),
        )

        instance = LlamaServerInstance(
            name=name,
            port=port,
            process=process,
            base_url=base_url,
            client=client,
            started_at=started_at,
            args=args,
        )
        # Les tubes du fils DOIVENT être vidés en continu. Un `PIPE` que personne ne lit se
        # remplit (64 Kio sous Linux) et `llama-server` se bloque alors sur son prochain `write`
        # — génération comprise. Le symptôme est trompeur : le modèle répond, mais quinze fois
        # trop lentement, et seulement pour les configurations bavardes (décodage spéculatif, qui
        # journalise à chaque brouillon). Mesuré : 2,2 tok/s tube plein contre 33,9 tok/s drainé.
        instance.drains = [
            asyncio.create_task(self._drain(process.stdout, instance)),
            asyncio.create_task(self._drain(process.stderr, instance)),
        ]

        try:
            await self._await_health(instance)
            instance.props = await self.fetch_props(instance)
        except BaseException:
            # Toute erreur — y compris une annulation — doit laisser le système propre : sans
            # cela un chargement interrompu laisserait un processus orphelin tenant un port et,
            # sur GPU, plusieurs gigaoctets de VRAM.
            await self.terminate(instance)
            raise

        emit(
            EVENT_LOAD_COMPLETE,
            model=name,
            port=port,
            pid=instance.pid,
            load_seconds=time.monotonic() - started_at,
        )
        return instance

    async def _await_health(self, instance: LlamaServerInstance) -> None:
        """Attend que `/health` réponde 200, ou échoue avec un diagnostic exploitable."""
        deadline = time.monotonic() + self._load_timeout_s

        while time.monotonic() < deadline:
            if not instance.is_running:
                detail = await self._read_failure_output(instance)
                raise UpstreamError(
                    f"llama-server exited with code {instance.exit_code} "
                    f"while loading '{instance.name}'{detail}"
                )
            try:
                response = await instance.client.get("/health", timeout=2.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                # Attendu tant que le port n'écoute pas encore : ce n'est pas une erreur.
                pass
            await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)

        raise UpstreamError(
            f"llama-server did not become ready within {self._load_timeout_s:g}s "
            f"for model '{instance.name}'"
        )

    @staticmethod
    async def _drain(stream, instance: LlamaServerInstance) -> None:
        """Vide un tube du fils en continu, en gardant la fin du journal.

        Deux rôles, indissociables : empêcher le blocage sur tube plein, et conserver de quoi
        diagnostiquer un échec. Lire à la demande ne suffirait pas — le fils serait déjà bloqué
        avant qu'on ait une raison de lire."""
        if stream is None:
            return
        with contextlib.suppress(asyncio.CancelledError, ValueError, OSError):
            while True:
                line = await stream.readline()
                if not line:                      # EOF : le fils a fermé le tube
                    return
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    instance.log_tail.append(text)

    async def _read_failure_output(self, instance: LlamaServerInstance) -> str:
        """Fin du journal de l'instance, pour rendre un échec de chargement diagnosticable.

        Sans cela, un modèle qui ne charge pas produirait un simple code de sortie, impossible à
        interpréter — exactement le genre d'erreur muette que la règle §18 interdit.

        Le journal vient du tampon alimenté par `_drain` : au moment où l'on veut diagnostiquer,
        le fils est mort et ses tubes sont fermés — il n'y aurait plus rien à y lire.
        """
        # Laisse aux tâches de drainage le temps d'absorber les dernières lignes avant l'EOF.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.gather(*instance.drains, return_exceptions=True),
                                   timeout=2.0)
        if not instance.log_tail:
            return ""
        return ": " + " / ".join(list(instance.log_tail)[-3:])

    async def fetch_props(self, instance: LlamaServerInstance) -> dict:
        """Lit `GET /props`, source de vérité des capacités et du contexte effectif (OC-032)."""
        try:
            response = await instance.client.get("/props", timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UpstreamError(f"unable to read properties of model '{instance.name}'") from exc
        return payload if isinstance(payload, dict) else {}

    async def terminate(self, instance: LlamaServerInstance) -> None:
        """Arrête une instance : `SIGTERM`, puis `SIGKILL` si elle s'obstine. Idempotent."""
        with contextlib.suppress(Exception):
            await instance.client.aclose()

        # Les tâches de drainage se terminent d'elles-mêmes sur EOF ; on les annule ici pour le
        # cas où le fils survivrait à son tube (enfant orphelin tenant l'extrémité d'écriture).
        for task in instance.drains:
            task.cancel()

        if instance.process.returncode is not None:
            return

        with contextlib.suppress(ProcessLookupError, OSError):
            # Le groupe entier, pour ne pas laisser d'enfants derrière.
            os.killpg(os.getpgid(instance.pid), signal.SIGTERM)

        try:
            await asyncio.wait_for(instance.process.wait(), timeout=_TERMINATE_GRACE_S)
            return
        except asyncio.TimeoutError:
            pass

        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(os.getpgid(instance.pid), signal.SIGKILL)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(instance.process.wait(), timeout=_TERMINATE_GRACE_S)
