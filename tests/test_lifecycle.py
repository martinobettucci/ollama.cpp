"""Tests d'intégration du cycle de vie des modèles.

@verifies docs/BACKLOG.md OC-032 « Détection de capacités », OC-033 « ModelLifecycleManager »,
          OC-034 « ModelScheduler »
@verifies docs/ollama.cpp-architecture.md §5.7 « États du cycle de vie », §5.8 « Scheduler »,
          risques R5 et R6
@verifies docs/DAT.md §3.2 « Chargement d'un modèle »

De vrais processus `llama-server` (factices) sont lancés et arrêtés : le single-flight, le
`keep_alive` et l'éviction sont vérifiés sur le comportement réel, pas sur des doublures.
"""

from __future__ import annotations

import asyncio

import pytest

from ollamacpp.durations import KeepAlive, parse_keep_alive
from ollamacpp.errors import ModelNotFound, UpstreamError
from ollamacpp.runtime import ModelLifecycleManager, ModelScheduler, ModelState
from ollamacpp.storage import Artifacts, LifecycleConfig, Manifest, RuntimeConfig
from ollamacpp.names import parse

from .conftest import install_model
from .ggufbuild import DEFAULT_KV, build_gguf


class TestChargement:
    async def test_modele_charge_et_pret(self, lifecycle, registry):
        install_model(registry)
        resident = await lifecycle.ensure_ready("qwen3:8b")
        assert resident.state is ModelState.READY
        assert resident.instance is not None
        assert resident.instance.is_running

    async def test_etat_avant_chargement(self, lifecycle, registry):
        install_model(registry)
        assert lifecycle.state("qwen3:8b") is ModelState.UNLOADED

    async def test_modele_absent(self, lifecycle):
        assert lifecycle.state("fantome") is ModelState.NOT_PRESENT
        with pytest.raises(ModelNotFound):
            await lifecycle.ensure_ready("fantome")

    async def test_reutilisation_sans_second_processus(self, lifecycle, registry):
        install_model(registry)
        premier = await lifecycle.ensure_ready("qwen3:8b")
        second = await lifecycle.ensure_ready("qwen3:8b")
        assert premier is second
        assert premier.instance.pid == second.instance.pid

    async def test_contexte_effectif_issu_de_props(self, lifecycle, registry):
        install_model(registry, runtime=RuntimeConfig(context=16384))
        resident = await lifecycle.ensure_ready("qwen3:8b")
        assert resident.context_length == 16384

    async def test_artefact_obligatoire_manquant_interdit_ready(self, lifecycle, registry):
        """Mission §14 : sans son `mmproj`, un modèle multimodal ne devient jamais READY."""
        blob = registry.blobs.ingest_bytes(build_gguf(DEFAULT_KV))
        registry.install(
            Manifest(
                name=parse("borgne"),
                artifacts=Artifacts(model=blob.digest, mmproj="sha256:" + "a" * 64),
            )
        )
        with pytest.raises(UpstreamError, match="missing required artifacts"):
            await lifecycle.ensure_ready("borgne")
        assert lifecycle.residents() == []


class TestSingleFlight:
    """Mission §19 : deux requêtes concurrentes partagent un chargement unique."""

    async def test_chargements_concurrents_partages(self, lifecycle, registry):
        install_model(registry)
        residents = await asyncio.gather(*[lifecycle.ensure_ready("qwen3:8b") for _ in range(5)])
        assert all(r is residents[0] for r in residents)
        assert len({r.instance.pid for r in residents}) == 1

    async def test_un_seul_processus_lance(self, lifecycle, registry, monkeypatch):
        install_model(registry)
        appels = 0
        original = lifecycle._supervisor.spawn

        async def compter(**kwargs):
            nonlocal appels
            appels += 1
            return await original(**kwargs)

        monkeypatch.setattr(lifecycle._supervisor, "spawn", compter)
        await asyncio.gather(*[lifecycle.ensure_ready("qwen3:8b") for _ in range(4)])
        assert appels == 1

    async def test_annulation_dun_client_nannule_pas_le_chargement(self, lifecycle, registry):
        """Sans `shield`, un client qui abandonne casserait toutes les requêtes en attente."""
        install_model(registry)
        abandonne = asyncio.create_task(lifecycle.ensure_ready("qwen3:8b"))
        persistant = asyncio.create_task(lifecycle.ensure_ready("qwen3:8b"))
        await asyncio.sleep(0.05)
        abandonne.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandonne

        resident = await persistant
        assert resident.state is ModelState.READY
        assert resident.instance.is_running

    async def test_echec_de_chargement_nempoisonne_pas_les_suivants(
        self, lifecycle, registry, monkeypatch
    ):
        install_model(registry)
        monkeypatch.setenv("FAKE_LLAMA_EXIT_CODE", "1")
        with pytest.raises(UpstreamError):
            await lifecycle.ensure_ready("qwen3:8b")

        monkeypatch.delenv("FAKE_LLAMA_EXIT_CODE")
        resident = await lifecycle.ensure_ready("qwen3:8b")
        assert resident.state is ModelState.READY

    async def test_echec_ne_laisse_pas_de_reservation_memoire(
        self, lifecycle, registry, monkeypatch
    ):
        """Une réservation non libérée ferait fuir le budget mémoire à chaque échec."""
        install_model(registry)
        monkeypatch.setenv("FAKE_LLAMA_EXIT_CODE", "1")
        with pytest.raises(UpstreamError):
            await lifecycle.ensure_ready("qwen3:8b")
        assert lifecycle.resident_infos() == []


class TestUsageEtEtats:
    async def test_busy_pendant_la_requete_puis_idle(self, lifecycle, registry):
        install_model(registry)
        async with lifecycle.acquire("qwen3:8b") as resident:
            assert resident.state is ModelState.BUSY
            assert resident.active_requests == 1
        assert resident.state is ModelState.IDLE
        assert resident.active_requests == 0

    async def test_requetes_concurrentes_comptees(self, lifecycle, registry):
        install_model(registry)

        async def requete(barriere):
            async with lifecycle.acquire("qwen3:8b") as resident:
                await barriere.wait()
                return resident.active_requests

        barriere = asyncio.Barrier(3)
        resultats = await asyncio.gather(*[requete(barriere) for _ in range(3)])
        assert max(resultats) == 3

    async def test_compteur_libere_meme_en_cas_derreur(self, lifecycle, registry):
        """Sans `finally`, une requête interrompue laisserait le modèle éternellement BUSY."""
        install_model(registry)
        with pytest.raises(RuntimeError):
            async with lifecycle.acquire("qwen3:8b") as resident:
                raise RuntimeError("échec applicatif")
        assert resident.active_requests == 0
        assert resident.state is ModelState.IDLE


class TestKeepAlive:
    async def test_keep_alive_zero_decharge_a_la_fin_de_la_requete(self, lifecycle, registry):
        """Sémantique Ollama : `keep_alive: 0` décharge dès la fin de la requête (risque R5)."""
        install_model(registry)
        async with lifecycle.acquire("qwen3:8b", keep_alive=parse_keep_alive(0)):
            pass
        assert lifecycle.residents() == []

    async def test_keep_alive_du_manifest_applique(self, lifecycle, registry):
        install_model(registry, lifecycle_config=LifecycleConfig(keep_alive="15m"))
        resident = await lifecycle.ensure_ready("qwen3:8b")
        assert resident.keep_alive.seconds == 900.0

    async def test_keep_alive_de_la_requete_prime_sur_le_manifest(self, lifecycle, registry):
        install_model(registry, lifecycle_config=LifecycleConfig(keep_alive="15m"))
        resident = await lifecycle.ensure_ready("qwen3:8b", keep_alive=parse_keep_alive("1h"))
        assert resident.keep_alive.seconds == 3600.0

    async def test_defaut_global_si_rien_nest_precise(self, lifecycle, registry):
        install_model(registry)
        resident = await lifecycle.ensure_ready("qwen3:8b")
        assert resident.keep_alive.seconds == 300.0

    async def test_requete_ulterieure_change_la_politique(self, lifecycle, registry):
        """`keep_alive` accompagne chaque appel : il peut modifier un modèle déjà chargé."""
        install_model(registry)
        await lifecycle.ensure_ready("qwen3:8b")
        resident = await lifecycle.ensure_ready("qwen3:8b", keep_alive=parse_keep_alive("1h"))
        assert resident.keep_alive.seconds == 3600.0

    async def test_balayage_decharge_les_modeles_expires(self, config, registry, scheduler):
        """Le balayage rend `keep_alive` effectif sans attendre une autre requête."""
        from ollamacpp.runtime import LlamaServerSupervisor

        horloge = {"t": 1000.0}
        supervisor = LlamaServerSupervisor(
            binary=config.llama_server_bin, host=config.llama_server_host,
            port_min=config.llama_server_port_min, port_max=config.llama_server_port_max,
            load_timeout_s=config.load_timeout_s, request_timeout_s=config.request_timeout_s,
        )
        manager = ModelLifecycleManager(
            registry=registry, supervisor=supervisor, scheduler=scheduler, config=config,
            clock=lambda: horloge["t"],
        )
        try:
            install_model(registry)
            await manager.ensure_ready("qwen3:8b", keep_alive=parse_keep_alive("60s"))
            assert await manager.sweep() == []

            horloge["t"] += 61
            assert await manager.sweep() == ["qwen3:8b"]
            assert manager.residents() == []
        finally:
            await manager.shutdown()

    async def test_keep_alive_illimite_jamais_balaye(self, config, registry, scheduler):
        from ollamacpp.runtime import LlamaServerSupervisor

        horloge = {"t": 1000.0}
        supervisor = LlamaServerSupervisor(
            binary=config.llama_server_bin, host=config.llama_server_host,
            port_min=config.llama_server_port_min, port_max=config.llama_server_port_max,
            load_timeout_s=config.load_timeout_s, request_timeout_s=config.request_timeout_s,
        )
        manager = ModelLifecycleManager(
            registry=registry, supervisor=supervisor, scheduler=scheduler, config=config,
            clock=lambda: horloge["t"],
        )
        try:
            install_model(registry)
            await manager.ensure_ready("qwen3:8b", keep_alive=KeepAlive(float("inf")))
            horloge["t"] += 100_000
            assert await manager.sweep() == []
        finally:
            await manager.shutdown()


class TestEviction:
    async def test_limite_de_modeles_charges_respectee(self, lifecycle, registry):
        """`max_loaded_models` vaut 2 dans la configuration de test."""
        for nom in ("a", "b", "c"):
            install_model(registry, nom)
            await lifecycle.ensure_ready(nom)
        residents = [r.name for r in lifecycle.residents()]
        assert len(residents) == 2
        assert "c:latest" in residents
        assert "a:latest" not in residents, "le plus ancien doit avoir été évincé"

    async def test_modele_busy_survit_a_la_pression(self, lifecycle, registry):
        for nom in ("a", "b", "c"):
            install_model(registry, nom)

        async with lifecycle.acquire("a"):
            await lifecycle.ensure_ready("b")
            await lifecycle.ensure_ready("c")
            noms = [r.name for r in lifecycle.residents()]
            assert "a:latest" in noms, "un modèle BUSY ne doit jamais être évincé"

    async def test_processus_evince_reellement_arrete(self, lifecycle, registry):
        for nom in ("a", "b", "c"):
            install_model(registry, nom)
        premier = await lifecycle.ensure_ready("a")
        instance = premier.instance
        await lifecycle.ensure_ready("b")
        await lifecycle.ensure_ready("c")
        await asyncio.sleep(0.2)
        assert not instance.is_running, "l'éviction doit arrêter le processus, pas l'oublier"


class TestDechargement:
    async def test_dechargement_explicite(self, lifecycle, registry):
        install_model(registry)
        await lifecycle.ensure_ready("qwen3:8b")
        assert await lifecycle.unload("qwen3:8b") is True
        assert lifecycle.residents() == []

    async def test_dechargement_idempotent(self, lifecycle, registry):
        install_model(registry)
        assert await lifecycle.unload("qwen3:8b") is False

    async def test_arret_general_ne_laisse_aucun_processus(self, lifecycle, registry):
        for nom in ("a", "b"):
            install_model(registry, nom)
            await lifecycle.ensure_ready(nom)
        instances = [r.instance for r in lifecycle.residents()]
        await lifecycle.shutdown()
        await asyncio.sleep(0.2)
        assert all(not instance.is_running for instance in instances)


class TestCapacites:
    """OC-032 : les capacités viennent de faits observables, jamais d'une déclaration."""

    async def test_outils_detectes_depuis_le_template(self, lifecycle, registry):
        install_model(registry)
        resident = await lifecycle.ensure_ready("qwen3:8b")
        assert "tools" in resident.capabilities
        assert resident.capabilities.evidence["tools"].startswith("props.chat_template_caps")

    async def test_absence_doutils_respectee(self, lifecycle, registry, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_TOOLS", "0")
        install_model(registry)
        resident = await lifecycle.ensure_ready("qwen3:8b")
        assert "tools" not in resident.capabilities

    async def test_vision_exige_mmproj_et_modalite(self, lifecycle, registry):
        install_model(registry, "vision-modele", with_mmproj=True)
        resident = await lifecycle.ensure_ready("vision-modele")
        assert "vision" in resident.capabilities

    async def test_pas_de_vision_sans_mmproj(self, lifecycle, registry):
        """Le cœur de l'exigence : jamais `vision=true` sur un modèle qui ne voit rien."""
        install_model(registry, "texte-seul")
        resident = await lifecycle.ensure_ready("texte-seul")
        assert "vision" not in resident.capabilities

    async def test_raisonnement_detecte(self, lifecycle, registry, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_THINKING", "1")
        install_model(registry)
        resident = await lifecycle.ensure_ready("qwen3:8b")
        assert "thinking" in resident.capabilities

    async def test_manifest_peut_retirer_une_capacite(self, lifecycle, registry):
        blob = registry.blobs.ingest_bytes(build_gguf(DEFAULT_KV))
        registry.install(
            Manifest(
                name=parse("bride"),
                artifacts=Artifacts(model=blob.digest),
                capabilities_override={"tools": False},
            )
        )
        resident = await lifecycle.ensure_ready("bride")
        assert "tools" not in resident.capabilities
        assert resident.capabilities.evidence["tools"] == "désactivée par le manifest"
