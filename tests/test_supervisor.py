"""Tests d'intégration du superviseur `llama-server`.

@verifies docs/BACKLOG.md OC-030 « LlamaServerSupervisor », OC-031 « arguments runtime »
@verifies docs/ollama.cpp-architecture.md §1.2 « Le mode routeur de llama-server »,
          §1.3 « /props », §8 risque R10
@verifies docs/DAT.md §2 « Services et processus »

Tests d'**intégration**, pas unitaires : un vrai processus est lancé, un vrai port alloué, un
vrai dialogue HTTP a lieu, et l'arrêt passe par un vrai signal. Seule l'inférence est simulée.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest

from ollamacpp.errors import UpstreamError
from ollamacpp.runtime.supervisor import allocate_port
from ollamacpp.storage import RuntimeConfig


@pytest.fixture
def modele_factice(tmp_path) -> Path:
    """Le faux serveur ne lit pas le fichier ; il doit simplement exister comme un vrai chemin."""
    chemin = tmp_path / "modele.gguf"
    chemin.write_bytes(b"GGUF")
    return chemin


class TestAllocationDePort:
    def test_port_dans_la_plage(self):
        port = allocate_port("127.0.0.1", 19500, 19599)
        assert 19500 <= port <= 19599

    def test_port_occupe_evite(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as pris:
            pris.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            pris.bind(("127.0.0.1", 19600))
            pris.listen(1)
            assert allocate_port("127.0.0.1", 19600, 19601) == 19601

    def test_plage_saturee(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as pris:
            pris.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            pris.bind(("127.0.0.1", 19700))
            pris.listen(1)
            with pytest.raises(UpstreamError, match="no free port"):
                allocate_port("127.0.0.1", 19700, 19700)


class TestDemarrage:
    async def test_instance_prete_et_interrogeable(self, supervisor, modele_factice):
        instance = await supervisor.spawn(
            name="qwen3:8b", model_path=modele_factice, runtime=RuntimeConfig()
        )
        try:
            assert instance.is_running
            assert instance.pid > 0
            reponse = await instance.client.get("/health")
            assert reponse.status_code == 200
        finally:
            await supervisor.terminate(instance)

    async def test_props_lues_au_demarrage(self, supervisor, modele_factice):
        """`/props` doit être lu avant de déclarer l'instance prête (source des capacités)."""
        instance = await supervisor.spawn(
            name="qwen3:8b", model_path=modele_factice,
            runtime=RuntimeConfig(context=8192, parallel=2),
        )
        try:
            assert instance.props["default_generation_settings"]["n_ctx"] == 8192
            assert instance.props["model_alias"] == "qwen3:8b"
            assert "chat_template_caps" in instance.props
        finally:
            await supervisor.terminate(instance)

    async def test_arguments_reellement_transmis(self, supervisor, modele_factice):
        """Vérifie de bout en bout que le manifest devient bien une ligne de commande (OC-031)."""
        instance = await supervisor.spawn(
            name="qwen3:8b",
            model_path=modele_factice,
            runtime=RuntimeConfig(cache_type_k="iq4_nl", cache_type_v="q8_0",
                                  flash_attention=True, context=16384),
        )
        try:
            argv = (await instance.client.get("/debug/argv")).json()["argv"]
            assert "--cache-type-k" in argv and argv[argv.index("--cache-type-k") + 1] == "iq4_nl"
            assert "--cache-type-v" in argv and argv[argv.index("--cache-type-v") + 1] == "q8_0"
            assert argv[argv.index("--flash-attn") + 1] == "on"
            assert argv[argv.index("--ctx-size") + 1] == "16384"
        finally:
            await supervisor.terminate(instance)

    async def test_instances_multiples_sur_des_ports_distincts(self, supervisor, modele_factice):
        premiere = await supervisor.spawn(name="a", model_path=modele_factice,
                                          runtime=RuntimeConfig())
        seconde = await supervisor.spawn(name="b", model_path=modele_factice,
                                         runtime=RuntimeConfig())
        try:
            assert premiere.port != seconde.port
        finally:
            await supervisor.terminate(premiere)
            await supervisor.terminate(seconde)


class TestEchecs:
    async def test_sortie_immediate_diagnostiquee(self, supervisor, modele_factice, monkeypatch):
        """Un échec de chargement doit produire un message exploitable, pas un code muet."""
        monkeypatch.setenv("FAKE_LLAMA_EXIT_CODE", "3")
        with pytest.raises(UpstreamError) as info:
            await supervisor.spawn(name="casse", model_path=modele_factice,
                                   runtime=RuntimeConfig())
        message = info.value.message
        assert "exited with code 3" in message
        assert "simulated load failure" in message, "la fin de stderr doit être remontée"

    async def test_depassement_du_delai(self, config, modele_factice, monkeypatch):
        from ollamacpp.runtime import LlamaServerSupervisor

        monkeypatch.setenv("FAKE_LLAMA_START_DELAY", "30")
        rapide = LlamaServerSupervisor(
            binary=config.llama_server_bin, host=config.llama_server_host,
            port_min=19800, port_max=19850, load_timeout_s=1.0, request_timeout_s=5.0,
        )
        with pytest.raises(UpstreamError, match="did not become ready"):
            await rapide.spawn(name="lent", model_path=modele_factice, runtime=RuntimeConfig())

    async def test_aucun_processus_orphelin_apres_echec(self, config, modele_factice, monkeypatch):
        """Un chargement échoué ne doit laisser ni processus ni port pris (fuite de VRAM)."""
        from ollamacpp.runtime import LlamaServerSupervisor

        monkeypatch.setenv("FAKE_LLAMA_START_DELAY", "30")
        rapide = LlamaServerSupervisor(
            binary=config.llama_server_bin, host=config.llama_server_host,
            port_min=19860, port_max=19860, load_timeout_s=1.0, request_timeout_s=5.0,
        )
        with pytest.raises(UpstreamError):
            await rapide.spawn(name="lent", model_path=modele_factice, runtime=RuntimeConfig())

        # Le port doit être immédiatement réutilisable : la preuve que rien ne l'occupe plus.
        await asyncio.sleep(0.3)
        assert allocate_port("127.0.0.1", 19860, 19860) == 19860


class TestArret:
    async def test_terminaison_propre(self, supervisor, modele_factice):
        instance = await supervisor.spawn(name="qwen3:8b", model_path=modele_factice,
                                          runtime=RuntimeConfig())
        await supervisor.terminate(instance)
        assert not instance.is_running

    async def test_terminaison_idempotente(self, supervisor, modele_factice):
        instance = await supervisor.spawn(name="qwen3:8b", model_path=modele_factice,
                                          runtime=RuntimeConfig())
        await supervisor.terminate(instance)
        await supervisor.terminate(instance)
        assert not instance.is_running

    async def test_port_libere_apres_arret(self, supervisor, modele_factice):
        instance = await supervisor.spawn(name="qwen3:8b", model_path=modele_factice,
                                          runtime=RuntimeConfig())
        port = instance.port
        await supervisor.terminate(instance)
        await asyncio.sleep(0.3)
        assert allocate_port("127.0.0.1", port, port) == port

class TestDrainageDesTubes:
    """Les tubes du fils sont vidés en continu, sans quoi `llama-server` se bloque.

    Un `PIPE` que personne ne lit se remplit (64 Kio sous Linux) et le fils se bloque sur son
    prochain `write` — génération comprise. Le symptôme est trompeur : le modèle répond, mais
    quinze fois trop lentement, et seulement pour les configurations bavardes (décodage
    spéculatif, qui journalise à chaque brouillon). Mesuré sur un cas réel : 2,2 tok/s tube plein
    contre 33,9 tok/s drainé.
    """

    async def test_une_instance_tres_bavarde_ne_se_bloque_pas(self, supervisor, modele_factice,
                                                              monkeypatch):
        """Bien plus d'un tampon de tube écrit sur la sortie : l'instance reste interrogeable."""
        monkeypatch.setenv("FAKE_LLAMA_LOG_BYTES", "400000")
        instance = await supervisor.spawn(
            name="bavard", model_path=modele_factice, runtime=RuntimeConfig(),
        )
        try:
            # Si les tubes n'étaient pas drainés, le fils serait bloqué et ne répondrait plus.
            reponse = await instance.client.get("/health", timeout=5.0)
            assert reponse.status_code == 200
            assert instance.is_running
        finally:
            await supervisor.terminate(instance)

    async def test_le_journal_conserve_est_borne(self, supervisor, modele_factice, monkeypatch):
        """Le tampon de diagnostic garde la FIN du journal, sans croître indéfiniment."""
        from ollamacpp.runtime.supervisor import _LOG_TAIL_LINES
        monkeypatch.setenv("FAKE_LLAMA_LOG_BYTES", "400000")
        instance = await supervisor.spawn(
            name="bavard", model_path=modele_factice, runtime=RuntimeConfig(),
        )
        try:
            await asyncio.sleep(0.3)
            assert len(instance.log_tail) <= _LOG_TAIL_LINES
        finally:
            await supervisor.terminate(instance)

    async def test_les_taches_de_drainage_sont_annulees_a_l_arret(self, supervisor,
                                                                 modele_factice):
        """Aucune tâche ne survit à l'instance : sinon elles s'accumuleraient à chaque bascule."""
        instance = await supervisor.spawn(
            name="qwen3:8b", model_path=modele_factice, runtime=RuntimeConfig()
        )
        assert instance.drains and all(not t.done() for t in instance.drains)
        await supervisor.terminate(instance)
        await asyncio.sleep(0.1)
        assert all(t.done() for t in instance.drains)

