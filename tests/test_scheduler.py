"""Tests du scheduler et de l'estimation mémoire.

@verifies docs/BACKLOG.md OC-034 « ModelScheduler », OC-035 « Observabilité des décisions »
@verifies docs/ollama.cpp-architecture.md §5.8 « Scheduler », risques R6 et R11
@verifies docs/DAT.md §3.2 « Chargement d'un modèle »

Les décisions sont testées sans processus ni mémoire réels : le scheduler est une fonction pure
de son entrée, précisément pour être vérifiable sans GPU (risque R11).
"""

from __future__ import annotations

import pytest

from ollamacpp.gguf import GGUFMetadata
from ollamacpp.runtime.memory import (
    FixedMemoryProbe,
    HostMemoryProbe,
    cache_type_bytes,
    estimate_kv_bytes,
    estimate_model_memory,
    resolve_memory_budget,
)
from ollamacpp.runtime.scheduler import (
    REASON_ALREADY_RESIDENT,
    REASON_ENOUGH_MEMORY,
    REASON_INSUFFICIENT_MEMORY,
    REASON_KEEP_ALIVE_EXPIRED,
    REASON_MEMORY_PRESSURE,
    REASON_SLOT_PRESSURE,
    ModelScheduler,
    ResidentInfo,
)

GO = 1024 ** 3


def resident(nom: str, *, octets=GO, actives=0, inactivite=0.0, priorite=0,
             expire=False, epingle=False) -> ResidentInfo:
    return ResidentInfo(
        name=nom, estimated_bytes=octets, active_requests=actives,
        idle_seconds=inactivite, priority=priorite, keep_alive_expired=expire, pinned=epingle,
    )


class TestAdmissionSimple:
    def test_modele_deja_resident(self):
        plan = ModelScheduler(memory_budget_bytes=10 * GO, max_loaded_models=3).plan(
            name="a", required_bytes=GO, residents=[resident("a")]
        )
        assert plan.admitted
        assert plan.reason == REASON_ALREADY_RESIDENT
        assert not plan.needs_eviction

    def test_memoire_suffisante(self):
        plan = ModelScheduler(memory_budget_bytes=10 * GO, max_loaded_models=3).plan(
            name="b", required_bytes=2 * GO, residents=[resident("a")]
        )
        assert plan.admitted
        assert plan.reason == REASON_ENOUGH_MEMORY
        assert not plan.needs_eviction

    def test_budget_indisponible_nempeche_pas_de_charger(self):
        """Une sonde mémoire défaillante ne doit pas bloquer le service (budget = 0)."""
        plan = ModelScheduler(memory_budget_bytes=0, max_loaded_models=3).plan(
            name="b", required_bytes=999 * GO, residents=[]
        )
        assert plan.admitted


class TestPressionMemoire:
    def test_eviction_sous_pression(self):
        plan = ModelScheduler(memory_budget_bytes=4 * GO, max_loaded_models=10).plan(
            name="neuf", required_bytes=3 * GO,
            residents=[resident("vieux", octets=3 * GO, inactivite=500)],
        )
        assert plan.admitted
        assert [e.model for e in plan.evictions] == ["vieux"]
        assert plan.evictions[0].reason == REASON_MEMORY_PRESSURE

    def test_pression_de_slots(self):
        """Limite de nombre de modèles atteinte, mais mémoire disponible."""
        plan = ModelScheduler(memory_budget_bytes=100 * GO, max_loaded_models=2).plan(
            name="c", required_bytes=GO,
            residents=[resident("a", inactivite=10), resident("b", inactivite=500)],
        )
        assert plan.admitted
        assert [e.model for e in plan.evictions] == ["b"]
        assert plan.evictions[0].reason == REASON_SLOT_PRESSURE

    def test_evictions_multiples_si_necessaire(self):
        plan = ModelScheduler(memory_budget_bytes=6 * GO, max_loaded_models=10).plan(
            name="gros", required_bytes=6 * GO,
            residents=[resident("a", octets=3 * GO, inactivite=100),
                       resident("b", octets=3 * GO, inactivite=200)],
        )
        assert plan.admitted
        assert len(plan.evictions) == 2
        assert plan.freed_bytes == 6 * GO

    def test_modele_trop_gros_pour_le_budget(self):
        plan = ModelScheduler(memory_budget_bytes=4 * GO, max_loaded_models=10).plan(
            name="enorme", required_bytes=100 * GO, residents=[]
        )
        assert not plan.admitted
        assert plan.reason == REASON_INSUFFICIENT_MEMORY
        assert plan.evictions == (), "aucune éviction inutile si le modèle ne tient pas de toute façon"


class TestRegleAbsolueBusy:
    """Un modèle BUSY n'est jamais évincé : cela tuerait une requête en cours."""

    def test_modele_busy_jamais_evince(self):
        plan = ModelScheduler(memory_budget_bytes=4 * GO, max_loaded_models=10).plan(
            name="neuf", required_bytes=3 * GO,
            residents=[resident("occupe", octets=3 * GO, actives=1, inactivite=9999)],
        )
        assert not plan.admitted
        assert plan.evictions == ()

    def test_modele_epingle_jamais_evince(self):
        """Un modèle en cours de chargement a déjà réservé sa mémoire."""
        plan = ModelScheduler(memory_budget_bytes=4 * GO, max_loaded_models=10).plan(
            name="neuf", required_bytes=3 * GO,
            residents=[resident("en-chargement", octets=3 * GO, epingle=True, inactivite=9999)],
        )
        assert not plan.admitted

    def test_busy_epargne_mais_idle_evince(self):
        plan = ModelScheduler(memory_budget_bytes=6 * GO, max_loaded_models=10).plan(
            name="neuf", required_bytes=3 * GO,
            residents=[resident("occupe", octets=3 * GO, actives=2, inactivite=9999),
                       resident("libre", octets=3 * GO, inactivite=5)],
        )
        assert [e.model for e in plan.evictions] == ["libre"]

    def test_modele_demande_jamais_evince_pour_lui_meme(self):
        scheduler = ModelScheduler(memory_budget_bytes=4 * GO, max_loaded_models=1)
        plan = scheduler.plan(name="a", required_bytes=GO, residents=[resident("a")])
        assert plan.admitted
        assert plan.evictions == ()


class TestOrdreDesCandidats:
    def test_keep_alive_expire_prioritaire(self):
        """Un modèle expiré part avant un modèle plus ancien mais encore valide."""
        plan = ModelScheduler(memory_budget_bytes=100 * GO, max_loaded_models=2).plan(
            name="c", required_bytes=GO,
            residents=[resident("tres-vieux", inactivite=9999),
                       resident("expire", inactivite=10, expire=True)],
        )
        assert [e.model for e in plan.evictions] == ["expire"]
        assert plan.evictions[0].reason == REASON_KEEP_ALIVE_EXPIRED

    def test_priorite_basse_evincee_en_premier(self):
        plan = ModelScheduler(memory_budget_bytes=100 * GO, max_loaded_models=2).plan(
            name="c", required_bytes=GO,
            residents=[resident("important", priorite=100, inactivite=9999),
                       resident("secondaire", priorite=1, inactivite=10)],
        )
        assert [e.model for e in plan.evictions] == ["secondaire"]

    def test_lru_a_priorite_egale(self):
        plan = ModelScheduler(memory_budget_bytes=100 * GO, max_loaded_models=2).plan(
            name="c", required_bytes=GO,
            residents=[resident("recent", inactivite=5), resident("ancien", inactivite=900)],
        )
        assert [e.model for e in plan.evictions] == ["ancien"]


class TestExpiration:
    def test_modeles_expires_listes(self):
        expires = ModelScheduler(memory_budget_bytes=GO, max_loaded_models=3).expired(
            [resident("a", expire=True, inactivite=700), resident("b")]
        )
        assert [e.model for e in expires] == ["a"]

    def test_modele_expire_mais_busy_epargne(self):
        """Une requête en cours l'emporte sur l'expiration : on ne coupe pas une génération."""
        expires = ModelScheduler(memory_budget_bytes=GO, max_loaded_models=3).expired(
            [resident("a", expire=True, actives=1)]
        )
        assert expires == []


class TestJournalisationExplicable:
    def test_format_de_la_decision(self):
        """Format exigé par la mission §30, greppable et stable."""
        plan = ModelScheduler(memory_budget_bytes=4 * GO, max_loaded_models=10).plan(
            name="neuf", required_bytes=3 * GO,
            residents=[resident("qwen3.6", octets=3 * GO, inactivite=731, priorite=50)],
        )
        ligne = ModelScheduler.log(plan.evictions[0])
        assert "model=qwen3.6" in ligne
        assert "action=evict" in ligne
        assert "reason=memory_pressure" in ligne
        assert "idle_seconds=731" in ligne
        assert "priority=50" in ligne


# --- Estimation mémoire ------------------------------------------------------------------------


def metadata(**kv) -> GGUFMetadata:
    base = {
        "general.architecture": "qwen3",
        "qwen3.block_count": 36,
        "qwen3.embedding_length": 4096,
    }
    base.update(kv)
    return GGUFMetadata(version=3, tensor_count=0, kv=base, file_size=0)


class TestEstimationKV:
    def test_proportionnelle_au_contexte(self):
        petit = estimate_kv_bytes(metadata(), context=4096)
        grand = estimate_kv_bytes(metadata(), context=8192)
        assert grand == 2 * petit

    def test_grouped_query_attention_prise_en_compte(self):
        """Sans cette correction l'estimation serait fausse d'un facteur 8 (risque R6)."""
        sans_gqa = estimate_kv_bytes(metadata(), context=4096)
        avec_gqa = estimate_kv_bytes(
            metadata(**{"qwen3.attention.head_count": 32, "qwen3.attention.head_count_kv": 4}),
            context=4096,
        )
        assert avec_gqa == sans_gqa // 8

    def test_type_de_cache_reduit_lempreinte(self):
        f16 = estimate_kv_bytes(metadata(), context=4096, cache_type_k="f16", cache_type_v="f16")
        quantise = estimate_kv_bytes(metadata(), context=4096,
                                     cache_type_k="iq4_nl", cache_type_v="q8_0")
        assert quantise < f16

    def test_parallelisme_multiplie(self):
        seul = estimate_kv_bytes(metadata(), context=4096, parallel=1)
        quatre = estimate_kv_bytes(metadata(), context=4096, parallel=4)
        assert quatre == 4 * seul

    def test_metadonnees_insuffisantes(self):
        """Mieux vaut une estimation ouvertement nulle qu'un chiffre inventé."""
        assert estimate_kv_bytes(None, context=4096) == 0
        assert estimate_kv_bytes(metadata(**{"qwen3.block_count": 0}), context=4096) == 0

    def test_type_inconnu_retombe_sur_le_defaut_prudent(self):
        assert cache_type_bytes("type_du_futur") == cache_type_bytes("f16")
        assert cache_type_bytes(None) == 2.0


class TestEstimationTotale:
    def test_somme_des_composantes(self):
        estimation = estimate_model_memory(
            artifacts_bytes=4 * GO, metadata=metadata(), context=4096
        )
        assert estimation.weights_bytes == 4 * GO
        assert estimation.kv_bytes > 0
        assert estimation.total_bytes == (
            estimation.weights_bytes + estimation.kv_bytes + estimation.overhead_bytes
        )

    def test_champs_journalisables(self):
        champs = estimate_model_memory(artifacts_bytes=GO, metadata=None, context=0).as_fields()
        assert set(champs) == {"weights_bytes", "kv_bytes", "overhead_bytes", "total_bytes"}


class TestBudget:
    def test_limite_explicite_prime(self):
        budget = resolve_memory_budget(
            FixedMemoryProbe(100 * GO), configured_limit=8 * GO, safety_margin=0.0
        )
        assert budget == 8 * GO

    def test_marge_de_securite_appliquee(self):
        budget = resolve_memory_budget(
            FixedMemoryProbe(10 * GO), configured_limit=0, safety_margin=0.10
        )
        assert budget == pytest.approx(9 * GO, rel=0.01)

    def test_sonde_indisponible(self):
        assert resolve_memory_budget(FixedMemoryProbe(0), configured_limit=0,
                                     safety_margin=0.1) == 0

    def test_sonde_hote_renvoie_une_valeur_plausible(self):
        """Vérification que la sonde réelle fonctionne sur cet hôte, sans figer de valeur."""
        total = HostMemoryProbe().total_bytes()
        assert total > 0
