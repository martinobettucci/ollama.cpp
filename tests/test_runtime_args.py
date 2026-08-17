"""Tests de la construction des arguments `llama-server`.

@verifies docs/BACKLOG.md OC-031 « Construction des arguments runtime »
@verifies docs/ollama.cpp-architecture.md §1.5 « Arguments CLI pertinents », §5.2

Chaque drapeau attendu ici a été vérifié dans `common/arg.cpp` à la révision `39be55c`. C'est le
point où le projet tient sa promesse centrale : exposer les capacités de `llama.cpp` qu'Ollama
masque.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ollamacpp.runtime.args import RESERVED_FLAGS, build_args, filter_extra_args
from ollamacpp.storage import RuntimeConfig

MODELE = Path("/blobs/sha256-abc")


def args_for(runtime: RuntimeConfig, **kwargs) -> list[str]:
    return build_args(
        model_path=MODELE, alias="qwen3:8b", host="127.0.0.1", port=18000,
        runtime=runtime, **kwargs
    )


def valeur_de(args: list[str], drapeau: str) -> str | None:
    return args[args.index(drapeau) + 1] if drapeau in args else None


class TestArgumentsDeBase:
    def test_identite_du_modele(self):
        args = args_for(RuntimeConfig())
        assert valeur_de(args, "--model") == str(MODELE)
        assert valeur_de(args, "--alias") == "qwen3:8b"

    def test_ecoute_locale(self):
        args = args_for(RuntimeConfig())
        assert valeur_de(args, "--host") == "127.0.0.1"
        assert valeur_de(args, "--port") == "18000"

    def test_interface_web_desactivee(self):
        """Une instance interne n'a pas d'interface web : mémoire et surface inutiles."""
        assert "--no-webui" in args_for(RuntimeConfig())

    def test_jinja_active(self):
        """Sans `--jinja`, ni les chat templates ni le parsing des appels d'outils natifs."""
        assert "--jinja" in args_for(RuntimeConfig())

    def test_props_non_active(self):
        """`GET /props` est servi sans condition ; `--props` n'ouvrirait que le POST mutant."""
        assert "--props" not in args_for(RuntimeConfig())

    def test_contexte_par_defaut_applique(self):
        args = args_for(RuntimeConfig(), default_context=8192)
        assert valeur_de(args, "--ctx-size") == "8192"


class TestCapacitesAvancees:
    """Le cœur du projet : ce qu'Ollama ne permet pas de configurer."""

    def test_types_de_cache_kv(self):
        """L'exemple exact de la mission §16 : K en IQ4_NL, V en Q8_0."""
        args = args_for(RuntimeConfig(cache_type_k="iq4_nl", cache_type_v="q8_0"))
        assert valeur_de(args, "--cache-type-k") == "iq4_nl"
        assert valeur_de(args, "--cache-type-v") == "q8_0"

    @pytest.mark.parametrize("actif, attendu", [(True, "on"), (False, "off")])
    def test_flash_attention(self, actif, attendu):
        """`--flash-attn` attend `on|off|auto` (`common/arg.cpp` l. 1744)."""
        assert valeur_de(args_for(RuntimeConfig(flash_attention=actif)), "--flash-attn") == attendu

    def test_flash_attention_non_precisee_laisse_le_defaut(self):
        assert "--flash-attn" not in args_for(RuntimeConfig())

    def test_repartition_gpu(self):
        args = args_for(RuntimeConfig(gpu_layers=99, tensor_split="0.6,0.4", main_gpu=1))
        assert valeur_de(args, "--n-gpu-layers") == "99"
        assert valeur_de(args, "--tensor-split") == "0.6,0.4"
        assert valeur_de(args, "--main-gpu") == "1"

    def test_contexte_batch_parallelisme(self):
        args = args_for(RuntimeConfig(context=262144, batch=2048, ubatch=512, parallel=4))
        assert valeur_de(args, "--ctx-size") == "262144"
        assert valeur_de(args, "--batch-size") == "2048"
        assert valeur_de(args, "--ubatch-size") == "512"
        assert valeur_de(args, "--parallel") == "4"

    def test_decoding_speculatif(self):
        args = args_for(RuntimeConfig(draft_max=16, draft_min=4, draft_p_min=0.75),
                        draft_path=Path("/blobs/sha256-draft"))
        assert valeur_de(args, "--model-draft") == "/blobs/sha256-draft"
        assert valeur_de(args, "--draft-max") == "16"
        assert valeur_de(args, "--draft-p-min") == "0.75"

    def test_multimodal(self):
        args = args_for(RuntimeConfig(), mmproj_path=Path("/blobs/sha256-mmproj"))
        assert valeur_de(args, "--mmproj") == "/blobs/sha256-mmproj"

    def test_adaptateurs_multiples(self):
        args = args_for(RuntimeConfig(),
                        adapter_paths=(Path("/blobs/sha256-a"), Path("/blobs/sha256-b")))
        assert args.count("--lora") == 2

    def test_modes_embedding_et_rerank(self):
        assert "--embedding" in args_for(RuntimeConfig(embedding=True))
        assert "--reranking" in args_for(RuntimeConfig(reranking=True))
        assert valeur_de(args_for(RuntimeConfig(pooling="mean")), "--pooling") == "mean"

    def test_drapeaux_negatifs(self):
        args = args_for(RuntimeConfig(mmap=False, kv_offload=False))
        assert "--no-mmap" in args
        assert "--no-kv-offload" in args

    def test_drapeaux_negatifs_absents_par_defaut(self):
        args = args_for(RuntimeConfig())
        assert "--no-mmap" not in args
        assert "--no-kv-offload" not in args

    def test_raisonnement(self):
        args = args_for(RuntimeConfig(reasoning_format="deepseek", reasoning_budget=2048))
        assert valeur_de(args, "--reasoning-format") == "deepseek"
        assert valeur_de(args, "--reasoning-budget") == "2048"


class TestDrapeauxReserves:
    """Un manifest ne doit jamais pouvoir détourner l'identité ou l'exposition d'une instance."""

    @pytest.mark.parametrize("drapeau", ["--host", "--port", "--model", "-m", "--alias", "--api-key"])
    def test_drapeau_reserve_filtre(self, drapeau):
        assert filter_extra_args((drapeau, "valeur")) == []

    def test_valeur_du_drapeau_reserve_aussi_retiree(self):
        """Laisser une valeur orpheline décalerait toute la ligne de commande."""
        assert filter_extra_args(("--host", "0.0.0.0", "--verbose")) == ["--verbose"]

    def test_forme_avec_egal_filtree(self):
        assert filter_extra_args(("--host=0.0.0.0", "--verbose")) == ["--verbose"]

    def test_extra_args_legitimes_conserves(self):
        assert filter_extra_args(("--cache-reuse", "256")) == ["--cache-reuse", "256"]

    def test_manifest_ne_peut_pas_exposer_linstance(self):
        """Cas d'attaque concret : un manifest tentant d'écouter sur toutes les interfaces."""
        args = args_for(RuntimeConfig(extra_args=("--host", "0.0.0.0")))
        assert args.count("--host") == 1
        assert valeur_de(args, "--host") == "127.0.0.1"

    def test_manifest_ne_peut_pas_changer_de_modele(self):
        args = args_for(RuntimeConfig(extra_args=("--model", "/etc/passwd")))
        assert args.count("--model") == 1
        assert valeur_de(args, "--model") == str(MODELE)

    def test_liste_des_reserves_couvre_les_formes_courtes(self):
        for court in ("-m", "-a"):
            assert court in RESERVED_FLAGS


class TestPrecedence:
    def test_contexte_du_manifest_prime_sur_le_defaut(self):
        args = args_for(RuntimeConfig(context=16384), default_context=4096)
        assert valeur_de(args, "--ctx-size") == "16384"
        assert args.count("--ctx-size") == 1
