"""Contrat avec le VRAI binaire `llama-server`.

@verifies docs/BACKLOG.md OC-031 « Construction des arguments runtime », OC-085 « bout en bout »
@verifies docs/ollama.cpp-architecture.md §1.5 « Arguments CLI pertinents », §8 risque R10
@verifies docs/DAT.md §5.2 « Interfaces consommées »

**Risque R10 — divergence de l'upstream.** `ollama.cpp` ne dépend de `llama.cpp` que par sa
surface publique : les drapeaux CLI et les endpoints HTTP. Ce fichier vérifie la première moitié
de ce contrat contre le **binaire réellement compilé**, et non contre une doublure : si une mise à
jour de `llama.cpp` renommait ou supprimait un drapeau que le middleware émet, ce test échouerait
au lieu de laisser le défaut se manifester en production.

La méthode exploite le fait que `llama-server` valide ses arguments **avant** de toucher au
modèle :

- un drapeau inconnu produit `error: invalid argument: --xxx` et un arrêt immédiat ;
- des arguments valides mènent jusqu'au chargement, qui échoue alors sur le fichier absent.

Atteindre l'erreur de chargement prouve donc que toute la ligne de commande a été acceptée.

**Limite assumée** : ces tests ne couvrent pas l'inférence réelle. L'environnement de
construction de référence n'a pas accès au réseau de téléchargement de modèles (politique
réseau), donc aucun GGUF n'y est disponible. La partie « inférence sur un vrai modèle » d'OC-085
reste **non vérifiée** et est suivie comme telle dans `docs/BACKLOG.md`.

Activation : les tests s'exécutent si `OLLAMACPP_TEST_LLAMA_SERVER` désigne un binaire
`llama-server`, sinon ils sont ignorés.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ollamacpp.runtime.args import build_args
from ollamacpp.storage import RuntimeConfig

pytestmark = pytest.mark.e2e


def _binary() -> str | None:
    explicite = os.environ.get("OLLAMACPP_TEST_LLAMA_SERVER")
    if explicite and Path(explicite).is_file():
        return explicite
    return shutil.which("llama-server")


BINAIRE = _binary()

besoin_binaire = pytest.mark.skipif(
    BINAIRE is None,
    reason="binaire llama-server absent : définir OLLAMACPP_TEST_LLAMA_SERVER",
)

#: Marqueurs d'un refus au niveau de l'analyse des arguments.
REFUS_ARGUMENT = ("invalid argument", "unknown argument", "unrecognized")


def lancer(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BINAIRE, *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def args_pour(runtime: RuntimeConfig, **kwargs) -> list[str]:
    return build_args(
        model_path=Path("/inexistant-pour-le-test.gguf"),
        alias="contrat", host="127.0.0.1", port=18999, runtime=runtime, **kwargs
    )


def assert_arguments_acceptes(resultat: subprocess.CompletedProcess, args: list[str]) -> None:
    """Vérifie qu'aucun argument n'a été refusé par l'analyseur de `llama-server`."""
    sortie = (resultat.stdout + resultat.stderr).lower()
    for marqueur in REFUS_ARGUMENT:
        assert marqueur not in sortie, (
            f"llama-server a refusé un argument de la ligne suivante :\n"
            f"  {' '.join(args)}\n"
            f"sortie : {(resultat.stdout + resultat.stderr)[-400:]}"
        )


@besoin_binaire
class TestSanityDuHarnais:
    """Contre-épreuves : sans elles, un test qui n'échoue jamais ne prouve rien."""

    def test_un_drapeau_inconnu_est_bien_refuse(self):
        resultat = lancer(["--model", "/x.gguf", "--drapeau-qui-nexiste-pas"])
        sortie = (resultat.stdout + resultat.stderr).lower()
        assert any(marqueur in sortie for marqueur in REFUS_ARGUMENT)

    def test_binaire_construit_depuis_la_revision_auditee(self):
        """Trace la version réellement testée, pour que l'échec soit interprétable."""
        resultat = lancer(["--version"])
        sortie = resultat.stdout + resultat.stderr
        assert "version" in sortie.lower()


@besoin_binaire
class TestLigneDeCommandeAcceptee:
    """Chaque configuration runtime que le middleware sait produire doit être acceptée."""

    def test_configuration_minimale(self):
        args = args_pour(RuntimeConfig())
        resultat = lancer(args)
        assert_arguments_acceptes(resultat, args)

    def test_arguments_toujours_emis(self):
        """`--no-webui` et `--jinja` sont posés systématiquement : ils doivent exister."""
        args = args_pour(RuntimeConfig())
        assert "--no-webui" in args and "--jinja" in args
        assert_arguments_acceptes(lancer(args), args)

    def test_contexte_batch_parallelisme(self):
        args = args_pour(RuntimeConfig(context=2048, batch=256, ubatch=64, parallel=2,
                                       threads=2, threads_batch=2))
        assert_arguments_acceptes(lancer(args), args)

    def test_cache_kv_quantise(self):
        """L'exemple exact de la mission §16 : K en IQ4_NL, V en Q8_0."""
        args = args_pour(RuntimeConfig(cache_type_k="iq4_nl", cache_type_v="q8_0",
                                       flash_attention=True))
        assert_arguments_acceptes(lancer(args), args)

    @pytest.mark.parametrize("actif", [True, False])
    def test_flash_attention_dans_les_deux_sens(self, actif):
        """`--flash-attn` attend `on|off|auto` : une valeur booléenne serait refusée."""
        args = args_pour(RuntimeConfig(flash_attention=actif))
        assert_arguments_acceptes(lancer(args), args)

    def test_repartition_gpu(self):
        args = args_pour(RuntimeConfig(gpu_layers=0, tensor_split="1.0", main_gpu=0))
        assert_arguments_acceptes(lancer(args), args)

    def test_drapeaux_negatifs(self):
        args = args_pour(RuntimeConfig(mmap=False, kv_offload=False, mlock=False))
        assert_arguments_acceptes(lancer(args), args)

    def test_decoding_speculatif(self):
        args = args_pour(RuntimeConfig(draft_max=8, draft_min=2, draft_p_min=0.7),
                         draft_path=Path("/inexistant-draft.gguf"))
        assert_arguments_acceptes(lancer(args), args)

    def test_multimodal_et_adaptateurs(self):
        args = args_pour(RuntimeConfig(),
                         mmproj_path=Path("/inexistant-mmproj.gguf"),
                         adapter_paths=(Path("/inexistant-lora.gguf"),))
        assert_arguments_acceptes(lancer(args), args)

    def test_modes_embedding_et_rerank(self):
        for runtime in (RuntimeConfig(embedding=True, pooling="mean"),
                        RuntimeConfig(reranking=True)):
            args = args_pour(runtime)
            assert_arguments_acceptes(lancer(args), args)

    def test_raisonnement(self):
        args = args_pour(RuntimeConfig(reasoning_format="auto", reasoning_budget=-1))
        assert_arguments_acceptes(lancer(args), args)

    def test_configuration_complete(self):
        """Toutes les options modélisées à la fois : la combinaison doit rester valide."""
        args = args_pour(
            RuntimeConfig(
                context=4096, batch=512, ubatch=128, parallel=2, threads=2, threads_batch=2,
                gpu_layers=0, main_gpu=0, flash_attention=True,
                cache_type_k="q8_0", cache_type_v="q8_0", kv_unified=True,
                mmap=False, mlock=False, draft_max=8, draft_min=2, draft_p_min=0.7,
                reasoning_format="auto",
            )
        )
        assert_arguments_acceptes(lancer(args), args)

    def test_arrive_bien_jusquau_chargement(self):
        """Preuve positive : l'échec porte sur le MODÈLE, donc les arguments ont tous été acceptés."""
        args = args_pour(RuntimeConfig(context=2048))
        resultat = lancer(args)
        sortie = resultat.stdout + resultat.stderr
        assert resultat.returncode != 0
        assert "inexistant-pour-le-test.gguf" in sortie
        assert "failed to load model" in sortie.lower()


@besoin_binaire
class TestDrapeauxDocumentes:
    """Les drapeaux dont dépend le middleware doivent figurer dans l'aide du binaire."""

    ATTENDUS = [
        "--model", "--alias", "--host", "--port", "--jinja",
        "--ctx-size", "--batch-size", "--ubatch-size", "--parallel", "--threads",
        "--n-gpu-layers", "--tensor-split", "--main-gpu",
        "--flash-attn", "--cache-type-k", "--cache-type-v",
        "--mmproj", "--model-draft", "--lora",
        "--embedding", "--reranking", "--pooling",
        "--chat-template", "--reasoning-format", "--reasoning-budget",
    ]

    @pytest.fixture(scope="class")
    @classmethod
    def aide(cls) -> str:
        resultat = lancer(["--help"])
        return resultat.stdout + resultat.stderr

    @pytest.mark.parametrize("drapeau", ATTENDUS)
    def test_drapeau_present_dans_laide(self, aide: str, drapeau: str):
        assert drapeau in aide, (
            f"{drapeau} a disparu de llama-server : le middleware doit être adapté (risque R10)"
        )


@besoin_binaire
class TestSurfaceHttpReelle:
    """Surface HTTP vérifiée sur un `llama-server` réellement démarré.

    Le mode routeur (`--models-dir`) démarre **sans charger de modèle** : c'est ce qui permet de
    vérifier en vrai le contrat HTTP dont dépend le superviseur, même sans GGUF disponible.
    """

    @pytest.fixture
    def routeur(self, tmp_path):
        """Démarre un vrai `llama-server` en mode routeur et attend qu'il réponde."""
        import socket
        import time

        with socket.socket() as sonde:
            sonde.bind(("127.0.0.1", 0))
            port = sonde.getsockname()[1]

        modeles = tmp_path / "models"
        modeles.mkdir()
        processus = subprocess.Popen(
            [BINAIRE, "--models-dir", str(modeles), "--host", "127.0.0.1",
             "--port", str(port), "--no-webui"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            import httpx

            limite = time.monotonic() + 30
            while time.monotonic() < limite:
                if processus.poll() is not None:
                    pytest.fail("llama-server s'est arrêté avant d'écouter")
                try:
                    if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.2)
            else:
                pytest.fail("llama-server n'a pas répondu dans le délai imparti")
            yield base
        finally:
            processus.terminate()
            try:
                processus.wait(timeout=15)
            except subprocess.TimeoutExpired:
                processus.kill()

    def test_health_repond_comme_attendu(self, routeur):
        """Contrat exact sur lequel repose `LlamaServerSupervisor._await_health`."""
        import httpx

        reponse = httpx.get(f"{routeur}/health", timeout=5.0)
        assert reponse.status_code == 200
        assert reponse.json() == {"status": "ok"}

    def test_v1_models_sert_la_forme_openai(self, routeur):
        import httpx

        payload = httpx.get(f"{routeur}/v1/models", timeout=5.0).json()
        assert payload["object"] == "list"
        assert isinstance(payload["data"], list)

    def test_endpoints_dinference_presents_dans_limplementation(self):
        """Les chemins consommés existent bien dans le code servi.

        Les vérifier en HTTP demanderait un modèle chargé, impossible ici (cf. l'en-tête de ce
        fichier). La recherche porte sur le binaire **et** ses bibliothèques : `llama-server` est
        un lanceur mince dont l'implémentation vit dans `libllama-server-impl`.
        """
        racine = Path(BINAIRE).parent
        contenu = Path(BINAIRE).read_bytes()
        for bibliotheque in racine.glob("libllama-server*"):
            if bibliotheque.is_file():
                contenu += bibliotheque.read_bytes()

        for chemin in (b"/health", b"/props", b"/v1/chat/completions", b"/v1/embeddings",
                       b"/tokenize", b"/v1/models"):
            assert chemin in contenu, f"{chemin!r} absent de llama-server (risque R10)"
