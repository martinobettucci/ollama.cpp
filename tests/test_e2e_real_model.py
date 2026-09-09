"""Bout en bout : `ollama.cpp` complet, vrai `llama-server`, vrai modèle GGUF.

@verifies docs/BACKLOG.md OC-085 « Bout en bout avec un vrai llama-server »
@verifies docs/ollama.cpp-architecture.md §5.1 « Vue d'ensemble », §8 risques R3, R4, R10
@verifies docs/DAT.md §2 « Services et processus », §3.1 « Requête d'inférence »

C'est la vérification que rien d'autre ne remplace : l'application réelle, un binaire
`llama-server` réellement compilé, un modèle GGUF réellement chargé, des tokens réellement
générés par `llama.cpp`. Aucune doublure nulle part.

Le modèle est produit par `scripts/make_test_model.py` : une architecture `llama` complète et
valide d'environ 460 Kio, aux poids aléatoires mais déterministes. Il **génère du charabia**, et
c'est sans importance — ce qui est vérifié ici, c'est la chaîne, pas la qualité du texte :
chargement, `/props`, tokenisation, rendu du chat template, génération, streaming, conversion
canonique, sérialisation des quatre façades, cycle de vie et ordonnancement.

Ce que ces tests ne prouvent PAS : qu'un modèle entraîné produit des réponses pertinentes, ni le
comportement d'offload GPU (risque R11, pas de GPU ici).

Activation : `OLLAMACPP_TEST_LLAMA_SERVER` doit désigner un binaire `llama-server`, et les
sources de `llama.cpp` doivent être trouvables pour `gguf-py` (`LLAMA_CPP_SOURCE`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ollamacpp.config import Config
from ollamacpp.names import parse as parse_name
from ollamacpp.storage import Artifacts, Manifest, RuntimeConfig

pytestmark = pytest.mark.e2e

RACINE = Path(__file__).resolve().parent.parent
GENERATEUR = RACINE / "scripts" / "make_test_model.py"


def _binaire() -> str | None:
    explicite = os.environ.get("OLLAMACPP_TEST_LLAMA_SERVER")
    if explicite and Path(explicite).is_file():
        return explicite
    return shutil.which("llama-server")


def _sources_llama_cpp() -> str | None:
    for candidat in (os.environ.get("LLAMA_CPP_SOURCE"), "/home/user/llama.cpp",
                     str(Path.home() / "llama.cpp")):
        if candidat and (Path(candidat) / "gguf-py" / "gguf" / "__init__.py").is_file():
            return candidat
    return None


BINAIRE = _binaire()
SOURCES = _sources_llama_cpp()

besoin_environnement = pytest.mark.skipif(
    BINAIRE is None or SOURCES is None,
    reason="requiert OLLAMACPP_TEST_LLAMA_SERVER et les sources de llama.cpp (gguf-py)",
)


@pytest.fixture(scope="session")
def modele_gguf(tmp_path_factory) -> Path:
    """Génère une fois par session le modèle GGUF de test, réellement chargeable."""
    if BINAIRE is None or SOURCES is None:
        pytest.skip("environnement incomplet")

    destination = tmp_path_factory.mktemp("gguf") / "test-llama.gguf"
    resultat = subprocess.run(
        [sys.executable, str(GENERATEUR), "--output", str(destination),
         "--llama-cpp", SOURCES],
        capture_output=True, text=True, timeout=180, check=False,
    )
    if resultat.returncode != 0 or not destination.is_file():
        pytest.skip(f"génération du modèle impossible : {resultat.stderr[-300:]}")
    return destination


@pytest.fixture
def config_reelle(tmp_path) -> Config:
    """Configuration pointant sur le VRAI binaire `llama-server`."""
    return Config(
        models_dir=tmp_path / "models",
        llama_server_bin=BINAIRE or "llama-server",
        llama_server_port_min=19200,
        llama_server_port_max=19399,
        # Un vrai chargement, même minuscule, demande plus qu'un faux serveur.
        load_timeout_s=90.0,
        request_timeout_s=90.0,
        max_loaded_models=2,
        memory_limit_bytes=8 * 1024 ** 3,
        memory_safety_margin=0.0,
        default_context=512,
    )


@pytest.fixture
def client_reel(config_reelle, modele_gguf):
    """Application complète, adossée au vrai `llama-server`, avec le modèle installé.

    Le modèle est installé par le **vrai** magasin et le **vrai** registre : le manifest écrit ici
    est identique à celui qu'aurait produit `/api/create`.
    """
    import warnings

    from fastapi.testclient import TestClient

    from ollamacpp.app import create_app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(create_app(config_reelle)) as client:
            registre = client.app.state.service.registry
            blob = registre.blobs.ingest_file(modele_gguf)

            registre.install(Manifest(
                name=parse_name("tiny:test"),
                artifacts=Artifacts(model=blob.digest),
                runtime=RuntimeConfig(context=512, parallel=1, threads=2),
            ))
            # Second modèle, en mode embeddings : `llama-server` refuse les embeddings sans
            # `--embedding`, et refuse la génération avec. Deux instances distinctes sont donc la
            # seule configuration réaliste.
            registre.install(Manifest(
                name=parse_name("tiny-embed:test"),
                artifacts=Artifacts(model=blob.digest),
                runtime=RuntimeConfig(context=512, threads=2, embedding=True, pooling="mean"),
            ))
            yield client


def ndjson(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def sse(response) -> list[tuple[str | None, dict]]:
    events: list[tuple[str | None, dict]] = []
    nom: str | None = None
    for ligne in response.text.splitlines():
        if ligne.startswith("event:"):
            nom = ligne[6:].strip()
        elif ligne.startswith("data:"):
            charge = ligne[5:].strip()
            if charge != "[DONE]":
                events.append((nom, json.loads(charge)))
            nom = None
    return events


# --- Catalogue -------------------------------------------------------------------------------------


@besoin_environnement
class TestCatalogueReel:
    def test_tags_expose_une_identite_reelle(self, client_reel, modele_gguf):
        """Risque R4 : `llama-server` renvoie des bouchons ; le registre doit faire mieux."""
        entrees = {m["name"]: m for m in client_reel.get("/api/tags").json()["models"]}
        entree = entrees["tiny:test"]

        assert entree["size"] == modele_gguf.stat().st_size
        assert len(entree["digest"]) == 64
        assert entree["details"]["family"] == "llama"
        assert entree["details"]["quantization_level"] == "F32"

    def test_show_lit_le_gguf_reel(self, client_reel):
        corps = client_reel.post("/api/show", json={"model": "tiny:test"}).json()
        assert corps["model_info"]["general.architecture"] == "llama"
        assert corps["model_info"]["llama.block_count"] == 2
        assert corps["model_info"]["llama.context_length"] == 512
        assert "{% for message in messages %}" in corps["template"]


# --- Inférence réelle -------------------------------------------------------------------------------


@besoin_environnement
class TestInferenceOllama:
    def test_chat_genere_des_tokens_reels(self, client_reel):
        corps = client_reel.post("/api/chat", json={
            "model": "tiny:test", "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "options": {"num_predict": 12, "temperature": 0, "seed": 7},
        }).json()

        assert corps["done"] is True
        assert corps["message"]["role"] == "assistant"
        # Le contenu est du charabia, mais il existe et il a été produit par llama.cpp.
        assert isinstance(corps["message"]["content"], str)
        assert corps["eval_count"] > 0
        assert corps["prompt_eval_count"] > 0

    def test_sortie_utf8_valide(self, client_reel):
        """Le modèle ne peut émettre que des octets ASCII : la sortie est toujours décodable.

        Vérifié plutôt que supposé — c'est la propriété qui rend ce modèle non entraîné
        exploitable de bout en bout (cf. `scripts/make_test_model.py`).
        """
        contenu = client_reel.post("/api/chat", json={
            "model": "tiny:test", "stream": False,
            "messages": [{"role": "user", "content": "hello world"}],
            "options": {"num_predict": 32, "temperature": 0},
        }).json()["message"]["content"]

        contenu.encode("utf-8").decode("utf-8")
        assert all(ord(c) < 128 for c in contenu), "seuls des octets ASCII sont émettables"

    def test_metriques_reelles_en_nanosecondes(self, client_reel):
        """Risque R3, sur de vraies mesures cette fois — pas un faux serveur."""
        corps = client_reel.post("/api/chat", json={
            "model": "tiny:test", "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
            "options": {"num_predict": 8},
        }).json()

        assert isinstance(corps["eval_duration"], int)
        assert corps["eval_duration"] > 0
        # Une durée réelle de génération de quelques tokens se compte en millisecondes, donc en
        # millions de nanosecondes. Un chiffre en secondes serait ici inférieur à 1.
        assert corps["eval_duration"] > 1000, "durée manifestement pas en nanosecondes"
        debit = corps["eval_count"] / corps["eval_duration"] * 1e9
        assert 0.1 < debit < 1_000_000, f"débit implausible : {debit:g} tokens/s"

    def test_chat_streame_en_ndjson(self, client_reel):
        reponse = client_reel.post("/api/chat", json={
            "model": "tiny:test",
            "messages": [{"role": "user", "content": "hello"}],
            "options": {"num_predict": 10, "temperature": 0},
        })
        assert reponse.headers["content-type"].startswith("application/x-ndjson")

        chunks = ndjson(reponse)
        assert len(chunks) > 1
        assert chunks[-1]["done"] is True
        assert all(c["done"] is False for c in chunks[:-1])
        assemble = "".join(c["message"]["content"] for c in chunks[:-1])
        assert assemble, "le flux doit porter du contenu"
        assert chunks[-1]["message"]["content"] == "", "pas de duplication du contenu"

    def test_generate(self, client_reel):
        corps = client_reel.post("/api/generate", json={
            "model": "tiny:test", "prompt": "hello", "stream": False,
            "options": {"num_predict": 8, "temperature": 0},
        }).json()
        assert corps["done"] is True
        assert isinstance(corps["response"], str)

    def test_embeddings_reels(self, client_reel):
        """Une instance distincte, lancée avec `--embedding`, produit de vrais vecteurs."""
        corps = client_reel.post("/api/embed", json={
            "model": "tiny-embed:test", "input": ["hello", "world"],
        }).json()

        assert len(corps["embeddings"]) == 2
        assert len(corps["embeddings"][0]) == 64, "dimension d'embedding du modèle"
        assert all(isinstance(v, float) for v in corps["embeddings"][0])

    def test_deux_entrees_donnent_deux_vecteurs_differents(self, client_reel):
        corps = client_reel.post("/api/embed", json={
            "model": "tiny-embed:test", "input": ["aaaa", "zzzz"],
        }).json()
        assert corps["embeddings"][0] != corps["embeddings"][1]


# --- Les quatre façades sur le même moteur ------------------------------------------------------------


@besoin_environnement
class TestQuatreFacadesSurLeMemeModele:
    """Exigence centrale de la mission : un seul runtime derrière les quatre APIs."""

    def test_openai_chat_completions(self, client_reel):
        corps = client_reel.post("/v1/chat/completions", json={
            "model": "tiny:test", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10, "temperature": 0,
        }).json()
        assert corps["object"] == "chat.completion"
        assert corps["usage"]["completion_tokens"] > 0
        assert isinstance(corps["choices"][0]["message"]["content"], str)

    def test_openai_streaming(self, client_reel):
        reponse = client_reel.post("/v1/chat/completions", json={
            "model": "tiny:test", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10, "stream": True, "temperature": 0,
        })
        assert reponse.text.rstrip().endswith("data: [DONE]")
        evenements = [charge for _nom, charge in sse(reponse)]
        assert evenements[-1]["choices"][0]["finish_reason"] in ("stop", "length")

    def test_openai_completions(self, client_reel):
        corps = client_reel.post("/v1/completions", json={
            "model": "tiny:test", "prompt": "hello", "max_tokens": 8, "temperature": 0,
        }).json()
        assert corps["object"] == "text_completion"

    def test_openai_responses(self, client_reel):
        corps = client_reel.post("/v1/responses", json={
            "model": "tiny:test", "input": "hello", "max_output_tokens": 10, "temperature": 0,
        }).json()
        assert corps["object"] == "response"
        assert corps["status"] == "completed"
        assert isinstance(corps["output_text"], str)

    def test_anthropic_messages(self, client_reel):
        corps = client_reel.post("/v1/messages", json={
            "model": "tiny:test", "max_tokens": 10, "temperature": 0,
            "messages": [{"role": "user", "content": "hello"}],
        }).json()
        assert corps["type"] == "message"
        assert corps["stop_reason"] in ("end_turn", "max_tokens")
        assert corps["usage"]["input_tokens"] > 0

    def test_anthropic_streaming(self, client_reel):
        reponse = client_reel.post("/v1/messages", json={
            "model": "tiny:test", "max_tokens": 10, "stream": True, "temperature": 0,
            "messages": [{"role": "user", "content": "hello"}],
        })
        noms = [nom for nom, _charge in sse(reponse)]
        assert noms[0] == "message_start"
        assert noms[-1] == "message_stop"

    def test_comptage_de_tokens_par_le_vrai_tokenizer(self, client_reel):
        """`count_tokens` passe par `/tokenize` de l'instance : c'est le vrai tokenizer SPM."""
        corps = client_reel.post("/v1/messages/count_tokens", json={
            "model": "tiny:test",
            "messages": [{"role": "user", "content": "hello world"}],
        }).json()
        assert corps["input_tokens"] > 0

    def test_une_seule_instance_pour_les_quatre_facades(self, client_reel):
        """Preuve qu'il n'y a pas quatre runtimes : un seul processus sert tout le monde."""
        for chemin, charge in [
            ("/api/chat", {"model": "tiny:test", "stream": False,
                           "messages": [{"role": "user", "content": "hi"}],
                           "options": {"num_predict": 4}}),
            ("/v1/chat/completions", {"model": "tiny:test", "max_tokens": 4,
                                      "messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/responses", {"model": "tiny:test", "input": "hi", "max_output_tokens": 4}),
            ("/v1/messages", {"model": "tiny:test", "max_tokens": 4,
                              "messages": [{"role": "user", "content": "hi"}]}),
        ]:
            assert client_reel.post(chemin, json=charge).status_code == 200, chemin

        residents = client_reel.get("/api/ps").json()["models"]
        assert [m["name"] for m in residents] == ["tiny:test"]


# --- Cycle de vie réel --------------------------------------------------------------------------------


@besoin_environnement
class TestCycleDeVieReel:
    def test_ps_reflete_une_instance_reellement_chargee(self, client_reel):
        assert client_reel.get("/api/ps").json()["models"] == []

        client_reel.post("/api/chat", json={
            "model": "tiny:test", "stream": False,
            "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 4},
        })

        resident = client_reel.get("/api/ps").json()["models"][0]
        assert resident["name"] == "tiny:test"
        assert resident["context_length"] == 512
        assert resident["size"] > 0
        # Aucune couche déportée : la seule réponse honnête est zéro, ce dont le CLI Ollama
        # déduit « 100% CPU ».
        assert resident["size_vram"] == 0
        assert resident["expires_at"]

    def test_keep_alive_zero_arrete_le_processus(self, client_reel):
        client_reel.post("/api/chat", json={
            "model": "tiny:test", "stream": False,
            "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 4},
        })
        assert len(client_reel.get("/api/ps").json()["models"]) == 1

        client_reel.post("/api/chat", json={"model": "tiny:test", "messages": [],
                                            "keep_alive": 0})
        assert client_reel.get("/api/ps").json()["models"] == []

    def test_num_ctx_recharge_avec_le_nouveau_contexte(self, client_reel):
        """Risque R2, vérifié sur une vraie instance : le plafond atteint `--ctx-size`."""
        client_reel.post("/api/chat", json={
            "model": "tiny:test", "stream": False,
            "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 4},
        })
        assert client_reel.get("/api/ps").json()["models"][0]["context_length"] == 512

        client_reel.post("/api/chat", json={
            "model": "tiny:test", "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 4, "num_ctx": 256},
        })
        assert client_reel.get("/api/ps").json()["models"][0]["context_length"] == 256

    def test_capacites_issues_du_vrai_props(self, client_reel):
        """OC-032 sur un vrai modèle : pas de `mmproj`, donc jamais `vision`."""
        client_reel.post("/api/chat", json={
            "model": "tiny:test", "stream": False,
            "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 4},
        })
        capacites = client_reel.post("/api/show", json={"model": "tiny:test"}).json()["capabilities"]
        assert "completion" in capacites
        assert "vision" not in capacites

    def test_eviction_reelle_sous_pression(self, client_reel):
        """`max_loaded_models` vaut 2 : le troisième chargement évince réellement un processus."""
        registre = client_reel.app.state.service.registry
        source = registre.get("tiny:test")
        for nom in ("a:test", "b:test"):
            registre.install(Manifest(
                name=parse_name(nom),
                artifacts=source.manifest.artifacts,
                runtime=RuntimeConfig(context=512, threads=2),
            ))

        for nom in ("tiny:test", "a:test", "b:test"):
            client_reel.post("/api/chat", json={
                "model": nom, "stream": False,
                "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 4},
            })

        residents = [m["name"] for m in client_reel.get("/api/ps").json()["models"]]
        assert len(residents) == 2
        assert "b:test" in residents
        assert "tiny:test" not in residents, "le plus ancien doit avoir été évincé"
