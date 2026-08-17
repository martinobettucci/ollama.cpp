"""Compatibilité avec le **vrai binaire** `ollama`.

@verifies docs/BACKLOG.md OC-084 « Compatibilité du CLI Ollama », OC-043 « /api/ps »
@verifies docs/ollama.cpp-architecture.md §9 « Résultat visé », §2.4 « Schémas »
@verifies docs/DAT.md §5.1 « Interfaces exposées »

C'est la vérification du résultat annoncé par la mission (§39) :

    OLLAMA_HOST=http://localhost:11434 ollama list

Le CLI Ollama officiel est lancé, sans adaptation d'aucune sorte, contre un `ollama.cpp` réel
servant un vrai modèle via un vrai `llama-server`. Un client qui ne sait rien de `ollama.cpp` doit
le prendre pour un Ollama.

Ces tests ont trouvé un défaut réel que la suite HTTP ne pouvait pas voir : `/api/ps` renvoyait
`size_vram = size` en toutes circonstances, ce qui faisait afficher « 100% GPU » par le CLI sur un
serveur calculant intégralement sur CPU. Le CLI déduit cette colonne du rapport `size_vram/size`
(`cmd/cmd.go` l. 1141-1150) — un détail invisible en JSON, mensonger à l'écran.

Activation : `OLLAMACPP_TEST_OLLAMA_CLI` doit désigner un binaire `ollama`, en plus des
prérequis de `test_e2e_real_model.py`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

RACINE = Path(__file__).resolve().parent.parent


def _cli() -> str | None:
    explicite = os.environ.get("OLLAMACPP_TEST_OLLAMA_CLI")
    if explicite and Path(explicite).is_file():
        return explicite
    return shutil.which("ollama")


def _llama_server() -> str | None:
    explicite = os.environ.get("OLLAMACPP_TEST_LLAMA_SERVER")
    if explicite and Path(explicite).is_file():
        return explicite
    return shutil.which("llama-server")


def _sources() -> str | None:
    for candidat in (os.environ.get("LLAMA_CPP_SOURCE"), "/home/user/llama.cpp"):
        if candidat and (Path(candidat) / "gguf-py" / "gguf" / "__init__.py").is_file():
            return candidat
    return None


CLI = _cli()
LLAMA_SERVER = _llama_server()
SOURCES = _sources()

besoin_cli = pytest.mark.skipif(
    CLI is None or LLAMA_SERVER is None or SOURCES is None,
    reason="requiert OLLAMACPP_TEST_OLLAMA_CLI, OLLAMACPP_TEST_LLAMA_SERVER et gguf-py",
)


@pytest.fixture(scope="module")
def service_ollama_cpp(tmp_path_factory):
    """Démarre un `ollama.cpp` réel, en processus séparé, avec un vrai modèle installé.

    Un vrai processus est indispensable : le CLI parle en HTTP à une adresse, il ne peut pas être
    branché sur un client de test en mémoire.
    """
    if CLI is None or LLAMA_SERVER is None or SOURCES is None:
        pytest.skip("environnement incomplet")

    base = tmp_path_factory.mktemp("cli")
    gguf = base / "test-llama.gguf"
    genere = subprocess.run(
        [sys.executable, str(RACINE / "scripts" / "make_test_model.py"),
         "--output", str(gguf), "--llama-cpp", SOURCES],
        capture_output=True, text=True, timeout=180, check=False,
    )
    if genere.returncode != 0 or not gguf.is_file():
        pytest.skip(f"génération du modèle impossible : {genere.stderr[-200:]}")

    modeles = base / "models"
    sys.path.insert(0, str(RACINE))
    from ollamacpp.config import Config
    from ollamacpp.names import parse as parse_name
    from ollamacpp.registry import ModelRegistry
    from ollamacpp.storage import Artifacts, BlobStore, Manifest, ManifestStore, RuntimeConfig

    config = Config(models_dir=modeles)
    config.ensure_layout()
    blobs = BlobStore(config.blobs_dir, config.tmp_dir)
    blobs.ensure_layout()
    registre = ModelRegistry(blobs, ManifestStore(config.manifests_dir))
    blob = blobs.ingest_file(gguf)
    registre.install(Manifest(
        name=parse_name("tiny:test"),
        artifacts=Artifacts(model=blob.digest),
        runtime=RuntimeConfig(context=512, threads=2),
    ))

    port = 11577
    service = subprocess.Popen(
        [sys.executable, "-m", "ollamacpp"],
        cwd=str(RACINE),
        env={
            **os.environ,
            "OLLAMACPP_MODELS": str(modeles),
            "OLLAMACPP_HOST": "127.0.0.1",
            "OLLAMACPP_PORT": str(port),
            "OLLAMACPP_LLAMA_SERVER_BIN": LLAMA_SERVER,
            "OLLAMACPP_LLAMA_SERVER_PORT_MIN": "19500",
            "OLLAMACPP_LLAMA_SERVER_PORT_MAX": "19599",
            "OLLAMACPP_LOAD_TIMEOUT_S": "90",
            "OLLAMACPP_DEFAULT_CONTEXT": "512",
        },
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    adresse = f"http://127.0.0.1:{port}"
    try:
        import httpx

        limite = time.monotonic() + 60
        while time.monotonic() < limite:
            if service.poll() is not None:
                pytest.fail(f"ollama.cpp s'est arrêté : {service.stdout.read()[-500:]}")
            try:
                if httpx.get(f"{adresse}/api/version", timeout=2.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.3)
        else:
            pytest.fail("ollama.cpp n'a pas démarré dans le délai imparti")
        yield adresse
    finally:
        service.terminate()
        try:
            service.wait(timeout=25)
        except subprocess.TimeoutExpired:
            service.kill()


def ollama(service_ollama_cpp: str, *args: str, timeout: float = 60.0):
    """Exécute le CLI Ollama contre `ollama.cpp`, sans aucune adaptation."""
    return subprocess.run(
        [CLI, *args],
        env={**os.environ, "OLLAMA_HOST": service_ollama_cpp,
             "NO_PROXY": "*", "no_proxy": "*"},
        capture_output=True, text=True, timeout=timeout, check=False,
    )


@besoin_cli
class TestCliOllama:
    """Les commandes du CLI officiel, dans leur ordre d'usage naturel."""

    def test_list(self, service_ollama_cpp):
        """`ollama list` : la commande de la mission §39."""
        resultat = ollama(service_ollama_cpp, "list")
        assert resultat.returncode == 0, resultat.stderr
        assert "tiny:test" in resultat.stdout
        assert "NAME" in resultat.stdout and "SIZE" in resultat.stdout

    def test_list_affiche_une_taille_reelle(self, service_ollama_cpp):
        """Le CLI formate `size` : une taille vide ou nulle serait immédiatement visible."""
        sortie = ollama(service_ollama_cpp, "list").stdout
        assert "KB" in sortie or "MB" in sortie or "GB" in sortie

    def test_list_affiche_un_identifiant(self, service_ollama_cpp):
        """Le CLI tronque le digest à 12 caractères : un digest vide le ferait planter."""
        lignes = [l for l in ollama(service_ollama_cpp, "list").stdout.splitlines()
                  if "tiny:test" in l]
        assert lignes
        colonnes = lignes[0].split()
        assert len(colonnes[1]) == 12, f"identifiant inattendu : {colonnes[1]!r}"

    def test_show(self, service_ollama_cpp):
        resultat = ollama(service_ollama_cpp, "show", "tiny:test")
        assert resultat.returncode == 0, resultat.stderr
        sortie = resultat.stdout
        assert "architecture" in sortie and "llama" in sortie
        assert "context length" in sortie and "512" in sortie
        assert "Capabilities" in sortie and "completion" in sortie

    def test_ps_vide_avant_chargement(self, service_ollama_cpp):
        resultat = ollama(service_ollama_cpp, "ps")
        assert resultat.returncode == 0, resultat.stderr
        assert "NAME" in resultat.stdout

    def test_run_produit_une_reponse(self, service_ollama_cpp):
        """`ollama run` : chargement, rendu du template, génération, streaming, affichage."""
        resultat = ollama(service_ollama_cpp, "run", "tiny:test", "hello", timeout=180)
        assert resultat.returncode == 0, resultat.stderr
        assert resultat.stdout.strip(), "le CLI doit afficher du texte généré"

    def test_ps_apres_chargement(self, service_ollama_cpp):
        """Le modèle doit apparaître résident, avec son contexte et son échéance."""
        ollama(service_ollama_cpp, "run", "tiny:test", "hi", timeout=180)
        sortie = ollama(service_ollama_cpp, "ps").stdout
        assert "tiny:test" in sortie
        assert "512" in sortie, "la colonne CONTEXT doit refléter le contexte réel"
        assert "from now" in sortie, "la colonne UNTIL doit refléter expires_at"

    def test_ps_nannonce_pas_de_gpu_sur_un_serveur_cpu(self, service_ollama_cpp):
        """Défaut trouvé par ce test : `size_vram = size` faisait afficher « 100% GPU ».

        Le CLI déduit la colonne PROCESSOR du rapport `size_vram/size` (`cmd/cmd.go`
        l. 1141-1150). Sans couches déportées, la seule réponse honnête est « 100% CPU ».
        """
        ollama(service_ollama_cpp, "run", "tiny:test", "hi", timeout=180)
        lignes = [l for l in ollama(service_ollama_cpp, "ps").stdout.splitlines()
                  if "tiny:test" in l]
        assert lignes
        assert "100% CPU" in lignes[0], f"colonne PROCESSOR trompeuse : {lignes[0]!r}"

    def test_cp(self, service_ollama_cpp):
        resultat = ollama(service_ollama_cpp, "cp", "tiny:test", "copie-cli:v1")
        assert resultat.returncode == 0, resultat.stderr
        assert "copie-cli:v1" in ollama(service_ollama_cpp, "list").stdout

    def test_rm(self, service_ollama_cpp):
        ollama(service_ollama_cpp, "cp", "tiny:test", "jetable-cli:v1")
        resultat = ollama(service_ollama_cpp, "rm", "jetable-cli:v1")
        assert resultat.returncode == 0, resultat.stderr
        assert "jetable-cli:v1" not in ollama(service_ollama_cpp, "list").stdout

    def test_rm_dun_modele_absent_echoue(self, service_ollama_cpp):
        """Contre-épreuve : le CLI doit voir l'échec, pas un succès silencieux."""
        resultat = ollama(service_ollama_cpp, "rm", "nexiste-pas:v1")
        assert resultat.returncode != 0

    def test_show_dun_modele_absent_echoue(self, service_ollama_cpp):
        resultat = ollama(service_ollama_cpp, "show", "nexiste-pas:v1")
        assert resultat.returncode != 0
        assert "not found" in (resultat.stdout + resultat.stderr).lower()
