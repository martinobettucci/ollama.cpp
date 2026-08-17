"""Pull réel depuis Hugging Face, puis inférence sur le modèle téléchargé.

@verifies docs/BACKLOG.md OC-061 « Source Hugging Face », OC-050 « /api/pull »,
          OC-023 « Métadonnées GGUF », OC-085 « Vérification sur un vrai llama-server »
@verifies docs/ollama.cpp-architecture.md §5.10 « Arborescence de données », §8 risque R8
@verifies docs/DAT.md §3.3 « Téléchargement »
@verifies README.md « Accès réseau requis par `pull` depuis Hugging Face »

Le reste de la suite exerce le chemin Hugging Face contre un serveur local qui **reproduit** le
contrat de l'API : c'est ce qui rend la suite exécutable hors ligne, et c'est aussi sa limite. Un
serveur de test ne redirige pas vers un hôte de stockage distinct, ne signe pas d'URL, ne renvoie
pas d'en-tête `x-xet-hash` — autant de traits du vrai Hugging Face qui n'apparaissent que face à
lui.

Ce module fait donc le trajet réel, de bout en bout :

    huggingface.co/api/models → /resolve/main/<fichier> → 302 → hôte de stockage → blob → manifest

et vérifie ensuite que le modèle ainsi obtenu **se charge et génère**.

Prérequis, tous nécessaires ; à défaut le module est ignoré :

- `OLLAMACPP_TEST_HF_PULL=1`, parce que ces tests téléchargent près de 500 Mo ;
- `OLLAMACPP_TEST_LLAMA_SERVER` ou un `llama-server` dans le `PATH` ;
- un accès sortant vers `huggingface.co` **et** vers son hôte de stockage — voir la section
  « Accès réseau requis » du README : autoriser `huggingface.co` seul ne suffit pas, la
  résolution réussit alors que le téléchargement échoue.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e

#: Dépôt de référence : petit (491 Mo en Q4_K_M), publié par l'éditeur du modèle, réellement
#: entraîné, doté d'un template de conversation et de la capacité `tools`.
REPO = "hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FICHIER = "q4_k_m"
MODELE = f"{REPO}:{FICHIER}"

#: Taille exacte du fichier publié, annoncée par l'en-tête `x-linked-size` de Hugging Face. La
#: figer transforme « le téléchargement s'est terminé » en « le fichier reçu est le bon ».
TAILLE_ATTENDUE = 491_400_032


def _llama_server() -> str | None:
    explicite = os.environ.get("OLLAMACPP_TEST_LLAMA_SERVER")
    if explicite and Path(explicite).is_file():
        return explicite
    return shutil.which("llama-server")


LLAMA_SERVER = _llama_server()
ACTIVE = os.environ.get("OLLAMACPP_TEST_HF_PULL") == "1"

besoin_hf = pytest.mark.skipif(
    not ACTIVE or LLAMA_SERVER is None,
    reason="requiert OLLAMACPP_TEST_HF_PULL=1, OLLAMACPP_TEST_LLAMA_SERVER et un accès réseau à "
           "huggingface.co ainsi qu'à son hôte de stockage",
)


@pytest.fixture(scope="module")
def service_reel(tmp_path_factory):
    """Un `ollama.cpp` complet, avec le vrai binaire amont, sur un stockage vierge."""
    if not ACTIVE or LLAMA_SERVER is None:
        pytest.skip("environnement incomplet")

    from fastapi.testclient import TestClient

    from ollamacpp.app import create_app
    from ollamacpp.config import Config

    config = Config(
        models_dir=tmp_path_factory.mktemp("hf"),
        llama_server_bin=LLAMA_SERVER,
        llama_server_port_min=19800,
        llama_server_port_max=19899,
        default_context=2048,
        load_timeout_s=180.0,
    )
    with TestClient(create_app(config)) as client:
        yield client


@besoin_hf
class TestPullReel:
    """Le téléchargement lui-même, et ce qu'il dépose sur le disque."""

    def test_pull_aboutit(self, service_reel):
        """Le flux de progression doit se terminer par `success`, format d'Ollama."""
        reponse = service_reel.post("/api/pull", json={"model": MODELE, "stream": False},
                                    timeout=1800)
        assert reponse.status_code == 200, reponse.text
        assert reponse.json()["status"] == "success"

    def test_le_modele_est_liste(self, service_reel):
        noms = [m["name"] for m in service_reel.get("/api/tags").json()["models"]]
        assert MODELE in noms

    def test_taille_reelle_du_fichier_publie(self, service_reel):
        """La taille doit être celle annoncée par Hugging Face, à l'octet près."""
        modele = next(m for m in service_reel.get("/api/tags").json()["models"]
                      if m["name"] == MODELE)
        assert modele["size"] == TAILLE_ATTENDUE

    def test_le_blob_porte_le_digest_calcule_localement(self, service_reel):
        """Le nom du blob est un digest **recalculé** ici, jamais une valeur annoncée (risque R8)."""
        import hashlib

        modele = next(m for m in service_reel.get("/api/tags").json()["models"]
                      if m["name"] == MODELE)
        digest = modele["digest"]
        blob = Path(service_reel.app.state.service.config.blobs_dir) / f"sha256-{digest}"
        assert blob.is_file(), f"blob absent : {blob}"

        somme = hashlib.sha256()
        with blob.open("rb") as flux:
            for bloc in iter(lambda: flux.read(1024 * 1024), b""):
                somme.update(bloc)
        assert somme.hexdigest() == digest

    def test_la_source_est_tracee_dans_le_manifest(self, service_reel):
        """Le manifest doit retenir d'où vient le modèle, pour un `pull` ultérieur."""
        installe = service_reel.app.state.service.registry.get(MODELE)
        source = installe.manifest.source
        assert source is not None
        assert source.type == "huggingface"
        assert source.reference == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"


@besoin_hf
class TestModeleTelecharge:
    """Ce que `/api/show` annonce du modèle réel, et ce qu'il sait faire."""

    def test_capacites_observees(self, service_reel):
        """Qwen2.5-Instruct sait appeler des outils et n'a pas d'encodeur d'image.

        `vision` doit être **absent** : la mission (§13) interdit d'annoncer une capacité que le
        modèle ne possède pas, et ce dépôt ne publie aucun projecteur.
        """
        capacites = service_reel.post("/api/show", json={"model": MODELE}).json()["capabilities"]
        assert "completion" in capacites
        assert "tools" in capacites
        assert "vision" not in capacites

    def test_compte_de_parametres_calcule(self, service_reel):
        """Ce GGUF ne porte **pas** `general.parameter_count` ; il doit être calculé.

        Défaut trouvé sur ce modèle : s'en remettre à la clé laissait `parameter_size` vide et
        `ollama show` affichait une ligne « parameters » sans valeur. Le compte vient désormais de
        la table des tenseurs, comme chez Ollama. Le contrôle croisé est `general.size_label`,
        renseigné indépendamment par l'éditeur : 630M.
        """
        corps = service_reel.post("/api/show", json={"model": MODELE}).json()
        compte = corps["model_info"]["general.parameter_count"]
        assert compte == 630_167_424
        assert corps["details"]["parameter_size"] == "630.17M"
        assert corps["model_info"]["general.size_label"] == "630M"

    def test_generation_reelle(self, service_reel):
        """Un modèle entraîné doit produire une réponse **juste**, pas seulement du texte.

        C'est ce que le modèle de test fabriqué localement ne pouvait pas prouver : ses poids
        étant aléatoires, il générait des jetons valides et un contenu dépourvu de sens.
        """
        reponse = service_reel.post("/api/chat", json={
            "model": MODELE, "stream": False,
            "options": {"temperature": 0, "seed": 7},
            "messages": [{"role": "user",
                          "content": "En une phrase : quelle est la capitale de la France ?"}],
        }, timeout=600)
        assert reponse.status_code == 200, reponse.text
        corps = reponse.json()
        assert corps["done"] is True
        assert corps["done_reason"] == "stop"
        assert "Paris" in corps["message"]["content"]
        assert corps["eval_count"] > 0
        assert corps["prompt_eval_count"] > 0

    def test_appel_doutil_reel(self, service_reel):
        """Un vrai appel d'outil, décidé par le modèle et non par une simulation."""
        reponse = service_reel.post("/api/chat", json={
            "model": MODELE, "stream": False,
            "options": {"temperature": 0, "seed": 7},
            "messages": [{"role": "user", "content": "Quelle est la météo à Lyon ? Utilise l'outil."}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "Météo actuelle d'une ville",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}],
        }, timeout=600)
        assert reponse.status_code == 200, reponse.text
        appels = reponse.json()["message"].get("tool_calls") or []
        assert appels, "le modèle devait demander l'outil"
        assert appels[0]["function"]["name"] == "get_weather"
        assert appels[0]["function"]["arguments"] == {"city": "Lyon"}

    def test_boucle_doutil_complete(self, service_reel):
        """Le résultat d'outil doit revenir au modèle et nourrir la réponse finale (risque R7)."""
        premier = service_reel.post("/api/chat", json={
            "model": MODELE, "stream": False,
            "options": {"temperature": 0, "seed": 7},
            "messages": [{"role": "user", "content": "Quelle est la météo à Lyon ? Utilise l'outil."}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "Météo actuelle d'une ville",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}],
        }, timeout=600).json()

        second = service_reel.post("/api/chat", json={
            "model": MODELE, "stream": False,
            "options": {"temperature": 0, "seed": 7},
            "messages": [
                {"role": "user", "content": "Quelle est la météo à Lyon ? Utilise l'outil."},
                premier["message"],
                {"role": "tool", "tool_name": "get_weather",
                 "content": '{"city": "Lyon", "temperature_c": 17, "condition": "ensoleille"}'},
            ],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "Météo actuelle d'une ville",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}],
        }, timeout=600).json()
        assert "17" in second["message"]["content"], second["message"]["content"]


@besoin_hf
class TestReferenceInvalide:
    """Contre-épreuves : un échec doit être un échec, pas un succès silencieux."""

    def test_depot_inexistant(self, service_reel):
        reponse = service_reel.post(
            "/api/pull",
            json={"model": "hf.co/ollamacpp-tests/depot-qui-nexiste-pas", "stream": False},
            timeout=120,
        )
        assert reponse.status_code >= 400
        assert "error" in reponse.json()

    def test_fichier_inexistant_dans_un_depot_reel(self, service_reel):
        reponse = service_reel.post(
            "/api/pull",
            json={"model": f"{REPO}:quantification-inexistante", "stream": False},
            timeout=120,
        )
        assert reponse.status_code >= 400
        assert "error" in reponse.json()


@besoin_hf
def test_la_redirection_vers_lhote_de_stockage_est_bien_suivie():
    """Documente le prérequis réseau en le mesurant, plutôt qu'en l'affirmant.

    Autoriser `huggingface.co` ne suffit pas : la route `/resolve/` répond `302` vers un hôte
    distinct qui porte l'octet. Ce test échoue précisément lorsque seule l'API est ouverte — le
    diagnostic est alors immédiat au lieu de se manifester par un téléchargement interrompu.
    """
    url = (f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
           f"/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    with httpx.Client(follow_redirects=False, timeout=60.0) as client:
        premiere = client.head(url)
    assert premiere.status_code in (301, 302, 307, 308), (
        f"redirection attendue, reçu {premiere.status_code}")
    cible = premiere.headers["location"]
    assert cible.startswith("https://"), cible

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        finale = client.head(url)
    assert finale.status_code == 200, (
        f"hôte de stockage inaccessible ({finale.status_code}) : autoriser le domaine de "
        f"{cible.split('/')[2]} en sortie — voir README, « Accès réseau requis »")
    assert int(finale.headers.get("content-length", 0)) == TAILLE_ATTENDUE
