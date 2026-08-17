"""Tests du téléchargement de modèles : registre privé et Hugging Face.

@verifies docs/BACKLOG.md OC-050 « /api/pull », OC-061 « Source Hugging Face »,
          OC-062 « Registre privé natif »
@verifies docs/ollama.cpp-architecture.md §5.10 « Arborescence de données », risques R8 et R9
@verifies docs/DAT.md §3.3 « Téléchargement », §7 « Sécurité »

Les registres sont de **vrais serveurs HTTP** locaux, pas des doublures : le client httpx réel
négocie une vraie connexion, télécharge un vrai flux et vérifie un vrai checksum. Seul le contenu
servi est déterministe.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from .ggufbuild import DEFAULT_KV, build_gguf, build_mmproj

MODELE = build_gguf(DEFAULT_KV)
MODELE_DIGEST = "sha256:" + hashlib.sha256(MODELE).hexdigest()
MMPROJ = build_mmproj()
MMPROJ_DIGEST = "sha256:" + hashlib.sha256(MMPROJ).hexdigest()


class RegistreHandler(BaseHTTPRequestHandler):
    """Registre privé minimal, conforme au contrat décrit dans `ollamacpp/sources.py`."""

    protocol_version = "HTTP/1.1"
    #: En-têtes reçus, pour vérifier que le jeton est bien transmis mais jamais journalisé.
    recus: list[dict[str, str]] = []
    #: Permet de simuler un artefact dont le contenu ne correspond pas au checksum annoncé.
    corrompre = False

    def log_message(self, *_args) -> None:
        pass

    def _send(self, body: bytes, content_type: str = "application/json", status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        type(self).recus.append(dict(self.headers))

        if self.path.startswith("/v1/models/"):
            nom = self.path[len("/v1/models/"):]
            if nom == "absent":
                return self._send(json.dumps({"error": "not found"}).encode(), status=404)
            if nom == "interdit":
                return self._send(json.dumps({"error": "forbidden"}).encode(), status=403)
            if nom == "sans-checksum":
                return self._send(json.dumps({
                    "artifacts": {"model": {"url": "/blobs/model.gguf"}}
                }).encode())

            artifacts = {"model": {"url": "/blobs/model.gguf", "digest": MODELE_DIGEST,
                                   "name": "model.gguf"}}
            if nom == "vision":
                artifacts["mmproj"] = {"url": "/blobs/mmproj.gguf", "digest": MMPROJ_DIGEST,
                                       "name": "mmproj.gguf"}
            return self._send(json.dumps({"artifacts": artifacts}).encode())

        if self.path == "/blobs/model.gguf":
            contenu = b"contenu substitue" if type(self).corrompre else MODELE
            return self._send(contenu, content_type="application/octet-stream")
        if self.path == "/blobs/mmproj.gguf":
            return self._send(MMPROJ, content_type="application/octet-stream")

        # --- Surface Hugging Face ---
        if self.path.startswith("/api/models/"):
            depot = self.path[len("/api/models/"):]
            if depot == "org/absent":
                return self._send(json.dumps({"error": "not found"}).encode(), status=404)
            if depot == "org/sans-gguf":
                return self._send(json.dumps({"siblings": [{"rfilename": "README.md"}]}).encode())
            return self._send(json.dumps({"siblings": [
                {"rfilename": "modele-Q4_K_M.gguf"},
                {"rfilename": "modele-Q8_0.gguf"},
                {"rfilename": "mmproj-f16.gguf"},
            ]}).encode())

        if "/resolve/main/" in self.path:
            fichier = self.path.rsplit("/", 1)[-1]
            contenu = MMPROJ if "mmproj" in fichier else MODELE
            return self._send(contenu, content_type="application/octet-stream")

        self._send(json.dumps({"error": "not found"}).encode(), status=404)


@pytest.fixture
def registre_url():
    """Démarre un vrai serveur HTTP local jouant le registre privé et Hugging Face."""
    RegistreHandler.recus = []
    RegistreHandler.corrompre = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), RegistreHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def client_registre(config, registre_url):
    """Application configurée pour utiliser le registre local, en privé comme en Hugging Face."""
    import dataclasses
    import warnings

    from fastapi.testclient import TestClient

    from ollamacpp.app import create_app

    configuree = dataclasses.replace(
        config,
        registry_url=registre_url,
        registry_token="jeton-de-test-tres-secret",
        hf_endpoint=registre_url,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(create_app(configuree)) as test_client:
            yield test_client


def ndjson(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


# --- Registre privé ------------------------------------------------------------------------------


class TestRegistrePrive:
    def test_pull_installe_le_modele(self, client_registre):
        response = client_registre.post("/api/pull", json={"model": "qwen3-test",
                                                           "stream": False})
        assert response.status_code == 200
        assert response.json() == {"status": "success"}

        noms = [m["name"] for m in client_registre.get("/api/tags").json()["models"]]
        assert "qwen3-test:latest" in noms

    def test_artefact_installe_avec_le_bon_digest(self, client_registre):
        client_registre.post("/api/pull", json={"model": "qwen3-test", "stream": False})
        entree = client_registre.get("/api/tags").json()["models"][0]
        assert entree["digest"] == MODELE_DIGEST.split(":", 1)[1]
        assert entree["size"] == len(MODELE)

    def test_modele_multimodal_avec_projecteur(self, client_registre):
        client_registre.post("/api/pull", json={"model": "vision", "stream": False})
        body = client_registre.post("/api/show", json={"model": "vision"}).json()
        assert "vision" in body["capabilities"]
        assert body.get("projector_info")

    def test_checksum_non_conforme_refuse(self, client_registre):
        """Un artefact substitué ne doit jamais être installé (risque R8)."""
        RegistreHandler.corrompre = True
        response = client_registre.post("/api/pull", json={"model": "qwen3-test",
                                                           "stream": False})
        assert response.status_code == 400
        assert "checksum" in response.json()["error"]
        assert client_registre.get("/api/tags").json()["models"] == []

    def test_artefact_sans_checksum_refuse(self, client_registre):
        """Sans checksum, rien ne distingue un artefact légitime d'un artefact substitué."""
        response = client_registre.post("/api/pull", json={"model": "sans-checksum",
                                                           "stream": False})
        assert response.status_code == 400
        assert "checksum" in response.json()["error"]

    def test_modele_absent(self, client_registre):
        response = client_registre.post("/api/pull", json={"model": "absent", "stream": False})
        assert response.status_code == 400
        assert "not found" in response.json()["error"]

    def test_identifiants_refuses(self, client_registre):
        response = client_registre.post("/api/pull", json={"model": "interdit", "stream": False})
        assert response.status_code == 400
        assert "credentials" in response.json()["error"]


class TestSecuriteDuRegistre:
    """Risque R9 : le jeton doit être transmis à l'amont et n'apparaître nulle part ailleurs."""

    def test_jeton_transmis_en_authorization(self, client_registre):
        client_registre.post("/api/pull", json={"model": "qwen3-test", "stream": False})
        autorisations = [h.get("Authorization") for h in RegistreHandler.recus]
        assert "Bearer jeton-de-test-tres-secret" in autorisations

    def test_jeton_absent_des_journaux(self, client_registre, caplog):
        import logging

        with caplog.at_level(logging.DEBUG, logger="ollamacpp"):
            client_registre.post("/api/pull", json={"model": "qwen3-test", "stream": False})
        assert "jeton-de-test-tres-secret" not in caplog.text

    def test_jeton_absent_des_reponses_derreur(self, client_registre):
        """Un message d'erreur ne doit divulguer ni le jeton ni l'URL du registre."""
        reponse = client_registre.post("/api/pull", json={"model": "absent", "stream": False})
        assert "jeton-de-test-tres-secret" not in reponse.text

    def test_configuration_journalisee_masque_les_secrets(self, config, registre_url):
        import dataclasses

        configuree = dataclasses.replace(config, registry_token="ultra-secret")
        assert configuree.redacted()["registry_token"] == "<set>"
        assert "ultra-secret" not in repr(configuree.redacted())


class TestProgression:
    def test_flux_ndjson_de_progression(self, client_registre):
        """Format `api.ProgressResponse`, pour qu'`ollama pull` affiche une progression réelle."""
        response = client_registre.post("/api/pull", json={"model": "qwen3-test"})
        assert response.headers["content-type"].startswith("application/x-ndjson")

        evenements = ndjson(response)
        statuts = [e.get("status") for e in evenements]
        assert statuts[0] == "pulling manifest"
        assert statuts[-1] == "success"
        assert any(s and s.startswith("pulling model") for s in statuts)

    def test_progression_porte_total_et_completed(self, client_registre):
        evenements = ndjson(client_registre.post("/api/pull", json={"model": "qwen3-test"}))
        telechargements = [e for e in evenements if e.get("total")]
        assert telechargements
        assert telechargements[-1]["completed"] == telechargements[-1]["total"]

    def test_deduplication_dun_second_pull(self, client_registre):
        """Un artefact déjà présent n'est pas retéléchargé : le `pull` est alors quasi instantané."""
        client_registre.post("/api/pull", json={"model": "qwen3-test", "stream": False})
        avant = len(RegistreHandler.recus)
        client_registre.post("/api/pull", json={"model": "copie", "stream": False})
        apres = len(RegistreHandler.recus)
        # Un seul appel supplémentaire : la résolution du manifest. Pas de re-téléchargement.
        assert apres - avant == 1

    def test_erreur_en_cours_de_flux_signalee_dans_le_flux(self, client_registre):
        """Le code HTTP est déjà envoyé : Ollama émet alors un objet `{"error": ...}`."""
        RegistreHandler.corrompre = True
        evenements = ndjson(client_registre.post("/api/pull", json={"model": "qwen3-test"}))
        assert any("error" in e for e in evenements)


# --- Hugging Face ----------------------------------------------------------------------------------


class TestHuggingFace:
    def test_pull_depuis_un_depot(self, client_registre):
        response = client_registre.post("/api/pull", json={"model": "hf.co/org/depot",
                                                           "stream": False})
        assert response.status_code == 200
        noms = [m["name"] for m in client_registre.get("/api/tags").json()["models"]]
        assert "hf.co/org/depot:latest" in noms

    def test_projecteur_associe_automatiquement(self, client_registre):
        client_registre.post("/api/pull", json={"model": "hf.co/org/depot", "stream": False})
        body = client_registre.post("/api/show", json={"model": "hf.co/org/depot"}).json()
        assert "vision" in body["capabilities"]

    def test_selection_dun_fichier_precis(self, client_registre):
        response = client_registre.post("/api/pull", json={
            "model": "hf.co/org/depot:Q8_0.gguf", "stream": False,
        })
        assert response.status_code == 200

    def test_depot_absent(self, client_registre):
        response = client_registre.post("/api/pull", json={"model": "hf.co/org/absent",
                                                           "stream": False})
        assert response.status_code == 400
        assert "not found" in response.json()["error"]

    def test_depot_sans_gguf(self, client_registre):
        response = client_registre.post("/api/pull", json={"model": "hf.co/org/sans-gguf",
                                                           "stream": False})
        assert response.status_code == 400
        assert "GGUF" in response.json()["error"]

    def test_prefixe_huggingface_co_equivalent(self, client_registre):
        response = client_registre.post("/api/pull", json={"model": "huggingface.co/org/depot",
                                                           "stream": False})
        assert response.status_code == 200


# --- Validation défensive ---------------------------------------------------------------------------


class TestValidationDefensive:
    @pytest.mark.parametrize("nom", ["../../etc/passwd", "..", "a/b", "nom avec espaces", ""])
    def test_nom_dartefact_distant_valide(self, nom):
        """Risque R8 : aucun nom annoncé par une source distante ne doit circuler tel quel."""
        from ollamacpp.sources import PullError, _safe_filename

        with pytest.raises(PullError):
            _safe_filename(nom)

    @pytest.mark.parametrize("nom", ["modele.gguf", "mmproj-f16.gguf", "a_b-c.1.gguf"])
    def test_noms_legitimes_acceptes(self, nom):
        from ollamacpp.sources import _safe_filename

        assert _safe_filename(nom) == nom

    def test_url_relative_reste_dans_le_registre(self):
        """Une URL relative fournie par le registre ne doit pas pouvoir pointer ailleurs."""
        from ollamacpp.sources import _absolute_url

        assert _absolute_url("https://reg.test", "/blobs/x") == "https://reg.test/blobs/x"
        assert _absolute_url("https://reg.test", "blobs/x") == "https://reg.test/blobs/x"
