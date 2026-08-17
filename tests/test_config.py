"""Tests de la configuration centralisée.

@verifies docs/BACKLOG.md OC-010 « Configuration centralisée »
@verifies docs/ollama.cpp-architecture.md §8 risque R9 « Fuite de secrets »
@verifies docs/DAT.md §7 « Sécurité »
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ollamacpp.config import SECRET_FIELDS, Config, ConfigError

# Toutes les variables lues par `Config.from_env`, à neutraliser entre deux tests pour que
# l'environnement de la machine hôte n'influence jamais un résultat.
_ENV_VARS = [
    "OLLAMACPP_HOST",
    "OLLAMACPP_PORT",
    "OLLAMACPP_VERSION",
    "OLLAMACPP_LOG_LEVEL",
    "OLLAMACPP_MODELS",
    "OLLAMACPP_LLAMA_SERVER_BIN",
    "OLLAMACPP_LLAMA_SERVER_HOST",
    "OLLAMACPP_LLAMA_SERVER_PORT_MIN",
    "OLLAMACPP_LLAMA_SERVER_PORT_MAX",
    "OLLAMACPP_LOAD_TIMEOUT_S",
    "OLLAMACPP_REQUEST_TIMEOUT_S",
    "OLLAMACPP_KEEP_ALIVE",
    "OLLAMACPP_MAX_LOADED_MODELS",
    "OLLAMACPP_MEMORY_LIMIT_BYTES",
    "OLLAMACPP_MEMORY_SAFETY_MARGIN",
    "OLLAMACPP_DEFAULT_CONTEXT",
    "OLLAMACPP_DOWNLOAD_CONCURRENCY",
    "OLLAMACPP_REGISTRY_URL",
    "OLLAMACPP_REGISTRY_TOKEN",
    "OLLAMACPP_HF_ENDPOINT",
    "OLLAMACPP_HF_TOKEN",
    "OLLAMACPP_API_KEY",
    "OLLAMACPP_MANAGEMENT_ENABLED",
]


@pytest.fixture(autouse=True)
def env_propre(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestValeursParDefaut:
    def test_port_ollama_par_defaut(self):
        """11434 est le port qu'attendent les clients Ollama sans configuration."""
        assert Config.from_env().port == 11434

    def test_keep_alive_par_defaut_de_cinq_minutes(self):
        assert Config.from_env().default_keep_alive.seconds == 300.0

    def test_binaire_amont_par_defaut(self):
        assert Config.from_env().llama_server_bin == "llama-server"

    def test_gestion_du_catalogue_activee_par_defaut(self):
        assert Config.from_env().management_enabled is True

    def test_aucun_secret_par_defaut(self):
        config = Config.from_env()
        assert config.api_key == ""
        assert config.registry_token == ""
        assert config.hf_token == ""


class TestLectureDeLenvironnement:
    def test_surcharge_du_port(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_PORT", "8080")
        assert Config.from_env().port == 8080

    def test_repertoire_de_modeles_avec_tilde(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_MODELS", "~/modeles")
        assert Config.from_env().models_dir == Path.home() / "modeles"

    def test_keep_alive_au_format_duree(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_KEEP_ALIVE", "15m")
        assert Config.from_env().default_keep_alive.seconds == 900.0

    def test_keep_alive_illimite(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_KEEP_ALIVE", "-1s")
        assert Config.from_env().default_keep_alive.is_infinite

    @pytest.mark.parametrize("valeur, attendu", [("true", True), ("false", False), ("0", False),
                                                 ("1", True), ("yes", True), ("off", False)])
    def test_booleens(self, monkeypatch, valeur, attendu):
        monkeypatch.setenv("OLLAMACPP_MANAGEMENT_ENABLED", valeur)
        assert Config.from_env().management_enabled is attendu

    def test_variable_vide_traitee_comme_absente(self, monkeypatch):
        """Une variable définie à la chaîne vide dans un Compose ne doit pas écraser le défaut."""
        monkeypatch.setenv("OLLAMACPP_HOST", "")
        assert Config.from_env().host == "0.0.0.0"


class TestValidation:
    """Une configuration invalide doit empêcher le démarrage, pas dégrader silencieusement."""

    @pytest.mark.parametrize("valeur", ["abc", "12.5", ""])
    def test_entier_invalide(self, monkeypatch, valeur):
        if valeur == "":
            pytest.skip("la chaîne vide est traitée comme une absence")
        monkeypatch.setenv("OLLAMACPP_PORT", valeur)
        with pytest.raises(ConfigError):
            Config.from_env()

    def test_port_hors_borne(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_PORT", "0")
        with pytest.raises(ConfigError):
            Config.from_env()

    def test_plage_de_ports_amont_incoherente(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_LLAMA_SERVER_PORT_MIN", "19000")
        monkeypatch.setenv("OLLAMACPP_LLAMA_SERVER_PORT_MAX", "18000")
        with pytest.raises(ConfigError):
            Config.from_env()

    def test_marge_memoire_hors_borne(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_MEMORY_SAFETY_MARGIN", "1.0")
        with pytest.raises(ConfigError):
            Config.from_env()

    def test_booleen_invalide(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_MANAGEMENT_ENABLED", "peut-être")
        with pytest.raises(ConfigError):
            Config.from_env()

    def test_keep_alive_invalide(self, monkeypatch):
        from ollamacpp.durations import DurationError

        monkeypatch.setenv("OLLAMACPP_KEEP_ALIVE", "toujours")
        with pytest.raises(DurationError):
            Config.from_env()


class TestSecrets:
    """Risque R9 : aucun jeton ne doit pouvoir atteindre les journaux."""

    def test_secrets_masques_dans_la_vue_journalisable(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_REGISTRY_TOKEN", "tok_ultra_secret")
        monkeypatch.setenv("OLLAMACPP_API_KEY", "cle_api_secrete")
        monkeypatch.setenv("OLLAMACPP_HF_TOKEN", "hf_secret")

        redacted = Config.from_env().redacted()
        serialise = repr(redacted)

        for secret in ("tok_ultra_secret", "cle_api_secrete", "hf_secret"):
            assert secret not in serialise

    def test_masque_distingue_defini_et_absent(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_REGISTRY_TOKEN", "tok")
        redacted = Config.from_env().redacted()
        assert redacted["registry_token"] == "<set>"
        assert redacted["api_key"] == "<unset>"

    def test_masque_ne_revele_pas_la_longueur(self, monkeypatch):
        monkeypatch.setenv("OLLAMACPP_API_KEY", "x" * 64)
        assert Config.from_env().redacted()["api_key"] == "<set>"

    def test_tous_les_champs_secrets_sont_couverts(self):
        """Garde-fou : un nouveau champ secret doit être ajouté à `SECRET_FIELDS`."""
        redacted = Config.from_env().redacted()
        for name in SECRET_FIELDS:
            assert redacted[name] in {"<set>", "<unset>"}

    def test_vue_journalisable_est_serialisable(self):
        """Les journaux structurés sérialisent la configuration : aucun objet opaque ne doit rester."""
        import json

        json.dumps(Config.from_env().redacted())


class TestArborescence:
    def test_chemins_derives(self, tmp_path):
        config = Config(models_dir=tmp_path)
        assert config.blobs_dir == tmp_path / "blobs"
        assert config.manifests_dir == tmp_path / "manifests"
        assert config.tmp_dir == tmp_path / "tmp"

    def test_creation_idempotente(self, tmp_path):
        config = Config(models_dir=tmp_path / "modeles")
        config.ensure_layout()
        config.ensure_layout()
        assert config.blobs_dir.is_dir()
        assert config.manifests_dir.is_dir()
        assert config.tmp_dir.is_dir()


class TestImmutabilite:
    def test_configuration_non_modifiable(self):
        """La configuration est figée : aucun composant ne peut la muter en cours d'exécution."""
        config = Config.from_env()
        with pytest.raises((AttributeError, TypeError)):
            config.port = 9999  # type: ignore[misc]
