"""Tests de la représentation conversationnelle canonique.

@verifies docs/BACKLOG.md OC-014 « Représentation conversationnelle canonique »
@verifies docs/ollama.cpp-architecture.md §5.4 « Modèle conversationnel canonique », risque R7

Les quatre invariants du §5.4 sont testés ici au niveau des types eux-mêmes. Leur vérification
de bout en bout, sur les quatre façades, relève d'OC-082 et OC-083.
"""

from __future__ import annotations

import base64

import pytest

from ollamacpp.canonical import (
    AssistantMessage,
    CanonicalResult,
    FinishReason,
    ImageInput,
    ReasoningBlock,
    SamplingOptions,
    SystemMessage,
    TextBlock,
    Timings,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ollamacpp.errors import BadRequest

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class TestInvariantSeparationDuRaisonnement:
    """Invariant n° 3 : un bloc de raisonnement n'est jamais fusionné dans le texte visible."""

    def test_message_assistant_separe_texte_et_raisonnement(self):
        message = AssistantMessage(
            content=(
                ReasoningBlock(text="l'utilisateur veut la météo"),
                TextBlock(text="Il fait beau."),
            )
        )
        assert message.text == "Il fait beau."
        assert message.reasoning == "l'utilisateur veut la météo"

    def test_resultat_separe_texte_et_raisonnement(self):
        result = CanonicalResult(
            model="m",
            content=(ReasoningBlock(text="analyse"), TextBlock(text="réponse")),
        )
        assert result.text == "réponse"
        assert result.reasoning == "analyse"

    def test_raisonnement_seul_ne_produit_pas_de_texte(self):
        assert AssistantMessage(content=(ReasoningBlock(text="secret"),)).text == ""


class TestInvariantResultatDoutil:
    """Invariant n° 2 : un résultat d'outil n'est jamais un message utilisateur."""

    def test_type_distinct_de_user_message(self):
        resultat = ToolResultMessage(call_id="call_1", name="meteo", content="18°C")
        assert not isinstance(resultat, UserMessage)
        assert resultat.role == "tool"

    def test_role_non_modifiable(self):
        """Le rôle est figé à la construction : aucune façade ne peut le réécrire."""
        resultat = ToolResultMessage(call_id="call_1")
        with pytest.raises((AttributeError, TypeError)):
            resultat.role = "user"  # type: ignore[misc]

    def test_identifiant_dappel_obligatoire_pour_la_correlation(self):
        assert ToolResultMessage(call_id="call_abc").call_id == "call_abc"


class TestInvariantIdentifiantsDappels:
    """Invariant n° 1 : `ToolCall.id` traverse les conversions inchangé."""

    @pytest.mark.parametrize(
        "identifiant",
        ["call_abc123", "toolu_01A09q90qw90lq917835lq9", "fc_68a9", "0", "a" * 128],
    )
    def test_identifiants_de_formats_varies_preserves(self, identifiant: str):
        """Les quatre façades ont des formats d'identifiants différents ; aucun n'est normalisé."""
        assert ToolCall(id=identifiant, name="f").id == identifiant

    def test_correlation_appel_resultat(self):
        appel = ToolCall(id="call_1", name="meteo", arguments={"ville": "Paris"})
        resultat = ToolResultMessage(call_id=appel.id, name=appel.name, content="18°C")
        assert resultat.call_id == appel.id


class TestInvariantOrdreDesArguments:
    """Invariant n° 4 : l'ordre des clés d'arguments est préservé."""

    def test_ordre_dinsertion_conserve(self):
        appel = ToolCall(id="c", name="f", arguments={"z": 1, "a": 2, "m": 3})
        assert list(appel.arguments.keys()) == ["z", "a", "m"]

    def test_ordre_conserve_apres_serialisation_json(self):
        import json

        appel = ToolCall(id="c", name="f", arguments={"ville": "Paris", "unite": "C"})
        assert json.dumps(appel.arguments) == '{"ville": "Paris", "unite": "C"}'


class TestImageInput:
    """Les trois façades expriment les images différemment ; la normalisation est unique."""

    def test_base64_nu_style_ollama(self):
        image = ImageInput.from_base64(base64.b64encode(PNG_1PX).decode())
        assert image.data == PNG_1PX
        assert image.media_type == "image/png"

    def test_data_uri_style_openai(self):
        uri = f"data:image/png;base64,{base64.b64encode(PNG_1PX).decode()}"
        assert ImageInput.from_base64(uri).data == PNG_1PX

    def test_type_declare_prioritaire_sur_la_detection(self):
        encoded = base64.b64encode(PNG_1PX).decode()
        assert ImageInput.from_base64(encoded, media_type="image/webp").media_type == "image/webp"

    def test_detection_jpeg(self):
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 16
        assert ImageInput.from_bytes(jpeg).media_type == "image/jpeg"

    def test_rembourrage_manquant_tolere(self):
        """Certains clients omettent le « = » final ; le refuser casserait des images valides."""
        encoded = base64.b64encode(PNG_1PX).decode().rstrip("=")
        assert ImageInput.from_base64(encoded).data == PNG_1PX

    @pytest.mark.parametrize("invalide", ["", "!!!!", "data:image/png;base64,!!!"])
    def test_image_illisible_leve_une_erreur_de_requete(self, invalide: str):
        """Une image illisible doit produire un 400, pas une image vide passée au modèle."""
        with pytest.raises(BadRequest):
            ImageInput.from_base64(invalide)

    def test_aller_retour_data_uri(self):
        image = ImageInput.from_bytes(PNG_1PX)
        assert ImageInput.from_base64(image.to_data_uri()).data == PNG_1PX


class TestUserMessage:
    def test_texte_et_images_separes(self):
        message = UserMessage(
            content=(TextBlock(text="Décris "), ImageInput.from_bytes(PNG_1PX), TextBlock(text="ceci."))
        )
        assert message.text == "Décris ceci."
        assert len(message.images) == 1


class TestSamplingOptions:
    def test_fusion_priorise_la_surcharge(self):
        """Règle de précédence `requête > manifest` (architecture §5.6)."""
        socle = SamplingOptions(num_ctx=4096, temperature=0.7)
        surcharge = SamplingOptions(temperature=0.2)
        fusion = socle.merged_with(surcharge)
        assert fusion.num_ctx == 4096
        assert fusion.temperature == 0.2

    def test_valeur_absente_ne_efface_pas_le_socle(self):
        fusion = SamplingOptions(num_ctx=8192).merged_with(SamplingOptions())
        assert fusion.num_ctx == 8192

    def test_zero_explicite_surcharge_bien(self):
        """`0` est une valeur, pas une absence : `temperature=0` doit être respectée."""
        fusion = SamplingOptions(temperature=0.7).merged_with(SamplingOptions(temperature=0.0))
        assert fusion.temperature == 0.0

    def test_extra_fusionne(self):
        fusion = SamplingOptions(extra={"a": 1}).merged_with(SamplingOptions(extra={"b": 2}))
        assert fusion.extra == {"a": 1, "b": 2}

    def test_options_inconnues_conservees(self):
        """Une option ajoutée par une version ultérieure d'Ollama traverse sans casser."""
        assert SamplingOptions(extra={"future_option": 42}).extra["future_option"] == 42


class TestFinishReason:
    @pytest.mark.parametrize(
        "canonique, ollama, openai, anthropic",
        [
            (FinishReason.STOP, "stop", "stop", "end_turn"),
            (FinishReason.LENGTH, "length", "length", "max_tokens"),
            (FinishReason.TOOL_CALLS, "stop", "tool_calls", "tool_use"),
            (FinishReason.STOP_SEQUENCE, "stop", "stop", "stop_sequence"),
        ],
    )
    def test_correspondance_des_vocabulaires(self, canonique, ollama, openai, anthropic):
        """Chaque façade a son vocabulaire ; la correspondance est faite une seule fois."""
        assert canonique.to_ollama() == ollama
        assert canonique.to_openai() == openai
        assert canonique.to_anthropic() == anthropic

    def test_ollama_ne_connait_pas_tool_calls(self):
        """Ollama signale un appel d'outil par `message.tool_calls`, pas par `done_reason`."""
        assert FinishReason.TOOL_CALLS.to_ollama() == "stop"


class TestMetriquesOllama:
    def test_conversion_en_nanosecondes(self):
        timings = Timings(total_s=1.5, load_s=0.5, prompt_eval_s=0.25, eval_s=0.75)
        metrics = timings.to_ollama_metrics(Usage(prompt_eval_count=26, eval_count=298))
        assert metrics["total_duration"] == 1_500_000_000
        assert metrics["load_duration"] == 500_000_000
        assert metrics["prompt_eval_duration"] == 250_000_000
        assert metrics["eval_duration"] == 750_000_000

    def test_compteurs_de_tokens_repris(self):
        metrics = Timings(total_s=1.0).to_ollama_metrics(Usage(prompt_eval_count=26, eval_count=298))
        assert metrics["prompt_eval_count"] == 26
        assert metrics["eval_count"] == 298

    def test_champs_nuls_omis(self):
        """Les champs de `Metrics` sont `omitempty` côté Ollama : zéro n'est pas émis."""
        metrics = Timings(total_s=1.0).to_ollama_metrics(Usage())
        assert "eval_count" not in metrics
        assert "load_duration" not in metrics
        assert metrics["total_duration"] == 1_000_000_000

    def test_tous_les_champs_sont_entiers(self):
        metrics = Timings(1.5, 0.5, 0.25, 0.75).to_ollama_metrics(Usage(1, 2))
        assert all(isinstance(value, int) for value in metrics.values())


class TestUsage:
    def test_total(self):
        assert Usage(prompt_eval_count=10, eval_count=32).total_tokens == 42


class TestMessagesSysteme:
    def test_role_fige(self):
        assert SystemMessage(text="tu es utile").role == "system"
