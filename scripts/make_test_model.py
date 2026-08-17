#!/usr/bin/env python3
"""Génère un modèle GGUF minuscule mais **réellement chargeable** par `llama.cpp`.

@spec docs/BACKLOG.md OC-085 « Bout en bout avec un vrai llama-server », OC-092 « Seed »
@spec docs/ollama.cpp-architecture.md §8 risque R11
@spec docs/DAT.md §13 « Données de développement »

Pourquoi ce script existe : vérifier `ollama.cpp` de bout en bout demande un vrai modèle, donc
normalement un téléchargement de plusieurs centaines de mégaoctets — impossible dans un
environnement sans accès au réseau de distribution, et lent partout ailleurs. Ce script produit à
la place une architecture `llama` complète et valide, d'environ un mégaoctet, aux poids
aléatoires mais déterministes.

Le modèle **génère du charabia** : ses poids n'ont jamais été entraînés. Ce n'est pas le sujet.
Ce qu'il permet de vérifier pour de vrai, c'est toute la chaîne : chargement par `llama-server`,
lecture de `/props`, tokenisation, rendu du chat template, génération de tokens, streaming,
conversion canonique et sérialisation Ollama. Le seul élément non représentatif est la *qualité*
du texte produit.

Le tokenizer est un SPM à repli d'octets : il sait donc encoder n'importe quel texte, ce qui rend
le comptage de tokens et la longueur de prompt réalistes.

Usage :

    python scripts/make_test_model.py --output modele-test.gguf [--llama-cpp /chemin/llama.cpp]

`gguf-py` est fourni par les sources de `llama.cpp` ; le chemin peut aussi être donné par
`LLAMA_CPP_SOURCE`. Aucune écriture n'est faite dans l'arbre `llama.cpp`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: Dimensions : le plus petit modèle `llama` qui reste structurellement crédible. Deux couches
#: suffisent à exercer le cache KV, l'attention et le réseau feed-forward.
N_EMBD = 64
N_LAYER = 2
N_HEAD = 4
N_HEAD_KV = 4
N_FF = 128
N_CTX = 512
HEAD_DIM = N_EMBD // N_HEAD
RMS_EPS = 1e-5

#: Graine fixe : le modèle produit est reproductible octet pour octet, condition pour que les
#: tests qui s'appuient dessus soient déterministes (CLAUDE.md §8).
SEED = 20260817

#: Chat template minimal mais réel : `/api/chat` et les façades OpenAI et Anthropic ont besoin
#: d'un template pour rendre une conversation. Il est écrit en Jinja, comme ceux des vrais
#: modèles, et gère système, utilisateur et assistant.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>\n{{ message['content'] }}<|end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


#: Octets que le modèle a le droit d'ÉMETTRE : tabulation, saut de ligne et ASCII imprimable.
#:
#: Le vocabulaire contient bien les **256** octets — SPM l'exige : `llama_vocab::byte_to_token`
#: cherche `<0xXX>` puis, à défaut, l'octet brut, avec un `.at()` qui lève. Or SPM remplace les
#: espaces par « ▁ » (U+2581), qui s'encode sur trois octets non-ASCII : un vocabulaire amputé
#: fait donc échouer la tokenisation du moindre prompt contenant une espace.
#:
#: La restriction porte sur la SORTIE, via la projection finale (cf. `build_model`). Un modèle
#: aux poids aléatoires tire ses tokens au hasard : s'il peut émettre n'importe quel octet, il
#: produit des suites qui ne forment pas de l'UTF-8 valide et `llama-server` rejette la réponse
#: avant même de la renvoyer. En annulant les poids de sortie des octets non-ASCII, toute sortie
#: atteignable est de l'UTF-8 valide, et la chaîne complète devient vérifiable.
EMITTABLE_BYTES = [0x09, 0x0A] + list(range(0x20, 0x7F))


def build_vocabulary() -> tuple[list[str], list[float], list[int]]:
    """Construit un vocabulaire SPM à repli d'octets, couvrant les 256 octets.

    Trois jetons de contrôle, puis un jeton par octet. Le repli d'octets permet d'encoder
    n'importe quel texte : sans lui, la tokenisation dépendrait d'un vocabulaire de mots et les
    longueurs de prompt ne seraient pas réalistes.
    """
    import gguf

    tokens: list[str] = ["<unk>", "<s>", "</s>"]
    scores: list[float] = [0.0, 0.0, 0.0]
    types: list[int] = [
        gguf.TokenType.UNKNOWN,
        gguf.TokenType.CONTROL,
        gguf.TokenType.CONTROL,
    ]

    for value in range(256):
        tokens.append(f"<0x{value:02X}>")
        scores.append(0.0)
        types.append(gguf.TokenType.BYTE)

    return tokens, scores, types


def build_model(output: Path) -> Path:
    """Écrit le GGUF complet : métadonnées, vocabulaire et tenseurs."""
    import gguf
    import numpy as np

    rng = np.random.default_rng(SEED)

    def weights(*shape: int) -> "np.ndarray":
        """Poids aléatoires de faible amplitude, pour que les logits restent finis."""
        return rng.normal(0.0, 0.02, size=shape).astype(np.float32)

    def norm(size: int) -> "np.ndarray":
        """Poids de normalisation RMS : proches de 1, comme dans un modèle entraîné."""
        return np.ones(size, dtype=np.float32)

    tokens, scores, types = build_vocabulary()
    n_vocab = len(tokens)

    writer = gguf.GGUFWriter(str(output), "llama")

    # --- Métadonnées d'architecture ---------------------------------------------------------
    writer.add_name("ollama.cpp test model")
    writer.add_description("Modèle de test minuscule, poids aléatoires, non entraîné")
    writer.add_context_length(N_CTX)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_layer_norm_rms_eps(RMS_EPS)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    # --- Tokenizer ---------------------------------------------------------------------------
    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    writer.add_chat_template(CHAT_TEMPLATE)

    # --- Tenseurs ------------------------------------------------------------------------------
    # Convention GGUF : `ne = {n_entrée, n_sortie}`, ce qui correspond à une forme numpy
    # (n_sortie, n_entrée). Se tromper d'ordre produit un modèle qui se charge puis calcule faux.
    writer.add_tensor("token_embd.weight", weights(n_vocab, N_EMBD))

    for layer in range(N_LAYER):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", norm(N_EMBD))
        writer.add_tensor(f"blk.{layer}.attn_q.weight", weights(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{layer}.attn_k.weight", weights(N_HEAD_KV * HEAD_DIM, N_EMBD))
        writer.add_tensor(f"blk.{layer}.attn_v.weight", weights(N_HEAD_KV * HEAD_DIM, N_EMBD))
        writer.add_tensor(f"blk.{layer}.attn_output.weight", weights(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{layer}.ffn_norm.weight", norm(N_EMBD))
        writer.add_tensor(f"blk.{layer}.ffn_gate.weight", weights(N_FF, N_EMBD))
        writer.add_tensor(f"blk.{layer}.ffn_up.weight", weights(N_FF, N_EMBD))
        writer.add_tensor(f"blk.{layer}.ffn_down.weight", weights(N_EMBD, N_FF))

    writer.add_tensor("output_norm.weight", norm(N_EMBD))

    # Projection finale contrainte : seuls les octets émettables reçoivent des poids, et ils sont
    # d'amplitude volontairement forte. Les autres lignes valent zéro, donc leur logit vaut
    # exactement zéro, tandis que les logits des tokens autorisés s'étalent largement de part et
    # d'autre. Le maximum est donc pris parmi les tokens autorisés, et toute sortie est de
    # l'UTF-8 valide. C'est ce qui rend ce modèle non entraîné réellement exploitable de bout en
    # bout — le test `test_sortie_utf8_valide` le vérifie plutôt que de le supposer.
    output_projection = np.zeros((n_vocab, N_EMBD), dtype=np.float32)
    emittable_ids = [3 + value for value in EMITTABLE_BYTES]  # 3 jetons de contrôle en tête
    output_projection[emittable_ids] = rng.normal(
        0.0, 0.5, size=(len(emittable_ids), N_EMBD)
    ).astype(np.float32)
    writer.add_tensor("output.weight", output_projection)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return output


def locate_gguf_py(explicit: str | None) -> Path:
    """Trouve `gguf-py` dans les sources de `llama.cpp`, sans jamais y écrire."""
    candidats = [
        explicit,
        os.environ.get("LLAMA_CPP_SOURCE"),
        "/home/user/llama.cpp",
        str(Path.home() / "llama.cpp"),
        "../llama.cpp",
    ]
    for candidat in candidats:
        if not candidat:
            continue
        chemin = Path(candidat) / "gguf-py"
        if (chemin / "gguf" / "__init__.py").is_file():
            return chemin
    raise SystemExit(
        "gguf-py introuvable : indiquer les sources de llama.cpp avec --llama-cpp "
        "ou la variable LLAMA_CPP_SOURCE"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="modele-test.gguf", type=Path)
    parser.add_argument("--llama-cpp", default=None,
                        help="racine des sources de llama.cpp (contenant gguf-py)")
    args = parser.parse_args()

    sys.path.insert(0, str(locate_gguf_py(args.llama_cpp)))

    try:
        import gguf  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"dépendance manquante : {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    chemin = build_model(args.output)
    taille = chemin.stat().st_size

    print(f"écrit : {chemin} ({taille / 1024:.0f} Kio)")
    print(f"  architecture : llama, {N_LAYER} couches, {N_EMBD} dimensions, contexte {N_CTX}")
    print(f"  vocabulaire  : 259 jetons, dont {len(EMITTABLE_BYTES)} octets émettables")
    print("  poids aléatoires : le texte produit est du charabia, c'est attendu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
