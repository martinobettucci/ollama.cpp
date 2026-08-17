"""Construction de la ligne de commande d'une instance `llama-server`.

@spec docs/BACKLOG.md OC-031 « Construction des arguments runtime »
@spec docs/ollama.cpp-architecture.md §1.5 « Arguments CLI pertinents », §5.2 « Choix
      technologiques », §5.6 « Ordre de précédence de la configuration »
@spec docs/DAT.md §2 « Services et processus »

C'est ici que se matérialise l'objectif n° 1 du projet : **ne pas masquer les capacités de
`llama.cpp`**. Chaque drapeau émis a été vérifié dans `common/arg.cpp` à la révision auditée
`39be55c` ; le tableau du §1.5 de l'architecture en donne la liste et la source.

Deux garde-fous.

**Les drapeaux réservés au superviseur ne sont jamais délégables.** `--host`, `--port`, `--model`
et `--alias` sont posés par `ollama.cpp` ; un manifest qui tenterait de les redéfinir via
`extra_args` serait ignoré, sinon une instance pourrait s'exposer hors de la boucle locale ou
charger un autre modèle que celui demandé.

**Le contexte historique de `llama-server` reste intact.** `ollama.cpp` n'ajoute que des
arguments : `llama-server -m modele.gguf` continue de fonctionner tel quel (mission §32).
"""

from __future__ import annotations

from pathlib import Path

from ..storage.manifests import RuntimeConfig

#: Drapeaux posés par le superviseur, qu'un manifest ne peut pas redéfinir. Le préfixe court est
#: inclus quand `llama-server` en accepte un, pour qu'aucune forme ne passe au travers.
RESERVED_FLAGS = frozenset(
    {"--host", "--port", "--model", "-m", "--alias", "-a", "--api-key", "--api-key-file",
     "--models-dir", "--models-max", "--mmproj", "--path"}
)


def _flag(args: list[str], name: str, value: object | None) -> None:
    if value is not None:
        args.extend([name, str(value)])


def filter_extra_args(extra: tuple[str, ...]) -> list[str]:
    """Retire les drapeaux réservés d'une liste d'arguments supplémentaires.

    Le filtrage retire le drapeau **et** sa valeur éventuelle : laisser passer une valeur
    orpheline décalerait toute la ligne de commande et produirait une erreur incompréhensible.
    """
    out: list[str] = []
    skip_next = False
    for token in extra:
        if skip_next:
            skip_next = False
            continue
        base = token.split("=", 1)[0]
        if base in RESERVED_FLAGS:
            # `--flag=valeur` porte sa valeur ; `--flag valeur` la met dans le jeton suivant.
            skip_next = "=" not in token
            continue
        out.append(token)
    return out


def build_args(
    *,
    model_path: Path,
    alias: str,
    host: str,
    port: int,
    runtime: RuntimeConfig,
    mmproj_path: Path | None = None,
    draft_path: Path | None = None,
    adapter_paths: tuple[Path, ...] = (),
    default_context: int = 4096,
) -> list[str]:
    """Construit les arguments d'une instance `llama-server` pour un modèle logique.

    `runtime` est supposé **déjà fusionné** selon la précédence `requête > manifest > GGUF >
    défauts` : cette fonction ne décide de rien, elle traduit.
    """
    args: list[str] = [
        "--model", str(model_path),
        "--alias", alias,
        "--host", host,
        "--port", str(port),
        # L'interface web n'a aucun sens sur une instance interne, et la servir consommerait de
        # la mémoire et exposerait une surface inutile.
        "--no-webui",
        # Volontairement PAS de `--props` : `GET /props` — la source de vérité de la détection
        # de capacités (OC-032) — est servi sans condition (`server-context.cpp` l. 4566). Le
        # drapeau n'active que le `POST /props` MUTANT (l. 4620), dont l'instance n'a aucun
        # besoin et qui ne ferait qu'ouvrir une surface de modification.
        # Le rendu Jinja des templates est ce qui permet de ne pas réimplémenter les chat
        # templates (mission §2) et d'obtenir le parsing natif des appels d'outils.
        "--jinja",
    ]

    # --- Artefacts liés ---------------------------------------------------------------------
    _flag(args, "--mmproj", mmproj_path)
    _flag(args, "--model-draft", draft_path)
    for adapter in adapter_paths:
        args.extend(["--lora", str(adapter)])

    # --- Contexte et parallélisme -----------------------------------------------------------
    _flag(args, "--ctx-size", runtime.context if runtime.context is not None else default_context)
    _flag(args, "--batch-size", runtime.batch)
    _flag(args, "--ubatch-size", runtime.ubatch)
    _flag(args, "--parallel", runtime.parallel)

    # --- CPU ---------------------------------------------------------------------------------
    _flag(args, "--threads", runtime.threads)
    _flag(args, "--threads-batch", runtime.threads_batch)
    _flag(args, "--numa", runtime.numa)

    # --- GPU ---------------------------------------------------------------------------------
    _flag(args, "--n-gpu-layers", runtime.gpu_layers)
    _flag(args, "--tensor-split", runtime.tensor_split)
    _flag(args, "--main-gpu", runtime.main_gpu)

    # --- Attention et cache KV ---------------------------------------------------------------
    # Le cœur de la valeur ajoutée : Ollama n'expose pas ces réglages, `llama.cpp` si.
    if runtime.flash_attention is not None:
        args.extend(["--flash-attn", "on" if runtime.flash_attention else "off"])
    _flag(args, "--cache-type-k", runtime.cache_type_k)
    _flag(args, "--cache-type-v", runtime.cache_type_v)
    if runtime.kv_unified is True:
        args.append("--kv-unified")
    if runtime.kv_offload is False:
        args.append("--no-kv-offload")

    # --- Mémoire -----------------------------------------------------------------------------
    if runtime.mmap is False:
        args.append("--no-mmap")
    if runtime.mlock is True:
        args.append("--mlock")

    # --- Decoding spéculatif -----------------------------------------------------------------
    _flag(args, "--draft-max", runtime.draft_max)
    _flag(args, "--draft-min", runtime.draft_min)
    _flag(args, "--draft-p-min", runtime.draft_p_min)

    # --- Modes ------------------------------------------------------------------------------
    if runtime.embedding is True:
        args.append("--embedding")
    if runtime.reranking is True:
        args.append("--reranking")
    _flag(args, "--pooling", runtime.pooling)

    # --- Templates et raisonnement -----------------------------------------------------------
    _flag(args, "--chat-template", runtime.chat_template)
    _flag(args, "--reasoning-format", runtime.reasoning_format)
    _flag(args, "--reasoning-budget", runtime.reasoning_budget)

    # --- Soupape ------------------------------------------------------------------------------
    args.extend(filter_extra_args(runtime.extra_args))

    return args


def describe_args(args: list[str]) -> str:
    """Rend une ligne de commande lisible pour les journaux.

    Aucun secret ne transite par ces arguments — les drapeaux `--api-key` sont réservés et
    filtrés — mais la fonction reste le point unique où l'on pourrait masquer si cela changeait.
    """
    return " ".join(args)
