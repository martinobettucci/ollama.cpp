# ollama.cpp

**Middleware de remplacement compatible Ollama, adossé à `llama-server`.**

`ollama.cpp` permet de remplacer Ollama par `llama-server` sans casser les clients, outils et
services qui parlent aujourd'hui à Ollama — tout en donnant accès aux capacités avancées de
`llama.cpp` qu'Ollama masque (types de cache K/V, Flash Attention, `tensor-split`, decoding
spéculatif, contexte et parallélisme par modèle).

```
Clients
    ↓
ollama-gateway        clés API, quotas, targets, routage, usage
    ↓
ollama.cpp            registry, manifests, lifecycle, scheduler, façades d'API
    ↓
llama-server          HTTP d'inférence, slots, batching, chat templates
    ↓
llama.cpp             inférence, GGUF, tokenizer, KV cache, GPU
```

## Objectif

Qu'une cible `Ollama` puisse être remplacée par une cible `ollama.cpp` **sans modification du
client**, et notamment sans aucune modification d'`ollama-gateway` :

```bash
OLLAMA_HOST=http://localhost:11434 ollama list
```

et simultanément, sur les mêmes modèles et le même runtime :

```
POST /api/chat              (Ollama natif)
POST /v1/chat/completions   (OpenAI)
POST /v1/responses          (OpenAI Responses)
POST /v1/messages           (Anthropic Messages)
```

## Ce que `ollama.cpp` n'est pas

- **Pas un moteur d'inférence.** L'inférence, le GGUF, la tokenisation, les chat templates, le KV
  cache, l'offload GPU, la Flash Attention, le multimodal, le decoding spéculatif, les slots, le
  batching et le sampling restent la responsabilité de `llama.cpp` / `llama-server`.
- **Pas un fork de `llama.cpp`.** Aucune ligne de `llama.cpp` n'est modifiée : `ollama.cpp` pilote
  le binaire `llama-server` par ligne de commande et par HTTP.
- **Pas une passerelle.** Clés API, quotas, targets, contrôle d'accès aux modèles, routage,
  endpoints VS Code, suivi d'usage et administration restent la responsabilité
  d'`ollama-gateway`.

## Ce que `ollama.cpp` ajoute

Registre de modèles, manifests, téléchargement et registre privé, cycle de vie des modèles
(chargement, déchargement, `keep_alive`), ordonnancement conscient de la mémoire, détection de
capacités, et les trois familles d'API ci-dessus au-dessus d'une représentation conversationnelle
canonique unique.

## Stack

| Élément | Choix |
|---|---|
| Langage | Python 3.11 |
| Framework HTTP | FastAPI / Starlette / uvicorn |
| Client amont | httpx |
| Tests | pytest, pytest-asyncio |
| Backend d'inférence | binaire `llama-server` (externe, non modifié) |
| Conteneurisation | Docker + Compose (dev / staging / prod) |

Les versions de dépendances sont alignées sur celles d'`ollama-gateway` pour éviter deux
écosystèmes divergents dans la même chaîne. Justification et compromis :
`docs/ollama.cpp-architecture.md` §5.2.

## Prérequis

- Python 3.11 ou plus récent
- Un binaire `llama-server` accessible (compilé depuis `llama.cpp` ou fourni par l'image Docker)
- Docker et Docker Compose pour les environnements conteneurisés

## État du projet

**En cours de construction.** L'audit des dépôts amont et l'architecture sont terminés et
documentés ; l'implémentation suit le plan du §7 de `docs/ollama.cpp-architecture.md`.

L'état réel, unité par unité, est tenu dans **`docs/BACKLOG.md`**, qui fait foi. Aucune
fonctionnalité n'est annoncée ici avant d'y être marquée `[x]`.

## Installation

_À compléter avec l'unité de backlog OC-010 (configuration) et OC-090 (conteneurisation)._

## Commandes principales

_À compléter avec les unités de backlog OC-090 et OC-091._

Les commandes cibles sont `./runDev`, `./runStaging`, `./runProd` pour les environnements
conteneurisés, `pytest` pour les tests, et `python -m ollamacpp` pour un lancement direct.

## Variables d'environnement

_À compléter avec l'unité de backlog OC-010._ Chaque variable sera documentée avec son rôle, son
format attendu, son caractère obligatoire ou facultatif et une valeur d'exemple non sensible.

## Structure du dépôt

```
docs/
  ollama.cpp-architecture.md   document fondateur : audit, matrice, plan, risques
  DAT.md                       dossier d'architecture technique
  BACKLOG.md                   état réel du projet, unités OC-xxx (fait foi)
  JOURNAL.md                   décisions et investigations
  DESIGN_SYSTEM.md             charte d'interface P2Enjoy
CHANGELOG.md
README.md
```

## Limites connues

- **Génération d'images non supportée.** `llama.cpp` ne génère pas d'images : les capacités
  `ollama-image` (modèles `x/…`) et `openai-image` (`/v1/images/generations`) d'`ollama-gateway`
  resteront non servies. Incompatibilité assumée et documentée.
- **`POST /api/push` non implémenté.** Publier vers un registre distant n'a pas de sens sans
  registre Ollama ; l'endpoint répondra un code et un message explicites.
- **Endpoints de compte cloud Ollama hors périmètre** (`/api/me`, `/api/signout`,
  `/api/experimental/*`), ainsi que les modèles distants fédérés (`remote_host`, `remote_model`).
- **Layout de stockage inspiré d'Ollama, pas identique.** Un répertoire `~/.ollama` existant n'est
  pas lu tel quel.
- **Pas de GPU dans l'environnement de développement de référence** : le comportement d'offload et
  l'occupation VRAM ne sont pas vérifiables localement. Suivi en risque R11.

## Documentation

| Document | Contenu |
|---|---|
| `docs/ollama.cpp-architecture.md` | Audit des dépôts amont, matrice de compatibilité `ollama-gateway`, manques, interfaces, plan, risques |
| `docs/DAT.md` | Composants, flux, données, interfaces, sécurité, déploiement |
| `docs/BACKLOG.md` | État réel, unité par unité |
| `docs/JOURNAL.md` | Décisions structurantes et leurs justifications |
| `CHANGELOG.md` | Changements non publiés et publiés |

## Licence

Voir `LICENSE`.
