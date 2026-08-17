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

Les quatre façades, le registre, le cycle de vie, l'ordonnanceur et le téléchargement sont
implémentés et couverts par 640 tests, dont des tests d'API de bout en bout et une suite de
conformité rejouant la logique d'`ollama-gateway`.

La chaîne est vérifiée de bout en bout sur un **vrai `llama-server`** compilé depuis l'upstream,
chargeant un **vrai modèle GGUF** et générant de vrais tokens.

L'état réel, unité par unité, est tenu dans **`docs/BACKLOG.md`**, qui fait foi — une unité n'y
passe `[x]` qu'après validation complète de sa Definition of Done.

## Installation

### Avec Docker (recommandé)

L'image compile `llama-server` depuis l'upstream à une révision épinglée, puis installe le
service. Aucune source de `llama.cpp` n'est vendorée dans ce dépôt.

```bash
cp .env.example .env.prod        # puis compléter
./runProd
```

### Sans Docker

```bash
pip install -r requirements.txt
export OLLAMACPP_LLAMA_SERVER_BIN=/chemin/vers/llama-server
python -m ollamacpp
```

Le binaire `llama-server` se compile depuis les sources de `llama.cpp` :

```bash
cmake -S llama.cpp -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TOOLS=ON
cmake --build build --target llama-server -j"$(nproc)"
```

## Commandes principales

| Commande | Effet |
|---|---|
| `./runDev` | environnement de développement conteneurisé |
| `./runStaging` | environnement de staging (exige `.env.staging`) |
| `./runProd` | environnement de production (exige `.env.prod`) |
| `python -m ollamacpp` | lancement direct, sans conteneur |
| `python scripts/seed.py --verify` | installe un modèle de démonstration et vérifie l'inférence |
| `python scripts/make_test_model.py --output m.gguf` | génère un vrai GGUF minuscule, sans téléchargement |
| `pytest` | suite de tests (les tests marqués `e2e` sont ignorés sans binaire amont) |
| `pytest -m conformance` | conformité vis-à-vis d'`ollama-gateway` uniquement |
| `OLLAMACPP_TEST_LLAMA_SERVER=/chemin/llama-server pytest -m e2e` | bout en bout sur le vrai `llama-server` et un vrai modèle |
| `docker compose -f docker-compose.dev.yml down -v` | arrêt et réinitialisation des données locales |

Il n'y a **pas d'étape de build** pour le service lui-même : c'est du Python pur. Le seul artefact
compilé est `llama-server`, produit par l'image Docker ou fourni par l'exploitant.

## Variables d'environnement

Toutes les variables sont documentées dans **`.env.example`** — rôle, format, caractère
obligatoire, valeur d'exemple non sensible. Les principales :

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMACPP_PORT` | `11434` | port d'écoute, celui qu'attendent les clients Ollama |
| `OLLAMACPP_MODELS` | `~/.ollama.cpp/models` | répertoire des blobs et manifests |
| `OLLAMACPP_LLAMA_SERVER_BIN` | `llama-server` | binaire d'inférence |
| `OLLAMACPP_KEEP_ALIVE` | `5m` | résidence par défaut (nombre = secondes, négatif = illimité) |
| `OLLAMACPP_MAX_LOADED_MODELS` | `3` | nombre maximal de modèles résidents |
| `OLLAMACPP_MEMORY_LIMIT_BYTES` | `0` (auto) | budget mémoire de l'ordonnanceur |
| `OLLAMACPP_MANAGEMENT_ENABLED` | `true` | autorise `pull`/`create`/`copy`/`delete`/`blobs` |
| `OLLAMACPP_API_KEY` | vide | si défini, exige `Authorization: Bearer` |
| `OLLAMACPP_REGISTRY_URL` / `_TOKEN` | vide | registre privé |

Les variables marquées SECRET dans `.env.example` ne doivent jamais être committées. Leur valeur
est masquée dans les journaux, ce que vérifie un test dédié.

## Structure du dépôt

```
ollamacpp/
  config.py errors.py names.py durations.py   configuration, erreurs, nommage, keep_alive
  gguf.py observability.py backend.py         métadonnées GGUF, journal, pont llama-server
  sources.py service.py app.py                téléchargement, assemblage, application ASGI
  canonical/       représentation conversationnelle unique des quatre façades
  storage/         magasin de blobs adressé par contenu, manifests
  registry/        registre des modèles installés
  runtime/         superviseur, capacités, cycle de vie, ordonnanceur, mémoire
  api/             façades Ollama, OpenAI, Responses, Anthropic
tests/             unitaires, intégration, API, conformité, contrat llama-server
scripts/seed.py            données de démonstration, via les vraies API
scripts/make_test_model.py modèle GGUF de test, réellement chargeable
docs/
  ollama.cpp-architecture.md   document fondateur : audit, matrice, plan, risques
  DAT.md                       dossier d'architecture technique
  BACKLOG.md                   état réel du projet, unités OC-xxx (fait foi)
  JOURNAL.md                   décisions et investigations
  PROD_MIGRATIONS.md           contrat de déploiement
  MODEL_CONFIG.md              manifest et pilotage de llama-server, référence complète
  REGISTRY.md                  protocole du registre privé
  manual.md                    manuel d'exploitation
  DESIGN_SYSTEM.md             charte d'interface P2Enjoy
Dockerfile  docker-compose.{dev,staging,prod}.yml  runDev runStaging runProd
.env.example
CHANGELOG.md  README.md
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
- **Le modèle de test intégré produit du charabia.** `scripts/make_test_model.py` génère un vrai
  GGUF `llama` (~460 Kio) chargeable par `llama-server`, ce qui permet d'exécuter toute la suite
  de bout en bout sans télécharger de modèle. Ses poids sont aléatoires : la chaîne complète est
  vérifiée, mais pas la **qualité** des réponses d'un modèle entraîné.
- **`/api/pull` n'accepte pas un chemin de fichier local** : ce n'est pas un nom de modèle Ollama
  valide, et Ollama ne l'accepte pas davantage. Un GGUF local s'installe par
  `POST /api/blobs/<digest>` puis `POST /api/create`.
- **Pas de quantisation à la volée** dans `/api/create` : le GGUF doit être quantifié en amont.
  L'endpoint répond `501` avec un message explicite plutôt que d'ignorer le champ.

## Documentation

| Document | Contenu |
|---|---|
| `docs/ollama.cpp-architecture.md` | Audit des dépôts amont, matrice de compatibilité `ollama-gateway`, manques, interfaces, plan, risques |
| `docs/DAT.md` | Composants, flux, données, interfaces, sécurité, déploiement |
| `docs/BACKLOG.md` | État réel, unité par unité |
| `docs/MODEL_CONFIG.md` | Manifest, `runtime` et correspondance complète avec les drapeaux `llama-server` |
| `docs/REGISTRY.md` | Protocole du registre privé : résolution, artefacts, checksums, sécurité |
| `docs/manual.md` | Manuel d'exploitation |
| `docs/PROD_MIGRATIONS.md` | Contrat de déploiement |
| `docs/JOURNAL.md` | Décisions structurantes et leurs justifications |
| `CHANGELOG.md` | Changements non publiés et publiés |

## Licence

Voir `LICENSE`.
