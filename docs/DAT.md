# DAT — Dossier d'Architecture Technique de `ollama.cpp`

Document vivant. Il décrit ce qui **est**, pas ce qui est souhaité. Toute divergence entre ce
document et le code est un défaut à corriger dans le même chunk.

Le document fondateur — audit des dépôts amont, matrice de compatibilité, justification des choix
et risques — est `docs/ollama.cpp-architecture.md`. Le présent DAT est la vue opérationnelle :
composants, flux, données, interfaces, déploiement.

---

## 1. Composants

| Composant | Module | Responsabilité |
|---|---|---|
| Configuration | `ollamacpp/config.py` | Chargement centralisé depuis l'environnement, valeurs par défaut, validation |
| Erreurs | `ollamacpp/errors.py` | Schéma d'erreur unique compatible Ollama `{"error": "..."}` |
| Nommage | `ollamacpp/names.py` | Analyse et normalisation `[host/][namespace/]model[:tag]` |
| Durées | `ollamacpp/durations.py` | `keep_alive` (sémantique Ollama) et durées en nanosecondes |
| Canonique | `ollamacpp/canonical/` | Représentation conversationnelle unique des 4 façades |
| Stockage | `ollamacpp/storage/` | `BlobStore` adressé par contenu, lecture/écriture des manifests |
| Registre | `ollamacpp/registry/` | `ModelRegistry`, sources (fichiers, Hugging Face, registre privé) |
| Runtime | `ollamacpp/runtime/` | Superviseur `llama-server`, détection de capacités, lifecycle, scheduler |
| Façades | `ollamacpp/api/` | Ollama natif, OpenAI Chat/Completions/Responses, Anthropic Messages |
| Application | `ollamacpp/app.py` | Assemblage FastAPI, cycle de vie, gestionnaires d'erreurs |

## 2. Services et processus

| Processus | Rôle | Réseau |
|---|---|---|
| `ollama.cpp` (uvicorn) | Façades HTTP, registre, lifecycle, scheduler | écoute `OLLAMACPP_HOST:OLLAMACPP_PORT` (défaut `0.0.0.0:11434`) |
| `llama-server` (N instances) | Inférence, une instance par modèle logique résident | loopback uniquement, port éphémère alloué par le superviseur |

Les instances `llama-server` ne sont **jamais** exposées hors de la boucle locale. Elles sont des
enfants directs du processus `ollama.cpp`, terminées avec lui.

## 3. Flux principaux

### 3.1 Requête d'inférence

```
client → façade → CanonicalRequest
                → ModelRegistry.resolve(nom)
                → ModelLifecycleManager.ensure_ready(ref, keep_alive)   [single-flight]
                      → ModelScheduler.admit(...)  → éviction éventuelle
                      → LlamaServerSupervisor.spawn(...) si nécessaire
                → sérialisation backend → llama-server /v1/chat/completions
                → CanonicalResult
                → sérialiseur de la façade d'origine → client
```

En streaming, `CanonicalResult` est produit incrémentalement : chaque delta backend est converti
en delta canonique, puis sérialisé au format de la façade (NDJSON pour Ollama, SSE pour
OpenAI/Anthropic).

### 3.2 Chargement d'un modèle

```
ensure_ready(ref)
  état READY               → réutilisation immédiate, last_used mis à jour
  état LOADING             → attente sur le même future (single-flight)
  état UNLOADED            → admit() → spawn() → attente de /health → READY
  état NOT_PRESENT         → erreur 404 « model '<nom>' not found »
```

### 3.3 Téléchargement

```
/api/pull → résolution de la source (privée | Hugging Face | fichiers)
          → téléchargement vers tmp/<uuid>.partial
          → vérification du checksum
          → renommage atomique vers blobs/sha256-<hex>
          → écriture du manifest
          → flux de progression NDJSON, statut final « success »
```

## 4. Modèle de données

### 4.1 Arborescence

```
$OLLAMACPP_MODELS/
  blobs/sha256-<hex>                      artefacts adressés par contenu
  manifests/<host>/<namespace>/<model>/<tag>   manifest JSON
  tmp/<uuid>.partial                      téléchargements en cours
```

### 4.2 Manifest

Schéma versionné (`schema_version: 1`). Voir `docs/ollama.cpp-architecture.md` §5.5 pour le
schéma complet et la sémantique de `capabilities_override` (restriction seule, jamais élévation).

### 4.3 Précédence de configuration

```
override de requête  >  manifest  >  métadonnées GGUF  >  défauts ollama.cpp
```

## 5. Interfaces

### 5.1 Interfaces exposées

Trois familles, décrites en détail dans `docs/ollama.cpp-architecture.md` §2.1 et §3.2 :
API native Ollama (`/api/*`), API OpenAI (`/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/models`, `/v1/responses`), API Anthropic (`/v1/messages`,
`/v1/messages/count_tokens`).

### 5.2 Interfaces consommées

`llama-server`, surface HTTP publique uniquement :

| Endpoint | Usage |
|---|---|
| `GET /health` | attente de disponibilité après `spawn` |
| `GET /props` | contexte effectif, modalités, capacités de template, chat template, tokens spéciaux |
| `POST /v1/chat/completions` | inférence conversationnelle (toutes façades) |
| `POST /v1/embeddings` | embeddings |
| `POST /v1/rerank` | reranking |
| `POST /tokenize`, `/detokenize` | comptage de tokens, `context` de `/api/generate` |
| `GET /metrics` | observabilité |

Aucune dépendance à un endpoint non documenté ou privé de `llama-server`.

### 5.3 Interfaces internes

Signatures dans `docs/ollama.cpp-architecture.md` §6.

## 6. Authentification et autorisation

`ollama.cpp` est un composant **de confiance interne**, placé derrière `ollama-gateway` qui porte
les clés API, quotas et contrôles d'accès aux modèles (`docs/ollama.cpp-architecture.md` §0.2).

Il fournit néanmoins deux garde-fous, appliqués **côté serveur** :

| Mécanisme | Variable | Comportement |
|---|---|---|
| Jeton d'accès amont | `OLLAMACPP_API_KEY` | Si défini, tout appel doit porter `Authorization: Bearer <clé>`. Sinon l'en-tête est ignoré. |
| Verrouillage du plan de contrôle | `OLLAMACPP_MANAGEMENT_ENABLED` | Si `false`, `pull`/`create`/`copy`/`delete`/`blobs` répondent `403` avec le schéma d'erreur Ollama. |

Ces règles sont appliquées dans le middleware serveur, jamais côté client, et vérifiées par des
tests qui contournent toute interface.

## 7. Sécurité

- Aucun jeton de registre, aucun en-tête `Authorization`, aucune URL signée n'est journalisé.
- Aucun chemin issu d'un manifest distant n'est utilisé pour écrire : le `BlobStore` est adressé
  par contenu, les noms de fichiers sont dérivés du digest calculé localement.
- Les checksums sont vérifiés avant installation ; un artefact non conforme est détruit.
- Les instances `llama-server` n'écoutent que sur la boucle locale.

## 8. Stratégie de déploiement

| Environnement | Fichier | Particularités |
|---|---|---|
| Développement | `docker-compose.dev.yml` | autonome, modèle de test minuscule, aucun service payant |
| Staging | `docker-compose.staging.yml` | variables dédiées, données non partagées avec la production |
| Production | `docker-compose.prod.yml` | variables dédiées, limites mémoire explicites |

Commandes : `./runDev`, `./runStaging`, `./runProd`.

## 9. Reprise

`ollama.cpp` est sans état durable en mémoire : le registre est reconstruit depuis le système de
fichiers au démarrage. Un redémarrage ne perd que la résidence des modèles, qui est rechargée à la
demande. Les téléchargements interrompus laissent un fichier `tmp/*.partial` collecté au démarrage.

## 10. Choix techniques importants et compromis

Documentés en un seul endroit, avec leur justification :
`docs/ollama.cpp-architecture.md` §5.2 et §5.3.

## 11. Dépendances structurantes

| Dépendance | Version | Rôle | Justification |
|---|---|---|---|
| `fastapi` | 0.139.2 | Framework HTTP | Déjà retenu par `ollama-gateway`, même écosystème et mêmes outils de test |
| `starlette` | 1.3.1 | Socle ASGI, streaming | Dépendance directe (réponses en flux) |
| `uvicorn[standard]` | 0.51.0 | Serveur ASGI | Standard de l'écosystème |
| `httpx` | 0.28.1 | Client HTTP amont, streaming | Déjà retenu par `ollama-gateway` |
| `pytest`, `pytest-asyncio` | 9.1.1 / 1.4.0 | Tests | Alignement avec `ollama-gateway` |

Versions alignées sur celles de `ollama-gateway` pour éviter deux écosystèmes divergents dans la
même chaîne. Aucune dépendance d'inférence : `llama-server` est un binaire externe.

## 12. Commandes

Voir `README.md`. Les commandes de référence sont `./runDev`, `./runStaging`, `./runProd`,
`pytest`, et le lancement direct `python -m ollamacpp`.

## 13. Données de développement

L'environnement de développement doit pouvoir démontrer chaque fonctionnalité sans GPU ni service
payant. Le seed installe un modèle GGUF minuscule dans le `BlobStore` et son manifest, de sorte que
`/api/tags`, `/api/show`, `/api/ps`, `/api/chat` et les trois façades soient démontrables
localement.
