# Architecture de `ollama.cpp`

> Document d'architecture fondateur, exigé avant toute implémentation.
> Statut : **architecture arrêtée et implémentée**. Le §7 donne l'état d'avancement réel étape
> par étape ; `docs/BACKLOG.md` fait foi unité par unité. Les sections « Plan d'implémentation »
> et « Risques » sont vivantes et mises à jour au fil des chunks.
>
> Toutes les affirmations sur `llama.cpp`, `ollama` et `ollama-gateway` de ce document ont été
> vérifiées par lecture directe des dépôts aux révisions indiquées au §0. Ce sont des **faits
> observés**, pas des souvenirs de modèle.

---

## 0. Périmètre, sources et méthode

### 0.1 Objectif

`ollama.cpp` est un **middleware de remplacement compatible Ollama**, adossé à `llama-server`.
Il doit permettre de remplacer une cible Ollama par une cible `ollama.cpp` sans modifier les
clients, et idéalement sans modifier `ollama-gateway`.

```
Clients
    ↓
ollama-gateway          (clés API, quotas, targets, routage, usage — déjà existant)
    ↓
ollama.cpp              (registry, manifests, lifecycle, scheduler, façades d'API)
    ↓
llama-server            (HTTP d'inférence, slots, batching, templates)
    ↓
llama.cpp               (inférence, GGUF, tokenizer, KV cache, GPU)
```

### 0.2 Ce que `ollama.cpp` n'est pas

- **Ce n'est pas un moteur d'inférence.** L'inférence, le GGUF, la tokenisation, les chat
  templates, le KV cache, l'offload GPU, la Flash Attention, le multimodal, le decoding
  spéculatif, les slots, le batching et le sampling restent **intégralement** la responsabilité de
  `llama.cpp` / `llama-server`.
- **Ce n'est pas un fork de `llama.cpp`.** Aucune ligne de `llama.cpp` n'est modifiée.
  `ollama.cpp` pilote le binaire `llama-server` par processus et par HTTP.
- **Ce n'est pas une passerelle.** Les clés API, quotas, targets, contrôle d'accès aux modèles,
  routage, endpoints VS Code, suivi d'usage et administration sont la responsabilité de
  `ollama-gateway` et ne sont pas dupliqués ici.

### 0.3 Sources auditées

| Dépôt | Révision auditée | Rôle |
|---|---|---|
| `llama.cpp` | `39be55c` (`vendor: move hash to vendor (#27262)`) | Backend d'inférence, référence du routeur multi-modèles |
| `ollama` | `d67ad83` (`mlx update (#17761)`) | **Spécification comportementale** de l'API à reproduire |
| `ollama-gateway` | `e26fe13` | **Consommateur réel** dont le contrat doit être satisfait |

Les trois dépôts sont montés en lecture seule pendant les travaux. Aucune modification n'y est
apportée.

### 0.4 Méthode

Ollama est traité comme une **spécification comportementale**, pas comme une base de code à
recopier. La compatibilité visée est comportementale : mêmes chemins, mêmes méthodes, mêmes champs,
mêmes types JSON, mêmes codes HTTP, même format de flux, même schéma d'erreur.

---

## 1. Architecture actuelle de `llama-server`

### 1.1 Surface HTTP réellement exposée

Relevé exhaustif depuis `tools/server/server.cpp` (enregistrement des routes, l. 226-359) :

**Inférence et compatibilité**

| Méthode | Chemin | Note |
|---|---|---|
| POST | `/completion`, `/completions` | natif llama.cpp |
| POST | `/v1/completions` | OpenAI legacy |
| POST | `/chat/completions`, `/v1/chat/completions` | **OpenAI Chat Completions** |
| POST | `/v1/chat/completions/control` | contrôle de génération |
| POST | `/responses`, `/v1/responses` | **OpenAI Responses** |
| POST | `/v1/messages` | **Anthropic Messages** |
| POST | `/v1/messages/count_tokens` | comptage de tokens Anthropic |
| POST | `/v1/audio/transcriptions` | transcription |
| POST | `/infill` | FIM |
| POST | `/embedding`, `/embeddings`, `/v1/embeddings` | embeddings |
| POST | `/rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking` | reranking |
| POST | `/tokenize`, `/detokenize`, `/apply-template` | utilitaires |
| POST | `/chat/completions/input_tokens`, `/v1/responses/input_tokens` | comptage |

**Introspection et contrôle**

| Méthode | Chemin | Note |
|---|---|---|
| GET | `/health`, `/v1/health` | public, sans clé API |
| GET | `/metrics` | métriques Prometheus |
| GET / POST | `/props` | **propriétés du modèle chargé** |
| GET | `/models`, `/v1/models` | listing |
| GET / POST | `/lora-adapters` | adaptateurs LoRA |
| GET | `/slots`, POST `/slots/:id_slot` | slots |
| GET / POST / DELETE | `/v1/stream`, `/v1/streams/lookup` | streaming reprenable |

**Constat majeur n° 1 : `llama-server` implémente déjà nativement les trois familles d'API
demandées** — OpenAI Chat Completions, OpenAI Responses et Anthropic Messages. Il ne manque que
**l'API native Ollama** et tout le plan de contrôle des modèles.

### 1.2 Le mode routeur de `llama-server`

`tools/server/server-models.{h,cpp}` (2 507 lignes) implémente déjà un **routeur multi-modèles**.
C'est la brique la plus proche de ce que `ollama.cpp` doit faire, et elle mérite d'être comprise
avant d'écrire quoi que ce soit.

Routes du routeur (`server.cpp` l. 226-230) :

```
POST   /models          création / enregistrement
POST   /models/load     chargement
POST   /models/unload   déchargement
GET    /models/sse      flux d'événements de cycle de vie
DELETE /models          suppression
```

Machine à états documentée dans `server-models.h` l. 19-38 :

```
DOWNLOADING ──► DOWNLOADED ──► (remplacé par une nouvelle instance)

UNLOADED ──► LOADING ──► LOADED ◄──── SLEEPING
 ▲            │            │               ▲
 └───failed───┘            │               │
 ▲                         └──sleeping─────┘
 └────────unloaded─────────┘
```

Structure `server_model_meta` (l. 72-106) : `source` (preset / models_dir / cache), `preset`,
`name`, `aliases`, `tags`, `port`, `status`, `last_used` (LRU), `args` (arguments passés à
l'instance), `loaded_info`, `progress`, `exit_code`, `multimodal` (`mtmd_caps`).

Mécanique (`server_models`, l. 112-294) :

- **un processus fils `llama-server` par modèle**, sur un port propre, supervisé par un thread ;
- `proxy_request()` relaie les requêtes HTTP vers l'instance fille ;
- `ensure_model_ready()` charge à la demande et **attend** — c'est déjà un single-flight ;
- `server_lru_sched` évince en LRU quand `--models-max` est atteint ;
- notifications SSE de changement d'état ;
- `handle_child_state()` reçoit les transitions d'état remontées par le fils.

**Constat majeur n° 2 : le pattern « un processus `llama-server` par modèle logique, piloté par un
superviseur qui proxifie » est le pattern retenu en amont par le projet `llama.cpp` lui-même.**
`ollama.cpp` reprend ce pattern plutôt que d'en inventer un autre.

### 1.3 `/props` : la source de vérité runtime

`server-context.cpp` l. 4576-4611. Champs exposés :

| Champ | Usage pour `ollama.cpp` |
|---|---|
| `default_generation_settings.n_ctx` | contexte effectif → `/api/show`, `/api/ps.context_length` |
| `model_path` | chemin du GGUF chargé |
| `model_alias`, `model_ftype` | identité et quantisation |
| `modalities` : `{vision, video, audio}` | **détection de capacité vision/audio observable** |
| `chat_template` | template effectif → `/api/show.template` |
| `chat_template_caps` | **détection de capacité tools / thinking observable** |
| `bos_token`, `eos_token` | métadonnées de tokenizer |
| `build_info` | version du backend |
| `total_slots` | parallélisme effectif |
| `is_sleeping` | état de veille |

`chat_template_caps` provient de `common_chat_templates_get_caps()` (`common/chat.cpp` l. 3900) et
sérialise `jinja::caps` (`common/jinja/caps.h`) :

```
supports_tools, supports_tool_calls, supports_system_role,
supports_parallel_tool_calls, supports_preserve_reasoning,
supports_reasoning_effort, supports_string_content,
supports_typed_content, supports_object_arguments
```

**Constat majeur n° 3 : `ollama.cpp` peut dériver les capacités de faits observables plutôt que de
les déclarer.** `vision` vient de `modalities.vision`, `tools` de `chat_template_caps.supports_tools`,
`thinking` de `supports_reasoning_effort` / `supports_preserve_reasoning`. C'est exactement
l'exigence « ne pas annoncer `vision=true` si le modèle ne peut pas traiter une image ».

### 1.4 `/v1/models` de `llama-server`

`server-context.cpp` l. 4883-4906. La réponse contient **déjà** une double forme : une clé
`models` de style Ollama et une clé `data` de style OpenAI. Mais les valeurs Ollama sont des
bouchons : `modified_at: ""`, `size: ""`, `digest: ""` (commentaire explicite : *« dummy value,
llama.cpp does not support managing model file's hash »*).

Cette réponse **n'est pas exploitable** par `ollama-gateway`, qui attend `size: int` et un vrai
`digest`. C'est précisément le trou que le `ModelRegistry` de `ollama.cpp` comble.

### 1.5 Arguments CLI pertinents

Relevés dans `common/arg.cpp`. Ils constituent la surface de configuration runtime par modèle que
`ollama.cpp` doit pouvoir piloter — c'est **le cœur de la valeur ajoutée du projet** (§16 de la
mission : ne pas masquer les capacités de `llama.cpp` comme le fait Ollama).

| Domaine | Flags |
|---|---|
| Modèle | `--model`, `--alias`, `--mmproj`, `--model-draft` / `--spec-draft-model`, `--lora`, `--lora-scaled` |
| Contexte | `--ctx-size`, `--batch-size`, `--ubatch-size`, `--parallel`, `--keep` |
| Threads / CPU | `--threads`, `--threads-batch`, `--cpu-mask`, `--cpu-range`, `--numa`, `--prio` |
| GPU | `--n-gpu-layers`, `--tensor-split`, `--main-gpu`, `--device`, `--n-cpu-moe`, `--override-tensor` |
| Attention / cache | `--flash-attn`, `--cache-type-k`, `--cache-type-v`, `--cache-type-k-draft`, `--cache-type-v-draft`, `--kv-unified`, `--no-kv-offload`, `--cache-ram`, `--cache-reuse` |
| Spéculatif | `--draft-max`, `--draft-min`, `--draft-p-min`, `--spec-draft-ngl`, `--spec-draft-n-max` |
| Templates | `--jinja`, `--chat-template`, `--chat-template-file`, `--chat-template-kwargs` |
| Raisonnement | `--reasoning-format`, `--reasoning-budget`, `--reasoning-effort`, `--reasoning-preserve` |
| Modes | `--embedding` / `--embeddings`, `--rerank` / `--reranking`, `--pooling` |
| Serveur | `--host`, `--port`, `--api-key`, `--no-webui`, `--slots`, `--props`, `--metrics`, `--sleep-idle-seconds` |
| Routeur | `--models-dir`, `--models-max`, `--models-preset`, `--models-autoload` |

`--cache-type-k iq4_nl` et `--cache-type-v q8_0` sont donc directement atteignables : l'exemple
donné dans la mission est réalisable sans aucune modification de `llama.cpp`.

---

## 2. Architecture Ollama pertinente

### 2.1 Routes

Relevé depuis `server/routes.go`, `GenerateRoutes()` l. 1823-1911.

| Méthode | Chemin | Handler | Reproduit par `ollama.cpp` |
|---|---|---|---|
| HEAD/GET | `/` | `"Ollama is running"` (texte brut) | oui |
| HEAD/GET | `/api/version` | version | oui |
| GET | `/api/status` | `StatusHandler` | oui (dégradé, pas de cloud) |
| POST | `/api/pull` | `PullHandler` | oui |
| POST | `/api/push` | `PushHandler` | **non** — voir §5.3 |
| HEAD/GET | `/api/tags` | `ListHandler` | oui |
| POST | `/api/show` | `ShowHandler` | oui |
| DELETE | `/api/delete` | `DeleteHandler` | oui |
| POST | `/api/create` | `CreateHandler` | oui (sous-ensemble, voir §5.3) |
| POST/HEAD | `/api/blobs/:digest` | `CreateBlobHandler` / `HeadBlobHandler` | oui |
| POST | `/api/copy` | `CopyHandler` | oui |
| GET | `/api/ps` | `PsHandler` | oui |
| POST | `/api/generate` | `GenerateHandler` | oui |
| POST | `/api/chat` | `ChatHandler` | oui |
| POST | `/api/embed` | `EmbedHandler` | oui |
| POST | `/api/embeddings` | `EmbeddingsHandler` | oui |
| POST | `/v1/chat/completions` | `ChatMiddleware()` → `ChatHandler` | oui |
| POST | `/v1/completions` | `CompletionsMiddleware()` → `GenerateHandler` | oui |
| POST | `/v1/embeddings` | `EmbeddingsMiddleware()` → `EmbedHandler` | oui |
| GET | `/v1/models` | `ListMiddleware()` → `ListHandler` | oui |
| GET | `/v1/models/:model` | `RetrieveMiddleware()` → `ShowHandler` | oui |
| POST | `/v1/responses` | `ResponsesMiddleware()` → `ChatHandler` | oui |
| POST | `/v1/messages` | `AnthropicMessagesMiddleware()` → `ChatHandler` | oui |
| POST | `/api/me`, `/api/signout`, `/api/experimental/*` | compte cloud, web search | **non** — hors périmètre |

**Constat majeur n° 4 : Ollama lui-même fait converger ses façades.** Les routes
`/v1/chat/completions`, `/v1/completions`, `/v1/responses` et `/v1/messages` sont des
*middlewares de traduction* branchés sur les mêmes `ChatHandler` / `GenerateHandler` que l'API
native. C'est exactement l'architecture « représentation canonique unique » demandée par la
mission, et elle est validée par l'implémentation de référence.

### 2.2 Conventions comportementales

Depuis `docs/api.md`, section « Conventions » :

- **Noms de modèles** : format `model:tag`, avec namespace optionnel `example/model`. Le tag est
  optionnel et vaut `latest` par défaut.
- **Durées** : toutes les durées des réponses sont exprimées **en nanosecondes** (entiers).
- **Streaming** : réponses en flux d'objets JSON (NDJSON), désactivable par `{"stream": false}`.
- **`keep_alive`** : par défaut `5m`.

Sérialisation de `keep_alive` en **entrée** (`api/types.go`, `Duration.UnmarshalJSON` l. 1243-1271) :

| Valeur JSON | Interprétation |
|---|---|
| absente | 5 minutes |
| nombre `n >= 0` | `n` **secondes** |
| nombre `n < 0` | durée infinie (`MaxInt64`) |
| chaîne | durée Go (`"10m"`, `"1h30m"`, `"300ms"`) |
| chaîne négative | durée infinie |
| autre type | erreur |

Note : un JSON invalide produit une erreur, mais la valeur par défaut est posée **avant** le
switch — un point de détail reproduit tel quel.

### 2.3 Schéma d'erreur

Uniforme : `{"error": "<message>"}` (`server/routes.go`, ~40 occurrences de `gin.H{"error": ...}`).

Cas notables :

| Situation | Code | Corps |
|---|---|---|
| corps manquant | 400 | `{"error": "missing request body"}` |
| modèle inconnu | 404 | `{"error": "model '<nom>' not found"}` |
| nom de modèle invalide | 400 | `{"error": "<InvalidModelNameErrMsg>"}` |
| capacité absente | 400 | `{"error": "\"<modèle>\" does not support thinking"}` / `does not support generate` |
| `raw` incompatible | 400 | `{"error": "raw mode does not support template, system, or context"}` |
| erreur interne | 500 | `{"error": "<message>"}` |

### 2.4 Schémas de requêtes et de réponses

Relevés depuis `api/types.go`. Le détail exhaustif sert de référence d'implémentation.

**`GenerateRequest`** (l. 62-129) : `model`, `prompt`, `suffix`, `system`, `template`, `context[]`,
`stream`, `raw`, `format`, `keep_alive`, `images[]`, `options{}`, `think`, `truncate`, `shift`,
`logprobs`, `top_logprobs`, `_debug_render_only`.

**`ChatRequest`** (l. 133-179) : `model`, `messages[]`, `stream`, `format`, `keep_alive`, `tools`,
`options{}`, `think`, `truncate`, `shift`, `logprobs`, `top_logprobs`.

**`Message`** (l. 197-207) : `role`, `content`, `thinking`, `images[]`, `tool_calls[]`,
`tool_name`, `tool_call_id`. Le rôle est **normalisé en minuscules** au décodage (l. 209-219).

**`ToolCall`** (l. 221-231) : `{id?, function: {index, name, arguments}}`. Les arguments sont un
**ordered map** : l'ordre d'insertion des clés est préservé à la sérialisation.

**`Metrics`** (l. 557-566), embarqué à plat dans `ChatResponse` et `GenerateResponse` :
`total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`,
`eval_duration` — tous `omitempty`, tous en nanosecondes.

**`ChatResponse`** (l. 520-550) : `model`, `created_at`, `message`, `done`, `done_reason?`,
`logprobs?`, + `Metrics` à plat.

**`GenerateResponse`** (l. 891-930) : `model`, `created_at`, `response`, `thinking?`, `done`,
`done_reason?`, `context?`, + `Metrics` à plat, `tool_calls?`, `logprobs?`.

**`ListModelResponse`** (l. 822-833), élément de `/api/tags` :
`name`, `model`, `modified_at`, `size` (int64), `digest`, `details`, `capabilities?`.

**`ProcessModelResponse`** (l. 835-844), élément de `/api/ps` :
`name`, `model`, `size`, `digest`, `details`, `expires_at`, `size_vram`, `context_length`.

**`ModelDetails`** (l. 933-943) :
`parent_model`, `format`, `family`, `families[]`, `parameter_size`, `quantization_level`,
`context_length?`, `embedding_length?`.

**`ShowResponse`** (l. 737-756) : `license?`, `modelfile?`, `parameters?`, `template?`, `system?`,
`renderer?`, `parser?`, `details`, `messages?`, `model_info` (**non-omitempty**), `projector_info?`,
`tensors?`, `capabilities?`, `modified_at?`.

**`EmbedRequest` / `EmbedResponse`** (l. 599-629) : entrée `input` (`string` ou `[]string`),
`truncate`, `dimensions`, `keep_alive`, `options`. Sortie `model`, `embeddings` (`[][]float32`),
`total_duration?`, `load_duration?`, `prompt_eval_count?`.

**`EmbeddingRequest` / `EmbeddingResponse`** (legacy, l. 631-650) : entrée `prompt` (singulier),
sortie `{"embedding": []float64}` (singulier, non imbriqué).

**`ProgressResponse`** (l. 777-783), flux de `/api/pull` : `status`, `digest?`, `total?`,
`completed?`.

**`CreateRequest`** (l. 652-712) : `model`, `stream`, `quantize`, `from`, `files{}`, `adapters{}`,
`template`, `license`, `system`, `parameters{}`, `messages[]`, `renderer`, `parser`, `info{}`, …

**`Capability`** (`types/model/capability.go`) — énumération fermée :
`completion`, `tools`, `insert`, `vision`, `embedding`, `thinking`, `image`, `audio`.

### 2.5 Nommage

`types/model/name.go` : hôte par défaut `registry.ollama.ai`, namespace par défaut `library`, tag
par défaut `latest`. `Name.String()` **omet** l'hôte et le namespace quand ils valent les valeurs
par défaut : `library/llama3:latest` s'affiche `llama3:latest`.

### 2.6 Stockage

Ollama utilise un stockage adressé par contenu : `blobs/sha256-<hex>` + `manifests/<host>/<ns>/<model>/<tag>`.
Les propriétés qui nous intéressent — déduplication, checksums, artefacts partagés, installation
atomique — sont des propriétés du **layout**, pas du protocole de registre OCI. `ollama.cpp`
reprend le layout, pas le protocole (§5.3).

---

## 3. Architecture `ollama-gateway` pertinente

`ollama-gateway` est le consommateur réel. Son contrat est **la** contrainte dure du projet.
C'est une application Python / FastAPI (`app/`, `requirements.txt`, `pytest.ini`).

### 3.1 Ce que la passerelle attend de l'amont

Source de vérité : `app/apis.py`, `app/servers.py`, `app/proxy.py`, `app/context.py`.

**Catalogue d'endpoints sondés** (`app/apis.py`, `CATALOG` l. 45-77) :

```python
"ollama":    GET  /api/version, GET /api/tags, GET /api/ps, POST /api/show,
             POST /api/generate, POST /api/chat, POST /api/embed, POST /api/embeddings
"openai":    GET  /v1/models, POST /v1/chat/completions, POST /v1/completions,
             POST /v1/embeddings, POST /v1/responses
"anthropic": POST /v1/messages, POST /v1/messages/count_tokens
```

**Détection de disponibilité** (`app/servers.py::probe` l. 251-266) : `GET {base}/api/tags`.
Doit répondre **200** avec `{"models": [ {...} ]}` ; la passerelle lit `m.get("name") or m.get("model")`.

**Filtrage des listings** (`app/proxy.py::_filter_models` l. 47-63) : sur `/api/tags` elle filtre
`obj["models"]` par `m.get("name") in allowed or m.get("model") in allowed` ; sur `/v1/models` par
`m.get("id") in allowed`.
→ **`/api/tags` doit porter `name` ET `model`, `/v1/models` doit porter `id`.**

**Injection de contexte** (`app/context.py::inject_num_ctx` l. 198-222) : la passerelle **réécrit
le corps** des requêtes `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings` pour y forcer
`options.num_ctx = min(demandé, plafond_de_la_clé)`.
→ **`options.num_ctx` doit être accepté et honoré sur ces quatre chemins.** Un 400 casserait toutes
les clés à plafond de contexte.

**Gestion du catalogue** (`app/servers.py`) :
- `pull_model` l. 402-433 : `POST {base}/api/pull` avec `{"model": ..., "stream": false}`.
  Attend **200** et `{"status": "success"}` — toute autre valeur de `status` est traitée en échec.
- `delete_model` l. 436-470 : `DELETE {base}/api/delete` avec `{"model": ...}`.
  Attend **200 ou 204** ; **404** est interprété comme « modèle déjà absent ».

**Sonde de compatibilité** (`app/servers.py::_is_served` l. 301-312) — subtilité critique :

> Un chemin est considéré comme « servi » sauf s'il renvoie un 404 de **routeur**. La sonde envoie
> un corps `{}` vide. Ollama répond alors 404 `model '' not found` : le handler existe, donc le
> chemin est servi. La passerelle distingue les deux cas **au mot `model` dans le corps de la
> réponse**.

→ **Contrainte de conformité non évidente** : sur un corps `{}`, chaque endpoint POST doit répondre
soit un code ≠ 404, soit un 404 dont le corps contient le mot `model`. Un 404 générique
(`{"detail":"Not Found"}` de Starlette, `404 page not found` de Gin) ferait apparaître l'endpoint
comme **absent** dans la matrice de compatibilité de la passerelle. Les erreurs de validation
doivent donc suivre le schéma Ollama (`400 {"error": ...}`), pas le `422` par défaut de FastAPI.

**Endpoints jamais proxifiés** (`app/apis.py::MANAGEMENT_PATHS` l. 85-91) : `pull`, `push`,
`delete`, `create`, `copy`, `blobs` sont refusés en 403 à toute clé cliente ; ils ne sont appelés
que depuis la console d'administration, en LAN. `ollama.cpp` n'a donc pas à durcir ces endpoints
contre un usage public — mais il ne doit pas non plus les supposer inatteignables.

**Authentification amont** : `Authorization: Bearer <token>` optionnel par serveur
(`app/servers.py::_upstream_info`). `ollama.cpp` doit accepter — et ignorer sans erreur — un
`Authorization` amont, ou le valider s'il est configuré.

### 3.2 Matrice de compatibilité `ollama-gateway` → `ollama.cpp`

Légende de la colonne « Criticité » : **P0** = la passerelle est cassée sans cela ; **P1** = une
fonctionnalité de la passerelle est cassée ; **P2** = confort.

| # | Endpoint | Usage passerelle | Champs consommés | Streaming | Criticité |
|---|---|---|---|---|---|
| 1 | `GET /api/tags` | sonde de disponibilité + listing + filtrage | `models[].name`, `models[].model` | non | **P0** |
| 2 | `POST /api/chat` | proxy d'inférence | passthrough + `options.num_ctx` injecté | NDJSON | **P0** |
| 3 | `POST /api/generate` | proxy d'inférence, y compris images `x/…` | idem | NDJSON | **P0** |
| 4 | `POST /api/embed` | proxy | idem | non | **P0** |
| 5 | `POST /api/embeddings` | proxy legacy | idem | non | **P1** |
| 6 | `GET /api/version` | matrice de compatibilité | `version` | non | **P1** |
| 7 | `GET /api/ps` | matrice de compatibilité | liste | non | **P1** |
| 8 | `POST /api/show` | matrice de compatibilité | servi sur `{}` | non | **P1** |
| 9 | `GET /v1/models` | listing OpenAI + filtrage | `data[].id` | non | **P0** |
| 10 | `POST /v1/chat/completions` | proxy | passthrough | SSE | **P0** |
| 11 | `POST /v1/completions` | proxy | passthrough | SSE | **P1** |
| 12 | `POST /v1/embeddings` | proxy | passthrough | non | **P1** |
| 13 | `POST /v1/responses` | proxy | passthrough | SSE | **P1** |
| 14 | `POST /v1/messages` | proxy | passthrough | SSE | **P1** |
| 15 | `POST /v1/messages/count_tokens` | matrice | passthrough | non | **P2** |
| 16 | `POST /api/pull` | console d'admin | `{"status":"success"}` | non (`stream:false`) | **P1** |
| 17 | `DELETE /api/delete` | console d'admin | 200/204, 404 = absent | non | **P1** |
| 18 | `POST /v1/images/generations` | capacité image OpenAI | — | — | **hors périmètre** |
| 19 | `/api/generate` avec modèle `x/…` | capacité image Ollama | — | — | **hors périmètre** |

Les lignes 18-19 concernent la génération d'images. `llama.cpp` ne génère pas d'images ;
`ollama.cpp` ne prétendra donc pas à ces capacités. La passerelle les traite comme des capacités
séparées par clé : une cible qui ne les sert pas est simplement marquée non compatible dans la
matrice, sans casser le reste. **C'est une incompatibilité assumée et documentée.**

### 3.3 Conclusion sur la substituabilité

Passer `target = Ollama` à `target = ollama.cpp` ne demande **aucune modification** de
`ollama-gateway`, à condition que les lignes 1-17 soient satisfaites comportementalement. Aucune
modification de `ollama-gateway` n'est prévue ni autorisée dans ce projet.

---

## 4. Fonctionnalités déjà disponibles vs manquantes

### 4.1 Déjà fournies par `llama-server` — à ne surtout pas réimplémenter

| Domaine | Disponible |
|---|---|
| Inférence, GGUF, tokenisation, KV cache, GPU, Flash Attention | oui |
| Chat templates Jinja, `tool_use` template, parsing des tool calls | oui (`--jinja`) |
| Raisonnement / thinking, `reasoning_format`, budget de raisonnement | oui |
| Multimodal (vision, audio) via `mmproj` | oui |
| Decoding spéculatif, draft model | oui |
| Slots, batching continu, parallélisme | oui |
| **OpenAI Chat Completions** | oui, natif |
| **OpenAI Responses** | oui, natif |
| **Anthropic Messages** + `count_tokens` | oui, natif |
| Embeddings, reranking | oui |
| Introspection `/props`, `/slots`, `/metrics`, `/health` | oui |
| Supervision multi-modèles, LRU, SSE, single-flight | oui, en mode routeur |
| Téléchargement Hugging Face (`--hf-repo`, `--hf-file`, `--docker-repo`) | oui |

### 4.2 Manquantes — périmètre réel de `ollama.cpp`

| Manque | Conséquence |
|---|---|
| **API native Ollama (`/api/*`)** | aucun client Ollama ne fonctionne |
| **Registre de modèles avec identité stable** | `/v1/models` renvoie des bouchons (`size: ""`, `digest: ""`) |
| **Manifests de modèles** | pas de configuration runtime par modèle, pas d'artefacts liés |
| **`size` et `digest` réels** | `/api/tags` et `/api/ps` inexploitables par les outils Ollama |
| **`keep_alive` par requête** | sémantique Ollama absente |
| **`expires_at`, `size_vram`, `context_length` par modèle résident** | `/api/ps` inexploitable |
| **Capacités canoniques Ollama** | `/api/show.capabilities` absent |
| **Scheduler conscient de la mémoire, avec priorités** | LRU simple par compte de modèles, pas par mémoire |
| **Registre privé, checksums, installation atomique** | absent |
| **`pull` / `create` / `copy` / `delete` / `blobs`** | absent |
| **Nommage Ollama (`ns/model:tag`)** | absent |
| **Représentation canonique commune aux quatre façades** | chaque façade est indépendante dans `llama-server` |

---

## 5. Architecture cible de `ollama.cpp`

### 5.1 Vue d'ensemble

```
                              ollama.cpp
          ┌──────────────┬───────────────┬───────────────┬──────────────┐
       Ollama API    OpenAI Chat    OpenAI Resp.   Anthropic Msg.   (façades)
       /api/*        /v1/chat/…     /v1/responses  /v1/messages
          └──────────────┴───────┬───────┴───────────────┘
                                 ▼
                   couche canonique  (CanonicalRequest / CanonicalResult)
                                 ▼
                          ModelRegistry        (noms, manifests, artefacts, digests)
                                 ▼
                      ModelLifecycleManager    (états, single-flight, keep_alive)
                                 ▼
                          ModelScheduler       (mémoire, priorités, éviction)
                                 ▼
                     LlamaServerSupervisor     (1 processus par modèle logique)
                                 ▼
                           llama-server  →  llama.cpp
```

### 5.2 Choix technologiques et compromis

| Décision | Justification | Compromis assumé |
|---|---|---|
| **Processus séparé, pas un fork de `llama.cpp`** | §33 de la mission : conserver l'upstream. Zéro patch sur `llama.cpp`, mise à jour du backend sans rebase. | Un saut réseau supplémentaire en loopback (négligeable devant le coût d'inférence). |
| **Python 3.11 + FastAPI + httpx** | Stack déjà retenue par `ollama-gateway` (même écosystème, mêmes outils de test). CLAUDE.md §3 privilégie Python pour les services backend. Traduction JSON et proxy de flux : le coût CPU est marginal devant l'inférence. | Pas de binaire statique unique comme Ollama. Mitigé par la conteneurisation. |
| **Un processus `llama-server` par modèle logique** | Pattern déjà retenu en amont par `llama.cpp` (`server-models.cpp`). Permet des flags **par modèle** — c'est l'objectif n° 1 du projet. | Surcoût mémoire par processus ; mitigé par le scheduler. |
| **Pilotage par CLI + HTTP, pas par le mode routeur** | Le mode routeur impose sa propre notion de modèle (presets / models-dir) et n'expose pas de crochet pour les manifests, `keep_alive` par requête ou l'éviction par mémoire. | Duplication partielle de la supervision. Atténué en calquant la machine à états sur celle de l'amont. |
| **Façades → canonique → `/v1/chat/completions`** | Un seul sérialiseur backend à maintenir et à tester ; surface `llama-server` la plus complète et la plus stable. | Les façades Responses / Messages ne passent pas par les implémentations natives de `llama-server`. En contrepartie, la cohérence inter-façades devient testable (§35). |
| **Layout de stockage inspiré d'Ollama** | Déduplication, checksums et artefacts partagés viennent du layout, pas du protocole. | Pas de compatibilité binaire avec un répertoire `~/.ollama` existant : c'est un layout **inspiré**, pas identique. |

### 5.3 Périmètre explicitement exclu

| Exclu | Raison |
|---|---|
| `POST /api/push` | Publier vers un registre distant n'a pas de sens sans registre Ollama. Répondra `501` avec un message explicite. |
| `/api/me`, `/api/signout`, `/api/user/keys/*` | Comptes cloud Ollama. Hors périmètre. |
| `/api/experimental/web_search`, `web_fetch` | Services Ollama hébergés. Hors périmètre. |
| Génération d'images (`x/…`, `/v1/images/generations`) | `llama.cpp` ne génère pas d'images. |
| Modèles distants (`remote_host`, `remote_model`) | Fédération Ollama Cloud. Hors périmètre. |
| Clés API, quotas, contrôle d'accès aux modèles | Responsabilité de `ollama-gateway` (§0.2). |
| Protocole de registre OCI Ollama | Seul le *layout* est repris, pas le protocole. |

Ces exclusions sont **documentées et non silencieuses** : chaque endpoint exclu répond un code et
un message explicites, jamais un 404 de routeur — ce qui préserve la sonde de la passerelle (§3.1).

### 5.4 Modèle conversationnel canonique

Toute requête entrante est convertie en une représentation unique, et tout résultat backend est
converti en un résultat unique avant sérialisation par la façade d'origine.

```
CanonicalRequest
  model: ModelRef                  nom Ollama normalisé
  messages: [CanonicalMessage]
  tools: [ToolDefinition]
  tool_choice: ToolChoice
  options: SamplingOptions         num_ctx, temperature, top_p, stop, seed, …
  response_format: ResponseFormat  texte | json | json_schema
  thinking: ThinkingRequest        off | on | low|medium|high|max
  stream: bool
  keep_alive: KeepAlive
  source_api: "ollama"|"openai-chat"|"openai-responses"|"anthropic"

CanonicalMessage  (union discriminée par `role`)
  SystemMessage    (text)
  UserMessage      (content: [TextBlock | ImageInput])
  AssistantMessage (content: [TextBlock | ReasoningBlock], tool_calls: [ToolCall])
  ToolResultMessage(call_id, name, content)

ToolCall     : id, name, arguments (dict ordonné), index
ImageInput   : media_type, data (bytes) — normalisation base64 / data URI / URL
ReasoningBlock: text, signature?, redacted?
```

Invariants à garantir par les tests (§35, §36) :

1. `call_id` est **préservé bit à bit** dans les deux sens sur les quatre façades.
2. Un `function_call_output` (Responses) ou un `tool_result` (Messages) devient un
   `ToolResultMessage`, **jamais** un `UserMessage`.
3. Les blocs de raisonnement ne sont jamais fusionnés dans le contenu textuel.
4. L'ordre des clés d'arguments de tool call est préservé.
5. Le même échange conceptuel exprimé dans les quatre formats produit un `CanonicalRequest`
   **structurellement égal**, hors champ `source_api`.

### 5.5 Manifest de modèle

Fichier JSON versionné, décrivant un modèle logique.

```json
{
  "schema_version": 1,
  "name": "qwen3.6-35b",
  "source":     { "type": "private", "model": "qwen3.6-35b" },
  "artifacts":  { "model": "sha256:...", "mmproj": "sha256:...", "draft": null,
                  "adapters": [], "template": null },
  "capabilities_override": { "vision": null, "tools": null, "thinking": null },
  "runtime":    { "context": 262144, "flash_attention": true,
                  "cache_type_k": "iq4_nl", "cache_type_v": "q8_0", "parallel": 4 },
  "lifecycle":  { "keep_alive": "15m", "priority": 100 }
}
```

`capabilities_override` n'est pas une déclaration : c'est une **restriction**. Une capacité ne peut
être forcée à `true` que si elle est observable ; elle peut en revanche être forcée à `false`. Cela
respecte l'exigence « ne pas annoncer `vision=true` si le modèle ne peut pas traiter une image ».

### 5.6 Ordre de précédence de la configuration

```
override de requête  >  manifest  >  métadonnées GGUF  >  défauts ollama.cpp
```

Le GGUF reste la source de vérité privilégiée pour l'architecture, le tokenizer, le chat template
et le contexte natif. Le manifest ne duplique pas ces informations : il les **surcharge**
seulement lorsque c'est nécessaire.

### 5.7 États du cycle de vie

```
NOT_PRESENT ──pull──► DOWNLOADING ──► UNLOADED ──load──► LOADING ──► READY
                           │                                │          │ ▲
                           └──────échec───────► FAILED ◄─────┘   requête│ │fin
                                                                        ▼ │
                                                        IDLE ◄──────── BUSY
                                                          │
                                          keep_alive / pression mémoire
                                                          ▼
                                                     UNLOADING ──► UNLOADED
```

- **single-flight** : deux requêtes concurrentes sur le même modèle partagent un chargement unique ;
- un modèle **BUSY n'est jamais évincé** ;
- un modèle multimodal ne passe **jamais** `READY` si un artefact obligatoire (`mmproj`) manque.

### 5.8 Scheduler

Entrées : modèle demandé, modèles chargés, mémoire utilisée et disponible, taille estimée du
modèle, allocation KV, requêtes actives, `last_used`, `keep_alive`, `priority`, coût de chargement.

Politique initiale :

```
modèle READY                      → réutiliser
modèle déchargé + mémoire libre   → charger
pression mémoire                  → chercher un candidat IDLE à évincer
keep_alive expiré                 → candidat privilégié
sinon                             → LRU pondéré par la priorité
```

Chaque décision est journalisée de façon explicable :

```
model=qwen3.6 action=evict reason=memory_pressure idle_seconds=731 priority=50
```

### 5.9 Plan de contrôle

Conformément au §24 de la mission, l'administration passe par **l'API Ollama elle-même**
(`/api/tags`, `/api/show`, `/api/pull`, `/api/create`, `/api/copy`, `/api/delete`, `/api/ps`).
Aucun namespace `/admin/*` propriétaire n'est créé.

Les fonctions sans équivalent Ollama — configuration runtime fine par modèle, registre privé —
sont portées par le **manifest**, donc par `/api/create` et le système de fichiers, et non par une
API parallèle.

### 5.10 Arborescence de données

```
$OLLAMACPP_MODELS/
  blobs/
    sha256-<hex>                       artefacts adressés par contenu
  manifests/
    <host>/<namespace>/<model>/<tag>   manifest JSON
  tmp/
    <uuid>.partial                     téléchargements en cours (installation atomique)
```

---

## 6. Interfaces internes proposées

```python
class ModelRegistry:
    def resolve(self, name: str) -> ModelRef                    # nom → référence normalisée
    def get(self, ref: ModelRef) -> RegisteredModel | None
    def list(self) -> list[RegisteredModel]
    def install(self, ref: ModelRef, manifest: Manifest) -> RegisteredModel
    def copy(self, src: ModelRef, dst: ModelRef) -> None
    def delete(self, ref: ModelRef) -> bool

class BlobStore:
    def path(self, digest: str) -> Path
    def has(self, digest: str) -> bool
    def ingest(self, source, expected_digest: str | None) -> str   # atomique + checksum

class ModelLifecycleManager:
    async def ensure_ready(self, ref, keep_alive) -> LoadedModel   # single-flight
    async def unload(self, ref) -> None
    def status(self, ref) -> ModelState
    def residents(self) -> list[LoadedModel]

class ModelScheduler:
    async def admit(self, ref, estimated) -> AdmissionDecision     # évince si nécessaire
    def sweep(self) -> list[EvictionDecision]                      # keep_alive expiré

class LlamaServerSupervisor:
    async def spawn(self, ref, runtime: RuntimeConfig) -> Instance
    async def terminate(self, instance) -> None
    async def props(self, instance) -> dict                        # GET /props

class CapabilityDetector:
    def detect(self, props: dict, manifest: Manifest, artifacts) -> set[Capability]
```

---

## 7. Plan d'implémentation

Ordre imposé par la mission (§37), adapté à l'audit. Chaque étape se termine par build, tests,
correction des régressions, commit et push.

| # | Étape | État |
|---|---|---|
| 1 | Audit du routeur `llama-server` | fait |
| 2 | Audit d'Ollama | fait |
| 3 | Audit d'`ollama-gateway` | fait |
| 4 | Matrice de compatibilité | fait (§3.2) |
| 5 | Documentation socle (ce document, DAT, backlog, journal) | fait |
| 6 | Configuration centralisée, erreurs, nommage | fait, tests unitaires |
| 7 | Représentation canonique | fait, tests unitaires et d'équivalence |
| 8 | `BlobStore`, manifests, `ModelRegistry` | fait, tests unitaires |
| 9 | `LlamaServerSupervisor` | fait, tests d'intégration sur processus réels |
| 10 | `ModelLifecycleManager` + single-flight | fait, tests d'intégration |
| 11 | `ModelScheduler` + `keep_alive` | fait, tests unitaires et d'intégration |
| 12 | Détection de capacités | fait, tests d'intégration |
| 13 | Ollama `/api/tags`, `/api/show`, `/api/ps`, `/api/version` | fait, tests d'API |
| 14 | Ollama `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings` | fait, tests d'API |
| 15 | Ollama `/api/pull`, `/api/create`, `/api/copy`, `/api/delete`, `/api/blobs` | fait, tests d'API |
| 16 | Registre privé, source Hugging Face | fait, tests sur registres HTTP réels |
| 17 | OpenAI Chat Completions, Completions, Embeddings, Models | fait, tests d'API |
| 18 | OpenAI Responses | fait, tests d'API et d'équivalence |
| 19 | Anthropic Messages, `count_tokens` | fait, tests d'API |
| 20 | Vision, thinking | fait, tests d'intégration |
| 21 | Tests de conformité | fait : conformité `ollama-gateway`, équivalence des 4 façades, multi-tours ≥ 10 |
| 22 | Conteneurisation, `runDev` / `runStaging` / `runProd` | fait |

**Reste non vérifié** : l'inférence sur un vrai modèle GGUF (OC-085). Le binaire `llama-server` a
été compilé depuis l'upstream et son contrat vérifié — toute ligne de commande produite par le
middleware est acceptée, `/health` et `/v1/models` répondent conformément sur une instance
réellement démarrée — mais la politique réseau de l'environnement de construction bloque le
téléchargement de modèles, si bien qu'aucun GGUF réel n'a pu être chargé. Le reste de la chaîne
est couvert par un faux `llama-server` implémentant le contrat HTTP amont.

Le suivi fin, unité par unité, vit dans `docs/BACKLOG.md`, qui fait foi.

---

## 8. Risques

| # | Risque | Impact | Mitigation |
|---|---|---|---|
| R1 | **Sonde de compatibilité de la passerelle** : un 404 générique fait apparaître un endpoint comme absent (§3.1) | La matrice de la passerelle déclare `ollama.cpp` incompatible à tort | Gestionnaire d'erreurs global au schéma Ollama ; test de conformité dédié rejouant `_is_served` sur chaque endpoint avec un corps `{}` |
| R2 | **`options.num_ctx` injecté par la passerelle** sur 4 chemins | Toute clé à plafond de contexte échoue en 400 | `num_ctx` accepté et honoré ; test dédié |
| R3 | **Durées en nanosecondes** | Les clients calculent des débits faux d'un facteur 10⁹ | Type dédié, jamais de flottant de secondes ; tests sur les 6 champs de `Metrics` |
| R4 | **`size` / `digest` bouchons** hérités de `llama-server` | `ollama list` affiche des tailles vides ; la passerelle filtre mal | Le registre calcule un digest réel et une taille réelle depuis le `BlobStore` |
| R5 | **Dérive de la sémantique `keep_alive`** (secondes vs durée Go, négatif = infini, 0 = déchargement) | Modèles jamais déchargés, ou déchargés en boucle | Parseur dédié, table de vérité issue de `Duration.UnmarshalJSON`, tests exhaustifs |
| R6 | **Estimation mémoire imprécise** → OOM ou sous-utilisation | Crash de l'hôte ou gâchis de VRAM | Estimation prudente (taille GGUF + KV), plafond configurable, marge de sécurité, journalisation explicable |
| R7 | **Perte sémantique du tool calling** entre façades | Boucles d'agents cassées, `call_id` perdus | Représentation canonique unique + tests multi-tours ≥ 10 appels sur les 4 façades |
| R8 | **Chemins issus de manifests distants** (`../`, chemins absolus) | Écrasement arbitraire de fichiers | Aucune écriture par chemin fourni : le `BlobStore` est adressé par contenu ; validation stricte, refus de tout chemin non canonique |
| R9 | **Fuite de secrets** (jetons de registre, `Authorization`) | Compromission | Interdiction de journaliser les en-têtes d'authentification ; test de non-régression sur les logs |
| R10 | **Divergence de l'upstream `llama.cpp`** | Le middleware casse à la mise à jour | Aucun patch sur `llama.cpp` ; dépendance limitée à la surface HTTP publique et aux flags CLI documentés ; tests de contrat sur `/props` |
| R11 | **Absence de GPU dans l'environnement de développement** | Le comportement d'offload et l'occupation VRAM ne sont pas vérifiables ici | Tests sur CPU avec de très petits modèles ; les décisions dépendantes du GPU sont isolées derrière une abstraction de mesure mémoire, testée par injection |
| R12 | **Génération d'images non supportée** | Deux cases de la matrice de la passerelle restent rouges | Incompatibilité assumée, documentée (§5.3) ; aucune capacité mensongère annoncée |

---

## 9. Résultat visé

```bash
OLLAMA_HOST=http://localhost:11434 ollama list     # parle à ollama.cpp
```

```
ollama-gateway  →  http://ollama-cpp:11434         # cible Ollama interchangeable
```

et simultanément :

```
POST /api/chat              ┐
POST /v1/chat/completions   ├── mêmes modèles, même runtime, même registre
POST /v1/responses          │
POST /v1/messages           ┘
```
