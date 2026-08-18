# Configuration d'un modèle — manifest et pilotage de `llama-server`

@spec docs/BACKLOG.md OC-021 « Manifests de modèles », OC-031 « Construction des arguments
      runtime », OC-032 « Détection de capacités », OC-047 « options.num_ctx »
@spec docs/ollama.cpp-architecture.md §1.5 « Arguments CLI pertinents », §5.5 « Manifest de
      modèle », §5.6 « Ordre de précédence »
@spec docs/DAT.md §4.2 « Manifest », §4.3 « Précédence de configuration »

C'est **la raison d'être de `ollama.cpp`** : donner à chaque modèle sa propre configuration
`llama-server`, là où Ollama impose la sienne. Ce document est la référence complète de ce qui
est réglable, où le régler, et vers quel drapeau chaque réglage se traduit.

Implémentation : `ollamacpp/storage/manifests.py` (schéma) et `ollamacpp/runtime/args.py`
(traduction en ligne de commande). Tests : `tests/test_runtime_args.py` et
`tests/test_llama_server_contract.py`, qui vérifient que chaque ligne produite est acceptée par
le binaire réel.

---

## 1. Où vit la configuration

```
$OLLAMACPP_MODELS/manifests/<hôte>/<namespace>/<modèle>/<tag>
```

Par exemple, `qwen3:8b` installé depuis le registre par défaut :

```
$OLLAMACPP_MODELS/manifests/registry.ollama.ai/library/qwen3/8b
```

Le fichier est un JSON. Il est relu à chaque démarrage : le registre n'a pas de base de données.

---

## 2. Schéma complet du manifest

```json
{
  "schema_version": 1,
  "name": "qwen3:8b",
  "source":    { "type": "private", "reference": "qwen3.6-35b" },
  "artifacts": {
    "model":    "sha256:5f3a…",
    "mmproj":   "sha256:9b21…",
    "draft":    "sha256:c4e7…",
    "adapters": ["sha256:11aa…"],
    "template": null
  },
  "capabilities_override": { "vision": false },
  "runtime":   { "context": 262144, "flash_attention": true,
                 "cache_type_k": "iq4_nl", "cache_type_v": "q8_0", "parallel": 4 },
  "lifecycle": { "keep_alive": "15m", "priority": 100 },
  "system":     "Tu es un assistant concis.",
  "template":   "",
  "license":    "",
  "parameters": { "temperature": 0.7, "stop": ["<|end|>"] },
  "modified_at": "2026-08-17T18:00:00+00:00"
}
```

### 2.1 `artifacts` — les fichiers du modèle

| Clé | Rôle | Effet |
|---|---|---|
| `model` | poids, **obligatoire** | `--model` |
| `mmproj` | projecteur multimodal | `--mmproj`, active la capacité `vision` |
| `draft` | modèle de brouillon | `--model-draft`, active le decoding spéculatif |
| `adapters` | adaptateurs LoRA | un `--lora` par entrée |
| `template` | template externe | réservé, non encore appliqué |

Les artefacts sont **toujours des digests**, jamais des chemins. Un manifest venu d'un registre
distant ne peut donc pas désigner un fichier arbitraire du système (risque R8). Un artefact
déclaré mais absent du magasin empêche le modèle d'atteindre l'état `READY` — un modèle
multimodal sans son `mmproj` échoue au chargement plutôt que d'accepter des images qu'il ne peut
pas traiter.

Lors d'un `pull` depuis Hugging Face, le `mmproj` est retenu automatiquement et **apparié à la
quantification** du fichier de poids choisi lorsque le dépôt en publie plusieurs. Vérifié sur
`ggml-org/SmolVLM-256M-Instruct-GGUF` :

| Référence tirée | `model` | `mmproj` |
|---|---|---|
| `hf.co/ggml-org/SmolVLM-256M-Instruct-GGUF` | `…-Q8_0.gguf` | `mmproj-…-Q8_0.gguf` |
| `…-GGUF:f16` | `…-f16.gguf` | `mmproj-…-f16.gguf` |
| `ggml-org/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M` | `…-Q4_K_M.gguf` | `mmproj-…-Q8_0.gguf` (repli : aucun Q4_K_M publié) |

### 2.2 `capabilities_override` — restreindre, jamais accorder

```json
"capabilities_override": { "vision": false, "tools": false }
```

Seule la valeur `false` est acceptée. Écrire `"vision": true` fait **échouer la validation du
manifest** :

```
capabilities_override ne peut que désactiver une capacité ('vision' = True)
```

Les capacités sont constatées, pas déclarées : `vision` exige un `mmproj` installé *et* la
modalité active dans `/props` du modèle chargé ; `tools` vient de `chat_template_caps`. Ce champ
sert à retirer une capacité réelle mais indésirable — par exemple interdire les outils sur un
modèle partagé.

Capacités désactivables : `completion`, `tools`, `insert`, `vision`, `embedding`, `thinking`,
`reranking`.

### 2.3 `lifecycle` — résidence en mémoire

| Champ | Défaut | Effet |
|---|---|---|
| `keep_alive` | `OLLAMACPP_KEEP_ALIVE` | durée de résidence après la dernière requête |
| `priority` | `0` | résistance à l'éviction : à égalité de LRU, la priorité basse part d'abord |

`keep_alive` accepte la syntaxe Ollama : nombre = **secondes**, chaîne = durée (`"15m"`,
`"1h30m"`), négatif = résidence illimitée, `0` = déchargement dès la fin de la requête.

### 2.4 Champs de présentation

`system`, `template`, `license` et `parameters` alimentent `/api/show` et sont renseignés par
`/api/create`. `parameters` est rendu au format Modelfile dans la réponse de `/api/show`.

---

## 3. `runtime` — table complète des réglages

Chaque champ correspond à un drapeau réel de `common/arg.cpp`, vérifié contre le binaire compilé.
Un champ absent ou `null` laisse `llama-server` appliquer son propre défaut : `ollama.cpp`
n'émet **que** ce qui est explicitement demandé.

### 3.1 Contexte et débit

| Champ | Type | Drapeau | Notes |
|---|---|---|---|
| `context` | entier | `--ctx-size` | fenêtre de contexte. Principal consommateur de mémoire |
| `batch` | entier | `--batch-size` | taille de lot logique |
| `ubatch` | entier | `--ubatch-size` | taille de micro-lot physique |
| `parallel` | entier | `--parallel` | requêtes simultanées. **Multiplie la mémoire du cache KV d'autant** |

### 3.2 CPU

| Champ | Type | Drapeau |
|---|---|---|
| `threads` | entier | `--threads` |
| `threads_batch` | entier | `--threads-batch` |
| `numa` | chaîne | `--numa` |

### 3.3 GPU

| Champ | Type | Drapeau | Notes |
|---|---|---|---|
| `gpu_layers` | entier | `--n-gpu-layers` | `0` force le CPU ; une valeur élevée déporte tout |
| `tensor_split` | chaîne | `--tensor-split` | répartition multi-GPU, ex. `"0.6,0.4"` |
| `main_gpu` | entier | `--main-gpu` | GPU portant les tenseurs non répartis |

### 3.4 Attention et cache KV — le cœur du sujet

| Champ | Type | Drapeau | Notes |
|---|---|---|---|
| `flash_attention` | booléen | `--flash-attn on\|off` | non précisé = défaut `auto` de `llama.cpp` |
| `cache_type_k` | chaîne | `--cache-type-k` | quantisation du cache K |
| `cache_type_v` | chaîne | `--cache-type-v` | quantisation du cache V |
| `kv_unified` | booléen | `--kv-unified` si `true` | cache KV unifié entre slots |
| `kv_offload` | booléen | `--no-kv-offload` si `false` | garde le cache KV hors GPU |

Types de cache reconnus par l'estimateur mémoire : `f32`, `f16`, `bf16`, `q8_0`, `q5_0`, `q5_1`,
`q4_0`, `q4_1`, `iq4_nl`. Un type inconnu est accepté et transmis à `llama-server`, mais estimé
comme `f16` — le côté prudent.

L'exemple de la mission :

```json
"runtime": { "cache_type_k": "iq4_nl", "cache_type_v": "q8_0", "flash_attention": true }
```

devient :

```
--cache-type-k iq4_nl --cache-type-v q8_0 --flash-attn on
```

C'est ce qu'Ollama ne permet pas d'exprimer par modèle.

### 3.5 Mémoire

| Champ | Type | Drapeau |
|---|---|---|
| `mmap` | booléen | `--no-mmap` si `false` |
| `mlock` | booléen | `--mlock` si `true` |

### 3.6 Decoding spéculatif

| Champ | Type | Drapeau |
|---|---|---|
| `draft_max` | entier | `--draft-max` |
| `draft_min` | entier | `--draft-min` |
| `draft_p_min` | flottant | `--draft-p-min` |

Exige `artifacts.draft`, qui produit `--model-draft`.

### 3.7 Modes de service

| Champ | Type | Drapeau | Notes |
|---|---|---|---|
| `embedding` | booléen | `--embedding` si `true` | **exclusif** de la génération |
| `reranking` | booléen | `--reranking` si `true` | idem |
| `pooling` | chaîne | `--pooling` | stratégie d'agrégation des embeddings |

`llama-server` refuse les embeddings sans `--embedding`, et la génération avec. Un modèle servant
les deux usages doit donc être installé **deux fois**, sous deux noms, avec deux `runtime`
différents. La capacité `completion` n'est d'ailleurs pas annoncée sur un modèle en mode
embedding : l'annoncer serait faux.

### 3.8 Templates et raisonnement

| Champ | Type | Drapeau |
|---|---|---|
| `chat_template` | chaîne | `--chat-template` |
| `reasoning_format` | chaîne | `--reasoning-format` |
| `reasoning_budget` | entier | `--reasoning-budget` |

### 3.9 `extra_args` — la soupape

```json
"runtime": { "extra_args": ["--cache-reuse", "256", "--slot-prompt-similarity", "0.5"] }
```

Les arguments sont ajoutés tels quels en fin de ligne de commande. Cela évite qu'un besoin
ponctuel impose de faire évoluer le schéma, tout en restant visible dans le manifest et dans le
journal.

**Drapeaux réservés, filtrés silencieusement d'`extra_args`** : `--host`, `--port`, `--model`,
`-m`, `--alias`, `-a`, `--api-key`, `--api-key-file`, `--models-dir`, `--models-max`, `--mmproj`,
`--path`. Le filtrage retire le drapeau **et sa valeur**. Sans cette règle, un manifest pourrait
exposer une instance hors de la boucle locale ou lui faire charger un autre fichier que celui
demandé.

### 3.10 Drapeaux posés systématiquement

| Drapeau | Motif |
|---|---|
| `--no-webui` | une instance interne n'a pas d'interface web à servir |
| `--jinja` | active le rendu des chat templates et le parsing natif des appels d'outils |
| `--host`, `--port`, `--model`, `--alias` | identité et exposition, décidées par le superviseur |

`--props` n'est **pas** passé : `GET /props` est servi sans condition, et le drapeau n'activerait
que le `POST /props` mutant, dont l'instance n'a aucun besoin.

---

## 4. Ordre de précédence

```
option de la requête   >   manifest   >   métadonnées GGUF   >   défauts du service
```

### 4.1 Ce qu'une requête peut surcharger

Les options Ollama qui configurent l'**instance** — et non l'échantillonnage — provoquent un
**rechargement** du modèle si elles diffèrent de la configuration en place :

| Option de requête | Champ `runtime` | Drapeau |
|---|---|---|
| `options.num_ctx` | `context` | `--ctx-size` |
| `options.num_batch` | `batch` | `--batch-size` |
| `options.num_gpu` | `gpu_layers` | `--n-gpu-layers` |
| `options.num_thread` | `threads` | `--threads` |

C'est la sémantique d'Ollama, où ces options sont des options de *runner*. C'est aussi ce qui
rend effectif le plafond de contexte injecté par `ollama-gateway` : traiter `num_ctx` comme un
paramètre de génération le rendrait silencieusement inopérant.

Un modèle **occupé** n'est pas rechargé au milieu d'une requête : la requête est servie avec la
configuration en place, et l'événement `runtime_override_ignored_while_busy` est journalisé.

Toutes les autres options (`temperature`, `top_p`, `top_k`, `seed`, `stop`, `num_predict`…) sont
des paramètres de génération : elles n'entraînent aucun rechargement.

### 4.2 Ce que le GGUF fournit

Le GGUF reste la source de vérité pour l'architecture, le tokenizer, le chat template et le
contexte natif. Le manifest ne les duplique pas : il les surcharge seulement quand c'est
nécessaire. En l'absence de `runtime.context`, le contexte natif du GGUF est utilisé, puis
`OLLAMACPP_DEFAULT_CONTEXT`.

---

## 5. Modifier la configuration d'un modèle

1. éditer la section `runtime` de son manifest ;
2. décharger le modèle pour que la nouvelle configuration prenne effet :

```bash
curl http://localhost:11434/api/chat -d '{"model":"qwen3:8b","messages":[],"keep_alive":0}'
```

3. relancer une requête et vérifier :

```bash
curl http://localhost:11434/api/ps        # context_length reflète la nouvelle valeur
```

La ligne de commande réellement transmise apparaît dans le journal :

```
event=load_started model=qwen3:8b port=18000 args="--model … --cache-type-k iq4_nl --flash-attn on …"
```

C'est le premier endroit où regarder quand un modèle ne se comporte pas comme prévu.

---

## 6. Effet mémoire des réglages

L'ordonnanceur estime l'empreinte d'un modèle avant de l'admettre :

```
octets ≈ poids + cache KV + surcoût de processus (256 Mio)
cache KV ≈ 2 × contexte × couches × dimension_kv × octets_par_élément × parallel
```

`dimension_kv` tient compte de la *grouped-query attention* quand le GGUF publie
`attention.head_count` et `attention.head_count_kv` — sans quoi l'estimation serait fausse d'un
facteur 4 à 8 sur les modèles récents.

Conséquences pratiques :

| Action | Effet mémoire |
|---|---|
| doubler `context` | double le cache KV |
| `parallel: 4` | quadruple le cache KV |
| `cache_type_k`/`_v` en `q8_0` | divise le cache KV par ~1,8 |
| `cache_type_k` en `iq4_nl`, `_v` en `q8_0` | divise le cache KV par ~2,3 |

Un refus d'admission est explicite :

```
cannot load model 'qwen3:8b': not enough memory (19595788770 bytes required)
```

Réduire `context`, quantifier le cache KV, ou augmenter `OLLAMACPP_MEMORY_LIMIT_BYTES`.

---

## 7. Exemples

### Modèle de production, contexte long et cache quantifié

```json
"runtime": {
  "context": 131072, "parallel": 4,
  "flash_attention": true, "cache_type_k": "iq4_nl", "cache_type_v": "q8_0",
  "gpu_layers": 99, "tensor_split": "0.5,0.5"
},
"lifecycle": { "keep_alive": "-1", "priority": 100 }
```

Résidence illimitée et priorité haute : ce modèle ne sera jamais évincé au profit d'un autre.

### Modèle multimodal

```json
"artifacts": { "model": "sha256:…", "mmproj": "sha256:…" },
"runtime":   { "context": 32768, "flash_attention": true }
```

La capacité `vision` apparaît dès lors que le projecteur est installé et que l'instance annonce
la modalité image.

### Modèle d'embeddings

```json
"runtime": { "context": 8192, "embedding": true, "pooling": "mean" },
"lifecycle": { "keep_alive": "1h" }
```

Aucune capacité `completion` : ce modèle ne génère pas de texte.

### Modèle avec decoding spéculatif

```json
"artifacts": { "model": "sha256:…", "draft": "sha256:…" },
"runtime":   { "context": 32768, "draft_max": 16, "draft_min": 4, "draft_p_min": 0.75 }
```

### Poste de développement, mémoire contrainte

```json
"runtime":   { "context": 4096, "gpu_layers": 0, "threads": 4, "mmap": true },
"lifecycle": { "keep_alive": "2m", "priority": 0 }
```
