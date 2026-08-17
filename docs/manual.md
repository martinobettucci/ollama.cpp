# Manuel d'exploitation de `ollama.cpp`

@spec docs/BACKLOG.md OC-094 « Manuel utilisateur »
@spec docs/DAT.md §12 « Commandes »
@spec README.md

Ce manuel s'adresse à la personne qui **exploite** `ollama.cpp` : installer des modèles, régler
leur configuration, comprendre ce que fait l'ordonnanceur, diagnostiquer un problème.

Il décrit le comportement réel du service. Aucune adresse interne, aucune clé et aucune valeur de
secret n'y figure ; seuls les **noms** de variables d'environnement sont mentionnés.

---

## 1. Ce que fait `ollama.cpp`

`ollama.cpp` parle le protocole d'Ollama, mais exécute l'inférence avec `llama-server`. Pour un
client, c'est un Ollama. Pour l'exploitant, c'est un `llama-server` dont on peut régler finement
chaque modèle.

Il expose quatre familles d'API sur le **même** catalogue et le **même** runtime :

| Famille | Endpoints |
|---|---|
| Ollama natif | `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`, `/api/tags`, `/api/show`, `/api/ps`, `/api/pull`, `/api/create`, `/api/copy`, `/api/delete`, `/api/blobs`, `/api/version` |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses` |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` |

---

## 2. Premiers pas

### 2.1 Vérifier que le service répond

```bash
curl http://localhost:11434/                # → Ollama is running
curl http://localhost:11434/api/version     # → {"version":"..."}
curl http://localhost:11434/api/tags        # → {"models":[...]}
```

`/api/tags` répond `200` même sans aucun modèle installé : un catalogue vide n'est pas une panne.

### 2.2 Installer un modèle de démonstration

```bash
python scripts/seed.py --verify
```

Le script télécharge un petit modèle, lui donne le nom court `demo:latest` et exécute une
inférence de contrôle. Il passe par les mêmes API que n'importe quel client.

### 2.3 Utiliser le CLI Ollama

```bash
export OLLAMA_HOST=http://localhost:11434
ollama list
ollama show demo
ollama ps
```

---

## 3. Gérer le catalogue

Ces opérations exigent `OLLAMACPP_MANAGEMENT_ENABLED=true`. En production, laisser à `false` et
n'ouvrir que le temps d'une intervention.

### 3.1 Télécharger depuis Hugging Face

```bash
curl http://localhost:11434/api/pull -d '{"model":"hf.co/<propriétaire>/<dépôt>"}'
```

Un fichier précis peut être choisi en le suffixant : `hf.co/<propriétaire>/<dépôt>:Q4_K_M.gguf`.
Sans précision, le premier GGUF non-projecteur du dépôt est retenu, et un éventuel `mmproj` est
associé automatiquement — le modèle devient alors capable de vision.

### 3.2 Télécharger depuis le registre privé

Avec `OLLAMACPP_REGISTRY_URL` et `OLLAMACPP_REGISTRY_TOKEN` configurés :

```bash
curl http://localhost:11434/api/pull -d '{"model":"<nom-dans-le-registre>"}'
```

Le registre doit fournir un **checksum** pour chaque artefact. Sans checksum, le téléchargement
est refusé : rien ne distinguerait alors un artefact légitime d'un artefact substitué.

Le protocole complet qu'un registre doit implémenter — résolution, format des artefacts, codes de
réponse, règles de sécurité, exemple minimal — est décrit dans **`docs/REGISTRY.md`**.

### 3.3 Installer un GGUF déjà présent sur la machine

```bash
DIGEST="sha256:$(sha256sum modele.gguf | cut -d' ' -f1)"
curl -X POST --data-binary @modele.gguf "http://localhost:11434/api/blobs/$DIGEST"
curl http://localhost:11434/api/create -d "{
  \"model\": \"mon-modele:v1\",
  \"files\": {\"modele.gguf\": \"$DIGEST\"},
  \"stream\": false
}"
```

C'est le chemin natif d'Ollama. `/api/pull` n'accepte **pas** un chemin de fichier : ce n'est pas
un nom de modèle valide.

### 3.4 Renommer, dupliquer, supprimer

```bash
curl http://localhost:11434/api/copy -d '{"source":"mon-modele:v1","destination":"prod:latest"}'
curl -X DELETE http://localhost:11434/api/delete -d '{"model":"mon-modele:v1"}'
```

Une copie ne duplique **aucun octet** : le stockage est adressé par contenu, les deux noms
partagent les mêmes fichiers. Supprimer l'un ne casse donc jamais l'autre.

---

## 4. Régler un modèle

C'est l'apport principal de `ollama.cpp` : chaque modèle porte sa propre configuration
`llama-server`, dans son **manifest**.

Le manifest se trouve sous :

```
$OLLAMACPP_MODELS/manifests/<hôte>/<namespace>/<modèle>/<tag>
```

Section `runtime` (extrait) :

```json
{
  "runtime": {
    "context": 32768,
    "parallel": 4,
    "flash_attention": true,
    "cache_type_k": "iq4_nl",
    "cache_type_v": "q8_0",
    "gpu_layers": 99,
    "tensor_split": "0.6,0.4"
  },
  "lifecycle": { "keep_alive": "30m", "priority": 100 }
}
```

Après modification, décharger le modèle pour que la nouvelle configuration s'applique :

```bash
curl http://localhost:11434/api/chat -d '{"model":"mon-modele:v1","messages":[],"keep_alive":0}'
```

La table **complète** des réglages, avec le drapeau `llama-server` correspondant pour chacun,
est dans **`docs/MODEL_CONFIG.md`**. Les plus utiles au quotidien :

### 4.1 Réglages les plus utiles

| Réglage | Effet | Quand l'utiliser |
|---|---|---|
| `context` | fenêtre de contexte | le principal consommateur de mémoire |
| `cache_type_k` / `cache_type_v` | quantisation du cache KV | diviser la mémoire du cache par deux ou plus, à contexte égal |
| `flash_attention` | attention optimisée | presque toujours bénéfique quand le matériel le permet |
| `parallel` | requêtes simultanées par instance | multiplie la mémoire du cache d'autant |
| `gpu_layers` | couches déportées sur GPU | `0` force le CPU |
| `tensor_split` | répartition multi-GPU | machines à plusieurs cartes |
| `priority` | résistance à l'éviction | protéger un modèle critique |

### 4.2 Priorité de configuration

```
option de la requête  >  manifest  >  métadonnées GGUF  >  défauts du service
```

Une requête peut donc surcharger le contexte via `options.num_ctx` — c'est ce que fait
`ollama-gateway` pour plafonner une clé. Le modèle est alors **rechargé** avec le nouveau
contexte, car ce réglage s'applique au chargement et non à la génération.

---

## 5. Comprendre la résidence en mémoire

### 5.1 `keep_alive`

| Valeur envoyée | Effet |
|---|---|
| absente | 5 minutes (ou la valeur d'`OLLAMACPP_KEEP_ALIVE`) |
| `0` | déchargement dès la fin de la requête |
| `300` | 300 **secondes** |
| `"10m"`, `"1h30m"` | durée |
| `-1` | résidence illimitée |

Décharger un modèle immédiatement :

```bash
curl http://localhost:11434/api/chat -d '{"model":"demo","messages":[],"keep_alive":0}'
```

### 5.2 Ce que fait l'ordonnanceur

Quand un modèle est demandé :

1. s'il est déjà chargé → réutilisé ;
2. s'il y a de la place → chargé ;
3. sinon → un modèle est évincé, choisi dans cet ordre : `keep_alive` expiré d'abord, puis
   priorité la plus basse, puis le plus anciennement utilisé.

**Un modèle qui sert une requête n'est jamais évincé.** Si tous les modèles résidents sont
occupés et qu'il n'y a pas assez de mémoire, la requête échoue avec un message explicite plutôt
que d'interrompre une génération en cours.

### 5.3 Voir l'état réel

```bash
curl http://localhost:11434/api/ps
```

`size_vram` est l'empreinte estimée, `context_length` le contexte réellement alloué, `expires_at`
l'instant de déchargement prévu.

---

## 6. Diagnostic

### 6.1 Lire les journaux

Les événements sont structurés en `clé=valeur`, donc greppables :

```
event=load_requested model=qwen3:8b keep_alive=300s
event=load_started model=qwen3:8b port=18000 args="--model … --cache-type-k iq4_nl …"
event=load_complete model=qwen3:8b port=18000 pid=1234 load_seconds=8.412
event=eviction model=autre:latest action=evict reason=memory_pressure idle_seconds=731 priority=50
event=keep_alive_expired model=demo:latest
```

`args` montre la ligne de commande exacte transmise à `llama-server` : c'est le premier endroit
où regarder quand un modèle se comporte autrement que prévu.

### 6.2 Problèmes courants

| Symptôme | Cause probable | Action |
|---|---|---|
| `model '<nom>' not found` | modèle non installé, ou nom mal orthographié | `GET /api/tags` pour voir les noms exacts, tag compris |
| `not enough memory (N bytes required)` | contexte trop grand pour le budget | réduire `context`, quantifier le cache KV, ou augmenter `OLLAMACPP_MEMORY_LIMIT_BYTES` |
| `is missing required artifacts` | un artefact du manifest est absent du magasin | retélécharger le modèle |
| `"<modèle>" does not support tools` / `vision` / `thinking` | capacité réellement absente | `POST /api/show` pour voir les capacités constatées |
| `llama-server exited with code N` suivi d'un message | échec au chargement | le message reprend la fin de la sortie d'erreur de `llama-server` |
| `did not become ready within Ns` | modèle très gros ou disque lent | augmenter `OLLAMACPP_LOAD_TIMEOUT_S` |
| `model management is disabled` | `OLLAMACPP_MANAGEMENT_ENABLED=false` | comportement attendu en production |

### 6.3 Pourquoi une capacité est-elle absente ?

Les capacités ne sont jamais déclarées : elles sont **constatées**. `vision` exige un projecteur
installé et la modalité active dans le modèle chargé ; `tools` vient des capacités du chat
template. Un manifest peut retirer une capacité, jamais en ajouter une.

```bash
curl http://localhost:11434/api/show -d '{"model":"demo"}' | python3 -m json.tool | head -40
```

---

## 7. Ce que `ollama.cpp` ne fait pas

| Fonction | État | Raison |
|---|---|---|
| Génération d'images | non servie | `llama.cpp` ne génère pas d'images |
| `POST /api/push` | `501` explicite | pas de registre de publication |
| Comptes cloud Ollama, recherche web | non servis | hors périmètre |
| Quantisation à la volée dans `/api/create` | `501` explicite | quantifier le GGUF en amont |
| Clés API par client, quotas, suivi d'usage | non servis | responsabilité d'`ollama-gateway` |

Ces endpoints répondent un **code et un message explicites**, jamais un 404 muet : un outil qui
les sonde apprend qu'ils sont refusés, pas qu'ils sont absents.
