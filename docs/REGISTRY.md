# Registre privé de `ollama.cpp` — spécification du protocole

@spec docs/BACKLOG.md OC-062 « Registre privé natif », OC-050 « /api/pull »
@spec docs/ollama.cpp-architecture.md §5.10 « Arborescence de données », §8 risques R8 et R9
@spec docs/DAT.md §3.3 « Téléchargement », §7 « Sécurité »

Ce document décrit **le contrat qu'un serveur doit implémenter** pour servir de registre privé à
`ollama.cpp`. Il s'adresse à qui écrit ou exploite un tel registre.

Implémentation côté client : `ollamacpp/sources.py`, fonctions `_resolve_private` et
`_fetch_artifact`. Tests : `tests/test_sources.py`, qui exécutent un registre réel.

---

## 1. Pourquoi un registre propre plutôt que le protocole d'Ollama

Ollama distribue ses modèles par un protocole de type OCI. Le reprendre imposerait d'implémenter
un registre OCI complet côté serveur pour distribuer trois fichiers.

`ollama.cpp` retient donc les **propriétés** utiles — déduplication, checksums, artefacts
partagés, installation atomique — qui viennent du *layout de stockage*, et non du protocole. Le
protocole de distribution se réduit à deux verbes HTTP : décrire un modèle, servir un fichier.
Un registre peut donc être un simple serveur de fichiers statiques plus un JSON.

---

## 2. Configuration côté `ollama.cpp`

| Variable | Rôle |
|---|---|
| `OLLAMACPP_REGISTRY_URL` | URL de base du registre. Vide = aucun registre privé configuré. |
| `OLLAMACPP_REGISTRY_TOKEN` | Jeton envoyé en `Authorization: Bearer`. Facultatif. |

Sans `OLLAMACPP_REGISTRY_URL`, `/api/pull` ne résout que les références `hf.co/<propriétaire>/<dépôt>`.

---

## 3. Résolution d'un modèle

### Requête

```
GET {OLLAMACPP_REGISTRY_URL}/v1/models/{nom}
Authorization: Bearer {OLLAMACPP_REGISTRY_TOKEN}      (si un jeton est configuré)
```

`{nom}` est le nom demandé par le client, tel quel. Il n'est ni normalisé ni suffixé d'un tag
avant l'appel : un registre peut donc accepter `qwen3.6-35b` comme `qwen3.6-35b:latest`.

### Réponse attendue — `200`

```json
{
  "artifacts": {
    "model":  { "url": "/blobs/qwen3.6-35b.gguf", "digest": "sha256:5f3a…", "name": "model.gguf" },
    "mmproj": { "url": "/blobs/mmproj.gguf",      "digest": "sha256:9b21…", "name": "mmproj.gguf" }
  }
}
```

| Champ | Obligatoire | Sémantique |
|---|---|---|
| `artifacts` | oui | Objet non vide. Un `artifacts` absent ou vide est une erreur. |
| `artifacts.model` | oui | Les poids du modèle. Sans lui, le pull échoue. |
| `artifacts.mmproj` | non | Projecteur multimodal. Sa présence rend la vision possible. |
| `<artefact>.url` | **oui** | Absolue, ou relative à l'URL de base du registre. |
| `<artefact>.digest` | **oui** | `sha256:` suivi de 64 caractères hexadécimaux **minuscules**. |
| `<artefact>.name` | non | Nom de fichier informatif. Validé, jamais utilisé comme chemin. |

Toute autre clé d'`artifacts` est acceptée et téléchargée, mais seuls `model` et `mmproj` sont
actuellement rattachés au manifest installé.

### Codes de réponse

| Code | Interprétation par `ollama.cpp` |
|---|---|
| `200` | Manifest exploitable |
| `404` | `model '<nom>' not found in the private registry` |
| `401`, `403` | `private registry refused the credentials` |
| autre `4xx`/`5xx`, ou erreur réseau | `unable to reach the private registry for model '<nom>'` |

Aucun de ces messages ne divulgue l'URL du registre ni le jeton.

---

## 4. Téléchargement d'un artefact

```
GET {url résolue}
Authorization: Bearer {OLLAMACPP_REGISTRY_TOKEN}      (même jeton que la résolution)
```

Le client suit les redirections, lit la réponse en flux par blocs d'un mégaoctet, et utilise
`Content-Length` — quand il est présent — pour renseigner la progression. Un artefact peut donc
être servi depuis un CDN, à condition que celui-ci accepte l'en-tête d'autorisation ou que l'URL
soit signée.

### Vérification et installation

1. Le SHA-256 est calculé **pendant** le téléchargement, jamais par une relecture.
2. Le fichier est écrit dans `tmp/<uuid>.partial`.
3. Si le digest calculé diffère du `digest` annoncé, le fichier temporaire est **détruit** et le
   pull échoue avec `checksum verification failed for <rôle>`.
4. Sinon, `os.replace` installe le blob dans `blobs/sha256-<hex>` — une opération atomique.
5. Le nom du fichier installé est dérivé du digest **calculé localement**. Aucun nom fourni par
   le registre ne devient un chemin (risque R8).

### Déduplication

Avant de télécharger, `ollama.cpp` regarde si le `digest` annoncé est déjà présent dans son
magasin. Si oui, **rien n'est téléchargé**. Deux conséquences pour un registre :

- publier un modèle sous un nouveau nom sans changer ses poids ne coûte aucun transfert ;
- un `digest` erroné mais correspondant par hasard à un blob existant installerait ce blob : le
  digest est donc l'identité de l'artefact, il doit être exact.

---

## 5. Le checksum est obligatoire, et c'est délibéré

Un artefact sans `digest` est refusé avant tout téléchargement :

```
private registry artifact 'model' has no checksum
```

C'est un choix de sécurité, pas une facilité d'implémentation. Sans checksum fourni **par le
manifest**, un artefact substitué en chemin — CDN compromis, cache empoisonné, erreur
d'exploitation — serait installé et exécuté sans que rien ne le distingue de l'original. Le
digest est la seule information qui lie le nom logique du modèle au contenu réellement attendu.

Un registre doit donc calculer et publier le SHA-256 de chaque fichier qu'il sert.

---

## 6. Sécurité

| Règle | Mise en œuvre |
|---|---|
| Le jeton ne circule que dans `Authorization` | Jamais en paramètre d'URL, jamais dans un corps |
| Le jeton n'est jamais journalisé | Les événements ne portent que le modèle et le rôle d'artefact ; `Config.redacted()` masque le champ. Vérifié par `test_jeton_absent_des_journaux` |
| Le jeton n'apparaît pas dans les erreurs | Vérifié par `test_jeton_absent_des_reponses_derreur` |
| Une URL relative reste dans le registre | `_absolute_url` la joint à l'URL de base |
| Un nom de fichier distant est validé | `[A-Za-z0-9._-]+`, `.` et `..` refusés |
| Aucun chemin distant n'est écrit | Le magasin est adressé par contenu |

**Ce que `ollama.cpp` ne fait pas** : il ne vérifie aucune signature (le digest authentifie le
contenu, pas son émetteur) et ne valide pas de certificat au-delà de la vérification TLS
standard. Un registre exposé publiquement doit donc être servi en HTTPS avec un certificat
valide, et son jeton traité comme un secret.

---

## 7. Exemple minimal de registre

Un registre valide peut tenir en quelques lignes : un JSON par modèle, et des fichiers statiques.

```
registre/
  v1/models/qwen3.6-35b        → JSON de manifest (§3)
  blobs/qwen3.6-35b.gguf       → fichier
  blobs/mmproj.gguf            → fichier
```

Le JSON correspondant :

```json
{
  "artifacts": {
    "model":  { "url": "/blobs/qwen3.6-35b.gguf", "digest": "sha256:<sha256 du fichier>" },
    "mmproj": { "url": "/blobs/mmproj.gguf",      "digest": "sha256:<sha256 du fichier>" }
  }
}
```

Calcul des digests :

```bash
printf 'sha256:%s\n' "$(sha256sum blobs/qwen3.6-35b.gguf | cut -d' ' -f1)"
```

Si le registre exige un jeton, il doit refuser en `401` ou `403` les requêtes sans
`Authorization` valide, **y compris sur les fichiers de blobs** — sinon les poids sont publics
quoi qu'annonce le manifest.

---

## 8. Configuration runtime des modèles distribués

Le registre distribue des **artefacts**, pas de la configuration. Le contexte, les types de cache
K/V, la Flash Attention et le reste sont posés dans le manifest local du modèle après
installation — voir `docs/MODEL_CONFIG.md`.

C'est volontaire : la configuration runtime dépend de la machine qui exécute le modèle — mémoire
disponible, GPU présents, parallélisme voulu —, pas de celle qui le distribue. Un même modèle
tourne avec 262 144 tokens de contexte sur un serveur et 8 192 sur un poste de développement.

---

## 9. Utilisation

```bash
export OLLAMACPP_REGISTRY_URL=https://registre.interne.example
export OLLAMACPP_REGISTRY_TOKEN=…            # jamais committé

curl http://localhost:11434/api/pull -d '{"model":"qwen3.6-35b"}'
```

Le flux de progression suit le format d'Ollama (`status`, `digest`, `total`, `completed`), donc
`ollama pull` et la console d'administration d'`ollama-gateway` affichent une progression réelle.
En `{"stream": false}`, la réponse est `{"status": "success"}` — ce qu'attend précisément
`ollama-gateway` (`app/servers.py::pull_model`).
