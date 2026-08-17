# Contrat de déploiement de `ollama.cpp`

@spec docs/BACKLOG.md OC-093 « Contrat de déploiement »
@spec docs/DAT.md §8 « Stratégie de déploiement », §9 « Reprise »

Ce document décrit **ce qu'un humain doit appliquer** pour déployer ou mettre à jour
`ollama.cpp`. Il ne doit jamais dériver de l'état réel du projet : toute modification du schéma
de stockage, des variables d'environnement ou des services déployés le met à jour dans le même
changement.

---

## 1. Baseline de production

**Aucune.** `ollama.cpp` n'a jamais été déployé en production à ce jour. La section
« [Publié] » du `CHANGELOG.md` est vide, et le backlog ne comporte aucune unité `[x]` en dehors
des fondations documentaires.

Le premier déploiement est donc une **installation initiale**, pas une migration.

---

## 2. Opérations en attente pour le premier déploiement

| # | Opération | Obligatoire | Notes |
|---|---|---|---|
| 1 | Construire l'image (`docker compose -f docker-compose.prod.yml build`) | oui | compile `llama-server` depuis l'upstream à la révision épinglée par `LLAMA_CPP_REF` |
| 2 | Créer `.env.prod` depuis `.env.example` | oui | aucun secret n'est fourni par défaut |
| 3 | Provisionner le volume de modèles | oui | prévoir la taille cumulée des GGUF, pas celle d'un seul |
| 4 | Installer les modèles | oui | voir §4 |
| 5 | Vérifier la sonde de disponibilité | oui | voir §5 |
| 6 | Basculer la cible dans `ollama-gateway` | oui | voir §6 |

### 2.1 Il n'y a pas de migration de schéma

`ollama.cpp` n'utilise **aucune base de données**. Son état persistant se limite à une
arborescence de fichiers :

```
$OLLAMACPP_MODELS/
  blobs/sha256-<hex>
  manifests/<host>/<namespace>/<model>/<tag>
  tmp/
```

Le registre est reconstruit depuis le disque à chaque démarrage. Il n'y a donc ni migration à
ordonner, ni verrou de migration, ni risque de schéma désynchronisé. Le manifest porte un champ
`schema_version` : une version inconnue fait **échouer la lecture de ce modèle** plutôt que de
l'interpréter de travers ; les autres modèles restent servis.

---

## 3. Variables d'environnement à définir

Toutes sont documentées dans `.env.example`, avec leur rôle, leur format et leur caractère
obligatoire ou non. Les variables **structurantes en production** :

| Variable | Valeur recommandée en production | Motif |
|---|---|---|
| `OLLAMACPP_MODELS` | `/models` (volume persistant) | sans persistance, chaque redémarrage retélécharge tout |
| `OLLAMACPP_MANAGEMENT_ENABLED` | `false` | le catalogue se gère depuis la console d'administration, jamais depuis le flux d'inférence |
| `OLLAMACPP_MEMORY_LIMIT_BYTES` | valeur explicite | sur une machine partagée, la déduction automatique surestime ce qui est réellement disponible |
| `OLLAMACPP_MAX_LOADED_MODELS` | selon la mémoire | borne haute complémentaire du budget mémoire |
| `OLLAMACPP_KEEP_ALIVE` | selon l'usage | une valeur trop longue immobilise la mémoire, trop courte fait payer des rechargements |
| `OLLAMACPP_API_KEY` | vide derrière la passerelle | `ollama-gateway` porte déjà les clés par client |
| `OLLAMACPP_REGISTRY_TOKEN` | secret, hors dépôt | jamais journalisé (vérifié par test) |

**Aucun secret ne doit être écrit dans un fichier versionné.** `.env.prod` et `.env.staging` sont
exclus du dépôt.

---

## 4. Installation des modèles

Deux voies, selon la provenance.

**Depuis un registre (privé ou Hugging Face)** — nécessite `OLLAMACPP_MANAGEMENT_ENABLED=true`
le temps de l'opération, ou une exécution depuis le réseau d'administration :

```
POST /api/pull   {"model": "hf.co/<propriétaire>/<dépôt>", "stream": false}
POST /api/pull   {"model": "<nom-dans-le-registre-privé>", "stream": false}
```

**Depuis un GGUF local** — chemin natif d'Ollama, sans accès réseau :

```
POST /api/blobs/sha256:<digest>     (corps = le fichier)
POST /api/create  {"model": "<nom>", "files": {"modele.gguf": "sha256:<digest>"}, "stream": false}
```

Le script `scripts/seed.py` automatise la première voie pour un environnement de démonstration.

**Configuration runtime par modèle.** C'est l'apport principal du projet : contexte, Flash
Attention, types de cache K/V, `tensor-split`, modèle de brouillon. Elle vit dans le manifest du
modèle (`docs/ollama.cpp-architecture.md` §5.5), pas dans les variables d'environnement.

---

## 5. Vérifications après déploiement

À exécuter dans l'ordre ; chacune doit passer avant la suivante.

| # | Vérification | Attendu |
|---|---|---|
| 1 | `GET /api/version` | `200`, corps `{"version": "..."}` |
| 2 | `GET /` | `200`, texte `Ollama is running` |
| 3 | `GET /api/tags` | `200`, `models[]` contenant les modèles installés, avec `name`, `model`, `size` et `digest` non vides |
| 4 | `POST /api/show {"model": "<un modèle>"}` | `200`, `details`, `capabilities` cohérentes avec le modèle |
| 5 | `POST /api/chat` non streamé | `200`, `message.content` non vide, `done_reason: "stop"` |
| 6 | `GET /api/ps` | le modèle apparaît, avec `size_vram`, `context_length` et `expires_at` |
| 7 | Attendre `keep_alive` + 10 s, puis `GET /api/ps` | le modèle a disparu — preuve que le déchargement automatique fonctionne |
| 8 | `POST /v1/chat/completions` | `200`, forme OpenAI |
| 9 | `POST /v1/messages` (avec `max_tokens`) | `200`, forme Anthropic |
| 10 | Journaux | présence d'`event=load_complete`, absence de tout jeton |

---

## 6. Bascule depuis une cible Ollama

Dans `ollama-gateway`, un serveur d'exécution est décrit par une URL de base. La bascule consiste
à remplacer l'URL de l'Ollama existant par celle de `ollama.cpp`.

**Aucune modification du code d'`ollama-gateway` n'est nécessaire** : la matrice de compatibilité
(`docs/ollama.cpp-architecture.md` §3.2) et la suite `tests/test_conformance_gateway.py`
vérifient les dix-sept comportements dont la passerelle dépend.

Procédure recommandée :

1. déclarer `ollama.cpp` comme **serveur supplémentaire**, sans y rattacher de clé ;
2. lancer le test de disponibilité de la console : il doit trouver le serveur en ligne et lister
   ses modèles ;
3. lancer le test de compatibilité d'API : les familles `ollama`, `openai` et `anthropic` doivent
   apparaître servies. **Les familles `ollama-image` et `openai-image` resteront non servies** —
   `llama.cpp` ne génère pas d'images, incompatibilité assumée et documentée ;
4. utiliser « Essayer maintenant » sur chaque famille ;
5. basculer **une seule clé** de test vers le nouveau serveur, et observer les journaux d'usage ;
6. basculer les autres clés une fois le comportement confirmé.

**Retour arrière** : réaffecter les clés au serveur Ollama d'origine. Aucune donnée n'est
transformée par la bascule, et `ollama.cpp` ne modifie rien chez la passerelle : le retour est
immédiat et sans perte.

---

## 7. Mise à jour de `llama.cpp`

Le backend est compilé depuis l'upstream à la révision épinglée par `LLAMA_CPP_REF` dans le
`Dockerfile`. Pour le mettre à jour :

1. changer `LLAMA_CPP_REF` ;
2. reconstruire l'image ;
3. exécuter `pytest -m e2e` avec `OLLAMACPP_TEST_LLAMA_SERVER` pointant sur le nouveau binaire —
   cette suite vérifie que **tous les drapeaux** émis par le middleware sont encore acceptés et
   que les endpoints HTTP consommés existent toujours (risque R10) ;
4. déployer seulement si cette suite passe.

Aucun patch n'est appliqué à `llama.cpp` : une mise à jour ne demande jamais de rebase.

---

## 8. Reprise et arrêt

- **Arrêt** : le service termine ses instances `llama-server` à l'arrêt. `stop_grace_period` est
  fixé à 60 s en production pour laisser le temps aux requêtes en cours. Un arrêt brutal peut
  laisser des processus orphelins qui retiendraient de la VRAM.
- **Redémarrage** : sans état à restaurer. Le registre est relu depuis le disque et les modèles
  sont rechargés à la première requête. Les téléchargements interrompus (`tmp/*.partial`) sont
  collectés au démarrage.
- **Perte du volume de modèles** : aucune donnée métier n'est perdue — seulement les artefacts,
  retéléchargeables. Le coût est le temps de téléchargement.

---

## 9. Risques connus au déploiement

| Risque | Effet | Atténuation |
|---|---|---|
| Budget mémoire mal calibré | refus de chargement, ou pression sur l'hôte | fixer `OLLAMACPP_MEMORY_LIMIT_BYTES` explicitement et surveiller `event=eviction` |
| Volume de modèles trop petit | échec de `pull` en cours de téléchargement | dimensionner sur la somme des GGUF, pas sur le plus gros |
| `OLLAMACPP_MANAGEMENT_ENABLED=true` en production | un client pourrait muter le catalogue s'il atteignait le service | laisser à `false` ; le port n'est de toute façon publié que sur la boucle locale |
| Génération d'images attendue | deux familles restent rouges dans la matrice de la passerelle | incompatibilité documentée : conserver un Ollama pour ces modèles |
| Mise à jour de `llama.cpp` non testée | drapeau disparu, instances qui ne démarrent plus | exécuter `pytest -m e2e` avant de déployer (§7) |
