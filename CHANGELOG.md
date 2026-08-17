# Changelog

Toutes les modifications notables de `ollama.cpp`.

## [Non publié]

### Ajouté

- Document d'architecture fondateur `docs/ollama.cpp-architecture.md` : audit de `llama-server`,
  d'Ollama et d'`ollama-gateway`, matrice de compatibilité, fonctionnalités déjà disponibles et
  manquantes, interfaces proposées, plan d'implémentation et risques.
- Dossier d'architecture technique `docs/DAT.md`.
- Backlog `docs/BACKLOG.md` avec des identifiants d'unités stables (`OC-xxx`) servant d'ancrage
  aux commentaires de traçabilité `@spec` / `@verifies`.
- Journal de décisions `docs/JOURNAL.md`.
- `README.md` décrivant l'objectif, la stack, l'installation et les limites connues.
- Noyau du service (OC-010 à OC-014) : configuration centralisée sans secret en dur, schéma
  d'erreur compatible Ollama appliqué globalement, nommage `[host/][namespace/]model[:tag]`,
  sémantique `keep_alive` et durées en nanosecondes, représentation conversationnelle canonique
  vers laquelle convergeront les quatre façades.
- Suite de tests unitaires du noyau, dont la reproduction fidèle de la sonde `_is_served`
  d'`ollama-gateway` et la table de vérité complète de `keep_alive`.
- Stockage et registre (OC-020 à OC-023) : magasin d'artefacts adressé par contenu avec
  installation atomique, vérification de checksum et déduplication ; manifests de modèles
  versionnés dont les artefacts sont des digests et jamais des chemins ; registre donnant à
  chaque modèle une taille et un digest réels, et lecteur de métadonnées GGUF permettant de
  servir `/api/tags` et `/api/show` sans charger le modèle.
- Runtime (OC-030 à OC-035) : superviseur lançant une instance `llama-server` par modèle
  logique avec ses propres drapeaux (`cache_type_k`, `cache_type_v`, Flash Attention,
  `tensor_split`, decoding spéculatif) ; détection de capacités à partir de faits observables
  (`/props`, artefacts présents) ; cycle de vie avec chargement single-flight et sémantique
  `keep_alive` d'Ollama ; ordonnanceur mémoire évinçant en LRU pondéré par priorité sans jamais
  toucher à un modèle occupé ; journal de décisions explicable.
- Faux `llama-server` exécutable pour les tests d'intégration : vrai processus, vrai port, vrai
  HTTP, vrais signaux — seule l'inférence est déterministe.
- Façade Ollama native (OC-040 à OC-047, OC-051 à OC-055) : `/`, `/api/version`, `/api/status`,
  `/api/tags`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate`, `/api/embed`,
  `/api/embeddings`, `/api/copy`, `/api/delete`, `/api/blobs`, `/api/create`, avec streaming
  NDJSON, durées en nanosecondes, `done_reason` et sémantique `keep_alive` complète.
- Pont canonique vers `llama-server` : sérialiseur backend unique partagé par les quatre
  façades, avec correspondance vérifiée du raisonnement, des formats de réponse structurée et
  des mesures.
- Application ASGI assemblant les quatre façades sur un registre et un runtime uniques, avec
  verrou du plan de contrôle et clé d'accès optionnelle appliqués côté serveur.
- Façades OpenAI et Anthropic (OC-070 à OC-076) : `/v1/models`, `/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/messages` et
  `/v1/messages/count_tokens`, chacune convertissant directement vers la représentation
  canonique, avec leurs formats de streaming propres (SSE OpenAI, événements nommés Anthropic
  et Responses).
- Corrélation du nom d'outil : un résultat d'outil sans nom explicite — cas d'Anthropic, dont le
  bloc `tool_result` ne porte que `tool_use_id` — retrouve son nom depuis l'appel corrélé.
- Tests d'équivalence des quatre façades (OC-082) et de boucles de douze appels d'outils
  successifs (OC-083).
- Téléchargement de modèles (OC-050, OC-061, OC-062) : registre privé natif avec jeton, checksum
  obligatoire et installation atomique ; source Hugging Face avec sélection de fichier et
  association automatique du projecteur ; flux de progression au format Ollama.
- Suite de conformité (OC-080, OC-081) rejouant la logique exacte d'`ollama-gateway` — sonde
  `_is_served` sur les quinze endpoints de son catalogue, filtrage des listings et injection de
  `options.num_ctx` — contre le service réel.
- Vérification de bout en bout sur un vrai `llama-server` et un vrai modèle GGUF (OC-085) :
  `scripts/make_test_model.py` produit une architecture `llama` complète et chargeable d'environ
  460 Kio, ce qui rend la suite exécutable partout sans téléchargement. 22 tests couvrent
  l'inférence réelle, le streaming, les embeddings, les quatre façades sur une instance unique,
  le cycle de vie et l'éviction ; 44 tests vérifient que toute ligne de commande produite est
  acceptée par le binaire.
- `docs/REGISTRY.md` : spécification du protocole du registre privé, à destination de qui en
  implémente un.
- `docs/MODEL_CONFIG.md` : référence complète du manifest et des 26 réglages `runtime`, avec le
  drapeau `llama-server` correspondant à chacun.
- Compatibilité vérifiée avec le vrai binaire `ollama` (OC-084) : `list`, `show`, `ps`, `cp`,
  `rm` et `run` fonctionnent sans adaptation contre `ollama.cpp`.
- Prérequis réseau d'un `pull` depuis Hugging Face documenté dans `README.md` et rappelé dans
  `.env.example` : l'API et le stockage des fichiers sont deux domaines distincts, autoriser
  `huggingface.co` seul laisse la résolution réussir puis le téléchargement échouer.
- Vérification de bout en bout contre le **vrai Hugging Face** (OC-061) :
  `tests/test_e2e_huggingface.py` tire `Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m`, contrôle la
  taille à l'octet près et le digest recalculé, puis charge le modèle et vérifie une réponse
  **juste**, un appel d'outil réel et la boucle complète sur son résultat. Activé par
  `OLLAMACPP_TEST_HF_PULL=1`.

### Corrigé

- Le nombre de paramètres n'était lu que dans `general.parameter_count`, clé absente de beaucoup
  de GGUF publiés — dont ceux de Qwen. `ollama show` affichait alors une ligne « parameters »
  vide. Il est désormais **calculé** en additionnant les éléments de la table des tenseurs, comme
  le fait Ollama, et écrit dans les métadonnées.
- `/api/ps` annonçait `size_vram = size` en toutes circonstances, ce dont le CLI Ollama déduisait
  « 100% GPU » même sur un serveur calculant intégralement sur CPU. La part en VRAM est désormais
  déduite des couches réellement déportées.
- Conteneurisation (OC-090 à OC-094) : `Dockerfile` multi-étapes compilant `llama-server` depuis
  l'upstream à une révision épinglée, fichiers Compose dev/staging/prod, scripts `runDev`,
  `runStaging` et `runProd`, `.env.example` documentant chaque variable, seed de démonstration
  passant par les vraies API, contrat de déploiement et manuel d'exploitation.

## [Publié]

_Rien à publier pour le moment._
