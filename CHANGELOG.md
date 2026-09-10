# Changelog

Toutes les modifications notables de `ollama.cpp`.

## [Non publié]

### Corrigé

- **Un modèle qui vient de répondre n'est plus évincé.** Entre deux appels d'outils d'un même
  échange, un modèle est IDLE pour l'ordonnanceur alors que la conversation continue : la façade
  tient une requête HTTP ouverte et s'apprête à relancer. L'évincer coupe ce flux **en plein
  chunk** — le client reçoit un corps incomplet, pas une erreur lisible.

  Relevé en production : `event=eviction model=fast:latest reason=max_loaded_models
  idle_seconds=11.265`, puis `TransferEncodingError: Not enough data to satisfy transfer length
  header` chez le client **15 ms plus tard**.

  Un délai de grâce (`OLLAMACPP_EVICTION_GRACE_S`, 30 s par défaut) rend inévinçable un modèle qui
  a servi récemment. Un `keep_alive` expiré prime : ce modèle devait partir de toute façon.
  L'ordonnanceur reste neutre par défaut (`0.0`) ; c'est la configuration du service qui décide.

  **Limite assumée** : un outil qui attend l'utilisateur laisse le modèle inactif bien plus
  longtemps — deux minutes côté Open WebUI. Couvrir ce cas par le délai bloquerait toute bascule
  de modèle pendant ce temps. La couverture complète suppose que la façade signale qu'un échange
  est en cours, ce que l'ordonnanceur ne peut pas deviner.

- **« Place prise » et « ne tient pas » sont deux refus distincts.** Les deux rendaient
  `not enough memory (N bytes required)`. L'utilisateur lisait que son modèle était trop gros,
  alors qu'il lui suffisait de réessayer — la place était occupée par un modèle en train de
  servir, qu'on refuse d'évincer pour ne pas tuer sa requête. Le refus porte désormais le motif
  `all_residents_busy` et le message le dit : *another model is currently serving requests and
  cannot be evicted — retry in a moment*.

- **Les appels d'outils fragmentés par le streaming sont réassemblés.** En flux, `llama-server`
  découpe `function.arguments` en fragments de tokens répartis sur plusieurs chunks et corrélés
  par `index` — `{`, puis `"id":"`, puis `document`, puis `-word`… Chaque fragment pris isolément
  n'est pas du JSON valide.

  `parse_chunk` les traitait comme des appels **complets** : un appel d'outil était émis par
  fragment, aux arguments inexploitables (`{"_raw": "{"}`). Un agent qui reçoit cela rappelle
  l'outil, reçoit à nouveau des miettes, et **boucle**. Constaté en production sur une seule
  question : **5 422 appels à `ask_user` et 4 167 à `view_skill`**, 918 messages dans la dernière
  requête, aucune réponse produite.

  `chat_stream` accumule désormais les fragments par `index` et ne décode qu'à la clôture du flux
  (`_ToolCallAssembler`). Les appels sortent en un seul delta, complets, tels que le modèle les a
  formés. Correction faite au niveau du backend : les quatre façades — Ollama, OpenAI, Responses,
  Anthropic — en bénéficient ensemble. Le texte, lui, reste streamé token par token.

- **Les tubes de `llama-server` sont drainés en continu.** `stdout` et `stderr` étaient ouverts en
  `PIPE` sans jamais être lus — seul un échec de chargement en consommait 4 Kio. Un tube que
  personne ne vide se remplit (64 Kio sous Linux) et le fils se bloque alors sur son prochain
  `write`, génération comprise. Le symptôme est trompeur : le modèle répond, mais des dizaines de
  fois trop lentement, et uniquement pour les configurations bavardes — un décodage spéculatif
  journalise à chaque brouillon. Mesuré sur Qwen3.8 27B avec DFlash2 : **2,2 tok/s tube plein
  contre 33,9 tok/s drainé**, à configuration identique.

  Deux tâches vident désormais les deux flux dès le démarrage de l'instance et en conservent la
  fin dans un tampon circulaire borné (40 lignes). Le diagnostic d'échec de chargement y puise :
  au moment où l'on veut lire, le fils est mort et ses tubes sont fermés — il n'y aurait plus rien
  à récupérer. Les tâches sont annulées à l'arrêt de l'instance.

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
- Vérification de la vision sur un vrai modèle multimodal : `ggml-org/SmolVLM-256M-Instruct-GGUF`
  est tiré avec son projecteur, et décrit correctement quatre couleurs distinctes sur les quatre
  façades ainsi que depuis le binaire `ollama` officiel (`ollama run modèle "question image.png"`).
- `scripts/make_test_image.py` : générateur d'images PNG de test sans dépendance externe (aplat,
  disque, carré, palette de la charte), pour que la vérification de la vision repose sur une
  observation et non sur un fichier versionné.

### Corrigé

- Le garde-fou de capacités n'existait que sur les façades Ollama et OpenAI. Une image envoyée à
  un modèle sans projecteur traversait les façades Responses et Anthropic jusqu'à `llama-server`,
  dont le refus remontait en `502` accompagné d'un conseil destiné à l'exploitant. Les quatre
  façades partagent désormais `reject_unsupported` et répondent `400 … does not support vision`.
- Le projecteur d'un dépôt Hugging Face était choisi par tri alphabétique, sans rapport avec le
  fichier de poids retenu : demander `:f16` livrait un modèle f16 et un encodeur d'image en Q8_0,
  silencieusement. Le projecteur est maintenant apparié à la quantification du poids, avec repli
  sur le premier lorsqu'aucun ne correspond.
- La sélection d'un fichier précis (`dépôt:motif`) cherchait parmi **tous** les GGUF du dépôt,
  projecteurs compris : un `mmproj` pouvait être retenu comme modèle. La recherche est restreinte
  aux poids.
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
