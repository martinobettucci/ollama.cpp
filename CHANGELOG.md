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

## [Publié]

_Rien à publier pour le moment._
