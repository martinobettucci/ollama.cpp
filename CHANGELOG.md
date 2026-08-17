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

## [Publié]

_Rien à publier pour le moment._
