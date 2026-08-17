# Backlog de `ollama.cpp`

Statuts : `[ ]` non commencé · `[~]` en cours ou implémenté mais insuffisamment vérifié ·
`[x]` terminé et intégralement vérifié.

Une unité ne passe `[x]` qu'après validation de sa Definition of Done (CLAUDE.md §17) : code,
tests unitaires **et** tests d'intégration/E2E propres à l'unité, documentation à jour,
commentaires de traçabilité `@spec` / `@verifies` présents, commit poussé.

Les identifiants `OC-xxx` sont **stables** : ils sont cités par les commentaires `@spec` du code et
les `@verifies` des tests. Ils ne sont jamais réutilisés ni renumérotés.

---

## Lot 0 — Fondations documentaires

- [x] **OC-001** — Audit de `llama-server`, d'Ollama et d'`ollama-gateway`
      *DoD : constats vérifiés par lecture directe, révisions citées.*
      → `docs/ollama.cpp-architecture.md` §1, §2, §3
- [x] **OC-002** — Matrice de compatibilité `ollama-gateway` → `ollama.cpp`
      *DoD : chaque endpoint consommé listé avec ses champs, son streaming et sa criticité.*
      → `docs/ollama.cpp-architecture.md` §3.2
- [x] **OC-003** — Document d'architecture fondateur
      *DoD : architectures amont, matrice, manques, interfaces, plan, risques.*
      → `docs/ollama.cpp-architecture.md`
- [x] **OC-004** — Documentation socle (README, CHANGELOG, DAT, backlog, journal)

## Lot 1 — Noyau

- [~] **OC-010** — Configuration centralisée
      *Variables documentées, valeurs par défaut, validation au démarrage, aucun secret en dur.*
- [~] **OC-011** — Schéma d'erreur compatible Ollama
      *`{"error": "..."}` sur toutes les façades ; jamais de 404 de routeur ni de 422 FastAPI sur
      un endpoint servi (risque R1).*
- [~] **OC-012** — Nommage des modèles
      *`[host/][namespace/]model[:tag]`, tag par défaut `latest`, namespace par défaut `library`,
      omission des valeurs par défaut à l'affichage.*
- [~] **OC-013** — Durées et `keep_alive`
      *Nanosecondes en sortie ; entrée : absent = 5 m, nombre = secondes, négatif = infini,
      chaîne = durée Go, `0` = déchargement (risques R3, R5).*
- [~] **OC-014** — Représentation conversationnelle canonique
      *Messages, tool calls, tool results, images, blocs de raisonnement ; invariants
      `docs/ollama.cpp-architecture.md` §5.4.*

## Lot 2 — Registre et stockage

- [~] **OC-020** — `BlobStore` adressé par contenu
      *Ingestion atomique, vérification de checksum, déduplication, refus des chemins non
      canoniques (risque R8).*
- [~] **OC-021** — Manifests de modèles
      *Schéma versionné, lecture/écriture, validation, `capabilities_override` restrictif.*
- [~] **OC-022** — `ModelRegistry`
      *Résolution de nom, `list`, `get`, `install`, `copy`, `delete`, `size` et `digest` réels
      (risque R4).*
- [~] **OC-023** — Métadonnées GGUF
      *Lecture de l'en-tête GGUF : architecture, quantisation, contexte, taille de paramètres.*

## Lot 3 — Runtime

- [~] **OC-030** — `LlamaServerSupervisor`
      *Lancement d'une instance par modèle logique, allocation de port, attente de `/health`,
      arrêt propre, remontée du code de sortie.*
- [~] **OC-031** — Construction des arguments runtime
      *Contexte, batch, ubatch, parallel, threads, GPU layers, tensor split, Flash Attention,
      `cache_type_k` / `cache_type_v`, draft model, mmproj, adaptateurs.*
- [~] **OC-032** — Détection de capacités observables
      *Depuis `/props` (`modalities`, `chat_template_caps`), le manifest et les artefacts
      présents. Jamais de capacité déclarée non observable.*
- [~] **OC-033** — `ModelLifecycleManager`
      *États `NOT_PRESENT` → `FAILED`, single-flight, refus de `READY` si un artefact obligatoire
      manque.*
- [~] **OC-034** — `ModelScheduler`
      *Admission, estimation mémoire, éviction LRU pondérée par priorité, `keep_alive` expiré
      privilégié, `BUSY` jamais évincé (risque R6).*
- [~] **OC-035** — Observabilité des décisions
      *Journal structuré et explicable de chaque décision du scheduler et de chaque transition.*

## Lot 4 — Façade Ollama

- [~] **OC-040** — `GET /api/version`, `GET /`, `GET /api/status`
- [~] **OC-041** — `GET /api/tags`
      *`models[].name` **et** `models[].model`, `size` et `digest` réels, `details`,
      `capabilities` (criticité P0 de la matrice).*
- [~] **OC-042** — `POST /api/show`
      *`details`, `model_info`, `template`, `system`, `parameters`, `capabilities`, `modified_at`.*
- [~] **OC-043** — `GET /api/ps`
      *État réel du lifecycle : `expires_at`, `size_vram`, `context_length`. Jamais simulé.*
- [~] **OC-044** — `POST /api/chat`
      *Messages, tools, tool results, images, `format`, `options`, `stream` NDJSON, `keep_alive`,
      `think`, `done_reason`, métriques.*
- [~] **OC-045** — `POST /api/generate`
      *`prompt`, `system`, `template`, `raw`, `format`, `options`, `stream`, `keep_alive`,
      `images`, `think`, `context`.*
- [~] **OC-046** — `POST /api/embed` et `POST /api/embeddings`
      *Formes plurielle et legacy singulière.*
- [~] **OC-047** — `options.num_ctx` honoré sur les 4 chemins natifs
      *Contrainte dure d'`ollama-gateway` (risque R2).*

## Lot 5 — Plan de contrôle

- [~] **OC-050** — `POST /api/pull`
      *Flux de progression NDJSON, statut final `success`, mode `stream:false` attendu par la
      console d'administration d'`ollama-gateway`.*
- [~] **OC-051** — `DELETE /api/delete`
- [~] **OC-052** — `POST /api/copy`
- [~] **OC-053** — `POST /api/create`
      *Sous-ensemble : `from`, `files`, `template`, `system`, `parameters`, `license`.*
- [~] **OC-054** — `POST` et `HEAD /api/blobs/:digest`
- [~] **OC-055** — Endpoints hors périmètre explicites
      *`/api/push` et les endpoints cloud répondent un code et un message clairs, jamais un 404
      de routeur.*

## Lot 6 — Sources de modèles

- [x] **OC-060** — Source système de fichiers
      *Décision : un chemin local n'est pas un nom de modèle Ollama valide et Ollama ne l'accepte
      pas non plus sur `/api/pull`. L'installation d'un GGUF local passe donc par le chemin natif
      `POST /api/blobs/<digest>` + `POST /api/create` (OC-053, OC-054), vérifié de bout en bout.*
- [~] **OC-061** — Source Hugging Face
- [~] **OC-062** — Registre privé natif
      *URL, jeton, en-tête `Authorization`, checksums, cache local, installation atomique,
      reprise de téléchargement ; aucun secret journalisé (risque R9).*

## Lot 7 — Façades OpenAI et Anthropic

- [~] **OC-070** — `GET /v1/models` et `GET /v1/models/:model`
      *`data[].id` (contrainte de filtrage d'`ollama-gateway`).*
- [~] **OC-071** — `POST /v1/chat/completions`
      *Streaming SSE, tools, `tool_choice`, vision, `response_format`, `usage`.*
- [~] **OC-072** — `POST /v1/completions`
- [~] **OC-073** — `POST /v1/embeddings`
- [~] **OC-074** — `POST /v1/responses`
      *`input`/`output`, `reasoning`, `function_call`, `function_call_output`, `call_id`, `tools`,
      événements de streaming, `usage`. Un `function_call_output` ne devient jamais un message
      `user` (risque R7).*
- [~] **OC-075** — `POST /v1/messages`
      *Blocs de contenu, `tool_use`, `tool_result`, images, `max_tokens`, streaming, `usage`.*
- [~] **OC-076** — `POST /v1/messages/count_tokens`

## Lot 8 — Vérification

- [~] **OC-080** — Tests de conformité Ollama sur fixtures
      *Statut, schéma JSON, champs, types, chunks de flux, erreurs.*
- [~] **OC-081** — Test de la sonde `_is_served` d'`ollama-gateway`
      *Chaque endpoint POST, corps `{}` : code ≠ 404, ou 404 contenant le mot `model` (risque R1).*
- [~] **OC-082** — Équivalence des quatre façades
      *Le même échange conceptuel produit un `CanonicalRequest` structurellement égal.*
- [~] **OC-083** — Multi-tours ≥ 10 appels d'outils
      *Rôles, identifiants d'appels, résultats, raisonnement, streaming, intention initiale.*
- [ ] **OC-084** — Compatibilité du CLI Ollama
      *`list`, `show`, `ps`, `pull`, `cp`, `rm`, `run` contre `ollama.cpp`.*
- [ ] **OC-085** — Bout en bout avec un vrai `llama-server`
      *Modèle minuscule, sans GPU, sans service payant.*

## Lot 9 — Exploitation

- [ ] **OC-090** — Conteneurisation dev / staging / prod
- [ ] **OC-091** — `runDev`, `runStaging`, `runProd`
- [ ] **OC-092** — Seed de démonstration reproductible
- [ ] **OC-093** — Contrat de déploiement (`docs/PROD_MIGRATIONS.md`)
- [ ] **OC-094** — Manuel utilisateur

---

## Hors périmètre (décidé, documenté, non planifié)

Justifications dans `docs/ollama.cpp-architecture.md` §5.3.

- `POST /api/push` — pas de registre de publication.
- Endpoints de compte cloud Ollama (`/api/me`, `/api/signout`, `/api/user/keys/*`).
- `/api/experimental/web_search`, `/api/experimental/web_fetch`.
- Génération d'images (`x/…`, `/v1/images/generations`) — `llama.cpp` ne génère pas d'images.
- Modèles distants fédérés (`remote_host`, `remote_model`).
- Clés API, quotas, contrôle d'accès aux modèles — responsabilité d'`ollama-gateway`.
