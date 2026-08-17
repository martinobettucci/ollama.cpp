# Journal de `ollama.cpp`

Trace chronologique des décisions et investigations significatives. Une entrée par décision
structurante : problème, hypothèses, observations, options, décision, conséquences, vérifications.

---

## 2026-08-17 — Audit des dépôts amont

**Problème.** Construire un middleware compatible Ollama adossé à `llama-server` sans savoir
précisément ce que `llama-server` fait déjà, ce qu'Ollama garantit, et ce dont `ollama-gateway`
dépend réellement.

**Méthode.** Lecture directe des trois dépôts, montés en lecture seule. Aucune déduction à partir
des noms de fichiers. Révisions : `llama.cpp` `39be55c`, `ollama` `d67ad83`, `ollama-gateway`
`e26fe13`.

**Observations.**

1. `llama-server` expose **déjà nativement** les trois familles d'API demandées :
   `/v1/chat/completions`, `/v1/responses` et `/v1/messages` (+ `/v1/messages/count_tokens`).
   Relevé dans `tools/server/server.cpp` l. 243-267.
2. `llama-server` possède **déjà un routeur multi-modèles** (`tools/server/server-models.cpp`,
   2 507 lignes) : un processus fils par modèle, proxy HTTP, machine à états, LRU, SSE,
   single-flight via `ensure_model_ready()`.
3. `/props` expose `modalities`, `chat_template_caps`, `n_ctx`, `chat_template`, `model_path` :
   de quoi dériver les capacités de **faits observables**.
4. `/v1/models` de `llama-server` renvoie des bouchons pour `size` et `digest`
   (`server-context.cpp` l. 4888-4889, commentaire explicite dans le code).
5. Ollama fait lui-même converger ses façades : `/v1/chat/completions`, `/v1/responses` et
   `/v1/messages` sont des middlewares de traduction branchés sur `ChatHandler`
   (`server/routes.go` l. 1898-1908).
6. `ollama-gateway` dépend de comportements non évidents, en particulier `_is_served`
   (`app/servers.py` l. 301-312) et l'injection de `options.num_ctx`
   (`app/context.py` l. 198-222).

**Conséquences.** Le périmètre réel de `ollama.cpp` est plus étroit et plus net que prévu :
l'API native Ollama et **tout le plan de contrôle des modèles** (identité, manifests, digests,
téléchargement, `keep_alive`, résidence mémoire). L'inférence et les trois façades avancées
existent déjà en amont.

**Vérifications.** Chaque constat est référencé par fichier et par ligne dans
`docs/ollama.cpp-architecture.md` §1 à §3. Les noms de flags CLI cités ont été vérifiés un par un
dans `common/arg.cpp`.

---

## 2026-08-17 — Deux contraintes dures d'`ollama-gateway`

**Problème.** Une compatibilité « mêmes URLs » ne suffit pas : la passerelle a des attentes
comportementales qui casseraient silencieusement.

**Observations.**

1. **Sonde de compatibilité.** `_is_served()` considère un endpoint comme **absent** s'il répond
   404 sans le mot `model` dans le corps. La sonde envoie `{}`. Un `422` de validation FastAPI ne
   pose pas problème (≠ 404), mais un 404 générique de routeur ou un `{"detail":"Not Found"}`
   Starlette ferait apparaître l'endpoint comme non servi dans la matrice de la passerelle.
2. **Injection de `num_ctx`.** La passerelle **réécrit** le corps des requêtes `/api/chat`,
   `/api/generate`, `/api/embed`, `/api/embeddings` pour forcer `options.num_ctx` au plafond de la
   clé. Un refus de ce champ casserait toutes les clés à plafond de contexte.

**Décision.** Ces deux points deviennent des unités de backlog à part entière (OC-011, OC-047) et
des risques suivis (R1, R2), avec un test de conformité dédié qui rejoue la logique exacte de
`_is_served` sur chaque endpoint.

**Conséquence.** Le gestionnaire d'erreurs global de `ollama.cpp` ne peut pas être celui de
FastAPI par défaut : il doit produire le schéma Ollama sur toutes les façades.

---

## 2026-08-17 — Choix de la stack

**Problème.** `ollama.cpp` porte `.cpp` dans son nom mais ne doit pas être un fork de `llama.cpp`
(mission §33). Quel langage et quel mode d'intégration ?

**Options envisagées.**

| Option | Avantages | Inconvénients |
|---|---|---|
| Patch dans `tools/server` de `llama.cpp` | pas de saut réseau, accès direct aux structures | fork intrusif, rebase permanent sur l'upstream, contraire à la mission §33 |
| Service Go | binaire statique, écosystème d'Ollama, réutilisation possible des types `api/` | second écosystème à maintenir face à `ollama-gateway`, cycle de test plus lourd ici |
| Service Python / FastAPI | même stack qu'`ollama-gateway`, cycle de test rapide, CLAUDE.md §3 | pas de binaire unique, surcoût runtime |

**Décision.** Service Python 3.11 / FastAPI / httpx, processus séparé, pilotant le binaire
`llama-server` par CLI et par HTTP. Versions de dépendances **alignées sur celles
d'`ollama-gateway`** pour éviter deux écosystèmes divergents dans la même chaîne.

**Justification.** Le travail de `ollama.cpp` est de la traduction JSON, de la supervision de
processus et de l'ordonnancement — pas du calcul. Le surcoût CPU est marginal devant l'inférence.
Le bénéfice — zéro patch sur `llama.cpp`, mise à jour du backend sans rebase — est exactement
l'objectif §33.

**Compromis assumé.** Un saut réseau supplémentaire en loopback, et pas de binaire statique unique
comme Ollama. Mitigé par la conteneurisation.

---

## 2026-08-17 — Convergence des façades

**Problème.** `llama-server` implémente déjà nativement Responses et Messages. Faut-il les
proxifier telles quelles, ou les faire passer par une représentation canonique ?

**Options.**

1. **Passe-plat** vers `/v1/responses` et `/v1/messages` de `llama-server`. Fidélité maximale à
   chaque spécification, zéro réimplémentation. Mais chaque façade devient indépendante : aucune
   garantie que les quatre chemins traitent le tool calling de la même façon, et le §35 de la
   mission (« les quatre chemins doivent produire la même représentation interne ») devient
   invérifiable.
2. **Canonique → `/v1/chat/completions`.** Une seule conversion par façade — jamais une chaîne
   façade → façade. Un seul sérialiseur backend à maintenir et à tester. La cohérence
   inter-façades devient une propriété testable.

**Décision.** Option 2. La mission interdit explicitement les chaînes de conversion
(`Messages → OpenAI → Ollama → llama.cpp`) ; une conversion unique vers une représentation
canonique n'est pas une chaîne, c'est le contraire.

**Conséquence.** Les invariants du §5.4 de l'architecture (préservation de `call_id`, un
`function_call_output` n'est jamais un message `user`, blocs de raisonnement non fusionnés, ordre
des arguments préservé) deviennent des propriétés vérifiables par test, sur les quatre façades à
la fois.

**Compromis assumé.** Les implémentations natives Responses et Messages de `llama-server` ne sont
pas empruntées. En contrepartie, la cohérence du tool calling est garantie et testée.

---

## 2026-08-17 — Un processus `llama-server` par modèle logique

**Problème.** Faut-il utiliser le mode routeur intégré de `llama-server` (`--models-dir`,
`--models-max`) ou superviser les instances directement ?

**Observations.** Le mode routeur choisit ses modèles par presets, `models-dir` ou cache. Il n'a
pas de notion de manifest, ni de `keep_alive` par requête, ni d'éviction par mémoire avec
priorités. Ses arguments par modèle viennent de presets INI, pas d'un manifest versionné.

**Décision.** Superviser directement des instances `llama-server` mono-modèle, une par modèle
logique, en calquant la machine à états et le pattern de proxy sur ceux de `server-models.cpp`.

**Justification.** L'objectif n° 1 du projet est de **ne pas masquer les capacités de
`llama.cpp`** : `--cache-type-k iq4_nl`, `--cache-type-v q8_0`, `--tensor-split`, `--model-draft`
doivent être configurables **par modèle**. Cela exige de contrôler la ligne de commande de chaque
instance.

**Compromis assumé.** Duplication partielle de la supervision déjà présente en amont. Atténuée en
reprenant explicitement le même pattern, ce qui laisse la porte ouverte à une délégation ultérieure
au mode routeur si celui-ci gagne les crochets manquants.

**Vérification restante.** Le comportement d'offload GPU et l'occupation VRAM ne sont pas
vérifiables dans l'environnement de développement courant (pas de GPU). Suivi en risque R11 :
les décisions dépendantes du GPU sont isolées derrière une abstraction de mesure mémoire, testée
par injection.
