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

---

## 2026-08-17 — Vérification contre le vrai `llama-server`

**Problème.** Toute la chaîne était vérifiée contre un faux `llama-server`. Cela prouve la
cohérence interne du middleware, mais pas que `llama.cpp` accepte réellement ce qu'on lui envoie —
précisément le risque R10.

**Observations.**

1. `llama-server` a été compilé hors source depuis l'arbre amont, à la révision auditée
   (`build 50, commit 39be55c`). L'arbre `llama.cpp` est resté intact, ce qui a été vérifié après
   coup par `git status`.
2. `llama-server` valide ses arguments **avant** de toucher au modèle : un drapeau inconnu produit
   `error: invalid argument: --xxx` et un arrêt immédiat. Atteindre l'erreur de chargement du
   modèle prouve donc que toute la ligne de commande a été acceptée.
3. Le mode routeur (`--models-dir`) démarre **sans charger de modèle**. C'est ce qui a permis de
   vérifier en vrai le contrat HTTP dont dépend le superviseur : `/health` répond exactement
   `{"status": "ok"}` et `/v1/models` la forme OpenAI.

**Décision.** Écrire `tests/test_llama_server_contract.py`, activé par
`OLLAMACPP_TEST_LLAMA_SERVER` et ignoré sinon. Il couvre chaque configuration runtime que le
middleware sait produire, la présence de chaque drapeau dans l'aide, et la surface HTTP réelle.
Deux contre-épreuves y figurent : un drapeau inconnu doit bien être refusé, et l'échec de
chargement doit bien porter sur le fichier de modèle — sans elles, un test qui ne peut pas
échouer ne prouverait rien.

**Vérifications réalisées.** 44 tests passent contre le binaire réel. La ligne de commande la plus
complète que le middleware sache produire — contexte, batch, ubatch, parallélisme, threads,
couches GPU, `tensor-split`, Flash Attention, `cache-type-k`/`v`, `kv-unified`, `no-mmap`,
decoding spéculatif, `reasoning-format` — est acceptée sans réserve.

**Limite, explicitement non résolue.** L'inférence sur un vrai GGUF n'a **pas** pu être vérifiée :
la politique réseau de l'environnement de construction bloque `huggingface.co` (403 sur CONNECT),
donc aucun modèle réel n'était disponible. OC-085 reste `[~]`. Ce qui est prouvé est le contrat
d'interface ; ce qui ne l'est pas est la génération de tokens de bout en bout.

---

## 2026-08-17 — `/api/pull` et les chemins de fichiers locaux

**Problème.** La première implémentation de `/api/pull` acceptait un chemin de fichier local comme
source. Le test de conformité a montré qu'elle renvoyait un 404 : le registre résout d'abord le
nom, et un chemin comme `/tmp/x/modele.gguf` n'est pas un nom de modèle Ollama valide.

**Observation.** Ollama ne l'accepte pas davantage. Son chemin natif pour un GGUF local est
`POST /api/blobs/<digest>` puis `POST /api/create` avec `files` — déjà implémenté et testé ici
(OC-053, OC-054).

**Décision.** Retirer la branche « fichier local » de `pull`, plutôt que d'inventer une règle de
dérivation de nom qui aurait surpris l'utilisateur (quel nom porterait
`/tmp/x/qwen3-8b-instruct.gguf` ?). Le message d'erreur de `pull` énumère désormais les trois
voies possibles.

**Conséquence.** Une capacité en moins, mais alignée sur Ollama et sans comportement surprenant.
OC-060 passe `[x]` avec cette justification explicite, plutôt que de rester une case à moitié
cochée.

---

## 2026-08-17 — Inférence réelle : générer un modèle plutôt que le télécharger

**Problème.** La vérification de bout en bout butait sur l'absence de modèle GGUF : la politique
réseau bloque les CDN de distribution. L'API `huggingface.co` répond, mais tout fichier LFS
redirige vers `us.aws.cdn.hf.co`, qui refuse la connexion en 403. Sans modèle, OC-085 restait à
moitié vérifié.

**Observation.** Le problème n'était pas d'obtenir *ce* modèle-là, mais d'obtenir *un* modèle
valide. Or `llama.cpp` fournit `gguf-py`, qui sait écrire un GGUF, et l'architecture `llama` est
entièrement décrite par ses métadonnées et ses tenseurs. Un modèle aux poids aléatoires se charge
et s'exécute exactement comme un modèle entraîné : seule la qualité du texte diffère.

**Décision.** Écrire `scripts/make_test_model.py`, qui produit une architecture `llama` complète
(2 couches, 64 dimensions, contexte 512, tokenizer SPM, chat template) d'environ 460 Kio. Le
script devient un outil du projet, pas un artifice de test : il rend la suite de bout en bout
exécutable partout, sans téléchargement.

**Deux obstacles rencontrés, et ce qu'ils ont appris.**

1. *Premier essai* : vocabulaire des 256 octets. Le modèle a chargé et généré, mais
   `llama-server` a rejeté la réponse — « output that does not match the expected format ». Les
   poids aléatoires tiraient des octets au hasard, qui ne formaient pas de l'UTF-8 valide.
2. *Deuxième essai* : vocabulaire restreint à l'ASCII. Échec à la tokenisation,
   `unordered_map::at`. Lecture de `llama_vocab::byte_to_token` (`src/llama-vocab.cpp` l. 3900) :
   SPM cherche `<0xXX>`, puis se rabat sur l'octet brut avec un `.at()` qui lève. Or SPM remplace
   les espaces par « ▁ » (U+2581), trois octets non-ASCII — un vocabulaire amputé casse donc tout
   prompt contenant une espace.
3. *Solution* : conserver les 256 octets pour l'entrée, et contraindre la **sortie** en annulant
   les lignes de la projection finale correspondant aux octets non-ASCII. Leur logit vaut alors
   exactement zéro, tandis que celui des tokens autorisés s'étale largement : le maximum est
   toujours pris parmi les tokens émettables. Un test vérifie la propriété au lieu de la
   supposer.

**Vérifications réalisées.** `tests/test_e2e_real_model.py` : 22 tests sur l'application
complète, le vrai binaire `llama-server` et ce vrai modèle. Chargement, `/props`, `/api/chat`
streamé et non streamé, `/api/generate`, embeddings sur une seconde instance lancée avec
`--embedding`, les quatre façades servies par une instance unique, `/api/ps`, `keep_alive: 0`,
rechargement sur `num_ctx`, capacités issues du vrai `/props`, et éviction réelle sous pression —
avec vérification que le processus évincé est bien arrêté.

**Conséquence.** OC-085 passe `[x]`. La limite restante est nommée explicitement : les poids
étant aléatoires, la *qualité* des réponses d'un modèle entraîné n'est pas vérifiée. La chaîne,
elle, l'est intégralement.

---

## 2026-08-17 — Deux manques de documentation signalés par le responsable

**Problème.** Le responsable a relevé deux absences : le protocole du registre privé n'était
décrit que dans une docstring, et la configuration runtime par modèle n'apparaissait que par
fragments dans le manuel.

**Observation.** Les deux sont des contrats destinés à des lecteurs extérieurs au code. Qui écrit
un registre privé a besoin de connaître les codes de réponse, la forme du manifest et le
caractère obligatoire du checksum. Qui exploite un modèle a besoin de la table complète des
réglages et de leur drapeau `llama-server`. Une docstring ne remplit ni l'un ni l'autre rôle :
elle n'est lue que par qui modifie l'implémentation.

**Décision.** Deux documents dédiés.

- `docs/REGISTRY.md` : spécification du protocole — résolution, forme des artefacts, codes de
  réponse, téléchargement et vérification, raison pour laquelle le checksum est obligatoire,
  règles de sécurité, et exemple d'un registre minimal tenant en un JSON et des fichiers
  statiques.
- `docs/MODEL_CONFIG.md` : référence complète du manifest, table exhaustive des 26 champs
  `runtime` avec leur drapeau, drapeaux réservés, drapeaux posés systématiquement, ordre de
  précédence, effet mémoire de chaque réglage, et exemples par cas d'usage.

**Conséquence.** Le manuel et le README renvoient vers ces documents plutôt que d'en dupliquer
des extraits, qui dériveraient.

---

## 2026-08-17 — Domaines à autoriser pour un `pull` réel depuis Hugging Face

**Problème.** Le responsable demande quels domaines ouvrir dans la politique réseau pour qu'un
`pull` depuis Hugging Face aboutisse réellement. Les tentatives précédentes échouaient alors que
`huggingface.co` semblait joignable, ce qui avait été résumé à tort par « Hugging Face est
bloqué ».

**Observation.** Mesure directe, sans supposition :

- `https://huggingface.co/` répond `200` ;
- `https://us.aws.cdn.hf.co/`, `eu.aws.cdn.hf.co`, `cdn-lfs*.huggingface.co`,
  `cas-bridge.xethub.hf.co` et `transfer.xethub.hf.co` échouent tous à la connexion ;
- le mandataire qualifie l'échec sans ambiguïté : `connect_rejected`, « gateway answered 403 to
  CONNECT (policy denial) », pour `us.aws.cdn.hf.co:443`.

La trace de redirection d'un GGUF réel
(`Qwen/Qwen2.5-0.5B-Instruct-GGUF/.../qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491 Mo, et
`ggml-org/models-moved/.../stories260K.gguf`, 1,19 Mo) donne la cause exacte : `huggingface.co`
répond `302` vers `https://us.aws.cdn.hf.co/xet-bridge-us/<id>/<hash>?...&Signature=...`, avec les
en-têtes `x-linked-size` et `x-xet-hash`. Le stockage Xet sert l'octet ; l'API ne fait
qu'indiquer où.

**Conclusion.** L'API des métadonnées et le stockage des fichiers sont deux domaines distincts.
Une politique n'autorisant que `huggingface.co` laisse la résolution réussir puis le
téléchargement échouer — d'où le diagnostic initial erroné. Le domaine bloquant est
`us.aws.cdn.hf.co` ; `eu.aws.cdn.hf.co` et les hôtes `cdn-lfs*` couvrent respectivement l'autre
région et les dépôts non migrés vers Xet. Les hôtes `*.xethub.hf.co` relèvent du client Xet
natif, que `ollama.cpp` n'utilise pas : il télécharge en HTTP simple en suivant les redirections.

**Décision.** Documenter le prérequis réseau plutôt que de le laisser dans une conversation :
tableau des domaines et de leur nécessité dans `README.md`, renvoi depuis `.env.example` à côté
d'`OLLAMACPP_HF_ENDPOINT`, avec une commande de vérification qui observe l'URL finale sans rien
télécharger.

**Conséquence.** Aucun changement de code : `OLLAMACPP_HF_ENDPOINT` ne désigne que l'API et le
client suit déjà les redirections. OC-061 reste `[~]` : le chemin est couvert par des tests
contre un serveur local qui reproduit le contrat HF, mais le `pull` contre le vrai Hugging Face
n'a pas pu être exécuté dans cet environnement, l'hôte de stockage y étant refusé.

---

## 2026-08-17 — Pull réel depuis Hugging Face, et ce qu'un vrai modèle a révélé

**Contexte.** L'hôte de stockage `us.aws.cdn.hf.co` a été autorisé dans la politique réseau. Le
`pull` réel, jusque-là impossible, devient exécutable.

**Observations.**

- `POST /api/pull` sur `hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m` aboutit en 37 s : 475
  événements de progression, 491 400 032 octets, statut final `success`.
- Le digest recalculé localement (`74a4da8c…`) correspond au contenu du blob, et la taille à
  l'octet près à l'en-tête `x-linked-size` annoncé par Hugging Face.
- Le modèle se charge et répond juste : « La capitale de la France est Paris. »
- `capabilities` vaut `["completion", "tools"]`, issues du `/props` du modèle chargé. `vision` est
  correctement absent : le dépôt ne publie aucun projecteur.
- Un appel d'outil réel est émis par le modèle, puis la boucle complète — résultat renvoyé,
  réponse finale exploitant la valeur — fonctionne.

**Défaut trouvé — le compte de paramètres.** `ollama show` affichait une ligne « parameters »
vide. Ce GGUF ne porte pas `general.parameter_count` ; seul `general.size_label` (« 630M »), une
chaîne libre, est renseigné. La lecture d'en-tête s'arrêtait après les paires clé/valeur et ne
lisait jamais la table des tenseurs.

Vérification dans l'amont : Ollama ne lit pas cette clé, il la **calcule**. `fs/ggml/gguf.go`
l. 239-251 additionne les éléments de chaque tenseur (`Tensor.elements()`, `fs/ggml/ggml.go`
l. 523-532) puis écrit le résultat dans les métadonnées, écrasant la clé si elle existait.

**Correction.** `read_metadata` parcourt désormais la table des tenseurs — nom, forme, type,
décalage, soit quelques dizaines d'octets par tenseur ; le blob de poids n'est toujours jamais
touché — et écrit `general.parameter_count`. Bornes reprises de l'amont : `MaxTensorDims = 4`, et
une table tronquée fait échouer la lecture plutôt que de renvoyer un compte partiel, un compte
faux étant affiché comme un fait par `ollama show`.

Six tests écrits **avant** la correction, tous en échec puis tous passants. Résultat sur le vrai
modèle : 630 167 424 paramètres, `parameter_size = "630.17M"`, et `ollama show` affiche enfin
`parameters 630.17M`. Le contrôle croisé est le `size_label` de 630M déclaré indépendamment par
l'éditeur.

**Équivalence des quatre façades (§35), mesurée sur le vrai modèle.** Le même échange
— question, appel d'outil, résultat d'outil, réponse — joué sur les quatre façades donne, à
`temperature: 0`, **la même phrase caractère pour caractère**. Les charges utiles sérialisées vers
`llama-server` sont identiques entre OpenAI Chat, Responses et Anthropic ; la façade Ollama n'en
diffère que par l'identifiant d'appel, que son protocole ne transporte pas.

Une première mesure, à température par défaut, donnait une réponse divergente sur la façade
Responses. La comparaison des charges utiles sérialisées a montré qu'elles étaient identiques :
la divergence venait de l'échantillonnage, pas du code. Consigné parce qu'une conclusion hâtive
aurait fait chercher un défaut inexistant.

**Dix appels d'outils successifs (§36).** Deux limites, toutes deux hors du périmètre de
`ollama.cpp`, mesurées séparément :

1. le modèle de 0,5 milliard de paramètres abandonne la boucle après un appel et conclut en
   texte ;
2. `llama-server` ignore `tool_choice: "required"` sur un tour succédant à un résultat d'outil —
   vérifié en l'interrogeant **directement**, sans `ollama.cpp` dans le chemin.

Ce qui relève d'`ollama.cpp` a donc été vérifié pour lui-même : une conversation portant dix
appels et dix résultats traverse les quatre façades, produit 702 jetons de prompt **identiques**
sur les quatre, et le prompt rendu par `/apply-template` contient les dix blocs `<tool_call>`, les
dix `<tool_response>`, et le témoin posé au premier tour. Rien n'est tronqué, aucun résultat
d'outil n'est requalifié en instruction utilisateur.

**Conséquence.** OC-061 passe `[x]`, couvert par `tests/test_e2e_huggingface.py` (13 tests, réseau
requis, activés par `OLLAMACPP_TEST_HF_PULL=1`). La limite « le modèle de test produit du
charabia » du README est désormais compensée : un vrai modèle entraîné vérifie la justesse des
réponses. La limite `tool_choice` de `llama-server` est ajoutée aux limites connues.

---

## 2026-08-18 — Vision réelle : un projecteur, quatre couleurs, trois défauts

**Contexte.** Le responsable demande de vérifier le `mmproj` sur un petit modèle de vision.
`ggml-org/SmolVLM-256M-Instruct-GGUF` est retenu : 256 M de paramètres, publié par l'équipe de
`llama.cpp`, et — trait décisif pour la suite — il publie **un projecteur par quantification**.

**Observation nominale.** Le `pull` tire les deux artefacts en 19 s : modèle 175 054 528 octets,
projecteur 103 769 856 octets, chacun avec son digest recalculé. `/api/show` annonce
`["completion", "vision"]`.

Mais une capacité annoncée n'est pas une capacité observée. La mission (§13) l'interdit
explicitement. J'ai donc fabriqué des images — `scripts/make_test_image.py`, encodeur PNG de
quelques lignes plutôt qu'une dépendance à Pillow — et soumis quatre couleurs franches. Le modèle
répond `Red`, `Blue`, `Green`, `Yellow` : **4 sur 4**. Une couleur ne se devine pas ; l'image
traverse réellement le projecteur.

Le binaire `ollama` officiel confirme le parcours canonique de bout en bout : `ollama show`
affiche une section « Projector » (`clip`, 93,51 M de paramètres), et
`ollama run <modèle> "What color… /chemin/image.png"` imprime « Added image » puis répond `Blue`.

**Défaut 1 — le garde-fou de capacités manquait sur deux façades.** La contre-épreuve, une image
envoyée au Qwen textuel, a révélé une asymétrie : Ollama et OpenAI répondaient `400 … does not
support vision`, mais Responses et Anthropic laissaient l'image atteindre `llama-server`. Son
refus remontait en `502` portant un message d'amont — « you may need to provide the mmproj ».
Deux défauts en un : un `502` annonce une panne du serveur là où la requête est simplement
invalide, et ce conseil s'adresse à l'exploitant, pas au client.

Cause : le contrôle était **dupliqué**, une copie dans `api/ollama.py`, une autre dans
`api/openai.py`, aucune ailleurs. Un garde-fou par façade est un garde-fou qu'on oublie.
Correction : une seule fonction `api/common.py::reject_unsupported`, appelée par les quatre
façades ; les deux copies sont supprimées. Treize tests, dont quatre écrits en échec avant la
correction, et un test dédié à l'absence de fuite du mot `mmproj` vers le client.

**Défaut 2 — le projecteur n'était pas apparié.** Le projecteur était choisi par
`sorted(projectors)[0]`, sans rapport avec le fichier de poids retenu. Demander `:f16` livrait
donc le modèle en f16 et l'encodeur d'image en Q8_0. Le résultat fonctionne — `llama.cpp` accepte
l'écart — ce qui rend la surprise d'autant plus silencieuse : la demande explicite de
l'utilisateur était contredite sans le moindre signal. Un dépôt qui prend la peine de publier les
deux variantes exprime une intention ; la suivre est le comportement le moins surprenant.
`_pair_projector` compare le dernier segment du nom de fichier, avec repli sur le premier
projecteur lorsqu'aucun ne correspond — cas de `ggml-org/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M`, dont
le dépôt ne publie pas de projecteur en Q4_K_M.

**Défaut 3 — un projecteur pouvait être retenu comme modèle.** La sélection par motif
(`dépôt:motif`) cherchait parmi **tous** les GGUF, projecteurs compris. Sur un dépôt publiant
`mmproj-…-f16.gguf`, `:f16` pouvait donc désigner un projecteur comme fichier de poids. Corrigé
en restreignant la recherche aux poids ; le message d'erreur reste inchangé.

**Hypothèse invalidée, consignée.** Un test supposait qu'envoyer une image à un modèle textuel
depuis le CLI produirait une erreur. Il a échoué : le CLI sort en `0`. Lecture de l'amont —
`cmd/cmd.go` l. 854-867 — le CLI décide **lui-même** d'attacher ou non un fichier, à partir de
`Capabilities`, de `ProjectorInfo` et des clés `.vision.` que le serveur annonce. Il n'attache
donc rien et envoie le chemin comme du texte. Le comportement était correct, mon attente ne
l'était pas. Le test a été réécrit pour vérifier ce qui relève réellement d'`ollama.cpp` : qu'aucun
de ces trois signaux ne fuit sur un modèle purement textuel.

**Conséquence.** 774 tests passent avec le vrai `llama-server`, le vrai binaire `ollama`, Hugging
Face et un vrai modèle multimodal.
