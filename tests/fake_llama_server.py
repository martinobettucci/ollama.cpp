#!/usr/bin/env python3
"""Faux `llama-server` : implémentation du contrat HTTP amont, sans inférence.

@verifies docs/BACKLOG.md OC-030 « LlamaServerSupervisor », OC-032 « Détection de capacités »,
          OC-033 « ModelLifecycleManager »
@verifies docs/ollama.cpp-architecture.md §1.1 « Surface HTTP réellement exposée »,
          §1.3 « /props : la source de vérité runtime »
@verifies docs/DAT.md §5.2 « Interfaces consommées »

Ce n'est **pas un mock** : c'est un vrai processus, lancé par le vrai superviseur, qui alloue un
vrai port, répond en vrai HTTP et se termine sur un vrai signal. Seule l'inférence est
déterministe au lieu d'être calculée.

Ce choix suit CLAUDE.md §15 : les composants critiques ne sont pas remplacés par des mocks, et le
contrat simulé est documenté. Il permet de vérifier tout le chemin — allocation de port, attente
de `/health`, lecture de `/props`, détection de capacités, proxy de streaming, arrêt propre —
sans avoir à compiler `llama.cpp` ni à télécharger un modèle.

Les réponses reproduisent la forme réelle relevée dans `tools/server/server-context.cpp`
(révision auditée `39be55c`), en particulier `/props` (l. 4576-4611).

Variables d'environnement de simulation :

- `FAKE_LLAMA_EXIT_CODE`  : sort immédiatement avec ce code (échec de chargement) ;
- `FAKE_LLAMA_START_DELAY`: secondes avant d'écouter (chargement lent, dépassement de délai) ;
- `FAKE_LLAMA_VISION`     : `1` pour annoncer `modalities.vision` ;
- `FAKE_LLAMA_TOOLS`      : `0` pour un template sans support d'outils ;
- `FAKE_LLAMA_THINKING`   : `1` pour un template gérant le raisonnement ;
- `FAKE_LLAMA_LOG_BYTES`  : écrit ce volume sur stdout/stderr avant d'écouter, pour éprouver le
  drainage des tubes par le superviseur (un `PIPE` non lu bloque le fils dès 64 Kio).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGV: list[str] = []
OPTIONS: argparse.Namespace


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip() in {"1", "true", "yes", "on"}


def build_props() -> dict:
    """Réponse `/props`, calquée sur `server-context.cpp` l. 4576-4611."""
    return {
        "default_generation_settings": {"n_ctx": OPTIONS.ctx_size, "params": {}},
        "total_slots": OPTIONS.parallel,
        "model_alias": OPTIONS.alias,
        "model_path": OPTIONS.model,
        "model_ftype": "Q4_K_M",
        "modalities": {
            "vision": bool(OPTIONS.mmproj) and _env_flag("FAKE_LLAMA_VISION", True),
            "video": False,
            "audio": False,
        },
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        "chat_template_caps": {
            "supports_tools": _env_flag("FAKE_LLAMA_TOOLS", True),
            "supports_tool_calls": _env_flag("FAKE_LLAMA_TOOLS", True),
            "supports_system_role": True,
            "supports_parallel_tool_calls": True,
            "supports_preserve_reasoning": _env_flag("FAKE_LLAMA_THINKING", False),
            "supports_reasoning_effort": _env_flag("FAKE_LLAMA_THINKING", False),
            "supports_string_content": True,
            "supports_typed_content": False,
            "supports_object_arguments": False,
        },
        "bos_token": "<s>",
        "eos_token": "</s>",
        "build_info": "fake-llama-server",
        "is_sleeping": False,
    }


def _echo_reply(payload: dict) -> str:
    """Réponse déterministe : renvoie le dernier contenu utilisateur, préfixé.

    Déterministe et traçable : un test peut vérifier que le prompt a bien traversé toute la
    chaîne canonique sans dépendre d'un modèle réel.
    """
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return f"echo: {content}"
            if isinstance(content, list):
                texts = [part.get("text", "") for part in content if isinstance(part, dict)]
                return "echo: " + "".join(texts)
    return "echo:"


def _wants_tool_call(payload: dict) -> bool:
    """Émet un appel d'outil dès que la requête en propose un, pour couvrir le chemin tool call."""
    return bool(payload.get("tools"))


def build_chat_completion(payload: dict) -> dict:
    """Réponse `/v1/chat/completions` non streamée, au format OpenAI servi par `llama-server`."""
    message: dict = {"role": "assistant", "content": _echo_reply(payload)}
    finish_reason = "stop"

    if _wants_tool_call(payload):
        tool = payload["tools"][0]
        name = (tool.get("function") or {}).get("name", "unknown")
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_fake_1",
                    "type": "function",
                    "function": {"name": name, "arguments": '{"ok":true}'},
                }
            ],
        }
        finish_reason = "tool_calls"

    if _env_flag("FAKE_LLAMA_THINKING", False):
        message["reasoning_content"] = "réflexion simulée"

    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1700000000,
        "model": payload.get("model") or OPTIONS.alias,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        "timings": {
            "prompt_n": 11,
            "prompt_ms": 12.5,
            "predicted_n": 7,
            "predicted_ms": 42.0,
        },
    }


def build_chat_chunks(payload: dict) -> list[dict]:
    """Suite de chunks SSE, découpée par mots pour exercer réellement l'agrégation."""
    model = payload.get("model") or OPTIONS.alias
    base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 1700000000,
            "model": model}

    if _wants_tool_call(payload):
        tool = payload["tools"][0]
        name = (tool.get("function") or {}).get("name", "unknown")
        return [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                  "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_fake_1", "type": "function",
                 "function": {"name": name, "arguments": '{"ok":true}'}}]},
                "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}},
        ]

    chunks = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                    "finish_reason": None}]}]
    if _env_flag("FAKE_LLAMA_THINKING", False):
        chunks.append({**base, "choices": [{"index": 0,
                                            "delta": {"reasoning_content": "réflexion simulée"},
                                            "finish_reason": None}]})
    text = _echo_reply(payload)
    for index, word in enumerate(text.split(" ")):
        piece = word if index == 0 else f" {word}"
        chunks.append({**base, "choices": [{"index": 0, "delta": {"content": piece},
                                            "finish_reason": None}]})
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                   "timings": {"prompt_n": 11, "prompt_ms": 12.5,
                               "predicted_n": 7, "predicted_ms": 42.0}})
    return chunks


def build_embeddings(payload: dict) -> dict:
    """Réponse `/v1/embeddings`. Vecteurs déterministes, dérivés de la longueur de l'entrée."""
    raw = payload.get("input")
    inputs = raw if isinstance(raw, list) else [raw if raw is not None else ""]
    return {
        "object": "list",
        "model": payload.get("model") or OPTIONS.alias,
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": [round(0.1 * (len(str(item)) % 10 + offset), 4) for offset in range(4)],
            }
            for index, item in enumerate(inputs)
        ],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # noqa: D102 - silence, les tests lisent stderr
        pass

    # --- utilitaires ---------------------------------------------------------------------------

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            return {}

    # --- routes --------------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - imposé par BaseHTTPRequestHandler
        if self.path in ("/health", "/v1/health"):
            self._send_json({"status": "ok"})
        elif self.path == "/props":
            self._send_json(build_props())
        elif self.path == "/debug/argv":
            # Endpoint propre au faux serveur : permet aux tests de vérifier la ligne de commande
            # réellement reçue, donc la traduction du manifest en drapeaux (OC-031).
            self._send_json({"argv": ARGV})
        else:
            self._send_json({"error": {"message": "not found"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()

        if self.path in ("/v1/chat/completions", "/chat/completions"):
            if payload.get("stream"):
                self._stream_sse(build_chat_chunks(payload))
            else:
                self._send_json(build_chat_completion(payload))
        elif self.path in ("/v1/embeddings", "/embeddings", "/embedding"):
            self._send_json(build_embeddings(payload))
        elif self.path == "/tokenize":
            text = payload.get("content") or ""
            self._send_json({"tokens": list(range(1, len(str(text).split()) + 1))})
        elif self.path == "/detokenize":
            self._send_json({"content": " ".join(str(t) for t in payload.get("tokens") or [])})
        else:
            self._send_json({"error": {"message": "not found"}}, status=404)

    def _stream_sse(self, chunks: list[dict]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Accepte les drapeaux que `ollama.cpp` émet, et ignore silencieusement les autres.

    Le vrai `llama-server` en accepte des centaines ; refuser un drapeau inconnu ferait échouer
    les tests pour une raison sans rapport avec ce qu'ils vérifient.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default="")
    parser.add_argument("--alias", default="fake")
    parser.add_argument("--mmproj", default="")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--parallel", type=int, default=1)
    known, _unknown = parser.parse_known_args(argv)
    return known


def main() -> int:
    global ARGV, OPTIONS
    ARGV = sys.argv[1:]
    OPTIONS = parse_args(ARGV)

    exit_code = os.environ.get("FAKE_LLAMA_EXIT_CODE")
    if exit_code:
        print("simulated load failure: unable to load model", file=sys.stderr, flush=True)
        return int(exit_code)

    delay = float(os.environ.get("FAKE_LLAMA_START_DELAY") or 0)
    if delay:
        time.sleep(delay)

    # Déluge de journal : reproduit un amont bavard (décodage spéculatif, journalisation par
    # brouillon). Sans drainage côté superviseur, l'écriture ci-dessous BLOQUE et le serveur
    # n'écoute jamais — exactement la panne que ce mécanisme doit empêcher.
    volume = int(os.environ.get("FAKE_LLAMA_LOG_BYTES") or 0)
    if volume:
        ligne = "x" * 120
        for flux in (sys.stdout, sys.stderr):
            ecrit = 0
            while ecrit < volume:
                print(ligne, file=flux, flush=True)
                ecrit += len(ligne) + 1

    server = ThreadingHTTPServer((OPTIONS.host, OPTIONS.port), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
