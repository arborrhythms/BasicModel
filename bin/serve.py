"""BasicModel HTTP server -- WikiOracle integration.

Provides an OpenAI-compatible /chat/completions endpoint so WikiOracle
can query BasicModel the same way it queries NanoChat.

Security:
    - Bearer token auth via BASICMODEL_API_TOKEN env var (disabled when unset)
    - Request body size limit (1MB)
    - Input validation and length truncation
    - Rate limiting per IP (configurable via BASICMODEL_RATE_LIMIT)
    - Error sanitization (no internal details in responses)
    - CORS restricted to localhost

Usage:
    python serve.py                  # defaults from BasicModel.xml
    python serve.py -p 8001          # override port
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from flask import Flask, jsonify, request

# Ensure basicmodel bin/ is on the path for local imports
_BIN = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_BIN)
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
from secure import guard_input

logger = logging.getLogger("basicmodel")
logging.basicConfig(level=logging.INFO, format="[BasicModel] %(message)s")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1_000_000  # 1MB request body limit

# Global model reference -- set by main()
_model = None
_model_config = {}

# --- Rate limiting (in-process sliding window) ---
_rate_limit = int(os.getenv("BASICMODEL_RATE_LIMIT", "60"))  # requests per minute per IP
_rate_window: dict[str, list[float]] = {}

def _check_rate_limit(ip: str) -> bool:
    """Return True if request is within rate limit, False if exceeded.

    Uses a 60-second sliding window keyed on client IP. ``_rate_limit``
    of 0 disables the cap. Mutates ``_rate_window[ip]`` in place: prunes
    entries older than the window and appends ``now`` on accept.
    """
    if _rate_limit <= 0:
        return True
    now = time.monotonic()
    window = _rate_window.setdefault(ip, [])
    # Prune entries older than 60 seconds
    cutoff = now - 60
    _rate_window[ip] = window = [t for t in window if t > cutoff]
    if len(window) >= _rate_limit:
        return False
    window.append(now)
    return True


# --- Auth and security ---
_api_token = os.getenv("BASICMODEL_API_TOKEN", "")


@app.before_request
def auth_and_rate_check():
    """Check bearer token and rate limit on all endpoints except /health.

    Flask ``before_request`` hook. Returns a JSON error response (401 or
    429) on rejection; returns ``None`` to let the request continue.
    Bearer auth is skipped when ``BASICMODEL_API_TOKEN`` is unset.
    """
    if request.path == "/health":
        return None

    # Bearer token auth (skip when token not configured)
    if _api_token:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {_api_token}":
            return jsonify({"error": "Unauthorized"}), 401

    # Rate limiting
    ip = request.remote_addr or "unknown"
    if not _check_rate_limit(ip):
        resp = jsonify({"error": "Rate limit exceeded"})
        resp.headers["Retry-After"] = "60"
        return resp, 429

    return None


@app.after_request
def add_security_headers(response):
    """Restrict CORS to localhost (BasicModel only serves WikiOracle locally).

    Flask ``after_request`` hook. Echoes the ``Origin`` header back only
    for 127.0.0.1 / localhost callers and sets the allow-methods /
    allow-headers pair on every response.
    """
    origin = request.headers.get("Origin", "")
    if origin.startswith(("http://127.0.0.1", "https://127.0.0.1",
                          "http://localhost", "https://localhost")):
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


# --- Model loading ---

def _load_model(config_path=None):
    """Load the XML-selected model class.

    Defaults to ``../data/BasicModel.xml`` when ``config_path`` is None.
    Loads any configured dataset into ``TheData``, constructs the model
    via ``BaseModel.from_config``, evaluates and compiles it. Returns
    ``(model, cfg)``.
    """
    from Models import BaseModel
    from data import TheData

    if config_path is None:
        config_path = os.path.join(_PROJECT, "data", "BasicModel.xml")

    cfg = BaseModel.load_config(config_path)
    arch = cfg.get("architecture", {})
    dat = arch.get("data", {})
    dataset = dat.get("dataset")  # None when serving without training data
    if dataset is not None:
        TheData.load(dataset,
                     num_shards=int(dat.get("numShards")),
                     max_docs=int(dat.get("maxDocs")),
                     shard_dir=dat.get("shardDir"))

    model, cfg = BaseModel.from_config(config_path, data=TheData)

    model.eval()
    from util import compile
    model = compile(model)
    return model, cfg


# --- Endpoints ---

_MAX_MSG_LEN = 10_000  # max chars per user message
_MAX_MESSAGES = 50     # max messages in a request

# HTTP generation budget. AR reruns the model per generated token,
# so a request that uses the XML-wide maxResponseLength (typically
# thousands of characters) can sit in inference long enough for the
# urllib client to time out.  Cap server-side: default 64 generated
# characters, honor explicit max_tokens requests but clamp to 128.
_DEFAULT_GEN_BUDGET = 64
_MAX_GEN_BUDGET = 128


@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint.

    Accepts messages in the standard format and returns a response.
    BasicModel processes the last user message through its forward pass.
    """
    global _model
    if _model is None:
        return jsonify({"error": "Model not loaded"}), 503

    body = request.get_json(force=True) or {}
    messages = body.get("messages", [])

    # Input validation
    if not isinstance(messages, list):
        return jsonify({"error": "messages must be a list"}), 400
    if len(messages) > _MAX_MESSAGES:
        return jsonify({"error": f"Too many messages (max {_MAX_MESSAGES})"}), 400

    # Find the last user message
    user_msg = ""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if not isinstance(content, str):
                return jsonify({"error": "message content must be a string"}), 400
            user_msg = content[:_MAX_MSG_LEN]
            break

    if not user_msg:
        return jsonify({"error": "No user message found"}), 400

    # Resolve generation budget: honor explicit max_tokens but clamp,
    # fall back to the default when unset / invalid.
    raw_budget = body.get("max_tokens")
    try:
        gen_budget = int(raw_budget) if raw_budget is not None else _DEFAULT_GEN_BUDGET
    except (TypeError, ValueError):
        gen_budget = _DEFAULT_GEN_BUDGET
    gen_budget = max(1, min(gen_budget, _MAX_GEN_BUDGET))

    # Prompt injection guard
    injection = guard_input(user_msg)
    if injection:
        logger.warning("Input blocked: %s", injection)
        return jsonify({"error": "Input rejected"}), 400

    # Process truth entries if provided (WikiOracle TruthSet)
    truth_entries = body.get("truth", [])
    if truth_entries and isinstance(truth_entries, list):
        try:
            _model.store_truths(truth_entries)
        except Exception as exc:
            logger.warning("Failed to store truths: %s", exc)

    # Shamatha speech: restrict grammar to S -> C only
    thought_free = body.get("thought_free", False)
    try:
        from basicmodel.bin.Language import TheGrammar
    except ModuleNotFoundError:
        from Language import TheGrammar
    try:
        if thought_free:
            TheGrammar.thought_free = True

        # Autoregressive inference: extend input text word by word,
        # bounded by the server's generation budget so slow AR runs
        # cannot sit longer than the HTTP client's read timeout.
        predicted_words = _model.infer(
            user_msg, mode='AR', max_length=gen_budget)
        response_text = " ".join(predicted_words)

        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"<conversation>{response_text}</conversation>",
                }
            }]
        }
        clarifications = getattr(_model, "_last_clarifications", None)
        if clarifications:
            response["clarifications"] = list(clarifications)
        return jsonify(response)
    except Exception as exc:
        logger.exception("Inference error")
        return jsonify({"error": "Internal model error"}), 500
    finally:
        TheGrammar.thought_free = False


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Returns a JSON body with ``ok`` (True once the model is loaded),
    the model name, and a flattened copy of the architecture config so
    monitors can sanity-check what is actually running.
    """
    return jsonify({
        "ok": _model is not None,
        "model": "BasicModel",
        "config": {k: v for k, v in _model_config.get("architecture", {}).items()}
        if _model_config else {},
    })


def main():
    """CLI entry point: parse args, load model, warm up, run Flask app.

    Resolves the config path from ``--config``, ``$BASIC_XML`` or the
    XML default. Pays the ``torch.compile`` warmup cost up-front so the
    first real request does not trip the client's read timeout. Mutates
    the module-level ``_model`` and ``_model_config`` globals.
    """
    global _model, _model_config

    parser = argparse.ArgumentParser(
        prog="serve.py",
        description=(
            "Start the BasicModel HTTP inference server.\n\n"
            "Exposes a chat-completions endpoint compatible with the OpenAI\n"
            "API schema.  The model config is loaded from --config (XML), an\n"
            "environment variable, or the server block inside the XML.\n\n"
            "Examples:\n"
            "  python serve.py --config data/BasicModel.xml\n"
            "  python serve.py --config data/BasicModel.xml --port 8080\n"
            "  BASIC_XML=data/BasicModel.xml python serve.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-p", "--port", type=int, default=None,
                        help="TCP port to listen on (overrides XML <server><port>).")
    parser.add_argument("--host", default=None,
                        help="Bind address (overrides XML <server><host>; default 127.0.0.1).")
    parser.add_argument("--config", "--model", dest="config",
                        default=None,
                        help="Path to the XML config file (also accepted as --model).")
    args = parser.parse_args()

    config_path = args.config or os.getenv("BASIC_XML")

    logger.info("Loading model...")
    _model, _model_config = _load_model(config_path)
    logger.info("Model loaded: %d parameters",
                sum(p.numel() for p in _model.parameters()))

    # Warm up the inference path so that torch.compile's one-time
    # compilation cost is paid BEFORE /health reports ready.  Without
    # this, the first /chat/completions request pays ~24s of compile
    # overhead on top of normal inference and trips the client's
    # 30s read timeout.
    logger.info("Warming up inference path...")
    try:
        _model.infer("warmup", mode='AR', max_length=4)
        logger.info("Warmup complete.")
    except Exception as exc:
        logger.warning("Warmup failed (continuing anyway): %s", exc)

    arch = _model_config.get("architecture", {})
    srv = arch.get("server", {})
    host = args.host or srv.get("host")
    port = args.port or srv.get("port")

    logger.info("Serving on %s:%s", host, port)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
