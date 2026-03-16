"""BasicModel HTTP server — WikiOracle integration.

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

# Ensure bin/ is on the path for local imports
_BIN = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_BIN)
_WO_BIN = os.path.join(os.path.dirname(_PROJECT), "bin")  # WikiOracle bin/
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _WO_BIN not in sys.path:
    sys.path.insert(0, _WO_BIN)

import torch
from security import guard_input, detect_identifiability, detect_asymmetric_claim

logger = logging.getLogger("basicmodel")
logging.basicConfig(level=logging.INFO, format="[BasicModel] %(message)s")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1_000_000  # 1MB request body limit

# Global model reference — set by main()
_model = None
_model_config = {}

# --- Rate limiting (in-process sliding window) ---
_rate_limit = int(os.getenv("BASICMODEL_RATE_LIMIT", "60"))  # requests per minute per IP
_rate_window: dict[str, list[float]] = {}

def _check_rate_limit(ip: str) -> bool:
    """Return True if request is within rate limit, False if exceeded."""
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
    """Check bearer token and rate limit on all endpoints except /health."""
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
    """Restrict CORS to localhost (BasicModel only serves WikiOracle locally)."""
    origin = request.headers.get("Origin", "")
    if origin.startswith(("http://127.0.0.1", "https://127.0.0.1",
                          "http://localhost", "https://localhost")):
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


# --- Model loading ---

def _load_model(config_path=None):
    """Load BasicModel with config from BasicModel.xml."""
    from BasicModel import BasicModel, TheData

    if config_path is None:
        config_path = os.path.join(_PROJECT, "data", "BasicModel.xml")

    cfg = BasicModel.load_config(config_path)
    arch = cfg.get("architecture", {})
    dataset = arch.get("dataset", "xor")
    TheData.load(dataset)

    model = BasicModel()
    cfg = model.create_from_config(config_path, data=TheData)

    model.eval()
    return model, cfg


# --- Endpoints ---

_MAX_MSG_LEN = 10_000  # max chars per user message
_MAX_MESSAGES = 50     # max messages in a request


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

    # Prompt injection guard
    injection = guard_input(user_msg)
    if injection:
        logger.warning("Input blocked: %s", injection)
        return jsonify({"error": "Input rejected"}), 400

    try:
        # Tokenize: convert text to input tensor via the model's Embedding
        input_space = _model.inputSpace
        vs = input_space.vectors()

        # Encode text as word embeddings
        words = user_msg.split()
        nVec = input_space.outputShape[0]  # max tokens (nActive)
        emb_dim = vs.embeddingSize
        input_tensor = torch.zeros(1, nVec, emb_dim, device=next(_model.parameters()).device)

        codebook = vs._emb.weight.detach()
        word_list = vs.wv.index_to_key
        word_to_idx = {w: i for i, w in enumerate(word_list)}

        for i, word in enumerate(words[:nVec]):
            w_lower = word.lower()
            if w_lower in word_to_idx:
                idx = word_to_idx[w_lower]
                input_tensor[0, i, :codebook.shape[1]] = codebook[idx]

        # Forward pass
        with torch.no_grad():
            input_emb, symbols, output_pred = _model.forward(input_tensor)

        # Decode: map output vector to nearest word in InputSpace codebook
        predicted_words = input_space.predict(output_pred)
        response_text = " ".join(predicted_words)

        # Output safety filtering
        if detect_identifiability(response_text):
            logger.warning("Output blocked: identifiability detected")
            response_text = ""
        asym = detect_asymmetric_claim(response_text)
        if asym:
            logger.warning("Output blocked: %s", asym)
            response_text = ""

        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"<conversation>{response_text}</conversation>",
                }
            }]
        })
    except Exception as exc:
        logger.exception("Inference error")
        return jsonify({"error": "Internal model error"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "ok": _model is not None,
        "model": "BasicModel",
        "config": {k: v for k, v in _model_config.get("architecture", {}).items()}
        if _model_config else {},
    })


def main():
    global _model, _model_config

    parser = argparse.ArgumentParser(description="BasicModel HTTP server")
    parser.add_argument("-p", "--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--config", default=None, help="Path to XML config file")
    args = parser.parse_args()

    config_path = args.config or os.getenv("BASIC_XML")

    logger.info("Loading model...")
    _model, _model_config = _load_model(config_path)
    logger.info("Model loaded: %d parameters",
                sum(p.numel() for p in _model.parameters()))

    host = args.host or _model_config.get("server", {}).get("host", "127.0.0.1")
    port = args.port or _model_config.get("server", {}).get("port", 8001)

    logger.info("Serving on %s:%s", host, port)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
