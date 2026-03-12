"""BasicModel HTTP server — WikiOracle integration.

Provides an OpenAI-compatible /chat/completions endpoint so WikiOracle
can query BasicModel the same way it queries NanoChat.

Usage:
    python serve.py                  # defaults from model.xml
    python serve.py -p 8001          # override port
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

from flask import Flask, jsonify, request

# Ensure bin/ is on the path for local imports
_BIN = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_BIN)
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

app = Flask(__name__)

# Global model reference — set by main()
_model = None
_model_config = {}


def _load_model():
    """Load BasicModel with config from model.xml."""
    from BasicModel import BasicModel, TheData, TheObjectEncoding

    model = BasicModel()
    cfg = model.load_config(os.path.join(_PROJECT, "model.xml"))
    arch = cfg.get("architecture", {})

    # Data must be loaded before model creation (sets dimensions)
    dataset = cfg.get("training", {}).get("dataset", "xor")
    TheData.load(dataset)

    model.create(
        nInput=arch.get("nInput", 32),
        nPercepts=arch.get("nPercepts", 64),
        nConcepts=arch.get("nConcepts", 256),
        nSymbols=arch.get("nSymbols", 2),
        nWords=arch.get("nWords", 16),
        nOutput=arch.get("nOutput", 32),
        reversePass=arch.get("reversePass", True),
    )

    # Load weights if available
    wcfg = cfg.get("weights", {})
    if wcfg.get("autoload", True):
        wpath = wcfg.get("path", "output/weights.pt")
        if not os.path.isabs(wpath):
            wpath = os.path.join(_PROJECT, wpath)
        model.load_weights(wpath)

    model.eval()
    return model, cfg


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

    # Find the last user message
    user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    if not user_msg:
        return jsonify({"error": "No user message found"}), 400

    try:
        # BasicModel works with tensor input — for now, echo the query
        # with a model-status response.  Full tensor inference will be
        # added when the training pipeline is connected.
        response_text = (
            f"<conversation>BasicModel received your query. "
            f"The model has {sum(p.numel() for p in _model.parameters())} parameters "
            f"across {len(list(_model.named_modules()))} modules.</conversation>"
        )
        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text,
                }
            }]
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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
    args = parser.parse_args()

    print("[BasicModel] Loading model...")
    _model, _model_config = _load_model()
    print(f"[BasicModel] Model loaded: {sum(p.numel() for p in _model.parameters())} parameters")

    host = args.host or _model_config.get("server", {}).get("host", "127.0.0.1")
    port = args.port or _model_config.get("server", {}).get("port", 8001)

    print(f"[BasicModel] Serving on {host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
