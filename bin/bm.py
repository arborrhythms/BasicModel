"""BasicModel chat client.

Two modes:

* **In-process** (``--config PATH``): load the model directly and call
  ``model.infer(...)`` without any HTTP layer.  This is the easiest
  path for local debugging -- any exception surfaces with a full
  traceback instead of being swallowed into a 500.

* **HTTP** (default): POST to a running ``serve.py`` at the given
  ``--host`` / ``--port``.

Usage:
    # In-process (no server running)
    python bm.py --config ../data/MM_20M.xml
    python bm.py --config ../data/MM_20M.xml "hello"

    # Talk to a running serve.py
    python bm.py --port 8003
    python bm.py --port 8003 "hello"
    python bm.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


# --- HTTP client --------------------------------------------------------

def chat_http(text, host="127.0.0.1", port=8001):
    """Send a chat query to a running serve.py and return the response.

    POSTs a single-message OpenAI-style payload to ``/chat/completions``
    and extracts the assistant text. Returns an ``[error] ...`` string
    on URL failure rather than raising.
    """
    url = f"http://{host}:{port}/chat/completions"
    payload = json.dumps({
        "messages": [{"role": "user", "content": text}]
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        return f"[error] {e}"


# --- In-process client --------------------------------------------------

def _load_model_in_process(config_path, max_length=64):
    """Load the XML-selected model class; return a callable ``chat(text)``.

    Reads the XML config, loads the configured dataset into ``TheData``
    (forwarding only the shard / maxDocs / shardDir keys actually
    present), constructs the model via ``BaseModel.from_config``, runs
    it through ``util.compile`` and returns a closure that invokes
    ``model.infer`` in autoregressive mode.
    """
    # Ensure bin/ is importable when run as a script
    bin_dir = os.path.dirname(os.path.abspath(__file__))
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)

    from Models import BaseModel
    from data import TheData

    cfg = BaseModel.load_config(config_path)
    arch = cfg.get("architecture", {})
    dat = arch.get("data", {})
    dataset = dat.get("dataset")
    if dataset is not None:
        # Inline / xor-style datasets don't carry shard / maxDocs /
        # shardDir keys; only forward them when the XML supplies them.
        load_kwargs = {}
        for src_key, dst_key, cast in (
                ("numShards", "num_shards", int),
                ("maxDocs",   "max_docs",   int),
                ("shardDir",  "shard_dir",  str)):
            v = dat.get(src_key)
            if v is None:
                continue
            try:
                load_kwargs[dst_key] = cast(v)
            except (TypeError, ValueError):
                pass
        TheData.load(dataset, **load_kwargs)

    model, _ = BaseModel.from_config(config_path, data=TheData)
    model.eval()
    from util import compile
    model = compile(model)

    def chat_inproc(text):
        """In-process chat closure: tokenize, run AR inference, join."""
        tokens = model.infer(text, mode="AR", max_length=max_length)
        return " ".join(tokens) if tokens else ""

    return chat_inproc


# --- REPL / entry -------------------------------------------------------

DEMO_QUERIES = [
    "The quick brown fox",
    "Once upon a time",
    "Scientists discovered that",
    "The weather today is",
    "In the beginning there was",
]


def main():
    """CLI entry point: parse args and dispatch to demo / one-shot / REPL.

    Selects in-process mode when ``--config`` is given, HTTP mode
    otherwise. Three flow modes: ``--demo`` cycles fixed prompts,
    positional ``query`` is one-shot, default is an interactive REPL
    that prints full tracebacks instead of swallowing exceptions.
    """
    parser = argparse.ArgumentParser(description="BasicModel chat client")
    parser.add_argument("query", nargs="?", help="Single query to send")
    parser.add_argument(
        "--config",
        help="XML config path. If given, load the model in-process and "
             "skip the HTTP layer.")
    parser.add_argument(
        "--max-length", type=int, default=64,
        help="Max generated tokens (in-process mode only).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--demo", action="store_true",
                        help="Run sample queries")
    args = parser.parse_args()

    if args.config:
        print(f"[bm] loading model from {args.config}...")
        chat = _load_model_in_process(args.config, max_length=args.max_length)
        print("[bm] model ready.")
    else:
        def chat(text):
            """HTTP chat closure bound to the parsed --host / --port."""
            return chat_http(text, args.host, args.port)

    if args.demo:
        for q in DEMO_QUERIES:
            print(f">> {q}")
            print(f"<< {chat(q)}")
            print()
        return

    if args.query:
        print(chat(args.query))
        return

    # Interactive mode
    mode = "in-process" if args.config else f"http://{args.host}:{args.port}"
    print(f"BasicModel chat ({mode}) -- Ctrl+D to quit")
    while True:
        try:
            text = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            continue
        try:
            print(f"<< {chat(text)}")
        except Exception:
            # Don't suppress -- surface the full traceback in-process.
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
