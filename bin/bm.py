"""BasicModel chat client — sends queries to the /chat/completions endpoint.

Usage:
    python bm.py                          # interactive mode
    python bm.py "What is the capital"    # single query
    python bm.py --demo                   # run sample queries
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


def chat(text, host="127.0.0.1", port=8001):
    """Send a chat query and return the response text."""
    url = f"http://{host}:{port}/chat/completions"
    payload = json.dumps({
        "messages": [{"role": "user", "content": text}]
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        return f"[error] {e}"


DEMO_QUERIES = [
    "The quick brown fox",
    "Once upon a time",
    "Scientists discovered that",
    "The weather today is",
    "In the beginning there was",
]


def main():
    parser = argparse.ArgumentParser(description="BasicModel chat client")
    parser.add_argument("query", nargs="?", help="Single query to send")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--demo", action="store_true", help="Run sample queries")
    args = parser.parse_args()

    if args.demo:
        for q in DEMO_QUERIES:
            print(f">> {q}")
            print(f"<< {chat(q, args.host, args.port)}")
            print()
        return

    if args.query:
        print(chat(args.query, args.host, args.port))
        return

    # Interactive mode
    print("BasicModel chat (Ctrl+D to quit)")
    while True:
        try:
            text = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            continue
        print(f"<< {chat(text, args.host, args.port)}")


if __name__ == "__main__":
    main()
