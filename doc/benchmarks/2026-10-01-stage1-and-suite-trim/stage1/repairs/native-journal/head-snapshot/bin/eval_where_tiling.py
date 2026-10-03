#!/usr/bin/env python3
"""Corpus evaluator for the overlapping PS/WS `.where` tiling.

Input is JSONL with ``text`` and gold UTF-8 byte ``spans``.  This evaluator is
deliberately independent of the lexer: PS starts from byte atoms, WS proposes
overlapping typed/word/sentence wholes, and :class:`WhereTilingLayer` must
settle the requested surface objects through its fixed refinement passes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch

_BIN = Path(__file__).resolve().parent
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import Meronomy
from Layers import WHITESPACE, WhereTilingLayer


KIND_TYPED = 1
KIND_WORD = 2
KIND_SEPARATOR = 3
KIND_SENTENCE = 4


def _candidates(text: str):
    """Return byte atoms plus overlapping whole spans/kinds."""
    raw = text.encode("utf-8")
    parts = [(i, i + 1) for i in range(len(raw))]
    words = set(Meronomy.word_spans(raw))
    wholes, kinds, seen = [], [], set()

    def add(span, kind):
        pair = (int(span[0]), int(span[1]))
        if pair[1] <= pair[0] or pair in seen:
            return
        seen.add(pair)
        wholes.append(pair)
        kinds.append(int(kind))

    # Complete surface tiling first: words and separators are peers.
    for span in Meronomy.word_tiling(raw):
        if tuple(span) in words:
            kind = KIND_WORD
        else:
            kind = KIND_SEPARATOR
        add(span, kind)
    # Typed refinements overlap the word candidates (e.g. R2D2 has alternating
    # letter/digit children) and exercise the multi-level schedule.
    for _cls, start, end in Meronomy.class_segments(raw):
        add((start, end), KIND_TYPED)
    if raw:
        add((0, len(raw)), KIND_SENTENCE)
    return raw, parts, wholes, kinds, words


def _is_whitespace(raw: bytes, span):
    chunk = raw[int(span[0]):int(span[1])]
    return bool(chunk) and all(Meronomy._byte_class(b) == WHITESPACE
                               for b in chunk)


def parse_surface(text: str, passes: int = 3):
    """Parse one surface and return accepted basic spans plus a route trace."""
    raw, parts, wholes, kinds, word_spans = _candidates(text)
    if not raw:
        return [], {"passes": 0, "overflow": 0, "unresolved": 0,
                    "sigma": 0, "pi": 0, "raise": 0}
    layer = WhereTilingLayer()
    p = torch.tensor(parts, dtype=torch.float32).unsqueeze(0)
    w = torch.tensor(wholes, dtype=torch.float32).unsqueeze(0)
    schedule = layer.build_schedule(p, w, passes)
    accepted = schedule[-1]["accepted_whole"][0]
    # Surface basic objects are separator-bounded words plus non-whitespace
    # separator runs (punctuation).  Typed descendants and the sentence parent
    # remain in the accepted multi-level tiling but are not scored as words.
    predicted = []
    for j, span in enumerate(wholes):
        if not bool(accepted[j]):
            continue
        if tuple(span) in word_spans or (
                kinds[j] == KIND_SEPARATOR and not _is_whitespace(raw, span)):
            predicted.append(tuple(span))
    predicted = sorted(set(predicted))
    trace = {
        "passes": len(schedule),
        "overflow": int(schedule[-1]["overflow"][0]),
        "unresolved": int((schedule[-1]["whole_valid"][0]
                           & ~accepted).sum()),
        "sigma": sum(int(x["sigma_whole"][0].sum()) for x in schedule),
        "pi": sum(int(x["pi_part"][0].sum()) for x in schedule),
        "raise": sum(int(x["raise_whole"][0].sum()
                         + x["raise_part"][0].sum()) for x in schedule),
        "accepted_all": [list(wholes[j]) for j in range(len(wholes))
                         if bool(accepted[j])],
    }
    return [list(x) for x in predicted], trace


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{lineno}: text must be a string")
            spans = row.get("spans")
            if not isinstance(spans, list):
                raise ValueError(f"{path}:{lineno}: spans must be a list")
            records.append(row)
    return records


def evaluate_records(records, passes=3):
    tp = fp = fn = exact = 0
    failures = []
    route_totals = {"overflow": 0, "unresolved": 0,
                    "sigma": 0, "pi": 0, "raise": 0}
    for i, row in enumerate(records):
        pred, trace = parse_surface(row["text"], passes=passes)
        gold = {tuple(map(int, x)) for x in row.get("spans", [])}
        got = {tuple(map(int, x)) for x in pred}
        tp += len(gold & got)
        fp += len(got - gold)
        fn += len(gold - got)
        exact += int(gold == got)
        for key in route_totals:
            route_totals[key] += int(trace[key])
        if gold != got and len(failures) < 100:
            failures.append({"row": i, "text": row["text"],
                             "gold": sorted(gold), "predicted": sorted(got),
                             "trace": trace})
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / max(1e-12, precision + recall))
    n = len(records)
    return {
        "sentences": n, "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "precision": precision, "recall": recall,
        "f1": f1, "exact_sentence": exact / max(1, n),
        **route_totals, "failures": failures,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--json", dest="json_path", type=Path)
    ap.add_argument("--require-f1", type=float)
    args = ap.parse_args(argv)
    result = evaluate_records(load_jsonl(args.corpus), passes=args.passes)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(payload + "\n", encoding="utf-8")
    if args.require_f1 is not None and result["f1"] < args.require_f1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
