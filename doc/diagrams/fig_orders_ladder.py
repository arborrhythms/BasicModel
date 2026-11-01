#!/usr/bin/env python3
"""Orders are learned: definitions raise by equivalence class, NP + VP lowers, determiners select.

Source: doc/Orders.md §10 (decided 2026-10-21, amended 2026-10-22/23) and the raise rule in
doc/Architecture.md (raiseCount, raiseEvidence). Embedded in doc/Orders.md §10.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_diagrams import *  # noqa: F401,F403

FILENAME = "orders_ladder.svg"


def chip(x, y, label, fill, stroke, w=None, fs=13):
    w = w or int(len(label) * fs * 0.56) + 22
    return rect(x, y, w, 30, fill, stroke=stroke, sw=1.4, rx=15) + txt(x + w / 2, y + 20, label, fs=fs, fill=BLUE_DK)


def make():
    W, H = 1180, 670
    b = []
    bands = [  # order, label, y, fill, stroke, chips
        (2, "order 2 · kinds", 96, ORG_L, ORG, ["cat", "dodo", "animal"]),
        (1, "order 1 · individuals", 222, "#f5eef8", PURP, ["Felix", "Ciao", "word types, by form sameness"]),
        (0, "order 0 · occurrences", 348, BLUE_L, BLUE, ["the event 'this' points at", "'Felix runs'",
                                                         "sign and no-op tokens"]),
    ]
    X0, X1 = 330, 890
    for order, label, y, fill, stroke, chips in bands:
        b.append(rect(X0, y, X1 - X0, 100, fill, stroke=stroke, sw=1.8, rx=12))
        b.append(txt(X0 + 16, y + 26, label, anchor="start", fs=15, fill=stroke, bold=True))
        x = X0 + 16
        for c in chips:
            b.append(chip(x, y + 50, c, "#fff", stroke))
            x += int(len(c) * 13 * 0.56) + 22 + 12

    # Raise, on the left: two IS statements with a shared side.
    b.append(ortho([(310, 362), (310, 290)], PURP, "arrp", sw=2.4))
    b.append(ortho([(310, 236), (310, 164)], ORG, "arro", sw=2.4))
    b.append(txt(296, 310, "“This is Felix.", anchor="end", fs=13, fill=BLUE_DK, italic=True))
    b.append(txt(296, 328, "That is Felix.”", anchor="end", fs=13, fill=BLUE_DK, italic=True))
    b.append(txt(296, 346, "Felix: the class of his occurrences", anchor="end", fs=11.5, fill=GRY))
    b.append(txt(296, 184, "“Felix is a cat.", anchor="end", fs=13, fill=BLUE_DK, italic=True))
    b.append(txt(296, 202, "Ciao is a cat.”", anchor="end", fs=13, fill=BLUE_DK, italic=True))
    b.append(txt(296, 220, "cat: a kind", anchor="end", fs=11.5, fill=GRY))
    b.append(txt(196, 122, "DEFINITIONS RAISE", fs=14, fill=PURP, bold=True))

    # Lower, on the right: NP + VP lands among the members.
    b.append(ortho([(910, 164), (910, 236)], RED, "arrr", sw=2.4))
    b.append(ortho([(910, 290), (910, 362)], RED, "arrr", sw=2.4))
    b.append(txt(924, 184, "“Dodos died out.”", anchor="start", fs=13, fill=BLUE_DK, italic=True))
    b.append(txt(924, 202, "a change across the kind's", anchor="start", fs=11.5, fill=GRY))
    b.append(txt(924, 218, "members, at order 1", anchor="start", fs=11.5, fill=GRY))
    b.append(txt(924, 310, "“Felix runs.”", anchor="start", fs=13, fill=BLUE_DK, italic=True))
    b.append(txt(924, 328, "an occurrence of Felix,", anchor="start", fs=11.5, fill=GRY))
    b.append(txt(924, 344, "at order 0", anchor="start", fs=11.5, fill=GRY))
    b.append(txt(1010, 122, "NP + VP LOWERS", fs=14, fill=RED, bold=True))

    b.append(card(30, 476, 360, 164, "The raise", [
        "an IS row whose shared side gathers",
        "raiseCount distinct members (different",
        "tokens or values; events at different",
        "places or times), with evidence above",
        "raiseEvidence, makes a class one order",
        "above its members; the word denotes it"], "#f5eef8", PURP, fs=12.5))
    b.append(card(410, 476, 360, 164, "Selection, not lowering", [
        "the determiner selects: the phrase takes",
        "the order of what it picks ('the cat' is",
        "an individual)",
        "the modal withholds the lowering: the",
        "clause stays above order 0, extending an",
        "alternative timeline of the one identity"], GRY_L, BLUE_DK, fs=12.5))
    b.append(card(790, 476, 360, 164, "Raises nothing", [
        "identity: 'Paris is the capital of",
        "France'; 'y is three. x is three.'",
        "sign and no-op tokens are occurrences",
        "at order 0 and never raise",
        "the null-concept has no order",
        ""], GRY_L, BLUE_DK, fs=12.5))
    return figure(W, H, "Orders are learned: definitions raise, NP + VP lowers",
                  "Orders.md §10 · orders live in the concepts, not the grammar; parts of speech stay order-free",
                  b, MARKERS)


if __name__ == "__main__":
    write(FILENAME, make())
