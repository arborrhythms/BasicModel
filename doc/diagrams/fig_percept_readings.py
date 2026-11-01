#!/usr/bin/env python3
"""The readings of a percept: interpret it or not; if so, the interpretation is a sign or a symbol.

Source: doc/Orders.md §11 ("The three readings forced", refined 2026-11-01), doc/Language.md
(the amendment under interpret), thinking spec §32.3-§32.4 item 22. Embedded in doc/Orders.md §11.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_diagrams import *  # noqa: F401,F403

FILENAME = "percept_readings.svg"


def make():
    W, H = 1180, 690
    b = []
    b.append(card(380, 84, 420, 62, "A percept at a leaf", [
        "a word-occurrence, read as its form; with a field, a native percept"], BLUE_L, BLUE,
        center=True, fs=12))
    b.append(diamond(590, 214, 190, 84, "interpret?", "#fff", BLUE_DK, fs=15))
    b.append(ortho([(590, 146), (590, 172)], BLUE_DK, "arrd"))
    for k, row in enumerate(["optional in the architecture;", "initially forced in reading mode,",
                             "since words are so often", "meaningful beyond themselves"]):
        b.append(txt(700, 196 + k * 18, row, anchor="start", fs=12.5, fill=BLUE_DK, italic=True))

    noop = (30, 300, 360, 196)
    b.append(ortho([(495, 214), (210, 214), (210, noop[1])], TEAL, "arrt"))
    b.append(txt(470, 206, "no", fs=13, fill=TEAL, bold=True))
    b.append(card(*noop, "no-op · Peirce's icon", [
        "the percept stands for itself",
        "value: its form code (interpret_I1)",
        "token: its own occurrence, content",
        "the form, kind the null-concept",
        "reverse: the form itself, so a mention",
        "is said as it was",
        "e.g. 'woof', a quoted word, a word",
        "heard only as a sound"], TEAL_L, TEAL, step=19))

    b.append(diamond(790, 340, 210, 80, "a sign or a symbol?", "#fff", BLUE_DK, fs=13))
    b.append(ortho([(590, 256), (590, 278), (790, 278), (790, 300)], BLUE_DK, "arrd"))
    b.append(txt(604, 274, "yes", anchor="start", fs=13, fill=BLUE_DK, bold=True))
    sign = (420, 430, 350, 168)
    symbol = (800, 430, 350, 168)
    b.append(ortho([(685, 340), (595, 340), (595, sign[1])], ORG, "arro"))
    b.append(ortho([(895, 340), (975, 340), (975, symbol[1])], PURP, "arrp"))
    b.append(card(*sign, "sign · Peirce's index", [
        "the percept points at something co-present",
        "token: the pointed-at occurrence, kind",
        "the null-concept",
        "content: zero in a text; with a field,",
        "the pointed-at percept",
        "e.g. 'this', 'that'"], ORG_L, ORG, step=19))
    b.append(card(*symbol, "symbol · Peirce's symbol", [
        "the word's conventional object",
        "the interpreted leaf stands as its",
        "object, with no rule of its own",
        "a pending word is minted at its first",
        "kept definition",
        "e.g. 'dog', 'bark'"], "#f5eef8", PURP, step=19))
    b.append(txt(30, 530, "Certificate (v):", anchor="start", fs=12, fill=TEAL, bold=True))
    b.append(txt(30, 548, "“The dog went woof. That was a bark.”", anchor="start", fs=12, fill=BLUE_DK, italic=True))
    b.append(txt(30, 566, "the no-op at 'woof' opens a token that", anchor="start", fs=12, fill=GRY))
    b.append(txt(30, 582, "'that' binds and the definition kinds bark", anchor="start", fs=12, fill=GRY))

    b.append(rect(30, 620, W - 60, 52, GRY_L, stroke=GRY_MID, sw=1, rx=8))
    b.append(txt(46, 641, "Built (item 22, Codex, 2026-11-01): interpret mandatory in reading mode; sign at every "
                 "interpreted leaf; the no-op's own-occurrence token when the forcing is lifted on the layer instance; no parameter.",
                 anchor="start", fs=12, fill=BLUE_DK))
    b.append(txt(46, 661, "The forcing stays on outside certificate (v). Lifting it in production is a later decision, "
                 "with a field (FutureWork, \"Renamed sign\").",
                 anchor="start", fs=12, fill=BLUE_DK))
    return figure(W, H, "The readings of a percept: interpret or not; if so, a sign or a symbol",
                  "Orders.md §11 · Language.md, interpret · thinking spec §32.3–§32.4, item 22 (Alec, 2026-11-01)",
                  b, MARKERS)


if __name__ == "__main__":
    write(FILENAME, make())
