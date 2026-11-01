#!/usr/bin/env python3
"""Memory: what is in focus, familiar and known, and the long-term strata the kept close writes.

Sources: doc/Architecture.md (the Rows, Identity and MTM bullets; "LTM0 occurrence rows";
the familiar tier), doc/STM.md, doc/Orders.md §10. Embedded in doc/STM.md.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_diagrams import *  # noqa: F401,F403

FILENAME = "memory_strata.svg"


def badge(x, y, n, color=BLUE_DK):
    return circle(x, y, 10, "#fff", color, sw=1.5, op=1) + txt(x, y + 4, str(n), fs=11, fill=color, bold=True)


def make():
    W, H = 1240, 836
    b = []
    stm = (30, 92, 320, 228)
    mtm = (450, 92, 380, 252)
    con = (910, 92, 300, 228)
    ltm0 = (30, 448, 380, 196)
    ltm1 = (450, 448, 380, 196)
    folds = (910, 448, 300, 196)
    b.append(card(*stm, "STM · in focus", [
        "a fixed working set of stmCapacity (8)",
        "chunks, newest at slot 0; the same",
        "tensors serve parsing, recent entries,",
        "recall and serial thought",
        "each chunk: content, references and",
        "its two evidence lanes (chunk_truth)",
        "MTM symbols prepended at the open",
        "bind's intersection, computed on read"], BLUE_L, BLUE, step=18.5))
    b.append(card(*mtm, "MTM · familiar", [
        "ConceptualSpace.mtm, capacity 64 tokens",
        "each token: kind, access count, last model",
        "time, introduction and use addresses",
        "its current value: the state after its latest verb",
        "names mapped to identities; event histories;",
        "an actual and alternative timelines",
        "familiar-tier candidates with sparse support",
        "expires after 16 kept closes unreferenced;",
        "cleared at the epoch boundary"], TEAL_L, TEAL, step=18.5))
    b.append(card(*con, "Concepts · known", [
        "a minted row at the shared index",
        "its content: the fold of its",
        "occurrences, rebuildable",
        "sigma/pi definitions and an order",
        "ids and ordered records owned",
        "by the ConceptAllocator"], "#f5eef8", PURP))
    b.append(card(*ltm0, "LTM0 · the occurrence store", [
        "one row per kept occurrence: its slot,",
        ".when band, truth_plus and truth_minus,",
        "access count and surprise",
        "the occurrence index: stable addresses,",
        "stream, document and content keys",
        "order read from the referent"], GRY_L, BLUE_DK))
    b.append(card(*ltm1, "LTM1 · the relation stratum", [
        "a row is two tagged references; its kind",
        "is read from the referents: definition,",
        "attribution or conditional",
        "owned with the symbol taxonomy by the",
        "ConceptAllocator",
        "LTM2 names relations over relations"], GRY_L, BLUE_DK))
    b.append(card(*folds, "Folds, not rows", [
        "parthood, subsumption and kind",
        "statements are folds in the",
        "meronomy and the taxonomy over",
        "symbols", "equality is membership in",
        "a singleton"], GRY_L, GRY))

    # 1 kept close STM -> MTM; 2 bind reads MTM; 3 promotion; 4 occurrence row;
    # 5 definition row; 6 raise; 7 recall by cue; 8 folds organize concepts.
    b.append(ortho([(350, 160), (450, 160)], TEAL, "arrt")); b.append(badge(400, 146, 1, TEAL))
    b.append(ortho([(450, 250), (350, 250)], BLUE, "arrb", dash="7,4")); b.append(badge(400, 236, 2, BLUE))
    b.append(ortho([(830, 190), (910, 190)], PURP, "arrp")); b.append(badge(870, 176, 3, PURP))
    b.append(ortho([(110, 320), (110, 448)], BLUE_DK, "arrd")); b.append(badge(126, 386, 4))
    b.append(ortho([(250, 448), (250, 320)], BLUE, "arrb", dash="7,4")); b.append(badge(266, 386, 7, BLUE))
    b.append(ortho([(330, 320), (330, 396), (560, 396), (560, 448)], BLUE_DK, "arrd")); b.append(badge(445, 396, 5))
    b.append(ortho([(760, 448), (760, 404), (1000, 404), (1000, 320)], PURP, "arrp")); b.append(badge(880, 404, 6, PURP))
    b.append(line(1140, 320, 1140, 448, stroke=GRY, sw=1.8, dash="3,3")); b.append(badge(1140, 384, 8, GRY))

    legend = [
        (1, "the kept close writes bindings, current values and opened tokens; trial-local records before it"),
        (2, "bind reads an accessible antecedent, which must beat a zero-logit dummy"),
        (3, "promotion by recurrence: conceptMintCount observations within the horizon, outside the both corner"),
        (4, "the kept close writes the occurrence row"),
        (5, "a kept definition writes its relation row; it can mint its subject"),
        (6, "raise: raiseCount distinct members sharing a side make a class one order up"),
        (7, "recall by cue; a query with only its .when filled returns the entries at that time"),
        (8, "the meronomy and the taxonomy organize the concepts"),
    ]
    for k, (n, text) in enumerate(legend):
        x = 40 + (k // 4) * 600
        y = 690 + (k % 4) * 30
        b.append(badge(x, y - 4, n, GRY))
        b.append(txt(x + 18, y, text, anchor="start", fs=12, fill=BLUE_DK))
    b.append(txt(W // 2, H - 12, "in focus: the STM frames · familiar: the MTM pool · known: a minted concept",
                 fs=12, fill=GRY, italic=True))
    return figure(W, H, "Memory: in focus, familiar, known, and the long-term strata",
                  "Architecture.md: Rows, Identity and the LTM0 occurrence rows · STM.md · Orders.md §10",
                  b, MARKERS)


if __name__ == "__main__":
    write(FILENAME, make())
