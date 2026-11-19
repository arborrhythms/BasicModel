#!/usr/bin/env python3
"""The sentence loop of one path: what is read, composed, generated, thought and answered.

Source: doc/Architecture.md, "Whole paths and concept episodes". Embedded there.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_diagrams import *  # noqa: F401,F403

FILENAME = "sentence_loop.svg"


def node(x, y, w, h, lines, fill, stroke, rx=8, mono=False):
    out = [rect(x, y, w, h, fill, stroke=stroke, sw=1.8, rx=rx)]
    top = y + h / 2 + 4.5 - (len(lines) - 1) * 8
    for k, text in enumerate(lines):
        out.append(txt(x + w / 2, top + k * 16, text, fs=12.5, fill=BLUE_DK,
                       ff="Menlo, monospace" if mono else None))
    return "\n".join(out)


def label(x, y, text, anchor="middle", fill=GRY):
    return txt(x, y, text, anchor=anchor, fs=11.5, fill=fill, italic=True)


def make():
    W, H = 1332, 668
    b = []
    # The two systems.
    b.append(rect(200, 96, 850, 262, BLUE_L, stroke=BLUE, sw=1.5, rx=12, op=0.45))
    b.append(txt(216, 118, "System 1: order zero", anchor="start", fs=14, fill=BLUE, bold=True))
    b.append(rect(200, 404, 850, 176, PURP_L, stroke=PURP, sw=1.5, rx=12, op=0.45))
    b.append(txt(216, 426, "System 2: symbols and higher orders", anchor="start", fs=14, fill=PURP, bold=True))

    b.append(node(30, 353, 130, 44, ["input words"], GRY_L, BLUE_DK))
    # System 1.
    b.append(node(230, 130, 220, 52, ["observed field a_0", "(open attention pass)"], "#fff", BLUE))
    b.append(node(610, 136, 120, 40, ["reverse"], TEAL_L, TEAL, mono=True))
    b.append(node(810, 205, 120, 40, ["reverse"], TEAL_L, TEAL, mono=True))
    b.append(node(480, 282, 230, 52, ["knowing: the generated", "order-zero field"], "#fff", BLUE))
    b.append(node(900, 288, 120, 40, ["reverse"], TEAL_L, TEAL, mono=True))
    # System 2.
    b.append(node(230, 465, 140, 52, ["compose"], "#fff", PURP))
    b.append(node(420, 461, 190, 60, ["generate walk +", "occurrence writer"], ORG_L, ORG))
    b.append(node(665, 465, 150, 52, ["closing thought"], "#fff", PURP))
    b.append(node(855, 461, 190, 60, ["generate walk +", "occurrence writer"], ORG_L, ORG))
    # The two costs and the comparison.
    b.append(node(1090, 128, 200, 60, ["R: reconstruction_reverse", "+ reconstruction_order_zero"], GRN_L, GRN_DK, rx=28))
    b.append(node(1090, 280, 200, 56, ["A: answer"], GRN_L, GRN_DK, rx=28))
    b.append(diamond(1190, 491, 216, 116, "", "#fff", BLUE_DK))
    b.append(txt(1190, 487, "keep the cheapest", fs=12.5, fill=BLUE_DK, bold=True))
    b.append(txt(1190, 504, "R + A path", fs=12.5, fill=BLUE_DK, bold=True))

    # Values (solid arrows).
    b.append(ortho([(160, 365), (180, 365), (180, 156), (230, 156)]))
    b.append(ortho([(160, 385), (180, 385), (180, 491), (230, 491)]))
    b.append(ortho([(450, 156), (610, 156)]))
    b.append(ortho([(730, 156), (1090, 156)]))
    b.append(ortho([(370, 491), (420, 491)], PURP, "arrp"))
    b.append(label(395, 452, "end slots", fill=PURP))
    b.append(ortho([(515, 461), (515, 334)], ORG, "arro"))
    b.append(ortho([(710, 298), (770, 298), (770, 225), (810, 225)]))
    b.append(ortho([(930, 225), (1062, 225), (1062, 174), (1090, 174)]))
    b.append(ortho([(300, 517), (300, 556), (740, 556), (740, 517)], PURP, "arrp"))
    b.append(label(410, 572, "understanding", fill=PURP))
    b.append(ortho([(655, 334), (655, 440), (712, 440), (712, 465)], BLUE, "arrb"))
    b.append(label(647, 392, "read", anchor="end", fill=BLUE))
    b.append(ortho([(768, 465), (768, 378), (690, 378), (690, 334)], PURP, "arrp"))
    b.append(label(776, 420, "intuit deposits", anchor="start", fill=PURP))
    b.append(ortho([(815, 491), (855, 491)], PURP, "arrp"))
    b.append(label(835, 452, "conclusion", fill=PURP))
    b.append(ortho([(960, 461), (960, 328)], ORG, "arro"))
    b.append(ortho([(1020, 308), (1090, 308)]))
    b.append(ortho([(1290, 158), (1306, 158), (1306, 491), (1298, 491)]))
    b.append(ortho([(1190, 336), (1190, 433)]))

    # One set of parameters (dotted links, no arrowhead).
    b.append(path("M 670 176 L 670 192 L 870 192 L 870 205", stroke=TEAL, sw=1.4, dash="3,4"))
    b.append(label(770, 187, "one decoder", fill=TEAL))
    b.append(path("M 900 245 L 940 288", stroke=TEAL, sw=1.4, dash="3,4"))
    b.append(label(930, 266, "one decoder", anchor="start", fill=TEAL))
    b.append(path("M 515 521 L 515 538 L 950 538 L 950 521", stroke=ORG, sw=1.4, dash="3,4"))
    b.append(label(840, 533, "one walk", fill=ORG))

    b.append(txt(W // 2, 622, "Solid arrows carry values. Dotted links join calls of one set of parameters:",
                 fs=12.5, fill=GRY, italic=True))
    b.append(txt(W // 2, 642, "one reverse decoder (SentenceField.reconstruct, OutputOccurrences.realize) "
                 "and one generate walk (BasicModel.generate).", fs=12.5, fill=GRY, italic=True))
    return figure(W, H, "The sentence loop of one path",
                  "Architecture.md: Whole paths and concept episodes · each stream keeps its cheapest R + A path",
                  b, MARKERS)


if __name__ == "__main__":
    write(FILENAME, make())
