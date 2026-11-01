#!/usr/bin/env python3
"""A concept: a graded DNF definition (sigma over pi), one signed code, two evidence lanes.

Sources: doc/Architecture.md ("a concept is sigma over pi", 2026-09-21/22), doc/Spaces.md
(ConceptualSpace: Geometry, the symbolic phase), bin/Interpret.py activate_code. Widths are read
from data/BasicModel.xml. Embedded in doc/Spaces.md, ConceptualSpace.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_diagrams import *  # noqa: F401,F403

FILENAME = "concept_geometry.svg"


def sup(s):
    return f"<tspan baseline-shift='super' font-size='70%'>{s}</tspan>"


def make():
    W, H = 1240, 650
    b = []
    panels = [(30, "Definition: graded DNF", PURP), (440, "The code", BLUE), (850, "Two symbols, two lanes", ORG)]
    for x, title, col in panels:
        b.append(rect(x, 84, 380, 520, "#fff", stroke=col, sw=1.6, rx=10))
        b.append(title_bar(x, 84, 380, 32, title, bg=col, fs=14))

    # A. sigma over pi.
    node = lambda cx, cy, label, col: (circle(cx, cy, 22, "#fff", col, sw=1.8, op=1) +
                                       txt(cx, cy + 5, label, fs=13, fill=col, bold=True))
    edges = [((220, 150), (220, 218)), ((220, 218), (130, 300)), ((220, 218), (310, 300)),
             ((130, 300), (80, 384)), ((130, 300), (175, 384)), ((310, 300), (265, 384)), ((310, 300), (360, 384))]
    for (x1, y1), (x2, y2) in edges:
        b.append(line(x1, y1, x2, y2, stroke=GRY_MID, sw=1.6))
    b.append(node(220, 150, "c", PURP))
    b.append(node(220, 218, "σ", PURP))
    b.append(node(130, 300, "π", BLUE))
    b.append(node(310, 300, "π", BLUE))
    for cx, lab, col in [(80, "part", BLUE), (175, "prop", GRN), (265, "part", BLUE), (360, "prop", GRN)]:
        b.append(rect(cx - 30, 372, 60, 26, BLUE_L if col == BLUE else GRN_L, stroke=col, sw=1.2, rx=6))
        b.append(txt(cx, 390, lab, fs=12, fill=BLUE_DK))
    b.append(txt(262, 222, "max over cases", anchor="start", fs=11.5, fill=PURP, italic=True))
    b.append(txt(220, 344, "min within a case", fs=11.5, fill=BLUE, italic=True))
    for k, row in enumerate(["order 0: sigma over pi of memberships in",
                             "PartSpace parts and WholeSpace properties;",
                             "each pole reduced independently, with",
                             "nonnegative exponents; pi only at order 0",
                             "higher orders: sigma alone over the",
                             "previous order's symbols, the taxonomy"]):
        b.append(txt(46, 434 + k * 22, row, anchor="start", fs=12.5, fill=BLUE_DK))

    # B. code = signed activation x softplus(atom).
    b.append(txt(630, 146, "code = signed activation × softplus(atom)", fs=13.5, fill=BLUE, bold=True))
    atom = [0.55, 0.3, 0.8, 0.45, 0.65, 0.25, 0.7, 0.5]
    b.append(bar_vector(462, 216, atom, 12, 60, 3, colors=[BLUE] * 8))
    b.append(txt(596, 202, "× (+0.8)", anchor="start", fs=13, fill=GRN_DK, bold=True))
    b.append(bar_vector(676, 216, [0.8 * v for v in atom], 12, 60, 3, colors=[GRN] * 8))
    b.append(txt(522, 236, "softplus(atom) > 0", fs=11.5, fill=GRY))
    b.append(txt(736, 236, "present", fs=11.5, fill=GRN_DK))
    b.append(bar_vector(462, 300, atom, 12, 60, 3, colors=[BLUE] * 8))
    b.append(txt(596, 296, "× (−0.6)", anchor="start", fs=13, fill=RED_DK, bold=True))
    b.append(bar_vector(676, 300, [-0.6 * v for v in atom], 12, 60, 3))
    b.append(txt(736, 352, "anti-present", fs=11.5, fill=RED_DK))
    get = model_reader(REFERENCE_MODEL)
    cdim, mw = int(get("ConceptualSpace", "nDim")), int(get("ConceptualSpace", "meaningWidth"))
    form = cdim - mw
    bx, bw = 470, 320
    fw = max(30, int(bw * form / cdim))
    b.append(rect(bx, 388, fw, 30, BLUE_L, stroke=BLUE, sw=1.2, rx=0))
    b.append(rect(bx + fw, 388, bw - fw, 30, PURP_L, stroke=PURP, sw=1.2, rx=0))
    b.append(txt(bx + fw / 2, 408, f"{form}", fs=12, fill=BLUE_DK, bold=True))
    b.append(txt(bx + fw + (bw - fw) / 2, 408, f"meaning: {mw // 2} pole pairs", fs=12, fill=BLUE_DK, bold=True))
    b.append(txt(bx, 436, f"form {form} + meaning {mw} = {cdim}  (data/BasicModel.xml)", anchor="start", fs=11.5, fill=GRY))
    for k, row in enumerate(["the sign of the activation: present or",
                             "anti-present; its magnitude: certainty",
                             "two symbols share one stored code"]):
        b.append(txt(456, 480 + k * 22, row, anchor="start", fs=12.5, fill=BLUE_DK))

    # C. the two lanes and their four corners.
    b.append(txt(1040, 146, f"symbol 2i = +a{sup('+')} · code{sub('i')}", fs=13.5, fill=ORG, bold=True))
    b.append(txt(1040, 170, f"symbol 2i+1 = −a{sup('−')} · code{sub('i')}", fs=13.5, fill=ORG, bold=True))
    sx, sy, s = 930, 210, 220
    b.append(rect(sx, sy, s, s, "#fffaf3", stroke=ORG, sw=1.2, rx=0))
    b.append(line(sx, sy + s, sx + s + 16, sy + s, stroke=GRY, sw=1.4))
    b.append(line(sx, sy + s, sx, sy - 16, stroke=GRY, sw=1.4))
    b.append(txt(sx + s + 22, sy + s + 5, f"a{sup('+')}", anchor="start", fs=14, fill=GRY, bold=True))
    b.append(txt(sx, sy - 24, f"a{sup('−')}", fs=14, fill=GRY, bold=True))
    corners = [(sx, sy + s, "neither", GRY, "middle", 0, 26), (sx + s, sy + s, "true", GRN_DK, "middle", 0, 26),
               (sx, sy, "false", RED_DK, "end", -12, 5), (sx + s, sy, "both", PURP, "start", 12, 5)]
    for cx, cy, lab, col, anchor, dx, dy in corners:
        b.append(circle(cx, cy, 6, col, col, sw=1, op=1))
        b.append(txt(cx + dx, cy + dy, lab, anchor=anchor, fs=13, fill=col, bold=True))
    for k, row in enumerate(["truth_plus and truth_minus are two",
                             "independent positive lanes, not one",
                             "signed number: both never becomes neither",
                             "(Interpret.activate_code)"]):
        b.append(txt(866, 488 + k * 22, row, anchor="start", fs=12.5, fill=BLUE_DK))
    return figure(W, H, "A concept: sigma over pi, one signed code, two evidence lanes",
                  "Architecture.md, a concept is sigma over pi · Spaces.md, ConceptualSpace geometry",
                  b, MARKERS)


if __name__ == "__main__":
    write(FILENAME, make())
