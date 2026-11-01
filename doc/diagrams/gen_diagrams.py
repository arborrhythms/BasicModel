#!/usr/bin/env python3
"""Generate WikiOracle conceptual diagrams as SVG files.

    python gen_diagrams.py                        # write every diagram, including fig_*.py
    python gen_diagrams.py grammar_operators.svg  # write only the named ones
    python fig_orders_ladder.py                   # one figure script on its own

This file holds the shared palette and drawing primitives, the figures that
predate the one-figure scripts, and the grammar operator sheet. Each fig_*.py
beside it defines FILENAME and make() and imports the primitives from here.
Every figure is embedded in at least one doc/*.md page.
"""
import os, math, sys
from html import escape

DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE_DK = "#2c3e50"; BLUE    = "#2980b9"; BLUE_L  = "#d6eaf8"
PURP    = "#8e44ad"; PURP_L  = "#e8daef"
ORG     = "#ca6f1e"; ORG_L   = "#fdebd0"
GRN_DK  = "#1a7a38"; GRN     = "#27ae60"; GRN_L   = "#d5f5e3"
RED_DK  = "#922b21"; RED     = "#c0392b"; RED_L   = "#fadbd8"
GRY     = "#626567"; GRY_L   = "#f2f3f4"; GRY_MID = "#aab7b8"
TEAL    = "#16a085"; TEAL_L  = "#d1f2eb"
BG      = "#fafafa"

FF = "font-family='Georgia, serif'"

# ── Low-level SVG helpers ──────────────────────────────────────────────────────

def svg(w, h, content, extra=""):
    return (f"<svg width='{w}' height='{h}' xmlns='http://www.w3.org/2000/svg'"
            f" {FF} {extra}>\n  <rect width='{w}' height='{h}' fill='{BG}'/>\n"
            f"{content}\n</svg>")

def g(content, transform=""):
    t = f" transform='{transform}'" if transform else ""
    return f"<g{t}>{content}</g>"

def rect(x, y, w, h, fill, stroke=None, sw=1.5, rx=6, op=1.0, dash=""):
    s = f" stroke='{stroke}' stroke-width='{sw}'" if stroke else ""
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{fill}'"
            f" fill-opacity='{op}' rx='{rx}'{s}{d}/>")

def txt(x, y, s, anchor="middle", fs=14, fill=BLUE_DK, bold=False, italic=False, ff=None):
    fw = " font-weight='bold'" if bold else ""
    fi = " font-style='italic'" if italic else ""
    family = f" font-family='{ff}'" if ff else ""
    return (f"<text x='{x}' y='{y}' text-anchor='{anchor}' font-size='{fs}'"
            f" fill='{fill}'{fw}{fi}{family}>{s}</text>")

def line(x1, y1, x2, y2, stroke=GRY_MID, sw=1.5, dash=""):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{stroke}' stroke-width='{sw}'{d}/>"

def arrow_marker(mid, color=BLUE_DK, size=8):
    return (f"<marker id='{mid}' markerWidth='{size}' markerHeight='{size*0.7}'"
            f" refX='{size-1}' refY='{size*0.35}' orient='auto'>"
            f"<polygon points='0 0, {size} {size*0.35}, 0 {size*0.7}' fill='{color}'/></marker>")

def arrow(x1, y1, x2, y2, color=BLUE_DK, sw=1.8, mid="arrd"):
    return (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}'"
            f" stroke='{color}' stroke-width='{sw}' marker-end='url(#{mid})'/>")

def ellipse(cx, cy, rx, ry, fill, stroke, sw=2, op=0.6):
    return (f"<ellipse cx='{cx}' cy='{cy}' rx='{rx}' ry='{ry}' fill='{fill}'"
            f" fill-opacity='{op}' stroke='{stroke}' stroke-width='{sw}'/>")

def circle(cx, cy, r, fill, stroke, sw=2, op=0.8):
    return (f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{fill}'"
            f" fill-opacity='{op}' stroke='{stroke}' stroke-width='{sw}'/>")

def path(d, fill="none", stroke=BLUE_DK, sw=1.5, dash=""):
    da = f" stroke-dasharray='{dash}'" if dash else ""
    return f"<path d='{d}' fill='{fill}' stroke='{stroke}' stroke-width='{sw}'{da}/>"

def hbar(x, y, val, maxval, width, height, fill, bg=GRY_L, label=""):
    """Horizontal progress bar."""
    frac = max(0, min(1, val / maxval))
    s = rect(x, y, width, height, bg, stroke=GRY_MID, sw=1, rx=3)
    if frac > 0:
        s += rect(x, y, width * frac, height, fill, rx=3)
    if label:
        s += txt(x + width + 6, y + height//2 + 5, label, anchor="start", fs=12, fill=GRY)
    return s

def title_bar(x, y, w, h, label, bg=BLUE_DK, fg="white", fs=15, rx=8):
    """Rounded title bar for a box."""
    return (rect(x, y, w, h, bg, rx=rx) +
            rect(x, y + h//2, w, h//2, bg, rx=0) +
            txt(x + w//2, y + h - 7, label, fs=fs, fill=fg, bold=True))

def ortho(points, color=BLUE_DK, mid="arrd", sw=1.8, dash=""):
    """An arrow along straight segments through ``points``."""
    d = "M " + " L ".join(f"{px} {py}" for px, py in points)
    da = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{sw}'{da}"
            f" marker-end='url(#{mid})'/>")

def card(x, y, w, h, title, rows, fill, stroke, title_fs=15, fs=12.5, step=19, center=False, sw=1.8):
    """A rounded box with a bold title and one text row per entry."""
    out = [rect(x, y, w, h, fill, stroke=stroke, sw=sw, rx=10)]
    if center:
        out.append(txt(x + w/2, y + 24, title, fs=title_fs, fill=stroke, bold=True))
        out += [txt(x + w/2, y + 48 + k*step, r, fs=fs, fill=BLUE_DK) for k, r in enumerate(rows)]
    else:
        out.append(txt(x + 14, y + 24, title, anchor="start", fs=title_fs, fill=stroke, bold=True))
        out += [txt(x + 14, y + 48 + k*step, r, anchor="start", fs=fs, fill=BLUE_DK) for k, r in enumerate(rows)]
    return "\n".join(out)

def diamond(cx, cy, w, h, label, fill, stroke, fs=13):
    pts = f"{cx},{cy - h/2} {cx + w/2},{cy} {cx},{cy + h/2} {cx - w/2},{cy}"
    return (f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-width='1.8'/>" +
            txt(cx, cy + 5, label, fs=fs, fill=stroke, bold=True))

def figure(W, H, title, subtitle, body, markers=(("arrd", BLUE_DK),)):
    """The common frame of the architecture figures: title bar, subtitle, body."""
    defs = "<defs>" + "".join(arrow_marker(m, c) for m, c in markers) + "</defs>"
    head = [rect(0, 0, W, 44, BLUE_DK, rx=0), txt(W//2, 29, title, fs=18, fill="white", bold=True)]
    if subtitle:
        head.append(txt(W//2, 66, subtitle, fs=13, fill=GRY, italic=True))
    return svg(W, H, "\n".join([defs, *head, *body]))

MARKERS = (("arrd", BLUE_DK), ("arrb", BLUE), ("arrg", GRN), ("arro", ORG),
           ("arrp", PURP), ("arrr", RED), ("arrt", TEAL))


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 – TERNARY TRUTH TABLES
# ══════════════════════════════════════════════════════════════════════════════
# The tables are computed, not typed: each function below mirrors one kernel in
# bin/Layers.py (class Ops) exactly on the three ternary values.

TERNARY = ["+1", "0", "−1"]
_TV = {"+1": 1.0, "0": 0.0, "−1": -1.0}

def _sign(v):
    return (v > 0) - (v < 0)

def ops_intersection(x, y):
    """Ops.intersection (default): min, a zero coordinate is silent."""
    return y if x == 0 else x if y == 0 else min(x, y)

def ops_union(x, y):
    """Ops.union (default, kind='strict'): elementwise max."""
    return max(x, y)

def ops_radmin(x, y):
    """Ops._radmin (kind='radial'): same sign -> sign * min |.|, else 0."""
    return _sign(x) * min(abs(x), abs(y)) if x * y > 0 else 0.0

def ops_radmax(x, y):
    """Ops._radmax (kind='radial'): same sign -> sign * max |.|; zero passes the other through."""
    core = _sign(x) * max(abs(x), abs(y)) if x * y > 0 else 0.0
    return core + (y if x == 0 else 0.0) + (x if y == 0 else 0.0)

def ops_part_scalar(x, y):
    """Ops._part_kernel(scalar=True): clipped cosine in [0, 1]; part(0, y) = 1, part(x, 0) = 0."""
    if x == 0:
        return 1.0
    if y == 0:
        return 0.0
    return max(0.0, x * y) / (abs(x) * abs(y))

def ops_non(x):
    """Ops._non_kernel (bitonic): the triangular residual 1 - |x|."""
    return 1.0 - abs(max(-1.0, min(1.0, x)))

def ternary_label(v):
    return "+1" if v > 0 else "−1" if v < 0 else "0"

def unit_label(v):
    return "1" if v >= 1 else "0" if v <= 0 else f"{v:.2f}"

SIGNED = {"+1": (GRN_L, GRN_DK), "0": (GRY_L, GRY), "−1": (RED_L, RED_DK)}
UNIT = {"1": (TEAL_L, TEAL), "0": (GRY_L, GRY)}
NONMASS = {"1": (PURP_L, PURP), "0": (GRY_L, GRY)}

def cell_color(v, palette=SIGNED):
    return palette.get(v, ("#fff", BLUE_DK))

def truth_cell(x, y, w, h, val, header=False, palette=SIGNED):
    bg = BLUE_DK if header else cell_color(val, palette)[0]
    fg = "white" if header else cell_color(val, palette)[1]
    return (rect(x, y, w, h, bg, stroke="#ccc", sw=1, rx=0) +
            txt(x + w//2, y + h//2 + 5, val, fs=14, fill=fg, bold=not header))

def binary_table(x, y, op_label, op, label=ternary_label, palette=SIGNED, cell_w=60, cell_h=36):
    """A 3×3 binary truth table computed from ``op`` over the ternary values."""
    out = [txt(x + (4*cell_w)//2, y - 8, op_label, fs=13, fill=BLUE_DK, bold=True),
           truth_cell(x, y, cell_w, cell_h, "a \\ b", header=True)]
    for j, v in enumerate(TERNARY):
        out.append(truth_cell(x + (j+1)*cell_w, y, cell_w, cell_h, v, header=True))
    for i, a in enumerate(TERNARY):
        ry = y + (i+1)*cell_h
        out.append(truth_cell(x, ry, cell_w, cell_h, a, header=True))
        for j, b in enumerate(TERNARY):
            out.append(truth_cell(x + (j+1)*cell_w, ry, cell_w, cell_h,
                                  label(op(_TV[a], _TV[b])), palette=palette))
    return "\n".join(out)

def unary_table(x, y, op_label, col_label, op, label=ternary_label, palette=SIGNED, cell_w=70, cell_h=36):
    """Unary table: a | op(a), computed from ``op``."""
    out = [txt(x + cell_w, y - 8, op_label, fs=13, fill=BLUE_DK, bold=True),
           truth_cell(x, y, cell_w, cell_h, "a", header=True),
           truth_cell(x + cell_w, y, cell_w, cell_h, col_label, header=True)]
    for i, a in enumerate(TERNARY):
        ry = y + (i+1)*cell_h
        out.append(truth_cell(x, ry, cell_w, cell_h, a))
        out.append(truth_cell(x + cell_w, ry, cell_w, cell_h, label(op(_TV[a])), palette=palette))
    return "\n".join(out)

def make_ternary():
    W, H = 1000, 742
    parts = ["<defs>" + arrow_marker("arrd", BLUE_DK) + "</defs>"]
    parts.append(rect(0, 0, W, 44, BLUE_DK, rx=0))
    parts.append(txt(W//2, 29, "Ternary Logic Operators", fs=18, fill="white", bold=True))
    parts.append(txt(W//2, 68, "+1 = true  ·  0 = unknown  ·  −1 = false  ·  "
                     "every cell computed from the Ops kernels in bin/Layers.py", fs=13, fill=GRY))

    # Row 1: the two negations, with the note that separates them.
    parts.append(unary_table(40, 110, "NOT  ( neg(a) = −a )", "¬a", lambda a: -a))
    parts.append(unary_table(240, 110, "NON  ( non(a) = 1 − |a| )", "non(a)", ops_non,
                             label=unit_label, palette=NONMASS))
    note = ["NOT inverts an assertion; NON withdraws it.",
            "non(a) is the indeterminate mass, not a truth value:",
            "true + false + non = 1, so non(0) = 1 and non(±1) = 0.",
            "The monotonic form is relu(a − τ) when a threshold",
            "τ is supplied, and 0 otherwise."]
    parts.append(rect(470, 92, 500, 162, GRY_L, stroke=GRY_MID, sw=1, rx=8))
    for k, row in enumerate(note):
        parts.append(txt(490, 122 + k*27, row, anchor="start", fs=13, fill=BLUE_DK))

    # Row 2: the binary operators as built.
    parts.append(line(0, 274, W, 274, stroke="#ddd", sw=1.5))
    parts.append(txt(30, 298, "As built: the default kernels behind Ops.intersection, "
                     "Ops.union and Ops.part", anchor="start", fs=13, fill=GRY, italic=True))
    parts.append(binary_table(30, 330, "Intersection  ( min; a 0 is silent )", ops_intersection))
    parts.append(binary_table(330, 330, "Union  ( elementwise max )", ops_union))
    parts.append(binary_table(630, 330, "PART  ( scalar form, range [0, 1] )", ops_part_scalar,
                              label=unit_label, palette=UNIT))
    parts.append(txt(750, 494, "1 when a ⊆ b; 0 is part of everything; opposite signs → 0",
                     fs=11, fill=GRY, italic=True))

    # Row 3: the radial forms, which survive only under kind='radial'.
    parts.append(line(0, 512, W, 512, stroke="#ddd", sw=1.5))
    parts.append(txt(30, 536, "kind='radial' only: the hard radial forms of Logic.md §7",
                     anchor="start", fs=13, fill=GRY, italic=True))
    parts.append(binary_table(30, 568, "RadMin  ( agree → min|a, b|; 0 absorbs )", ops_radmin))
    parts.append(binary_table(330, 568, "RadMax  ( agree → max|a, b|; 0 passes )", ops_radmax))
    radial = ["Both collapse to 0 when the signs disagree.",
              "Their LogSumExp forms, under kind='soft',",
              "approach these as τ → 0 and give both",
              "operands gradient. Neither is the default."]
    parts.append(rect(630, 568, 340, 144, GRY_L, stroke=GRY_MID, sw=1, rx=8))
    for k, row in enumerate(radial):
        parts.append(txt(648, 600 + k*28, row, anchor="start", fs=13, fill=BLUE_DK))
    return svg(W, H, "\n".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 – LUMINOSITY
# ══════════════════════════════════════════════════════════════════════════════
# TruthLayer.luminosity (bin/Layers.py), the catuṣkoṭi coverage measure:
#     T_k = max_i relu(+t_ik),  F_k = max_i relu(-t_ik)
#     L   = mean_k [ (T_k - F_k) - min(T_k, F_k) ]   clamped to [-1, 1]

def bar_vector(x, y, values, bar_w, bar_h, spacing, colors=None, neg_color=RED, label=""):
    """Draw a row of vertical bars representing a vector."""
    out = []
    for i, v in enumerate(values):
        bx = x + i * (bar_w + spacing)
        if v > 0:
            ph = int(round(v * bar_h))
            out.append(rect(bx, y - ph, bar_w, ph, colors[i] if colors else GRN, rx=2))
        if v < 0:
            out.append(rect(bx, y, bar_w, int(round(-v * bar_h)), neg_color, rx=2))
        out.append(line(bx - 1, y, bx + bar_w + 1, y, stroke="#bbb", sw=1))
    if label:
        out.append(txt(x + len(values) * (bar_w + spacing) / 2, y + bar_h + 16, label, fs=11, fill=GRY))
    return "\n".join(out)

def coverage(truths):
    D = len(truths[0])
    T = [max(max(t[k], 0.0) for t in truths) for k in range(D)]
    F = [max(max(-t[k], 0.0) for t in truths) for k in range(D)]
    corner = [(T[k] - F[k]) - min(T[k], F[k]) for k in range(D)]
    return T, F, corner, max(-1.0, min(1.0, sum(corner) / D))

def sub(s):
    return f"<tspan baseline-shift='sub' font-size='70%'>{s}</tspan>"

def make_luminosity():
    W, H = 960, 742
    parts = ["<defs>" + arrow_marker("arrt", TEAL, 7) + arrow_marker("arrr", RED, 7) + "</defs>"]
    parts.append(rect(0, 0, W, 44, BLUE_DK, rx=0))
    parts.append(txt(W//2, 29, "Luminosity: Truth Coherence Measure", fs=18, fill="white", bold=True))

    parts.append(rect(170, 54, 620, 60, TEAL_L, stroke=TEAL, sw=1.5, rx=8))
    parts.append(txt(W//2, 78, f"L  =  mean{sub('k')} [ (T{sub('k')} − F{sub('k')}) − "
                     f"min(T{sub('k')}, F{sub('k')}) ]  ∈  [−1, 1]", fs=16, fill=TEAL, bold=True))
    parts.append(txt(W//2, 102, f"T{sub('k')} = max{sub('i')} relu(+t{sub('ik')})  ·  "
                     f"F{sub('k')} = max{sub('i')} relu(−t{sub('ik')})  ·  coverage, not mass",
                     fs=13, fill=TEAL))

    D, BH = 7, 46
    cases = [
        (20, TEAL, "#f8fffe", "Consistent truths  →  high luminosity", "arrt",
         [[0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6],
          [0.7, 0.8, 0.7, 0.9, 0.6, 0.6, 0.7],
          [0.9, 0.7, 0.8, 0.6, 0.8, 0.7, 0.8]], ["t₁", "t₂", "t₃"], "high: the truths agree"),
        (490, RED, "#fff8f8", "Contradictory truths  →  negative luminosity", "arrr",
         [[0.8, 0.7, 0.9, -0.7, 0.6, 0.8, -0.5],
          [-0.6, -0.8, 0.5, 0.9, -0.7, -0.6, 0.7]], ["t₄", "t₅"], "negative: contradiction"),
    ]
    PW = 450
    for x0, col, bg, title, mid, truths, names, verdict in cases:
        parts.append(rect(x0, 126, PW, 548, bg, stroke=col, sw=1.5, rx=10))
        parts.append(title_bar(x0, 126, PW, 32, title, bg=col, fs=13))
        cx = x0 + PW // 2
        T, F, corner, L = coverage(truths)

        parts.append(txt(cx, 180, "Stored truths  (signed references)", fs=12, fill=GRY))
        bw, sp, gap = 12, 3, 18
        width = len(truths) * D * (bw + sp) + (len(truths) - 1) * gap
        for i, (t, name) in enumerate(zip(truths, names)):
            parts.append(bar_vector(cx - width // 2 + i * (D * (bw + sp) + gap), 240,
                                    t, bw, BH, sp, label=name))

        bw, sp = 20, 6
        vx = cx - D * (bw + sp) // 2
        parts.append(f"<line x1='{cx}' y1='312' x2='{cx}' y2='334' stroke='{col}' stroke-width='1.8' marker-end='url(#{mid})'/>")
        parts.append(txt(cx, 352, f"per dimension: T{sub('k')} up (green), F{sub('k')} down (red)",
                         fs=12, fill=col))
        parts.append(bar_vector(vx, 412, T, bw, BH, sp))
        parts.append(bar_vector(vx, 412, [-f for f in F], bw, BH, sp))

        parts.append(f"<line x1='{cx}' y1='466' x2='{cx}' y2='488' stroke='{col}' stroke-width='1.8' marker-end='url(#{mid})'/>")
        parts.append(txt(cx, 506, f"corner c{sub('k')} = (T{sub('k')} − F{sub('k')}) − "
                         f"min(T{sub('k')}, F{sub('k')})", fs=12, fill=col))
        parts.append(bar_vector(vx, 566, corner, bw, BH, sp, colors=[TEAL] * D))

        good = L > 0.5
        parts.append(rect(x0 + 25, 624, PW - 50, 34, GRN_L if good else RED_L,
                          stroke=GRN if good else RED, sw=1.5, rx=6))
        parts.append(txt(cx, 646, f"L = mean c ≈ {L:+.2f}  ({verdict})".replace("-", "−"), fs=13,
                         fill=GRN_DK if good else RED_DK, bold=True))

    parts.append(txt(W//2, 698, "Corners per dimension: T only → +T · F only → −F · "
                     "both → −min(T, F) · neither → 0.", fs=12, fill=GRY, italic=True))
    parts.append(txt(W//2, 720, "The meet form ‖relu(min truths)‖ survives only as the "
                     "orthogonalization criterion (TruthLayer._luminosity_without).",
                     fs=12, fill=GRY, italic=True))
    return svg(W, H, "\n".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 – VECTOR SPACES
# ══════════════════════════════════════════════════════════════════════════════
# Content follows doc/Spaces.md (Overview; "Perceptual and conceptual
# ownership"; the space sections) and the Spaces table in doc/Architecture.md.

def make_vector_spaces():
    W, H = 1440, 876
    SANS = "Helvetica, Arial, sans-serif"
    defs = ("<defs>" + arrow_marker("arrd", BLUE_DK) + arrow_marker("arrb", BLUE)
            + arrow_marker("arrg", GRN) + arrow_marker("arro", ORG)
            + arrow_marker("arrp", PURP) + "</defs>")
    parts = [defs]
    parts.append(txt(W//2, 40, "BasicModel Space Hierarchy", fs=30, fill="#222222", bold=True, ff=SANS))
    parts.append(txt(W//2, 66, "Forward (ownership): IS → {PS, WS, SS, LS} → CS → OS  ·  "
                     "Reverse: OS → SS → CS → {PS, WS} → IS  ·  "
                     "CS is the sole reducer and owner of every commit", fs=14, fill=GRY))

    def bullets(x, y, rows, fs=14.2, step=24, fill=BLUE_DK):
        return [txt(x, y + i*step, "• " + r, anchor="start", fs=fs, fill=fill) for i, r in enumerate(rows)]

    def ortho(points, color=BLUE_DK, mid="arrd", sw=1.8, dash=""):
        d = "M " + " L ".join(f"{px} {py}" for px, py in points)
        da = f" stroke-dasharray='{dash}'" if dash else ""
        return (f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{sw}'{da}"
                f" marker-end='url(#{mid})'/>")

    def space(x, y, w, h, code, label, rows, fill, stroke_col, step=24):
        out = [rect(x, y, w, h, fill, stroke=stroke_col, sw=1.8, rx=10),
               txt(x + 20, y + 36, code, anchor="start", fs=28, fill=stroke_col, bold=True),
               txt(x + 22 + 26 * len(code), y + 36, label, anchor="start", fs=20, fill=BLUE_DK, bold=True)]
        return out + bullets(x + 18, y + 70, rows, step=step)

    # InputSpace above the perceptual towers.
    ix, iy, iw, ih = 380, 88, 200, 48
    parts.append(rect(ix, iy, iw, ih, BLUE_L, stroke=BLUE, sw=2, rx=10))
    parts.append(txt(ix + iw//2, iy + 31, "InputSpace (IS)", fs=18, fill=BLUE_DK, bold=True))

    # Perceptual container: two peer towers, no PS -> WS edge.
    px0, py0, pw, ph = 40, 176, 880, 318
    parts.append(rect(px0, py0, pw, ph, "#eef6fb", stroke=BLUE, sw=2, rx=14))
    parts.append(txt(px0 + pw//2, py0 + 32, "Perceptual Spaces", fs=18, fill=BLUE_DK, bold=True))
    parts.append(txt(px0 + pw//2, py0 + ph - 10, "peers: the input fans to both towers; no PS → WS edge",
                     fs=12.5, fill=BLUE_DK, italic=True))
    ps = (70, 228, 400, 230)
    ws = (490, 228, 400, 230)
    parts += space(*ps, "PS", "Part Space", [
        "bottom-up synthesis of ordered parts",
        "native radix ids and ordered witnesses",
        "code = max of constituent codes",
        "one percept store; ids decide presence",
        "no learned fold inside the tower",
        "the ordered witness gives the exact reverse",
    ], BLUE_L, BLUE)
    parts += space(*ws, "WS", "Whole Space", [
        "top-down division by primitive properties",
        "one property codebook (ASCII classes)",
        "max over allowed primitives per position",
        "min pervasion over a run",
        "no perceptual pi layer",
        "reads conceptual context as its input",
    ], GRN_L, GRN)

    # Symbol and language: references, grammar and its scheduling.
    sx0, sy0, sw0, sh0 = 950, 176, 450, 318
    parts.append(rect(sx0, sy0, sw0, sh0, "#fdf5ec", stroke=ORG, sw=2, rx=14))
    parts.append(txt(sx0 + sw0//2, sy0 + 32, "Symbol and Language", fs=18, fill=BLUE_DK, bold=True))
    ls = (970, 216, 410, 96)
    ss = (970, 326, 410, 136)
    parts.append(rect(*ls, "#fff", stroke=ORG, sw=1.5, rx=10))
    parts.append(txt(ls[0] + 18, ls[1] + 30, "LS", anchor="start", fs=22, fill=ORG, bold=True))
    parts.append(txt(ls[0] + 58, ls[1] + 30, "LanguageSpace", anchor="start", fs=18, fill=BLUE_DK, bold=True))
    parts += bullets(ls[0] + 18, ls[1] + 58, [
        "scheduling: chooses grammar operations",
        "returns choice tensors; CS applies them",
    ], fs=13.5, step=22)
    parts.append(rect(*ss, ORG_L, stroke=ORG, sw=1.5, rx=10))
    parts.append(txt(ss[0] + 18, ss[1] + 30, "SS", anchor="start", fs=22, fill=ORG, bold=True))
    parts.append(txt(ss[0] + 58, ss[1] + 30, "SymbolSpace", anchor="start", fs=18, fill=BLUE_DK, bold=True))
    parts += bullets(ss[0] + 18, ss[1] + 58, [
        "zero-dimensional references to CS concepts",
        "LanguageLayer: compose, thought, generate",
        "TruthLayer; LTM0, the occurrence store",
        "no percept codebook, no atomic forward",
    ], fs=13.5, step=21)

    # ConceptualSpace: the sole reducer.
    cx0, cy0, cw0, ch0 = 190, 556, 1060, 200
    parts.append(rect(cx0, cy0, cw0, ch0, "#f5eef8", stroke=PURP, sw=2, rx=14))
    parts.append(txt(cx0 + 24, cy0 + 36, "Conceptual Space (CS)", anchor="start", fs=22, fill=PURP, bold=True))
    parts += bullets(cx0 + 24, cy0 + 70, [
        "sole reducer: commits concepts, STM and grammar transactions (CSsub, CSsym, CSLang)",
        "concept dictionary: code = signed activation × softplus(atom); two symbols per concept",
        "definitions: pi (conjunctive) only at order 0, sigma (disjunctive) at every order",
        "ConceptAllocator: ids and ordered records; ConceptualAttentionLayer relation store; LTM1",
        "STM (8 chunks) · MTM (tokens, familiar tier) · ActivationPredictor (expectation)",
    ], step=26)

    # OutputSpace below.
    ox, oy, ow, oh = 620, 790, 200, 48
    parts.append(rect(ox, oy, ow, oh, ORG_L, stroke=ORG, sw=2, rx=10))
    parts.append(txt(ox + ow//2, oy + 31, "OutputSpace (OS)", fs=18, fill=ORG, bold=True))

    # Forward arrows.
    ps_c, ws_c = ps[0] + ps[2]//2, ws[0] + ws[2]//2
    parts.append(ortho([(ix + 50, iy + ih), (ix + 50, 156), (ps_c, 156), (ps_c, ps[1])], BLUE, "arrb"))
    parts.append(ortho([(ix + 150, iy + ih), (ix + 150, 156), (ws_c, 156), (ws_c, ws[1])], GRN, "arrg"))
    parts.append(ortho([(ps_c, ps[1] + ps[3]), (ps_c, cy0)], BLUE, "arrb"))
    parts.append(ortho([(ws_c, ws[1] + ws[3]), (ws_c, cy0)], GRN, "arrg"))
    parts.append(ortho([(cx0 + cw0//2, cy0 + ch0), (cx0 + cw0//2, oy)], ORG, "arro"))
    # CS <-> SS: symbolicOrder reads CS -> SS -> CS; SS hands its result to LS.
    up_x, down_x = 1080, 1160
    parts.append(ortho([(up_x, cy0), (up_x, ss[1] + ss[3])], PURP, "arrp"))
    parts.append(ortho([(down_x, ss[1] + ss[3]), (down_x, cy0)], ORG, "arro"))
    parts.append(txt(down_x + 10, 524, "symbolicOrder × (CS → SS → CS)", anchor="start", fs=12.5, fill=BLUE_DK, italic=True))
    parts.append(ortho([(1300, ss[1]), (1300, ls[1] + ls[3])], ORG, "arro"))
    # LS returns choices; CS applies them (the CSLang transaction).
    parts.append(ortho([(ls[0] + ls[2], ls[1] + ls[3]//2), (1418, ls[1] + ls[3]//2),
                        (1418, cy0 + 120), (cx0 + cw0, cy0 + 120)], ORG, "arro"))
    parts.append(txt(1334, cy0 + 112, "choices (CSLang)", fs=12.5, fill=BLUE_DK, italic=True))
    # Conceptual pass-back: attention to PS (positive floor), context to WS.
    parts.append(ortho([(ps_c - 60, cy0), (ps_c - 60, ps[1] + ps[3])], PURP, "arrp", sw=2.2, dash="8,5"))
    parts.append(ortho([(ws_c + 90, cy0), (ws_c + 90, ws[1] + ws[3])], PURP, "arrp", sw=2.2, dash="8,5"))
    parts.append(txt(ps_c - 70, 524, "attention (positive floor)", anchor="end", fs=12.5, fill=PURP, italic=True))
    parts.append(txt(ws_c + 100, 524, "conceptual context", anchor="start", fs=12.5, fill=PURP, italic=True))

    parts.append(txt(40, H - 12, "solid: forward · dashed: conceptual pass-back · "
                     "sources: Spaces.md (Overview, ownership), Architecture.md (Spaces)",
                     anchor="start", fs=12, fill=GRY, italic=True))
    return svg(W, H, "\n".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4 – REFERENCE CONFIGURATION (read from data/BasicModel.xml)
# ══════════════════════════════════════════════════════════════════════════════
# Replaces the retired MM_5M "progressive bottleneck" figure. Every number is
# read from the model file at generation time, so regenerating keeps it current.

REFERENCE_MODEL = os.path.join(DIR, os.pardir, os.pardir, "data", "BasicModel.xml")
EVENT_BAND = 8    # .where 4 + .when 4 (Spaces.md, InputSpace)

def model_reader(path):
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    def get(section, key, default="—"):
        node = root.find(f"{section}/{key}")
        return node.text.strip() if node is not None and node.text else default
    return get

def make_reference_architecture():
    get = model_reader(REFERENCE_MODEL)
    num = lambda s: f"{int(s):,}" if str(s).isdigit() else s
    W, H = 1180, 744
    parts = ["<defs>" + arrow_marker("arrd", BLUE_DK) + arrow_marker("arrb", BLUE)
             + arrow_marker("arrg", GRN) + arrow_marker("arro", ORG) + arrow_marker("arrp", PURP) + "</defs>"]
    parts.append(rect(0, 0, W, 44, BLUE_DK, rx=0))
    parts.append(txt(W//2, 29, "BasicModel reference configuration  (data/BasicModel.xml)",
                     fs=18, fill="white", bold=True))
    parts.append(txt(W//2, 68, "read from the model file when the figure is generated  ·  "
                     "rows × width of each dictionary; live locations × event width",
                     fs=13, fill=GRY, italic=True))

    def box(x, y, w, h, title, rows, fill, stroke_col):
        out = [rect(x, y, w, h, fill, stroke=stroke_col, sw=2, rx=10),
               txt(x + w//2, y + 26, title, fs=16, fill=stroke_col, bold=True)]
        for k, r in enumerate(rows):
            out.append(txt(x + w//2, y + 52 + k*22, r, fs=13, fill=BLUE_DK))
        return out

    def ortho(points, color, mid, dash=""):
        d = "M " + " L ".join(f"{px} {py}" for px, py in points)
        da = f" stroke-dasharray='{dash}'" if dash else ""
        return f"<path d='{d}' fill='none' stroke='{color}' stroke-width='1.8'{da} marker-end='url(#{mid})'/>"

    I = lambda k: get("InputSpace", k)
    P = lambda k: get("PartSpace", k)
    Wh = lambda k: get("WholeSpace", k)
    C = lambda k: get("ConceptualSpace", k)
    O = lambda k: get("OutputSpace", k)
    A = lambda k: get("architecture", k)

    ib = (390, 86, 400, 98)
    parts += box(*ib, "InputSpace", [
        f"byte surface, ceiling {num(I('nOutput'))}",
        f"lift {num(I('nVectors'))} × {num(I('nDim'))}",
    ], BLUE_L, BLUE)

    pdim = int(P("nDim"))
    pb = (40, 236, 340, 164)
    parts += box(*pb, "PartSpace", [
        f"percept dictionary {num(P('nVectors'))} × {pdim}",
        f"event {pdim} = what {pdim - EVENT_BAND} + where/when {EVENT_BAND}",
        f"{P('nOutput')} live locations × {P('nOutputDim')}",
        f"synthesis: {P('synthesis')}; {A('bindingDepth')} native passes",
        f"chunk promotion at {P('chunkPromotionThreshold')} uses",
    ], BLUE_L, BLUE)

    wb = (420, 236, 340, 164)
    parts += box(*wb, "WholeSpace", [
        f"property basis {num(Wh('nVectors'))} × {Wh('nDim')}",
        f"input {Wh('nInput')} × {num(Wh('nInputDim'))}: conceptual context",
        f"{Wh('nOutput')} live locations × {Wh('nOutputDim')}",
        f"analysis: {Wh('analysis')}; butterfly: {Wh('butterfly')}",
        "",
    ], GRN_L, GRN)

    sb = (800, 236, 340, 164)
    lang = get("SymbolSpace", "language/grammar")
    parts += box(*sb, "SymbolSpace + LanguageSpace", [
        f"grammar: {lang}",
        f"LTM0 occurrence store: {num(get('SymbolSpace', 'ltmCapacity'))} rows",
        "references to CS concepts",
        "no percept codebook",
        "",
    ], ORG_L, ORG)

    cdim, mw = int(C("nDim")), int(C("meaningWidth"))
    layers = int(A("conceptLayers"))
    cb = (190, 460, 800, 164)
    parts += box(*cb, "ConceptualSpace", [
        f"concept dictionary {num(C('nVectors'))} rows ({num(C('activeVectors'))} active) × {num(cdim)}",
        f"code {num(cdim)} = form {cdim - mw} + meaning {mw}  ({mw // 2} pole pairs)",
        f"input {C('nInput')} × {num(C('nInputDim'))}; output {C('nOutput')} × {num(C('nOutputDim'))}",
        f"orders 0 to {layers - 1} (conceptLayers {layers}); STM {C('stmCapacity')} ideas; "
        f"binding: {A('conceptBinding')}",
        f"sigma/pi: {A('sigmaPi')}; word axis capacity {num(A('serialWordCapacity'))} (storage, not length)",
    ], "#f5eef8", PURP)

    ob = (440, 664, 300, 64)
    parts += box(*ob, "OutputSpace", [
        f"{O('nInput')} × {num(O('nInputDim'))}  →  {O('nOutput')} × {O('nOutputDim')}",
    ], ORG_L, ORG)

    # Arrows: input fans to the two towers; both feed CS; CS and SS exchange;
    # CS passes context back to WS; CS feeds the output head.
    parts.append(ortho([(ib[0] + 80, ib[1] + ib[3]), (ib[0] + 80, 212), (pb[0] + pb[2]//2, 212),
                        (pb[0] + pb[2]//2, pb[1])], BLUE, "arrb"))
    parts.append(ortho([(ib[0] + 200, ib[1] + ib[3]), (ib[0] + 200, wb[1])], GRN, "arrg"))
    parts.append(ortho([(pb[0] + pb[2]//2, pb[1] + pb[3]), (pb[0] + pb[2]//2, 430),
                        (cb[0] + 120, 430), (cb[0] + 120, cb[1])], BLUE, "arrb"))
    parts.append(ortho([(wb[0] + 120, wb[1] + wb[3]), (wb[0] + 120, cb[1])], GRN, "arrg"))
    parts.append(ortho([(wb[0] + 220, cb[1]), (wb[0] + 220, wb[1] + wb[3])], PURP, "arrp", dash="8,5"))
    parts.append(txt(wb[0] + 230, 436, "context", anchor="start", fs=12, fill=PURP, italic=True))
    parts.append(ortho([(cb[0] + cb[2] - 60, cb[1]), (cb[0] + cb[2] - 60, sb[1] + sb[3])], PURP, "arrp"))
    parts.append(ortho([(sb[0] + sb[2] - 60, sb[1] + sb[3]), (sb[0] + sb[2] - 60, 520),
                        (cb[0] + cb[2], 520)], ORG, "arro"))
    parts.append(ortho([(cb[0] + cb[2]//2 - 50, cb[1] + cb[3]), (cb[0] + cb[2]//2 - 50, ob[1])], ORG, "arro"))

    run = get("architecture", "training/batchSize")
    parts.append(txt(W - 20, H - 14, f"batch {run}  ·  max sentence {get('architecture', 'data/maxSentenceWords')} words"
                     f"  ·  grammar {lang}", anchor="end", fs=12, fill=GRY, italic=True))
    return svg(W, H, "\n".join(parts))


# ── Grammar operators sheet ────────────────────────────────────────────────────
# Data for doc/specs/2026-09-20-accessible-mind-subsystems.md. Monochrome and
# sans-serif so the sheet prints and takes ink; the blank rows are for the pen.

GO_REV = "2026-10-03 (operators candidate; awaiting review)"

# (id, title, three body lines, dashed = reducible or derived)
GO_SUBSYSTEMS = [
    ("1", "Perceptual knowing",
     ["activation of the two meronymic", "towers (PartSpace, WholeSpace);", "the word stream, read by bracket"], False),
    ("2", "Conceptual · order 0",
     ["FIELD: whole 0-order codebook at", "once; TWO symbols per concept,", "c+/c−; memberships over bytes"], False),
    ("3", "Conceptual · higher",
     ["FIELD over higher-order symbols;", "one point = a REGION of order 0,", "possibly discontinuous (rows as in 2)"], False),
    ("4", "Serial thinking",
     ["CODES enter projected; operators", "move / extend them into one IDEA,", "an off-codebook, generative vector"], True),
    ("5", "Priming",
     ["spreading activation over the", "concept store's edges; lives with", "the codebook; decays"], False),
    ("6", "Expectation",
     ["NEGATIVE IMAGE: predicted idea,", "sign-reversed, added at the closing;", "learns o − ê. > per bracket at 6.8"], True),
    ("7", "LTM",
     ["serial form ONLY: ideas, any order;", "> one row per S with (c+, c−) at 7;", "reached by CUE only (no last-N)"], False),
    ("8", "Budget",
     ["remaining work + closure pressure;", "READ to know cutoff is near,", "CHARGED by every thought op"], False),
    ("9", "Meronymic access",
     ["residual of part(x,y)=x·(y/|y|),", "left as an idea VECTOR (serial, 4),", "not a 1/0); order 0; no store"], True),
    ("10", "Taxonomic access",
     ["relations over higher-order symbols;", "emits a SYMBOLIC value (graded);", "> at 7 it is the store's sigma rows"], False),
]

GO_SUBSYSTEM_NOTE = (
    "field (parallel, graded) → CODE = nearest-row projection; words arrive as codes → "
    "operators compose an IDEA: one off-codebook vector that must regenerate its codes + "
    "operations (generativity)   ·   stores: perceptual, conceptual, priming, LTM   ·   "
    "resource: budget   ·   relation: taxonomy   ·   > = decided and specified, not yet landed")

GO_GRAMMARS = [
    ("<compose>  ·  understanding", [
        "reads: 1 perceptual knowing (word stream, row-owned), 2/3, 4, 5, 9",
        "never 6: PURITY — the composed idea o is the same whatever was predicted",
        "writes: 4 (push / fold), 2/3 (the composed idea); never 7, 10, 8",
        "7.5 ONE operation per round: one softmax over every operator × every location;",
        "binary shortens the sequence by one, unary does not; a fixed round budget;",
        "STOP is a candidate only once the sequence fits its LTM row; no DP of any kind",
        "two hard derivations: exploit (tempered argmax, item 8's tie rule on logits)",
        "and one distinct sampled explore, which trains and is not recorded",
        "gradient: live — straight-through, weighted by the chooser's own softmax"]),
    ("<thought>  ·  thinking  (per-model allow-list)", [
        "reads: 4 the completed idea + recency buffer, 2/3, 6 (what is conceived), 8, 9, 10,",
        "and 7 only as frames a what() brought into STM by cue",
        "writes: effects on 2/3 and 4; the controller alone records / writes 7",
        "chooser: the SAME selection layer as compose and generate (7.5) —",
        "(operator, operands, open roles, level) | conclude",
        "gradient: none through effects; chooser on policy credit only",
        "cost: every choice / execute / descend / return charges 8",
        "> 6.8 adds the bracket candidates and 6.5 identity binding to this same",
        "softmax, credited the same way; neither is a separate policy"]),
    ("<generate>  ·  speech production", [
        "reads: the concluded idea in 4 (GIVEN, on-manifold), 2/3,",
        "its OWN emitted prefix — never the input stream or a parse trace",
        "writes: 1 the output stream (words via the reverse chain)",
        "never: 7, 10, thought execution",
        "policy: the same selection layer once more; the serial STM reducer is",
        "that layer with a two-slot window, not a second mechanism",
        "gradient: live in generate; stops at the concluded idea",
        "cost: output walk budget (not 8)",
        ""]),
]

GO_GRAMMAR_NOTE = (
    "one vocabulary of operator identities; shared operators are the ONE gradient "
    "coupling between objectives (dissonance = per-operator gradient cosine); a compose face "
    "alone grants no thought permission   ·   the ENDING (not an operator) adds the negative "
    "image and, per sentence, commits the better of the two derivations")

# group -> rows of (operator, faces, operand kind, roles, {subsystem id: access}, note)
# kind: F = a pooled reading over a bracket, S = one identified concept (6.8, 6a).
# W* = FutureWork §7; > = decided and specified, not yet landed.
# Section D's inverses are read from Language.reverse_binary_step /
# reverse_unary_step and Layers._binary_op_inverse_impl, not from the prose.
GO_OPERATORS = [
    ("concept and symbol folds — declarations specify operand kinds and all effects", [
        ("not", "C T G", "F/S", "I1→O1", {"4": "RW"}, "exchange evidence poles; self-inverse; trust is unchanged"),
        ("non", "C – –", "F/S", "I1→O1", {"4": "RW"}, "exclude the expressed pole; no affirmation and no inverse"),
        ("intersection / union", "C G", "S", "I1,I2→O1", {"2": "R", "4": "RW"}, "intersection: signed coordinate minimum, zero is silent; union: log-odds sum"),
        ("conjunction / disjunction", "C G", "S", "I1,I2→O1", {"3": "R", "4": "RW"}, "product binding / mean bundling of symbol codes; inverse searches primed operands"),
        ("sum / chunk / product", "C G", "S", "I1,I2→O1", {"4": "RW"}, "sum is the mean; chunk is additive; product is coordinate multiplication"),
        ("compound", "C G", "S", "modifier,head→O1", {"2": "R", "3": "R", "4": "RW"}, "reverse the head's sigma, select cases by modifier, refold; no new parameters"),
        ("verb / adverb", "C G", "S", "I1,I2→O1", {"4": "RW"}, "verb head is I1; multiplicative spectral gain; the right operand witnesses the inverse"),
        ("lift / lower / generic", "C G", "S", "declared roles", {"4": "RW"}, "declared order and head properties; generic is independent of retired exist"),
        ("surface / preposition / bind", "C G", "S", "I1,I2→O1", {"4": "RW"}, "declared witness and reference policies; surface renames are deferred"),
        ("tense / morphology / aspect / null", "C G", "S", "I1→O1", {"4": "RW"}, "surface carriers; their planned dimensional rewrite remains future work"),
        ("interpret", "C G", "S", "word→object", {"2": "RW", "3": "RW", "4": "RW"}, "native definition replaces the word; admission uses the existing owner"),
    ]),
    ("checked thought — permission masks the same exhaustive effect declaration", [
        ("part / whole", "C T G", "S", "I1,I2→O1; open sides", {"4": "RW", "9": "R", "10": "R"}, "full conceptual content and paired evidence; native taxonomy at higher order"),
        ("equal", "C T G", "S", "I1,I2→O1", {"4": "RW"}, "stable question-formula identity; closing still writes two part rows"),
        ("what (query content)", "C T G", "S", "description→content", {"4": "RW", "7": "R"}, "one retrieval for what, where and when cues; lookup is retired; rename is deferred"),
        ("quantize / arma", "– T –", "S", "I1→O1", {"3": "RW", "6": "RW"}, "codebook projection / expectation content; structural identity carriers retired"),
    ]),
    ("retired / pending", [
        ("exist / true / binary symbolize", "– – –", "—", "—", {}, "no scalar truth output, no unused binary carrier"),
        ("legacy query / queryPart / queryEqual", "– – –", "—", "—", {}, "geometric scalar experiments retired; the content operator above remains"),
        ("divide / descend / gloss (6.8)", "C – –", "F", "bracket→brackets/symbol", {"8": "W"}, "same candidate softmax; typed bracket and shared perceptual budget"),
        ("STOP / closing", "C T G", "—", "eligible stop / commit", {"7": "W"}, "cost greedy and one-departure paths before training; strict improvement selects explore"),
    ]),
]

# group -> rows of (operator, form, what the reverse program does, what it needs / loses)
GO_INVERSES = [
    ("exact with the declared witness", [
        ("not", "self", "exchange the poles again", "no witness; evidence only"),
        ("sum / chunk", "residual", "2 × parent − witness / parent − witness", "free decoding searches the primed bank"),
        ("verb / adverb", "gain", "divide by the same positive multiplicative gain", "right operand witness; no iterative inverse"),
        ("surface / product / part / whole", "declared", "use the declared witness and inverse contract", "missing witnesses mask unavailable candidates"),
    ]),
    ("bounded search through the same forward face", [
        ("compound", "case search", "primed native modifier/head pairs, conditioned sigma cases", "both poles retained; no occurrence trace or fabricated split"),
        ("conjunction / disjunction / intersection / union / equal", "search", "cost the recomposed primed pairs", "shared fixed reconstruction basis limit"),
        ("lift / lower", "tied + search", "the forward map's own inverse and bounded operands", "no stored derivation is replayed"),
    ]),
    ("identity or unavailable", [
        ("non", "unavailable", "the expressed evidence was discarded", "never presented as its own inverse"),
        ("generic / tense / morphology / aspect / null", "identity", "the declared carrier preserves the conceptual value", "surface revisions remain future work"),
        ("interpret", "native index", "read the existing definition's word", "same symbol/concept owner"),
        ("quantize / arma", "no generate face", "thought-only content", "no identity pseudo-operator"),
    ]),
]


GO_INVERSE_NOTE = [
    "every reverse step reports availability, and a missing inverse is explicit: the "
    "traversal never substitutes an identity pseudo-inverse   ·   the reverse loop is a "
    "bounded while_loop over the recorded derivation, so only recorded operators execute, "
    "and the lossy folds share one candidate search",
    "WITNESS = the retained constituent reference, the dictionary row the word was pushed "
    "with. It belongs to its own occurrence, so it is present for reconstruction and absent "
    "in free generation — which is why the same operator has two inverses here",
]

GO_QUESTIONS = [
    "Open questions for the pen:  (1) is priming read by thought, or only as a retrieval cue "
    "and by compose?   (2) lookup without a generate face — intended?   (3) which structural "
    "operators may a model think with — `not`, to conclude absences?",
    "(4) why / how as walks over implies / operator rows — wait for two truths (item 7)?   "
    "(5) 7.5 deferred two levers: a nonzero training temperature, and a parsimony / work term "
    "against the round budget — take either up after the conference?",
    "(6) forgetting of the concept inventory itself — order-0 definitions, alternatives, "
    "feature groups — has no rule (item 5 or FutureWork?)   (7) item 8's propertyBasis: port "
    "it to XOR_grammar, or leave that gate as written?",
    "On the inverses (D):  (8) the lossy folds are the only operators with no closed form — "
    "should compose PREFER an invertible fold where both fit, the way item 8 prefers the "
    "structural face at equal fit?   (9) lift / lower reference-free recovery is item 6's "
    "blocker: is clean-up decoding through the forward kernel — the search the lossy folds "
    "already use — the intended answer?",
    "(10) adverb: keep the eight-step iterative inverse, or declare it non-invertible and "
    "let it fall to the same candidate search?   (11) surface's free-generation prior is a "
    "guess about the word rather than a recovery — is that the right division between "
    "reconstruction and generation, or should generation also search?",
]


# Helvetica advance widths (AFM, per 1000 em), so a string that would overflow
# its box is an error at generation time rather than a clipped sheet.
_HELV = dict(zip(
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~",
    [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
     556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015,
     667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778,
     722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556, 333,
     556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556,
     333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]))
_HELV.update({"·": 278, "×": 584, "—": 1000, "–": 556, "→": 1000, "−": 584, "§": 556,
              "ê": 556, "…": 1000, "’": 222, "> ": 556, "²": 333, "≤": 584, "δ": 528, "κ": 500, "Σ": 667, "⁰": 333})


def _helv_width(s, fs, bold=False):
    return sum(_HELV.get(c, 556) for c in s) / 1000 * fs * (1.06 if bold else 1.0)


def make_grammar_operators():
    INK, SOFT, FAINT, MUTE = "#111", "#333", "#444", "#666"
    SANS = "Helvetica, Arial, sans-serif"
    W, X0 = 2100, 40

    def t(x, y, s, fs=13, fill=INK, bold=False, italic=False, right=None):
        # right = the x the text must not pass (default: the sheet's right margin)
        over = x + _helv_width(s, fs, bold) - ((W - X0) if right is None else right - 2)
        if over > 0:
            raise ValueError(f"grammar_operators: {over:.0f}px too wide at {fs}px: {s!r}")
        return txt(x, y, escape(s, quote=False), anchor="start", fs=fs, fill=fill,
                   bold=bold, italic=italic, ff=SANS)

    def table(y, cols, groups, cells_of):
        """One bordered matrix: a header strip, italic group bands, and a
        row per entry.  ``cells_of`` maps a row tuple to one value per column."""
        total, top, rowh = sum(w for _, w in cols), y, 24
        parts.append(rect(X0, y, total, 26, "#eee", stroke=SOFT, sw=1, rx=0))
        cx = X0
        for name, w in cols:
            parts.append(t(cx + 6, y + 17, name, fs=11, bold=True, right=cx + w))
            cx += w
        y += 26
        for group, rows in groups:
            parts.append(rect(X0, y, total, rowh, "#f7f7f7", stroke=SOFT, sw=0.8, rx=0))
            parts.append(t(X0 + 6, y + 16, group, fs=11.5, fill=SOFT, bold=True, italic=True))
            y += rowh
            for row in rows:
                parts.append(rect(X0, y, total, rowh, "#fff", stroke="#999", sw=0.6, rx=0))
                cx = X0
                for (col, w), value in zip(cols, cells_of(row)):
                    if value:
                        narrow = w <= 48
                        parts.append(t(cx + 6, y + 16, value, fs=11.5 if narrow else 10.8,
                                       bold=narrow, right=cx + w))
                    cx += w
                y += rowh
        cx = X0
        for _, w in cols:
            parts.append(line(cx, top, cx, y, stroke="#bbb", sw=0.6))
            cx += w
        return y

    parts = [None]                       # the white ground, sized once H is known
    parts.append(t(X0, 44, "BasicModel — the three grammars, their operators, their "
                   "inverses, and what each face may touch", fs=22, bold=True))
    parts.append(t(X0, 68, f"Draft for annotation · {GO_REV} · operators have no return "
                   "values, only effects: a signature is roles × faces × read (R) / write (W) "
                   "over the accessible mind", fs=13, fill=FAINT, italic=True))

    # A. subsystems
    y = 100
    parts.append(t(X0, y, "A.  The accessible mind — ten subsystems (dashed = reducible or "
                   "derived; every access is bounded)", fs=16, bold=True))
    y += 14
    bw, bh, gap = 194, 98, 8
    for i, (sid, title, body, dashed) in enumerate(GO_SUBSYSTEMS):
        x = X0 + i * (bw + gap)
        parts.append(rect(x, y, bw, bh, "#fff", stroke=SOFT, sw=1.2,
                          dash="5,4" if dashed else ""))
        parts.append(t(x + 8, y + 18, sid, fs=12, fill=MUTE, bold=True))
        parts.append(t(x + 28, y + 18, title, fs=11.5, bold=True, right=x + bw))
        for j, row in enumerate(body):
            parts.append(t(x + 8, y + 38 + j * 15, row, fs=9.1, fill=SOFT, right=x + bw))
    y += bh + 16
    parts.append(t(X0, y, GO_SUBSYSTEM_NOTE, fs=11, fill=FAINT, italic=True))

    # B. the three grammars
    y += 26
    parts.append(t(X0, y, "B.  The three grammars (faces) and their contexts", fs=16, bold=True))
    y += 14
    gw, gh, ggap = 660, 204, 20
    for i, (title, body) in enumerate(GO_GRAMMARS):
        x = X0 + i * (gw + ggap)
        parts.append(rect(x, y, gw, gh, "#fff", stroke=SOFT, sw=1.4))
        parts.append(t(x + 10, y + 22, title, fs=14, bold=True, right=x + gw))
        for j, row in enumerate(body):
            parts.append(t(x + 10, y + 44 + j * 17, row, fs=11.3, fill="#222", right=x + gw))
    for i in range(len(GO_GRAMMARS) - 1):
        x, ym = X0 + (i + 1) * gw + i * ggap, y + gh / 2
        parts.append(line(x + 2, ym, x + ggap - 7, ym, stroke=SOFT, sw=1.5))
        parts.append(path(f"M{x + ggap - 8},{ym - 5} L{x + ggap - 1},{ym} "
                          f"L{x + ggap - 8},{ym + 5} z", fill=SOFT, stroke=SOFT, sw=1))
    y += gh + 14
    parts.append(t(X0, y, GO_GRAMMAR_NOTE, fs=11, fill=FAINT, italic=True))

    # C. the signature matrix
    y += 28
    parts.append(t(X0, y, "C.  Operator signatures — roles × faces × subsystem access",
                   fs=16, bold=True))
    y += 12
    ids = [s[0] for s in GO_SUBSYSTEMS]
    cols = [("operator", 270), ("faces", 58), ("kind", 46), ("roles", 250)] \
        + [(i, 42) for i in ids] + [("note", 970)]
    y = table(y, cols,
              GO_OPERATORS,
              lambda r: [r[0], r[1], r[2], r[3]] + [r[4].get(i, "") for i in ids] + [r[5]])

    # D. the inverses
    y += 26
    parts.append(t(X0, y, "D.  Operators and their inverses — the reverse program that "
                   "replays a recorded derivation backward", fs=16, bold=True))
    y += 12
    y = table(y, [("operator", 300), ("form", 96), ("what the reverse step does", 820),
                  ("what it needs · what is lost", 800)],
              GO_INVERSES, lambda r: list(r))
    y += 20
    for row in GO_INVERSE_NOTE:
        parts.append(t(X0, y, row, fs=11, fill=FAINT, italic=True))
        y += 16

    y += 20
    for q in GO_QUESTIONS:
        parts.append(t(X0, y, q, fs=11, fill=FAINT))
        y += 16
    H = y + 24
    parts[0] = rect(0, 0, W, H, "#fff", rx=0)
    return svg(W, H, "\n".join(parts))


# ── Write files ────────────────────────────────────────────────────────────────

DIAGRAMS = {
    "ternary_logic.svg":     make_ternary,
    "luminosity.svg":        make_luminosity,
    "vector_spaces.svg":     make_vector_spaces,
    "basicmodel_reference.svg": make_reference_architecture,
    "grammar_operators.svg": make_grammar_operators,
}


def write(fname, content):
    """Write one generated figure next to this script."""
    fp = os.path.join(DIR, fname)
    with open(fp, "w") as f:
        f.write(content)
    print(f"✓  {fp}")


def figure_modules():
    """The one-figure scripts beside this one (fig_*.py): FILENAME and make()."""
    import glob, importlib.util
    mods = {}
    for path in sorted(glob.glob(os.path.join(DIR, "fig_*.py"))):
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[mod.FILENAME] = mod.make
    return mods


def main(argv):
    makers = {**DIAGRAMS, **figure_modules()}
    names = argv[1:] or list(makers)
    unknown = [n for n in names if n not in makers]
    if unknown:
        sys.exit(f"unknown diagram(s): {', '.join(unknown)}; known: {', '.join(makers)}")
    for fname in names:
        content = makers[fname]()        # generate first: a width error must not truncate
        write(fname, content)


if __name__ == "__main__":
    main(sys.argv)
