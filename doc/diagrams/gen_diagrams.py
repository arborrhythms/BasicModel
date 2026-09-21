#!/usr/bin/env python3
"""Generate WikiOracle conceptual diagrams as SVG files.

    python gen_diagrams.py                        # write every diagram
    python gen_diagrams.py grammar_operators.svg  # write only the named ones
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


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 – TERNARY TRUTH TABLES
# ══════════════════════════════════════════════════════════════════════════════

def cell_color(v):
    m = {"+1": (GRN_L, GRN_DK), "0": (GRY_L, GRY), "−1": (RED_L, RED_DK)}
    return m.get(v, ("#fff", BLUE_DK))

def truth_cell(x, y, w, h, val, header=False):
    bg = BLUE_DK if header else cell_color(val)[0]
    fg = "white" if header else cell_color(val)[1]
    return (rect(x, y, w, h, bg, stroke="#ccc", sw=1, rx=0) +
            txt(x + w//2, y + h//2 + 5, val, fs=14, fill=fg, bold=not header))

def binary_table(x, y, op_label, rows, cell_w=60, cell_h=36):
    """Draw a 3×3 binary truth table with header row/col."""
    vals = ["+1", "0", "−1"]
    out = []
    # table title
    out.append(txt(x + (4*cell_w)//2, y - 8, op_label, fs=13, fill=BLUE_DK, bold=True))
    # corner cell
    out.append(truth_cell(x, y, cell_w, cell_h, "a \\ b", header=True))
    # column headers
    for j, v in enumerate(vals):
        bg, fg = cell_color(v)
        out.append(rect(x + (j+1)*cell_w, y, cell_w, cell_h, BLUE_DK, stroke="#ccc", sw=1, rx=0))
        out.append(txt(x + (j+1)*cell_w + cell_w//2, y + cell_h//2 + 5, v, fs=14, fill="white", bold=True))
    # row headers + data
    for i, rv in enumerate(vals):
        ry = y + (i+1)*cell_h
        bg, fg = cell_color(rv)
        out.append(rect(x, ry, cell_w, cell_h, BLUE_DK, stroke="#ccc", sw=1, rx=0))
        out.append(txt(x + cell_w//2, ry + cell_h//2 + 5, rv, fs=14, fill="white", bold=True))
        for j, cv in enumerate(vals):
            v = rows[i][j]
            out.append(truth_cell(x + (j+1)*cell_w, ry, cell_w, cell_h, v))
    return "\n".join(out)

def unary_table(x, y, op_label, col_label, rows, cell_w=70, cell_h=36):
    """Unary table: a | op(a) with 3 rows."""
    out = []
    out.append(txt(x + cell_w, y - 8, op_label, fs=13, fill=BLUE_DK, bold=True))
    # headers
    out.append(rect(x, y, cell_w, cell_h, BLUE_DK, stroke="#ccc", sw=1, rx=0))
    out.append(txt(x + cell_w//2, y + cell_h//2 + 5, "a", fs=14, fill="white", bold=True))
    out.append(rect(x + cell_w, y, cell_w, cell_h, BLUE_DK, stroke="#ccc", sw=1, rx=0))
    out.append(txt(x + cell_w + cell_w//2, y + cell_h//2 + 5, col_label, fs=13, fill="white", bold=True))
    # rows
    for i, (av, rv) in enumerate(rows):
        ry = y + (i+1)*cell_h
        out.append(truth_cell(x, ry, cell_w, cell_h, av))
        # Result cell with special handling for non-T/U/F values
        rbg, rfg = cell_color(rv)
        out.append(rect(x + cell_w, ry, cell_w, cell_h, rbg, stroke="#ccc", sw=1, rx=0))
        out.append(txt(x + cell_w + cell_w//2, ry + cell_h//2 + 5, rv, fs=13, fill=rfg, bold=True))
    return "\n".join(out)

def make_ternary():
    W, H = 970, 530
    defs = ("<defs>" +
            arrow_marker("arrd", BLUE_DK) +
            "</defs>")

    parts = [defs]

    # Title
    parts.append(rect(0, 0, W, 44, BLUE_DK, rx=0))
    parts.append(txt(W//2, 29, "Ternary Logic Operators", fs=18, fill="white", bold=True))

    # Legend
    parts.append(txt(W//2, 66, "+1 = true     0 = unknown     −1 = false", fs=13, fill=GRY))

    # ── Row 1: Unary operators ─────────────────────────────────────

    # NOT / neg(a) = −a
    parts.append(unary_table(40, 100, "NOT  ( neg(a) = −a )",  "¬a",
        [("+1","−1"),("0","0"),("−1","+1")]))

    # NON / non(a) = 0  (bitonic: complete withdrawal of assertion)
    parts.append(unary_table(230, 100, "NON  ( non(a) = 0 )",  "non(a)",
        [("+1","0"),("0","0"),("−1","0")]))

    # Separator
    parts.append(line(0, 255, W, 255, stroke="#ddd", sw=1.5))

    # ── Row 2: Binary operators ────────────────────────────────────

    # Intersection: sign-aware min magnitude (Basis.conjunction, bitonic)
    and_rows = [
        ["+1","0","0"],
        ["0","0","0"],
        ["0","0","−1"],
    ]
    parts.append(binary_table(30, 287, "Intersection  ( agree → min|a,b| )", and_rows))

    # Union: sign-aware max magnitude + absorb zeros (Basis.disjunction, bitonic)
    or_rows = [
        ["+1","+1","0"],
        ["+1","0","−1"],
        ["0","−1","−1"],
    ]
    parts.append(binary_table(330, 287, "Union  ( agree → max|a,b|; 0 absorbs )", or_rows))

    # PART: mereological containment score [0, 1]  — Basis.part(a, b)
    part_rows = [
        ["+1","0","0"],
        ["+1","+1","+1"],
        ["0","0","+1"],
    ]
    parts.append(binary_table(630, 287, "PART  ( mereological, range [0, 1] )", part_rows))

    # Footer note on PART
    parts.append(txt(630 + 120, 487,
        "PART(a,b): 1 when a ⊆ b; 0 is part of everything; opposite signs → 0",
        fs=11, fill=GRY, anchor="middle", italic=True))

    return svg(W, H, "\n".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 – LUMINOSITY
# ══════════════════════════════════════════════════════════════════════════════

def bar_vector(x, y, values, bar_w, bar_h, spacing, colors=None, label=""):
    """Draw a row of vertical bars representing a vector."""
    out = []
    for i, v in enumerate(values):
        bx = x + i * (bar_w + spacing)
        h_pos = max(0, v)
        h_neg = min(0, v)
        # positive part (up from baseline)
        if h_pos > 0:
            ph = int(h_pos * bar_h)
            col = colors[i] if colors else GRN
            out.append(rect(bx, y - ph, bar_w, ph, col, rx=2))
        # negative part (down from baseline)
        if h_neg < 0:
            nh = int(abs(h_neg) * bar_h)
            out.append(rect(bx, y, bar_w, nh, RED, rx=2))
        # baseline tick
        out.append(line(bx - 1, y, bx + bar_w + 1, y, stroke="#bbb", sw=1))
    if label:
        cx = x + len(values) * (bar_w + spacing) / 2
        out.append(txt(cx, y + 18, label, fs=11, fill=GRY))
    return "\n".join(out)

def make_luminosity():
    W, H = 900, 600
    defs = "<defs>" + arrow_marker("arrd", BLUE_DK) + arrow_marker("arrt", TEAL, 7) + "</defs>"
    parts = [defs]

    # Title bar
    parts.append(rect(0, 0, W, 44, BLUE_DK, rx=0))
    parts.append(txt(W//2, 29, "Luminosity: Truth Coherence Measure", fs=18, fill="white", bold=True))

    # Formula box
    parts.append(rect(220, 52, 460, 38, TEAL_L, stroke=TEAL, sw=1.5, rx=8))
    parts.append(txt(W//2, 77, "L  =  ‖ [min(truths)]⁺ ‖", fs=16, fill=TEAL, bold=True))

    # ── LEFT CASE: Consistent truths ──────────────────────────────
    CASE_W = 380
    parts.append(rect(20, 105, CASE_W, 460, "#f8fffe", stroke=TEAL, sw=1.5, rx=10))
    parts.append(title_bar(20, 105, CASE_W, 32, "Consistent Truths  →  High Luminosity", bg=TEAL, fs=13))

    # Three consistent truth vectors (all positive)
    D = 7
    t1 = [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6]
    t2 = [0.7, 0.8, 0.7, 0.9, 0.6, 0.6, 0.7]
    t3 = [0.9, 0.7, 0.8, 0.6, 0.8, 0.7, 0.8]

    BW, BH, SP = 12, 60, 3
    base_y = 230

    parts.append(txt(210, 155, "Stored truths  (activation × DoT):", fs=12, fill=GRY))

    for idx, (tvec, lbl) in enumerate([(t1, "t₁"), (t2, "t₂"), (t3, "t₃")]):
        bx = 35 + idx * (D*(BW+SP) + 15)
        parts.append(bar_vector(bx, base_y, tvec, BW, BH, SP, label=lbl))

    # Conjunction = min
    conj = [min(t1[i], t2[i], t3[i]) for i in range(D)]
    parts.append(arrow(210, base_y + 28, 210, base_y + 60, color=TEAL, mid="arrt"))
    parts.append(txt(210, base_y + 76, "min( t₁, t₂, t₃ )", fs=12, fill=TEAL))
    parts.append(bar_vector(35 + D*(BW+SP)//2 - D*(BW+SP)//2, base_y + 90,
                            conj, BW, BH, SP, label="conjunction"))

    # positive part = keep positive, zero out negatives
    relu_conj = [max(0, v) for v in conj]
    parts.append(arrow(210, base_y + 168, 210, base_y + 198, color=TEAL, mid="arrt"))
    parts.append(txt(210, base_y + 214, "[conjunction]⁺  (positive part)", fs=12, fill=TEAL))
    parts.append(bar_vector(35 + D*(BW+SP)//2 - D*(BW+SP)//2, base_y + 228,
                            relu_conj, BW, BH, SP, colors=[GRN]*D, label="positive part"))

    # Luminosity value
    L_val = math.sqrt(sum(v*v for v in relu_conj))
    parts.append(rect(35, base_y + 320, CASE_W - 50, 34, GRN_L, stroke=GRN, sw=1.5, rx=6))
    parts.append(txt(35 + (CASE_W-50)//2, base_y + 342,
                     f"Luminosity  L ≈ {L_val:.2f}   (high — truths agree)",
                     fs=13, fill=GRN_DK, bold=True))

    # ── RIGHT CASE: Contradictory truths ──────────────────────────
    parts.append(rect(W - CASE_W - 20, 105, CASE_W, 460, "#fff8f8", stroke=RED, sw=1.5, rx=10))
    parts.append(title_bar(W - CASE_W - 20, 105, CASE_W, 32,
                           "Contradictory Truths  →  Low Luminosity", bg=RED, fs=13))

    # Two contradictory truth vectors
    t4 = [ 0.8,  0.7,  0.9, -0.7,  0.6,  0.8, -0.5]  # mixed
    t5 = [-0.6, -0.8,  0.5,  0.9, -0.7, -0.6,  0.7]  # opposite

    XR = W - CASE_W - 20 + 30
    for idx, (tvec, lbl) in enumerate([(t4, "t₄  (pos/neg mixed)"), (t5, "t₅  (neg dominant)")]):
        bx = XR + idx * (D*(BW+SP) + 15)
        parts.append(bar_vector(bx, base_y, tvec, BW, BH, SP, label=lbl))

    # conjunction
    conj2 = [min(t4[i], t5[i]) for i in range(D)]
    parts.append(arrow(XR + D*(BW+SP)*1 - 10, base_y + 28,
                       XR + D*(BW+SP)*1 - 10, base_y + 60, color=RED, mid="arrd"))
    parts.append(txt(XR + D*(BW+SP) + 10, base_y + 76, "min( t₄, t₅ )", fs=12, fill=RED))
    conj2_x = XR + 20
    parts.append(bar_vector(conj2_x, base_y + 90, conj2, BW, BH, SP, label="conjunction (many negative)"))

    relu_conj2 = [max(0, v) for v in conj2]
    parts.append(arrow(XR + D*(BW+SP)//2, base_y + 168,
                       XR + D*(BW+SP)//2, base_y + 198, color=RED, mid="arrd"))
    parts.append(txt(XR + D*(BW+SP)//2, base_y + 214, "[conjunction]⁺  (positive part)", fs=12, fill=RED))
    parts.append(bar_vector(conj2_x, base_y + 228, relu_conj2,
                            BW, BH, SP, colors=[GRN if v > 0 else GRY_L for v in relu_conj2],
                            label="positive part (most dims zeroed)"))

    L_val2 = math.sqrt(sum(v*v for v in relu_conj2))
    parts.append(rect(XR - 15, base_y + 320, CASE_W - 50, 34, RED_L, stroke=RED, sw=1.5, rx=6))
    parts.append(txt(XR - 15 + (CASE_W-50)//2, base_y + 342,
                     f"Luminosity  L ≈ {L_val2:.2f}   (low — contradiction)",
                     fs=13, fill=RED_DK, bold=True))

    return svg(W, H, "\n".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 – VECTOR SPACES
# ══════════════════════════════════════════════════════════════════════════════

def make_vector_spaces():
    W, H = 1440, 780
    defs = (
        "<defs>"
        + arrow_marker("arrd", BLUE_DK)
        + arrow_marker("arrb", BLUE)
        + arrow_marker("arrg", GRN)
        + arrow_marker("arro", ORG)
        + arrow_marker("arrp", PURP)
        + "</defs>"
    )
    parts = [defs]

    parts.append(txt(W//2, 38, "BasicModel Space Hierarchy", fs=30, fill="#222222", bold=True,
                     ff="Helvetica, Arial, sans-serif"))

    def bullet_list(x, y, rows, fill=BLUE_DK, fs=9.4, step=17):
        out = []
        for i, row in enumerate(rows):
            out.append(txt(x, y + i*step, "• " + row, anchor="start", fs=fs, fill=fill))
        return out

    def feedback_path(d, marker=True):
        marker_attr = " marker-end='url(#arrp)'" if marker else ""
        return (
            f"<path d='{d}' fill='none' stroke='{PURP}' stroke-width='2.3'"
            f" stroke-dasharray='8,5'{marker_attr}/>"
        )

    def ortho_arrow(points, color=BLUE_DK, sw=1.8, mid="arrd", dash=""):
        d = "M " + " L ".join(f"{x} {y}" for x, y in points)
        da = f" stroke-dasharray='{dash}'" if dash else ""
        return (
            f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{sw}'"
            f"{da} marker-end='url(#{mid})'/>"
        )

    # Small boundary spaces sit above and below the main hierarchy.
    input_box = (620, 64, 200, 48)
    output_box = (620, 718, 200, 48)
    parts.append(rect(*input_box, BLUE_L, stroke=BLUE, sw=2, rx=10))
    parts.append(txt(input_box[0] + input_box[2]//2, input_box[1] + 31, "InputSpace (IS)", fs=18, fill=BLUE_DK, bold=True))
    parts.append(rect(*output_box, ORG_L, stroke=ORG, sw=2, rx=10))
    parts.append(txt(output_box[0] + output_box[2]//2, output_box[1] + 31, "OutputSpace (OS)", fs=18, fill=ORG, bold=True))

    # Perceptual container with three horizontal internal spaces.
    pc_x, pc_y, pc_w, pc_h = 40, 150, 1360, 285
    parts.append(rect(pc_x, pc_y, pc_w, pc_h, "#eef6fb", stroke=BLUE, sw=2, rx=14))
    child_y, child_h = 200, 210
    child_w, gap = 330, 145
    children = [
        ("PS", "Part Space", [
            "synthesizes atoms into part-percepts",
            "atom-view stem from InputSpace",
            "Sigma synthesis fold",
            "MPHF + index table",
            "Lexicon: surface rows",
            "part codebook: part-percepts",
        ], pc_x + 40, BLUE_L, BLUE),
        ("WS", "Whole Space", [
            "analyzes properties, regions and wholes",
            "unity / property view",
            "Pi analysis fold",
            "property basis: regions",
            "whole-percept codebook",
            "paired orth / semantic rows",
        ], pc_x + 40 + child_w + gap, GRN_L, GRN),
        ("SS", "Symbolic Space", [
            "zero-dimensional references",
            "symbol codebook",
            "grammar / signal router",
            "operators live in codebook",
            "TruthLayer stores DoT propositions",
            "TruthLayer: record / query / field",
        ], pc_x + 40 + 2*(child_w + gap), ORG_L, ORG),
    ]
    for code, label, rows, cx, fill, stroke_col in children:
        parts.append(rect(cx, child_y, child_w, child_h, fill, stroke=stroke_col, sw=1.8, rx=10))
        parts.append(txt(cx + 22, child_y + 38, code, anchor="start", fs=30, fill=stroke_col, bold=True))
        parts.append(txt(cx + 82, child_y + 38, label, anchor="start", fs=20, fill=BLUE_DK, bold=True))
        parts.extend(bullet_list(cx + 18, child_y + 80, rows, fs=14.2, step=24))

    # Conceptual space receives perceptual and symbolic streams.
    cs_x, cs_y, cs_w, cs_h = 190, 485, 1060, 190
    parts.append(rect(cs_x, cs_y, cs_w, cs_h, "#f5eef8", stroke=PURP, sw=2, rx=14))
    parts.append(txt(cs_x + 24, cs_y + 38, "Conceptual Space (CS)", anchor="start", fs=22, fill=PURP, bold=True))
    cs_rows = [
        "STM plus concept relations",
        "ties part-percepts to whole-percepts",
        "Concept codebook: relation rows",
        "ConceptualAttentionLayer: symbolic wave over concept inventory",
        "ConceptAllocator: ids, ordered records, relation pool",
        "Truth-gated acceptance and reasoning hooks",
    ]
    parts.extend(bullet_list(cs_x + 24, cs_y + 72, cs_rows, fs=14.2, step=20))

    # Flow arrows.
    ps_mid = (pc_x + 40 + child_w//2, child_y + child_h)
    ws_mid = (pc_x + 40 + child_w + gap + child_w//2, child_y + child_h)
    ss_mid = (pc_x + 40 + 2*(child_w + gap) + child_w//2, child_y + child_h)
    cs_top = cs_y
    input_bottom = input_box[1] + input_box[3]
    input_join_y = pc_y + 20
    ps_input_entry = (pc_x + 40 + 210, child_y)
    ws_input_entry = (pc_x + 40 + child_w + gap + 125, child_y)
    parts.append(ortho_arrow(
        [(input_box[0] + 50, input_bottom), (input_box[0] + 50, input_join_y),
         (ps_input_entry[0], input_join_y), ps_input_entry],
        color=BLUE,
        mid="arrb",
    ))
    parts.append(ortho_arrow(
        [(input_box[0] + 150, input_bottom), (input_box[0] + 150, input_join_y),
         (ws_input_entry[0], input_join_y), ws_input_entry],
        color=GRN,
        mid="arrg",
    ))
    parts.append(ortho_arrow(
        [(ps_mid[0], ps_mid[1]), (ps_mid[0], cs_top - 20), (cs_x + 220, cs_top - 20), (cs_x + 220, cs_top)],
        color=BLUE,
        mid="arrb",
    ))
    parts.append(ortho_arrow(
        [(ws_mid[0], ws_mid[1]), (ws_mid[0], cs_top)],
        color=GRN,
        mid="arrg",
    ))
    parts.append(ortho_arrow(
        [(ss_mid[0], ss_mid[1]), (ss_mid[0], cs_top - 20), (cs_x + cs_w - 220, cs_top - 20), (cs_x + cs_w - 220, cs_top)],
        color=ORG,
        mid="arro",
    ))
    parts.append(ortho_arrow(
        [(cs_x + cs_w//2, cs_y + cs_h), (cs_x + cs_w//2, output_box[1])],
        color=ORG,
        mid="arro",
    ))

    # Recurrent paths from the CS output back into perceptual and symbolic entry points.
    cs_out_y = cs_y + cs_h
    ps_entry = (pc_x + 40 + 70, child_y)
    ws_entry = (pc_x + 40 + child_w + gap + child_w - 45, child_y)
    ss_entry = (pc_x + 40 + 2*(child_w + gap) + child_w//2, child_y)
    parts.append(feedback_path(
        f"M {cs_x + 400} {cs_out_y} L {cs_x + 400} {output_box[1] - 18} "
        f"L 20 {output_box[1] - 18} L 20 {pc_y - 18} "
        f"L {ps_entry[0]} {pc_y - 18} L {ps_entry[0]} {ps_entry[1]}"
    ))
    parts.append(feedback_path(
        f"M {cs_x + 640} {cs_out_y} L {cs_x + 640} {output_box[1] - 28} "
        f"L {W - 52} {output_box[1] - 28} L {W - 52} {pc_y - 18} "
        f"L {ws_entry[0]} {pc_y - 18} L {ws_entry[0]} {ws_entry[1]}"
    ))
    parts.append(feedback_path(
        f"M {cs_x + 780} {cs_out_y} L {cs_x + 780} {output_box[1] - 10} "
        f"L {W - 20} {output_box[1] - 10} L {W - 20} {pc_y + 34} "
        f"L {ss_entry[0]} {pc_y + 34} L {ss_entry[0]} {ss_entry[1]}"
    ))
    parts.append(rect(pc_x + pc_w//2 - 180, pc_y + 14, 360, 34, "#eef6fb", rx=0))
    parts.append(txt(pc_x + pc_w//2, pc_y + 38, "Perceptual Spaces", fs=22, fill=BLUE_DK, bold=True))

    return svg(W, H, "\n".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4 – MM_5M ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def lerp_color(c1, c2, t):
    """Linearly interpolate two hex colors."""
    r1, g1, b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
    r2, g2, b2 = int(c2[1:3],16), int(c2[3:5],16), int(c2[5:7],16)
    r = int(r1 + (r2-r1)*t); g = int(g1 + (g2-g1)*t); b = int(b1 + (b2-b1)*t)
    return f"#{r:02x}{g:02x}{b:02x}"

def make_mm5m():
    W, H = 980, 760
    defs = "<defs>" + arrow_marker("arrd", BLUE_DK) + arrow_marker("arrg", GRN) + "</defs>"
    parts = [defs]

    # Title
    parts.append(rect(0, 0, W, 44, BLUE_DK, rx=0))
    parts.append(txt(W//2, 29, "MM_5M: Hierarchical Progressive Bottleneck  (N×D = 4096 = const)", fs=17, fill="white", bold=True))

    # Constants
    LEVELS = 8
    # (N, D) per level
    level_shapes = [(1024 >> t, 4 << t) for t in range(LEVELS)]  # perceptual + 8 concept levels
    # But first entry is Perceptual [1024×4], levels are [512×8], [256×16], ...
    concept_shapes = [(512 >> t, 8 << t) for t in range(LEVELS)]

    BOX_H = 36
    GAP   = 12
    CW    = 260  # conceptual box width
    SW_BOX = 100  # symbolic box width
    ARROW_W = 24
    LEFT_X  = 60
    SYM_X   = LEFT_X + CW + ARROW_W
    LABEL_X = SYM_X + SW_BOX + 16
    FIRST_Y = 150

    # Color gradient: BLUE → PURP → ORG across levels
    def level_color(t):
        if t < LEVELS//2:
            return lerp_color(BLUE, PURP, t / (LEVELS//2))
        else:
            return lerp_color(PURP, ORG, (t - LEVELS//2) / (LEVELS//2))

    # Input box
    parts.append(rect(LEFT_X, 55, CW, 32, GRY_L, stroke=GRY_MID, sw=1.5, rx=6))
    parts.append(txt(LEFT_X + CW//2, 77, "Input: byte tokens  (vocab 4096)", fs=13, fill=GRY))

    # Arrow to Perceptual
    parts.append(arrow(LEFT_X + CW//2, 87, LEFT_X + CW//2, 103, color=BLUE, mid="arrd"))

    # Perceptual box
    parts.append(rect(LEFT_X, 103, CW, BOX_H, BLUE_L, stroke=BLUE, sw=2, rx=6))
    parts.append(txt(LEFT_X + CW//2, 127, "Perceptual Space  [1024 × 4]   N×D = 4096", fs=12, fill=BLUE_DK, bold=True))

    # Arrow to first level
    parts.append(arrow(LEFT_X + CW//2, 139, LEFT_X + CW//2, FIRST_Y - 2, color=BLUE, mid="arrd"))

    # Level boxes
    for t in range(LEVELS):
        N, D = concept_shapes[t]
        sy = FIRST_Y + t * (BOX_H + GAP)
        col = level_color(t)
        col_l = lerp_color(col, "#ffffff", 0.6)

        # Conceptual box (width proportional to D on log scale, max CW)
        log_scale = (math.log2(D) - 3) / (math.log2(1024) - 3)  # 0 at D=8, 1 at D=1024
        box_w = int(180 + log_scale * (CW - 180))

        parts.append(rect(LEFT_X + (CW - box_w)//2, sy, box_w, BOX_H, col_l, stroke=col, sw=2, rx=4))
        parts.append(txt(LEFT_X + CW//2, sy + BOX_H//2 + 5,
                         f"L{t}  Conceptual  [{N} × {D}]", fs=12, fill=BLUE_DK, bold=True))

        # Arrow to symbolic
        parts.append(arrow(LEFT_X + CW + 4, sy + BOX_H//2,
                           SYM_X - 4, sy + BOX_H//2, color=col, mid="arrd"))

        # Symbolic box (fixed width)
        parts.append(rect(SYM_X, sy, SW_BOX, BOX_H, col_l, stroke=col, sw=1.5, rx=4))
        parts.append(txt(SYM_X + SW_BOX//2, sy + BOX_H//2 + 5,
                         f"[{N}×4]", fs=12, fill=BLUE_DK, bold=True))



    # Down arrow from last level
    last_y = FIRST_Y + (LEVELS - 1) * (BOX_H + GAP)
    parts.append(arrow(LEFT_X + CW//2, last_y + BOX_H + 2,
                       LEFT_X + CW//2, last_y + BOX_H + 30, color=ORG, mid="arrd"))

    # Output box
    out_y = last_y + BOX_H + 30
    parts.append(rect(LEFT_X, out_y, CW, 32, ORG_L, stroke=ORG, sw=2, rx=6))
    parts.append(txt(LEFT_X + CW//2, out_y + 22, "Output  [1 × 4]  logits", fs=12, fill=ORG, bold=True))

    # ── Right panel: N×D = const visualization ─────────────────────
    PLOT_X = SYM_X + SW_BOX + 50
    PLOT_W = W - PLOT_X - 30
    PLOT_Y = 80
    PLOT_H = 500

    parts.append(rect(PLOT_X, PLOT_Y, PLOT_W, PLOT_H, "#f8f9fa", stroke="#ddd", sw=1, rx=10))
    parts.append(txt(PLOT_X + PLOT_W//2, PLOT_Y + 22, "Information Density", fs=14, fill=BLUE_DK, bold=True))
    parts.append(txt(PLOT_X + PLOT_W//2, PLOT_Y + 40, "N × D = 4096 (constant)", fs=12, fill=GRY, italic=True))

    # Draw a log-log plot: x=log2(D), y=log2(N)
    AXIS_X = PLOT_X + 50
    AXIS_Y = PLOT_Y + PLOT_H - 60
    AXIS_W = PLOT_W - 80
    AXIS_H = PLOT_H - 120

    # Axes
    parts.append(line(AXIS_X, AXIS_Y, AXIS_X + AXIS_W, AXIS_Y, stroke=GRY_MID, sw=1.5))  # x
    parts.append(line(AXIS_X, AXIS_Y, AXIS_X, AXIS_Y - AXIS_H, stroke=GRY_MID, sw=1.5))  # y

    # Axis labels
    parts.append(txt(AXIS_X + AXIS_W//2, AXIS_Y + 28, "D  (dimension)", fs=12, fill=GRY))
    parts.append(txt(AXIS_X - 38, AXIS_Y - AXIS_H//2, "N", fs=12, fill=GRY))
    parts.append(txt(AXIS_X - 28, AXIS_Y - AXIS_H//2 + 14, "(tokens)", fs=11, fill=GRY))

    # Map (D, N) to plot coords
    # x range: log2(4)=2 to log2(1024)=10 → D: 4..1024
    # y range: log2(4)=2 to log2(1024)=10 → N: 4..1024
    def px(D_val):
        return AXIS_X + (math.log2(D_val) - 2) / 8 * AXIS_W
    def py(N_val):
        return AXIS_Y - (math.log2(N_val) - 2) / 8 * AXIS_H

    # Grid lines
    for v in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        xg = px(v); yg = py(v)
        parts.append(line(xg, AXIS_Y, xg, AXIS_Y + 5, stroke=GRY_MID, sw=1))
        parts.append(txt(xg, AXIS_Y + 18, str(v), fs=10, fill=GRY))
        parts.append(line(AXIS_X - 5, yg, AXIS_X, yg, stroke=GRY_MID, sw=1))
        parts.append(txt(AXIS_X - 8, yg + 4, str(v), fs=10, fill=GRY, anchor="end"))

    # Diagonal line N*D = 4096 → log2(N) = 12 - log2(D)
    d1, d2 = 4, 1024
    parts.append(line(px(d1), py(d2), px(d2), py(d1), stroke=TEAL, sw=1.5, dash="6,3"))

    # Points: Perceptual + 8 levels
    all_points = [(4, 1024, "P")] + [(D, N, f"L{t}") for t, (N, D) in enumerate(concept_shapes)]
    for i, (D_val, N_val, lbl) in enumerate(all_points):
        col = BLUE if i == 0 else level_color(i - 1)
        ppx = px(D_val); ppy = py(N_val)
        parts.append(circle(ppx, ppy, 7, col, BLUE_DK, sw=1.5))
        offset_x = 12 if D_val <= 256 else -14
        anchor = "start" if D_val <= 256 else "end"
        parts.append(txt(ppx + offset_x, ppy + 4, lbl, fs=11, fill=col, anchor=anchor, bold=True))

    # Annotation
    parts.append(txt(PLOT_X + PLOT_W//2, PLOT_Y + PLOT_H - 18,
        "Each step: N ÷ 2, D × 2  (same area = same information)",
        fs=11, fill=GRY, italic=True))


    return svg(W, H, "\n".join(parts))


# ── Grammar operators sheet ────────────────────────────────────────────────────
# Data for doc/specs/2026-09-20-accessible-mind-subsystems.md. Monochrome and
# sans-serif so the sheet prints and takes ink; the blank rows are for the pen.

GO_REV = "2026-09-20 (rev 10)"

# (id, title, three body lines, dashed = reducible or derived)
GO_SUBSYSTEMS = [
    ("1", "Perceptual knowing",
     ["activation of the two meronymic", "towers (PartSpace, WholeSpace);", "the word stream"], False),
    ("2", "Conceptual · order 0",
     ["FIELD: one activation per 0-order", "symbol, whole codebook at once;", "parallel"], False),
    ("3", "Conceptual · higher",
     ["FIELD over higher-order symbols;", "one point = a REGION of order 0,", "possibly discontinuous (rows as in 2)"], False),
    ("4", "Serial thinking",
     ["CODES enter projected; operators", "move / extend them into one IDEA,", "an off-codebook, generative vector"], True),
    ("5", "Priming",
     ["spreading activation over the", "concept store's edges; lives with", "the codebook; decays"], False),
    ("6", "Expectation",
     ["NEGATIVE IMAGE: predicted idea,", "sign-reversed, added at the seal:", "c = o − g·ê (order 1+); learns o − ê"], True),
    ("7", "LTM",
     ["serial form ONLY: ideas, any order", "episodes = chained NP/VP;", "reached by CUE only (no last-N)"], False),
    ("8", "Budget",
     ["remaining work + closure pressure;", "READ to know cutoff is near,", "CHARGED by every thought op"], False),
    ("9", "Meronymic access",
     ["residual of part(x,y)=x·(y/|y|),", "left as an idea VECTOR (serial, 4),", "not a 1/0); order 0; no store"], True),
    ("10", "Taxonomic access",
     ["relations over higher-order symbols;", "emits a SYMBOLIC value (graded);", "every symbol participates; bounded"], False),
]

GO_SUBSYSTEM_NOTE = (
    "field (parallel, graded) → CODE = nearest-row projection; words arrive as codes → "
    "operators compose an IDEA: one off-codebook vector that must regenerate its codes + "
    "operations (generativity)   ·   stores: perceptual, conceptual, priming, LTM   ·   "
    "resource: budget   ·   relation: taxonomy")

GO_GRAMMARS = [
    ("<compose>  ·  understanding", [
        "reads: 1 perceptual knowing (word stream, row-owned), 2/3, 4, 5, 9",
        "never 6: PURITY — the composed idea o is the same whatever was predicted",
        "writes: 4 (push / fold), 2/3 (the composed idea)",
        "never: 7 LTM, 10 taxonomy, allocation, subgoals, 8",
        "gradient: live — operands, outputs, shared operator parameters",
        "cost: not metered (reading is not thinking)"]),
    ("<thought>  ·  thinking  (per-model allow-list)", [
        "reads: 4 the completed idea + recency buffer, 2/3, 6 (what is conceived), 8, 9, 10,",
        "and 7 only as frames a what() brought into STM by cue",
        "writes: effects on 2/3 and 4; the controller alone records / writes 7",
        "chooser: (operator, operands, open roles, level) | conclude",
        "gradient: none through effects; chooser on policy credit only",
        "cost: every choice / execute / descend / return charges 8"]),
    ("<generate>  ·  speech production", [
        "reads: the concluded idea in 4 (GIVEN, on-manifold), 2/3,",
        "its OWN emitted prefix — never the input stream or a parse trace",
        "writes: 1 the output stream (words via the reverse chain)",
        "never: 7, 10, thought execution",
        "gradient: live in generate; stops at the concluded idea",
        "cost: output walk budget (not 8)"]),
]

GO_GRAMMAR_NOTE = (
    "one vocabulary of operator identities; shared operators are the ONE gradient "
    "coupling between objectives (dissonance = per-operator gradient cosine); a compose face "
    "alone grants no thought permission   ·   the SEAL (not an operator) adds the negative image")

# group -> rows of (operator, faces, roles, {subsystem id: access}, note); W* = FutureWork §7
GO_OPERATORS = [
    ("structural only (compose + generate)", [
        ("not / non", "C G", "I1→O1", {"2": "RW", "3": "RW", "4": "RW"},
         "not = sign reversal, order 1+ (expectation's image); non = withdrawal, any order "
         "(attention's exclusion)"),
        ("conjunction / disjunction", "C G", "I1,I2→O1", {"2": "R", "3": "RW", "4": "RW"},
         "symbolic tier"),
        ("intersection / union", "C G", "I1,I2→O1", {"2": "RW", "4": "RW"}, "subsymbolic tier"),
        ("sum / product", "C G", "I1,I2→O1", {"2": "RW", "4": "RW"},
         "additive / multiplicative concept ops"),
        ("lift · verb · adverb · lower", "C G", "I1,I2→O1", {"2": "RW", "4": "RW", "5": "R"},
         "VP application; lift = eig edit"),
        ("preposition · bind · tense · morphology", "C G", "I1(,I2)→O1",
         {"1": "R", "2": "RW", "4": "RW"}, "bind: referents from the serial stream"),
    ]),
    ("two-faced (compose + thought + generate)", [
        ("part / whole (one family, I2,I1)", "C T G", "I1,I2→O1; open I1=parts, I2=wholes",
         {"2": "R", "3": "R", "4": "RW", "8": "W", "9": "R", "10": "R"},
         "effect = idea-vector residual left in 4 (9); scalar truth derived; "
         "higher order → symbolic value via 10"),
        ("equal", "C T G", "I1,I2→O1", {"2": "R", "3": "R", "4": "RW", "8": "W"},
         "mutual parthood on payloads"),
        ("exist", "C T G", "I1→O1", {"2": "R", "4": "RW", "7": "R", "8": "W"},
         "T reads facts among frames in STM"),
        ("quantize", "C T G", "I1→O1", {"2": "R", "3": "W", "8": "W"},
         "snap to the codebook; keeps ideas on-manifold"),
        ("arma", "C T G", "I1→O1", {"4": "R", "6": "W", "8": "W"},
         "reads the recency buffer; its estimate, sign-reversed, is the NEGATIVE IMAGE the seal "
         "adds; positive as the <generate> seed; never a fact"),
        ("what (Q)", "C T G", "I1→O1",
         {"2": "W*", "3": "W*", "4": "RW", "7": "R", "8": "W"},
         "what(Q, where?, when?): wh-word = open role; cue → code postings → rank → frames; "
         "episodes return frame by frame  (* FutureWork §7)"),
    ]),
    ("asymmetric / deferred / planned", [
        ("lookup", "C T –", "I1,I2→O1", {"4": "W", "7": "R", "8": "W"},
         "same retrieval as what / parts / wholes — one mechanism; no generate face?"),
        ("true", "– T? –", "I1→O1 (sealed clause NP→REF(S))", {"7": "R", "10": "R"},
         "deferred to two-truths"),
        ("(subsymbolic LM operator)", "C   G", "I1..In→O1",
         {"1": "R", "2": "RW", "4": "RW", "5": "R"}, "SAME row as structural faces; nothing more"),
    ]),
    ("not operators", [
        ("thought chooser", "– T –", "(op, operands, open roles, level) | conclude",
         {"2": "R", "3": "R", "4": "R", "6": "R", "7": "R", "8": "R"},
         "context = recency buffer (live STM + last 8 ideas) + cued LTM frames; reads 6 as c "
         "per role; policy credit only, never residual credit"),
        ("the seal", "– – –", "o → c = o − g·(1 − m)·κ·ê",
         {"4": "R", "6": "RW", "7": "W"},
         "AFTER composition (purity); κ = predicted presence; m = open roles (object spared); "
         "empty expected role = absence, concluded in thought"),
    ]),
    ("… (add)", [("", "", "", {}, "")] * 3),
]

GO_QUESTIONS = [
    "Open questions for the pen:  (1) negating a 0-order point gives a REGION, so the image −ê "
    "is order 1+ even for a bare noun — right?   (2) is priming read by thought, or only as a "
    "retrieval cue + by compose?   (3) lookup without a generate face — intended?",
    "(4) which structural operators may a model think with — `not`, to conclude absences?   "
    "(5) why / how as walks over implies / operator rows — wait for two-truths?   "
    "(6) rows keep o with the estimate linked (c derived) — or should a row hold only its "
    "residual?",
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
              "ê": 556, "…": 1000, "’": 222})


def _helv_width(s, fs, bold=False):
    return sum(_HELV.get(c, 556) for c in s) / 1000 * fs * (1.06 if bold else 1.0)


def make_grammar_operators():
    INK, SOFT, FAINT, MUTE = "#111", "#333", "#444", "#666"
    SANS = "Helvetica, Arial, sans-serif"
    W, X0 = 1760, 40

    def t(x, y, s, fs=13, fill=INK, bold=False, italic=False, right=None):
        # right = the x the text must not pass (default: the sheet's right margin)
        over = x + _helv_width(s, fs, bold) - ((W - X0) if right is None else right - 2)
        if over > 0:
            raise ValueError(f"grammar_operators: {over:.0f}px too wide at {fs}px: {s!r}")
        return txt(x, y, escape(s, quote=False), anchor="start", fs=fs, fill=fill,
                   bold=bold, italic=italic, ff=SANS)

    parts = [None]                       # the white ground, sized once H is known
    parts.append(t(X0, 44, "BasicModel — the three grammars, their operators, "
                   "and what each face may touch", fs=22, bold=True))
    parts.append(t(X0, 68, f"Draft for annotation · {GO_REV} · operators have no return "
                   "values, only effects: a signature is roles × faces × read (R) / write (W) "
                   "over the accessible mind", fs=13, fill=FAINT, italic=True))

    # A. subsystems
    y = 100
    parts.append(t(X0, y, "A.  The accessible mind — ten subsystems (dashed = reducible or "
                   "derived; every access is bounded)", fs=16, bold=True))
    y += 14
    bw, bh, gap = 160, 98, 8
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
    gw, gh, ggap = 546, 150, 20
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
    cols = [("operator", 250), ("faces", 62), ("roles", 238)] + [(i, 44) for i in ids] \
        + [("note", 690)]
    total, top, rowh = sum(w for _, w in cols), y, 24
    parts.append(rect(X0, y, total, 26, "#eee", stroke=SOFT, sw=1, rx=0))
    cx = X0
    for name, w in cols:
        parts.append(t(cx + 6, y + 17, name, fs=11, bold=True, right=cx + w))
        cx += w
    y += 26
    for group, rows in GO_OPERATORS:
        parts.append(rect(X0, y, total, rowh, "#f7f7f7", stroke=SOFT, sw=0.8, rx=0))
        parts.append(t(X0 + 6, y + 16, group, fs=11.5, fill=SOFT, bold=True, italic=True))
        y += rowh
        for name, faces, roles, access, note in rows:
            parts.append(rect(X0, y, total, rowh, "#fff", stroke="#999", sw=0.6, rx=0))
            cells = [name, faces, roles] + [access.get(i, "") for i in ids] + [note]
            cx = X0
            for (col, w), value in zip(cols, cells):
                if value:
                    numeric = col in ids
                    parts.append(t(cx + 6, y + 16, value, fs=11.5 if numeric else 10.8,
                                   bold=numeric, right=cx + w))
                cx += w
            y += rowh
    cx = X0
    for _, w in cols:
        parts.append(line(cx, top, cx, y, stroke="#bbb", sw=0.6))
        cx += w

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
    "mm5m_architecture.svg": make_mm5m,
    "grammar_operators.svg": make_grammar_operators,
}


def main(argv):
    names = argv[1:] or list(DIAGRAMS)
    unknown = [n for n in names if n not in DIAGRAMS]
    if unknown:
        sys.exit(f"unknown diagram(s): {', '.join(unknown)}; known: {', '.join(DIAGRAMS)}")
    for fname in names:
        fp = os.path.join(DIR, fname)
        content = DIAGRAMS[fname]()      # generate first: a width error must not truncate
        with open(fp, "w") as f:
            f.write(content)
        print(f"✓  {fp}")


if __name__ == "__main__":
    main(sys.argv)
