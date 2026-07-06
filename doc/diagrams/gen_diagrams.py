#!/usr/bin/env python3
"""Generate WikiOracle conceptual diagrams as SVG files."""
import os, math

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

def rect(x, y, w, h, fill, stroke=None, sw=1.5, rx=6, op=1.0):
    s = f" stroke='{stroke}' stroke-width='{sw}'" if stroke else ""
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{fill}'"
            f" fill-opacity='{op}' rx='{rx}'{s}/>")

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


# ── Write files ────────────────────────────────────────────────────────────────

files = [
    ("ternary_logic.svg",    make_ternary()),
    ("luminosity.svg",       make_luminosity()),
    ("vector_spaces.svg",    make_vector_spaces()),
    ("mm5m_architecture.svg", make_mm5m()),
]

for fname, content in files:
    fp = os.path.join(DIR, fname)
    with open(fp, "w") as f:
        f.write(content)
    print(f"✓  {fp}")
