#!/usr/bin/env python3
"""Every grammar operator with its arguments and its output.

Faces, arities, output counts and rule attributes (head, referent, clause,
permutation) are parsed from data/complete.grammar, the reference model's
grammar, when the figure is generated. The meaning of each argument and output
comes from the table below. Generation fails when the table and the grammar
disagree, so a grammar change cannot leave this figure stale.

Sources for the table: the layer classes and their head roles (bin/Language.py,
bin/Interpret.py), the thought descriptors (bin/Queries.py), and the operator
sheet (grammar_operators.svg). Embedded in doc/Language.md.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_diagrams import *  # noqa: F401,F403
from gen_diagrams import _helv_width

FILENAME = "operator_signatures.svg"
GRAMMAR = os.path.join(DIR, os.pardir, os.pardir, "data", "complete.grammar")
SANS = "Helvetica, Arial, sans-serif"

# group key -> (title, subtitle, accent, light fill)
GROUPS = {
    "field": ("Field operations", "attention over a bracketed field; operands are field brackets, not symbols", BLUE, BLUE_L),
    "leaf": ("Leaf readings", "applied to a word at a leaf; interpret is forced at the word transaction in reading mode", TEAL, TEAL_L),
    "fold": ("Evidence folds", "on the meaning block each pole is reduced by the declared rule; numeric models use the Ops kernels", GRN_DK, GRN_L),
    "phrase": ("Phrase structure", "the filled argument heads the result; marker-headed operators also have a unary face", PURP, PURP_L),
    "relation": ("Relations", "structural carriers in compose and generate; checked thought faces run only at a completed boundary", ORG, ORG_L),
    "thought": ("Thought only", "boundary executors; every result is an evidence pair (for, against)", RED, RED_L),
    "cover": ("Perceptual cover (PartSpace)", "compose is Synthesize, generate is Analyze; U is the whole input field", GRY, GRY_L),
}

# operator -> (group, argument roles, head argument (1-based, 0 = none), output role(s), note)
SIGNATURES = {
    "divide": ("field", ["field bracket"], 0, "narrower brackets", "a heterogeneous bracket splits at a pole disagreement"),
    "descend": ("field", ["field bracket"], 0, "its byte parts", "an unknown word descends to its retained parts"),
    "gloss": ("field", ["field bracket"], 0, "one word", "a pure singular word is read as a word"),
    "and": ("field", ["field bracket"], 0, "reduced bracket", "for: min over positions; against: max"),
    "or": ("field", ["field bracket"], 0, "reduced bracket", "for: max over positions; against: min"),

    "interpret": ("leaf", ["word (its form)"], 0, "its object", "a pending word is minted at its first kept definition"),
    "witness": ("leaf", ["word handle"], 0, "MTM token, null kind", "reading mode: valid at an open handle; > renamed sign (thinking spec §32.4)"),

    "not": ("fold", ["meaning"], 0, "lanes exchanged", "self-inverse; a field operand in compose, a meaning in thought"),
    "non": ("fold", ["meaning"], 0, "for lane cleared", "the operand is kept as a marked witness; no inverse"),
    "intersection": ("fold", ["meaning", "meaning"], 0, "meet", "for: min, against: max"),
    "union": ("fold", ["meaning", "meaning"], 0, "join", "for: max, against: min"),
    "conjunction": ("fold", ["meaning", "meaning"], 0, "bound pair", "numeric: product binding; meaning block: as intersection"),
    "disjunction": ("fold", ["meaning", "meaning"], 0, "bundle", "numeric: a + b − ab; meaning block: as union"),
    "sum": ("fold", ["meaning", "meaning"], 0, "mean", "a known operand gives an exact inverse"),
    "product": ("fold", ["meaning", "meaning"], 0, "product", "numeric a · b; meaning block: a learned lane kernel"),

    "compound": ("phrase", ["modifier", "head"], 2, "head, its cases selected", "the modifier selects the head's native sigma cases"),
    "compound_unary": ("phrase", ["head"], 1, "head, implicit object", "the same case selection with no modifier"),
    "verb": ("phrase", ["verb", "object"], 1, "verb phrase", "the object's learned gain subtypes the verb"),
    "adverb": ("phrase", ["verb phrase", "adverb"], 1, "verb phrase", "multiplicative; exact given the adverb; withholds lowering"),
    "lift": ("phrase", ["noun phrase", "verb phrase"], 2, "S, the clause", "the sentence fold; NP + VP lowers one order"),
    "lower": ("phrase", ["marker", "description"], 2, "a member", "a determiner selects; the marker adds no content"),
    "definite": ("phrase", ["marker", "description"], 2, "a bound referent", "the lower layer, binding an address"),
    "universal": ("phrase", ["marker", "description"], 2, "the kind", "the lower layer, retaining the kind"),
    "lower_unary": ("phrase", ["description"], 1, "a member", "the unary face, with no marker"),
    "definite_unary": ("phrase", ["description"], 1, "a bound referent", "the unary face, with no marker"),
    "generic": ("phrase", ["description"], 1, "a generic type", "nominal type marker; the reference role supplies the order"),
    "preposition": ("phrase", ["marker", "phrase"], 2, "prepositional phrase", "the phrase passes with a .where edit"),
    "preposition_unary": ("phrase", ["phrase"], 1, "prepositional phrase", "the unary face, with no marker"),
    "surface": ("phrase", ["marker", "content"], 2, "content, marked", "the content is preserved; both operands kept"),
    "surface_unary": ("phrase", ["content"], 1, "content, marked", "the unary face, with no marker"),
    "transpose": ("phrase", ["left", "right"], 0, ["right", "left"], "an exact 2 → 2 rewrite of the two newest constituents"),
    "bind": ("phrase", ["mention", "location"], 1, "an identity", "an accessible location or a zero-logit dummy; thought: one open site"),

    "part": ("relation", ["part", "whole"], 0, "parthood relation", "thought: containment evidence, its residual and support"),
    "whole": ("relation", ["whole", "part"], 0, "parthood relation", "the converse of part"),
    "equal": ("relation", ["left", "right"], 0, "identity relation", "thought: conceptual identity evidence"),
    "implies": ("relation", ["antecedent", "consequent"], 0, "implication", "thought: containment evidence"),
    "ask": ("relation", ["description"], 0, "a subgoal", "thought: fill the open references, else the LTM best match"),

    "query": ("thought", ["description"], 0, "content (a set)", "the bounded LTM best match"),
    "isTrue": ("thought", ["description"], 0, "truth evidence", "an ended row's pair, its trust and witnessing"),
    "exist": ("thought", ["concept"], 0, "presence evidence", "the concept's extent; no scalar truth"),
    "isPart": ("thought", ["reference", "reference"], 0, "taxonomy evidence", "LTM1 relation evidence first; one side may be open"),
    "isEqual": ("thought", ["reference", "reference"], 0, "definition evidence", "reads LTM1 relation rows"),
    "infer": ("thought", ["premise", "rule"], 0, "the consequent", "or opens the antecedent as a subgoal; both addressed by .where"),
    "inferBackward": ("thought", ["rule", "premise"], 0, "the consequent", "infer with its arguments swapped"),
    "intuit": ("thought", ["meaning"], 0, "a proposal (knowing)", "order-0 sigma over pi over LTM1 rules; output-owned weights"),
    "gain": ("thought", ["concept"], 0, "expectation's gain", "set from the concept's magnitude"),

    "stop": ("cover", ["U"], 0, "U", "declared cover rule; no layer class or effect declaration"),
    "boundary": ("cover", ["U", "U"], 0, "U", "declared cover rule; no layer class or effect declaration"),
    "uniform": ("cover", ["U", "U"], 0, "U", "declared cover rule; no layer class or effect declaration"),
    "synthesize": ("cover", ["U", "U"], 0, "U", "residual-bearing additive synthesis; analyze is its exact inverse"),
}

FACE_OF_SECTION = {"compose": "C", "Synthesize": "C", "thought": "T", "generate": "G", "Analyze": "G"}


def read_grammar(path):
    """op -> dict(faces, arity per face, outputs, attributes) from the grammar file."""
    text = re.sub(r"<!--.*?-->", "", open(path, encoding="utf-8").read(), flags=re.S)
    ops = {}
    for section in re.finditer(r"<(compose|thought|generate|Synthesize|Analyze)>(.*?)</\1>", text, re.S):
        face = FACE_OF_SECTION[section.group(1)]
        for attrs, body in re.findall(r"<rule([^>]*)>(.*?)</rule>", section.group(2)):
            m = re.match(r"\s*(.*?)\s*=\s*(\w+)\.(\w+)\((.*?)\)\s*$", body)
            if m is None:
                raise ValueError(f"operator_signatures: unparsed rule {body!r}")
            lhs, op, method, args = m.groups()
            entry = ops.setdefault(op, dict(faces=set(), arity={}, outputs=1, attrs={}))
            entry["faces"].add(face)
            n_args = len([a for a in args.split(",") if a.strip()])
            n_lhs = len([a for a in lhs.split(",") if a.strip()])
            if face == "G":                       # reverse: output -> inputs
                entry["arity"].setdefault("G", n_lhs)
            else:
                entry["arity"][face] = n_args
                entry["outputs"] = max(entry["outputs"], n_lhs)
            for key, value in re.findall(r'(\w+)="([^"]*)"', attrs):
                entry["attrs"].setdefault(key, value)
    return ops


def check(ops):
    missing, extra = sorted(set(ops) - set(SIGNATURES)), sorted(set(SIGNATURES) - set(ops))
    if missing or extra:
        raise ValueError(f"operator_signatures: grammar and table disagree; missing {missing}, extra {extra}")
    for op, (group, args, head, out, note) in SIGNATURES.items():
        g = ops[op]
        arity = g["arity"].get("C", g["arity"].get("T"))
        if arity != len(args):
            raise ValueError(f"operator_signatures: {op} has {arity} arguments in the grammar, {len(args)} in the table")
        n_out = len(out) if isinstance(out, list) else 1
        if n_out != g["outputs"]:
            raise ValueError(f"operator_signatures: {op} has {g['outputs']} outputs in the grammar, {n_out} in the table")
        declared = g["attrs"].get("head")
        if declared is not None and int(declared[5:]) != head:
            raise ValueError(f"operator_signatures: {op} head is {declared} in the grammar, {head} in the table")


def make():
    ops = read_grammar(GRAMMAR)
    check(ops)
    W, X0, COLS, CW, RH = 1620, 30, 3, 500, 78
    gap = (W - 2 * X0 - COLS * CW) // (COLS - 1)
    b = []

    def t(x, y, s, fs=11.5, fill=BLUE_DK, bold=False, italic=False, anchor="start", limit=None):
        if limit is not None and _helv_width(s, fs, bold) > limit:
            raise ValueError(f"operator_signatures: {s!r} is wider than {limit}px at {fs}px")
        return txt(x, y, escape(s, quote=False), anchor=anchor, fs=fs, fill=fill, bold=bold, italic=italic, ff=SANS)

    def pill(x, y, w, label, accent, light, filled):
        return (rect(x, y, w, 22, accent if filled else "#fff", stroke=accent, sw=1.3, rx=11) +
                t(x + w / 2, y + 15, label, fill="white" if filled else BLUE_DK, bold=filled,
                  anchor="middle", limit=w - 10))

    def card(x, y, op):
        group, args, head, out, note = SIGNATURES[op]
        _, _, accent, light = GROUPS[group]
        g = ops[op]
        out = []
        ys = [y + 17] if len(args) == 1 else [y + 4, y + 30]
        for k, (role, py) in enumerate(zip(args, ys), start=1):
            out.append(line(x + 150, py + 11, x + 176, y + 28, stroke=accent, sw=1.3))
            out.append(pill(x, py, 150, role, accent, light, filled=(k == head)))
        out.append(rect(x + 176, y + 6, 140, 44, light, stroke=accent, sw=1.6, rx=8))
        out.append(t(x + 246, y + 25, op, fs=13, bold=True, anchor="middle", limit=134))
        faces = "".join(f"<tspan fill='{accent if f in g['faces'] else '#c8c8c8'}' font-weight='bold'>{f}</tspan>  "
                        for f in "CTG")
        out.append(f"<text x='{x + 246}' y='{y + 42}' text-anchor='middle' font-size='10.5' "
                   f"font-family='{SANS}'>{faces.strip()}</text>")
        outs = out_roles = SIGNATURES[op][3]
        out_roles = outs if isinstance(outs, list) else [outs]
        oys = [y + 17] if len(out_roles) == 1 else [y + 4, y + 30]
        for role, py in zip(out_roles, oys):
            out.append(f"<line x1='{x + 316}' y1='{y + 28}' x2='{x + 336}' y2='{py + 11}' stroke='{accent}' "
                       f"stroke-width='1.3' marker-end='url(#arrd)'/>")
            out.append(pill(x + 340, py, 160, role, accent, light, filled=False))
        tags = []
        a = g["attrs"]
        if "referent" in a:
            tags.append(f"referent {a['referent']}")
        if "clause" in a:
            tags.append(f"clause {a['clause']}")
        if "permutation" in a:
            tags.append(f"family {a.get('family', '?')}, arguments {a['permutation'].replace('argin', '')}")
        if g["arity"].get("T") is not None and g["arity"].get("C") not in (None, g["arity"]["T"]):
            tags.append(f"thought takes {g['arity']['T']}")
        text = ("[" + "; ".join(tags) + "]  " if tags else "") + note
        out.append(t(x, y + 66, text, fs=10.5, fill=GRY, italic=True, limit=CW))
        return "\n".join(out)

    b.append(t(X0, 40, "Operators: their arguments and outputs", fs=24, bold=True, fill="#222"))
    b.append(t(X0, 64, "read from data/complete.grammar when generated  ·  filled argument: the head  ·  "
               "faces: C compose, T thought, G generate (lit when declared)  ·  > decided, not built",
               fs=12.5, fill=GRY, italic=True))
    y = 86
    order = list(GROUPS)
    for group in order:
        title, subtitle, accent, light = GROUPS[group]
        members = [op for op, sig in SIGNATURES.items() if sig[0] == group]
        b.append(rect(X0, y, W - 2 * X0, 30, light, stroke=accent, sw=1.2, rx=6))
        b.append(t(X0 + 12, y + 20, title, fs=14, bold=True, fill=accent))
        b.append(t(X0 + 14 + _helv_width(title, 14, True) + 14, y + 20, subtitle, fs=11.5, fill=BLUE_DK, italic=True))
        y += 42
        for k, op in enumerate(members):
            col, row = k % COLS, k // COLS
            b.append(card(X0 + col * (CW + gap), y + row * RH, op))
        y += ((len(members) + COLS - 1) // COLS) * RH + 8
    b.append(t(X0, y + 10, f"{len(SIGNATURES)} operators. Roles in the code are argin1 and argin2; "
               "the names here are their meanings. Heads come from the grammar's head attribute or the layer's head role.",
               fs=11.5, fill=GRY, italic=True))
    H = y + 28
    return svg(W, H, "<defs>" + arrow_marker("arrd", GRY, 7) + "</defs>\n" + "\n".join(b))


if __name__ == "__main__":
    write(FILENAME, make())
