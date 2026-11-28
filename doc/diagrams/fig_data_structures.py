#!/usr/bin/env python3
"""Write data_structures.svg beside this script using only the standard library.

Run from the repository root with python3 doc/diagrams/fig_data_structures.py.
The STM and LTM column lists come from bin/Memory.py. The associations follow
BasicPercepts, Spaces, Layers, Understanding, SentenceStrata, RequestRead and ReadingMemory.
"""
import ast
from html import escape
from pathlib import Path
from textwrap import wrap

from gen_diagrams import BLUE, BLUE_DK, GRY, PURP, TEAL, figure, line, rect, txt, write

FILENAME = "data_structures.svg"
MEMORY_SOURCE = Path(__file__).resolve().parents[2] / "bin" / "Memory.py"
BOX_WIDTH = 334
GAP = 142
LEFT = 26
MIDDLE = LEFT + BOX_WIDTH + GAP
RIGHT = MIDDLE + BOX_WIDTH + GAP
WIDTH = RIGHT + BOX_WIDTH + LEFT


def memory_columns():
    """Read the declared storage schemas without importing torch or starting a model."""
    classes = {node.name: node for node in ast.parse(MEMORY_SOURCE.read_text(encoding="utf-8")).body
               if isinstance(node, ast.ClassDef)}
    columns = {}
    for name, attribute in (("STMChunks", "FIELDS"), ("IdeaRows", "columns"), ("LTM1", "columns")):
        for statement in classes[name].body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == attribute for target in statement.targets
            ):
                fields = ast.literal_eval(statement.value)
                columns[name] = wrap(", ".join(fields), width=38, break_long_words=False)
                break
        else:
            raise ValueError(f"{MEMORY_SOURCE} has no declaration for {name}.{attribute}")
    return columns


def class_box(body, x, y, title, implementation, color, sections):
    """Draw a UML classifier with one compartment per representation or API."""
    height = 55 + sum(34 + 18 * len(fields) for _, fields in sections)
    body.extend((
        rect(x, y, BOX_WIDTH, height, "white", color, rx=3),
        txt(x + BOX_WIDTH / 2, y + 23, title, fs=18, fill=color, bold=True),
        txt(x + BOX_WIDTH / 2, y + 43, implementation, fs=13, fill=GRY),
    ))
    cursor = y + 55
    for label, fields in sections:
        body.append(line(x, cursor, x + BOX_WIDTH, cursor, stroke=color, sw=1))
        body.append(txt(x + 13, cursor + 20, label, anchor="start", fs=14, fill=color, bold=True))
        for i, field in enumerate(fields):
            body.append(txt(x + 13, cursor + 41 + 18 * i, escape(field), anchor="start",
                            fs=13, ff="monospace"))
        cursor += 34 + 18 * len(fields)
    return y + height


def association(body, points, label, x, y, *, dependency=False):
    """Draw a reference or a dashed dependency between representations."""
    dash = ' stroke-dasharray="6 4"' if dependency else ""
    coords = " ".join(f"{px},{py}" for px, py in points)
    body.append(f'<polyline points="{coords}" fill="none" stroke="{BLUE_DK}" '
                f'stroke-width="1.5"{dash} marker-end="url(#association)"/>')
    body.append(txt(x, y, label, fs=13, fill=BLUE_DK))


def make():
    columns = memory_columns()
    body = ['<defs><marker id="association" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="8" markerHeight="8" orient="auto">'
            f'<path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="{BLUE_DK}"/></marker></defs>']
    top = 88
    percept_end = class_box(body, LEFT, top, "Percept", "«row» BasicPercepts", BLUE, (
        ("Basic percept identity", ["('bp', row)", "occurrences[row] = (parts, wholes)", "promoted[row]"]),
        ("Location-bound occurrences · PerceptBank", ["table: what | where | when", "two evidence lanes per channel", "span, token, time, activity"]),
        ("Percept input to order-zero Sigma", ["lanes(row)", "current and one-step-back activity", "when perceptDelay is enabled"]),
    ))
    concept_end = class_box(body, MIDDLE, top, "Concept", "«row» addressed by concept ID", PURP, (
        ("Identity · ConceptAllocator", ["('sym', cid)", "concepts, order_zero_members"]),
        ("Constituents · ConceptualAttentionLayer", ["constituents(cid)"]),
        ("Located evidence · ConceptualField", ["concept_ids, evidence", "where, when, percepts, wires"]),
        ("Unit delay · ConceptualSpace", ["current and previous observations", "kept priming trace → next request"]),
    ))
    relation_end = class_box(body, RIGHT, top, "LTM · relation", "LTM1", BLUE_DK, (
        ("Relation columns", columns["LTM1"]),
        ("Ordered pair of tagged endpoints", ["SYMBOL | OCCURRENCE | RELATION"]),
        ("Sequence of three occurrences", ["head = (A, tail)", "tail = (B, C)"]),
        ("Transitions · ideaTransitions (default off)", ["add_transition(before, after)", "marked pair of occurrence names"]),
        ("Find incoming references", ["rows_referencing(kind, value)"]),
    ))

    # Allow room between the two ranks for labels on the vertical associations.
    middle_y = max(percept_end, concept_end, relation_end) + 44
    bottom = middle_y + 55
    stm_end = class_box(body, LEFT, bottom, "STM", "ShortTermMemory", BLUE, (
        ("Working set · STMChunks", columns["STMChunks"]),
        ("Working ideas and ordered relations", ["strata.idea_stores → IdeaStack (STM0)", "strata.recent_stores → STM0",
          "_thought_stores → STM0", "STM0.parts → STM1"]),
        ("Attention state · ModelState", ["last_where: last native span", "when_time, when_code", "root_at(stream, tick) → kept head"]),
        ("Whole stack read by attend", ["STMChunks.request_view(state)", "newest first, followed by depth"]),
    ))
    mtm_end = class_box(body, MIDDLE, bottom, "MTM", "MediumTermMemory", TEAL, (
        ("Retained identities", ["entries → Token", "names → Name", "capacity, horizon"]),
        ("Token references", ["occurrences, symbol, transient", "timelines → Timeline → Event"]),
        ("Candidate stores", ["MTM1, MTMPairRows"]),
        ("Read occurrences under one head", ["read_occurrences(store, head)", "token_of(occurrence)"]),
    ))
    ltm_end = class_box(body, RIGHT, bottom, "LTM · occurrence", "LTM0 extends IdeaRows", BLUE_DK, (
        ("Occurrence columns · IdeaRows", columns["IdeaRows"]),
        ("Stable occurrence address", ["('ltm', 'name-v1', row_name)"]),
        ("Transitions within a stream", ["keep_transitions(stream, before)"]),
        ("Read by row", ["referent_of(index)", "meaning_of(index)"]),
    ))

    top_link = top + 78
    association(body, [(MIDDLE, top_link), (LEFT + BOX_WIDTH, top_link)],
                "order-zero members", MIDDLE - GAP / 2, top_link - 9)
    association(body, [(MIDDLE + BOX_WIDTH, top_link), (RIGHT, top_link)],
                "allocator.parts", RIGHT - GAP / 2, top_link - 9)
    left_centre = LEFT + BOX_WIDTH / 2
    concept_left = MIDDLE + 40
    association(body, [(left_centre, bottom), (left_centre, middle_y),
                       (concept_left, middle_y), (concept_left, concept_end)],
                "concept_rows", (left_centre + concept_left) / 2, middle_y - 8)
    concept_right = MIDDLE + BOX_WIDTH - 38
    association(body, [(concept_right, bottom), (concept_right, concept_end)],
                "Token.symbol (0..1)", concept_right - 64, middle_y + 14)
    relation_x = RIGHT + BOX_WIDTH / 2
    association(body, [(relation_x, bottom), (relation_x, relation_end)],
                "parts", relation_x + 23, middle_y + 14)
    bottom_link = bottom + 78
    association(body, [(MIDDLE + BOX_WIDTH, bottom_link), (RIGHT, bottom_link)],
                "kept occurrences", RIGHT - GAP / 2, bottom_link - 9)
    commit_y = max(stm_end, mtm_end, ltm_end) + 38
    association(body, [(left_centre, stm_end), (left_centre, commit_y),
                       (relation_x, commit_y), (relation_x, ltm_end)],
                "ReadingMemory.commit retains the STM0 ideas and STM1 pairs reachable from each selected head",
                WIDTH / 2, commit_y - 9, dependency=True)
    height = commit_y + 20
    drawing = figure(WIDTH, height,
                  "BasicModel data structures",
                  "Solid arrows show references. The dashed arrow shows how a grammar-free reading is committed.", body, markers=())
    return drawing.replace("<svg ", f'<svg viewBox="0 0 {WIDTH} {height}" ', 1)


if __name__ == "__main__":
    write(FILENAME, make())
