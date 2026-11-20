"""Thought reads the structure: the parts and the wholes of a concept, from
the Sigma and the Pi themselves (the edges of :func:`ConceptSpread.structure`).

``parts(c)``: what c reads, each entry with its weight and whether it is a
Not entry. An entry on a symbol is given as the concepts the symbol holds
(the symbols' Pi); an entry on a basic percept is given as itself.
``wholes(c)``: the concepts that read a symbol holding c, each with the
weight of that entry. A read moves nothing: no activation, weight or row.
"""
from dataclasses import dataclass

import torch

from util import TheXMLConfig


@dataclass(frozen=True)
class Entry:
    reference: tuple
    weight: float
    negated: bool
    words: tuple


def threshold():
    """The least weight at which an entry is read as present."""
    return float(TheXMLConfig.get('architecture.structureReadThreshold', default=1e-3))


def _reference(owner, graph, node):
    if node < graph.concepts:
        cid = owner.concept_id_at_row(node)
        return None if cid is None or int(cid) < 0 else ('sym', int(cid))
    if graph.percept_start <= node < graph.part_start:
        return ('percept', int(node - graph.percept_start))
    return None


def _words(owner, reference):
    definitions = getattr(owner, 'definitions', None)
    if reference is None or reference[0] != 'sym' or definitions is None:
        return ()
    return tuple(int(word) for word in definitions.words(reference[1]))


def _graph(owner):
    from ConceptSpread import structure
    with torch.no_grad():
        return structure(owner)


def _charge(work):
    return work is None or work.consume('expansion')


@torch.no_grad()
def parts(owner, cid, *, work=None):
    """The entries of concept ``cid``'s reading, ``[Entry]``."""
    row = owner._concept_row_of(int(cid))
    if row is None:
        return []
    graph = _graph(owner)
    least, found = threshold(), {}
    reads = (graph.parent == int(row)).nonzero().flatten().tolist()
    for e in reads:
        weight, negated = float(graph.weight[e]), bool(graph.negated[e])
        if weight < least:
            continue
        child = int(graph.child[e])
        if graph.concepts <= child < graph.percept_start:
            held = graph.child[graph.parent == child].tolist()
        else:
            held = [child]
        for node in held:
            reference = _reference(owner, graph, int(node))
            if reference is None or reference in found:
                continue
            if not _charge(work):
                return list(found.values())
            found[reference] = Entry(reference, weight, negated, _words(owner, reference))
    return list(found.values())


@torch.no_grad()
def wholes(owner, cid, *, work=None):
    """The concepts that read a symbol holding concept ``cid``, ``[Entry]``."""
    row = owner._concept_row_of(int(cid))
    if row is None:
        return []
    graph = _graph(owner)
    least, found = threshold(), {}
    symbols = graph.parent[(graph.child == int(row)) & (graph.parent >= graph.concepts)
                           & (graph.parent < graph.percept_start)].tolist()
    for symbol in symbols:
        for e in (graph.child == int(symbol)).nonzero().flatten().tolist():
            weight = float(graph.weight[e])
            reference = _reference(owner, graph, int(graph.parent[e]))
            if weight < least or reference is None or reference in found:
                continue
            if not _charge(work):
                return list(found.values())
            found[reference] = Entry(reference, weight, bool(graph.negated[e]), _words(owner, reference))
    return list(found.values())


@torch.no_grad()
def readers(owner, percept, *, work=None):
    """The concepts of order zero that read basic percept ``percept``."""
    sigma = getattr(owner, 'order_zero', None)
    if sigma is None:
        return []
    result = []
    for concept in sigma.rows_reading(int(percept)):
        cid = owner.concept_id_at_row(concept)
        entry = sigma.find(concept, int(percept))
        value = float(sigma.value[entry])
        if abs(value) < threshold() or cid is None or not _charge(work):
            continue
        reference = ('sym', int(cid))
        result.append(Entry(reference, abs(value), value < 0, _words(owner, reference)))
    return result
