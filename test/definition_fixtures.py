"""Definition owner for isolated conceptual-inventory mechanism fixtures."""
from Layers import TernaryTruthStore


def with_definitions(cs):
    from ClauseRow import attach_clause_index
    from types import SimpleNamespace
    store = TernaryTruthStore(cs.nDim, capacity=1024)
    object.__setattr__(cs, '_fixture_occurrences', store)
    attach_clause_index(SimpleNamespace(), store, cs)
    return cs


def witness_object(cs, word, *, point=None):
    """Supply an object with the form's observed parts, then witness its DEF.

    Reading an unfamiliar form does not itself testify that it has an object.
    Definition tests supply that testimony explicitly.
    """
    value = cs.interpret._pending[word]
    obj = cs.new_concept()
    cs._csw_concept_row(0, obj)
    for parts in value['parts']:
        for part in parts:
            cs.add_part(obj, part)
        cs._populate_concept_weights(
            obj, witness=(parts, value['wholes']), word_reading=True)
    for whole in value['wholes']:
        cs.add_whole(obj, whole)
    cs.interpret.define(word, obj)
    if point is not None:
        import torch
        with torch.no_grad():
            cs.similarity_codebook.W[cs._csw_row_of(obj)].copy_(point)
    return obj
