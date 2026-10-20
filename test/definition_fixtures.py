"""Definition owner for isolated conceptual-inventory mechanism fixtures."""
from Layers import TernaryTruthStore


def with_definitions(cs):
    object.__setattr__(cs, '_definition_store', TernaryTruthStore(cs.nDim, capacity=1024))
    return cs


def witness_object(cs, word):
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
    return obj
