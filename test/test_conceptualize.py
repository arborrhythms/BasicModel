"""SymbolSubSpace.conceptualize() -- the unified concept-formation API
(Alec 2026-06-21; doc/old/2026-06-21-higher-order-symbolic-composition.md
sections 2b/4b/4c). A concept is a flexible combination of two percepts;
conceptualize() dispatches the three orders to the ConceptualSpace symbol
tables (the duality: the SS subspace owns the method, CS owns the tables)."""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(name):
    import Models, Language
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def test_conceptualize_dispatch():
    m = _build("MM_symbol_tower.xml")
    ss = m.symbolSpace.subspace
    cs = ss.conceptualSpace

    # order 0: [part, whole] via relate -- idempotent, stored by reference.
    c0 = ss.conceptualize(0, part=5, whole=7)
    assert isinstance(c0, int)
    assert ss.conceptualize(0, part=5, whole=7) == c0           # idempotent
    assert sorted(cs._concept_parts.get(c0, ())) == [5]
    assert sorted(cs._concept_wholes.get(c0, ())) == [7]

    # order 1: [object isa word] meta -> (A=word, B=object, C=meta).
    c1 = ss.conceptualize(1, word_parts=[1, 2], word_whole=9, key="cat")
    assert c1 is not None and len(c1) == 3

    # order 2: higher-order object via synthesize_higher_order.
    c2 = ss.conceptualize(2, parts=[10, 11, 12])
    assert isinstance(c2, int)

    # missing inputs -> None (no spurious concepts minted).
    assert ss.conceptualize(0) is None
    assert ss.conceptualize(1) is None
    assert ss.conceptualize(2) is None


def test_conceptualize_chain():
    """order-3 = Gallistel sequence chain: a tail-recursive [whole, part] list
    over concept pairs (head whole = first concept, part = the rest-chain)."""
    m = _build("MM_symbol_tower.xml")
    ss = m.symbolSpace.subspace
    cs = ss.conceptualSpace
    a = ss.conceptualize(0, part=1, whole=2)
    b = ss.conceptualize(0, part=3, whole=4)
    c = ss.conceptualize(0, part=5, whole=6)

    head = ss.conceptualize(3, concept_ids=[a, b, c])
    assert isinstance(head, int)
    assert ss.conceptualize(3, concept_ids=[a, b, c]) == head     # idempotent
    assert ss.conceptualize(3, concept_ids=[c, b, a]) != head     # ORDERED
    assert ss.conceptualize(3, concept_ids=[a]) == a              # singleton
    assert ss.conceptualize(3, concept_ids=[]) is None            # empty
    # head's whole is the first concept (the part carries the rest-chain).
    assert ("sym", a) in cs._concept_wholes.get(head, set())
