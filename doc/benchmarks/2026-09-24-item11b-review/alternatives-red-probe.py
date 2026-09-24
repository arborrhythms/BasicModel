import Spaces
from test_cs_sparse_weights import _cs

def test_alternatives_do_not_trigger_conjunctive_overcollection():
    cs = _cs()
    for n in range(6):
        A, _, _ = cs.create_word_object_meta([20+n], [10+n], key='word')
    cs.refine_over_collected()
    assert A not in Spaces._concept_alloc_of(cs).retired
