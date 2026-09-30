"""Additional definition invariants probed before repairs."""
import copy
import pytest
import torch
from test_item7_definitions import model, admit


def test_copied_definition_index_is_owned_by_the_copied_store():
    m=model(); word,obj=admit(m,'hello'); original=m.symbolSpace.ltm_store
    restored=copy.deepcopy(original)
    restored.clear_origin(restored.ORIGIN_CONVERSATION)
    assert original.definitions.deref(word)==obj
    assert restored.definitions.deref(word) is None
    assert restored.definitions._store() is restored


def test_two_pending_words_cannot_overbook_the_last_definition_row():
    m=model(); cs=m._concept_owner(); store=m.symbolSpace.ltm_store
    while len(store)<store.capacity-1:store.append_idea(torch.zeros(store.nDim))
    first=cs.interpret.lookup_word([7],[],form='first')
    alloc=cs._concept_allocator
    before=alloc.next_id,dict(alloc.placement),dict(alloc.layer()._tensor_rows)
    with pytest.raises(RuntimeError,match='capacity'):
        cs.interpret.lookup_word([8],[],form='second')
    assert before==(alloc.next_id,dict(alloc.placement),dict(alloc.layer()._tensor_rows))
    obj=cs.interpret.forward(first)
    assert cs.definitions.deref(first)==obj


def test_word_concepts_exist_before_grounded_case_discovery(tmp_path):
    import inspect
    import test_grounded_xor as original
    source=inspect.getsource(original.learn_grounded_xor).split('    # The second presentation admits')[0]
    source+='    cs._commit_autobind_from_stash()\n'
    source+="    assert all(cs.word_concepts(word) for word in ('00', '01', '10', '11'))\n"
    namespace=dict(vars(original));exec(compile(source,__file__+':word-first','exec'),namespace)
    namespace['learn_grounded_xor'](tmp_path,4)


def test_word_definition_survives_identity_resolution_and_pruning():
    from test_cs_sparse_weights import _cs
    cs=_cs()
    word,obj=cs.interpret_word([7],[1],key='word')
    before=cs.concept_parts(word),cs.concept_wholes(word)
    cs.resolve_identities()
    cs.refine_over_collected()
    assert (cs.concept_parts(word),cs.concept_wholes(word))==before
