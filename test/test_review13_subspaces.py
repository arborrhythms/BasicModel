"""Native form and detached meaning faces, paired by the shared index."""
from types import SimpleNamespace
import torch
from torch import nn


def fixture():
    from test_cs_sparse_weights import _cs
    from Spaces import Codebook, _concept_alloc_of
    cs = _cs(nS=32, order=0)
    ps = Codebook()
    ps.W = nn.Parameter(torch.zeros(16, 3))
    with torch.no_grad():
        ps.W[4] = torch.tensor([.2, .6, .8])
        ps.W[7] = torch.tensor([.8, .2, .4])
    object.__setattr__(cs, '_model', SimpleNamespace(perceptualSpace=SimpleNamespace(
        subspace=SimpleNamespace(what=ps), nDim=3)))
    cb = Codebook()
    cb.W = nn.Parameter(torch.zeros(32, 5))
    cs.similarity_codebook = cb
    cb.enable_derived_codes(cs, percept_width=3, content_width=5)
    cs._online_learning_frozen = False
    return cs, cb, ps, _concept_alloc_of(cs)


def word(cs, parts):
    cid = cs.new_concept()
    row = cs._csw_concept_row(0, cid)
    for part, weight in parts:
        cs.add_concept_feature(row, 'ps', part, weight)
    return row


def test_native_percept_rows_and_net_evidence_are_detached_code_sources():
    cs, cb, ps, alloc = fixture()
    row = word(cs, [(4, .75), (7, -.25)])
    result = cb.lookup_rows(row)
    torch.testing.assert_close(result, torch.tensor([.2, .6, .8, 0., 0.]))
    assert not isinstance(cb.W, nn.Parameter) and not cb.W.requires_grad
    assert list(cb.parameters()) == []
    assert not result.requires_grad
    assert ps.W.grad is None and alloc.layer().features.values.grad is None
    with torch.no_grad():
        ps.W[4, 1] += .1
        cb.W.fill_(99.)
    torch.testing.assert_close(cb.lookup_rows(row)[1], result.detach()[1] + .1)


def test_group_uses_native_perceptual_fold_without_letter_concept_rows():
    cs, cb, ps, alloc = fixture()
    grouped = word(cs, [((4, 7), .5)])
    single = word(cs, [(7, 1.)])
    torch.testing.assert_close(cb.lookup_rows(grouped)[:3], ps.W[[4,7]].amax(0))
    torch.testing.assert_close(cb.lookup_rows(single)[:3], ps.W[7])
    assert set(alloc.layer()._tensor_row_keys) == {grouped, single}


def test_occurrence_mean_is_confined_to_detached_context_coordinates():
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    cs, cb, ps, _ = fixture()
    a, b = word(cs, [(4, 1.)]), word(cs, [(7, 1.)])
    store = TernaryTruthStore(5, capacity=8)
    cs.symbolSpace = SimpleNamespace(ltm_store=store)
    root = torch.tensor([9.,8.,7.,.2,.4], requires_grad=True)
    roles = torch.stack((root, root*0, root*0))
    row = store.append_meaning(ConceptualMeaning(roles, torch.tensor([True,False,False]),
        mode='assertive', sentence_kind='idea'), evidence=(0.,0.))
    store._append_leaf_terms(row, ((a,b),(),()), (True,True,True))
    codes = cb.lookup_rows(torch.tensor([a,b]))
    torch.testing.assert_close(codes[:,:3], ps.W[[4,7]])
    torch.testing.assert_close(codes[:,3:], root.detach()[None,3:].expand(2,-1))
    assert not codes.requires_grad
    assert root.grad is None and not store.slots.requires_grad
    newer = store.append_meaning(ConceptualMeaning(roles*2, torch.tensor([True,False,False]),
        mode='assertive', sentence_kind='idea'), evidence=(0.,0.))
    store._append_leaf_terms(newer, ((a,b),(),()), (True,True,True))
    mean = (root.detach()/2 + 2*root.detach())/1.5
    torch.testing.assert_close(cb.lookup_rows(a)[:3], ps.W[4])
    torch.testing.assert_close(cb.lookup_rows(a)[3:], mean[3:])
    assert len(store) == 2


def test_support_reports_exact_zeros_and_minimum_absolute_coordinate():
    cs, cb, ps, _ = fixture()
    a = word(cs, [(4, 1.)])
    with torch.no_grad():
        ps.W[4, 0] = 0.
    report = cb.mereology.support_audit([a])[0]
    assert report['row'] == a and report['dimension'] == 3
    assert report['nonzero_fraction'] == 2/3
    assert report['minimum_absolute_value'] == 0.
    assert report['minimum_nonzero_absolute_value'] == float(ps.W[4,1])


def test_actual_serial_code_uses_six_native_coordinates_and_no_context_bootstrap():
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    model.forward(model.inputSpace.prepInput(['hello world', 'hello there']))
    cb = model._concept_owner().similarity_codebook
    assert cb.mereology.percept_width == 6
    assert cb.mereology.context_width == 0
    assert not isinstance(cb.W, nn.Parameter) and list(cb.parameters()) == []
    rows = model.inputSpace._ar_grammar_object_rows
    atoms = model.inputSpace._ar_grammar_object_atoms
    torch.testing.assert_close(atoms[rows>=0], cb.lookup_rows(rows[rows>=0]))
    assert not cb.lookup_rows(rows[rows>=0])[...,6:].any()
    groups = model.objective_parameter_groups(model.getOptimizer(lr=.01))
    parameter = model.perceptualSpace.subspace.what.W
    assert any(p is parameter for p in groups['reconstruction'])
    assert all(p is not parameter for p in groups['output']+groups['expectation'])


def test_readback_uses_perceptual_cosine_without_context_or_scale():
    from SentenceUnderstanding import readback_scores
    codes = torch.tensor([[[1.,0.,9.],[0.,1.,1.]]])
    leaf = torch.tensor([[2.,0.,1.]])
    scores = readback_scores(leaf,codes,torch.ones(1,2),percept_width=2)
    torch.testing.assert_close(scores,torch.tensor([[1.,0.]]))
    torch.testing.assert_close(readback_scores(-leaf,codes,torch.ones(1,2),percept_width=2),scores)
