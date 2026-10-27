"""Native form and detached meaning faces, paired by the shared index."""
from order_fixtures import force_order
from types import SimpleNamespace
import torch
from torch import nn


def fixture(*, meaning_pairs=1, batch_size=1, property_width=0):
    from test_cs_sparse_weights import _cs
    from Spaces import Codebook, _concept_alloc_of
    cs = _cs(nS=32, order=0, batch_size=batch_size)
    ps = Codebook()
    width = 3 + property_width
    ps.W = nn.Parameter(torch.zeros(16, width))
    with torch.no_grad():
        ps.W[4, :3] = torch.tensor([.2, .6, .8])
        ps.W[7, :3] = torch.tensor([.8, .2, .4])
    object.__setattr__(cs, '_model', SimpleNamespace(perceptualSpace=SimpleNamespace(
        subspace=SimpleNamespace(what=ps), nDim=width)))
    cb = Codebook()
    cb.W = nn.Parameter(torch.zeros(32, width+2*meaning_pairs))
    cs.similarity_codebook = cb
    cb.enable_derived_codes(cs, percept_width=width, identity_width=3,
                            property_width=property_width, content_width=width+2*meaning_pairs)
    cs._online_learning_frozen = False
    return cs, cb, ps, _concept_alloc_of(cs)


def word(cs, parts):
    cid = cs.new_concept()
    force_order(cs, cid, 0)
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


def test_close_placement_is_confined_to_detached_context_coordinates():
    from Layers import LTM0
    from Meaning import ConceptualMeaning
    cs, cb, ps, _ = fixture()
    a, b = word(cs, [(4, 1.)]), word(cs, [(7, 1.)])
    store = LTM0(5, capacity=8)
    cs.symbolSpace = SimpleNamespace(ltm_store=store)
    part = store.parts.add(cs.concept_id_at_row(a), cs.concept_id_at_row(b),
                           symbol=cs.new_concept())
    root = torch.tensor([9.,8.,7.,.2,.4], requires_grad=True)
    roles = torch.stack((root, root*0, root*0))
    row = store.append_meaning(ConceptualMeaning(roles, torch.tensor([True,False,False]),
         sentence_kind='idea', role_refs=(('sym', part), None, None)),
        evidence=(0.,0.))
    reading = SimpleNamespace(word_rows=torch.tensor([a,b]), concept_ids=None)
    cs._sentence_seen_rows[0, :2] = torch.tensor([a,b])
    cs.prime_seen(torch.tensor([[a,b]]))
    cb.mereology.observe_close(reading)
    codes = cb.lookup_rows(torch.tensor([a,b]))
    torch.testing.assert_close(codes[:,:3], ps.W[[4,7]])
    torch.testing.assert_close(codes[:,3:], torch.tensor([[1/5,0.],[1/5,0.]]))
    assert not codes.requires_grad
    assert root.grad is None and not store.slots.requires_grad
    newer = store.append_meaning(ConceptualMeaning(roles*2, torch.tensor([True,False,False]),
         sentence_kind='idea', role_refs=(('sym', part), None, None)),
        evidence=(0.,0.))
    cs.prime_seen(torch.tensor([[a,b]]))
    cb.mereology.observe_close(reading)
    torch.testing.assert_close(cb.lookup_rows(a)[:3], ps.W[4])
    torch.testing.assert_close(cb.lookup_rows(a)[3:], torch.tensor([2/6,0.]))
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


def test_actual_serial_code_uses_two_form_subspaces_and_a_separate_bipolar_context():
    from test_mm_xor import _fresh_model
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    model.forward(model.inputSpace.prepInput(['hello world', 'hello there']))
    cb = model._concept_owner().similarity_codebook
    assert cb.mereology.identity_width == 96
    assert cb.mereology.property_width == 256
    assert cb.mereology.percept_width == 352
    assert cb.mereology.context_width == 128
    assert not isinstance(cb.W, nn.Parameter) and list(cb.parameters()) == []
    rows = model.inputSpace._ar_grammar_object_rows
    atoms = model.inputSpace._ar_grammar_object_atoms
    torch.testing.assert_close(atoms[rows>=0][..., :352], cb.lookup_rows(rows[rows>=0])[..., :352])
    assert not atoms[rows>=0][...,360:].any()
    assert cb.lookup_rows(rows[rows>=0])[...,360:].any()
    assert bool(cb.mereology.placement_witnesses[rows[rows>=0]].gt(0).all())
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
