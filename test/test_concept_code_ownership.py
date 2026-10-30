"""Host admission and reads preserve reconstruction-owned code parameters."""
import torch
from torch import nn
from Spaces import Codebook, ConceptualSpace


def test_known_concept_read_preserves_the_parameter_magnitude():
    cs = ConceptualSpace.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cb = Codebook()
    cb.W = nn.Parameter(torch.tensor([[2., -3.], [0., 4.]]))
    cs.similarity_codebook = cb
    object.__setattr__(cs, 'concept_codebook_row_of_percept', lambda pid: {3: 0}.get(pid))
    content, mask = cs.concept_row_content(torch.tensor([3, 99]))
    assert mask.tolist() == [True, False]
    torch.testing.assert_close(content[0], cb.W[0])
    assert not content[1].any()
    content.sum().backward()
    torch.testing.assert_close(cb.W.grad, torch.tensor([[1., 1.], [0., 0.]]))


def test_promoting_a_candidate_initializes_only_its_new_row():
    from test_attention_promotion import _fixture, _close
    from FamiliarTier import observe_candidate, centroid
    cs, rows = _fixture()
    support = {row: 1. for _, row in rows[:2]}
    for _ in range(3):
        key = observe_candidate(cs, 1, support)
        _close(cs)
    W = cs.similarity_codebook.W
    before = W.detach().clone()
    expected = centroid(cs, support)
    discovered = cs.mtm.promote_candidates(cs)
    assert len(discovered) == 1
    row = cs._csw_row_of(discovered[0])
    keep = torch.arange(len(W)) != row
    torch.testing.assert_close(W[keep], before[keep], rtol=0, atol=0)
    torch.testing.assert_close(W[row], expected)



def test_admitting_a_phrase_does_not_compose_or_normalize_its_code():
    from test_attention_promotion import _fixture
    cs, rows = _fixture()
    members = tuple(cid for cid, _ in rows[:2])
    cs.admission_count = 2
    counts = cs.utility_counts()
    counts['n'] = 100
    counts['n_c'].update({cid: 1 for cid in members})
    W = cs.similarity_codebook.getW()
    before = W.detach().clone()
    cs.__dict__['_chunk_proposals'] = [(0, members, 0)] * 2
    cs._commit_chunk_admissions()
    assert members in cs.__dict__['_chunk_admitted']
    assert members in cs.__dict__['_chunk_rows']
    torch.testing.assert_close(W, before, rtol=0, atol=0)


def test_concept_quantization_cannot_refresh_the_reconstruction_dictionary():
    from pathlib import Path
    import Models
    from test_mm_xor import _fresh_model
    # Field dictionaries still have direct parameters. Serial word views are
    # covered by test_derived_concept_codes, including their actual owner.
    model, _, _ = _fresh_model(str(Path(Models.__file__).resolve().parents[1]/'data/MM_xor.xml'))
    seen = set()
    for cs in model.conceptualSpaces:
        cb = cs.similarity_codebook
        if id(cb) in seen:
            continue
        seen.add(id(cb))
        assert isinstance(cb.W, nn.Parameter) and cb.W.requires_grad
        assert cb.vq is not None and not cb.vq.ema_update
        W, counts = cb.W.detach().clone(), cb.vq.cluster_size.clone()
        cb.train()
        cb.quantize(torch.randn(2, 3, cb.W.shape[-1]))
        torch.testing.assert_close(cb.W, W, rtol=0, atol=0)
        torch.testing.assert_close(cb.vq.cluster_size, counts, rtol=0, atol=0)
