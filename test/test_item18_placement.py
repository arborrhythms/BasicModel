"""Certificate 91: placement reads raw folded evidence, within each order."""
import torch
import pytest

pytestmark = pytest.mark.assessment_rung(1)


def test_91_context_includes_the_present_tokens_types_before_likeness():
    from test_priming_diffusion import class_fixture
    cs, codes, (a, b, parent), _ = class_fixture()
    cs._observed_knowing.zero_()
    cs._observed_knowing[0, a, 0] = 1.
    context = cs.placement_context(0)
    assert context[a] == 1.
    assert context[parent] > 0.
    # Neither expectation nor a changed placement can teach its own place.
    cs._expected_knowing.fill_(1.)
    codes.placement.copy_(torch.flip(codes.placement, (-1,)))
    torch.testing.assert_close(cs.placement_context(0), context, rtol=0, atol=0)


def test_91_observed_formal_membership_is_also_a_field_definition():
    from argument_fixtures import argument_inventory
    cs, speech, words = argument_inventory(('change_argin1',))
    word = words[0]
    speech.add(word, 'change_argin1')
    cs._observed_knowing[0, cs._csw_row_of(word), 0] = 1.
    context = cs.placement_context(0)
    assert context[cs._csw_row_of(speech.concepts['change_argin1'])] == 1.


def test_91_one_step_uses_only_context_and_negatives_of_the_rows_order():
    from test_priming_diffusion import class_fixture
    cs, codes, (a, b, parent), _ = class_fixture()
    codes.negative_count = 0
    codes.placement[a] = torch.tensor([1., 0.])
    codes.placement[parent] = torch.tensor([0., 1.])
    codes.placement_witnesses[parent] = 1
    before = codes.placement.clone()
    context = torch.zeros(cs.nVectors)
    context[parent] = 1.
    codes.place_rows([a], context, b'certificate-91')
    torch.testing.assert_close(codes.placement, before, rtol=0, atol=0)
    codes.negative_count = cs.nVectors
    negatives = codes._negatives(a, torch.zeros_like(context),
                                codes.placement_witnesses > 0, b'certificate-91')
    assert parent not in negatives.tolist()


def test_91_classes_have_no_form_and_are_placed_by_the_same_context():
    from test_priming_diffusion import class_fixture
    cs, codes, (a, b, parent), _ = class_fixture()
    assert not hasattr(codes, 'placement_derived')
    assert not hasattr(codes, 'derive_class_placements')
    assert not cs.similarity_codebook.lookup_rows(parent)[:codes.percept_width].any()
    codes.place_rows([parent], torch.zeros(cs.nVectors), b'certificate-91-class')
    assert codes.placement_witnesses[parent] == 1
    assert codes.placement[parent].norm() > 0.
    assert not cs.similarity_codebook.lookup_rows(parent)[:codes.percept_width].any()
