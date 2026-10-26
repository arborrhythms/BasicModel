"""Pre-layout estimates are rejected; current estimates authenticate and roundtrip."""
import copy
import pytest
import torch
from Layers import LTM0
from Meaning import ConceptualMeaning
from meaning_fixtures import symbol_prediction, estimate_meaning


def current_pair():
    store = LTM0(4, capacity=8)
    prior = symbol_prediction(torch.eye(4), torch.full((4,2), .5))
    actual = ConceptualMeaning.from_description(torch.eye(4)[:3].mean(0))
    e,o = store.append_expectation_pair(estimate_meaning(prior),actual,
        presence_logits=torch.zeros(3))
    store.surprise[o] = .123
    sidecar = store.semantic_extras()
    return store,e,o,sidecar


def test_pre_layout_estimate_rejected_and_current_roundtrips_once():
    old,e,o,sidecar = current_pair()
    new = LTM0(4,capacity=8)
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    historical = copy.deepcopy(sidecar)
    historical['version'] = 5
    with pytest.raises(ValueError,match='semantic checkpoint version'):
        new.load_semantic_extras(historical)
    new.load_semantic_extras(copy.deepcopy(sidecar))
    assert new.semantic_extras()['version'] == 10
    assert 'kind_logit' not in new.expectation_of(e)
    bank = dict(new.meaning_of(e).bindings)['_expected_symbols']
    assert [(identity,order) for identity,order,*_ in bank] == [(i,0) for i in range(4)]
    assert len(bank) == 4  # the complete native-symbol prior survives
    torch.testing.assert_close(new.meaning_of(o).roles, old.meaning_of(o).roles)
    assert new.row(o)['surprise'] == pytest.approx(.123)
    pair = new.expectation_pair(o)
    again = LTM0(4,capacity=8)
    again.load_state_dict(copy.deepcopy(new.state_dict()))
    again.load_semantic_extras(copy.deepcopy(new.semantic_extras()))
    torch.testing.assert_close(again.expectation_pair(o)['residual'],pair['residual'])
    assert again.semantic_extras() == new.semantic_extras()


def test_authentication_does_not_bless_corrupt_current_estimates():
    old,e,o,sidecar = current_pair()
    sidecar['records'][e]['expectation']['presence_logits'] = (9.,9.,9.)
    new = LTM0(4,capacity=8)
    new.load_state_dict(old.state_dict())
    with pytest.raises(ValueError,match='fingerprint'):
        new.load_semantic_extras(sidecar)
