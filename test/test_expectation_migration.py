"""Historical retained estimates migrate once; current reads have one schema."""
import copy
import pytest
import torch
from Layers import LTM0
from Meaning import ConceptualMeaning
from meaning_fixtures import symbol_prediction, estimate_meaning


def historical_pair():
    store = LTM0(4, capacity=8)
    prior = symbol_prediction(torch.eye(4), torch.full((4,2), .5))
    actual = ConceptualMeaning(torch.eye(4)[:3], torch.ones(3,dtype=torch.bool))
    e,o = store.append_expectation_pair(estimate_meaning(prior),actual,
        presence_logits=torch.zeros(3))
    identifier = int(store.address_keys[e])
    store._semantic_rows[identifier]['bindings'] = ()
    store._expectation_rows[identifier]['kind_logit'] = .3
    store._update_semantic_fingerprint(e)
    store.surprise[o] = .123
    sidecar = store.semantic_extras(); sidecar['version'] = 5
    return store,e,o,sidecar


def test_role_estimate_migrates_after_authentication_and_roundtrips_once():
    old,e,o,sidecar = historical_pair()
    new = LTM0(4,capacity=8)
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    new.load_semantic_extras(copy.deepcopy(sidecar))
    assert new.semantic_extras()['version'] == 6
    assert 'kind_logit' not in new.expectation_of(e)
    bank = dict(new.meaning_of(e).bindings)['_expected_symbols']
    assert all(identity == order == -1 for identity,order,*_ in bank)
    assert len(bank) == 4  # includes the historical occupied zero-valued role
    torch.testing.assert_close(new.meaning_of(o).roles, old.meaning_of(o).roles)
    assert new.row(o)['surprise'] == pytest.approx(.123)
    pair = new.expectation_pair(o)
    again = LTM0(4,capacity=8)
    again.load_state_dict(copy.deepcopy(new.state_dict()))
    again.load_semantic_extras(copy.deepcopy(new.semantic_extras()))
    torch.testing.assert_close(again.expectation_pair(o)['residual'],pair['residual'])
    assert again.semantic_extras() == new.semantic_extras()


def test_migration_does_not_bless_corrupt_historical_estimates():
    old,e,o,sidecar = historical_pair()
    sidecar['records'][e]['expectation']['presence_logits'] = (9.,9.,9.)
    new = LTM0(4,capacity=8)
    new.load_state_dict(old.state_dict())
    with pytest.raises(ValueError,match='fingerprint'):
        new.load_semantic_extras(sidecar)
