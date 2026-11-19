"""The old pass selector is replaced by a shared typed bracket allowance."""
from pathlib import Path
import pytest
import torch
from util import XMLConfig


@pytest.mark.parametrize('value',['all','off','1, 3','0','4','-1','1,no'])
def test_old_selector_is_rejected(value):
    with pytest.raises(ValueError,match='subsymbolicLoop'):
        XMLConfig._apply_legacy_renames({'architecture':{'subsymbolicLoop':value}},'probe.xml')


def test_default_and_overlay_use_one_attention_budget(tmp_path):
    config=XMLConfig();config.load(str(Path(__file__).resolve().parents[1]/'data/model.xml'))
    assert int(config.get('architecture.attentionBudget'))>0
    path=tmp_path/'attention.xml';path.write_text('<model><architecture><attentionBudget>7</attentionBudget></architecture></model>')
    config.overlay(str(path));assert config.get('architecture.attentionBudget')==7


def test_overlay_rejects_retired_spelling(tmp_path):
    path=tmp_path/'retired.xml';path.write_text('<model><architecture><subsymbolicNoop>1</subsymbolicNoop></architecture></model>')
    with pytest.raises(ValueError,match='subsymbolicNoop'):XMLConfig().overlay(str(path))


def test_selected_typed_scope_changes_the_attentive_extent(tmp_path):
    from test_grounded_xor import grounded_model
    model,inputs=grounded_model(tmp_path)
    cs,ws=model.conceptualSpaces[0],model.wholeSpace
    prior=ws.subspace.what.primitive_properties
    prior.teach(9,[48,49],[0.,1.]);prior.teach(10,[48,49],[1.,0.])
    from order_fixtures import force_order
    force_order(cs,10001,0)
    cs._csw_concept_row(0,10001)
    from nomination_fixtures import nominated_forms
    nominated_forms(model, ('0', '1'))
    cs.add_concept_feature(0,'ws',9,1.);cs.add_concept_feature(0,'ws',10,-1.)
    model.forward(inputs[2:3]);model.conceptualSpace._passback_scope_where=torch.tensor([[0.,1.]])
    ids,spans=model._subsymbolic_percepts[:2]
    # A property seen both true and false is read at its basic percepts
    # (sigma-pi §19.68, rule 6): the percept of its member (whole 9) and that
    # of its Not member (whole 10), each on its two lanes. Over "10" each
    # part is present and each property is witnessed both ways: both.
    def lanes(whole):
        field = cs._percept_field
        rows = [('bp', r) for r in cs.basic_percepts.carrying('ws', (whole,)) if ('bp', r) in field.keys]
        values = [field.values[field.keys.index(key), 0, 0] for key in rows]
        return torch.stack(values).amax(0) if values else torch.zeros(2)
    model._read_subsymbolic_field(ids,spans,1)
    torch.testing.assert_close(lanes(9),torch.tensor([1.,1.]))
    torch.testing.assert_close(lanes(10),torch.tensor([1.,1.]))
    model.conceptualSpace._passback_scope_where=torch.tensor([[1.,2.]])/inputs.shape[-1]
    model._read_subsymbolic_field(ids,spans,1)
    torch.testing.assert_close(lanes(9),torch.tensor([0.,0.]))
    torch.testing.assert_close(lanes(10),torch.tensor([1.,0.]))
