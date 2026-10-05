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
    cs,ws=model.conceptualSpaces[0],model.wholeSpaces[0]
    prior=ws.subspace.what.primitive_properties
    prior.teach(9,[48,49],[0.,1.]);prior.teach(10,[48,49],[1.,0.])
    cs._csw_concept_row(0,10001);cs.add_concept_feature(0,'ws',9,1.);cs.add_concept_feature(0,'ws',10,-1.)
    model.forward(inputs[2:3]);model.conceptualSpace._passback_scope_where=torch.tensor([[0.,1.]])
    ids,spans=model._subsymbolic_percepts[:2]
    model._read_subsymbolic_field(ids,spans,1)
    torch.testing.assert_close(cs._cs_last_a0[0,0,0],torch.tensor([1.,1.]))
    model.conceptualSpace._passback_scope_where=torch.tensor([[1.,2.]])/inputs.shape[-1]
    model._read_subsymbolic_field(ids,spans,1)
    torch.testing.assert_close(cs._cs_last_a0[0,0,0],torch.tensor([0.,1.]))
