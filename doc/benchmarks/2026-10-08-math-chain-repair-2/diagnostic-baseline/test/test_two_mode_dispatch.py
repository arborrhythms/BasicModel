"""Modes retire: topology has an inventory depth and one bracket allowance."""
from pathlib import Path
import pytest
import torch
from util import XMLConfig


@pytest.mark.parametrize('tag,value',[
    ('symbolicOrder',0),('symbolicOrder',1),('symbolicOrder',2),
    ('symbolicOrder',-1),('symbolicOrder','not_an_int'),
    ('serial',True),('serial',False),('subsymbolicOrder',1)])
def test_old_modes_and_order_budgets_cannot_be_selected(tag,value):
    with pytest.raises(ValueError,match=tag):
        XMLConfig._apply_legacy_renames({'architecture':{tag:value}},'old-mode.xml')


@pytest.mark.parametrize('name,grammatical',[('MM_xor_loopback.xml',True),('MM_xor.xml',False)])
def test_native_carrier_contract_replaces_traversal_switch(name,grammatical,eager_reading):
    from test_mm_xor import _fresh_model,_PROJECT
    model=_fresh_model(str(Path(_PROJECT)/'data'/name))[0]
    assert not hasattr(model,'serial') and not hasattr(model,'symbolicOrder')
    assert not hasattr(model,'subsymbolic_loop') and not hasattr(model,'mode_schedule')
    assert isinstance(model.attention_budget,int) and model.attention_budget>0
    assert model.word_brackets is grammatical
    assert model.inputSpace._per_word_enabled is grammatical
    with torch.no_grad():model(model.inputSpace.prepInput(['hello world']))
    assert not model._open_read.requires_grad
    assert model._attention_words.accepted.sum()==2


def test_all_shipped_configs_reject_legacy_mode_names():
    import xml.etree.ElementTree as E
    root=Path(__file__).resolve().parents[1]/'data'
    for path in root.rglob('*.xml'):
        config=E.parse(path).getroot()
        for name in ('serial','modeSchedule','symbolicOrder','subsymbolicOrder','subsymbolicLoop'):
            assert config.find('.//'+name) is None,(path,name)
