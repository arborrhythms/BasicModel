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

