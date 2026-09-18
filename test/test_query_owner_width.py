"""Declared thought types respect the capability owner's full width."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Meaning import ConceptualMeaning
from Queries import THOUGHT_EXECUTORS, ThoughtSignature
from test_cs_symbol_table import _cs
from test_query_vp_boundaries import _context


@pytest.mark.parametrize(('name', 'roles'), [
    ('equal', ('I1', 'I2')), ('what', ('I1',)), ('exist', ('I1',)),
])
def test_checked_call_rejects_foreign_width_before_any_executor(name, roles):
    cs = _cs()
    width = int(cs.outputShape[-1])
    calls = []
    descriptor = replace(THOUGHT_EXECUTORS[name],
                         executor=lambda *args: calls.append(args) or {})
    signature = ThoughtSignature(
        SimpleNamespace(semantic_id=name, operand_roles=roles), descriptor, roles)
    context = _context(cs)
    value = torch.ones(width + 1)
    arguments = ((value, value) if name == 'equal' else
                 (ConceptualMeaning.from_description(value),))
    with pytest.raises(ValueError, match='conceptual width'):
        signature.invoke(context, *arguments)
    assert calls == []
