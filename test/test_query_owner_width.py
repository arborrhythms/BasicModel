"""Declared query types must respect an available conceptual owner's width."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Meaning import ConceptualMeaning
from Queries import BUILTIN_QUERIES, QueryContext
from reasoning import TruthGroundedReasoner
from test_cs_symbol_table import _cs


@pytest.mark.parametrize('name', ['isEqual', 'what', 'exist'])
def test_checked_call_rejects_foreign_width_before_any_executor(name):
    cs = _cs()
    width = int(cs.outputShape[-1])
    calls = []
    signature = replace(BUILTIN_QUERIES[name],
                        executor=lambda *args: calls.append(args) or {})
    context = QueryContext(TruthGroundedReasoner(
        model=SimpleNamespace(conceptualSpace=cs)))
    value = torch.ones(width + 1)
    arguments = ((value, value) if name == 'isEqual' else
                 (ConceptualMeaning.from_description(value),))
    with pytest.raises(ValueError, match='conceptual width'):
        signature.invoke(context, *arguments)
    assert calls == []
