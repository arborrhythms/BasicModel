"""Small normal model shells for checked-boundary tests (no alternate executor)."""
from types import SimpleNamespace
from Language import Grammar
from Layers import WhatInteractionMemory, TernaryTruthStore
from Models import BasicModel
from Queries import GrammaticalThoughtRegistry


def model_for(cs, store=None):
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    registry = GrammaticalThoughtRegistry.install(cs, grammar)

    model = BasicModel()
    model.spaces = []
    model.eval()
    model.reasoning_iterations = model.thinking_budget = 128
    object.__setattr__(model, 'conceptualSpace', cs)
    object.__setattr__(model, 'grammatical_thoughts', registry)
    object.__setattr__(model, 'symbolSpace', SimpleNamespace(
        grammatical_thoughts=registry, ltm_store=store,
        what_memory=WhatInteractionMemory(batch=1, capacity=128, detach_mode='episode')))
    return model


def thought_config(tmp_path):
    import xml.etree.ElementTree as ET
    from test_ltm_consolidation import _STATEFUL_CONFIG
    document = ET.parse(_STATEFUL_CONFIG)
    grammar = document.find('SymbolSpace/language/grammar')
    grammar.clear()
    grammar.text = 'complete.grammar'
    document.find('ConceptualSpace/nVectors').text = '64'
    ET.SubElement(document.find('architecture'), 'conceptIndexRead').text = 'true'
    path = tmp_path / 'thought-model.xml'
    document.write(path)
    return str(path)
