"""Small normal model shells for checked-boundary tests (no alternate executor)."""
from types import SimpleNamespace
from Language import Grammar, OperationSelectionLayer
from Layers import WhatInteractionMemory, TernaryTruthStore
from Models import BasicModel
from Queries import GrammaticalThoughtRegistry


def force_requested_thought(model):
    """Supply the requested operation for evidence/boundary mechanism tests."""
    def choose(root, active, actions, **kwargs):
        if None in actions:
            return None
        try:
            name = model.grammatical_thoughts.signature_for(active,verify_reference=False).operation.semantic_id
        except (ValueError,TypeError):
            return actions[0]
        return next((item for item in actions if item.semantic_id==name),actions[0])
    model._choose_selected_thought_action=choose
    return model


def model_for(cs, store=None):
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    registry = GrammaticalThoughtRegistry.install(cs, grammar)

    model = BasicModel()
    model.spaces = []
    model.shared_grammar = OperationSelectionLayer(d_model=cs.outputShape[-1], chooser='mlp')
    object.__setattr__(model, 'languageSpace', SimpleNamespace(
        language_layer=SimpleNamespace(operation_layer=model.shared_grammar)))
    model.attention_budget = 64

    model.eval()
    model.reasoning_iterations = model.attention_budget = 128
    object.__setattr__(model, 'conceptualSpace', cs)
    object.__setattr__(model, 'grammatical_thoughts', registry)
    object.__setattr__(model, 'symbolSpace', SimpleNamespace(
        grammatical_thoughts=registry, ltm_store=store,
        what_memory=WhatInteractionMemory(batch=1, capacity=128, detach_mode='episode')))
    return force_requested_thought(model)


def thought_config(tmp_path):
    import xml.etree.ElementTree as ET
    from test_ltm_consolidation import _ON_CONFIG
    document = ET.parse(_ON_CONFIG)
    ET.SubElement(document.find('architecture'), 'stateless').text = 'false'
    grammar = document.find('SymbolSpace/language/grammar')
    grammar.clear()
    grammar.text = 'complete.grammar'
    document.find('ConceptualSpace/nVectors').text = '64'
    ET.SubElement(document.find('architecture'), 'conceptIndexRead').text = 'true'
    path = tmp_path / 'thought-model.xml'
    document.write(path)
    return str(path)
