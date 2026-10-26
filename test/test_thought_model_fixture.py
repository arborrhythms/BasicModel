"""Small normal model shells for checked-boundary tests (no alternate executor)."""
from types import SimpleNamespace
from Language import Grammar, OperationSelectionLayer
from Layers import WhatInteractionMemory, LTM0
from Models import BasicModel
from Queries import GrammaticalThoughtRegistry


def force_requested_thought(model):
    """Replay the requested opening operation through the ordinary chooser."""
    import torch
    choose_action = model._choose_selected_thought_action
    def choose(root, active, actions, **kwargs):
        state = model._what_memory().thought_state(kwargs.get('row', 0))
        if state is not None and not state.finished and state.work_spent and None in actions:
            selected = None
        else:
            try:
                name = model.grammatical_thoughts.signature_for(active, verify_reference=False).operation.semantic_id
            except (ValueError, TypeError):
                selected = actions[0]
            else:
                selected = next((item for item in actions if item is not None and item.semantic_id == name), actions[0])
        index = next(i for i, item in enumerate(actions) if item is selected)
        chooser = model._selected_thought_chooser(active)
        select = chooser.select_logits
        def replay(logits, **options):
            if options.get('masked_action') is None:
                options['replay_action'] = torch.tensor([index], device=logits.device)
            return select(logits, **options)
        chooser.select_logits = replay
        try:
            return choose_action(root, active, actions, **kwargs)
        finally:
            chooser.select_logits = select
    model._choose_selected_thought_action=choose
    return model


def model_for(cs, store=None):
    import torch
    with torch.no_grad():
        cs.similarity_codebook.W.sigmoid_()
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
    if store is not None:
        from ClauseRow import attach_clause_index
        attach_clause_index(model, store, cs)
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
