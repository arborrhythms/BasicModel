"""A legal unary preference must still fit the row by its closing."""
import torch

def test_shared_stack_contract_allows_unary_until_the_deadline(tmp_path, monkeypatch):
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from test_mm_xor import _fresh_model
    import test_subspace_what_stm_contract as contract
    # NOT became field-only in §3.1. The vector nominal marker preserves
    # these operands while testing the same unary deadline at the same size.
    grammar = tmp_path / 'deadline.grammar'
    grammar.write_text(Path('data/xor.grammar').read_text().replace('not_', 'generic_')
                       .replace('not.forward', 'generic.forward').replace('not.reverse', 'generic.reverse'))
    config = ET.parse('data/MM_text.xml')
    config.getroot().find('SymbolSpace/language/grammar').text = str(grammar)
    path = tmp_path / 'deadline.xml'
    config.write(path)
    model, _, _ = _fresh_model(str(path))
    assert [rule.method_name for rule in model.languageSpace._compose_unary_rules] == ['generic']

    chooser = model.languageSpace._tree_layer(2).chooser
    score = chooser.score_unary

    def prefer_unary(*args, **kwargs):
        stop, unary = score(*args, **kwargs)
        return stop, torch.full_like(unary, 1e6)

    monkeypatch.setattr(chooser, 'score_unary', prefer_unary)
    choose = model.languageSpace.choose_operation
    choices = []
    def record(*args, **kwargs):
        choice = choose(*args, **kwargs)
        if 'rounds_left' in kwargs:
            choices.append(choice)
        return choice
    monkeypatch.setattr(model.languageSpace, 'choose_operation', record)
    contract.test_shared_operation_writes_the_caller_owned_stack(model)
    assert len(choices) == 4
    assert [int(choice.kind) for choice in choices] == [2, 2, 2, 1], [(choice.kind, choice.applied, choice.stack_arity) for choice in choices]
