"""Hash seeds cannot choose collision work or symbolic host registration order."""
import inspect
import json
import os
from pathlib import Path
import sys


def test_collision_queue_follows_admission_order(monkeypatch):
    from test_operators_round3a import identity_bank
    from WordIdentity import WordIdentity

    bank = identity_bank()
    source, first = inspect.getsourcelines(WordIdentity.admit)
    queue_line = first + next(i for i, line in enumerate(source)
                              if line.strip() == 'key = self.key(self.form(word))')
    reads, mints = [], []
    form, mint = bank.form, bank._mint

    def observed_form(raw):
        caller = sys._getframe(1)
        if caller.f_code is WordIdentity.admit.__code__ and caller.f_lineno == queue_line:
            reads.append(bytes(raw).decode('ascii'))
        return form(raw)

    def observed_mint(left, right):
        mints.append([left.decode('ascii'), right.decode('ascii')])
        return mint(left, right)

    monkeypatch.setattr(bank, 'form', observed_form)
    monkeypatch.setattr(bank, '_mint', observed_mint)
    bank.admit(b'calaba')
    bank.admit(b'cabala')
    print('hash-order-certificate ' + json.dumps(dict(
        kind='collision', pythonhashseed=os.environ.get('PYTHONHASHSEED'),
        admission=[word.decode('ascii') for word in bank.words],
        queue_reads=reads, mint_pairs=mints, mint_records=bank.mints)))
    # Pending is a stack: append the touched words in admission order,
    # then process the most recently admitted first.
    assert reads == ['calaba', 'cabala', 'cabala', 'calaba']
    assert mints == [['calaba', 'cabala']]
    assert bank.mints[0]['atoms'] == ['cal@1', 'cab@1']
    assert all(bank.read(bank.form(word)) == word for word in bank.words)


def test_symbolic_hosts_follow_grammar_declaration_order(tmp_path):
    from test_mm_xor import _fresh_model
    from Language import TheGrammar

    # Native construction, with several generic symbolic methods and a
    # repeated declaration. Pre-wired substrate layers keep their owners.
    template = Path(__file__).resolve().parents[1] / 'data/XOR_grammar.xml'
    text = template.read_text().replace(
        '<rule>S = conjunction.forward(S, S)</rule>',
        '<rule>S = sum.forward(S, S)</rule>\n'
        '            <rule>S = equal.forward(S, S)</rule>\n'
        '            <rule>S = conjunction.forward(S, S)</rule>\n'
        '            <rule>S = sum.forward(S, S)</rule>', 1)
    config = tmp_path / 'host-order.xml'
    config.write_text(text)
    model, _, _ = _fresh_model(str(config))
    try:
        owner = model.symbolSpace.subspace
        syntactic = owner.syntacticLayer
        declared = list(dict.fromkeys(rule.method_name for rule in TheGrammar.rules
            if rule.space_role == 'SS' and rule.method_name in
            ('sum', 'equal', 'conjunction', 'disjunction')))
        modules = [name for layer in syntactic.layers
                   for name, value in syntactic._by_name.items()
                   if value is layer and name in declared]
        registry = [name for role, name in owner._host_layer_registry
                    if role == 'SS' and name in declared]
        print('hash-order-certificate ' + json.dumps(dict(
            kind='symbolic_hosts', pythonhashseed=os.environ.get('PYTHONHASHSEED'),
            declared=declared, module_order=modules, registry_order=registry,
            all_module_names=list(syntactic._by_name),
            all_registry_keys=list(owner._host_layer_registry))))
        assert len(declared) == 4
        assert modules == declared
        assert registry == declared
        assert all(owner._host_layer_registry[('SS', name)] is syntactic._by_name[name]
                   for name in declared)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
