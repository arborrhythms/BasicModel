"""Hash seeds cannot choose symbolic host registration order."""
import json
import os
from pathlib import Path


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
