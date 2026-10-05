"""Verify only the declared initialization/capacity changes, without training."""
import os
os.environ['BASICMODEL_DEVICE'] = 'cpu'
import ast
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

H = Path(__file__).resolve().parent
ROOT = H.parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT / 'bin')]
import bounded_tests
import torch
from Layers import RadixLayer

before = json.loads((H / 'review22-before/source.json').read_text())
after = bounded_tests.source_snapshot(ROOT)
changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
assert changed == ['bin/Layers.py', 'data/MM_grammar.xml', 'data/XOR_grammar.xml'], changed
capacities = []
with zipfile.ZipFile(H / 'review22-before/source.zip') as archive:
    old = archive.read('bin/Layers.py').decode()
    new = (ROOT / 'bin/Layers.py').read_text()
    def function(text):
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RadixLayer')
        return tree, next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'insert')
    tree0, fn0 = function(old)
    tree1, fn1 = function(new)
    old_body, new_body = ast.get_source_segment(old, fn0), ast.get_source_segment(new, fn1)
    fn0.body = fn1.body
    assert ast.dump(tree0, include_attributes=False) == ast.dump(tree1, include_attributes=False)
    for name in ('XOR_grammar', 'MM_grammar'):
        old_xml = ET.fromstring(archive.read(f'data/{name}.xml'))
        new_xml = ET.fromstring((ROOT / f'data/{name}.xml').read_text())
        for space in ('InputSpace', 'PartSpace', 'ConceptualSpace', 'WholeSpace'):
            a, b = old_xml.find(f'{space}/nDim'), new_xml.find(f'{space}/nDim')
            assert a.text == '14' and b.text == '22'
            a.text = b.text
        # Fixture comments may alter indentation tails; compare XML values.
        def values(node):
            return node.tag, node.attrib, (node.text or '').strip(), [values(child) for child in node]
        assert values(old_xml) == values(new_xml)
        capacities.append(dict(configuration=name,spaces=['InputSpace','PartSpace','ConceptualSpace','WholeSpace'],
            old_event_width=14,new_event_width=22,address_coordinates=8,old_content_width=6,new_content_width=14,
            conceptual_rows=int(new_xml.find('ConceptualSpace/nVectors').text),
            whole_output_width=new_xml.findtext('WholeSpace/nOutputDim'),other_xml_values_unchanged=True))

checks = []
for width in (6, 14):
    store = RadixLayer(dim=width, initial_cap=8)
    master = store._basis.W
    identity, pointer = id(master), master.data_ptr()
    old_rows = master.detach().clone()
    replay = torch.Generator()
    replay.set_state(torch.random.get_rng_state())
    raw = torch.empty(width).normal_(mean=0., std=1., generator=replay)
    expected = (raw / raw.norm().clamp(min=1e-8)).clamp(0., 1.)
    row = store.insert(b'x')
    assert row == 0 and torch.equal(master[row].detach(), expected)
    assert torch.equal(torch.random.get_rng_state(), replay.get_state())
    assert torch.equal(master[1:].detach(), old_rows[1:])
    assert id(store._basis.W) == identity and store._basis.W.data_ptr() == pointer and master.requires_grad
    state, saved = torch.random.get_rng_state(), master.detach().clone()
    assert store.insert(b'x') == row
    assert torch.equal(torch.random.get_rng_state(), state) and torch.equal(master.detach(), saved)
    explicit = torch.linspace(-2., 2., width)
    other = store.insert(b'y', init_vector=explicit)
    assert torch.equal(master[other].detach(), explicit) and torch.equal(torch.random.get_rng_state(), state)
    checks.append(dict(width=width,exact_l2_then_clamp=True,same_single_random_draw=True,
        parameter_identity_and_ownership_unchanged=True,duplicate_is_noop=True,explicit_initializer_unchanged=True,
        initialized_minimum=float(expected.min()),initialized_maximum=float(expected.max()),
        initialized_l2=float(expected.norm())))

result = dict(changed_runtime_files=changed,changed_function='RadixLayer.insert',old=old_body,new=new_body,
    byte_fallback_unchanged=True,tests_unchanged=True,seeds_bars_budgets_guards_unchanged=True,
    training_runs=0,seed_override=None,capacity_changes=capacities,checks=checks)
(H / 'review22-change-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('old','new')}))
