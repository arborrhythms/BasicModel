"""Read §11 artifacts only; no model construction, RNG or training."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
OLD = HERE / 'review11-measurements'
with zipfile.ZipFile(HERE / 'review11-source/source.zip') as archive:
    xml = archive.read('data/XOR_grammar.xml')
    # The frozen Grammar._fill_section reads space-role sections before direct
    # field rules. This fixture has one such section, <symbols>, in this order.
    rule_text = [rule.text.strip() for rule in ET.fromstring(xml).findall(
        './SymbolSpace/language/grammar/compose/symbols/rule')]
    assert rule_text == ['S = not.forward(S)', 'S = conjunction.forward(S, S)',
                         'S = disjunction.forward(S, S)']
    rules = {i: text.split('=')[1].strip().split('.')[0] for i, text in enumerate(rule_text)}
    kernel = archive.read('bin/Layers.py').decode()
    assert '"""Bundle two concepts by their arithmetic mean."""\n        return (x + y) * .5' in kernel

summary = json.loads((OLD / 'summary.json').read_text())
rows = []
for row in summary['xor']:
    folder = OLD / f'xor-{row["run"]:02}'
    events_file = folder / 'ownership/events.jsonl'
    final = None
    if events_file.exists():
        events = [json.loads(line) for line in events_file.read_text().splitlines()]
        evaluated = [event for event in events if event['kind'] == 'decoder' and not event['train']]
        final = evaluated[-1]
        assert final['trial'] == 'exploit'
    rows.append(dict(run=row['run'], mse=row['mse'], band=row['band'],
        final_greedy_compose=None if final is None else [
            [dict(rule_id=i, rule_name=rules[i], arity=a) for i, a in zip(ids, arities)]
            for ids, arities in zip(final['compose_rules'], final['compose_arities'])],
        evidence=None if final is None else str(events_file.relative_to(HERE)),
        status='not recorded in saved observations or logs' if final is None else 'observed',
        artifact_sha256={str(file.relative_to(HERE)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in [folder/'observations.jsonl', folder/'run.log', folder/'reports.jsonl']
            + ([events_file] if final is not None else [])}))
result = dict(source='saved §11 measurements; no new forward or training',
    rule_id_map=rules, frozen_xml_sha256=hashlib.sha256(xml).hexdigest(),
    rows=rows, recorded_runs=[row['run'] for row in rows if row['final_greedy_compose'] is not None],
    conclusion='Run 10 supports the mean hypothesis: all four final greedy compositions used '
        'rule 2, then mean disjunction, at MSE .250079. The other nine runs lack final compose '
        'observations, so the claim about all five quarter-band runs is unconfirmed. Run 3 is '
        '.232321, in the unchanged quarter band, not exactly at the additive floor.')
with (HERE / 'review12-retrospective.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(dict(recorded_runs=result['recorded_runs'], conclusion=result['conclusion'])))
