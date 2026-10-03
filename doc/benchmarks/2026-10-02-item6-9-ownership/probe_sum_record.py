"""Fixed-parameter diagnostic of the additive control's answer evidence."""
import json, os, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
os.environ.update(MODEL_COMPILE='none',BASICMODEL_DEVICE='cpu',BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false')
import torch
from test_mm_xor import _fresh_model
xml=(ROOT/'data/XOR_grammar.xml').read_text()
rules='<rule>S = not.forward(S)</rule>\n            <rule>S = conjunction.forward(S, S)</rule>\n            <rule>S = disjunction.forward(S, S)</rule>'
assert rules in xml
with tempfile.TemporaryDirectory() as directory:
    config=Path(directory)/'sum.xml'
    config.write_text(xml.replace(rules,'<rule>S = sum.forward(S, S)</rule>'))
    model,_,data=_fresh_model(str(config))
    samples=['hello world','hello there','loving world','loving there']
    rows=[]
    try:
        for presentation in range(3):
            with torch.no_grad():
                model(model.inputSpace.prepInput(samples))
            record=model._last_sentence_understanding
            features=record.reader_features(model.answer_record_reader.rule_count)
            contrast=lambda x:x[0]+x[3]-x[1]-x[2]
            rows.append(dict(presentation=presentation,
                root_contrast=contrast(record.root).tolist(),
                feature_contrast=contrast(features).tolist(),
                max_root_contrast=float(contrast(record.root).abs().max()),
                max_feature_contrast=float(contrast(features).abs().max()),
                primed_rows=record.primed.rows.tolist(),weights=record.primed.weights.tolist()))
            model.End();model.symbolSpace.soft_reset()
    finally:
        torch._dynamo.reset()
    path=Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k.startswith('max') or k=='presentation'} for r in rows]))
    assert all(r['max_root_contrast']<=1e-4 and r['max_feature_contrast']<=1e-4 for r in rows)
