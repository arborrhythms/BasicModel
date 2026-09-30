"""One unselected first forward, recording the clause that cannot be written."""
import dataclasses
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent / 'mm-admission'
HERE.mkdir(exist_ok=True)
os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='1',
                  BASIC_AUTOLOAD='false')
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
import torch
import ClauseJournal
from ClauseRow import ClauseRows
from test_mm_xor import _fresh_model
from util import init_device

def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value

readings = []
finish = ClauseJournal.finish_clause
def observe(language, program, **kwargs):
    clause = finish(language, program, **kwargs)
    readings.append(dict(program=plain(program), clause=plain(clause),
        registry=kwargs.get('registry') is not None,
        binary=[r.method_name for r in language._compose_binary_rules],
        unary=[r.method_name for r in language._compose_unary_rules]))
    return clause
ClauseJournal.finish_clause = observe
write = ClauseRows.write_clause
def admit(store, clause, **kwargs):
    try:
        return write(store, clause, **kwargs)
    except Exception as error:
        (HERE/'failure.json').write_text(json.dumps(dict(error=str(error),
            clause=plain(clause), readings=readings), indent=2)+'\n')
        raise
ClauseRows.write_clause = admit

init_device('cpu')
# The gate's class setup builds MM_xor before the grammar-specific test.
setup, _, _ = _fresh_model()
model, _, _ = _fresh_model(str(ROOT/'data/MM_grammar.xml'))
loader = model.inputSpace.data.data_loader(split='train', num_streams=4)
items, _ = next(iter(loader))
try:
    model.forward(model.inputSpace.prepInput(items))
finally:
    (HERE/'readings.json').write_text(json.dumps(readings, indent=2)+'\n')
    model.End()
