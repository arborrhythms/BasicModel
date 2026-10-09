"""Untrained ordinary-sentence construction check, not a learning attempt.

One fresh construction, eight corpus documents, no optimizer and no seed.
Keep the trace even when the untrained grammar leaves every reference bound.
"""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
import torch
torch.set_num_threads(1)
from test_math_chain import build_model
from math_chain_corpus import MathChainCorpus, flatten
from MathChainTraining import present
from ThoughtReferences import open_slots, bindings

config = HERE/'ordinary-construction.xml'
config.write_text((ROOT/'data/MM_math_chain.xml').read_text().replace(
    '<ltmCapacity>1048576</ltmCapacity>', '<ltmCapacity>8192</ltmCapacity>'))
model = build_model(config)
data = model.inputSpace.data
corpus = MathChainCorpus()
documents = tuple(corpus.problem(pair, split='train', training=True)
                  for pair in corpus.train_pairs[:8])
texts, labels, addresses = flatten(documents, supplied=False)
data.train_input, data.train_output = texts, [torch.zeros(1) for _ in texts]
data.text_answers['train'] = labels
data.source_addresses['train'] = [dict(a, split='train') for a in addresses]
observations = []
def observe(model, split, rows, result):
    fields = model._sentence_fields.get(0, ())
    episodes = dict(getattr(model, '_last_closing_thoughts', ()))
    for row, source in enumerate(rows):
        a = addresses[source]
        field = fields[row]
        text = texts[source]
        if a['sentence'] == documents[a['document']].question or text.startswith('the answer'):
            item = dict(source=source, text=text, pair=documents[a['document']].pair,
                episode=row in episodes, open=open_slots(field.meaning),
                query_open=() if field.query is None else open_slots(field.query),
                forward=bindings(field.meaning).get('_forward_references', ()))
            observations.append(item)
            print(json.dumps(item), flush=True)
try:
    print(present(model, split='train', after_batch=observe), flush=True)
finally:
    (HERE/'ordinary-construction.json').write_text(json.dumps(observations, indent=2)+'\n')
    model.End()
