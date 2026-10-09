"""One unforced initialization probe on eight real question sentences."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
import torch
from math_chain_corpus import ChainDocument, MathChainCorpus
from math_chain_ordinary import ordinary_model, train_documents

torch.set_num_threads(1)
folder = HERE/sys.argv[1]
folder.mkdir(exist_ok=False)
model = ordinary_model(folder)
corpus = MathChainCorpus()
docs = []
for index in range(8):
    source = corpus.problem((index, 0), split='development', training=True)
    docs.append(ChainDocument(source.key, (source.sentences[source.question],), question=0))
try:
    rows = train_documents(model, docs, folder)
    report = dict(kind='development', seed=None, questions=len(rows),
        committed_open=sum(bool(row['open']) for row in rows),
        episodes=sum(row['episode'] for row in rows),
        departures=[row['departure'] for row in rows],
        declaration='A single diagnostic construction, not a measurement attempt or a learning claim.')
    (folder/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
finally:
    model.End()
