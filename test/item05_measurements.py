"""One-attempt, unseeded grade-zero and whole-sentence signal dry runs."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
import numpy as np
import torch


def model_from(config):
    import Language, util
    from data import TheData
    from Models import BaseModel
    util.init_device('cpu')
    util.TheCompileBackend = 'none'
    TheData.processLM({split: dict(text=['placeholder'], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    model = BaseModel.from_config(str(config), data=TheData)[0]
    model._tensor_peer_while_eager = True
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('math', 'game'))
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.output
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.save(dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate()), folder/'initial-rng.pt')
    from bounded_tests import source_snapshot
    snapshot = source_snapshot(ROOT)
    (folder/'source.json').write_text(json.dumps(snapshot, sort_keys=True))
    report = dict(kind=args.kind, condition=args.condition, seed_policy='fresh unseeded; one attempt',
        source_sha256=hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        records=[])
    model = None
    started = time.perf_counter()
    try:
        from math_chain_corpus import ChainDocument, flatten
        from MathChainTraining import present
        if args.kind == 'math':
            if args.condition not in ('answer_expectation', 'expectation_only', 'zero_budget'):
                raise ValueError('unknown math condition')
            tree = ET.parse(ROOT/'data/MM_math_chain.xml')
            if args.condition == 'zero_budget':
                tree.find('./architecture/attentionBudget').text = '0'
            documents = [ChainDocument(**json.loads(line)['document'])
                         for line in (ROOT/'data/math/grade-00.jsonl').read_text().splitlines()]
            report.update(grade=0, epochs=1, documents=len(documents),
                          sentences=sum(len(doc.sentences) for doc in documents), batch_size=8,
                          attention_budget=0 if args.condition == 'zero_budget' else 32)
            config = folder/'model.xml'; tree.write(config)
            model = model_from(config)
            data = model.inputSpace.data
            texts, targets, addresses = flatten(documents, supplied=args.condition != 'expectation_only')
            data.train_input, data.train_output = texts, [torch.zeros(1) for _ in texts]
            data.text_answers = {'train': targets}
            data.has_supervised_outputs = any(target is not None for target in targets)
            data.source_addresses['train'] = [dict(address, split='train') for address in addresses]
            def observed(model, split, rows, result):
                report['records'].append(dict(rows=rows, reconstruction=float(result.lossIn.detach()),
                    answer=float(result.lossOut.detach())))
            report['presentation'] = present(model, split='train', optimizer=model.getOptimizer(lr=.001),
                                             batch_size=8, after_batch=observed)
        else:
            if args.condition not in ('future', 'supplied'):
                raise ValueError('unknown game signal')
            tree = ET.parse(ROOT/'data/MM_corpus_lessons.xml')
            config=folder/'model.xml'; tree.write(config)
            model=model_from(config)
            data=model.inputSpace.data
            row=next(row for row in map(json.loads,
                (ROOT/'data/games/not/clause.jsonl').read_text().splitlines())
                if row['split']=='train')
            source, target=row['sources'][0],row['target']
            report.update(key=row['key'], source=source, target=target, updates=16, learning_rate=.001)
            from corpus_lessons import stage_documents
            from What import What, WhatQuestion
            doc=ChainDocument('signal', (source, target) if args.condition=='future' else (source,),
                              question=None if args.condition=='future' else 0,
                              answer=None if args.condition=='future' else target)
            stage_documents(data, dict(documents=[dict(document='signal',records=[dataclasses.asdict(doc)])]))
            optimizer=model.getOptimizer(lr=.001)
            for update in range(16):
                model.begin_presentation()
                questions=(WhatQuestion.future(0,1),) if args.condition=='future' else (What.supervised(0),)
                batch=(model.inputSpace.prepInput([source]),model.outputSpace.prepOutput([torch.zeros(1)]))
                result,_=model.runBatch(train=True,batchNum=update,batchSize=1,split='train',optimizer=optimizer,
                    batch_override=batch,questions=questions,source_rows=[0])
                report['records'].append(dict(update=update+1,reconstruction=float(result.lossIn.detach()),
                    answer=float(result.lossOut.detach()),
                    produced=list(getattr(getattr(model,'_last_answer_construction',None),'texts',()) or ()),
                    losses=model.errors.breakdown()))
                model.flush_word_buffers();model.dispatch_per_row_reset([True]);model.dispatch_soft_reset();model.post_tick_compact()
        report['status']='complete'
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    finally:
        if model is not None:model.End()
        report['elapsed_seconds']=time.perf_counter()-started
        (folder/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({key:report[key] for key in ('kind','condition','status','elapsed_seconds')}))
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
