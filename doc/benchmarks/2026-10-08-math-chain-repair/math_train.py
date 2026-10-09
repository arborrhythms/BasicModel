"""One declared native training, with no restarts or outcome-based selection."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import random
import sys
import time
import traceback
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'bin'), str(HERE)]


def prepare(folder, epochs):
    """Capture a fresh entropy-initialized start for the three paired conditions."""
    import numpy as np
    import torch
    from math_chain_corpus import MathChainCorpus
    folder.mkdir(exist_ok=False)
    with (folder / 'initial-rng.pkl').open('wb') as stream:
        pickle.dump(dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state()), stream)
    corpus = MathChainCorpus()
    presentations = [{split: [asdict(doc) for doc in docs]
                      for split, docs in corpus.presentation().items()}
                     for _ in range(epochs)]
    (folder / 'presentations.json').write_text(json.dumps(presentations) + '\n')
    (folder / 'origin.json').write_text(json.dumps(dict(seed=None, source='fresh process entropy',
        paired_conditions=3, epochs=epochs, retries=0)) + '\n')


def stage(data, presentation, *, condition, epoch, run):
    import torch
    from math_chain_corpus import ChainDocument, flatten
    docs = {}
    for split, rows in presentation.items():
        docs[split] = []
        for row in rows:
            doc = ChainDocument(**{key: tuple(value) if isinstance(value, list) else value for key, value in row.items()})
            if condition == 'expectation_only' and split == 'train' and doc.question is not None:
                doc = replace(doc, sentences=tuple(text for text in doc.sentences
                                                   if not text.startswith('the answer is ')))
            docs[split].append(doc)
    flat = {split: flatten(rows, supplied=split == 'train' and condition != 'expectation_only')
            for split, rows in docs.items()}
    data.processLM({split: dict(text=flat[split][0], label=[])
                    for split in ('train', 'validation', 'test')})
    data.text_answers = {split: values[1] for split, values in flat.items()}
    data.has_supervised_outputs = condition != 'expectation_only'
    data.math_chain_documents = docs
    data.source_manifest = dict(dataset='math_chain')
    for split, (texts, _, addresses) in flat.items():
        data.source_addresses[split] = [dict(address, split=split,
            document_key=('math-chain', run, epoch, split, docs[split][address['document']].key))
            for address in addresses]
        if split == 'beyond':
            data.beyond_input = texts
            data.beyond_output = [torch.zeros(1) for _ in texts]


def train(folder, paired, condition, run):
    import numpy as np
    import torch
    import Language
    from data import TheData
    from Models import BaseModel
    from util import init_config
    from MathChainTraining import present
    from math_observer import Observer
    protocol = json.loads((HERE / 'protocol.json').read_text())
    start = time.monotonic()
    result = dict(condition=condition, run=run, seed=None, retries=0, completed_epochs=0,
                  required_epochs=protocol['epochs'], status='started')
    folder.mkdir(exist_ok=False)
    with (paired / 'initial-rng.pkl').open('rb') as stream:
        state = pickle.load(stream)
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_num_threads(1)
    presentations = json.loads((paired / 'presentations.json').read_text())
    assert len(presentations) == protocol['epochs']
    model = None

    def unseeded(*_args, **_kwargs):
        raise AssertionError('explicit learner seed prohibited')

    try:
        with patch.object(torch, 'manual_seed', unseeded), patch.object(np.random, 'seed', unseeded):
            config = ROOT / 'data/MM_math_chain.xml'
            init_config(path=str(config), defaults_path=str(ROOT / 'data/model.xml'))
            Language.TheGrammar._configured = False
            TheData.load('math_chain')
            stage(TheData, presentations[0], condition=condition, epoch=0, run=run)
            model = BaseModel.from_config(str(config), data=TheData)[0]
            assert tuple(model.outputSpace.outputShape) == (1, 1)
            assert not any(getattr(module, 'out_features', None) == 21 for module in model.modules())
            assert 'plus' not in model.grammatical_thoughts.executable_operation_ids
            if condition == 'zero_attention_budget':
                model.attention_budget = 0
            result['attention_budget'] = model.attention_budget
            chooser = model._selected_thought_chooser(None)
            before = {name: value.detach().clone() for name, value in chooser.named_parameters()}
            torch.save(before, folder / 'initial-chooser.pt')
            optimizer = model.getOptimizer(lr=.001)
            with Observer(folder) as observer:
                for epoch, presentation in enumerate(presentations):
                    stage(TheData, presentation, condition=condition, epoch=epoch, run=run)
                    if condition == 'expectation_only':
                        assert not TheData.has_supervised_outputs
                        assert not any('the answer is ' in text for text in TheData.train_input)
                        assert all(value is None for value in TheData.text_answers['train'])
                    observer.context = dict(epoch=epoch + 1, phase='train')
                    report = present(model, split='train', optimizer=optimizer,
                        batch_size=8, after_batch=observer.after_batch)
                    result['completed_epochs'] = epoch + 1
                    result['last_epoch'] = report
                    result['observer'] = observer.report()
                    result['seconds'] = time.monotonic() - start
                    (folder / 'progress.json').write_text(json.dumps(result, indent=2) + '\n')
                    print(json.dumps(dict(epoch=epoch + 1, seconds=result['seconds'], **report)), flush=True)
                observer.context = dict(epoch=protocol['epochs'], phase='evaluation')
                for split in ('test', 'beyond'):
                    present(model, split=split, batch_size=4, after_batch=observer.after_batch)
                    rows = [row for row in observer.questions if row['phase'] == 'evaluation' and row['split'] == split]
                    result[split] = dict(rows=rows, correct=sum(row['bound_correct'] for row in rows),
                                         chains=sum(row['chain_correct'] for row in rows),
                                         episodes=sum(row['episode'] for row in rows))
                result['held_out_bar'] = len(result['test']['rows']) == 4 and result['test']['correct'] == 4
                result['chain_bar'] = result['test']['chains'] == 4 and result['test']['episodes'] == 4
                result['observer'] = observer.report()
                result['chooser_movement'] = {name: float((value.detach() - before[name]).norm())
                    for name, value in chooser.named_parameters() if name in before}
                result['new_chooser_parameters'] = [name for name, _ in chooser.named_parameters() if name not in before]
                result['status'] = 'completed'
                torch.save(model.state_dict(), folder / 'final-state.pt')
    except BaseException as error:
        result['status'] = 'failed'
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        result['seconds'] = time.monotonic() - start
        (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        if model is not None:
            model.End()


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        train(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
