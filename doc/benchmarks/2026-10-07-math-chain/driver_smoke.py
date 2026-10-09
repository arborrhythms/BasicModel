"""Development-only check of the corpus driver and passive verifier plumbing."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / 'bin')]


def main():
    import torch
    import Language
    from data import TheData
    from Models import BaseModel
    from util import init_config
    from math_chain_corpus import MathChainCorpus
    from MathChainTraining import present
    from math_train import stage
    from math_observer import Observer
    torch.set_num_threads(1)
    folder = HERE / 'driver-smoke'
    folder.mkdir(exist_ok=False)
    config = ROOT / 'data/MM_math_chain.xml'
    init_config(path=str(config), defaults_path=str(ROOT / 'data/model.xml'))
    Language.TheGrammar._configured = False
    TheData.load('math_chain')
    corpus = MathChainCorpus()
    train = [*corpus.counting()[:2], *(corpus.problem((a, 0), split='train', training=True) for a in range(8))]
    docs = dict(corpus.presentation(), train=train)
    stage(TheData, {split: [asdict(doc) for doc in rows] for split, rows in docs.items()},
          condition='answer_and_expectation', epoch=0, run='development-smoke')
    model = BaseModel.from_config(str(config), data=TheData)[0]
    try:
        with Observer(folder) as observer:
            observer.context = dict(epoch=0, phase='construction-smoke')
            report = present(model, split='train', optimizer=model.getOptimizer(lr=.001),
                             batch_size=8, after_batch=observer.after_batch)
            assert report['sentences'] == 51
            assert len(observer.questions) == 8
            (folder / 'result.json').write_text(json.dumps(dict(report=report,
                observer=observer.report(), learning_attempt=False), indent=2) + '\n')
    finally:
        model.End()


if __name__ == '__main__':
    main()
