"""One sentence of the reading lesson presented compiled, as training presents
it: the output's generate walk is compiled whole on a training step (sigma-pi
plan, round 6d, the production pass)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# Measured alone on 2026-11-27: not ended after 15 minutes, 8.2 GiB at its peak.
@pytest.mark.slow
def test_one_lesson_sentence_trains_with_the_compiled_output_walk():
    import torch
    import util
    import Language
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    assert util.TheCompileBackend not in (None, '', 'none')
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    model = BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]
    try:
        optimizer = model.getOptimizer(lr=.001)
        curriculum = json.loads((ROOT/'data/corpus-lessons.json').read_text())
        texts, targets = stage_documents(TheData, curriculum['stages'][0])
        admit_training_forms(model, texts, targets)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        done = []

        class Complete(Exception):
            pass

        def batch(model, split, rows, result):
            done.append(rows)
            raise Complete
        try:
            present(model, split='train', optimizer=optimizer, batch_size=1, after_batch=batch)
        except Complete:
            pass
        assert len(done) == 1
        walk = model.__dict__.get('_output_walk_compiled')
        assert walk is not None, 'the step did not reach the output walk'
        assert getattr(walk, '__wrapped__', None) is not None, 'the walk ran eager'
        assert walk.__wrapped__ != model._output_generate_walk
        changed = [name for name, value in model.named_parameters()
                   if name in before and value.shape == before[name].shape
                   and not torch.equal(value.detach(), before[name])]
        assert changed, 'the step changed no parameter'
    finally:
        model.End()
