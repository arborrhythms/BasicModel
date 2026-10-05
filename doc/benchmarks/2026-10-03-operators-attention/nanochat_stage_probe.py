"""Bounded diagnosis of the frozen evaluator's word-only staging boundary."""
import json
import resource
import time
import torch
import eval_nanochat_grammar as gate


def test_native_word_staging():
    started = time.perf_counter()
    model, data = gate.build_eval_model(gate.DEFAULT_MODEL, autoload=False)
    item = gate.load_manifest(gate.DEFAULT_MANIFEST)['items'][0]
    texts = [item['prefix'] + candidate for candidate in item['candidates']]
    print('BUILT', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
    with gate.frozen_online_learning(model), torch.no_grad(), data.runtime_batch(texts):
        value = model.inputSpace.prepInput(list(data.train_input))
        model._lex_embed_stem(value)
        print('STAGED', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
        gate._validate_candidate_reached(model, [item], 'intact', 16)
        values = gate.word_candidate_scores(model, choices=16)
    print(json.dumps(dict(elapsed=time.perf_counter()-started,
        scores=values[0].tolist(), counts=values[3].tolist(),
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
