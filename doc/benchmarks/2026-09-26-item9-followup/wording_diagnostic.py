"""Inspect the unchanged 9000-update wording gate; retain its failure."""
import os
os.environ.setdefault('MODEL_COMPILE', 'eager')
os.environ.setdefault('BASICMODEL_DEVICE', 'cpu')
import json, math, sys, tempfile, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
import pytest, torch
import test_compiled_word_chunk as fixtures
import test_output_walk as captures
from test_surface_grammar import test_real_text_has_a_complete_selected_meaning
from GrammarLessons import compose_examples
from bounded_tests import source_snapshot
from Language import _FunctionalLanguageChooser

torch.set_num_threads(1)
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
report = {'source': source_snapshot(ROOT), 'seed': 931, 'updates': 9000}
started = time.perf_counter()
with tempfile.TemporaryDirectory() as td, pytest.MonkeyPatch.context() as mp:
    owners, saved = [], []
    constructor, capture = fixtures._tiny_canonical_model, captures._capture_program_probe
    def keep(*args, **kwargs):
        model = constructor(*args, **kwargs); owners.append(model); return model
    def remember(model, texts):
        out = capture(model, texts)
        saved.extend(zip(texts, out.answer_program))
        return out
    mp.setattr(fixtures, '_tiny_canonical_model', keep)
    mp.setattr(captures, '_capture_program_probe', remember)
    try:
        test_real_text_has_a_complete_selected_meaning(Path(td), mp)
        report['gate_passed'] = True
    except AssertionError as error:
        report['gate_passed'] = False
        report['failure'] = str(error)
        traceback.print_exc()
    model = owners[0]
    model.save_weights(OUT/'trained.ckpt')
    curriculum = json.loads((ROOT/'data/grammar_wording.json').read_text())
    old = dict(saved[:len(curriculum['train'])])
    heldout = dict(saved[len(curriculum['train']):])
    previous_leaves = {}
    for text, program in old.items():
        for word, value in zip(text.split(), program.leaves):
            previous_leaves.setdefault(word, value)
    details = []
    for row in curriculum['validation'] + curriculum['test']:
        program = heldout[row['text']]
        b, u = compose_examples(model.languageSpace, program, row['tree'])
        layer = model.languageSpace._tree_layer(2)
        values = torch.stack([x[0] for x in b])
        with torch.no_grad():
            _, _, routing = layer(values)
        confidence = _FunctionalLanguageChooser._reduce_confidence(routing)
        depth = torch.tensor([x[3] for x in b])
        threshold, _ = _FunctionalLanguageChooser._occupancy_threshold(depth, model.conceptualSpace.stm.capacity, model.stm_reduce_tau)
        forced = torch.tensor([x[2] for x in b]) | (depth >= model.conceptualSpace.stm.capacity)
        selected = torch.where(forced | (confidence > threshold), routing['reduce_marginal_op'][:, 0].argmax(-1), -1)
        names = [r.method_name for r in model.languageSpace._compose_binary_rules]
        steps = [dict(expected='copy' if entry[1] < 0 else names[entry[1]],
            actual='copy' if int(index) < 0 else names[int(index)],
            confidence=float(c), threshold=float(t), depth=entry[3], seal=entry[2])
            for entry, index, c, t in zip(b, selected, confidence, threshold)]
        details.append(dict(text=row['text'], steps=steps,
            actions=program.actions.tolist(), refs=program.concept_ids.tolist(),
            reference_ids=None if program.reference_ids is None else program.reference_ids.tolist(),
            leaf_drift={w:float((v-previous_leaves[w]).norm()) for w,v in zip(row['text'].split(), program.leaves) if w in previous_leaves}))
    report['heldout'] = details
    report['seconds'] = time.perf_counter()-started
    report['source_unchanged'] = report['source'] == source_snapshot(ROOT)
    (OUT/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('COMPLETE', report['gate_passed'], report['seconds'], flush=True)
