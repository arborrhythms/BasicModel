"""Observe live staging storage across complete intact/shuffled controls."""
import json
import resource
import torch
import eval_nanochat_grammar as gate


def test_control_cache_lifetime(monkeypatch):
    model, data = gate.build_eval_model(gate.DEFAULT_MODEL, autoload=False)
    def memory(label):
        seen = {}
        def visit(path, value, depth=0):
            if torch.is_tensor(value) and value.layout == torch.strided:
                storage = value.untyped_storage()
                if storage.nbytes() > 10_000_000:
                    seen.setdefault(storage._cdata, dict(bytes=storage.nbytes(), paths=[]))['paths'].append(path)
            elif depth < 3 and isinstance(value, (dict, tuple, list)):
                for name, item in (value.items() if isinstance(value, dict) else enumerate(value)):
                    visit(f'{path}.{name}', item, depth+1)
        for name, module in model.named_modules():
            for key, value in vars(module).items():
                if key != '_modules': visit(f'{name}.{key}', value)
        print(json.dumps(dict(label=label, peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            storages=sorted(seen.values(), key=lambda x:-x['bytes']))), flush=True)
    for name in ('_lex_embed_stem', '_stage_serial_concept_rows', '_stage_reading_word_concepts', '_stage_snapshot_bytes'):
        original = getattr(model, name)
        def wrap(*args, _name=name, _original=original, **kwargs):
            print('ENTER', _name, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
            result = _original(*args, **kwargs)
            print('EXIT', _name, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
            return result
        monkeypatch.setattr(model, name, wrap)
    memory('built')
    item = gate.load_manifest(gate.DEFAULT_MANIFEST)['items'][0]
    with gate.frozen_online_learning(model):
        for control in ('intact', 'shuffled'):
            gate._score_control_batch(model, data, [item], control, 16)
            memory(control)
