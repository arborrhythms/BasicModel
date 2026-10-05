"""Reproduce the ninth manifest item's rejection without changing its input."""
import json
import torch
import eval_nanochat_grammar as gate


def test_manifest_candidate_acceptance():
    model, data = gate.build_eval_model(gate.DEFAULT_MODEL, autoload=False)
    manifest = gate.load_manifest(gate.DEFAULT_MANIFEST)
    with gate.frozen_online_learning(model):
        for ordinal, item in enumerate(manifest['items'][:9], 1):
            for control in ('intact', 'shuffled'):
                try:
                    gate._score_control_batch(model, data, [item], control, 16)
                except RuntimeError:
                    reading = model._attention_words
                    forms, identities, _, live = model._attention_forms
                    rows = []
                    for b in range(len(forms)):
                        missing = live[b] & ~reading.accepted[b]
                        if not bool(missing.any()):
                            continue
                        rows.append(dict(row=b, candidate=item['candidates'][b],
                            words=[dict(form=form, identity=int(identities[b, w]),
                                poles=model._attention_native_poles[b, w].tolist(),
                                accepted=bool(reading.accepted[b, w]),
                                descended=bool(reading.descended[b, w]),
                                span=model._attention_spans[b, w].tolist())
                                for w, form in enumerate(forms[b]) if bool(live[b, w])],
                            spent=int(reading.table.spent[b]),
                            actions=reading.actions[b].tolist(),
                            intervals=reading.table.intervals[b, reading.table.valid[b]].tolist(),
                            done=reading.table.done[b, reading.table.valid[b]].tolist()))
                    print(json.dumps(dict(ordinal=ordinal, item=item, control=control, rows=rows)), flush=True)
                    raise
