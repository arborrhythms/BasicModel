"""Read-only WordIdentity order certificate, FutureWork clarification 2026-10-25."""
import json

import torch


def test_native_kept_readings_have_transitive_percept_codes(tmp_path, monkeypatch, eager_reading):
    from Models import BasicModel
    from Layers import Ops
    from PrependedSymbols import word_view
    from test_parts_of_speech import test_kept_readings_alone_supply_occupation_counts

    certificates = []
    commit = BasicModel._commit_sentence

    def committed(model, state, sid, active, observations, predictions, wins):
        result = commit(model, state, sid, active, observations, predictions, wins)
        derived = model._concept_owner().similarity_codebook.mereology
        towers = (derived._native(0), derived._native(1))
        identity = model.perceptualSpace.percept_store.identity
        before = {name:value.detach().clone() for name,value in model.state_dict().items()}
        ps = model.perceptualSpace._forward_input
        memberships = towers[1].primitive_properties.on_counts(model.wholeSpace._staged_word_primitive_counts)
        spans = model.wholeSpace._staged_word_property_spans
        staged = word_view(model.inputSpace, 'concept_ids')
        records, words, failures = [], {}, []

        def form(raw):
            code = identity.form(raw).to(towers[0].W)
            stored = towers[0].lookup_rows(identity.word_rows[raw])[:identity.width].detach()
            torch.testing.assert_close(code, stored, rtol=0, atol=0)
            return stored

        def edge(part_code, whole_code, **description):
            valid = bool(Ops.partOf(part_code, whole_code))
            record = dict(**description, valid=valid,
                          max_violation=float((part_code-whole_code).clamp_min(0).max()))
            if not valid:
                failures.append(record)
            return record

        for b in active.nonzero().flatten().tolist():
            reading = observations[int(bool(wins[b]))]['entries'][b]
            assert reading is not None and reading.word_ids.numel()
            kept = torch.isin(staged[b], reading.word_ids.to(staged.device))
            word_records, entropy = [], []
            for position in kept.nonzero().flatten().tolist():
                raw = ps['word_texts'][b][position].encode('latin1')
                code, atoms = form(raw), identity.part_atoms(raw)
                words[raw] = (code, atoms)
                atom_edges, nested_edges = [], []
                for atom in sorted(atoms):
                    atom_code = identity.atom_code(atom).to(code)
                    stored = towers[0].lookup_rows(identity.atoms[atom])[:identity.width].detach()
                    torch.testing.assert_close(atom_code, stored, rtol=0, atol=0)
                    atom_edges.append(edge(stored, code, atom=atom, row=identity.atoms[atom]))
                for part, lo, hi in identity.nested.get(raw, ()):
                    nested_edges.append(edge(form(part), code, part=part.hex(), span=[lo, hi]))
                word_records.append(dict(raw=raw.hex(), row=identity.word_rows[raw],
                                         atoms=atom_edges, nested=nested_edges))
                for slot, span in enumerate(spans[b,position].tolist()):
                    if span[1] <= span[0]:
                        continue
                    for row in (memberships[b,position,slot]>0).nonzero().flatten().tolist():
                        whole = towers[1].lookup_rows(row).detach()
                        entropy.append(dict(row=row, span=span,
                            distance_from_ones=float(torch.linalg.vector_norm(1-whole))))
            assert word_records and entropy
            records.append(dict(stream=b, words=word_records, whole_entropy=entropy))
        pairs = [edge(a, b, part=left.hex(), whole=right.hex())
                 for left,(a,aa) in words.items() for right,(b,ba) in words.items()
                 if aa <= ba]
        after = model.state_dict()
        assert before.keys()==after.keys()
        assert all(torch.equal(value,after[name]) for name,value in before.items()), 'certificate changed model tensors'
        certificates.append(dict(readings=records, word_pairs=pairs, failures=failures, read_only=True))
        return result

    monkeypatch.setattr(BasicModel,'_commit_sentence',committed)
    test_kept_readings_alone_supply_occupation_counts(tmp_path,monkeypatch,eager_reading)
    (tmp_path/'certificate.json').write_text(json.dumps(dict(closes=certificates),indent=2)+'\n')
    assert certificates
    assert not any(c['failures'] for c in certificates), 'native identity containment failed; stop and report'
