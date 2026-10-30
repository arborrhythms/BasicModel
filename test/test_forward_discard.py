"""Forward meaning loss pays locally; inverse undefinedness is unchanged."""
from types import SimpleNamespace

import torch

from SentenceCredit import forward_discard


def test_retained_four_clauses_recover_non_witnesses_without_a_discard_charge(tmp_path, monkeypatch):
    from pathlib import Path
    from identity_experiment import make_model
    from DerivationReconstruction import inverse_step
    from Layers import cube_unchart
    saved = torch.load(Path(__file__).resolve().parents[1] /
        'test/fixtures/2026-10-10-item6/identity-r2/validation/retained-trials.pt',
        map_location='cpu', weights_only=True)
    model = make_model(tmp_path, monkeypatch)
    try:
        for trial in saved['trials']:
            programs = [SimpleNamespace(**p) for p in trial['programs']]
            # The archive has signed, pre-centroid forms. Keep its actions and
            # all three meaning values; map only forms into the current cube
            # and express each local output as its parts' centroid.
            for program in programs:
                program.operation_values = program.operation_values.clone()
                width = saved['meaning_start']
                program.operation_values[..., :width] = cube_unchart(
                    program.operation_values[..., :width])
                for step, (kind, _, _) in enumerate(program.actions.tolist()):
                    if kind in (1, 2):
                        frame = program.operation_values[step]
                        frame[2, :width] = frame[:2 if kind == 1 else 1, :width].mean(0)
            charge = forward_discard(model.languageSpace, programs, saved['meaning_start'], trial['other_cost'])
            torch.testing.assert_close(charge, torch.zeros(4))
            torch.testing.assert_close(trial['other_cost'], torch.full((4,), 1.85))
        # Every non retains its occurrence operand, without claiming a free inverse.
        witnesses = 0
        for program in programs:
            for step, (kind, operation, _) in enumerate(program.actions.tolist()):
                if kind != 2:
                    continue
                wrapped = model.languageSpace._tree_layer(1).unary_ops[operation]
                op = getattr(wrapped, 'gl', wrapped)
                if op.rule_name != 'non':
                    continue
                frame = program.operation_values[step]
                ((value, domain),) = inverse_step(model.languageSpace, kind, operation, frame[2:3], frame)
                torch.testing.assert_close(value, frame[:1], rtol=0, atol=0)
                assert domain.all()
                witnesses += 1
        assert witnesses > 0
        # A lower step remains locally defined below a withheld parent.
        program = programs[0]
        index = int((program.actions[:, 0] == 1).nonzero()[0])
        kind, op, _ = program.actions[index].tolist()
        frame = program.operation_values[index]
        local = inverse_step(model.languageSpace, kind, op, frame[2:3], frame)
        inherited = inverse_step(model.languageSpace, kind, op, frame[2:3], frame,
            coordinates=torch.zeros_like(frame[2:3], dtype=torch.bool))
        assert all(domain.all() for _, domain in local)
        assert all(not domain.any() for _, domain in inherited)
    finally:
        model.End()


def test_exact_or_reading_pays_zero_forward_discard(monkeypatch):
    from Models import BasicModel
    from test_operators_final_b import forced_prefix_certificate
    score = BasicModel._sentence_path_cost
    charges = []
    def observed(model, *args, **kwargs):
        result = score(model, *args, **kwargs)
        registry = model._sentence_cost_registry
        charges.append(registry._value(registry._terms['reconstruction.forward_discard']).detach())
        return result
    monkeypatch.setattr(BasicModel, '_sentence_path_cost', observed)
    report = forced_prefix_certificate()
    assert report['reads'][1]['meaning_norm'] > 0
    assert all(read['multisets'] == 4 and read['R'] == read['leaf_cost']
               and read['byte_identity'] == [0.]*4 for read in report['reads'])
    assert charges and all(not value.any() for value in charges)


def test_search_inverse_is_exempt_while_field_coordinates_stay_undefined():
    from DerivationReconstruction import inverse_step
    from Interpret import activate_code
    from Language import DisjunctionLayer, LanguageSpace
    op = DisjunctionLayer()
    op.meaning_layout = (2, 6)
    codes = torch.tensor([[[1., 0., 1., 0., 0., 0.], [0., 1., 0., 1., 0., 0.]]])
    left = activate_code(codes[:, 0], torch.ones(1), 2, evidence=torch.tensor([[.7, .8]]))
    right = activate_code(codes[:, 1], torch.ones(1), 2, evidence=torch.tensor([[.3, .9]]))
    parent = op.compose(left, right)
    flag = torch.zeros(1, dtype=torch.bool)
    recovered_left, recovered_right, ready = LanguageSpace._bounded_binary_reconstruction(
        op, parent, torch.zeros_like(parent), flag, flag, codes, torch.ones(1, 2, dtype=torch.bool), 2)
    assert ready.all()
    torch.testing.assert_close(op.compose(recovered_left, recovered_right), parent, rtol=0, atol=0)
    frame = torch.stack((left[0], right[0], parent[0]))
    language = SimpleNamespace(_tree_layer=lambda arity: SimpleNamespace(ops=[op]))
    children = inverse_step(language, 1, 0, parent, frame)
    assert all(not domain.any() for _, domain in children)
    program = SimpleNamespace(actions=torch.tensor([[1, 0, -1]]), operation_values=frame[None])
    torch.testing.assert_close(forward_discard(language, [program], 2, parent.new_zeros(1)),
                               parent.new_zeros(1), rtol=0, atol=0)
