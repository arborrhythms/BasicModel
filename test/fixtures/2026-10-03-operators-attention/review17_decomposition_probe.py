"""Observe compose-pair coverage without a teacher, writes or RNG draws."""
from contextlib import contextmanager
from unittest.mock import patch
import torch


@torch.no_grad()
def pair_rows(model, entries, record):
    """Compare the fixed pair choice with operands read from the compose program."""
    from Generative import preserve_operator_activations
    language = model.languageSpace
    bank = record.primed

    @preserve_operator_activations
    def inspect(language):
        rows = []
        ops = language.language_layer.operation_layer.ops
        for batch, program in enumerate(entries['entries']):
            if program is None or program.operation_values is None:
                continue
            stack = []
            word_rows = record.word_rows[batch].index_select(0, model._sentence_leaf_positions(record, batch))
            for index, (kind, op_index, word) in enumerate(program.actions.detach().cpu().tolist()):
                if kind == 0:
                    stack.append(int(word_rows[word]))
                elif kind == 3:
                    stack[-2:] = reversed(stack[-2:])
                elif kind == 1:
                    if language._compose_binary_rules[op_index].method_name == 'bind':
                        continue
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(-1)
                    op = getattr(ops[op_index], 'gl', ops[op_index])
                    if getattr(op, 'inverse_kind', None) == 'case_search':
                        continue
                    row = dict(row=batch, operation=op_index, target=[left, right],
                               present=False, picked_true=False)
                    if bank.codes.shape[1]:
                        parent = program.operation_values[index, 2].detach()[None]
                        flag = torch.zeros(1, dtype=torch.bool, device=parent.device)
                        _, _, _, details = language._bounded_binary_reconstruction(
                            op, parent, torch.zeros_like(parent), flag, flag,
                            bank.codes[batch:batch+1], bank.valid[batch:batch+1],
                            model.reconstruction_basis_limit,
                            left_priming=bank.weights[batch:batch+1],
                            right_priming=bank.weights[batch:batch+1],
                            chooser=language.decomposition_chooser, return_details=True)
                        left_rows = bank.rows[batch:batch+1].gather(1, details['left_indices'])
                        right_rows = bank.rows[batch:batch+1].gather(1, details['right_indices'])
                        matches = ((left_rows[:, :, None] == left) & (right_rows[:, None, :] == right)
                                   & (left >= 0) & (right >= 0) & details['allowed']).flatten(1)
                        row.update(present=bool(matches.any()),
                                   picked_true=bool(matches.gather(1, details['selected'][:, None]).any()))
                    rows.append(row)
        return rows
    return inspect(language)


@contextmanager
def observe_decomposition(result):
    from Models import BasicModel
    observation = BasicModel._sentence_observation
    report = result.setdefault('decomposition_chooser', {})

    def snapshot(model, entries, record):
        chooser = model.languageSpace.decomposition_chooser
        rows = pair_rows(model, entries, record)
        count = len(rows)
        present = sum(row['present'] for row in rows)
        correct = sum(row['picked_true'] for row in rows)
        return dict(feature_names=list(chooser.feature_names),
            weights=chooser.weight.detach().cpu().tolist(), undos=count,
            true_pair_in_shortlist=present, absent_targets=count-present,
            pick_equals_true=correct,
            true_pair_in_shortlist_rate=present/count if count else None,
            pick_equals_true_rate=correct/count if count else None,
            pick_equals_true_given_present=correct/present if present else None,
            coverage_source='compose program operand identities', rows=rows)

    def observed_observation(model, state, sid, active, **kwargs):
        value = observation(model, state, sid, active, **kwargs)
        rng = torch.random.get_rng_state()
        reading = snapshot(model, value, model._trial_understanding(state, sid, active))
        assert torch.equal(rng, torch.random.get_rng_state())
        report.setdefault('start', reading)
        if getattr(model, '_sentence_training', False):
            report['training_calls'] = report.get('training_calls', 0) + 1
        else:
            report['end'] = reading
        return value

    with patch.object(BasicModel, '_sentence_observation', observed_observation):
        yield
