"""Chooser heads match the independent action replay during the migration."""
from types import SimpleNamespace

import pytest
import torch

from Memory import STMChunks, OCCURRENCE, SYMBOL, RELATION
from ReadingIndex import ReadingIndex
from Spaces import LanguageOperationChoice
import STMParse


@pytest.fixture(params=['cpu', 'mps'])
def device(request):
    if request.param == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    return torch.device(request.param)


def reference(index, symbols):
    """The old write_edges construction, including its pair allocation order."""
    pairs, heads = [], {}
    def join(left, right):
        pairs.append((left, right))
        return RELATION, len(pairs) - 1
    def sequence(children):
        head = None
        for child in reversed(children):
            head = child if head is None else join(child, head)
        return head
    for at, (kind, local, word) in enumerate(index.steps):
        if kind == 0:
            heads[at] = OCCURRENCE, 100 + word
        elif kind in (1, 2):
            op = symbols[local if kind == 1 else 2 + local]
            heads[at] = join((SYMBOL, op), sequence([heads[child] for child in index.operands[at]]))
    return pairs, sequence([heads[root] for root in index.roots])


READINGS = [
    [(0, -1, 0)],
    [(0, -1, 0), (0, -1, 1), (0, -1, 2)],
    [(0, -1, 0), (0, -1, 1), (1, 0, -1), (0, -1, 2), (1, 0, -1)],
    [(0, -1, 0), (0, -1, 1), (3, 0, -1), (1, 0, -1)],
    [(0, -1, 0), (0, -1, 1), (2, 0, 1), (1, 0, -1)],
    [(0, -1, 0), (1, 1, -1), (0, -1, 1), (2, 0, 0), (3, 0, -1)],
    [(0, -1, 0), (0, -1, 1), (1, 0, -1), (0, -1, 2), (0, -1, 3),
     (1, 0, -1), (3, 0, -1), (1, 0, -1), (2, 0, 0)],
]


def batch(device, *, compiled=False):
    count, capacity, width = len(READINGS), 8, 3
    longest = max(map(len, READINGS))
    state = STMChunks.empty(count, capacity, width, device=device)
    pairs = torch.zeros(count, 2 * longest, 2, 2, dtype=torch.long, device=device)
    order = torch.full((count, longest), -1, dtype=torch.long, device=device)
    symbols = torch.tensor([71, 72, 73], device=device)
    binary, unary = torch.tensor([0, 1], device=device), torch.tensor([2], device=device)
    execute = STMParse.apply
    if compiled:
        execute = torch.compile(execute, backend='eager', fullgraph=True)
    for turn in range(longest):
        step = torch.tensor([row[turn] if turn < len(row) else (-1, -1, -1)
                             for row in READINGS], device=device)
        kind, local, word = step.unbind(-1)
        state, _ = STMChunks.deposit(state, state[0].new_ones(count, width), kind == 0, stack=True)
        state = STMParse.push(state, word, kind == 0)
        active = kind > 0
        choice = LanguageOperationChoice(state[0].new_ones(count, width), kind, word.clamp_min(0),
            local, active, state[0].new_ones(count), local, active,
            head_role=torch.where(torch.arange(count, device=device) % 2 == 0, 1, 2),
            stack_arity=torch.where((kind == 1) & (local == 1), 1, 2))
        after = STMChunks.apply_choice(state, choice)
        # Fixations need not visit source columns in ascending order.
        column = torch.full((count,), longest - 1 - turn, device=device)
        state, pairs, order = execute(state, after, choice, column, pairs, order, symbols, binary, unary)
    return state, pairs, order


@pytest.mark.parametrize('compiled', [False, True])
def test_chooser_heads_match_replay_for_unequal_partial_and_transposed_rows(device, compiled):
    state, pairs, order = batch(device, compiled=compiled)
    language = SimpleNamespace(_compose_binary_rules=[SimpleNamespace(method_name='sum'),
                                SimpleNamespace(method_name='bind')])
    heads = STMParse.roots(state)
    for row, steps in enumerate(READINGS):
        words = sum(kind == 0 for kind, _, _ in steps)
        index = ReadingIndex(torch.tensor(steps), words, language)
        # The copy is one host transfer of its tensors together (7919cec8d5),
        # so the word positions are on the device of the pairs, as the
        # sentence trace holds them.
        result = STMParse.copy_pairs(pairs[row], heads[row], torch.arange(words, device=device),
                                    range(100, 100 + words), order[row])
        assert result == reference(index, [71, 72, 73])
        assert int((order[row] >= 0).sum()) == sum(kind > 0 for kind, _, _ in steps)
        assert int((pairs[row, 1::2, 0, 0] == SYMBOL).sum()) == sum(kind in (1, 2) for kind, _, _ in steps)


def test_mixed_winners_keep_whole_parse_and_fork_restoration(device):
    from SentenceFork import select_rows, snapshot
    state, pairs, order = batch(device)
    original = (state, (pairs, order))
    saved = snapshot(original, live=False)
    replaced = (tuple(value.clone() for value in state), (pairs.clone(), order.clone()))
    replaced[0][STMChunks.HEADS].zero_()
    replaced[1][0].zero_()
    mask = torch.arange(len(READINGS), device=device) % 2 == 0
    kept = select_rows(replaced, saved, mask)
    for field in (STMChunks.HEADS, STMChunks.STACK):
        torch.testing.assert_close(kept[0][field][mask], saved[0][field][mask], rtol=0, atol=0)
    torch.testing.assert_close(kept[1][0][mask], saved[1][0][mask], rtol=0, atol=0)


@pytest.mark.usefixtures('eager_reading')
def test_live_sentence_uses_heads_and_device_costs_without_index_replay(monkeypatch):
    from pathlib import Path
    from test_mm_xor import _fresh_model, _PROJECT
    from ParseForest import ParseForest
    from WholePaths import application_cost, load_cost
    model = _fresh_model(str(Path(_PROJECT) / 'data/XOR_grammar.xml'))[0]
    old_schedule, new_schedule = model._derivation_program, model._parse_program
    observed = []
    def schedule(sentence=None):
        result = new_schedule(sentence)
        expected = old_schedule(sentence, padded=True)
        for b in range(len(result[0])):
            torch.testing.assert_close(result[0][b][result[0][b] >= 0], expected[0][b][expected[0][b] >= 0], rtol=0, atol=0)
            torch.testing.assert_close(result[1][b][result[1][b, :, 0] >= 0],
                                       expected[1][b][expected[1][b, :, 0] >= 0], rtol=0, atol=0)
        observed.append(True)
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary sentence replayed an action stack')
    monkeypatch.setattr(model, '_parse_program', schedule)
    monkeypatch.setattr(model, '_derivation_program', forbidden)
    monkeypatch.setattr(ReadingIndex, '__init__', forbidden)
    original = model._sentence_observation
    def observation(*args, **kwargs):
        result = original(*args, **kwargs)
        like = result['mask'].float()
        legacy = {key: value for key, value in result.items() if key not in ('application_count', 'memory_load')}
        torch.testing.assert_close(application_cost(model.languageSpace, result, None, like),
            application_cost(model.languageSpace, legacy, None, like), rtol=0, atol=0)
        torch.testing.assert_close(load_cost(model, result, like), load_cost(model, legacy, like), rtol=0, atol=0)
        assert all(isinstance(entry.action_index, ParseForest) for entry in result['entries'] if entry is not None)
        return result
    monkeypatch.setattr(model, '_sentence_observation', observation)
    with torch.no_grad():
        for _ in range(3):
            model.understand(model.inputSpace.prepInput(['red blue red', 'blue red', 'red', '']))
            model.End()
            model.dispatch_per_row_reset([True] * 4)
    assert observed
