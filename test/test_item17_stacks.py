"""The fused Pi and Sigma pair, its stacks, and what they learn."""
import itertools
import json
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.assessment_rung(1)


def _signed(*shape):
    x = torch.rand(*shape)*3 - 1.5
    x.view(-1)[:5] = torch.tensor([-1., 1., 0., 1.5, -1.5])
    return x


# Certificate 7: the stacks.

def test_a_fused_pair_equals_its_two_layers():
    from Layers import PiLayer3, SigmaPiLayer
    torch.manual_seed(0)
    layer = SigmaPiLayer(6, 9, 4)
    with torch.no_grad():
        layer.pi.raw.mul_(torch.rand_like(layer.pi.raw)*20)
        layer.sigma.weight.copy_(torch.randn_like(layer.sigma.weight)*3)
    x = _signed(64, 6)
    lanes = torch.stack((x, -x), -2)
    fused = layer(lanes)
    rows = layer.pi(lanes)
    torch.testing.assert_close(fused, layer.sigma(PiLayer3.symbols(rows)), rtol=0, atol=1e-6)
    assert torch.isfinite(fused).all() and (fused.abs() <= 1).all()
    with pytest.raises(RuntimeError, match='no way back'):
        layer.reverse(fused)


def test_a_stack_starts_at_zero_and_reads_a_signed_input_as_its_lanes():
    from SigmaPiStack import SigmaPiStack
    stack = SigmaPiStack(5, 3, hidden=7, pairs=2, seed=4)
    assert len(stack.stages) == 2
    x = _signed(10, 5)
    assert torch.equal(stack(torch.stack((x, -x), -2)), torch.zeros(10, 3))
    with torch.no_grad():
        stack.stages[-1].sigma.weight.normal_()
    lanes = torch.stack((x.clamp(min=0), (-x).clamp(min=0)), -2)
    torch.testing.assert_close(stack(torch.stack((x, -x), -2)), stack(lanes), rtol=0, atol=0)
    with pytest.raises(ValueError, match='constructed width'):
        stack(x)


def test_a_stack_is_one_graph_across_batch_sizes():
    from torch._dynamo.testing import CompileCounter
    from SigmaPiStack import SigmaPiStack
    torch._dynamo.reset()
    stack = SigmaPiStack(12, 12, hidden=16, pairs=2)
    counter = CompileCounter()
    compiled = torch.compile(stack, backend=counter, fullgraph=True, dynamic=True)
    for batch in (3, 5, 8, 17):
        x = _signed(batch, 12)
        x = torch.stack((x, -x), -2)
        torch.testing.assert_close(compiled(x), stack(x), rtol=0, atol=1e-6)
    assert counter.frame_count == 1
    torch._dynamo.reset()


def _seconds(module, x, repeats=50):
    for _ in range(5):
        module(x).sum().backward()
    if x.device.type == 'mps':
        torch.mps.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        module(x).sum().backward()
    if x.device.type == 'mps':
        torch.mps.synchronize()
    return (time.perf_counter() - started)/repeats


def _word_combinations():
    # Each word owns a coordinate; the omitted last word has no input evidence.
    present = torch.zeros(4, 6, 2)
    for row, pair in enumerate(((0, 1), (2, 3), (0, 3), (2, 1))):
        present[row, list(pair), 0] = 1
    target = present.clone()
    target[torch.arange(4), torch.tensor([4, 4, 5, 5]), 0] = 1
    return present.transpose(-1, -2), target[..., 0] - target[..., 1]


def _learns_the_combinations(stack, lanes, target, presentations):
    optimizer = torch.optim.Adam(stack.parameters(), lr=.01)
    for _ in range(presentations):
        optimizer.zero_grad()
        (stack(lanes) - target).square().mean().backward()
        optimizer.step()
    read = stack(lanes).detach()
    expected = torch.tensor([0, 0, 1, 1])
    return bool(torch.equal(read[:, 4:].argmax(-1), expected)
                and (read[torch.arange(4), torch.tensor([4, 4, 5, 5])] > .5).all()
                and (read[torch.arange(4), torch.tensor([5, 5, 4, 4])] < .5).all())


def test_depth_one_two_and_four_pairs():
    """The target: the four word combinations, each read as its omitted
    word, with 1,000 presentations on five starts drawn from the system.
    Measured first: one pair and two pairs learn it on every start; four
    pairs is recorded."""
    from SigmaPiStack import SigmaPiStack
    lanes, target = _word_combinations()
    starts = _drawn_starts(5)
    learned = {pairs: [_learns_the_combinations(SigmaPiStack(6, 6, hidden=64, pairs=pairs, seed=seed),
                                                lanes, target, 1000) for seed in starts]
               for pairs in (1, 2, 4)}
    print(dict(depth={pairs: sum(values) for pairs, values in learned.items()}, starts=starts,
               presentations=1000))
    assert all(learned[1]) and all(learned[2])


_DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device', _DEVICES)
def test_seconds_against_an_mlp_layer(device, lesson):
    stack = lesson['expect_stack']
    width, hidden = stack.inputs, stack.hidden
    import copy
    pair = copy.deepcopy(stack).to(device)
    # The same weights: the Pi's two lanes by the hidden width, then the Sigma.
    mlp = torch.nn.Sequential(torch.nn.Flatten(-2), torch.nn.Linear(2*width, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, width)).to(device)
    result = {}
    for batch in (1, 64):
        x = _signed(batch, width).to(device)
        x = torch.stack((x, -x), -2)
        result[batch] = dict(stack=_seconds(pair, x), mlp=_seconds(mlp, x))
    print(dict(device=device, width=width, hidden=hidden, seconds=result))


# The lesson's sixteen sentences, with the root ideas the read is taught.

@pytest.fixture(scope='module')
def lesson():
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    import IdeaRead
    import Language
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    roots, predictions = [], []
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        complete = IdeaRead.complete
        def watched(model, record, present, registry):
            roots.append(record.meaning_root.detach().clone())
            owner = model._concept_owner()
            predictions.append(owner.idea_stack(
                model._idea_observation.transpose(-1, -2)).detach().clone())
            return complete(model, record, present, registry)
        patch.setattr(IdeaRead, 'complete', watched)
        TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                          for split in ('train', 'validation', 'test')})
        Language.TheGrammar._configured = False
        model = BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]
        model._tensor_peer_while_eager = True
        optimizer = model.getOptimizer(lr=.001)
        curriculum = json.loads(Path('data/corpus-lessons.json').read_text())
        texts, targets = stage_documents(TheData, curriculum['stages'][0])
        admit_training_forms(model, texts, targets)
        count = 0
        class Complete(Exception):
            pass
        def batch(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 16:
                raise Complete
        try:
            present(model, split='train', optimizer=optimizer, batch_size=1, after_batch=batch)
        except Complete:
            pass
        owner = model._concept_owner()
        result = dict(roots=roots, predictions=predictions, sentences=count,
                      expect_stack=owner.expect_stack, idea_stack=owner.idea_stack)
        yield result
        model.End()


def test_the_root_ideas_lie_within_what_the_read_can_give(lesson):
    roots = lesson['roots']
    assert lesson['sentences'] == 16 and roots
    least = min(float(root.min()) for root in roots)
    greatest = max(float(root.max()) for root in roots)
    readable = (-1., 1.)
    print(dict(sentences=lesson['sentences'], kept_roots=len(roots), least=least,
               greatest=greatest, read_range=readable,
               read_least=min(float(p.min()) for p in lesson['predictions']),
               read_greatest=max(float(p.max()) for p in lesson['predictions'])))
    assert readable[0] <= least and greatest <= readable[1]


# Certificate 22: and, or and not.

_VALUES = (1., 0., -1.)
_PAIRS = torch.tensor(list(itertools.product(_VALUES, _VALUES)))
_LANES = torch.stack((_PAIRS, -_PAIRS), -2)
_TASKS = dict(conjunction=_PAIRS.min(1).values, disjunction=_PAIRS.max(1).values,
              negation=-_PAIRS[:, 0])


def _answers(y):
    return (y > .5).long() - (y < -.5).long()


def _taught(task, seed, *, pi_penalty, sigma_penalty, presentations=2000):
    from SigmaPiStack import SigmaPiStack
    stack = SigmaPiStack(2, 1, hidden=4, pairs=1, seed=seed)
    pair = stack.stages[0]
    optimizer = torch.optim.Adam(stack.parameters(), lr=.05)
    target = .9*_TASKS[task]
    for _ in range(presentations):
        optimizer.zero_grad()
        loss = ((stack(_LANES)[:, 0] - target).square().mean()
                + pi_penalty*pair.pi.raw.abs().sum() + sigma_penalty*pair.sigma.weight.abs().sum())
        loss.backward()
        optimizer.step()
    return stack


def _read_back(stack, threshold=.1):
    """Present entries set to one, keeping a Not entry's sign; the rest to zero."""
    pair = stack.stages[0]
    with torch.no_grad():
        pair.pi.raw.copy_((pair.pi.raw.abs() > threshold).float())
        weight = pair.sigma.weight
        weight.copy_(torch.sign(weight)*(weight.abs() > threshold))
    return stack


def _drawn_starts(count):
    """Starts drawn from the system, never chosen."""
    import os
    return [int.from_bytes(os.urandom(4), 'little') for _ in range(count)]


def _outcomes(task, *, pi_penalty, sigma_penalty, starts=10):
    solved = read = not_entries = literals = 0
    seeds = _drawn_starts(starts)
    for seed in seeds:
        stack = _taught(task, seed, pi_penalty=pi_penalty, sigma_penalty=sigma_penalty)
        wanted = _TASKS[task].long()
        solved += bool(torch.equal(_answers(stack(_LANES)[:, 0].detach()), wanted))
        not_entries += int((stack.stages[0].sigma.weight < -.1).sum())
        literals += int((stack.stages[0].pi.raw.abs() > .1).sum())
        read += bool(torch.equal(_answers(_read_back(stack)(_LANES)[:, 0].detach()), wanted))
    return dict(solved=solved, read_back=read, not_entries=not_entries,
                literals=literals/starts, starts=starts, seeds=seeds)


@pytest.mark.parametrize('task', sorted(_TASKS))
def test_and_or_and_not_are_learned_and_read_back(task):
    both = _outcomes(task, pi_penalty=.003, sigma_penalty=.003)
    unpenalized = _outcomes(task, pi_penalty=0., sigma_penalty=0.)
    pi_alone = _outcomes(task, pi_penalty=.003, sigma_penalty=0.)
    print(dict(task=task, penalty_on_both=both, no_penalty=unpenalized, penalty_on_pi=pi_alone))
    assert both['solved'] == len(both['seeds']) and unpenalized['solved'] == len(unpenalized['seeds'])
    assert both['read_back'] == len(both['seeds'])
    if task == 'negation':
        assert both['not_entries'] >= len(both['seeds'])
    else:
        assert both['not_entries'] == 0
