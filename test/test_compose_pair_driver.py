"""Both compose trials train; the strictly lower-loss end state survives."""
import pytest
import os
import re
import sys
import tempfile
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

_DATA = os.path.join(_PROJECT, "data")
_GRAMMAR_CONFIG = os.path.join(_DATA, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(extra=""):
    import Models, Language
    from util import init_config, init_device
    init_device("cpu")
    with open(_GRAMMAR_CONFIG) as f:
        text = f.read()
    text = re.sub(r"\s*<learning>[^<]*</learning>\s*\n", "\n", text)
    if extra:
        text = text.replace("<architecture>", f"<architecture>\n    {extra}", 1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False)
    tmp.write(text)
    tmp.close()
    try:
        init_config(path=tmp.name, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(tmp.name)
        import Models as _M
        _M.TheData.load("xor")
        m.train()
        return m
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _batch(m):
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    inp_items, out_items = next(iter(loader))
    return (m.inputSpace.prepInput(inp_items),
            m.outputSpace.prepOutput(out_items))


@pytest.mark.usefixtures('eager_reading')
def test_complete_paths_train_in_one_step_and_commit_once(monkeypatch):
    m = _build('<separateChooser>true</separateChooser>')
    optimizer = torch.optim.SGD(m.parameters(), lr=0.0)
    batch = _batch(m)
    clock0 = m.present()
    steps0 = int(getattr(m, '_training_step_count', 0))
    observed = []
    constraints = []
    commits = []
    commit = m._commit_sentence
    def capture_commit(state, *args, **kwargs):
        selected = state[0][0].detach().clone()
        result = commit(state, *args, **kwargs)
        commits.append((selected, result[0][0].detach().clone(), result[0][1].clone()))
        return result
    monkeypatch.setattr(m, '_commit_sentence', capture_commit)
    import WholePaths
    finish = WholePaths.finish
    def capture(owner, state, *args, **kwargs):
        result = finish(owner, state, *args, **kwargs)
        alternative = kwargs.get('base') is not None
        observed.append((alternative, state[1][15].detach().clone(),
                         state[0][0].detach().clone(), float(result.parts.detach().sum())))
        if alternative:
            constraints.append((m._compose_fork.taken.clone(),
                                m._compose_forced_slots.detach().clone()))
        return result
    monkeypatch.setattr(WholePaths, 'finish', capture)
    backward = m._backward_training_loss
    updates = []
    def update(total, *args, **kwargs):
        updates.append(bool(m._sentence_backward))
        return backward(total, *args, **kwargs)
    monkeypatch.setattr(m, '_backward_training_loss', update)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            m.runBatch(train=True, batchSize=4, optimizer=optimizer, batch_override=batch)
        assert [row[0] for row in observed] == [False, True]
        assert updates == [True]
        assert all(torch.isfinite(torch.tensor(row[3])) for row in observed)
        # Both complete paths share one input tick. This numeric reader
        # produces no verbal sentence and no accepted thought deposit.
        assert not m.answer_synthesis
        assert m._sentence_ticks == {0: clock0 + 1}
        assert m.present() - clock0 == 1
        assert m._training_step_count - steps0 == 1
        assert len(commits) == 1
        chosen = torch.where(m._sentence_winners[0][:, None, None], observed[1][2], observed[0][2])
        selected, committed, depths = commits[0]
        torch.testing.assert_close(selected, chosen)
        for b, depth in enumerate(depths.tolist()):
            torch.testing.assert_close(committed[b, :depth], chosen[b, :depth])
        assert (m._reconstruction_stack()._choice_actions == -1).all()
        assert not m._reconstruction_stack()._choice_mask.any()
        audit=m._last_sentence_credit
        draw=audit['departure']
        narrowing=draw['narrowing']
        # One departure across the seven walks; later departures reuse
        # the already completed reading and its compose prefix.
        rounds = torch.stack([draw[name + '_round'] for name in
            ('attention', 'compose', 'candidate', 'bind', 'reconstruction', 'thought', 'output')], -1)
        assert rounds.ge(0).sum(-1).eq(draw['walk'].ge(0).long()).all()
        composed = draw['compose_round'] >= 0
        changed_compose=(observed[0][1] != observed[1][1]).any(-1)
        assert changed_compose[composed].all()
        a,b=audit['narrowing_actions']
        assert (a[narrowing]!=b[narrowing]).any(-1).all()
        taken, forced = constraints[0]
        assert torch.equal(forced.any(-1), composed)
        assert taken[composed].all()
        assert (observed[0][1][forced] != observed[1][1][forced]).all()
        assert m._exploration_trial is False
        assert not hasattr(m, '_compose_state_snapshot')
        assert m._compose_fork is None
    finally:
        m.End()
        m.symbolSpace.soft_reset()


@pytest.mark.usefixtures('eager_reading')
def test_evaluation_repeats_the_same_program_without_an_explore_trial(monkeypatch):
    m = _build('<composeTemperature>2</composeTemperature>')
    batch = _batch(m)
    original = m._run_batch_once
    from functools import wraps
    calls = []
    @wraps(original)
    def once(*args, **kwargs):
        calls.append((kwargs['train'], kwargs['exploration_trial']))
        return original(*args, **kwargs)
    monkeypatch.setattr(m, '_run_batch_once', once)
    def forbidden(*args, **kwargs):
        raise AssertionError('evaluation attempted a backward')
    monkeypatch.setattr(m, '_backward_training_loss', forbidden)
    # Let model time advance normally. Composition reads ages, so moving
    # the clock origin must not change this deterministic-choice probe.
    programs = []
    try:
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                m.runBatch(train=False, batchSize=4, batch_override=batch)
            programs.append(m._derivation_program()[1].detach().clone())
            m.End()
            m.symbolSpace.soft_reset()
        assert calls == [(False, False), (False, False)]
        assert torch.equal(programs[0], programs[1])
    finally:
        m.End()
        m.symbolSpace.soft_reset()
