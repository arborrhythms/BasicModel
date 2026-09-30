"""Both compose trials train; the strictly lower-loss end state survives."""
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


def test_pair_trains_twice_and_commits_once(monkeypatch):
    m = _build()
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
    backward = m._backward_training_loss
    def capture(total, *args, **kwargs):
        if getattr(m, '_sentence_backward', False):
            trace = m._reconstruction_stack()
            alternative = m._sentence_trial == 'explore'
            observed.append((alternative, trace._choice_actions.detach().clone(),
                             m.conceptualSpace.stm._buffer.detach().clone(),
                             float(total.detach())))
            if alternative:
                constraints.append((m._compose_prefix_slots.detach().clone(),
                                    m._compose_forced_slots.detach().clone()))
        return backward(total, *args, **kwargs)
    monkeypatch.setattr(m, '_backward_training_loss', capture)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            m.runBatch(train=True, batchSize=4, optimizer=optimizer, batch_override=batch)
        assert [row[0] for row in observed] == [False, True]
        assert all(torch.isfinite(torch.tensor(row[3])) for row in observed)
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
        assert (observed[0][1] != observed[1][1]).any(-1).all()
        prefix, forced = constraints[0]
        assert torch.equal(observed[0][1][prefix], observed[1][1][prefix])
        assert (observed[0][1][forced] != observed[1][1][forced]).all()
        assert m._exploration_trial is False
        assert not hasattr(m, '_compose_state_snapshot')
        assert m._compose_exploit_actions is None
    finally:
        m.End()
        m.symbolSpace.soft_reset()


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
    # The model normally stamps a new .when value on every public batch.
    # Identical words at different times are different numerical inputs.
    # Hold time fixed for this deterministic-choice probe; clock cadence is
    # checked separately above.
    monkeypatch.setattr(m, '_advance_when_time', lambda: None)
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


def test_flattened_temperature_driver_is_deleted():
    import Models
    assert not hasattr(Models.BasicModel, '_set_superposition_temperature')
