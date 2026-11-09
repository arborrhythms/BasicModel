"""Initial fixed graphs are allowed; later guard churn fails before fallback."""
import pytest
import torch
from torch._dynamo import config

from CompileGuard import guarded_compile, output_walk_phase, reconstruction_phase, percept_arrival_phase


@pytest.fixture(autouse=True)
def independent_dynamo_cache():
    torch._dynamo.reset()
    # ConfigModule settings require their own patch context: ordinary
    # monkeypatch teardown can hide a virtual setting instead of restoring it.
    with config.patch(recompile_limit=config.recompile_limit):
        yield
    torch._dynamo.reset()


def test_first_graph_reuses_then_shape_guard_fails_without_fallback():
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    def fixed_graph(x):
        return x.sin() + 1
    call = guarded_compile(fixed_graph, backend=backend, fullgraph=True, dynamic=False)
    x = torch.ones(2)
    torch.testing.assert_close(call(x), fixed_graph(x))
    torch.testing.assert_close(call(x + 1), fixed_graph(x + 1))
    assert len(graphs) == 1
    with pytest.raises(RuntimeError, match='Unexpected model recompilation.*fixed_graph') as error:
        call(torch.ones(3))
    assert 'guard failure' in str(error.value) and 'size mismatch' in str(error.value)
    assert len(graphs) == 1


def test_independent_wrappers_admit_initial_graphs_for_shared_code():
    def source(scale):
        def fixed_graph(x):
            return x * scale
        return fixed_graph
    first = guarded_compile(source(2), backend='eager', fullgraph=True, dynamic=False)
    second = guarded_compile(source(3), backend='eager', fullgraph=True, dynamic=False)
    x = torch.ones(2)
    for _ in range(2):
        torch.testing.assert_close(first(x), x * 2)
        torch.testing.assert_close(second(x), x * 3)


def test_train_eval_and_grad_phases_allow_initial_graphs_but_owner_drift_fails():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.))
            self.scale = 1
        def forward(self, x):
            return x * self.weight * self.scale + int(self.training)
    model = Model()
    call = guarded_compile(model, backend='eager', fullgraph=True, dynamic=False)
    assert isinstance(call, torch.nn.Module)
    assert set(call.state_dict()) == {'_orig_mod.weight'}
    x = torch.ones(2)
    for training in (True, False, True):
        call.train(training)
        for grad in (True, False, True):
            with torch.set_grad_enabled(grad):
                torch.testing.assert_close(call(x), model(x))
                torch.testing.assert_close(call(x), model(x))
    with torch.inference_mode():
        torch.testing.assert_close(call(x), model(x))
        torch.testing.assert_close(call(x), model(x))
    model.scale = 2
    before = (config.error_on_recompile, config.suppress_errors, config.fail_on_recompile_limit_hit)
    with pytest.raises(RuntimeError, match='guard failure') as error:
        call(x)
    assert 'scale' in str(error.value)
    assert (config.error_on_recompile, config.suppress_errors, config.fail_on_recompile_limit_hit) == before


@pytest.mark.parametrize('decoder', ['output', 'reconstruction'])
def test_declared_decoder_modes_are_initial_phases_not_shape_exceptions(decoder):
    if decoder == 'output':
        def walk(x, *, return_candidates=False, exploit_actions=None, kept_actions=None):
            return x + (1 if return_candidates else 2 if exploit_actions is not None else 3)
        phase = output_walk_phase
        modes = [dict(return_candidates=True), dict(exploit_actions=torch.zeros(2)),
                 dict(kept_actions=torch.zeros(2))]
    else:
        def walk(x, *, decoder_candidates=False, decoder_exploit=None, decoder_kept_actions=None):
            return x + (1 if decoder_candidates else 2 if decoder_exploit is not None else 3)
        phase = reconstruction_phase
        modes = [dict(decoder_candidates=True), dict(decoder_exploit=torch.zeros(2)),
                 dict(decoder_kept_actions=torch.zeros(2))]
    call = guarded_compile(walk, initial_phase=phase, backend='eager', fullgraph=True, dynamic=False)
    for _ in range(2):
        for mode in modes:
            torch.testing.assert_close(call(torch.ones(2), **mode), walk(torch.ones(2), **mode))
    with pytest.raises(RuntimeError, match='size mismatch'):
        call(torch.ones(3), **modes[0])


def test_partial_graph_guard_failure_raises_even_if_outer_policy_suppresses_errors():
    def graph_with_break(x, scale):
        value = x.sin()
        torch._dynamo.graph_break()
        return value * scale
    call = guarded_compile(graph_with_break, backend='eager', fullgraph=False, dynamic=False)
    x = torch.ones(2)
    with config.patch(suppress_errors=True):
        torch.testing.assert_close(call(x, 2), graph_with_break(x, 2))
        torch.testing.assert_close(call(x, 2), graph_with_break(x, 2))
        with pytest.raises(RuntimeError, match='guard failure'):
            call(x, 3)
        assert config.suppress_errors is True


def test_nested_fullgraph_entry_does_not_trace_the_guard_configuration():
    def word_brick(x):
        return x.sin()
    word = guarded_compile(word_brick, backend='eager', fullgraph=True, dynamic=False)
    def model_entry(x):
        return word(x) + 1
    model = guarded_compile(model_entry, backend='eager', fullgraph=True, dynamic=False)
    x = torch.ones(2)
    torch.testing.assert_close(model(x), word_brick(x) + 1)
    torch.testing.assert_close(model(x), word_brick(x) + 1)
    with pytest.raises(RuntimeError, match='size mismatch'):
        model(torch.ones(3))


def test_eager_mode_never_invokes_torch_compile(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('eager entry attempted compilation')
    monkeypatch.setattr(torch, 'compile', fail)
    def direct(x):
        return x + 1
    assert guarded_compile(direct, backend='none') is direct


def test_recompile_limit_hit_names_callable_and_phase():
    def limited_entry(x):
        return x.sin()
    call = guarded_compile(limited_entry, backend='eager', fullgraph=True, dynamic=False)
    with config.patch(recompile_limit=0):
        with pytest.raises(RuntimeError, match='Model compilation cache limit recompile_limit=0.*limited_entry.*declared phase'):
            call(torch.ones(2))


def test_independent_initial_graphs_name_accumulated_capacity_limit():
    def source(count):
        class Independent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offsets = tuple(range(count))
            def forward(self, x):
                for offset in self.offsets:
                    x = x + offset
                return x.sin()
        return Independent()
    # Three independent initial graphs, never a second call to any wrapper.
    # A small capacity makes the same per-code-object limit observable cheaply.
    calls = [guarded_compile(source(count), backend='eager', fullgraph=True, dynamic=False)
             for count in (1, 2, 3)]
    with config.patch(recompile_limit=64, accumulated_recompile_limit=2):
        calls[0](torch.ones(2))
        calls[1](torch.ones(2))
        with pytest.raises(RuntimeError, match='Model compilation cache limit accumulated_recompile_limit=2.*Independent.*declared phase'):
            calls[2](torch.ones(2))


def test_compose_closing_is_a_declared_phase_without_admitting_shape_changes():
    from CompileGuard import compose_choice_phase
    def choice(x, *, closing=False):
        return x + (2 if closing else 1)
    call = guarded_compile(choice, initial_phase=compose_choice_phase,
                           backend='eager', fullgraph=True, dynamic=False)
    for closing in (False, True, False, True):
        torch.testing.assert_close(call(torch.ones(2), closing=closing), choice(torch.ones(2), closing=closing))
    with pytest.raises(RuntimeError, match='Unexpected model recompilation.*choice') as error:
        call(torch.ones(3), closing=True)
    assert 'size mismatch' in str(error.value)


def test_carried_state_requires_a_declared_phase_and_only_two_initial_graphs():
    from CompileGuard import carried_state_phase
    def word_brick(word, gate, index, live, state):
        return word + state[0] * state[1]
    word = torch.ones(2)
    fresh = (torch.ones(2), torch.full((2,), 2.))
    carried = (torch.ones(2, requires_grad=True), torch.full((2,), 2.))
    third = (torch.ones(2, requires_grad=True), torch.full((2,), 2., requires_grad=True))
    args = (word, None, None, None)
    undeclared = guarded_compile(word_brick, backend='eager', fullgraph=True, dynamic=False)
    undeclared(*args, fresh)
    with pytest.raises(RuntimeError, match='Unexpected model recompilation.*word_brick'):
        undeclared(*args, carried)
    torch._dynamo.reset()
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    declared = guarded_compile(word_brick, initial_phase=carried_state_phase,
                               backend=backend, fullgraph=True, dynamic=False)
    for state in (fresh, carried, fresh, carried):
        torch.testing.assert_close(declared(*args, state), word_brick(*args, state))
    assert len(graphs) == 2
    assert not carried_state_phase((*args, fresh), {})
    assert carried_state_phase((*args, carried), {})
    with pytest.raises(RuntimeError, match='Unexpected model recompilation.*word_brick'):
        declared(*args, third)
    assert len(graphs) == 2


def test_whole_sentence_stm_tensor_write_does_not_guard_or_update_host_depth():
    from types import SimpleNamespace
    from Memory import ShortTermMemory
    from Spaces import ConceptualSpace
    owner = SimpleNamespace(stm=ShortTermMemory(batch=1, capacity=4, concept_dim=3))
    def write_slots(slab):
        ConceptualSpace._stm_set_all_slots(owner, slab)
        return owner.stm._buffer.clone()
    call = guarded_compile(write_slots, backend='eager', fullgraph=True, dynamic=False)
    slab = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    result = call(slab)
    assert owner.stm._max_depth_host == 0
    assert owner.stm._depth.tolist() == [2]
    torch.testing.assert_close(result[:, :2], slab.flip(1))
    assert not result[:, 2:].any()
    owner.stm._max_depth_host = 9
    torch.testing.assert_close(call(slab + 1)[:, :2], (slab + 1).flip(1))
    assert owner.stm._max_depth_host == 9


def test_output_semantic_banks_are_declared_but_their_widths_are_fixed():
    def walk(x, *, basis=None, constituents=None, case_bank=None, whole_bank=None):
        return x + sum(value.sum() for value in (basis, constituents, case_bank, whole_bank)
                       if value is not None)
    call = guarded_compile(walk, initial_phase=output_walk_phase,
                           backend='eager', fullgraph=True, dynamic=False)
    x = torch.ones(2)
    for _ in range(2):
        for banks in ({}, dict(basis=x), dict(basis=x, constituents=x, case_bank=x, whole_bank=x)):
            torch.testing.assert_close(call(x, **banks), walk(x, **banks))
    with pytest.raises(RuntimeError, match='Unexpected model recompilation.*walk'):
        call(x, basis=torch.ones(3))


def test_ownerless_percept_arrival_declares_training_and_keeps_shape_fixed():
    from Expectation import percept_arrival_brick
    inputs = (torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, dtype=torch.long),
              torch.zeros(1, 2), torch.zeros(2), torch.zeros(2), torch.zeros((), dtype=torch.long),
              torch.ones(1, 2), torch.full((1, 2), .5), torch.ones(1, dtype=torch.bool),
              torch.ones(1, 2), .05)
    call = guarded_compile(percept_arrival_brick, initial_phase=percept_arrival_phase,
                           backend='eager', fullgraph=True, dynamic=False)
    for training in (True, False, True, False):
        actual = call(*inputs, training)
        expected = percept_arrival_brick(*inputs, training)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_output_walk_construction_failure_propagates(monkeypatch):
    from Models import BasicModel
    from types import SimpleNamespace
    import CompileGuard
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'eager')
    def broken(*args, **kwargs):
        raise ValueError('invalid compiled walk construction')
    monkeypatch.setattr(CompileGuard, 'guarded_compile', broken)
    owner = SimpleNamespace(_output_generate_walk=lambda x: x)
    with pytest.raises(ValueError, match='invalid compiled walk construction'):
        BasicModel._compiled_output_walk(owner)
    assert '_output_walk_compiled' not in owner.__dict__


def test_native_primed_bank_keeps_constructed_width_across_sentence_occupancy(tmp_path, monkeypatch):
    from test_native_attention_field import _model
    model = _model(tmp_path, monkeypatch)
    native = model._prime_sentence_symbols
    banks = []
    def stage(sid):
        native(sid)
        bank = model._sentence_primed_bank
        if bank is not None:
            assert bank.bytes.dtype == torch.uint8
            assert bank.bytes.shape[-1] == model.word_surface_capacity + 1
            banks.append((bank.codes.detach(), bank.valid.detach(), bank.bytes.shape[-1]))
    monkeypatch.setattr(model, '_prime_sentence_symbols', stage)
    try:
        for text in ('alpha', 'alpha beta'):
            inputs = model.inputSpace.prepInput([text])
            model.runBatch(train=False, batchSize=1,
                           batch_override=(inputs, torch.empty(1, 0)))
        assert len(banks) >= 2
        assert len({values.shape[1] for values, _, _ in banks}) == 1
        assert len({width for _, _, width in banks}) == 1
        def read(values, valid):
            return torch.where(valid[..., None], values, 0.).sum(1)
        call = guarded_compile(read, backend='eager', fullgraph=True, dynamic=False)
        for values, valid, _ in banks:
            torch.testing.assert_close(call(values, valid), read(values, valid))
    finally:
        model.End()


def test_util_and_word_brick_keep_late_recompile_errors_out_of_fallback(monkeypatch):
    import SentenceCompose
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'eager')
    monkeypatch.setattr(util, 'TheCompileMode', 'default')
    def util_entry(x, scale):
        return x * scale
    def word_entry(x, scale):
        return x + scale
    calls = [util.compile(util_entry, verbose=False, fullgraph=True),
             SentenceCompose.compile_word_brick(word_entry)]
    for call in calls:
        call(torch.ones(2), 1)
        with pytest.raises(RuntimeError, match='guard failure'):
            call(torch.ones(2), 2)


@pytest.mark.parametrize('entry', ['reconstruction', 'output_walk'])
@pytest.mark.parametrize('backend', ['eager', 'none'])
def test_cached_model_factories_guard_compiled_calls_and_leave_eager_direct(monkeypatch, entry, backend):
    from types import SimpleNamespace
    from Models import BasicModel
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', backend)
    def numerical_step(x):
        return x.sin() * 2
    if entry == 'reconstruction':
        owner = SimpleNamespace(_reconstruct_sentences=numerical_step,
                                _recon_placement=lambda: 'compiled')
        factory = BasicModel._compiled_reconstruct
    else:
        owner = SimpleNamespace(_output_generate_walk=numerical_step)
        factory = BasicModel._compiled_output_walk
    if backend == 'none':
        def forbidden(*args, **kwargs):
            raise AssertionError('eager factory attempted compilation')
        monkeypatch.setattr(torch, 'compile', forbidden)
    call = factory(owner)
    assert factory(owner) is call
    x = torch.ones(2)
    torch.testing.assert_close(call(x), numerical_step(x))
    torch.testing.assert_close(call(x), numerical_step(x))
    if backend == 'none':
        assert call is numerical_step
        torch.testing.assert_close(call(torch.ones(3)), numerical_step(torch.ones(3)))
    else:
        with pytest.raises(RuntimeError, match='size mismatch') as error:
            call(torch.ones(3))
        assert 'numerical_step' in str(error.value) and 'guard failure' in str(error.value)
