"""Rejected inverse images and discarded walk residuals have no pullback."""
from types import SimpleNamespace

import pytest
import torch


def conditioned_root(dtype, *, form_value=.5, for_value=.5, against_value=0., symbolic=False):
    from Models import BasicModel

    class Reader(torch.nn.Module):
        _question_conditioner = BasicModel._question_conditioner
        _question_conditioner_modules = BasicModel._question_conditioner_modules
        _selected_thought_chooser_modules = BasicModel._selected_thought_chooser_modules
        _collect_fresh_synthesis_modules = BasicModel._collect_fresh_synthesis_modules

        def objective_parameter_groups(self, optimizer):
            return {'output': tuple(self.question_conditioners.parameters())}

        def condition(self, idea, context):
            return BasicModel._condition_answer_on_question(
                self, idea[:, None], context, identity=symbolic)[:, 0]

    model = Reader()
    model._question_conditioner(392, device=torch.device('cpu'), dtype=dtype)
    idea = torch.full((1, 392), form_value, dtype=dtype, device='cpu')
    idea[:, 328:360], idea[:, 360:] = for_value, against_value
    root = model.condition(idea, idea.new_ones(1, 29))
    return model.question_conditioners['392'].weight, root


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_rejected_inverse_cannot_write_one_conditioner_lane(dtype, monkeypatch):
    from Generative import InverseMenu
    from Models import BasicModel
    import Generative
    import util

    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    weight, root = conditioned_root(dtype)
    # The alternative has a finite image but a singular native derivative
    # on the empty against lane. STOP is the only admissible image.
    def menu(_language, parent, live, **options):
        rejected = torch.cat((parent[:, :360], parent[:, 360:].sqrt()), -1)
        left = torch.stack((rejected, parent), 1)
        legal = torch.stack((torch.zeros_like(live), live), 1)
        zero = parent.new_zeros(legal.shape)
        return InverseMenu(left, left, legal, legal, zero,
                           parent.new_zeros(*legal.shape, 2), zero)
    monkeypatch.setattr(Generative, 'inverse_menu', menu)
    op = SimpleNamespace(meaning_layout=(328, 392))
    language = SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=(op,),
        _generate_policy_width=392, choose_generate=lambda logits: logits.argmax(-1),
        generate_policy_logits=lambda parent, features: parent.new_zeros(len(parent), 2))
    result = BasicModel._output_generate_walk(SimpleNamespace(languageSpace=language),
        root[:, None], 1, return_trace=True)
    assert result[4].tolist() == [[1]]
    loss = result[0].sum() + result[3].sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(weight.grad).all()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_losing_output_residual_has_no_answer_exposure(dtype):
    from Layers import Error
    from WholePaths import objectives
    weight, root = conditioned_root(dtype)
    registries = []
    for departed in (False, True):
        registry = Error(row_mask=torch.ones(1,dtype=torch.bool))
        lane = root[:,360:].sqrt().mean(-1) if departed else root[:,360:].mean(-1)
        registry.add('output.walk_residual.against', lane, objective='reconstruction', side='A')
        registry.add('answer', root[:,0] + departed, objective='output')
        registries.append(SimpleNamespace(registry=registry))
    parts = torch.tensor([[[0.,0.,0.],[0.,0.,1.]]],dtype=dtype)
    costs, weights = objectives(registries, parts, torch.ones(1,dtype=torch.bool),
        [dict(compose_round=torch.tensor([-1]))])
    assert weights.tolist() == [[1.,0.]]
    sum(costs.values()).backward()
    assert torch.isfinite(weight.grad).all()
    torch.testing.assert_close(weight.grad[360:], torch.full_like(weight.grad[360:],1/32))


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_unary_chosen_by_another_stream_does_not_differentiate_a_rejected_operand(dtype):
    from Generative import selected_inverse
    from Language import LanguageSpace

    class OneUnary:
        _generate_binary_ops = ()
        _generate_unary_ops = (SimpleNamespace(generate=torch.sqrt),)
        generate_unary_step = LanguageSpace.generate_unary_step

    parent = torch.tensor([[.25], [0.]], dtype=dtype, requires_grad=True)
    choice = torch.tensor([0, 1])
    child, _ = selected_inverse(OneUnary(), parent, choice, torch.ones(2, dtype=torch.bool),
                                basis=None, basis_valid=None, require_symbols=False)
    torch.testing.assert_close(child, parent.new_tensor([[.5], [0.]]))
    child.sum().backward()
    torch.testing.assert_close(parent.grad, torch.ones_like(parent))


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_native_sum_inverse_has_finite_pullback_on_nearly_collinear_codes(dtype):
    from Language import LanguageSpace, SumLayer
    operator = SumLayer()
    operator.meaning_layout = (328, 392)
    weight, parent = conditioned_root(dtype, form_value=.25, for_value=0., against_value=.25, symbolic=True)
    # The same binary32 epsilon is used in both charts. The second recovery
    # selected by the native inverse has a squared support of 1.615587e-27.
    left, right = torch.full_like(parent, .1), torch.full_like(parent, .4)
    left[:, 328:], right[:, 328:] = 0., 0.
    left[:, 328:360], right[:, 328:360] = torch.finfo(torch.float32).eps, 1.
    valid = torch.ones(1, 1, dtype=torch.bool)
    flag = torch.zeros(1, dtype=torch.bool)
    a, b, available = LanguageSpace._bounded_binary_reconstruction(
        operator, parent, torch.zeros_like(parent), flag, flag, left[:, None], valid, 1,
        left_basis=left[:, None], right_basis=right[:, None], left_valid=valid, right_valid=valid)
    assert available.all() and torch.isfinite(a).all() and torch.isfinite(b).all()
    chosen = parent[:, :328].mean()
    loss = chosen + (a.sum() + b.sum()) * 0.
    torch.testing.assert_close(loss, chosen, rtol=0, atol=0)
    for objective in (loss, a.sum() + b.sum()):
        gradient, = torch.autograd.grad(objective, weight, retain_graph=True)
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize('streams', [1, 2])
def test_report_keeps_only_the_selected_whole_path_realization(streams):
    from Output import AnswerConstruction
    from WholePaths import keep_output
    values = torch.nn.Parameter(torch.arange(1., streams+1))
    def output(x):
        return AnswerConstruction(x[:,None], None, x[:,None], x[:,None], x[:,None], (), tuple(map(str,x.tolist())))
    paths = [SimpleNamespace(construction=output(values), output_trace=(torch.zeros(streams,1,dtype=torch.long),)),
             SimpleNamespace(construction=output(values*2), output_trace=(torch.ones(streams,1,dtype=torch.long),))]
    model = SimpleNamespace()
    keep_output(model, paths, torch.tensor([1,0])[:streams], 0)
    torch.testing.assert_close(model._kept_path_output.actual[:,0], torch.tensor([2.,2.])[:streams])
    assert model._kept_path_output_actions[:,0].tolist() == [1,0][:streams]
    assert not model._kept_path_output.actual.requires_grad


@pytest.mark.parametrize('streams', [1, 2])
def test_only_the_kept_sentence_understanding_has_a_pullback(streams):
    from SentenceUnderstanding import SentenceUnderstanding
    values = torch.nn.Parameter(torch.arange(1., streams + 1))
    loser = torch.nn.Parameter(torch.zeros(streams))
    snapshot = object()
    def record(field):
        x = field[:, None]
        rows = torch.arange(streams)[:, None]
        return SentenceUnderstanding(x, x[:, None], rows[:, 0], x[:, None], rows,
            x[:, None], rows, rows >= 0, snapshot, torch.tensor(0))
    greedy = record(torch.cat((loser[:1].sqrt(), values[1:])))
    explore = record(2 * values if streams == 1 else torch.cat((2 * values[:1], loser[1:].sqrt())))
    wins = torch.arange(streams) == 0
    def replay():
        assert streams == 2
        return record(torch.stack((2 * values[0], values[1])))
    kept = SentenceUnderstanding.select(greedy, explore, wins, live=True, replay=replay)
    kept.root.sum().backward()
    assert loser.grad is None
    torch.testing.assert_close(values.grad, torch.tensor([2., 1.])[:streams])


def test_native_mixed_sentence_keeps_one_live_composition_per_stream(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from SentenceUnderstanding import SentenceUnderstanding
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
                                  training_overrides={'reconstructInLoop': False})
    model.separate_chooser = False
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    try:
        optimizer = model.getOptimizer(lr=1e-4)
        score = model._sentence_path_cost
        def scored(*args, **kwargs):
            result = score(*args, **kwargs)
            # Keep opposite streams from the actual two native composition trials.
            registry = model._sentence_cost_registry
            ordinary = registry.total
            explore = model._sentence_trial == 'explore'
            def separated(*args, **kwargs):
                total = ordinary(*args, **kwargs)
                if kwargs.get('objective') == 'reconstruction':
                    fixed = total.new_tensor([0., 20.] if explore else [20., 0.])
                    return total - total.detach() + fixed
                return total
            registry.total = separated
            return result
        monkeypatch.setattr(model, '_sentence_path_cost', scored)
        observed = []
        select = SentenceUnderstanding.select.__func__
        def checked(cls, left, right, wins, **kwargs):
            assert wins.tolist() == [True, False] and kwargs['live']
            kept = select(cls, left, right, wins, **kwargs)
            assert kept.root.requires_grad
            observed.append(kept.root.detach().clone())
            return kept
        monkeypatch.setattr(SentenceUnderstanding, 'select', classmethod(checked))
        raw = model.inputSpace.prepInput(['a b', 'c d'])
        model.runBatch(train=True, optimizer=optimizer, batchSize=2, split='runtime',
                       batch_override=(raw, torch.empty(2, 0)))
        assert len(observed) == 1
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
