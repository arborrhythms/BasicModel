"""Exact score ties use grammar structure; unequal scores still decide."""
import pytest
import torch
from torch import nn

from Language import BinaryStructuredReductionLayer, SelectedThoughtChooser
from Models import StaticPeerPipeline
from GrammarPreference import structural_argmax


class _Candidate(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.routing_kind = kind

    def forward(self, left, right):
        return (left + right) / 2


def test_binary_equal_fit_prefers_structural_face_in_the_ordinary_mlp():
    layer = BinaryStructuredReductionLayer(d_model=4, chooser='mlp',
        ops=[_Candidate('opaque'), _Candidate('structural')])
    with torch.no_grad():
        for parameter in layer.chooser.parameters():
            parameter.zero_()
    _, _, routing = layer(torch.randn(2, 2, 4))
    assert routing['reduce_mask'][:, 0, 1].tolist() == [1., 1.]
    assert routing['reduce_mask'][:, 0, 0].tolist() == [0., 0.]


def test_thought_equal_fit_uses_the_same_mlp_and_keeps_live_credit():
    chooser = SelectedThoughtChooser(context_dim=4)
    index, logp = chooser.choose(torch.randn(2, 4), [False, False],
                                structural=(False, True))
    assert index == 1
    assert logp.requires_grad


def test_six_word_legacy_pipeline_retains_every_stage_and_feedback_delay():
    seen = []
    def stage_b(value, index, feedback):
        seen.append((index, feedback))
        return value
    trace = StaticPeerPipeline(6).run(list(range(6)), lambda word, i: word,
        stage_b, lambda word, i: ('grammar', i))
    assert [p for stage, p in trace if stage == 'C'] == list(range(6))
    assert seen == [(0, None), (1, None)] + [
        (i, ('grammar', i - 2)) for i in range(2, 6)]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_preference_never_overturns_a_better_score_or_unmasks_a_rule(dtype):
    scores = torch.tensor([[2, 2], [3, 2], [-2, -2], [0, -torch.inf]], dtype=dtype)
    assert structural_argmax(scores, (False, True)).tolist() == [1, 0, 1, 0]
    assert structural_argmax(scores, (True, True)).tolist() == scores.argmax(-1).tolist()
    # The smallest representable improvement still wins; no epsilon window.
    close = torch.ones(2, dtype=dtype)
    close[0] = torch.nextafter(close[0], torch.tensor(torch.inf, dtype=dtype))
    assert int(structural_argmax(close, (False, True))) == 0


def test_opaque_candidates_cannot_select_an_anchor_only_scorer():
    with pytest.raises(ValueError, match='ordinary mlp'):
        BinaryStructuredReductionLayer(d_model=4, chooser='anchordot',
            ops=[_Candidate('opaque'), _Candidate('structural')])


def test_unary_and_generate_use_the_same_exact_tie_order():
    from Language import UnaryStructuredLayer, LanguageSpace
    from types import SimpleNamespace as NS
    class Unary(_Candidate):
        def forward(self, x):
            return x
    ops = [Unary('opaque'), Unary('structural')]
    layer = UnaryStructuredLayer(d_model=4, chooser='mlp', ops=ops)
    with torch.no_grad():
        for p in layer.chooser.parameters():
            p.zero_()
        # Real MLP: the copy tool scores below the two exactly tied operators.
        layer.chooser.tool_embedding[0, 0] = -1
        layer.chooser.mlp[0].weight[0, 8] = 1
        layer.chooser.mlp[-1].weight[0, 0] = 1
    _, _, routing = layer(torch.randn(2, 1, 4))
    assert routing['action_op'].reshape(-1).tolist() == [1, 1]
    owner = NS(_generate_binary_ops=(), _generate_unary_ops=ops)
    scores = torch.tensor([[2., 2., -1.], [3., 2., -1.]])
    assert LanguageSpace.choose_generate(owner, scores).tolist() == [1, 0]


def test_preference_compiles_fullgraph_without_host_reads():
    def choose(scores):
        return structural_argmax(scores, (False, True, False))
    scores = torch.tensor([[1., 1., 0.], [0., 2., 3.]])
    captured = torch.compile(choose, backend='eager', fullgraph=True)
    assert captured(scores).tolist() == [1, 2]


def test_tie_preference_preserves_all_candidate_gradients_and_parameter_layout():
    layer = BinaryStructuredReductionLayer(d_model=4, chooser='mlp',
        ops=[_Candidate('opaque'), _Candidate('structural')])
    keys = set(layer.state_dict())
    with torch.no_grad():
        layer.chooser.mlp[-1].weight.zero_()
        layer.chooser.mlp[-1].bias.zero_()
    x = torch.randn(2, 2, 4)
    _, _, routing = layer(x)
    scores = routing['reduce_score']
    scores.retain_grad()
    loss = -scores.log_softmax(-1)[..., 1].sum()
    loss.backward()
    assert bool((scores.grad[..., 0] > 0).all())
    assert bool((scores.grad[..., 1] < 0).all())
    assert set(layer.state_dict()) == keys


def test_coverage_counts_only_committed_rules_and_refuses_missing_evidence():
    from types import SimpleNamespace as NS
    from GrammarEvidence import RoutingCoverage
    language = NS(_compose_binary_rules=[NS(method_name='blackbox'), NS(method_name='part')],
                  _compose_unary_rules=[], _tree_layer=lambda arity:
                  NS(ops=[_Candidate('opaque'), _Candidate('structural')]) if arity == 2 else None)
    coverage = RoutingCoverage(language, corpus='frozen-corpus-sha256', stage='base')
    program = NS(leaves=torch.randn(2, 4), actions=torch.tensor([
        [0, -1, 0], [0, -1, 1], [1, 1, -1], [-1, -1, -1]]))
    coverage.record(program, sentence_id='first')
    program.actions[2, 1] = 0
    coverage.record(program, sentence_id='second')
    report = coverage.report()
    assert report['sentences'] == report['operations'] == 2
    assert report['sentence_opaque_share'] == report['operation_opaque_share'] == .5
    assert report['learned_utility'] == 'unproven'
    with pytest.raises(ValueError, match='missing/empty'):
        coverage.record(None, sentence_id='missing')
    with pytest.raises(ValueError, match='unique'):
        coverage.record(program, sentence_id='first')
    program.actions[2, 1] = 42
    with pytest.raises(ValueError, match='unknown committed'):
        coverage.record(program, sentence_id='unknown')


def test_coverage_comparison_reuses_maturity_and_matches_corpus_before_quality():
    from copy import deepcopy
    from GrammarEvidence import compare_coverage
    base = dict(corpus='same', catalog={1: [dict(name='part', structural=True),
        dict(name='opaque', structural=False)]}, rows=[dict(sentence_id='a', opaque_operations=1)])
    later = deepcopy(base)
    later['catalog'][1].append(dict(name='whole', structural=True))
    later['rows'][0]['opaque_operations'] = 0
    report = compare_coverage([base, later], [{}, {}])
    assert report['status'] == 'skipped'
    assert report['learned_utility'] == 'unproven'
    later['corpus'] = 'changed'
    with pytest.raises(ValueError, match='same ordered corpus'):
        compare_coverage([base, later], [{}, {}])


@pytest.mark.parametrize('path', ['functional', 'trace'])
@pytest.mark.parametrize('flags, expected', [
    ((False, True), [0, 1, 1]),
    ((True, True), [0, 0, 1]),
    ((False, False), [0, 0, 1]),
])
def test_committed_marginal_controls_selection_for_every_catalog(path, flags, expected):
    from types import SimpleNamespace as NS
    from Language import _FunctionalLanguageChooser, ReconstructionStack
    from Models import BasicModel

    class CommittedReducer(nn.Module):
        structural_ops = flags

        def forward(self, window):
            parent = window.sum(dim=1, keepdim=True)
            routing = dict(chosen_reduced=parent,
                copy_score=window.new_zeros(3, 2, 1),
                # Adapters may publish a committed marginal distinct from
                # their scoring workspace. Catalog kind must not switch it.
                reduce_score=window.new_tensor([[[-1, 1]], [[1, -1]], [[1, -1]]]),
                reduce_marginal_op=window.new_tensor([[[.8, .2]], [[.5, .5]], [[.2, .8]]]),
                action_kind=torch.ones(3, 1, dtype=torch.long),
                src_left=torch.full((3, 1), -1, dtype=torch.long))
            return parent, parent, routing

    reducer = CommittedReducer()
    window = torch.ones(3, 2, 4)
    if path == 'functional':
        state = (window, torch.full((3,), 2), None, None, None, None)
        choice = _FunctionalLanguageChooser.choose_binary(
            state, reducer, torch.ones(3, dtype=torch.bool), base_tau=.75)
        assert choice[2].tolist() == expected
    else:
        trace = ReconstructionStack(batch=3, max_depth=4)
        trace.prepare_choices(3, 1, device='cpu', binary_rule_ids=(17, 23))
        owner = NS(_reconstruction_stack=lambda: trace, _stm_reducer=lambda: reducer)
        routing = reducer(window)[2]
        BasicModel._record_reconstruction_choice(owner, 0, routing,
            torch.ones(3, dtype=torch.bool), arity=2)
        ids, arities, active = trace.choices()
        assert ids[:, 0].tolist() == [[17, 23][i] for i in expected]
        assert arities[:, 0].tolist() == [2, 2, 2]
        assert active[:, 0].tolist() == [True, True, True]
