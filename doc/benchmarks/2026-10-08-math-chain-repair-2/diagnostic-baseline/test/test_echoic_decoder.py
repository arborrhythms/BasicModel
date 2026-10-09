"""The conceptual generate decoder is shared, reconstruction-owned and journal-free."""
from dataclasses import fields
from DecompositionChooser import DecompositionChooser
from types import SimpleNamespace, MethodType

import torch
import pytest


def test_answer_record_has_no_derivation_fields():
    from SentenceUnderstanding import SentenceUnderstanding
    assert not ({'rule_ids', 'arities', 'rule_valid', 'operand_positions',
                 'journal_columns', 'witness_offsets'} & {f.name for f in fields(SentenceUnderstanding)})




def test_generate_unary_calls_generate_face():
    from Language import LanguageSpace
    class Unary:
        def generate(self, value):
            return -value
        def reverse(self, value):
            raise AssertionError('decoder must use the generate face')
    language = SimpleNamespace(_generate_unary_ops=[Unary()])
    x = torch.tensor([[1., 2.], [3., 4.]])
    y, unavailable = LanguageSpace.generate_unary_step(language, x, torch.zeros(2, dtype=torch.long),
        torch.tensor([True, False]), return_status=True)
    torch.testing.assert_close(y, torch.tensor([[-1., -2.], [3., 4.]]))
    assert not unavailable.any()


def test_generate_chooser_belongs_to_reconstruction():
    from Models import BasicModel
    policy = torch.nn.Linear(2, 3)
    reader = torch.nn.Linear(2, 1)
    language = SimpleNamespace(generate_policy=policy)
    model = SimpleNamespace(languageSpace=language, outputSpace=reader,
        synthesis_parameters=lambda: tuple(policy.parameters()), conceptualSpaces=())
    optimizer = SimpleNamespace(param_groups=[{'params': [*policy.parameters(), *reader.parameters()]}])
    owners = BasicModel.objective_parameter_groups(model, optimizer)
    assert {id(p) for p in policy.parameters()} <= {id(p) for p in owners['reconstruction']}
    assert {id(p) for p in reader.parameters()} <= {id(p) for p in owners['output']}
    assert not {id(p) for p in owners['reconstruction']} & {id(p) for p in owners['output']}


def test_sentence_cost_is_not_diluted_by_other_packed_sentences(monkeypatch):
    import Models, util
    from SentenceUnderstanding import PrimedSymbols, SentenceUnderstanding
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    root = torch.tensor([[1., 2.]])
    reference = root[:, None].expand(1, 4, 2)
    bank = PrimedSymbols(torch.tensor([[0]]), root[:, None], torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool), torch.tensor([[[65]]]), torch.ones(1, 1, 1, dtype=torch.bool))
    end = torch.cat((root[:, None], root.new_zeros(1, 2, 2)), 1)
    record = SentenceUnderstanding(root, end, torch.ones(1, dtype=torch.long),
        end.flatten(1)[:, None].expand(1, 2, 6), torch.ones(1, 2, dtype=torch.long),
        reference, torch.zeros(1, 4, dtype=torch.long), torch.tensor([[True, True, False, False]]), bank, torch.tensor(0))
    owner = SimpleNamespace(inputSpace=SimpleNamespace(_word_active_mask=torch.ones(1, 4, dtype=torch.bool),
        _packed_sentence_ids=torch.tensor([[0, 0, 1, 1]])),
        _byte_tables=lambda B,W: (True, torch.ones(B,W,1,dtype=torch.long), torch.ones(B,W,1,dtype=torch.bool)),
        _byte_word_cost=lambda *a, **kw: torch.tensor([2.]),
        _decode_conceptual_sentence=lambda *a: (reference, torch.tensor([2]), torch.tensor([False]), root.sum(-1)*0, torch.zeros(1,4,dtype=torch.long)))
    result = Models.BasicModel._reconstruct_sentences(owner, root, reference, record.roots,
        record.depths, end, record.end_depth, record.sentence, understanding=record)
    torch.testing.assert_close(result[2], torch.tensor([2.]))
    torch.testing.assert_close(result[4], torch.tensor([[2., 0.]]))


def test_output_spelling_uses_the_echoic_shortlist():
    from Models import BasicModel
    from SentenceUnderstanding import PrimedSymbols
    bank = PrimedSymbols(torch.tensor([[8, 9]]), torch.tensor([[[1., 0.], [1., 1.]]]),
        torch.tensor([[.1, 2.]]), torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[[65], [66]]]), torch.ones(1, 2, 1, dtype=torch.bool))
    owner = SimpleNamespace(_concept_owner=lambda: SimpleNamespace(word_surface_for_row=lambda row: {8:b'A',9:b'B'}[row]))
    words = torch.tensor([[[1., 0.]]])
    assert BasicModel._generated_word_text(owner, words, torch.tensor([1]), bank=bank) == ('B',)




@pytest.mark.parametrize('compiled', [False, True])
def test_same_walk_infers_binary_and_unary_without_a_journal(monkeypatch, compiled):
    import Models, util
    from Language import LanguageSpace, ConjunctionLayer, GrammarLayer
    class Reflection(GrammarLayer):
        inverse_kind = "unary"
        invertible = True
        def forward(self, x): return -x
        def reverse(self, x): return -x
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    conjunction = ConjunctionLayer()
    policy = torch.nn.Linear(2, 3)
    with torch.no_grad():
        policy.weight.copy_(torch.tensor([[1., 1.], [-1., -1.], [0., 0.]]))
        # The composed root has unit activation, independent of form norms.
        policy.bias.copy_(torch.tensor([-1., -1., 0.]))
    language = SimpleNamespace(_generate_binary_ops=(conjunction,), _generate_unary_ops=(Reflection(0, 0),),
        generate_policy=policy, _generate_policy_width=2,
        decomposition_chooser=DecompositionChooser())
    for name in ('reverse_binary_step', '_reverse_of_binary_op', '_finish_binary_inverse',
                 'generate_unary_step', 'generate_policy_logits', 'choose_generate'):
        setattr(language, name, MethodType(getattr(LanguageSpace, name), language))
    language._bounded_binary_reconstruction = LanguageSpace._bounded_binary_reconstruction
    owner = SimpleNamespace(languageSpace=language)
    def forbidden():
        raise AssertionError('decoding accessed the compose journal')
    owner._reconstruction_stack = forbidden
    bank = torch.tensor([[[2., 1.], [1., 2.]]]).expand(2, -1, -1).clone().requires_grad_()
    root = conjunction.compose(bank[:, 0], bank[:, 1])
    signed = root * torch.tensor([[1.], [-1.]])
    event = torch.cat((signed[:, None], signed.new_zeros(2, 3, 2)), 1)
    def walk(event, bank):
        return Models.BasicModel._output_generate_walk(owner, event, 8,
            basis=bank, basis_valid=torch.ones(2, 2, dtype=torch.bool),
            basis_priming=torch.ones(2, 2), candidate_limit=2, return_trace=True)
    fn = torch.compile(walk, backend='inductor', fullgraph=True) if compiled else walk
    out, count, truncated, _, actions = fn(event, bank)
    assert count.tolist() == [2, 2]
    assert not truncated.any()
    torch.testing.assert_close(out[:, :2], bank)
    assert actions[0, :3].tolist() == [0, 2, 2]
    assert actions[1, :4].tolist() == [1, 0, 2, 2]
    # Even with multiple eligible actions, the free walk supplies no policy
    # gradient. The policy learns the compose structure in its teacher loss.
    def choice_at_compound(parent, lefts, rights, available, ops, codes, valid, **kwargs):
        # The unit-activation parent is shorter than these form codes. Keep
        # the same hard trace by offering alternatives at the compound, not
        # by expecting a norm threshold to distinguish roots from words.
        leaf = (valid & (codes == parent[:, None]).all(-1)).any(-1)
        stop = torch.arange(available.shape[-1], device=parent.device) == available.shape[-1] - 1
        return available & (~leaf[:, None] | stop)
    monkeypatch.setattr(LanguageSpace, 'decoder_eligibility', staticmethod(choice_at_compound))
    credited = fn(event, bank)[0]
    torch.testing.assert_close(credited, out)
    out = credited
    out.square().sum().backward()
    assert policy.weight.grad is None or not policy.weight.grad.any()
    assert bank.grad is None or not bank.grad.any()  # searched symbols remain detached
