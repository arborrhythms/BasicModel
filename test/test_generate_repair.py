"""Item 2.7 §34: bounded generation, completion, owned credit and orders."""
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


def controlled_walk(monkeypatch, mode):
    import Generative
    import util
    from Models import BasicModel
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    binary, unary = (1, 2) if mode == 'resets' else (0, 3)
    owner = SimpleNamespace(_generate_binary_ops=(SimpleNamespace(meaning_layout=None),)*binary,
        _generate_unary_ops=(SimpleNamespace(meaning_layout=None),)*unary, _generate_policy_width=2)
    owner.generate_policy_logits = lambda x,features=None: x.new_zeros(len(x), binary+unary+1)
    owner.choose_generate = lambda logits: logits.argmax(-1)
    def menu(language, parent, live, **options):
        value = parent[:, 0]
        stop = binary+unary
        left = parent[:, None].expand(-1, stop+1, -1).clone()
        right = left.clone()
        legal = torch.zeros(len(parent), stop+1, device=parent.device, dtype=torch.bool)
        if mode == 'drift':
            left[:, :unary] += 4e-5
            legal[:, :unary] = live[:, None]
        elif mode == 'chain':
            left[:, :unary] += 1
            for i in range(unary):
                legal[:, i] = live & (value == i+1)
            legal[:, stop] = live & (value == unary+1)
        else:
            split = (value == 3) | (value == 6)
            terminal = (value == 12) | (value == 22) | (value == 9)
            left[:, :stop] += 1
            right[:, 0] = torch.where((value == 3)[:, None], 10., 20.)
            legal[:, 0] = live & split
            legal[:, 1] = live & ~split & ~terminal
            legal[:, stop] = live & terminal
        from generation_fixtures import exact_menu
        return exact_menu(left,right,legal,binary)
    monkeypatch.setattr(Generative, 'inverse_menu', menu)
    model = SimpleNamespace(languageSpace=owner)
    def walk(event):
        with torch.no_grad():
            return BasicModel._output_generate_walk(model, event, 144,
                require_symbols=False, return_candidates=True)
    return walk, menu


@pytest.mark.parametrize('mode', ['drift', 'chain', 'resets'])
def test_unary_bound_and_slot_resets(monkeypatch, mode):
    import Models
    import Generative
    walk, menu = controlled_walk(monkeypatch, mode)
    calls, carries = [], []
    def counted(*args, **kwargs):
        calls.append(1)
        return menu(*args, **kwargs)
    monkeypatch.setattr(Generative, 'inverse_menu', counted)
    def observed_loop(cond, body, values):
        while bool(cond(*values)):
            values = body(*values)
            carries.append(values)
        return values
    monkeypatch.setattr(Models, '_reconstruction_while_loop', observed_loop)
    event = torch.tensor([[[1., 1.], [0., 0.], [0., 0.]]])
    result = walk(event)
    actions = result[4][0][result[4][0] >= 0].tolist()
    if mode == 'drift':
        assert actions == [0, 0, 0] and len(calls) == 4
        assert result[2].item() and result[1].item() == 0
    elif mode == 'chain':
        assert actions == [0, 1, 2, 3] and len(calls) == 4
        assert not result[2].item() and result[1].item() == 1
        assert result[0][0, 0].tolist() == [4., 4.]
    else:
        assert actions == [1, 1, 0, 1, 1, 3, 1, 1, 0, 1, 1, 3, 1, 1, 3]
        assert not result[2].item() and result[1].item() == 3
        assert carries[2][8][0].tolist() == [0, 0, 0]  # rewritten and pushed
        assert carries[5][8][0, 1].item() == 0  # vacated slot
        assert carries[8][8][0].tolist() == [0, 0, 0]  # reused slot
    assert result[-1].shape == (1, 144, 2) and not result[-1].requires_grad


@pytest.mark.parametrize('mode', ['drift', 'chain', 'resets'])
def test_unary_bound_eager_and_compiled_agree(monkeypatch, mode):
    walk, _ = controlled_walk(monkeypatch, mode)
    event = torch.tensor([[[1., 1.], [0., 0.], [0., 0.]]])
    expected = walk(event)
    actual = torch.compile(walk, backend='inductor', fullgraph=True)(event)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_native_walk_below_bound_matches_frozen_103c(monkeypatch, tmp_path):
    import Models
    import util
    from data import TheData
    from Models import BaseModel
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                       for split in ('train','validation','test')})
    model = BaseModel.from_config('data/MM_corpus_orders.xml', data=TheData)[0]
    try:
        namespace = dict(vars(Models))
        source = Path(__file__).with_name('fixtures')/'item27-103c-output-walk.py'
        exec(compile(source.read_text(), str(source), 'exec'), namespace)
        old = namespace['_output_generate_walk']
        language = model.languageSpace
        width = language._generate_policy_width
        event = torch.zeros(1, 8, width)
        event[0, 0] = .5
        options = dict(basis=event[:, :1], basis_valid=torch.ones(1, 1, dtype=torch.bool),
                       return_candidates=True)
        with torch.no_grad():
            import Generative
            native=Generative.inverse_menu
            with monkeypatch.context() as patch:
                patch.setattr(Generative,'inverse_menu',lambda *a,**k:native(*a,**k)[:3])
                patch.setattr(Generative,'InverseMenuRead',lambda *a:None)
                before = old(model, event, 144, **options)
            after = Models.BasicModel._output_generate_walk(model, event, 144, **options)
        assert before[1].item() == 1 and not before[2].item()
        assert before[4][0, 0] == len(language._generate_binary_ops)+len(language._generate_unary_ops)
        for a, b in zip(after[:len(before)], before):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        model.End()


@pytest.mark.parametrize('comparison', ['output', 'decoder'])
def test_finished_wins_and_matching_pending_compares_cost(comparison,monkeypatch):
    from Models import BasicModel
    from Output import AnswerConstruction
    from WalkTrials import output_pair
    costs = torch.tensor([[1., 9.], [9., 1.], [2., 1.], [2., 1.], [1., 1.], [1., 2.]])
    pending = torch.tensor([[True, False], [False, True], [True, True],
                            [False, False], [False, False], [True, True]])
    actions = torch.zeros(6, 2, dtype=torch.long)
    alternatives = torch.ones_like(actions, dtype=torch.bool)
    tops = torch.zeros(6, 2, 2)
    def realize(**options):
        side = int(options.get('exploit_actions') is not None)
        value = costs[:, side, None]
        return AnswerConstruction(value,None,value,value,None,(),()), (
            actions+side, alternatives, pending[:, side],value[:,0]*0,~pending[:,side],
            tops.new_zeros(6,2,1,5),tops)
    if comparison == 'output':
        _, audit = output_pair(realize, lambda x: x.actual[:, 0])
    else:
        def reconstruct(*args, **options):
            side = int(options.get('decoder_exploit') is not None)
            value = costs[:, side]
            trace = (tops, torch.ones(6,dtype=torch.long), pending[:, side], value*0, actions+side)
            if options.get('decoder_candidates'):
                trace = (*trace, alternatives)
            from generation_fixtures import suffix
            return (tops, value*0, value, pending[:, side], value[:, None]), (*trace, *suffix(tops,pending[:,side]))
        model = SimpleNamespace(_sentence_training=True, _compiled_reconstruct=lambda: reconstruct)
        from SentenceUnderstanding import SentenceUnderstanding
        record=SentenceUnderstanding(torch.zeros(6,2),tops,torch.ones(6),tops,torch.ones(6),
            tops,torch.zeros(6,2,dtype=torch.long),torch.ones(6,2,dtype=torch.bool),None,torch.tensor(0))
        import Generative
        monkeypatch.setattr(Generative,'walk_bank',lambda model,root,bank:bank)
        BasicModel._reconstruct_trial(model, record)
        audit = model._last_decoder_comparison
    assert audit['wins'].tolist() == [True, False, True, True, False, False]
    assert torch.equal(audit['pending'], pending)


def test_winning_departure_credit_reaches_only_generate_policy_and_its_owner():
    from Language import LanguageSpace
    from Models import BaseModel
    from WalkTrials import generate_credit, stage_generate_credit
    policy = torch.nn.Linear(2, 3)
    upstream = torch.nn.Linear(2, 2)
    language = SimpleNamespace(generate_policy=policy, _generate_policy_width=2)
    for name in ('generate_policy_logits', 'generate_policy_credit'):
        setattr(language, name, MethodType(getattr(LanguageSpace, name), language))
    tops = upstream(torch.ones(3, 4, 2))
    audit = dict(departure=torch.tensor([1, 2, -1]), wins=torch.tensor([True, False, False]),
        explore=torch.tensor([[0, 1, 2, 0]]*3), costs=torch.zeros(3,2))
    model = SimpleNamespace(languageSpace=language, _sentence_walk_credits=[])
    stage_generate_credit(model, audit, tops)
    credit, = model._sentence_walk_credits
    assert credit[0] > 0 and credit[1:].eq(0).all()
    credit.sum().backward()
    assert policy.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in upstream.parameters())
    optimizer = torch.optim.SGD([*policy.parameters(), *upstream.parameters()], lr=.01)
    groups = BaseModel.objective_parameter_groups(model, optimizer)
    assert {id(p) for p in policy.parameters()} <= {id(p) for p in groups['reconstruction']}
    assert not {id(p) for p in policy.parameters()} & {id(p) for p in groups['output']}
    audit['wins'].zero_()
    assert generate_credit(language, audit, tops).eq(0).all()


def identity_fixture():
    from Meaning import ConceptualMeaning
    from test_orders_learned import equality_fixture
    owner, store, alloc = equality_fixture()
    word = owner.interpret.lookup_word((4, 7), (), form='label')
    identity, row, order = owner.interpret.reading(word)
    value = owner.similarity_codebook.lookup_rows(row)
    index = store.append_meaning(ConceptualMeaning.from_description(value),
        order=order, document_key='first', sentence_index=0)
    occurrence = store.occurrence_of(index)
    key = owner.mtm.token_of(identity)
    owner.mtm.bind(key, occurrence, kind=None, when=0, value=value, reading=word)
    return owner, store, alloc, key, word, occurrence


def test_promoted_order_survives_early_occurrence_forgetting_and_two_epochs():
    from FamiliarTier import promote_identity
    from Forgetting import model_boundary
    from test_presentation_memory import model_for
    owner, store, alloc, key, word, occurrence = identity_fixture()
    cid = promote_identity(owner, key, word=word)
    assert alloc.reference_orders[cid] == 1
    store.reset()  # The occurrence is already gone before the epoch hook.
    assert store.location(occurrence) is None and alloc.order_of(cid) == 1
    for _ in range(2):
        model_boundary(model_for(owner, store), epoch_boundary=True)
        assert alloc.order_of(cid) == 1


def test_promotion_rejects_missing_witness_at_creation():
    from FamiliarTier import promote_identity
    owner, store, alloc, key, word, occurrence = identity_fixture()
    store.reset()
    before = alloc.next_id
    with pytest.raises(RuntimeError, match='identity promotion has no witnessed order: occurrence=.*record='):
        promote_identity(owner, key, word=word)
    assert alloc.next_id == before


def test_epoch_guard_still_rejects_unwitnessed_constructed_individual():
    from Forgetting import model_boundary
    from test_presentation_memory import model_for
    owner, store, alloc, key, word, occurrence = identity_fixture()
    cid = alloc.new_concept()
    alloc.store_of(cid).add_constituent(cid, 'part', ('occ', occurrence[-1]))
    alloc.individuals.add(cid)
    store.order[store.location(occurrence)] = -1
    with pytest.raises(RuntimeError, match='a promoted identity has no witnessed order at the epoch boundary'):
        model_boundary(model_for(owner, store), epoch_boundary=True)
