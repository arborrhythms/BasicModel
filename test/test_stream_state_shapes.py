"""Item 6.1: a batch position must not silently reuse another stream's state."""
from types import SimpleNamespace

import pytest
import torch


def _attention_model(keys):
    from Language import OperationSelectionLayer
    batch, words, width = keys.shape
    starts = torch.arange(words) * 2
    spans = torch.stack((starts, starts + 1), -1).expand(batch, -1, -1)
    chooser = OperationSelectionLayer(d_model=width)
    owner = SimpleNamespace(
        definitions=SimpleNamespace(word=lambda **kwargs: None),
        similarity_codebook=None, priming_weights=lambda **kwargs: None)
    return SimpleNamespace(
        inputSpace=SimpleNamespace(
            _ar_embedded_N=keys, _word_active_mask=torch.ones(batch, words, dtype=torch.bool),
            _ar_word_part_offsets=spans),
        perceptualSpace=SimpleNamespace(_forward_input={
            'word_texts': [[chr(ord('a') + w) for w in range(words)] for _ in range(batch)]}),
        conceptualSpace=SimpleNamespace(), attention_budget=8,
        _concept_owner=lambda: owner, _stm_reducer=lambda: chooser,
        languageSpace=SimpleNamespace(modules=lambda: []))


def test_attention_gist_is_invariant_to_other_batch_rows(monkeypatch):
    import ModelAttention
    monkeypatch.setattr(ModelAttention, 'native_word_poles',
                        lambda model, spans, forms, known: torch.zeros(*known.shape, 2))
    keys = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]],
                         [[0., 1., 0., 0.], [0., 1., 0., 0.]]])
    batched = _attention_model(keys)
    ModelAttention.stage_input(batched)
    assert batched._last_gist.shape == (2, 4)
    for b in range(2):
        alone = _attention_model(keys[b:b+1])
        ModelAttention.stage_input(alone)
        torch.testing.assert_close(batched._last_gist[b], alone._last_gist[0])


@pytest.mark.parametrize('rows', [None, [0]])
@pytest.mark.parametrize('has_vectors', [False, True])
def test_document_reset_clears_both_recall_histories(rows, has_vectors):
    from Models import BasicModel
    model = SimpleNamespace(
        _what_recall_history={0: ['a'], 1: ['b']} if has_vectors else {},
        _what_recall_sentence_history={0: ['field-a'], 1: ['field-b']})
    BasicModel._clear_recall_history(model, rows)
    expected = {} if rows is None else {1: ['field-b']}
    assert model._what_recall_sentence_history == expected
    assert 0 not in model._what_recall_history


def test_word_reference_commit_does_not_keep_old_order_shape():
    from Language import SymbolSpace
    owner = SimpleNamespace()
    SymbolSpace.commit_word_reference_slab(
        owner, torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, 1),
        torch.ones(1, 4, dtype=torch.bool), orders=torch.ones(1, 4, dtype=torch.long))
    SymbolSpace.commit_word_reference_slab(
        owner, torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2, 1),
        torch.ones(2, 2, dtype=torch.bool))
    assert getattr(owner, '_word_reference_orders', None) is None


def test_word_reference_reader_rejects_reshaped_stale_presences():
    from Models import BasicModel
    owner = SimpleNamespace(similarity_codebook=SimpleNamespace(
        lookup_rows=lambda rows: torch.ones(*rows.shape, 4)))
    model = SimpleNamespace(
        _concept_owner=lambda: owner, _tensor_pushed_ideas=torch.ones(2, 2, 4),
        symbolSpace=SimpleNamespace(_word_reference_presences=torch.ones(1, 4, 1),
                                   _word_reference_evidence=torch.zeros(2, 2, 2)))
    with pytest.raises(ValueError, match='word reference.*shape|word reference.*align'):
        BasicModel._answer_leaf_slab(model, torch.zeros(2, 2, dtype=torch.long))


def test_taxonomy_resize_cannot_reassign_live_priming_to_new_batch():
    from Language import Taxonomy
    tax = Taxonomy()
    tax.allocate_priming(1, 4, 4)
    tax.prime([1], batch=0)
    with pytest.raises(ValueError, match='priming.*batch|batch.*priming'):
        tax.allocate_priming(2, 4, 4)
    assert tax._priming.shape == (1, 4)


@pytest.mark.parametrize('reader', ['_recall_history', '_recall_sentence_history'])
@pytest.mark.parametrize('new_batch', [1, 3])
def test_recall_read_rejects_batch_change_until_all_history_is_reset(reader, new_batch):
    from Models import BasicModel
    model = BasicModel.__new__(BasicModel)
    torch.nn.Module.__init__(model)
    model.inputSpace = SimpleNamespace(_word_active_mask=torch.ones(2, 1, dtype=torch.bool))
    history = getattr(model, reader)()
    history[0], history[1] = ['a'], ['b']
    model.inputSpace._word_active_mask = torch.ones(new_batch, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match='recall.*batch|batch.*recall'):
        getattr(model, reader)()
    model._clear_recall_history([0])
    with pytest.raises(ValueError, match='recall.*batch|batch.*recall'):
        getattr(model, reader)()
    model._clear_recall_history()
    assert getattr(model, reader)() == {}


def _symbol_references():
    from Language import SymbolSpace
    symbol = SymbolSpace.__new__(SymbolSpace)
    torch.nn.Module.__init__(symbol)
    symbol.subspace = SimpleNamespace(Start=lambda: None, Reset=lambda **kwargs: None)
    symbol.commit_word_reference_slab(
        torch.tensor([[0, 1], [2, 3]]), torch.ones(2, 2, 1),
        torch.ones(2, 2, dtype=torch.bool), orders=torch.ones(2, 2, dtype=torch.long))
    return symbol


def test_word_references_clear_on_forward_start_and_document_reset():
    symbol = _symbol_references()
    symbol.Reset(batch=0, hard=False)
    assert symbol._word_reference_mask.all()
    symbol.Reset(batch=0, hard=True)
    assert not symbol._word_reference_mask[0].any()
    assert symbol._word_reference_mask[1].all()
    assert symbol._word_reference_rows[0].eq(-1).all()
    assert symbol._word_reference_orders[0].eq(-1).all()
    assert not symbol._word_reference_presences[0].any()
    assert not symbol._word_reference_evidence[0].any()
    symbol.Start()
    for name in ('rows', 'orders', 'mask', 'presences', 'evidence'):
        assert getattr(symbol, '_word_reference_' + name, None) is None


@pytest.mark.parametrize('column', ['orders', 'evidence'])
def test_invalid_reference_commit_keeps_previous_complete_handoff(column):
    symbol = _symbol_references()
    previous = {name: getattr(symbol, '_word_reference_' + name).clone()
                for name in ('rows', 'orders', 'mask', 'presences', 'evidence')}
    with pytest.raises(ValueError, match='word reference'):
        symbol.commit_word_reference_slab(torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, 1), torch.ones(1, 1, dtype=torch.bool),
            **{column: torch.zeros(2, 2)})
    for name, value in previous.items():
        torch.testing.assert_close(getattr(symbol, '_word_reference_' + name), value)


@pytest.mark.parametrize('column', ['orders', 'evidence'])
def test_program_reader_rejects_stale_reference_columns_even_with_empty_program(column):
    from Models import BasicModel
    symbol = SimpleNamespace(**{'_word_reference_' + column: torch.ones(1, 4)})
    model = SimpleNamespace(symbolSpace=symbol, _reconstruction_stack=lambda: SimpleNamespace())
    program = (torch.full((2, 2), -1), torch.zeros(2, 0, 3),
               torch.zeros(2, 0), torch.zeros(2, 0, dtype=torch.long))
    with pytest.raises(ValueError, match='word reference.*align|word reference.*shape'):
        BasicModel._program_entries(model, program, torch.zeros(2, 2, 4),
            torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 3, 4))


def test_rejected_taxonomy_resize_does_not_partially_resize_its_owner():
    from Language import SymbolSubSpace, Taxonomy
    tax = Taxonomy()
    tax.allocate_priming(1, 4, 4)
    tax.prime([1], batch=0)
    resized = []
    owner = SimpleNamespace(batch=1, taxonomy=tax,
        _knowledge=SimpleNamespace(_parent=torch.zeros(4), n_refs_live=4),
        _last_svo=torch.zeros(1, 3, 4), _svo_valid=torch.ones(1, dtype=torch.bool), svo_dim=4,
        category_stack=SimpleNamespace(ensure_batch=resized.append),
        reconstruction_stack=SimpleNamespace(ensure_batch=resized.append),
        _ensure_stm_batch=resized.append)
    with pytest.raises(ValueError, match='priming.*batch'):
        SymbolSubSpace.ensure_batch(owner, 2)
    assert owner.batch == 1 and owner._last_svo.shape == (1, 3, 4)
    assert resized == []
    tax.reset()
    SymbolSubSpace.ensure_batch(owner, 2)
    assert owner.batch == 2 and tax._priming.shape == (2, 4)
    assert tax._priming.eq(1).all()


def test_rejected_recall_write_does_not_first_mutate_discourse():
    from Models import BasicModel
    model = BasicModel.__new__(BasicModel)
    torch.nn.Module.__init__(model)
    model.inputSpace = SimpleNamespace(_word_active_mask=torch.ones(1, 1, dtype=torch.bool))
    model._recall_history()[0] = ['old sentence']
    observed = []
    disc = SimpleNamespace(_pool_sentence_rep=lambda value: value,
                           observe=observed.append)
    with pytest.raises(ValueError, match='recall.*batch'):
        model._observe_discourse(disc, torch.zeros(2, 4))
    assert observed == []


def test_document_reset_clears_only_the_finished_rows_priming():
    from Spaces import Space
    space = SimpleNamespace(layers=(), subspace=None,
                            _priming_boosts=torch.tensor([[2., 3.], [4., 5.]]))
    Space.Reset(space, batch=0, hard=False)
    torch.testing.assert_close(space._priming_boosts, torch.tensor([[2., 3.], [4., 5.]]))
    Space.Reset(space, batch=0, hard=True)
    torch.testing.assert_close(space._priming_boosts, torch.tensor([[1., 1.], [4., 5.]]))
    Space.Reset(space, hard=True)
    assert space._priming_boosts is None
