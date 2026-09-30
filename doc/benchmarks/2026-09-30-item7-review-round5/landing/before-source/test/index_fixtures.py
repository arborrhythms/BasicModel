"""Forced numerical unfolding for inverted-index mechanism tests."""
import torch
from types import SimpleNamespace


def terminal_model_index(model, references):
    """Force terminal generation over primed native codes for reader mechanisms.

    The production index still unfolds numeric values and descends sigma using
    each row's stamp. This fixture asserts no learned parsing or word recovery.
    """
    from MemoryIndex import configure_model_index
    from Queries import _existing_row
    model.languageSpace = SimpleNamespace(
        _generate_binary_ops=(), _generate_unary_ops=(),
        reverse_inverses=lambda _ops: (),
        generate_policy_logits=lambda values: values.new_zeros(len(values), 1))
    space = model.conceptualSpace
    rows = torch.tensor([_existing_row(space, reference) for reference in references
                         if reference is not None])
    space.prime_seen(rows)
    configure_model_index(model, model.symbolSpace.ltm_store)
    return max((space._row_order(int(row)) for row in rows), default=0)


def one_hot_unfold(value, limit, **kwargs):
    row = int(value.argmax())
    expected = torch.zeros_like(value)
    expected[row] = 1
    return ((row,), 1, True) if limit and torch.equal(value, expected) else ((), 0, False)


def append_indexed(store, meaning, *, terms, **kwargs):
    """Supply a forced grammar for this write; retain only inverted postings."""
    prior = store._index_unfold
    mapping = {tuple(value.detach().tolist()): tuple(codes)
               for value, live, codes in zip(meaning.roles, meaning.role_mask, terms) if live}
    def unfold(value, limit, **_kwargs):
        codes = mapping.get(tuple(value.detach().tolist()), ())
        return codes, min(1, limit), bool(limit)
    store._index_unfold = unfold
    try:
        return store.append_meaning(meaning, **kwargs)
    finally:
        store._index_unfold = prior or one_hot_unfold
