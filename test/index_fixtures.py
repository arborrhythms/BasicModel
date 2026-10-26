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
    from ClauseRow import predicate_relation
    model.languageSpace = SimpleNamespace(
        language_layer=getattr(getattr(model, 'languageSpace', None), 'language_layer', None),
        _generate_binary_ops=(), _generate_unary_ops=(),
        generate_policy_logits=lambda values: values.new_zeros(len(values), 1))
    space = model.conceptualSpace
    rows = torch.tensor([_existing_row(space, reference) for reference in references
                         if reference is not None
                         and predicate_relation(reference[1]) == 'operator'])
    space.prime_seen(rows)
    configure_model_index(model, model.symbolSpace.ltm_store)
    return max((space._row_order(int(row)) for row in rows), default=0)


def one_hot_unfold(value, limit, **kwargs):
    row = int(value.argmax())
    expected = torch.zeros_like(value)
    expected[row] = 1
    return ((row,), 1, True) if limit and torch.equal(value, expected) else ((), 0, False)


def append_content(store, meaning, **kwargs):
    """Each retrieval fixture write denotes one distinct source idea."""
    kwargs.setdefault('document_key', 'content-fixture')
    kwargs.setdefault('sentence_index', len(store))
    return store.append_meaning(meaning, **kwargs)
