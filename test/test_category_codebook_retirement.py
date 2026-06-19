"""Tests for the category_codebook → category_embedding retirement.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 deferred — "SymbolicSubSpace.category_codebook retirement". The plan
calls for replacing the heavyweight ``Codebook`` (which carried VQ /
polarity / meronomy / SVD machinery never used by the label consumers)
with a plain ``nn.Embedding[N_categories, pos_dim]``.

Consumers ported:
  * ``category_lookup(name)`` — direct row lookup by index
  * ``pos_lookup(active_symbols)`` — activation-similarity snap
  * ``SyntacticLayer._resetChart`` defensive seed (Language.py)

The ``category_codebook`` attribute no longer exists on SymbolicSpace.
"""
import os
import sys

import torch
import torch.nn as nn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)

# Reuse the proven SymbolicSubSpace builder from the existing partition test.
from test_partition_pos_codebook import _make_word_space  # noqa: E402


def test_category_embedding_replaces_codebook():
    """``category_embedding`` is the live attribute; the legacy
    ``category_codebook`` is no longer constructed."""
    ss = _make_word_space()
    assert hasattr(ss, 'category_embedding')
    assert isinstance(ss.category_embedding, nn.Embedding)
    assert not hasattr(ss, 'category_codebook')


def test_category_embedding_weight_shape():
    """``category_embedding.weight`` is ``[max(64, |grammar.categories|),
    pos_dim=4]``."""
    ss = _make_word_space()
    assert ss.category_embedding.weight.ndim == 2
    assert ss.category_embedding.weight.shape[0] >= 64
    assert ss.category_embedding.weight.shape[1] == 4


def test_category_lookup_returns_embedding_row():
    """``category_lookup(name)`` returns the embedding row for the
    given category index. Identical to ``category_embedding.weight[idx]``.
    """
    ss = _make_word_space()
    name = next(iter(ss.category_index.keys()))
    idx = ss.category_index[name]
    expected = ss.category_embedding.weight[idx]
    actual = ss.category_lookup(name)
    assert torch.allclose(actual, expected)


def test_category_lookup_by_int_index():
    """Integer-form lookup also works."""
    ss = _make_word_space()
    expected = ss.category_embedding.weight[3]
    actual = ss.category_lookup(3)
    assert torch.allclose(actual, expected)


def test_pos_lookup_returns_pos_dim_vector():
    """``pos_lookup(active_symbols)`` snaps to an embedding row and
    returns a ``[pos_dim]`` 1-D tensor."""
    ss = _make_word_space()
    active = torch.tensor([0.9, 0.0, 0.3])
    out = ss.pos_lookup(active)
    assert out.shape == (4,)


def test_category_embedding_registered_as_parameter():
    """The embedding's weight is a registered nn.Parameter (gradient-
    flowing). Codebook's wrapper was a Parameter too, but a plain
    Embedding makes the registration trivial / explicit."""
    ss = _make_word_space()
    params = dict(ss.named_parameters())
    # category_embedding.weight should appear in named_parameters
    matching = [k for k in params if 'category_embedding' in k]
    assert len(matching) >= 1
    found = params[matching[0]]
    assert found.requires_grad
