"""Reverse-side <attention> priming exercised on the word-grain gates
(open-fronts follow-on todo §3, 2026-07-13).

Open-fronts Task C widened the ``LanguageLayer.unreduce`` recommender-family
guard to Lift/Lower and added the ``<PartSpace><wordStore>`` word-whole-rows
restriction on the SS space_role. Both were landed DORMANT (no shipped config
flips ``<attention>``); these pins exercise the modes over the word-grain
gates with the REAL unreduce + REAL LiftLayer + REAL
``retrieval_candidates_for_slot`` (the test_heat_reverse_wiring harness
idiom, SS-role variant):

  * primer ON  -> the Lift-hosted reverse fires retrieval and carries the
    typed+heat candidate rows;
  * attention OFF + wordStore ON -> the WS word-whole rows restrict the
    reverse (THE Task-C SS-side pin);
  * primer ON + wordStore ON -> heat rows take PRECEDENCE over word rows
    (the ``'left_rows' not in reverse_kwargs`` gate);
  * retrieval failure -> degrades to the word-rows restriction, never
    breaking generation (the ON==OFF degradation contract, word-grain form).
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Language
from Language import (  # noqa: E402
    Grammar, LanguageLayer, LiftLayer, SymbolSubSpace, Taxonomy,
)


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(20260713)
    yield


class _FakeView:
    def __init__(self, by_cat, by_order):
        self._c = by_cat
        self._o = by_order

    def refs_by_category(self, name):
        return self._c.get(name, torch.empty(0, dtype=torch.long))

    def refs_by_order(self, o):
        return self._o.get(int(o), torch.empty(0, dtype=torch.long))


class _Basis:
    def __init__(self, W):
        self._W = W

    def getW(self):
        return self._W


class _StackSubSpace:
    def __init__(self, what, where, activation, basis, word_sub_space):
        self._what = what
        self._where = where
        self._activation = activation
        self.what = basis
        self.symbolSpace = word_sub_space

    def materialize(self, mode):
        return {"what": self._what,
                "where": self._where,
                "activation": self._activation}[mode]

    def set_what(self, t):
        self._what = t

    def set_where(self, t):
        self._where = t

    def set_activation(self, t):
        self._activation = t


def _grammar_with_lift():
    """Real Grammar holding one binary SS-role rule ``LP = lift(NP3, VP1)``."""
    g = Grammar()
    g.rules = [g._parse_rule("LP", "lift(NP3, VP1)", space_role='SS')]
    g.rule_table = {0: g.rules[0].canonical}
    g.symbol_vocab_size = 4
    g._configured = True
    return g


def _make_ss(view, *, attention_mode, hot_ref=None, word_registry=None,
             pos_to_row=None, top_order=3):
    """Real SymbolSubSpace; SS-role dispatch reads ``ss.wholeSpace`` for both
    the attention mode AND the word-whole registry."""
    ss = object.__new__(SymbolSubSpace)
    nn.Module.__init__(ss)
    ss.batch = 1
    tax = Taxonomy()
    tax.allocate_priming(batch_size=1, capacity=8, live=8)
    tax.configure_priming(priming_enabled=True)
    if hot_ref is not None:
        tax.prime([int(hot_ref)], batch=0, boost=4.0)
    ss.taxonomy = tax
    object.__setattr__(ss, '_knowledge', view)
    ss._order = torch.zeros(1, 8, dtype=torch.long)
    ss._order[0, 0] = int(top_order)
    object.__setattr__(ss, 'wholeSpace', SimpleNamespace(
        attention_mode=attention_mode,
        _word_whole_ss=dict(word_registry or {}),
        _ws_pos_to_row=dict(pos_to_row or {})))
    object.__setattr__(ss, 'conceptualSpace', None)
    return ss


_W = torch.tensor([
    [0.40, 0.30],   # row 0 -- NP-typed
    [0.41, 0.31],   # row 1 -- NP-typed
    [0.20, 0.10],   # row 2 -- VP-typed AND the word-whole row
])
_PARENT = torch.tensor([0.5, 0.4])
_VIEW = _FakeView(
    by_cat={'NP': torch.tensor([0, 1], dtype=torch.long),
            'VP': torch.tensor([2], dtype=torch.long)},
    # All rows admissible at the parent's order 3: the order-preserving
    # dispatch intersects BOTH slots' categories with refs_by_order(3),
    # and an empty intersection would drop the category filter (the
    # sanctioned untyped fallback), blurring the typed-rows assertion.
    by_order={3: torch.tensor([0, 1, 2], dtype=torch.long)},
)
# Word-whole registry: two surfaces resolving to row 2 only -- DISJOINT from
# the NP heat rows {0, 1} so precedence is observable.
_REGISTRY = {"hello": 7, "world": 9}
_POS_TO_ROW = {7: 2, 9: 2}


def _run_unreduce(attention_mode, *, hot_ref=None, word_store=False,
                  break_retrieval=False):
    """Drive the REAL unreduce; capture the reverse kwargs via a layer spy.

    Returns (captured_kwargs, retrieval_calls)."""
    g = _grammar_with_lift()
    D = _W.shape[1]
    K = 3
    what = torch.zeros(1, K, D)
    what[0, 0, :] = _PARENT
    where = torch.zeros(1, K, 1)
    where[0, 0, 0] = float(4 + 1 + 0)          # rule 0
    activation = torch.zeros(1, K)
    activation[0, 0] = 1.0
    ss = _make_ss(_VIEW, attention_mode=attention_mode, hot_ref=hot_ref,
                  word_registry=_REGISTRY, pos_to_row=_POS_TO_ROW)
    sub = _StackSubSpace(what, where, activation, _Basis(_W), ss)

    retrieval_calls = []
    orig_ret = ss.retrieval_candidates_for_slot

    def spy_ret(*a, **k):
        retrieval_calls.append(k.get('mode'))
        if break_retrieval:
            raise RuntimeError("forced retrieval failure")
        return orig_ret(*a, **k)

    object.__setattr__(ss, 'retrieval_candidates_for_slot', spy_ret)

    lift = LiftLayer(nInput=D, nOutput=D)
    syn = SimpleNamespace(space_role='SS', _by_name={'lift': lift})
    captured = {}

    def rev_spy(parent, basis=None, **kwargs):
        captured.update(kwargs)
        return parent, parent

    lift.reverse = rev_spy

    orig_space = Language.TheXMLConfig.space

    def patched_space(section, key, default=None, *a, **k):
        if section == 'PartSpace' and key == 'wordStore':
            return bool(word_store)
        return orig_space(section, key, default, *a, **k)

    Language.TheXMLConfig.space = patched_space
    try:
        lang = LanguageLayer.__new__(LanguageLayer)
        lang.unreduce(sub, syn, grammar=g)
    finally:
        Language.TheXMLConfig.space = orig_space
    return captured, retrieval_calls


def test_primer_fires_retrieval_on_lift_family():
    """Lift joined the recommender family (Task C): primer mode fires the
    retrieval helper on a Lift-hosted reverse and carries the typed rows."""
    kwargs, calls = _run_unreduce('primer', hot_ref=0)
    assert calls == ['primer', 'primer'], calls          # left + right slot
    assert 'left_rows' in kwargs
    assert sorted(kwargs['left_rows'].tolist()) == [0, 1]
    assert sorted(kwargs['right_rows'].tolist()) == [2]


def test_word_rows_restrict_when_attention_off():
    """THE Task-C SS-side pin: attention off + wordStore on -> the reverse is
    restricted to the WS word-whole rows (registry through _ws_pos_to_row)."""
    kwargs, calls = _run_unreduce('off', word_store=True)
    assert calls == []                                   # heat path dormant
    assert kwargs.get('left_rows') is not None
    assert kwargs['left_rows'].tolist() == [2]
    assert kwargs['right_rows'].tolist() == [2]


def test_off_without_word_store_is_plain_reverse():
    """Both gates off -> the pre-existing plain reverse (no kwargs at all)."""
    kwargs, calls = _run_unreduce('off', word_store=False)
    assert calls == [] and kwargs == {}


def test_heat_rows_take_precedence_over_word_rows():
    """A successful heat retrieval SUPPRESSES the word-rows restriction
    (the ``'left_rows' not in reverse_kwargs`` gate)."""
    kwargs, calls = _run_unreduce('primer', hot_ref=0, word_store=True)
    assert calls == ['primer', 'primer']
    assert sorted(kwargs['left_rows'].tolist()) == [0, 1]   # heat, not [2]


def test_retrieval_failure_degrades_to_word_rows():
    """The ON-path degradation contract, word-grain form: a retrieval
    failure empties the heat kwargs and the word-rows restriction still
    applies -- generation never breaks."""
    kwargs, calls = _run_unreduce('primer', hot_ref=0, word_store=True,
                                  break_retrieval=True)
    assert calls and calls[0] == 'primer'
    assert kwargs.get('left_rows') is not None
    assert kwargs['left_rows'].tolist() == [2]              # the word rows


def test_real_lift_reverse_completes_under_priming():
    """Non-breakage: the REAL LiftLayer.reverse runs to completion under the
    primer-mode kwargs on the word-grain gates (no spy)."""
    g = _grammar_with_lift()
    D = _W.shape[1]
    what = torch.zeros(1, 3, D)
    what[0, 0, :] = _PARENT
    where = torch.zeros(1, 3, 1)
    where[0, 0, 0] = 5.0
    activation = torch.zeros(1, 3)
    activation[0, 0] = 1.0
    ss = _make_ss(_VIEW, attention_mode='primer', hot_ref=0,
                  word_registry=_REGISTRY, pos_to_row=_POS_TO_ROW)
    sub = _StackSubSpace(what, where, activation, _Basis(_W), ss)
    syn = SimpleNamespace(space_role='SS',
                          _by_name={'lift': LiftLayer(nInput=D, nOutput=D)})
    lang = LanguageLayer.__new__(LanguageLayer)
    lang.unreduce(sub, syn, grammar=g)                   # must not raise
    out = sub.materialize(mode="what")
    assert torch.isfinite(out).all()
