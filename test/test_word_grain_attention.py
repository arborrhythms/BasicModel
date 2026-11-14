"""Plain word-grain inverse dispatch after symbol retrieval retirement."""
from types import SimpleNamespace
import torch
from Language import Grammar, LanguageLayer, LiftLayer

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


def test_attention_off_is_plain_reverse():
    grammar = _grammar_with_lift()
    bank = torch.tensor([[.4,.3],[.41,.31],[.2,.1]])
    what = torch.zeros(1,3,2)
    what[0,0] = torch.tensor([.5,.4])
    where = torch.zeros(1,3,1); where[0,0,0] = 5.
    active = torch.zeros(1,3); active[0,0] = 1.
    sub = _StackSubSpace(what, where, active, _Basis(bank), SimpleNamespace())
    lift = LiftLayer(nInput=2,nOutput=2)
    captured = {}
    def reverse(parent, basis=None, **kwargs):
        captured.update(kwargs)
        return parent,parent
    lift.reverse = reverse
    syn = SimpleNamespace(space_role='SS',_by_name={'lift':lift})
    LanguageLayer.__new__(LanguageLayer).unreduce(sub,syn,grammar=grammar)
    assert captured == {}
