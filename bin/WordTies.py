"""The certainty of each tie of a word to an object, held in MTM.

A tie gains a view whenever its word is read with a percept; it holds when
its object reads that percept wholly. Its certainty is the share of its
views that held. The ties are few and forgettable: when room is needed the
least certain is forgotten. An object is never forgotten with its tie; it
stays, more or less certain by the ties that remain to it.
"""
from dataclasses import dataclass

from util import TheXMLConfig


@dataclass
class Tie:
    views: int = 0
    agreement: float = 0.

    @property
    def certainty(self):
        return self.agreement / self.views if self.views else 0.


class WordTies:
    def __init__(self, capacity):
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError('wordTieCapacity must be positive')
        self.capacity = capacity
        self.ties = {}
        # The word that made each object: its readings widen the object.
        self.origin = {}

    def start(self, word, obj, *, views=0, agreement=0.):
        key = (int(word), int(obj))
        if key not in self.ties:
            if len(self.ties) >= self.capacity:
                weakest = min(self.ties, key=lambda k: (self.ties[k].certainty, self.ties[k].views))
                del self.ties[weakest]
            self.ties[key] = Tie(int(views), float(agreement))
        return self.ties[key]

    def observe(self, word, obj, holds):
        """One view of the tie: ``holds`` in ``[0, 1]``."""
        tie = self.start(word, obj)
        tie.views += 1
        tie.agreement += float(holds)
        return tie

    def state(self):
        return dict(capacity=self.capacity, origin=tuple(self.origin.items()),
                    ties=tuple((word, obj, tie.views, tie.agreement) for (word, obj), tie in self.ties.items()))

    @classmethod
    def restored(cls, state):
        held = cls(state['capacity'])
        held.origin = dict(state['origin'])
        held.ties = {(word, obj): Tie(views, agreement) for word, obj, views, agreement in state['ties']}
        return held

    def get(self, word, obj):
        return self.ties.get((int(word), int(obj)))

    def certainty(self, word, obj):
        tie = self.get(word, obj)
        return 0. if tie is None else tie.certainty

    def words_of(self, obj, least=0.):
        """The words tied to ``obj`` at a certainty of ``least`` or more, in
        the order of their ties."""
        return tuple(word for (word, held), tie in self.ties.items()
                     if held == int(obj) and tie.certainty >= least)


def ties(owner):
    """The ties of ``owner``'s MTM, made on first use; sized by
    ``architecture.wordTieCapacity``."""
    pool = owner.mtm
    held = getattr(pool, 'word_ties', None)
    if held is None:
        held = WordTies(TheXMLConfig.get('architecture.wordTieCapacity', default=256))
        pool.word_ties = held
    return held
