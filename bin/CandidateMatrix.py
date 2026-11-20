"""Candidate entries, tried in a small sparse matrix before they are kept.

Beside a Pi or a Sigma stands a matrix of candidate entries, a medium-term
structure. A candidate enters at nothing, so that entering changes no value;
it is read as an entry is; and it takes a step of the proximal kind with its
gradients averaged over the steps at which it was read (dual averaging),

    v = -sign(G) * max(|G| / n - penalty, 0),

``G`` the sum of its gradients over its ``n`` reads, so that it leaves
nothing exactly where its average gradient exceeds the penalty. It is
promoted into the matrix proper when it has held above nothing for ``hold``
reads in a row, and forgotten when the penalty returns it to nothing after it
had held, or when room is needed (the weakest first).
"""
from dataclasses import dataclass

from util import TheXMLConfig


def penalty():
    """The one strength of the penalty, shared by every candidate matrix."""
    return float(TheXMLConfig.get('architecture.candidatePenalty', default=.2))


@dataclass
class Candidate:
    value: float = 0.
    held: int = 0
    steps: int = 0
    gathered: float = 0.


class CandidateMatrix:
    def __init__(self, capacity, *, strength=None, hold=3):
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError('a candidate matrix needs room for a candidate')
        self.capacity, self.hold = capacity, int(hold)
        self.strength = penalty() if strength is None else float(strength)
        self.entries = {}
        self.entered = self.promoted = self.forgotten = 0

    def __contains__(self, key):
        return key in self.entries

    def __len__(self):
        return len(self.entries)

    def offer(self, key):
        """Enter ``key`` at nothing, making room by forgetting the weakest."""
        if key in self.entries:
            return self.entries[key]
        if len(self.entries) >= self.capacity:
            weakest = min(self.entries, key=lambda k: (abs(self.entries[k].value), self.entries[k].steps))
            del self.entries[weakest]
            self.forgotten += 1
        self.entries[key] = Candidate()
        self.entered += 1
        return self.entries[key]

    def value(self, key):
        found = self.entries.get(key)
        return 0. if found is None else found.value

    def step(self, gradients):
        """One proximal step of every candidate; ``gradients`` maps the key
        of each candidate read at this step to its gradient. A candidate
        not read is not stepped. One read
        above nothing ``hold`` times in a row is promoted; one the penalty
        returns to nothing after it had held is forgotten. Returns the keys
        promoted, with their values, and the keys forgotten."""
        promoted, forgotten = [], []
        for key, gradient in gradients.items():
            entry = self.entries.get(key)
            if entry is None:
                continue
            entry.gathered += float(gradient)
            entry.steps += 1
            mean = entry.gathered / entry.steps
            size = max(abs(mean) - self.strength, 0.)
            entry.value = -size if mean > 0 else size
            if entry.value != 0.:
                entry.held += 1
                if entry.held >= self.hold:
                    promoted.append((key, entry.value))
                    del self.entries[key]
                    self.promoted += 1
            elif entry.held:
                forgotten.append(key)
                del self.entries[key]
                self.forgotten += 1
        return promoted, forgotten

    def counts(self):
        return dict(entered=self.entered, promoted=self.promoted, forgotten=self.forgotten,
                    held=len(self.entries))


def pair_gradient(value, left, right):
    """A Pi candidate over two things (the row of a symbol of the pair): its
    reading ``value * max(left, right)`` against what was found together,
    ``min(left, right)``; the gradient of half the squared difference."""
    union, joint = max(left, right), min(left, right)
    return (value * union - joint) * union


def substitution_gradient(expected, read, present):
    """A Sigma candidate from a kind to a thing at one place: the kind was
    expected there and not read (``expected - read`` above nothing) and the
    thing is present; it gathers toward the entry by their product."""
    return -max(float(expected) - float(read), 0.) * float(present)
