"""Declared fixture forcing until occupation and definition learning supply orders."""
from Spaces import _concept_alloc_of


def force_order(cs, word, order):
    """Record the fixture's order and check it against any structural parts."""
    alloc = _concept_alloc_of(cs)
    alloc.reference_orders[int(word)] = int(order)
    alloc.settle(int(word))
