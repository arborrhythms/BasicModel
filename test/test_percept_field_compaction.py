"""The table of a sentence's occurrences: padding is no occurrence, and no
read crosses rows with an axis of events beside one of readings."""
from types import SimpleNamespace as NS

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from BasicPercepts import BasicPercepts


def parts_book(count, width):
    """A codebook in which each of ``count`` parts has one coordinate of
    its own, ``i mod width``."""
    from Spaces import Codebook
    book = Codebook()
    book.W = nn.Parameter(torch.zeros(count, width))
    with torch.no_grad():
        book.W[torch.arange(count), torch.arange(count) % width] = 1.
    return book


def test_padding_reads_nothing_and_teaches_no_property():
    from test_form_subspaces import property_book, P
    whole = property_book(rows=1)
    primitive = whole.primitive_properties
    with torch.no_grad():
        primitive.members[0, 65] = .3; primitive.members[0, 66] = .7
        primitive.observed[:, 65:67] = True
    whole.begin_forward()
    basic = BasicPercepts(NS(_basis=parts_book(9, 3)), 3, P, 8)
    basic.bind(whole)
    run, held = basic.reserve((7,), ()), basic.reserve((7,), (0,))
    raw = torch.tensor([[65, 66], [65, 66]])
    ids = torch.tensor([[7, 8], [7, 8]])
    spans = torch.tensor([[[0, 1], [1, 2]]]).expand(2, -1, -1)
    extent = torch.tensor([[[0, 2]]]).expand(2, -1, -1)
    def read(padded):
        parts = torch.cat((spans, torch.zeros(2, 13, 2, dtype=torch.long)), 1) if padded else spans
        rows = torch.cat((ids, torch.full((2, 13), -1)), 1) if padded else ids
        wholes = torch.cat((extent, torch.zeros(2, 7, 2, dtype=torch.long)), 1) if padded else extent
        return basic.read((rows, parts, primitive, raw, wholes), extent, (run, held))
    short, compact = read(False), read(True)
    torch.testing.assert_close(short.values, compact.values, rtol=0, atol=0)
    torch.testing.assert_close(short.events, compact.events, rtol=0, atol=0)
    assert compact.spans.shape[:2] == (2, 3)
    at = {tuple(span): a for a, span in enumerate(compact.spans[0].tolist())}
    # The run (7,) is held by its own event and by the whole; not at [1, 2].
    for span, value in (((0, 1), 1.), ((0, 2), 1.), ((1, 2), 0.)):
        assert compact.events[0, :, at[span], 0].eq(value).all()
    # The row of 7 in the whole: at its own byte the whole is held .3 for
    # and .7 against.
    torch.testing.assert_close(compact.events[1, :, at[0, 1]].detach(), torch.tensor([[.3, .7], [.3, .7]]))
    # The read takes the memberships' gradient: declared classes are not
    # fixed.
    assert compact.values.requires_grad


def test_captured_b28_w256_read_has_no_axis_of_events_beside_readings():
    """The captured allocation was [150,28,256,2058,2], or 17.7 GB."""
    B, W, P, K = 28, 256, 2058, 150
    class Bounded(TorchDispatchMode):
        def __init__(self): super().__init__(); self.shapes = set()
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            from torch.utils._pytree import tree_leaves
            self.shapes.update(tuple(x.shape) for x in tree_leaves(result) if torch.is_tensor(x))
            return result
    basic = BasicPercepts(NS(_basis=parts_book(K, 8)), 8, 0, 256)
    rows = tuple(basic.reserve((i,), ()) for i in range(K))
    spans = torch.zeros(B, P - 1, 2, dtype=torch.long)
    spans[:, 0] = torch.tensor([0, 1]); spans[:, 1] = torch.tensor([1, 2])
    ids = torch.full((B, P - 1), -1, dtype=torch.long); ids[:, 0] = 7; ids[:, 1] = 8
    raw = torch.full((B, 2), 65, dtype=torch.long)
    brackets = torch.zeros(B, W, 2, dtype=torch.long); brackets[:, :3] = torch.tensor([[0, 1], [1, 2], [0, 2]])
    observer = Bounded()
    with observer:
        field = basic.read((ids, spans, None, raw, spans[:, :0]), brackets, rows)
    assert field.events.shape[:2] == (K, B) and field.values.shape == (K, B, W, 2)
    at = {tuple(span): a for a, span in enumerate(field.spans[0].tolist())}
    assert field.events[7, :, at[0, 1], 0].eq(1).all() and field.events[8, :, at[1, 2], 0].eq(1).all()
    assert field.events[7, :, at[0, 2], 0].eq(1).all()
    assert not any(len(shape) == 5 and shape[:2] == (K, B) for shape in observer.shapes)
    assert not any(len(shape) >= 4 and shape[-2:] == (W, W) for shape in observer.shapes)


def test_interior_gap_does_not_become_an_ordered_literal_match():
    """A run of parts is an occurrence of its own only where its part events
    stand in order with nothing between them in the events' axis."""
    basic = BasicPercepts(NS(_basis=parts_book(9, 3)), 3, 0, 8)
    gap = basic._form_spans(torch.tensor([[7, -1, 8, -1]]), torch.tensor([[[0, 1], [0, 0], [1, 2], [0, 0]]]),
                            [('ps', (7, 8))])
    assert gap == [[]]
    found = basic._form_spans(torch.tensor([[7, 8]]), torch.tensor([[[0, 1], [1, 2]]]), [('ps', (7, 8))])
    assert found == [[(0, 2)]]
