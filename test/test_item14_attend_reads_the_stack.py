"""Item 1.4: ``attend`` reads the state of the parse, the whole STM stack
(plan §9.6, 1: "just pass the whole stack"). The request at a look reads
the stack as the last look's shift and reduces left it, through its own
weight, which starts at zero."""
import torch

from Memory import STMChunks


def recorded(model, monkeypatch):
    """Each look's read in the sentences read after this: the inputs of
    ``RequestRead.units`` and its result, and the stack each arrival left."""
    from RequestRead import RequestRead
    from SentenceLooks import Looks
    reads, arrivals = [], []
    units, run = RequestRead.units, Looks.run

    def read_units(self, x, state=None, stack=None):
        result = units(self, x, state, stack)
        reads.append(dict(x=x.detach().clone(), state=None if state is None else state.detach().clone(),
                          stack=None if stack is None else tuple(v.detach().clone() for v in stack),
                          units=result.detach().clone()))
        return result

    def looked(self, arrive, stack=None, **kwargs):
        if stack is None:
            return run(self, arrive, **kwargs)
        def arrived(*args):
            arrive(*args)
            arrivals.append((len(reads), tuple(v.detach().clone() for v in stack())))
        arrivals.append((0, tuple(v.detach().clone() for v in stack())))
        return run(self, arrived, stack=stack, **kwargs)

    monkeypatch.setattr(RequestRead, 'units', read_units)
    monkeypatch.setattr(Looks, 'run', looked)
    return reads, arrivals


def same(a, b):
    return all(torch.equal(u, v) for u, v in zip(a, b))


def request(model, x, state, stack):
    from SentenceLooks import attend
    read = model.request_read
    return attend(model, read.units(x, state, stack))


def test_a_the_request_at_the_next_look_reads_the_stack_the_last_look_left(tmp_path, monkeypatch):
    """(a) The stack each look's request reads is the STM as the arrivals
    before it left it (as the sentence begins at the first look). Two
    sentences that differ in the word reduced at the first look leave
    different stacks; read with the same field, place and flag, they give
    the same request while the stack's weight is zero and different
    requests once it is not."""
    from test_item17_looks import nominated_lesson
    from test_item18_expect import read
    model = nominated_lesson(tmp_path, monkeypatch)
    try:
        reader = model.request_read
        stm = model.conceptualSpace.stm
        assert reader.stack == (stm.capacity, stm.concept_dim)
        assert reader.stack_weight.shape[1] == STMChunks.request_width(*reader.stack)
        assert not reader.stack_weight.any()
        reads, arrivals = recorded(model, monkeypatch)
        stacks = {}
        for text in ('alpha gamma', 'beta gamma'):
            reads.clear()
            arrivals.clear()
            read(model, text)
            assert all(entry['stack'] is not None for entry in reads)
            # The stack read at each look is the last one an arrival left
            # before that look (at the first, the sentence's beginning).
            for k, entry in enumerate(reads):
                last = [held for at, held in arrivals if at <= k][-1]
                assert same(entry['stack'], last), (text, k)
            depths = [int(entry['stack'][STMChunks.DEPTH][0]) for entry in reads]
            print(text, 'depth read at each look', depths)
            assert depths[0] == 0 and depths[1] >= 1
            stacks[text] = reads[1]
        a, b = stacks['alpha gamma'], stacks['beta gamma']
        views = [STMChunks.request_view(s['stack']) for s in (a, b)]
        print('stack read at look 1 differs by', float((views[0] - views[1]).abs().max()))
        assert not torch.equal(*views)
        x, state = a['x'], a['state']
        with torch.no_grad():
            at_zero = [request(model, x, state, s['stack']) for s in (a, b)]
            assert torch.equal(*at_zero)
            # The read's tops start at zero (the request is its start on
            # every concept whatever it reads); they are drawn here as
            # training would move them, and the stack's weight with them.
            kept = {name: value.clone() for name, value in reader.state_dict().items()}
            torch.manual_seed(0)
            reader.ask.weight.normal_(0., 1.)
            at_zero = [request(model, x, state, s['stack']) for s in (a, b)]
            reader.stack_weight.normal_(0., 1.)
            try:
                moved = [request(model, x, state, s['stack']) for s in (a, b)]
            finally:
                reader.load_state_dict(kept)
        assert torch.equal(*at_zero)
        change = float((moved[0] - moved[1]).abs().max())
        print('request at look 1, stack weight zero: equal; stack weight drawn: differs by', change)
        assert change > 0
    finally:
        model.End()


def built_stack(values, orders):
    """A stack of one stream built by shifts, the first value deepest."""
    width = values.shape[-1]
    state = STMChunks.empty(1, 8, width)
    for value, order in zip(values, orders):
        state, _ = STMChunks.deposit(state, value[None], torch.ones(1, dtype=torch.bool),
                                     order=order, grammar_order=order, activation=1., stack=True)
    return state


def test_b_the_read_is_the_whole_stack(tmp_path, monkeypatch):
    """(b) An entry below the top changes the request: a stack of three that
    differs from another only in its bottom entry's value, or only in the
    order of the entry under the top, gives a different request once the
    stack's weight is not zero. The view puts each entry at its stack
    position, the top first."""
    from RequestRead import RequestRead
    torch.manual_seed(0)
    read = RequestRead(6, 16, stack=(8, 5))
    values = torch.rand(3, 5)
    base = built_stack(values, (0, 0, 0))
    bottom = built_stack(torch.cat((values[:1] + 1., values[1:])), (0, 0, 0))
    under = built_stack(values, (0, 1, 0))
    view = STMChunks.request_view(base)
    assert view.shape == (1, STMChunks.request_width(8, 5))
    entries = view[0, :-1].reshape(8, 5 + 6)
    assert torch.equal(entries[0, :5], values[2]) and torch.equal(entries[2, :5], values[0])
    assert entries[:3, 5].tolist() == [1., 1., 1.] and not entries[3:].any() and view[0, -1] == 3
    x = torch.rand(1, read.first.in_features)
    with torch.no_grad():
        read.stack_weight.normal_(0., 1.)
        first = read.units(x, None, base)
        for name, other, position in (('bottom value', bottom, 2), ('order under the top', under, 1)):
            moved = STMChunks.request_view(other)[0, :-1].reshape(8, 11)
            assert torch.equal(moved[0], entries[0]) and not torch.equal(moved[position], entries[position])
            change = float((read.units(x, None, other) - first).abs().max())
            print(name, 'at stack position', position, 'changes the units by', change)
            assert change > 0


def test_c_at_its_start_the_request_is_as_without_the_stack(tmp_path, monkeypatch):
    """(c) With the stack's weight at zero the read is exactly the read
    without the stack: at each look of a sentence, the stack the look read
    being the one the parse held, the units with it equal the units without
    it, bit for bit."""
    from test_item17_looks import nominated_lesson
    from test_item18_expect import read
    model = nominated_lesson(tmp_path, monkeypatch)
    try:
        reader = model.request_read
        reads, _ = recorded(model, monkeypatch)
        for text in ('alpha delta beta', 'delta gamma beta alpha'):
            reads.clear()
            read(model, text)
            looks = list(reads)
            assert looks and all(entry['stack'] is not None for entry in looks)
            with torch.no_grad():
                for entry in looks:
                    without = reader.units(entry['x'], entry['state'])
                    assert torch.equal(entry['units'], without)
                    assert torch.equal(request(model, entry['x'], entry['state'], entry['stack']),
                                       request(model, entry['x'], entry['state'], None))
            print(text, 'looks', len(looks), 'depths', [int(e['stack'][STMChunks.DEPTH][0]) for e in looks],
                  'units with the stack equal the units without it')
    finally:
        model.End()


def test_an_older_checkpoint_without_the_stack_weight_loads_it_at_zero():
    """A reader saved before it read the stack loads, its stack weight zero."""
    from RequestRead import RequestRead
    read = RequestRead(6, 16, stack=(8, 5))
    old = {key: value for key, value in read.state_dict().items() if key != 'stack_weight'}
    with torch.no_grad():
        read.stack_weight.fill_(1.)
    read.load_state_dict(old)
    assert read.stack_weight.shape == (16, STMChunks.request_width(8, 5)) and not read.stack_weight.any()
