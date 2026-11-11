"""Definition owner for isolated conceptual-inventory mechanism fixtures."""
from Memory import LTM0


def with_definitions(cs):
    from ClauseRow import attach_clause_index
    from types import SimpleNamespace
    from BasicPercepts import BasicPercepts
    from Layers import RadixLayer
    native = RadixLayer(cs.nDim, initial_cap=16384)
    basic = BasicPercepts(native, cs.nDim, 0, 16384)
    object.__setattr__(cs, '_fixture_percepts', native)
    cs._fixture_basic_percepts = basic
    object.__setattr__(cs, '_basic_percepts', basic)
    if not hasattr(cs, '_model'):
        object.__setattr__(cs, '_model', SimpleNamespace())
    store = LTM0(cs.nDim, capacity=1024)
    object.__setattr__(cs, '_fixture_occurrences', store)
    attach_clause_index(SimpleNamespace(), store, cs)
    return cs


def agreeing_views(owner, *identities, count=3):
    """Supply explicit agreeing content views, without changing row arithmetic."""
    from ConceptTrust import observe
    bank = owner.similarity_codebook
    for identity in identities:
        row = owner._csw_row_of(identity)
        assert row is not None, 'supplied concept needs its own seat'
        content = (bank.trust.content[row] if bool(bank.trust.known[row])
                   else bank.lookup_rows(row)).detach().clone()
        for _ in range(count):
            observe(owner, identity, content)


def projected_content(owner, identity, value):
    """Expected promoted content: unchanged form, meaning direction times radius."""
    import torch
    bank = owner.similarity_codebook
    trust = bank.trust
    expected = value.clone()
    if trust.tracked:
        row = owner._csw_row_of(identity)
        start = trust.start
        direction = bank.mereology.placement[row]
        expected[start:start+trust.width] = torch.nn.functional.normalize(direction, dim=0) * trust.lengths[row,0]
    return expected


def witness_object(cs, word, *, point=None):
    """Supply an object with the form's observed parts, then witness its DEF.

    Reading an unfamiliar form does not itself testify that it has an object.
    Definition tests supply that testimony explicitly.
    """
    value = cs.interpret._pending[word]
    obj = cs.new_concept()
    for parts in value['parts']:
        for part in parts:
            cs.add_part(obj, part)
        cs._csw_concept_row(0, obj)
        cs._populate_concept_weights(
            obj, witness=(parts, value['wholes']), word_reading=True)
    for whole in value['wholes']:
        cs.add_whole(obj, whole)
    agreeing_views(cs, word, obj)
    row = cs.interpret.define(word, obj)
    assert row is not None and cs.definitions.row(word, obj) == row
    if point is not None:
        import torch
        with torch.no_grad():
            cs.similarity_codebook.W[cs._csw_row_of(obj)].copy_(point)
    return obj


def learned_operation(occupations, operation, *occupants, count):
    """Supply kept input occupations through observe_kept's application writer.

    Each argument receives its fixture word(s), without inventing an output
    occupation. Output/head flow is tested by complete kept programs elsewhere.
    """
    if type(count) is not int or count < 1:
        raise ValueError('a learned operation needs a positive kept-use count')
    operands = tuple((int(words),) if isinstance(words, int) else tuple(words)
                     for words in occupants)
    for _ in range(count):
        occupations.observe_use(operation, operands)
        occupations.observe_seat(operation, operands)
        from ConceptHierarchy import refresh
        refresh(occupations.owner)
