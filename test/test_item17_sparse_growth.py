"""Item 1.7, rule 67 in tests only: a sparse Sigma that grows and is pruned
by its penalty, against the full matrix under the same penalty, the same
learning rate and the same readings (certificate 58 of the sigma-pi
specification, §21). Nothing of it is in bin/.

The two forms are taught by the same steps on the same readings, so they
share every draw; nothing is drawn here at all (the cases and their starts
are fixed, and the lesson's readings are what the lesson read).
"""
import pytest
import torch

from sparse_growth_trial import (case_readings, certificate_23_cases, compare,
                                 omission_readings, run)

LR = .5
TOLERANCE = 1e-9
SMALL, EFFECT = .25, .01


def _certify(name, result, lam):
    print(f'\n{name}, penalty {lam}: entries written {result["written"]}, removed '
          f'{result["removed"]}, held {result["sparse"]} (full {result["full"]}); largest entry '
          f'of the full matrix the sparse form lacks {result["lacks"]:.3g}; greatest difference '
          f'{result["difference"]:.3g}; table needed {result["needed"]} of {result["table"]}, '
          f'dropped {result["dropped"]}; small entries that tell: {result["small_telling"]}')
    # The entries the sparse form holds are the full matrix's that are not
    # zero, and their weights agree.
    assert result['lacking'] == 0 and result['extra'] == 0 and result['lacks'] == 0.
    assert result['difference'] <= TOLERANCE
    # A small entry whose removal changes a concept's reading is kept by both.
    assert all(entry['sparse'] for entry in result['small_telling'])
    # The table of gathered gradients stays within its size.
    assert result['needed'] <= result['table'] and result['dropped'] == 0


@pytest.mark.parametrize('lam', (.001, .01, .1))
@pytest.mark.parametrize('name', sorted(certificate_23_cases()))
def test_58_the_sparse_sigma_is_the_full_one_on_the_learning_cases(name, lam):
    cases, rows = certificate_23_cases()[name]
    S, T, M, start, protected = case_readings(cases, rows)
    full, sparse = run(S, T, M, concepts=S.shape[1], start=start, protected=protected,
                       lam=lam, lr=LR, steps=2000, table=16)
    result = compare(full, sparse, S, small=SMALL, effect=EFFECT)
    readings = sparse.read(S.to(torch.float64))[:, -1].tolist()
    print(f'{name}: targets {list(cases.values())}, read {[round(v, 3) for v in readings]}')
    _certify(name, result, lam)
    # Entries beyond a concept's own row are needed, and are written.
    assert result['written'] > 0


def test_58_the_sparse_sigma_is_the_full_one_on_the_lessons_readings():
    """The lesson's sixteen sentences read at a learning rate of zero; the
    rows the order-zero Sigma read, each reading with one word's row hidden
    and every concept taught its value on the whole reading."""
    from test_item17_symbols import _read_lesson_at_start
    model, patch, fields = _read_lesson_at_start(16)
    try:
        owner = model._concept_owner()
        alloc = owner._concept_allocator
        # A place's row reads the where wires, not a word's form: the
        # lesson's rows are its words'.
        places = owner.basic_percepts.place_rows
        own = {row: percept for percept, row in alloc.brought.items() if percept not in places}
        declared = {(row, percept) for row, percept, _ in alloc.order_zero_members if percept not in places}
        made = omission_readings(fields, own, declared)
    finally:
        model.End()
        patch.undo()
    if made is None:
        pytest.fail('the lesson read no reading with two nominated rows')
    S, T, M, start, protected, rows, concepts = made
    print(f'\nlesson: {len(concepts)} concepts on {len(rows)} rows, {S.shape[0]} readings')
    # A mark is a word (2e73d15cbe), so the lesson's readings hold more rows
    # together: its missing entries need more than the sixteen places the
    # table had (20 as read today), and a table that drops what it needs
    # does not follow the full matrix. The trial's table is sized for them.
    for lam in (.001, .01, .1):
        full, sparse = run(S, T, M, concepts=len(concepts), start=start, protected=protected,
                           lam=lam, lr=LR, steps=300, table=32)
        _certify('lesson', compare(full, sparse, S, small=SMALL, effect=EFFECT), lam)
