"""The trial of a row of the first Pi that reads the where and the when of its
reading (step 20 of round 6): the claims of its measures (i) and (iii).
The learning measure (ii) is printed by ``looks_where_trial.py``."""
import torch

from looks_round2 import Bank, context, lexicon
from looks_where_trial import N_WHERE, PiWhere, Reader, own_place_shares, read


def test_a_row_passed_by_its_entries_has_its_largest_share_at_its_own_place():
    for forward in (True, False):
        starts, rows = own_place_shares(forward=forward)
        for centre in (starts[0], starts[2], starts[4]):
            entries = rows[(centre, 'entries')]
            assert starts[max(range(len(entries)), key=entries.__getitem__)] == centre
            assert entries[starts.index(centre)] > 0
            # Passed by the derivative of its product, both lanes through the
            # wire's value, the row has no share at its own place.
            combined = rows[(centre, 'combined')]
            assert abs(combined[starts.index(centre)]) < 1e-9
            assert starts[max(range(len(combined)), key=combined.__getitem__)] != centre


def test_from_entries_at_nothing_the_entries_take_a_gradient_and_the_request_none():
    trial = PiWhere(lexicon())
    reader = Reader(trial, torch.Generator().manual_seed(0), noise=0.)
    Ep = trial.Ep.clone().requires_grad_()
    Em = trial.Em.clone().requires_grad_()
    bank = Bank.read('This is.', trial.lex)
    loss, _ = read(trial, reader, bank, train=True, Ep=Ep, Em=Em)
    loss.backward()
    assert reader.A.grad is None or float(reader.A.grad.norm()) == 0.
    where = torch.cat([Ep.grad[:, :N_WHERE], Em.grad[:, :N_WHERE]])
    assert float(where.norm()) > 0
    # Rows present in the sentence take it; an absent row, held at the Pi's
    # floor, takes a millionth or less of it.
    present = set(bank.row.tolist())
    size = where.abs().sum(-1)[:trial.lex.R] + where.abs().sum(-1)[trial.lex.R:]
    assert {int(r) for r in (size > 1e-6 * size.max()).nonzero()} <= present


def test_the_cone_over_a_place_keeps_its_where_and_drops_the_whats_that_vary():
    from looks_where_trial import place_forms
    starts, given, (Pw, Pp, Pm), record = place_forms(False)
    by_place = {r['place']: r for r in record}
    assert [r['place'] for r in record] == starts
    for place in (0, 8):
        # Met by forms that share no character: a place, with no what.
        assert by_place[place]['whats'] == []
        j = starts.index(place)
        assert float(Pw[j].abs().sum()) == 0.
        assert torch.allclose(Pp[j], given[1][j]) and torch.allclose(Pm[j], given[2][j])
    # Met by one form only, the row keeps that form's whats.
    assert 'is' in by_place[5]['whats']


def test_a_place_concept_asked_makes_its_own_reading_the_most_salient():
    from looks_where_trial import PlaceRequest, PlaceTrial, given_places
    from looks_round2 import DTYPE
    lex = lexicon()
    trial = PlaceTrial(lex, *given_places(PiWhere(lex), [0, 5, 8]))
    bank = Bank.read('This is Felix.', lex)
    starts = [bank.spans[k][0] for k in range(len(bank.spans))]
    for j, place in enumerate((0, 5, 8)):
        asked = torch.zeros(3, dtype=DTYPE)
        asked[j] = 1.
        request = PlaceRequest(words=1., concepts=torch.zeros(lex.R, dtype=DTYPE), places=asked)
        _, s, _, _ = trial.shares(bank, bank.live(), request)
        words = bank.word_tokens()
        assert starts[max(words, key=lambda k: float(s[k]))] == place
