"""The measure of reading a sentence's words in order (sigma-pi §19.124, 3;
round 6d, ninth pass, step 1): the interpreted arrivals of each row against
the row's live source slots, not against range(count) over padded forms."""
import torch


def spans_of(rows, width):
    """``spans`` ``[B, width, 2]`` with a one-character span at each listed
    slot of each row and empty spans elsewhere (gaps and padding)."""
    spans = torch.zeros(len(rows), width, 2, dtype=torch.long)
    for b, slots in enumerate(rows):
        for w in slots:
            spans[b, w] = torch.tensor([2 * w, 2 * w + 1])
    return spans


def test_a_correct_reading_with_gaps_and_padding_is_in_order_and_an_exchange_is_not():
    """Row 0 holds its words at slots 0, 2, 4 and 5 of 32 (Codex's corpus
    case); row 1 at slots 3 and 6 of 32, the slots before it another
    sentence's. Read in source order, both are in order; with two arrivals
    exchanged, neither is; with a word not interpreted, the row is not."""
    from AttentionLesson import in_order, live_slots
    spans = spans_of([[0, 2, 4, 5], [3, 6]], 32)
    assert live_slots(spans) == [[0, 2, 4, 5], [3, 6]]
    assert in_order([[0, 2, 4, 5], [3, 6]], spans) == [True, True]
    assert in_order([[2, 0, 4, 5], [6, 3]], spans) == [False, False]
    assert in_order([[0, 2, 5, 4], [3, 6]], spans) == [False, True]
    assert in_order([[0, 2, 4], [3, 6, 6]], spans) == [False, False]
    # The measure it replaces wanted the slots taken to be range(count):
    # the correct reading of row 0 failed it whatever count was given.
    assert all([0, 2, 4, 5] != list(range(count)) for count in (4, 32))


def test_a_supplied_reading_in_order_scores_in_order_and_the_exchanged_one_does_not(tmp_path, monkeypatch):
    """On the places' model, "This is Felix." is staged as 8 forms of which 4
    are live (padding after them). A reading supplied in the sentence's
    order is scored in order; the same reading with its first two words
    exchanged is not. The old measure (range over the staged forms) is
    printed beside."""
    from AttentionLesson import supplied_reading, looks_in_order, live_slots
    from test_item17_places import model_for, nominate, observe_runs, staged
    from test_item18_expect import read
    model = model_for(tmp_path, monkeypatch)
    try:
        nominate(model, ('This is Felix.',))
        seen = observe_runs(monkeypatch)
        text = 'This is Felix.'
        with torch.no_grad():
            read(model, text)
        model.End()
        forms, slots = staged(model), live_slots(seen[0].walk.spans)[0]
        words = [forms[w] for w in slots]
        scored = {}
        for name, parts in (('in order', words), ('exchanged', [words[1], words[0], *words[2:]])):
            seen.clear()
            with torch.no_grad(), supplied_reading(model, [parts]):
                read(model, text)
            model.End()
            orders = seen[0].source_orders()[0]
            old = orders == list(range(len(forms)))
            scored[name] = (looks_in_order(seen[0]), orders, old)
        print(dict(forms=forms, live_slots=slots, scored=scored))
        assert len(forms) > len(slots)
        assert scored['in order'][0] == [True] and scored['in order'][1] == slots
        assert scored['exchanged'][0] == [False]
    finally:
        model.End()
