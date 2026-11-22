"""The way back of reconstruction, de-muxed (sigma-pi §19.121, 4; rule 79;
round 6d of item 1.7, twelfth pass, a to e). The simulation first: on one
lesson sentence, the recreated `.where` of each reading (the first Pi's way
back at its rows, `BasicPercepts.kept_percept`, asked by the Sigma's way
back, `cs_percept_attribution`) matched against the `.where` parameter
(`WhereRegistry.code`) by shared size, a graded paste over the addresses,
and whether a cost on it passes a gradient to the parameter. Printed."""
import torch

from test_item17_places import minimal, model_for, nominate, places_of


def captured_field(model, monkeypatch, text):
    """The open field of ``text`` as reconstruction reads it."""
    import SentenceField
    from test_item18_expect import read
    reconstruct, seen = SentenceField.reconstruct, []
    def held(model, record, errors, active, **kw):
        seen.append(record.open_field)
        return reconstruct(model, record, errors, active, **kw)
    monkeypatch.setattr(SentenceField, 'reconstruct', held)
    with torch.no_grad():
        read(model, text)
    monkeypatch.setattr(SentenceField, 'reconstruct', reconstruct)
    return seen[0]


def recreated(model, field):
    """Each reading's recreated percept, de-muxed: the support of the first
    Pi's rows (the Sigma's way back), and the `.where` and `.when` the place
    rows keep, weighted by their support, ``[readings, wires, 2]`` each, with
    the support that reached place rows and other rows."""
    from SentenceField import inventory
    from BasicPercepts import held_entries, WHERE_WIRES, WHEN_WIRES
    owner = model._concept_owner()
    basic = owner.basic_percepts
    rows = inventory(owner, field)
    columns, attributed, _ = owner.cs_percept_attribution(field.evidence, observed=field.evidence, inventory_rows=rows)
    E = field.where.shape[1]
    where = torch.zeros(E, WHERE_WIRES, 2)
    when = torch.zeros(E, WHEN_WIRES, 2)
    placed, other = torch.zeros(E), torch.zeros(E)
    by_place = []
    for edge, col in enumerate(columns.tolist()):
        if col % 2:
            continue
        support = attributed[edge, 0, :, 0].detach().float().cpu()
        row = col // 2
        if row in basic.place_rows:
            w = held_entries(basic.where_entries[row].detach().float().cpu()).reshape(2, WHERE_WIRES).T
            t = held_entries(basic.when_entries[row].detach().float().cpu()).reshape(2, WHEN_WIRES).T
            where += support[:, None, None] * w
            when += support[:, None, None] * t
            placed += support
            by_place.append((row, support, w))
        else:
            other += support
    scale = placed.clamp_min(1e-12)[:, None, None]
    recreated.by_place = by_place
    return where / scale, when / scale, placed, other


import pytest


@pytest.mark.parametrize('periods', [None, (256, 32), (256, 16)])
def test_simulation_a_graded_paste_by_where_on_one_lesson_sentence(tmp_path, monkeypatch, periods):
    """``periods``: the `.where` ladder's two periods, the model's where
    ``None`` (8192 and 256 here), or a shorter pair patched in the test."""
    from BasicPercepts import field_factor
    start = int.from_bytes(__import__('os').urandom(6), 'little')
    torch.manual_seed(start)
    if periods is not None:
        import Spaces
        monkeypatch.setattr(Spaces, '_ladder_periods', lambda capacity: periods)
    model = model_for(tmp_path, monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson)
        basic, concept = places_of(model)
        registry = model.where_registry
        enc = registry.encoding
        print('start', start, 'places', len(basic.place_rows), 'with a concept', len(concept),
              'periods', enc.maxVal, enc.period_hf, 'text slice', registry.slices['text'])
        field = captured_field(model, monkeypatch, 'This is Felix.')
        model.End()
        where, when, placed, other = recreated(model, field)
        brackets = field.where[0].tolist()
        live = [r for r, (lo, hi) in enumerate(brackets) if hi > lo]
        print('readings', [(r, brackets[r]) for r in live])
        lo_, _ = model.where_registry.slices['text']
        own_code = torch.stack((model.where_registry.code[lo_:lo_ + 64].detach().float().cpu(),
                                1 - model.where_registry.code[lo_:lo_ + 64].detach().float().cpu()), -1)
        for r in [r for r, (a, b) in enumerate(field.where[0].tolist()) if b > a]:
            ranked = sorted(recreated.by_place, key=lambda e: -float(e[1][r]))[:5]
            print('reading', r, 'top place rows (support, the address their .where matches best)',
                  [(round(float(sup[r]), 3), int(field_factor(own_code[None], w[None])[0, 0].argmax()))
                   for _, sup, w in ranked])
        print('support at place rows', [round(float(placed[r]), 4) for r in live],
              'at other rows', [round(float(other[r]), 4) for r in live])
        lo, hi = registry.slices['text']
        width = max(b[1] for b in brackets) + 8
        addresses = torch.arange(width)
        code = registry.code
        lanes = lambda c: torch.stack((c, 1 - c), -1)
        candidates = lanes(code[lo:lo + width].float().cpu())               # [A, wires, 2]
        for r in live:
            if float(placed[r]) <= 0:
                print('reading', r, 'no place row asked: no .where recreated')
                continue
            match = field_factor(candidates[None], where[r][None])[0, 0]   # [A]
            true = int(brackets[r][0])
            best = int(match.argmax())
            rival = torch.cat((match[:true], match[true + 1:])).max()
            print('reading', r, 'bracket', true, 'best address', best,
                  'match at own', round(float(match[true]), 4), 'margin to best rival', round(float(match[true] - rival), 5),
                  'to the neighbours', [round(float(match[true] - match[a]), 5) for a in (true - 1, true + 1) if 0 <= a < width])
        # Compared: the mixture of each place row's own match, weighted by
        # its support, in place of the match of the supports' centroid.
        for r in live:
            mixture = sum(float(sup[r]) * field_factor(candidates[None], w[None])[0, 0]
                          for _, sup, w in recreated.by_place) / max(float(placed[r]), 1e-12)
            true = int(brackets[r][0])
            rival = torch.cat((mixture[:true], mixture[true + 1:])).max()
            print('mixture reading', r, 'bracket', true, 'best address', int(mixture.argmax()),
                  'margin to best rival', round(float(mixture[true] - rival), 5))
        # The graded paste: each reading's percept at each address in
        # proportion to its .where match there; a cost that asks it at its
        # own address, and the gradient it passes to the .where parameter.
        r = next((r for r in live if float(placed[r]) > 0), None)
        if r is not None:
            code.grad = None
            entries = where[r].clone().requires_grad_(True)
            match = field_factor(lanes(code[lo:lo + width].float())[None], entries[None])[0, 0]
            share = match / match.sum()
            cost = -share[int(brackets[r][0])].clamp_min(1e-12).log()
            cost.backward()
            print('graded paste: cost', round(float(cost), 5), 'gradient at WhereRegistry.code',
                  None if code.grad is None else float(code.grad.abs().sum()),
                  'at the recreated .where', float(entries.grad.abs().sum()))
        # How exact a recreated .where must be: the shared-size match of each
        # address's own code against its neighbours', at the model's periods.
        own = lanes(code[lo:lo + width].float().cpu().detach())
        m = field_factor(own[None], own)[:, 0]                             # [A, A]
        diffs = [float(m[a, a] - m[a, a + k]) for a in range(0, width - 4) for k in (1, 2, 3)]
        print('own address against a neighbour 1 to 3 away: least margin', round(min(diffs), 6),
              'mean', round(sum(diffs) / len(diffs), 6))
    finally:
        model.End()


def test_four_elements_against_three_for_an_axis():
    """Item d, the owner's added measure: for one axis of the `.where`, the
    shared-size margin of an address against its neighbours (1 to 3 away)
    and against every other address of a window of 256, with four elements
    (two periods, as built) against forms of three. Printed."""
    import math
    from BasicPercepts import field_factor
    a = torch.arange(256, dtype=torch.float64)
    unit = lambda x: (x + 1) / 2
    pair = lambda p: [unit(torch.sin(2 * math.pi * a / p)), unit(torch.cos(2 * math.pi * a / p))]
    forms = {
        'four: pairs at 8192 and 256 (as built)': pair(8192) + pair(256),
        'four: pairs at 256 and 16': pair(256) + pair(16),
        'four: pairs at 256 and 32': pair(256) + pair(32),
        'three: three phases of 16': [unit(torch.cos(2 * math.pi * a / 16 - k * 2 * math.pi / 3)) for k in range(3)],
        'three: three phases of 256': [unit(torch.cos(2 * math.pi * a / 256 - k * 2 * math.pi / 3)) for k in range(3)],
        'three: a pair at 16 and a ramp over 256': pair(16) + [a / 255],
        'three: a pair at 16 and a cosine of 512': pair(16) + [unit(torch.cos(2 * math.pi * a / 512))],
    }
    for name, wires in forms.items():
        code = torch.stack(wires, -1).float()
        lanes = torch.stack((code, 1 - code), -1)                   # [A, wires, 2]
        m = field_factor(lanes[None], lanes)[:, 0]                  # [A, A]
        near = min(float(m[i, i] - m[i, j]) for i in range(256) for j in (i - 3, i - 2, i - 1, i + 1, i + 2, i + 3)
                   if 0 <= j < 256)
        off = m - torch.eye(256) * 10
        alias = float((m.diagonal() - off.max(-1).values).min())
        print(f'{name}: least margin to a neighbour {near:.5f}; to any other address {alias:.5f}')
