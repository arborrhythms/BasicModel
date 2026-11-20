"""Item 1.7: the trial of the balanced code (sigma-pi §19.69, §19.70, §22.3b).

The point is numbers, printed. Assertions are only what holds by
construction (the mereological order of a code against its parts and its
wholes) or that the trial's stand-in codes the source's way. Draws come
from the system and are printed.
"""
import functools
from collections import Counter
import os

import numpy as np
import pytest
import torch

import code_trial as ct

WIDTHS = (64, 96, 128)


@functools.lru_cache(maxsize=None)
def words():
    return tuple(ct.dictionary())


@functools.lru_cache(maxsize=None)
def coder(layout, n, alternate):
    return ct.Coder(layout, n, alternate=alternate).admit_all(words())


@functools.lru_cache(maxsize=None)
def dictionary_holders():
    return ct.holders_of(words())


def layouts():
    yield 'built 96+256', coder('built', 64, False)
    for n in WIDTHS:
        yield f'{n}+{n} part only', coder('balanced', n, False)
        yield f'{n}+{n} part or whole', coder('balanced', n, True)


def _p5_combine(presence, codes):
    """P5's combination as the trial measured it: each row's one code taken
    at its for value onto the for lane and at its against value onto the
    against lane, ``1 - prod (1 - c)^b`` on each (the source now enters a
    row once at its net, with a read pair; OrderZero.combine)."""
    gaps = torch.log((1 - codes).clamp_min(1e-12))
    return 1 - torch.exp(torch.einsum('p...l,pw->...lw', presence, gaps))


# 0. The stand-in codes forms as WordIdentity does.

def test_the_stand_in_codes_as_word_identity_does():
    """On the first 3,000 words, the trial's coder as built and
    WordIdentity give the same row to every form that holds no part the two
    made differently. (Since round 3c, step 7, WordIdentity gives two forms
    with one code rule 56's next part where the stand-in raised bits or
    shared the row; the forms those parts touch are counted.)"""
    from Layers import RadixLayer
    from WordIdentity import WordIdentity, text_of
    sample = words()[:3000]
    # The source's parts half is 64 wide (rule 50); the stand-in as built
    # keys forms on their parts alone, as WordIdentity does with no wholes half.
    store = RadixLayer(64, initial_cap=8000)
    identity = WordIdentity(store, pair_dim=64)
    for w in sample:
        identity.admit(w.encode())
    mine = ct.Coder('built').admit_all(sample)
    runs = lambda bits, added: {run for run in added} | {str(atom[1] if isinstance(atom, tuple) else atom)
                                                          for atom in bits}
    apart = runs(identity.bits, identity.added) ^ runs(mine.bits, mine.added)
    apart |= {str(a[1] if isinstance(a, tuple) else a) for a in identity.bits} | {str(a) for a in mine.bits}
    apart = {text_of(r) if isinstance(r, bytes) else r for r in apart}
    kept = [w for w in sample if not any(run in w for run in apart) and w not in mine.shared
            and w not in mine.shared.values()]
    differ = sum(int(np.any(identity.form(w.encode()).numpy() != mine.parts_vector(w)[:64])) for w in kept)
    print(dict(words=len(sample), compared=len(kept), rows_that_differ=differ,
               shared=(0, len(mine.shared)), added=(len(identity.added), len(mine.added))))
    assert kept and differ == 0


# 1. The dictionary.

def test_1_the_dictionary_under_each_layout():
    print()
    for name, c in layouts():
        s = c.summary()
        a = c.accidents()
        print(name, s)
        print('   ', a)
        made = [sum(m in c.whole_of[w] for w in c.words) for m in c.wholes if m.startswith('made/')]
        if made:
            print('    forms under each made whole: mean', round(float(np.mean(made)), 1), 'greatest', max(made),
                  'forms in 1+ made whole', sum(any(x.startswith('made/') for x in c.whole_of[w]) for w in c.words))
        # By construction: each form's parts half is at or above each part's
        # code, and its wholes half at or below each whole's.
        for w in c.words[::997]:
            p = c.part_mask(w)
            assert all((c.atom_mask(atom) & ~p) == 0 for atom in c.parts[w])
            for whole in c.whole_of[w]:
                if c.layout == 'built':
                    assert (c.wholes_mask(w) & ~c.whole_code_mask(whole)) == 0
                else:
                    assert (c.whole_code_mask(whole) & ~c.wholes_mask(w)) == 0


def test_1_the_wholes_half_compared_where_it_differs():
    """The wholes half of a lowercase word is the same in nearly every word;
    count only the pairs whose wholes halves differ."""
    for name, c in layouts():
        P, W = c.matrices()
        wholes = [c.whole_of[w] for w in c.words]
        true = accident = 0
        for i in range(0, len(c.words), 40):
            le = (~W[i] | W).all(1) & (W != W[i]).any(1)
            for j in np.flatnonzero(le):
                if wholes[j] <= wholes[i]:
                    true += 1
                else:
                    accident += 1
        distinct = len({W[i].tobytes() for i in range(len(W))})
        print(name, dict(distinct_wholes_halves=distinct, strictly_below_true=true, strictly_below_accident=accident))


# 2. The wholes alone.

def test_2_the_declared_wholes_and_more_by_coordinates_of_their_own():
    seeds = ct.drawn(1)
    print(dict(seed=seeds[0]))
    for n in WIDTHS:
        for count in (9, 50, 200):
            wholes = ct.declared_wholes() if count == 9 else ct.nested_wholes(count, seeds[0])
            r = ct.whole_order(wholes, n)
            print(dict(n=n, **r))
            assert r['not_ordered'] == 0
    # The nine declared, at 64: what each lies in.
    c = ct.Coder('balanced', 64)
    for name in c.wholes:
        print(name, 'lies in', [o for o in c.ancestors(name) if o != name],
              'removed coordinates', ct.popcount(c.whole_code_mask(name)))


def test_2_a_set_of_wholes_as_an_address():
    """The wholes half as the key of a set of wholes: one probe of a
    dictionary, exact while no two sets fold to one mask."""
    for n in WIDTHS:
        c = coder('balanced', n, True)
        sets = {}
        for w in c.words:
            sets.setdefault(c.whole_of[w], c.wholes_mask(w))
        masks = {}
        for s, m in sets.items():
            masks.setdefault(m, []).append(s)
        clashes = sum(len(v) - 1 for v in masks.values() if len(v) > 1)
        print(dict(n=n, wholes=len(c.wholes), distinct_sets_of_wholes=len(sets), sets_sharing_a_mask=clashes))


# 3. The lesson.

def _lesson_forms(owner, basic):
    """The nominated rows and their forms (the lexical form of each row's
    word; a row's run of parts holds each character once)."""
    rows = [r for r in range(len(basic.index)) if bool(basic.promoted[r])]
    word_of = {row: word for word, row in basic.words.items()}
    forms = []
    for r in rows:
        form = owner.interpret.lexical_form(word_of[r])
        forms.append(form.decode() if isinstance(form, bytes) else str(form))
    return rows, forms


@pytest.fixture(scope='module')
def lesson():
    from test_item17_order_zero import _read_lesson
    model, patch = _read_lesson(16)
    owner = model._concept_owner()
    yield dict(model=model, owner=owner, basic=owner.basic_percepts,
               identity=model.perceptualSpace.percept_store.identity,
               store=model.perceptualSpace.percept_store)
    model.End()
    patch.undo()


def test_3_the_lessons_concepts_under_each_layout(lesson):
    owner, basic, identity, store = lesson['owner'], lesson['basic'], lesson['identity'], lesson['store']
    rows, forms = _lesson_forms(owner, basic)
    vocabulary = [w.decode() for w in identity.words]
    print(dict(nominated=forms, vocabulary=vocabulary))
    codes = basic.codes[rows].double().numpy()
    concepts = [owner.concept_of_basic_percept(r) for r in rows]
    model_cols = owner.order_zero.weight[:basic.width, concepts].detach().double().numpy().T
    print('model as built', ct.told_apart(ct.readings(codes, model_cols)))
    print('model codes, 8b columns', ct.told_apart(ct.readings(codes)))
    built = ct.Coder('built').admit_all(vocabulary)
    stand_in = np.stack([built.parts_vector(f)[:64] for f in forms])
    # The stand-in's parts half is the model's, 64 wide (rule 50).
    assert np.array_equal(stand_in, codes[:, :64])
    print('stand-in as built', ct.told_apart(ct.readings(np.stack([built.vector(f) for f in forms]))))
    holders = ct.holders_of(vocabulary)
    for n in WIDTHS:
        for alternate in (False, True):
            c = ct.Coder('balanced', n, alternate=alternate).admit_all(vocabulary)
            index = [c.words.index(f) for f in forms]
            label = f'{n}+{n} ' + ('part or whole' if alternate else 'part only')
            unsigned = np.stack([c.vector(f) for f in forms])
            print(label, 'P2 unsigned', ct.told_apart(ct.readings(unsigned)))
            print(label, 'P2 parts half only', ct.told_apart(ct.readings(unsigned[:, :n])))
            s, both = ct.signed(c, holders)
            print(label, 'P6 signed', ct.told_apart(ct.readings(s[index])), both)
            b, both = ct.signed(c, broad=True)
            print(label, 'P6 broad', ct.told_apart(ct.readings(b[index])), both)
    c = ct.Coder('balanced', 64).admit_all(vocabulary)
    R = ct.readings(ct.signed(c, holders)[0][[c.words.index(f) for f in forms]])
    U = ct.readings(np.stack([c.vector(f) for f in forms]))
    print('64+64 readings before the squash, row = reading, column = concept', forms)
    for f, u, r in zip(forms, U, R):
        print(f'{f:>6}', 'unsigned', np.round(u, 2).tolist(), 'signed', np.round(r, 2).tolist())


# 6 (a). A form and its holder: what each one's concept reads of the other.

def test_6a_a_part_and_its_whole_read_unsigned_and_signed():
    holders = dictionary_holders()
    for n in WIDTHS:
        c = coder('balanced', n, True)
        unsigned = np.stack([c.vector(w) for w in c.words])
        s, both = ct.signed(c, holders)
        b, both_b = ct.signed(c, broad=True)
        index = {w: i for i, w in enumerate(c.words)}
        P, _ = c.matrices()
        true, accidental = [], []
        for i in range(0, len(c.words), 40):
            le = (~P[i] | P).all(1)
            le[i] = False
            for j in np.flatnonzero(le):
                (true if c.parts[c.words[i]] <= c.parts[c.words[j]] else accidental).append((i, j))

        def read(codes, pairs, direction):
            cols = ct.columns(codes)
            if direction == 'part reads whole':
                v = np.array([codes[j] @ cols[i] for i, j in pairs])
            else:
                v = np.array([codes[i] @ cols[j] for i, j in pairs])
            return dict(mean=round(float(v.mean()), 3), least=round(float(v.min()), 3),
                        greatest=round(float(v.max()), 3), at_one_or_more=int((v >= 1 - 1e-9).sum()))

        print(dict(n=n, true_pairs=len(true), accidental_pairs=len(accidental), both_coordinates=both,
                   broad_both=both_b,
                   against_parts_mean=round(float((s[:, :n] < 0).sum(1).mean()), 2),
                   against_wholes_mean=round(float((s[:, n:] < 0).sum(1).mean()), 2),
                   broad_against_parts_mean=round(float((b[:, :n] < 0).sum(1).mean()), 2),
                   wholes=len(c.wholes)))
        for name, codes in (('unsigned', unsigned), ('signed', s), ('broad', b)):
            for direction in ('part reads whole', 'whole reads part'):
                print('  ', name, direction, 'true', read(codes, true, direction),
                      'accidental', read(codes, accidental, direction))
        for part, whole in (('archer', 'archery'), ('is', 'this'), ('ab', 'abaca')):
            if part in index and whole in index:
                i, j = index[part], index[whole]
                print('   ', part, whole, {name: (round(float(codes[j] @ ct.columns(codes[[i]])[0]), 3),
                                              round(float(codes[i] @ ct.columns(codes[[j]])[0]), 3))
                                       for name, codes in (('unsigned', unsigned), ('signed', s), ('broad', b))})
        # By construction: on the for lane of the parts half a part lies at
        # or below its holder.
        for i, j in true[:200]:
            assert np.all(np.clip(s[i, :n], 0, None) <= np.clip(s[j, :n], 0, None))
        # Distinct codes under the signed layout.
        print('   distinct signed codes', len({row.tobytes() for row in s}), 'of', len(s))


# 4. Centroids.

def test_4_centroids_of_classes():
    seed = ct.drawn(1)[0]
    rng = np.random.default_rng(seed)
    print(dict(seed=seed))
    proposal = coder('balanced', 64, True)
    made = max((n for n in proposal.wholes if n.startswith('made/')),
               key=lambda n: sum(n in proposal.whole_of[w] for w in proposal.words))
    added = max(proposal.added, key=lambda r: proposal.held[r])
    classes = {
        "holds 'q'": lambda c, w: 'q' in c.parts[w],
        f"holds the added part '{added}'": lambda c, w: added in w,
        f'under the made whole {made} ({len(proposal.wholes[made])} bytes)':
            lambda c, w: frozenset(w.encode()) <= proposal.wholes[made],
    }
    random_members = set(rng.choice(len(words()), 300, replace=False).tolist())
    holders = dictionary_holders()
    for name, c in (('built 96+256', coder('built', 64, False)), ('64+64 P2', proposal),
                    ('96+96 P2', coder('balanced', 96, True))):
        codes = np.stack([c.vector(w) for w in c.words])
        split = c.parts_n + c.empty
        variants = [('unsigned', codes)]
        if c.layout != 'built':
            variants.append(('P6 signed', ct.signed(c, holders)[0]))
        for variant, values in variants:
            for label, test in classes.items():
                members = [i for i, w in enumerate(c.words) if test(c, w)]
                print(name, variant, label, ct.centroid_report(values, members, split))
            print(name, variant, 'random 300', ct.centroid_report(values, sorted(random_members), split))


def test_4_the_lessons_centroid(lesson):
    identity = lesson['identity']
    vocabulary = [w.decode() for w in identity.words]
    rows, forms = _lesson_forms(lesson['owner'], lesson['basic'])
    holders = ct.holders_of(vocabulary)
    for name, c in (('built', ct.Coder('built').admit_all(vocabulary)),
                    ('64+64', ct.Coder('balanced', 64, alternate=True).admit_all(vocabulary))):
        members = [c.words.index(f) for f in forms]
        codes = np.stack([c.vector(w) for w in c.words])
        print(name, 'unsigned, the nominated forms', ct.centroid_report(codes, members, c.parts_n + c.empty))
        if c.layout != 'built':
            print(name, 'P6 signed, the nominated forms',
                  ct.centroid_report(ct.signed(c, holders)[0], members, c.parts_n))


# 5. Scalars.

def _scalar_table(read, values=(1, 2, 4)):
    return {x: {k: round(float(v), 3) for k, v in read(x).items()} for x in values}


def test_5_scalar_collections():
    collections = dict(alpha=1, beta=1, gamma=2)
    f = ct.felt
    # As built, the code alone: the felt value at coordinates 95 and 351.
    def built(v):
        code = np.zeros(352)
        code[95] = code[351] = f(v)
        return code
    cols = {k: ct.columns(built(v)[None])[0] for k, v in collections.items()}
    print('as built, the code alone', _scalar_table(lambda x: {k: np.tanh(built(x) @ u) for k, u in cols.items()}))

    # P3: an intensity a whole of its own kind; its 3 coordinates hold the
    # value, in the parts half as a lower bound and in the wholes half as an
    # upper bound (all other wholes coordinates one).
    n = 64
    kind_p, kind_w = ct.coords(b'\x00kind/intensity/parts', n), ct.coords(b'\x00whole/intensity', n)
    def p3(v):
        code = np.concatenate((np.zeros(n), np.ones(n)))
        code[list(kind_p)] = f(v)
        code[[n + i for i in kind_w]] = f(v)
        return code
    cols = {k: ct.columns(p3(v)[None])[0] for k, v in collections.items()}
    print('P3 64+64, unsigned', _scalar_table(lambda x: {k: np.tanh(p3(x) @ u) for k, u in cols.items()}))

    # P3 with P5's lanes: each value percept's row reads the input; for, its
    # parts (the input at or above it); against, its whole (the input above
    # it), graded as meet_ratios grades, or crisp. The concepts read the
    # combined vector of the rows present.
    for grade in ('graded', 'crisp'):
        def lanes_for(v, x):
            fv, fx = f(v), f(x)
            present = min(1., fx / fv)
            against = (1 - fv / fx if grade == 'graded' else 1.) if fx > fv + 1e-12 else 0.
            return present, against

        def read(x, grade=grade):
            seen = sorted(set(collections.values()) | {x})
            presence = torch.tensor([[lanes_for(v, x)] for v in seen], dtype=torch.float64)
            presence = presence.reshape(len(seen), 1, 2)
            codes = torch.tensor(np.stack([p3(v) for v in seen]))
            combined = _p5_combine(presence, codes)[0].numpy()
            return {k: np.tanh((combined[0] - combined[1]) @ u) for k, u in cols.items()}
        print(f'P3 + P5 lanes, {grade}', _scalar_table(read))

        def bare(v):
            code = p3(v)
            code[n:] = 0
            code[[n + i for i in kind_w]] = f(v)
            return code
        bare_cols = {k: ct.columns(bare(v)[None])[0] for k, v in collections.items()}

        def read_bare(x, grade=grade):
            seen = sorted(set(collections.values()) | {x})
            presence = torch.tensor([[lanes_for(v, x)] for v in seen], dtype=torch.float64).reshape(len(seen), 1, 2)
            combined = _p5_combine(presence, torch.tensor(np.stack([bare(v) for v in seen])))[0].numpy()
            return {k: np.tanh((combined[0] - combined[1]) @ u) for k, u in bare_cols.items()}
        print(f'P3 + P5 lanes, {grade}, wholes half zero but the kind', _scalar_table(read_bare))

    # Alec's sorted table: a value's parts are the values at or below it, its
    # wholes the values above (-1). Columns made from the member's code as the
    # table stands when the collection is made (stale), or from the member's
    # code now (re-made).
    for mode in ('exact', 'bins', 'hashed'):
        for remake in (False, True):
            table = ct.ValueTable(64, mode)
            made = {}
            for k, v in collections.items():
                made[k] = table.code(v)

            def read(x, table=table, made=made, remake=remake):
                code = table.code(x)
                width = len(code)
                result = {}
                for k, v in collections.items():
                    member = table.code(v) if remake else made[k]
                    u = ct.columns(ct.pad([member], width))[0]
                    result[k] = np.tanh(ct.pad([code], width)[0] @ u)
                return result
            rows = _scalar_table(read, (1, 2, 4, 1.5))
            print(f'table {mode}, columns {"re-made" if remake else "as made"}', rows,
                  'values', table.values)


def test_5_the_two_axes_of_the_numeric_xor():
    """Each axis's level percept at level one, read by each axis concept in
    the four cases; the conjunction feature is one on (1,1) only."""
    n = 64
    value = ct.coords(b'\x00value/1.0', n)
    kinds = {axis: ct.coords(b'\x00whole/axis/' + axis.encode(), n) for axis in 'xy'}

    def built(axis):
        code = np.zeros(352)
        code[95] = code[351] = ct.felt(1)
        return code

    def p2(axis):
        code = np.concatenate((np.zeros(n), np.ones(n)))
        code[list(value)] = 1
        code[[n + i for i in kinds[axis]]] = 0
        return code

    def p6(axis):
        code = np.zeros(2 * n)
        code[list(value)] = 1
        other = 'y' if axis == 'x' else 'x'
        code[[n + i for i in kinds[other]]] = -1
        code[[n + i for i in kinds[axis]]] = 1
        return code

    cases = ((0, 0), (1, 0), (0, 1), (1, 1))
    for name, coder_ in (('as built', built), ('P2 64+64', p2), ('P6 64+64', p6)):
        cols = {a: ct.columns(coder_(a)[None])[0] for a in 'xy'}
        features = []
        for case in cases:
            present = [coder_(a) for a, on in zip('xy', case) if on]
            if not present:
                features.append((0., 0., 0.))
                continue
            x = ct.combine_lanes(present)
            features.append((float(np.tanh((x[0] - x[1]) @ cols['x'])),
                             float(np.tanh((x[0] - x[1]) @ cols['y'])), float(case == (1, 1))))
        F = np.array(features)
        design = np.column_stack((np.ones(4), F))
        target = np.array([0., 1., 1., 0.])
        w, *_ = np.linalg.lstsq(design, target, rcond=None)
        print(name, dict(features={c: tuple(round(v, 4) for v in f) for c, f in zip(cases, features)},
                         axes_alike=bool(np.allclose(F[:, 0], F[:, 1])),
                         condition=float(np.linalg.cond(design)),
                         xor_residual=float(np.abs(design @ w - target).max()), weights=np.round(w, 3).tolist()))


# P7. Wholes learned downward under weights of zero or above (a toy).

def test_p7_learned_wholes_half_toy():
    seed = ct.drawn(1)[0]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sample = [words()[i] for i in rng.choice(len(words()), 2000, replace=False)]
    classes = {"holds 'er'": lambda w: 'er' in w, "holds 'ter'": lambda w: 'ter' in w,
               "holds 'in'": lambda w: 'in' in w,
               "holds 'an'": lambda w: 'an' in w, "holds 'z'": lambda w: 'z' in w,
               'nine letters or more': lambda w: len(w) >= 9,
               'random tenth': (lambda r: (lambda w: r.random() < .1))(np.random.default_rng(seed + 1))}
    M = torch.tensor([[float(test(w)) for test in classes.values()] for w in sample], dtype=torch.float64)
    n = 64
    print(dict(seed=seed, class_sizes=M.sum(0).tolist()))
    for form in ('union', 'sigma'):
        for against in (False, True):
            W = torch.nn.Parameter(torch.rand(len(classes), n, dtype=torch.float64) * .1)
            start_cos = None
            optimizer = torch.optim.Adam([W], lr=.02)

            def half(memberships):
                if form == 'union':
                    return 1 - torch.exp(memberships @ torch.log((1 - W).clamp_min(1e-9)))
                return torch.tanh(memberships @ W)

            def read():
                h = half(M)
                if against:
                    h = h - half(1 - M)
                cols = W / W.square().sum(1, keepdim=True).clamp_min(1e-12)
                return torch.tanh(h @ cols.T)

            def cosines():
                w = W.detach()
                u = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)
                c = u @ u.T
                return float(c[~torch.eye(len(c), dtype=torch.bool)].mean())
            start_cos = cosines()
            for _ in range(600):
                optimizer.zero_grad()
                loss = (read() - .76 * M).square().mean()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    W.clamp_(0, 1)
            with torch.no_grad():
                r = read().numpy()
                h = half(M)
                order = []
                for k in range(len(classes)):
                    members = M[:, k] > 0
                    above = bool((h[members] >= W[k] - 1e-9).all())
                    below = bool((h[members] <= W[k] + 1e-9).all())
                    order.append('above' if above else 'below' if below else 'neither')
                aucs = [round(ct.auc(r[M[:, k].numpy() > 0, k], r[M[:, k].numpy() == 0, k]), 3)
                        for k in range(len(classes))]
                support = (W > .05).sum(1).tolist()
                er, ter = W[0], W[1]
                nested = dict(ter_at_or_above_er=bool((ter >= er - 1e-6).all()),
                              ter_at_or_below_er=bool((ter <= er + 1e-6).all()),
                              shared_coordinates=int(((er > .05) & (ter > .05)).sum()))
                shared = int(((W > .05).sum(0) > 1).sum())
            print(dict(form=form, against_lane=against, loss=round(float(loss.detach()), 4),
                       member_vs_class_code=order, read_back_auc=aucs,
                       mean_cosine_between_classes=(round(start_cos, 3), round(cosines(), 3)),
                       coordinates_per_class=support, coordinates_used_by_several=shared, nested_er_ter=nested))


# §19.71, point 2: a row of the first Pi with against entries.

def test_6b_the_first_pi_with_against_entries():
    c = coder('balanced', 64, True)
    holders = dictionary_holders()
    against = ct.against_entries(c, holders)
    counts = [len(against[w]) for w in c.words]
    total = [len(c.parts[w]) + len(against[w]) for w in c.words]
    print(dict(against_entries=(min(counts), round(float(np.mean(counts)), 2), max(counts)),
               rows_with_none=sum(x == 0 for x in counts),
               all_entries_a_row=(min(total), round(float(np.mean(total)), 2), max(total)),
               rows_over_16=sum(x > 16 for x in total)))
    P, _ = c.matrices()
    pairs = {'held as a run': [], 'parts inside, not a run': [], 'accidental (code only)': []}
    for i in range(0, len(c.words), 40):
        le = (~P[i] | P).all(1)
        le[i] = False
        for j in np.flatnonzero(le):
            a, b = c.words[i], c.words[j]
            if c.parts[a] <= c.parts[b]:
                pairs['held as a run' if a in b else 'parts inside, not a run'].append((a, b))
            else:
                pairs['accidental (code only)'].append((a, b))
    for kind, group in pairs.items():
        for by in ('set', 'code'):
            row = {}
            for join in ('product', 'union'):
                lanes = np.array([ct.pi_row(c, a, b, against, join=join, by=by) for a, b in group])
                row[join] = dict(for_mean=round(float(lanes[:, 0].mean()), 3),
                                 against_fired=int((lanes[:, 1] > 0).sum()),
                                 net_mean=round(float((lanes[:, 0] - lanes[:, 1]).mean()), 3))
            print(kind, len(group), 'by', by, row)
    # By construction: a row reads its own form for, and never against.
    for w in c.words[::97]:
        for join in ('product', 'union'):
            assert ct.pi_row(c, w, w, against, join=join) == (1., 0.)
    for a, b in (('archer', 'archery'), ('archer', 'archers'), ('ab', 'abaca')):
        if a in c.order and b in c.order:
            print(a, 'on', b, 'against entries', sorted(against[a])[:12],
                  {join: ct.pi_row(c, a, b, against, join=join) for join in ('product', 'union')})


def test_6b_the_lessons_rows_with_against_entries(lesson):
    owner, basic, identity = lesson['owner'], lesson['basic'], lesson['identity']
    rows, forms = _lesson_forms(owner, basic)
    vocabulary = [w.decode() for w in identity.words]
    c = ct.Coder('balanced', 64, alternate=True).admit_all(vocabulary)
    against = ct.against_entries(c, ct.holders_of(vocabulary))
    print({f: sorted(against[f]) for f in forms})
    for by in ('set', 'code'):
        for join in ('product', 'union'):
            net = np.array([[np.subtract(*ct.pi_row(c, row, reading, against, join=join, by=by))
                             for row in forms] for reading in forms])
            own = np.diag(net).copy()
            other = net - np.diag(own) - 2 * np.eye(len(net))
            apart = int((own > other.max(1) + 1e-9).sum())
            print(dict(by=by, join=join, told_apart=apart, of=len(forms),
                       rows_read_both=int(sum(ct.pi_row(c, r, x, against, join=join, by=by) == (1., 1.)
                                              for r in forms for x in forms if r != x)),
                       others_net_mean=round(float(net[~np.eye(len(net), dtype=bool)].mean()), 3)))


# §19.71, point 3: the sorted table, graded by distance.

def test_5_the_sorted_table_graded_by_distance():
    collections = dict(alpha=1, beta=1, gamma=2)
    for mode in ('exact', 'bins'):
        for grade in ('crisp', 'against', 'both'):
            table = ct.ValueTable(64, mode)
            for v in collections.values():
                table.see(v)

            def read(x):
                table.see(x)
                code = ct.graded_table_code(table, x, grade)
                result = {}
                for k, v in collections.items():
                    member = ct.graded_table_code(table, v, grade)
                    width = max(len(member), len(code))
                    u = ct.columns(ct.pad([member], width))[0]
                    result[k] = np.tanh(ct.pad([code], width)[0] @ u)
                return result
            rows = _scalar_table(read, (1, 2, 4, 1.5))
            margins = {x: round(r['alpha'] - r['gamma'], 3) for x, r in rows.items()}
            print(f'table {mode}, {grade}', rows, 'alpha less gamma', margins)


# §19.72: the wholes half as an upper bound, read on its complement.

def test_7_the_wholes_half_read_on_its_complement_declared():
    c = ct.Coder('balanced', 64)
    wholes = ct.with_top(c.wholes)
    kids = ct.children(wholes)
    for name in wholes:
        cut = 0 if name == ct.TOP else ct.popcount(c.whole_code_mask(name))
        print(f'{name:>14}', dict(bytes=len(wholes[name]), cuts=cut, directly_inside=list(kids[name])))
    c.admit_all(['cat', 'Cat', 'CAT', 'c4t', 'is'])
    for w in c.words:
        f, a, witnessed = ct.complement_code(c, w, kids=kids)
        print(w, dict(wholes=sorted(c.whole_of[w]), cut=int(f[64:].sum()),
                      against=witnessed, against_coordinates=int(a[64:].sum())))
        # Rule 5 on the for lane, by construction: the wholes half (all ones
        # less the cut) is at or below each of its wholes' codes, and below
        # the top's (all ones).
        upper = 1 - f[64:]
        for name in c.whole_of[w]:
            whole = 1 - ct.bits(c.whole_code_mask(name), 64)
            assert np.all(upper <= whole)


def test_7_the_wholes_half_read_on_its_complement_dictionary():
    seed = ct.drawn(1)[0]
    print(dict(seed=seed))
    c = coder('balanced', 64, True)
    holders = dictionary_holders()
    codes_u, pairs_u = ct.complement_codes(c)
    codes_s, pairs_s = ct.complement_codes(c, holders)
    count = [len(w) for _f, _a, w in pairs_u]
    made_against = [sum(x.startswith('made/') for x in w) for _f, _a, w in pairs_u]
    coordinates = [int(a[64:].sum()) for _f, a, _w in pairs_u]
    print(dict(wholes=len(c.wholes) + 1,
               wholes_on_the_against_lane=(min(count), round(float(np.mean(count)), 1), max(count)),
               of_them_made=round(float(np.mean(made_against)), 1),
               against_coordinates_of_64=(min(coordinates), round(float(np.mean(coordinates)), 1), max(coordinates)),
               cut_coordinates_mean=round(float(np.mean([f[64:].sum() for f, _a, _w in pairs_u])), 2)))
    P, _ = c.matrices()
    true, accidental = [], []
    for i in range(0, len(c.words), 40):
        le = (~P[i] | P).all(1)
        le[i] = False
        for j in np.flatnonzero(le):
            (true if c.parts[c.words[i]] <= c.parts[c.words[j]] else accidental).append((i, j))
    for name, codes in (('complement, parts unsigned', codes_u), ('complement, parts P6', codes_s)):
        cols = ct.columns(codes)
        for direction in ('part reads whole', 'whole reads part'):
            out = {}
            for kind, group in (('true', true), ('accidental', accidental)):
                v = np.array([codes[j] @ cols[i] if direction == 'part reads whole' else codes[i] @ cols[j]
                              for i, j in group])
                out[kind] = dict(mean=round(float(v.mean()), 3), least=round(float(v.min()), 3),
                                 at_one=int((v >= 1 - 1e-9).sum()))
            print(name, direction, out)
    made = max((n for n in c.wholes if n.startswith('made/')),
               key=lambda n: sum(n in c.whole_of[w] for w in c.words))
    sample = set(range(0, len(c.words), 40))
    declared_differ = [n for n in ct.declared_wholes()
                       if len({n in c.whole_of[c.words[i]] for i in sample}) > 1]
    print(dict(declared_wholes_that_differ_in_the_sample=declared_differ, made=made,
               made_bytes=len(c.wholes[made])))
    members = [i for i, w in enumerate(c.words) if made in c.whole_of[w]]
    rng = np.random.default_rng(seed)
    random_members = sorted(rng.choice(len(c.words), 300, replace=False).tolist())
    for name, codes in (('complement, parts unsigned', codes_u), ('complement, parts P6', codes_s)):
        print(name, 'under', made, ct.centroid_report(codes, members, 64))
        print(name, 'random 300', ct.centroid_report(codes, random_members, 64))


def test_7_the_lesson_read_on_the_complement(lesson):
    owner, basic, identity = lesson['owner'], lesson['basic'], lesson['identity']
    rows, forms = _lesson_forms(owner, basic)
    vocabulary = [w.decode() for w in identity.words]
    c = ct.Coder('balanced', 64, alternate=True).admit_all(vocabulary)
    index = [c.words.index(f) for f in forms]
    holders = ct.holders_of(vocabulary)
    print(dict(forms=forms))
    unsigned = np.stack([c.vector(f) for f in forms])
    print('P2 unsigned', ct.told_apart(ct.readings(unsigned)))
    print('parts half only', ct.told_apart(ct.readings(unsigned[:, :64])))
    print('P6 signed', ct.told_apart(ct.readings(ct.signed(c, holders)[0][index])))
    for name, h in (('complement, parts unsigned', None), ('complement, parts P6', holders)):
        codes, pairs = ct.complement_codes(c, h)
        print(name, ct.told_apart(ct.readings(codes[index])))
        print(name, 'wholes half only', ct.told_apart(ct.readings(codes[index][:, 64:])))
    for f in forms:
        _f, a, witnessed = ct.complement_code(c, f)
        print(f, sorted(c.whole_of[f]), 'against', witnessed)


# §19.72: whether the two-sided code replaces the Not at the basic level.

WORDS3 = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
A, B, X, AB, AX = range(5)
TASKS = {
    'negative patterning': dict(words=WORDS3, conj=((0, 1), (0, 2)),
        cases={(A,): 1., (B,): 1., (A, B): 0.},
        probe={'A': (A,), 'B': (B,), 'A and B': (A, B), 'neither': ()},
        met=lambda r: r['A'] > .5 and r['B'] > .5 and r['A and B'] < .5),
    'conditioned inhibition': dict(words=WORDS3, conj=((0, 1), (0, 2)),
        cases={(A,): 1., (A, X): 0., (B,): 1.},
        probe={'A': (A,), 'A and X': (A, X), 'B': (B,), 'X': (X,), 'neither': ()},
        met=lambda r: r['A'] > .5 and r['A and X'] < .5 and r['B'] > .5),
    'XOR of two perceived words': dict(words=((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)),
        conj=((0, 2), (0, 3), (1, 2), (1, 3)),
        cases={(0, 2): 0., (0, 3): 1., (1, 2): 1., (1, 3): 0.},
        probe={'t1 t2': (0, 2), 't1 f2': (0, 3), 'f1 t2': (1, 2), 'f1 f2': (1, 3)},
        met=lambda r: r['t1 t2'] < .5 and r['t1 f2'] > .5 and r['f1 t2'] > .5 and r['f1 f2'] < .5),
    'X inhibits A, no row AX': dict(words=WORDS3, conj=((0, 1),),
        cases={(A,): 1., (A, X): 0., (B,): 1.},
        probe={'A': (A,), 'A and X': (A, X), 'B': (B,), 'X': (X,), 'neither': ()},
        met=lambda r: r['A'] > .5 and r['A and X'] < .5 and r['B'] > .5),
}


def test_8_whether_the_two_sided_code_replaces_the_not():
    seed = ct.drawn(1)[0]
    torch.manual_seed(seed)
    print(dict(seed=seed))
    for name, task in TASKS.items():
        for way, two_lane, nonnegative, own in (('1 as built, Not free', False, False, False),
                                               ('2 two-lane, untaught', True, None, False),
                                               ('3 two-lane, taught, weights >= 0', True, True, False),
                                               ('4 two-lane, taught, Not free', True, False, False),
                                               ('2o own coordinates, untaught', True, None, True),
                                               ('3o own coordinates, taught, weights >= 0', True, True, True),
                                               ('4o own coordinates, taught, Not free', True, False, True)):
            level = ct.BasicLevel(task['words'], task['conj'], two_lane=two_lane, own=own)
            if nonnegative is None:
                if name == 'XOR of two perceived words':
                    concepts = {'t1 f2 or f1 t2': level.column([5, 6])}
                else:
                    concepts = {'A': level.column([A]), 'B': level.column([B]), 'A, B': level.column([A, B]),
                                'AB (its row)': level.column([AB])}
                for concept, u in concepts.items():
                    readings = {k: round(level.read(u, p), 3) for k, p in task['probe'].items()}
                    print(name, '|', way, '| concept', concept, readings, 'Not entries', int((u < -.01).sum()))
                continue
            u = level.teach(task['cases'], nonnegative=nonnegative)
            readings = {k: round(level.read(u, p), 3) for k, p in task['probe'].items()}
            nots = int((u < -.01).sum())
            if nonnegative:
                assert nots == 0
            print(name, '|', way, '| met', task['met'](readings), 'Not entries', nots, readings)
        level = ct.BasicLevel(task['words'], task['conj'], two_lane=True)
        print('   against entries', {r: level.holders[r] for r in range(len(level.rows)) if level.holders[r]})


# §19.73: the wholes half on the basis of the part catalog.

COMBINATIONS = (('s', 'c'), ('s', 't'), ('i', 'c'), ('i', 't'))


def _catalog(c, combo, extra=None):
    return ct.catalog_codes(c, second=combo[0] == 'i', tight=combo[1] == 't', extra=extra)


def test_9_the_catalog_basis_of_the_wholes():
    c = coder('balanced', 64, False)
    for second in (False, True):
        basis = ct.catalog_basis(c, second)
        top = 0
        for mask in basis.values():
            top |= mask
        classes = {name: ct.popcount(ct.class_code(basis, members)) for name, members in c.wholes.items()}
        print('i' if second else 's', dict(catalog_parts=len(basis), top_ones=ct.popcount(top),
                                          coordinates_no_part_covers=64 - ct.popcount(top), classes=classes))
    for combo in COMBINATIONS:
        codes, parts, bounds, basis = _catalog(c, combo)
        keys = {(p, b) for p, b in zip(parts, bounds)}
        # Rule 5 by construction: the bound is at or below each whole's code,
        # and, on the same coordinates, the parts at or below the bound.
        for w, p, b in list(zip(c.words, parts, bounds))[::97]:
            for name in c.whole_of[w]:
                assert (b & ~ct.class_code(basis, c.wholes[name])) == 0
            if combo[0] == 's':
                assert (p & ~b) == 0
        a, b_ = c.words.index('archer'), c.words.index('archery')
        print(combo, dict(sharing=len(c.words) - len(keys), wholes_half_distinct=len(set(bounds)),
                          bound_ones=(min(map(ct.popcount, bounds)), round(float(np.mean([ct.popcount(x) for x in bounds])), 1),
                                      max(map(ct.popcount, bounds))),
                          archer_archery_differ=bool(np.any(codes[a] != codes[b_]))))
        # Parthoods of the sample, split as the first Pi's against entries were.
        P = np.array([[(p >> i) & 1 for i in range(64)] for p in parts], dtype=bool)
        groups = {'held as a run': [], 'parts inside, not a run': [], 'accidental': []}
        for i in range(0, len(c.words), 40):
            le = (~P[i] | P).all(1)
            le[i] = False
            for j in np.flatnonzero(le):
                u, v = c.words[i], c.words[j]
                key = ('held as a run' if u in v else 'parts inside, not a run') if c.parts[u] <= c.parts[v] else 'accidental'
                groups[key].append((i, j))
        cols = ct.columns(codes)
        for kind, group in groups.items():
            up = np.array([codes[j] @ cols[i] for i, j in group])
            down = np.array([codes[i] @ cols[j] for i, j in group])
            print('   ', kind, len(group), dict(part_reads_whole=(round(float(up.mean()), 3), int((up >= 1 - 1e-9).sum())),
                                              whole_reads_part=round(float(down.mean()), 3)))


def test_9_the_catalog_basis_centroids():
    seed = ct.drawn(1)[0]
    rng = np.random.default_rng(seed)
    c = coder('balanced', 64, False)
    proposal = coder('balanced', 64, True)
    made = max((n for n in proposal.wholes if n.startswith('made/')),
               key=lambda n: sum(n in proposal.whole_of[w] for w in proposal.words))
    extra = {'class of made whole 55': proposal.wholes[made]}
    random_members = sorted(rng.choice(len(c.words), 300, replace=False).tolist())
    print(dict(seed=seed, whole=made, bytes=len(proposal.wholes[made])))
    for combo in COMBINATIONS:
        codes, parts, bounds, basis = _catalog(c, combo, extra=extra)
        q = [i for i, w in enumerate(c.words) if 'q' in c.parts[w]]
        under = [i for i, w in enumerate(c.words) if set(w.encode()) <= extra['class of made whole 55']]
        print(combo, "holds 'q'", ct.centroid_report(codes, q, 64))
        print(combo, 'under the whole', ct.centroid_report(codes, under, 64))
        print(combo, 'random 300', ct.centroid_report(codes, random_members, 64))
    # A class of words coded by OR: how fast it fills, and the share of
    # non-members that lie inside it.
    basis = ct.catalog_basis(c, True)
    own = [0] * len(c.words)
    for k, w in enumerate(c.words):
        for atom in c.parts[w]:
            own[k] |= basis[atom]
    for size in (1, 2, 3, 5, 10):
        fills, inside = [], []
        for _ in range(20):
            members = rng.choice(len(c.words), size, replace=False)
            code = 0
            for k in members:
                code |= own[k]
            fills.append(ct.popcount(code))
            others = rng.choice(len(c.words), 2000, replace=False)
            inside.append(np.mean([(own[k] & ~code) == 0 for k in others if k not in members]))
        print(dict(words=size, ones=round(float(np.mean(fills)), 1), non_members_inside=round(float(np.mean(inside)), 3)))


def test_9_the_catalog_basis_on_the_lesson(lesson):
    owner, basic, identity = lesson['owner'], lesson['basic'], lesson['identity']
    rows, forms = _lesson_forms(owner, basic)
    vocabulary = [w.decode() for w in identity.words]
    c = ct.Coder('balanced', 64).admit_all(vocabulary)
    index = [c.words.index(f) for f in forms]
    print(dict(forms=forms))
    for combo in COMBINATIONS:
        codes, *_ = _catalog(c, combo)
        print(combo, ct.told_apart(ct.readings(codes[index])))


# §19.86: no code inside another by accident.

def test_10_the_linked_coder_codes_as_the_source_does():
    """The stand-in, without the repair, gives the first 3,000 words the
    source's codes (both halves) and the source's added parts."""
    from Layers import RadixLayer
    from WordIdentity import WordIdentity
    sample = words()[:3000]
    identity = WordIdentity(RadixLayer(128, initial_cap=8000), pair_dim=64, property_width=64)
    for w in sample:
        identity.admit(w.encode())
    mine = ct.LinkedCoder(64).admit_all(sample)
    differ = 0
    for w in sample:
        m = mine.code(w)
        differ += int(not np.array_equal(identity.code(w.encode()).cpu().numpy(),
                                         np.array([(m >> i) & 1 for i in range(128)], dtype=np.float32)))
    print(dict(words=len(sample), differ=differ, added=(len(identity.added), len(mine.added))))
    assert differ == 0
    assert {w.decode() if isinstance(w, bytes) else w for w in identity.added} == set(mine.added)


def _linked_report(c, label):
    accidents, true_out = c.all_accidents()
    shared = len(c.words) - len({c.code(w) for w in c.words})
    print(label, dict(words=len(c.words), accidents_all_pairs=accidents, true_parts_not_inside=true_out,
                      shared=shared, sample=ct.sample_accidents(c), failures=len(c.failures),
                      forms_given_coordinates=len(c.given_forms), **dict(c.stats)))
    print('   ones', ct.ones_in_halves(c))
    print('   ms to admit one form (mean, greatest) near', ct.admission_times(c),
          'earlier codes changed by an admission: mean', round(float(np.mean(c.changed_by)), 2),
          'greatest', max(c.changed_by), 'admissions changing any', sum(x > 0 for x in c.changed_by))
    # By construction: every form's runs among the stored forms lie inside it.
    assert true_out == 0
    return accidents


@functools.lru_cache(maxsize=None)
def linked(n, repair, count=None):
    forms = words() if count is None else words()[:count]
    return ct.LinkedCoder(n, repair=repair).admit_all(forms, index=ct.run_index(forms))


def test_10_the_source_coding_leaves_accidents():
    for n in WIDTHS:
        _linked_report(linked(n, None), f'{n}+{n} as the source codes')


def _repaired(n, repair, label):
    """All 19,821 words admitted in the order of the dictionary with the
    accidents removed as each is admitted; a cap of 3,000 repair steps an
    admission (counted and printed where it is reached)."""
    forms = words()
    c = ct.LinkedCoder(n, repair=repair, cap=3000).admit_all(forms, index=ct.run_index(forms))
    accidents = _linked_report(c, label)
    print('   failures', len(c.failures), 'example', c.failures[:2])
    return c, accidents


def test_10_accidents_removed_at_64_in_both_halves():
    """Way (c): the separating coordinate chosen among the zeros of either
    half, the one that adds the fewest ones to the form's holders. About
    150 s on a busy machine, the longest test of the file."""
    _repaired(64, 'ch', '64+64, (c) chosen in either half')


def test_10_accidents_removed_at_96_in_the_parts_half():
    _repaired(96, 'bh', '96+96, (b) chosen in the parts half')


def test_10_accidents_removed_at_128_in_the_parts_half():
    _repaired(128, 'bh', '128+128, (b) chosen in the parts half')
