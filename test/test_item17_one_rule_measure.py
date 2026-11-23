"""Which rule can be taught to be undone from the idea alone (round 6d of
item 1.7, eighteenth pass, item 2; sigma-pi §19.135, 6), a measure in tests
only. The lesson's model with one rule of the grammar available and no other
(``RULE``; the key ``rulesAvailable``) reads the 19 words alone, each with
its mark. The rules with parameters are taught by the model's own training
step on the words alone (the reconstruction's error and
``reconstruction.walk_policy``, nothing added), round by round; a rule with
none is measured once. After each round, input to output: of 19, the words
said back with their marks (the text the model's one reconstruction puts
back is the input); and for the rule, the cosine of each operand that the
walk's own inverse of the parent alone gives back to the operand as read
(`test_item17_free_undo_measure.FreeUndo`). At the root of each word and its
mark: whether the rule is on the walk's menu (`Generative.inverse_menu`) and,
where it is not, which of the menu's tests refuses it
(`LanguageSpace.decoder_eligibility`'s admissibility, read again here). A
round runs until every word is said back in two rounds running, or for
``ROUNDS`` (30) rounds: the measure's patience, no bar. A record: a start
drawn afresh and printed; nothing is asserted of what a start gives."""
import os
from collections import Counter

import pytest
import torch

from test_item17_free_undo_measure import FreeUndo
from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, minimal, model_for, nominate, words_of
from test_item17_reconstruction_said_back import Held as Said

WORDS = words_of(LESSON) + COUNTING[:6]
TAUGHT = ('surface', 'change', 'verb', 'modifier')
UNTAUGHT = ('part', 'whole', 'implies')


def refusal(undo, parent, action):
    """Which of the walk's menu tests refuse ``action`` at ``parent``
    ``[1, D]``: the menu's own tests (`decoder_eligibility`), read again on
    the same inputs. Empty where the action is admitted."""
    from Generative import EXACT_RESIDUAL_TOLERANCE as tolerance, in_cube
    language, bank, model = undo.language, undo.bank, undo.model
    binary = language._generate_binary_ops
    menu = undo.menu(parent)
    if bool(menu.admissible[0, action]):
        return []
    reasons = []
    if action >= len(binary):
        value = menu.left[:, action]
        if not bool(value.ne(parent).any()):
            reasons.append('the unary inverse gives its parent back')
        if not bool(in_cube(value)[0]):
            reasons.append('the operand is outside the cube')
        return reasons or ['refused by the menu (unary)']
    op = binary[action]
    live = torch.ones(1, dtype=torch.bool, device=parent.device)
    left, right, unavailable = language.reverse_binary_step(
        parent, torch.zeros(1, dtype=torch.long, device=parent.device), live, ops=(op,),
        basis=bank.codes.detach() if undo.require else None,
        basis_valid=bank.valid if undo.require else None, basis_priming=bank.weights,
        candidate_limit=model.reconstruction_basis_limit, free=undo.require, return_status=True,
        case_bank=None if bank.case_bank is None else bank.case_bank.detached(),
        whole_bank=undo.whole, action_candidates=undo.proposed)
    if bool(unavailable[0]):
        # Its operands are then zeros, and the tests below would only
        # repeat it.
        return ['the inverse finds no pair']
    content = next((getattr(o, 'gl', o).content_operand for o in binary
                    if getattr(getattr(o, 'gl', o), 'percept_band', None) is not None), lambda v: v)
    p, l, r, codes = content(parent), content(left), content(right), content(bank.codes.detach())
    layout = next((getattr(getattr(o, 'gl', o), 'meaning_layout', None) for o in binary
                   if getattr(getattr(o, 'gl', o), 'meaning_layout', None) is not None), None)
    width = layout[0] if layout is not None and layout[1] == codes.shape[-1] else codes.shape[-1]
    if not (bool(torch.isfinite(l).all()) and bool(torch.isfinite(r).all())
            and bool(((l >= 0) & (l <= 1)).all()) and bool(((r >= 0) & (r <= 1)).all())):
        reasons.append('an operand is outside the cube')
    valid = bank.valid & codes[..., :width].square().sum(-1).gt(0)
    if not bool((valid & (codes[..., :width] == l[:, None, :width]).all(-1)).any()):
        reasons.append('the first operand is no known code')
    if not bool((valid & (codes[..., :width] == r[:, None, :width]).all(-1)).any()):
        reasons.append('the second operand is no known code')
    value = p[..., :width]
    if not (bool((l[..., :width] - value).square().mean(-1).gt(tolerance).all())
            and bool((r[..., :width] - value).square().mean(-1).gt(tolerance).all())):
        reasons.append('an operand is its parent (no progress)')
    if not reasons:
        stop = menu.admissible.shape[1] - 1
        reasons.append('the parent is read exactly as one code (STOP alone is admitted)'
                       if bool(menu.exact[0, stop]) else
                       'another rule\'s pair is exact' if bool(menu.exact[0, :len(binary)].any()) else
                       'the recomposed pair is not exact while another explanation is')
    return reasons


class Undone:
    """The free undo of each reading's record, and the refusal at its root."""
    def __init__(self, model, monkeypatch):
        import SentenceField
        observe, reconstruct = model._sentence_observation, SentenceField.reconstruct
        self.held, self.rows = {}, []

        def observed(*args, **kw):
            result = observe(*args, **kw)
            self.held['entries'] = result['entries']
            return result

        def measured(model_, record, errors, active, **kw):
            result = reconstruct(model_, record, errors, active, **kw)
            row = dict(steps=[], root=None, error=None)
            try:
                with torch.no_grad():
                    program = self.held['entries'][0]
                    if program is not None:
                        undo = FreeUndo(model_, record, program)
                        row['steps'], _ = undo.run()
                        depth = int(record.end_depth[0])
                        if row['steps'] and row['steps'][0][2] is not None and depth:
                            row['root'] = refusal(undo, record.end_slots[0, depth - 1][None], row['steps'][0][2])
            except Exception as error:      # a measure: say what the inverse did
                row['error'] = f'{type(error).__name__}: {error}'[:300]
            self.rows.append(row)
            return result
        monkeypatch.setattr(model, '_sentence_observation', observed)
        monkeypatch.setattr(SentenceField, 'reconstruct', measured)


def measure(model, said, undone, alone, rule):
    from test_item18_expect import read
    tally, refusals = Counter(), Counter()
    cosines = {0: [], 1: []}
    for word in alone:
        said.rules.clear(); said.rows.clear(); undone.rows.clear()
        with torch.no_grad():
            read(model, word)
            model.End()
        out = said.rows[0] if said.rows else dict(joined=None, silent=True)
        row = undone.rows[0] if undone.rows else dict(steps=[], root=None, error='no reconstruction')
        tally['said back'] += out['joined'] == word and not out.get('silent')
        tally['composed by the rule'] += rule in said.rules
        root = row['steps'][0] if row['steps'] else None
        if root is not None and root[0] == rule:
            tally['root on the walk\'s menu'] += root[3]
            for n, cos in enumerate(root[5]):
                cosines[n].append(cos)
        for reason in row['root'] or ():
            refusals[reason] += 1
        tally['errors'] += row['error'] is not None
        print(f'   {word!r} -> {out["joined"]!r} | rules {said.rules} | root step '
              f'{None if root is None else (root[0], root[3], root[5])} | refused by {row["root"]}'
              + (f' | {row["error"]}' if row['error'] else ''))
    mean = {n: (round(sum(c) / len(c), 3) if c else None) for n, c in cosines.items()}
    back = {n: sum(c >= .99 for c in cosines[n]) for n in cosines}
    return dict(tally), mean, back, dict(refusals)


@pytest.mark.parametrize('rule', [os.environ.get('RULE', 'surface')])
def test_one_rule_taught_to_be_undone(tmp_path, monkeypatch, rule):
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start, 'rule', rule)
    model = model_for(tmp_path, monkeypatch, rules=rule)
    use_eager_reading(monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson + (' '.join(COUNTING[:4]) + '.', ' '.join(COUNTING[4:]) + '.'))
        undone = Undone(model, monkeypatch)
        said = Said(model, monkeypatch)
        alone = tuple(w + '.' for w in WORDS)
        optimizer = model.getOptimizer(lr=.01)
        rounds = int(os.environ.get('ROUNDS', '30')) if rule in TAUGHT else 0
        running, curve = 0, []
        for r in range(rounds + 1):
            if r:
                for w in alone:
                    read(model, w, training=True, optimizer=optimizer)
                    model.End()
            print(f' round {r}:')
            tally, mean, back, refusals = measure(model, said, undone, alone, rule)
            curve.append((r, tally.get('said back', 0), mean[0], mean[1]))
            print(f' round {r} of {rule}, start {start}: said back {tally.get("said back", 0)} of {len(alone)}'
                  f' | {tally} | operand cosine mean {mean} | operands back (cos >= .99) {back}'
                  f' | refused at the root by {refusals}', flush=True)
            running = running + 1 if tally.get('said back', 0) == len(alone) else 0
            if running == 2:
                break
        print('curve (round, said back of 19, mean cosine operand 0, operand 1):', rule, start, curve)
        assert curve
    finally:
        model.End()
