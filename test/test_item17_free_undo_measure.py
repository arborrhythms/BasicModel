"""What the idea alone gives back (round 6d of item 1.7, seventeenth pass,
item 2; §19.135, 1), a measure before the build, tests only. Each sentence is
read eager, no gradient, on the model as it runs. On the sentence's own
record, the recorded forest is undone from its roots in the walk's order
(roots newest first, the right child first) with every inverse given its
parent alone: the walk's own inverse, the menu the walk reads at that parent
(`Generative.inverse_menu`: `Language.reverse_binary_step` with `free=True`
and the walk's bank, `generate_unary_step`) at the recorded rule's action
(`SentenceField.walk_action`, by name). No frame of the journal, no
constituent of the reading, no `.where` or wires of the open field. A
recorded rule the walk has no rule for passes its parent through (`bind`)
or gives nothing, and is counted. Rule by rule: whether each operand comes
back (cosine to the frame's operand) and, for each leaf, its cosine to the
word as read and the word it spells (the walk's writer and the lower
stretch on that leaf alone). And the root's band of `.where` and `.when`
wires (`percept_band`). Nothing is repaired. A record: the start is drawn
afresh and printed; nothing is asserted of what a start gives."""
import os
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, minimal, model_for, nominate, words_of
from test_item17_undo_measure import forest_of, rule_name, text

WORDS = words_of(LESSON) + COUNTING[:6]


def cosine(a, b):
    a, b = a.detach().reshape(-1).float(), b.detach().reshape(-1).float()
    if not bool(a.ne(0).any()) or not bool(b.ne(0).any()):
        return 0.
    return round(float(F.cosine_similarity(a, b, dim=0)), 3)


class FreeUndo:
    """The record's forest undone by the walk's own inverses, each given its
    parent alone."""
    def __init__(self, model, record, program):
        from Generative import walk_bank, propose_actions
        from FormCentroids import prepare_inverse
        self.model, self.record, self.program = model, record, program
        language = self.language = model.languageSpace
        self.bank = bank = walk_bank(model, record.end_slots, record.primed)
        self.require = (not model._aligned_serial_word_mode()
                        or (bank.meaning_start is not None and bank.meaning_start < bank.codes.shape[-1]))
        self.whole = prepare_inverse(bank.form_wholes)
        self.proposed = propose_actions(language, bank.codes.detach(), bank.valid, groups=bank.verb_groups,
            terminal_valid=bank.terminal_valid, limit=model.reconstruction_basis_limit,
            whole_bank=self.whole) if self.require else None
        self.R2, self.R1 = len(language._generate_binary_ops), len(language._generate_unary_ops)

    def menu(self, parent):
        from Generative import inverse_menu
        bank = self.bank
        live = torch.ones(1, dtype=torch.bool, device=parent.device)
        return inverse_menu(self.language, parent, live, basis=bank.codes.detach(), basis_valid=bank.valid,
            priming=bank.weights, limit=self.model.reconstruction_basis_limit,
            case_bank=None if bank.case_bank is None else bank.case_bank.detached(),
            require_symbols=self.require, terminal_valid=bank.terminal_valid, whole_bank=self.whole,
            action_candidates=self.proposed)

    def run(self):
        """``(steps, leaves)``: each step ``(rule, kind, action, legal,
        in_cube, [(operand cosine), ...])`` and each leaf ``(position,
        value, cosine, stop legal)``, in the walk's order."""
        from Generative import in_cube
        from SentenceField import walk_action
        record, language = self.record, self.language
        forest = forest_of(language, self.program)
        depth = int(record.end_depth[0])
        values = record.end_slots[0, :depth]
        positions = self.model._sentence_leaf_positions(record, 0)
        pending = [(node, values[depth - 1 - i:depth - i]) for i, node in enumerate(forest)]
        steps, leaves = [], []
        stop = self.R2 + self.R1
        while pending:
            node, parent = pending.pop()
            if node[0] == 0:
                position = int(positions[node[1]])
                menu = self.menu(parent)
                # The walk's STOP: an exact STOP keeps the parent, else the
                # menu's terminal.
                value = parent if bool(menu.exact[0, stop]) else menu.left[:, stop]
                leaves.append((position, value[0], cosine(value, record.word_values[0, position]),
                               bool(menu.admissible[0, stop])))
                continue
            kind, operation, index = node[:3]
            frame = self.program.operation_values[index]
            name = rule_name(language, kind, operation)
            action = walk_action(language, kind, operation)
            children = node[3:]
            if action is None:
                # The walk has no rule of this name: bind passes its parent
                # through (one child), any other gives nothing.
                out = [parent if name == 'bind' else torch.zeros_like(parent) for _ in children]
                steps.append((name, kind, None, False, bool(in_cube(parent)[0]),
                              [cosine(v, frame[r]) for r, v in enumerate(out)]))
            else:
                menu = self.menu(parent)
                legal = bool(menu.admissible[0, action])
                left, right = menu.left[:, action], menu.right[:, action]
                out = [left, right][:len(children)]
                steps.append((name, kind, action, legal, bool(in_cube(parent)[0]),
                              [cosine(v, frame[r]) for r, v in enumerate(out)]))
            for child, value in zip(children, out):
                pending.append((child, value))
        return steps, leaves


def spelled(model, record, leaves):
    """The walk's writer and the lower stretch on ``leaves`` ``[n, D]``."""
    from Generative import walk_bank
    from OutputOccurrences import generated, realize
    n = len(leaves)
    language = model.languageSpace
    stop = len(language._generate_binary_ops) + len(language._generate_unary_ops)
    bank = walk_bank(model, record.end_slots, record.primed)
    actions = torch.full((1, n), stop, dtype=torch.long, device=leaves.device)
    generation = generated(model, leaves[None], torch.tensor([n]), torch.tensor([False]), bank, (actions,),
                           when=record.open_field.when[:, :1], depths=torch.tensor([n]))
    return text(realize(model, generation).materialize()[0, 0])


class Held:
    def __init__(self, model, monkeypatch):
        import SentenceField
        self.rows = []
        observe, reconstruct = model._sentence_observation, SentenceField.reconstruct
        held = {}

        def observed(*args, **kw):
            result = observe(*args, **kw)
            held['entries'] = result['entries']
            return result
        monkeypatch.setattr(model, '_sentence_observation', observed)

        def measured(model_, record, errors, active, **kw):
            result = reconstruct(model_, record, errors, active, **kw)
            row = dict(steps=[], leaves=[], error=None, root=None)
            try:
                with torch.no_grad():
                    program = held['entries'][0]
                    if program is not None:
                        undo = FreeUndo(model_, record, program)
                        row['steps'], leaves = undo.run()
                        for position, value, cos, legal in leaves:
                            word = spelled(model_, record, value[None])
                            row['leaves'].append((position, cos, legal, word))
                        row['joined'] = spelled(model_, record, torch.stack(
                            [v for _, v, _, _ in sorted(leaves, key=lambda leaf: leaf[0])])) if leaves else ''
                        band = model_.languageSpace.chunk_journal.percept_band
                        depth = int(record.end_depth[0])
                        if band is not None and depth:
                            root = record.end_slots[0, depth - 1]
                            row['root'] = [round(float(v), 3) for v in root[band[0]:band[1]]]
            except Exception as error:      # a measure: say what the inverse did
                row['error'] = f'{type(error).__name__}: {error}'[:300]
            self.rows.append(row)
            return result
        monkeypatch.setattr(SentenceField, 'reconstruct', measured)


def measured(model, held, sentences, rules):
    from test_item18_expect import read
    tally = Counter()
    for sentence in sentences:
        held.rows.clear()
        with torch.no_grad():
            read(model, sentence)
            model.End()
        row = held.rows[0] if held.rows else dict(steps=[], leaves=[], error='no reconstruction', root=None)
        for name, kind, action, legal, cube, cosines in row['steps']:
            entry = rules[name]
            entry['steps'] += 1
            entry['no walk rule'] += action is None
            entry['legal on the menu'] += legal
            entry['parent in the cube'] += cube
            for n, cos in enumerate(cosines):
                entry[f'operand {n} back (cos >= .99)'] += cos >= .99
                entry[f'operand {n} cos sum'] += cos
        words = sentence[:-1].split() + [sentence[-1]]
        said = [word for _, _, _, word in sorted(row['leaves'])]
        tally['said back (joined text is the input)'] += row.get('joined') == sentence
        tally['leaves spell the words'] += said == words
        for position, cos, legal, word in row['leaves']:
            tally['leaves'] += 1
            tally['leaves back (cos >= .99)'] += cos >= .99
            tally['leaves: STOP legal'] += legal
        tally['root band nonzero'] += bool(row['root'] and any(v != 0 for v in row['root']))
        tally['errors'] += row['error'] is not None
        print(f'  {sentence!r}: steps {[(s[0], s[3], s[4], s[5]) for s in row["steps"]]}'
              f' | leaves {[(p, c, l, w) for p, c, l, w in sorted(row["leaves"])]}'
              f' | joined {row.get("joined")!r} | root band {row["root"]}'
              + (f' | {row["error"]}' if row['error'] else ''))
    return dict(tally)


def test_what_the_idea_alone_gives_back(tmp_path, monkeypatch):
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    model = model_for(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson + (' '.join(COUNTING[:4]) + '.', ' '.join(COUNTING[4:]) + '.'))
        held = Held(model, monkeypatch)
        optimizer = model.getOptimizer(lr=.01)
        alone = tuple(w + '.' for w in WORDS)
        rounds = int(os.environ.get('ROUNDS', '3'))
        for r in (0, rounds):
            if r:
                for _ in range(rounds):
                    for w in alone:
                        read(model, w, training=True, optimizer=optimizer)
                        model.End()
            rules = defaultdict(Counter)
            print('taught', r, 'rounds')
            print(' words alone:', measured(model, held, alone, rules))
            print(' lesson sentences:', measured(model, held, ('This is Felix.',) + LESSON[1:4], rules))
            print(' two-word sentences:', measured(model, held, two, rules))
            print(' rules, taught', r, ':', {name: dict(entry) for name, entry in rules.items()})
        assert held.rows
    finally:
        model.End()
