"""What "undo" gives with the inverses as they are (round 6d of item 1.7,
sixteenth pass, item 1; a measure before the build, §19.134). Each sentence
is read eager, no gradient, on the model as it runs. On the sentence's own
record, `DerivationReconstruction.reconstruct` undoes the recorded forest.
For each step: the rule, and for each child whether it is the witness (the
operand the rule's inverse is given from the journal's frame, the record of
the way up), recovered from the parent (the idea), or given by no inverse,
with its cosine to the frame's operand. For each leaf: its source (record if
any step on its path gave it as the witness, none if a step had no inverse,
else idea), its cosine to the word read and `_last_derivation_certified`.
The leaves, in the input's order, go through the walk's writer
(`OutputOccurrences.generated`, `realize`) and the lower stretch, and the
text is compared with the input. A record: the start is drawn afresh and
printed; nothing is asserted of what a start gives."""
import os
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, minimal, model_for, nominate, words_of

WORDS = words_of(LESSON) + COUNTING[:6]


def text(t):
    return bytes(int(v) for v in t.detach().reshape(-1).tolist() if 0 < int(v) < 256).decode('latin1')


def rule_name(language, kind, operation):
    rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
    return getattr(rules[operation], 'method_name', type(rules[operation]).__name__)


def roles_of(language, kind, operation):
    """Each child's source as `DerivationReconstruction.inverse_step` makes
    it: 'record' (the journal's frame, the witness), 'idea' (recovered from
    the parent) or 'none' (no inverse: a zero)."""
    if kind == 2:
        op = language._tree_layer(1).unary_ops[operation]
        op = getattr(op, 'gl', op)
        return ('record',) if getattr(op, 'retains_operand', False) else ('idea',)
    op = language._tree_layer(2).ops[operation]
    op = getattr(op, 'gl', op)
    if getattr(op, 'field_operand', None) == 2:
        return ('record',)
    if getattr(op, 'meaning_layout', None) is not None and callable(getattr(op, 'compose_case_meaning', None)):
        return ('record', 'idea')
    if getattr(op, 'meaning_layout', None) is not None and (
            getattr(op, 'meaning_kernel', None) is not None or getattr(op, 'meaning_action', None) is not None or
            getattr(op, 'meaning_basis', None) is not None or op.meaning_rule == 'sum'):
        right = getattr(op, 'meaning_action', None) is not None or op.meaning_head_role == 0
        return ('idea', 'record') if right else ('record', 'idea')
    inverse = getattr(op, 'inverse_kind', None)
    if inverse == 'search':
        return ('none', 'none')
    return ('record', 'idea') if inverse in ('right', 'surface') else ('idea', 'record')


def forest_of(language, program):
    """The recorded forest, built as `DerivationReconstruction.reconstruct`
    builds it."""
    from ReadingIndex import steps as recorded_steps
    if program.tree is not None:
        return list(program.tree.forest())
    forest = []
    for index, (kind, operation, word) in enumerate(recorded_steps(program)):
        if kind == 0:
            forest.append((kind, word))
        elif kind == 1:
            if language._compose_binary_rules[operation].method_name == 'bind':
                forest[-1] = (kind, operation, index, forest[-1])
            else:
                right, left = forest.pop(), forest.pop()
                forest.append((kind, operation, index, left, right))
        elif kind == 2:
            position = len(forest) - 1 - max(0, word)
            forest[position] = (kind, operation, index, forest[position])
        elif kind == 3:
            forest[-2:] = reversed(forest[-2:])
    return forest


def leaf_sources(language, forest):
    """Each leaf's word index and source along its path from its root: a
    witness child is the record's whatever its parent was, a child of no
    inverse is none, a recovered child is what its parent was."""
    out = {}

    def walk(node, source):
        if node[0] == 0:
            out[node[1]] = source
            return
        roles = roles_of(language, node[0], node[1])
        for child, role in zip(node[3:], roles):
            walk(child, source if role == 'idea' else role)
    for root in forest:
        walk(root, 'idea')
    return out


class Undo:
    def __init__(self, model, monkeypatch):
        import DerivationReconstruction as DR
        import SentenceField
        self.model, self.rows = model, []
        language = model.languageSpace
        steps = []
        inverse = DR.inverse_step

        def logged(language, kind, operation, parent, frame, **kw):
            children = inverse(language, kind, operation, parent, frame, **kw)
            roles = []
            for role, ((value, _domain), source) in enumerate(zip(children, roles_of(language, kind, operation))):
                operand = frame[role:role + 1].detach()
                cos = float(F.cosine_similarity(value.detach().reshape(-1), operand.reshape(-1), dim=0))
                roles.append((source, round(cos, 3)))
            steps.append((rule_name(language, kind, operation), tuple(roles)))
            return children
        monkeypatch.setattr(DR, 'inverse_step', logged)

        observe, reconstruct = model._sentence_observation, SentenceField.reconstruct
        held = {}

        def observed(*args, **kw):
            result = observe(*args, **kw)
            held['entries'] = result['entries']
            return result
        monkeypatch.setattr(model, '_sentence_observation', observed)

        def undone(model_, record, errors, active, **kw):
            result = reconstruct(model_, record, errors, active, **kw)
            steps.clear()
            row = dict(steps=[], leaves=[], text=None, error=None)
            try:
                with torch.no_grad():
                    program = held['entries'][0]
                    positions = model_._sentence_leaf_positions(record, 0)
                    sources = ({} if program is None else
                               leaf_sources(language, forest_of(language, program)))
                    recovered, _missing, defined = DR.reconstruct(
                        language, record, held['entries'], lambda b: model_._sentence_leaf_positions(record, b))
                    certified = language._last_derivation_certified
                row['steps'] = list(steps)
                valid = record.word_valid[0].nonzero().flatten().tolist()
                at = {int(p): n for n, p in enumerate(positions.tolist())}
                for w in valid:
                    cos = float(F.cosine_similarity(recovered[0, w].detach(), record.word_values[0, w].detach(), dim=0))
                    row['leaves'].append(dict(position=w, cosine=round(cos, 3), certified=bool(certified[0, w].all()),
                                              source=sources.get(at.get(w), 'idea')))
                row['text'] = self.writer(record, recovered[:, valid].detach())
            except Exception as error:      # a measure: say what the inverse did
                row['error'] = f'{type(error).__name__}: {error}'[:200]
            self.rows.append(row)
            return result
        monkeypatch.setattr(SentenceField, 'reconstruct', undone)

    def writer(self, record, leaves):
        """The walk's writer and the lower stretch on ``leaves`` ``[1, n, D]``."""
        from Generative import walk_bank
        from OutputOccurrences import generated, realize
        model = self.model
        n = leaves.shape[1]
        language = model.languageSpace
        stop = len(language._generate_binary_ops) + len(language._generate_unary_ops)
        bank = walk_bank(model, record.end_slots, record.primed)
        actions = torch.full((1, n), stop, dtype=torch.long, device=leaves.device)
        generation = generated(model, leaves, torch.tensor([n]), torch.tensor([False]), bank, (actions,),
                               when=record.open_field.when[:, :1], depths=torch.tensor([n]))
        return text(realize(model, generation).materialize()[0, 0])


def measured(model, undo, sentences, rules):
    from test_item18_expect import read
    tally = Counter()
    for sentence in sentences:
        undo.rows.clear()
        with torch.no_grad():
            read(model, sentence)
            model.End()
        row = undo.rows[0] if undo.rows else dict(steps=[], leaves=[], text=None, error='no reconstruction')
        for name, roles in row['steps']:
            entry = rules[name]
            entry['steps'] += 1
            for n, (source, cos) in enumerate(roles):
                entry[f'operand {n}: {source}'] += 1
                entry[f'operand {n}: back (cos >= .99)'] += cos >= .99
        tally['text is the input'] += row['text'] == sentence
        for leaf in row['leaves']:
            tally['leaves'] += 1
            tally[f'leaves from the {leaf["source"]}'] += 1
            tally['leaves back (cos >= .99)'] += leaf['cosine'] >= .99
            tally['leaves certified'] += leaf['certified']
        steps = Counter(name for name, _ in row['steps'])
        print(f'  {sentence!r}: steps {[(n, r) for n, r in dict.fromkeys(row["steps"])]} x{dict(steps)}'
              f' | leaves {[(l["source"], l["cosine"], l["certified"]) for l in row["leaves"]]}'
              f' | text {row["text"]!r}' + (f' | {row["error"]}' if row['error'] else ''))
    return dict(tally)


def test_what_undo_gives(tmp_path, monkeypatch):
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
        undo = Undo(model, monkeypatch)
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
            print(' words alone:', measured(model, undo, alone, rules))
            print(' lesson sentences:', measured(model, undo, ('This is Felix.',) + LESSON[1:4], rules))
            print(' two-word sentences:', measured(model, undo, two, rules))
            print(' rules, taught', r, ':', {name: dict(entry) for name, entry in rules.items()})
        assert undo.rows
    finally:
        model.End()
