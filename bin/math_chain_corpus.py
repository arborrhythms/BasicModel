"""Data and verification records for the opaque-word arithmetic corpus.

This module belongs to the generator boundary, never to a learner operation.
The September generator owns randomness and equation objects. Only sentence
text and separately held answer words cross into the ordinary Data interface.
"""
from dataclasses import dataclass

from exact import Equation, MathProblemGenerator, num, var


NUMBER_WORDS = tuple('zero one two three four five six seven eight nine ten '
                     'eleven twelve thirteen fourteen fifteen sixteen '
                     'seventeen eighteen nineteen twenty'.split())
HELD_OUT_PAIRS = ((2, 7), (4, 8), (7, 3), (9, 6))
BEYOND_PAIRS = ((11, 2), (12, 3), (2, 11), (3, 12))


def number_word(value):
    if type(value) is not int or not 0 <= value < len(NUMBER_WORDS):
        raise ValueError('the corpus numeral must lie between zero and twenty')
    return NUMBER_WORDS[value]


def render(expr):
    if expr[0] == 'num':
        return number_word(expr[1])
    if expr[0] == 'var':
        return expr[1]
    if expr[0] == 'add':
        return f'{render(expr[1])} plus {render(expr[2])}'
    raise ValueError('math-chain presentations use addition only')


@dataclass(frozen=True)
class ChainDocument:
    key: str
    sentences: tuple
    question: int | None = None
    answer: str | None = None
    pair: tuple | None = None
    steps: tuple = ()
    statement_references: tuple = ()
    questions: tuple = ()

    def target(self, sentence, *, supplied=True):
        """The final answer, or a worked step's intermediate answer."""
        from What import ReferentTarget
        if not supplied:
            return None
        if sentence == self.question and self.answer is not None:
            return ReferentTarget(self.answer)
        for index, answer in self.questions:
            if sentence == index:
                return ReferentTarget(answer)
        return None


class MathChainCorpus:
    """All trained-range pairs except four fixed, unseen pairs.

    Counting covers every successor in the vocabulary in both stated forms.
    A fresh presentation shuffles premises independently of the answer and
    keeps the worked steps chronological. No chosen RNG seed is installed.
    """
    def __init__(self, *, max_addend=10, distractors=2):
        if max_addend != 10 or distractors < 1:
            raise ValueError('the declared gate uses addends through ten and distractors')
        self.generator = MathProblemGenerator(seed=None, range=len(NUMBER_WORDS),
            depths=(1,), distractors=(distractors,), stage=1, operators=('add',))
        self.max_addend, self.distractors = max_addend, int(distractors)
        self.train_pairs = tuple((a, b) for a in range(max_addend + 1)
                                 for b in range(max_addend + 1)
                                 if (a, b) not in HELD_OUT_PAIRS)

    def counting(self):
        records = []
        for value in range(len(NUMBER_WORDS) - 1):
            left, right = number_word(value), number_word(value + 1)
            records.append(ChainDocument(f'count:{value}:copula',
                (f'{left} plus one is {right}.',)))
            records.append(ChainDocument(f'count:{value}:reference',
                (f'{left.capitalize()} plus one.', f'Ref(1) is {right}.'),
                statement_references=((1, 0),)))
        return tuple(records)

    def problem(self, pair, *, split, training, answer_line=True):
        left, right = pair
        answer = number_word(left + right)
        equations = [Equation(var('x'), num(left)),
                     Equation(var('y'), ('add', var('x'), num(right)))]
        for name in ('u', 'v', 'w', 'z')[:self.distractors]:
            equations.append(Equation(var(name), num(
                self.generator.rng.randrange(len(NUMBER_WORDS)))))
        self.generator.rng.shuffle(equations)
        premises = tuple(f'{render(eq.lhs)} is {render(eq.rhs)}.' for eq in equations)
        sentences = [*premises, 'what is y ?']
        steps = tuple((number_word(left + offset), 'one',
                       number_word(left + offset + 1)) for offset in range(right))
        if training:
            if right:
                sentences.append(f'{number_word(right)} is {number_word(right - 1)} plus one.')
            sentences.extend(f'{a} plus {b} is {c}.' for a, b, c in steps)
            if answer_line:
                sentences.append(f'the answer is {answer}.')
        return ChainDocument(f'{split}:pair:{left}:{right}', tuple(sentences),
                             question=len(premises), answer=answer, pair=pair, steps=steps)

    def presentation(self, *, answer_line=True):
        train = [*self.counting(), *(self.problem(pair, split='train', training=True,
                        answer_line=answer_line) for pair in self.train_pairs)]
        self.generator.rng.shuffle(train)
        return dict(train=tuple(train),
                    validation=tuple(self.problem(pair, split='validation', training=False)
                                     for pair in HELD_OUT_PAIRS),
                    test=tuple(self.problem(pair, split='test', training=False)
                               for pair in HELD_OUT_PAIRS),
                    beyond=tuple(self.problem(pair, split='beyond', training=False)
                                 for pair in BEYOND_PAIRS))


def flatten(documents, *, supplied):
    texts, targets, addresses = [], [], []
    for document_index, document in enumerate(documents):
        for sentence, text in enumerate(document.sentences):
            row = len(texts)
            texts.append(text)
            targets.append(document.target(sentence, supplied=supplied))
            addresses.append(dict(row=row, document=document_index, sentence=sentence,
                                  char_start=0, char_end=len(text),
                                  external_id=None, source_time=None))
    return texts, targets, addresses


# -- the frozen graded corpus (item 0.5, plan section 7) -----------------------
#
# Grades follow correction 1 of the math-chain protocol (one-step problems
# first: identity, then a bound name, then plus zero, plus one and longer
# chains), with distractors and deeper chains as grades of their own.
# Worked steps are asked as intermediate questions whose answers are
# supplied (correction 2), so the existing answer supervision scores them.
# Rows are sentences of number words; nothing numeric reaches a learner.

GRADES = (
    ('counting', 'every successor fact, in both stated forms; train only'),
    ('identity', 'x is n. what is x ?'),
    ('alias', 'x is n. y is x. what is y ?'),
    ('plus_zero', 'x is n. y is x plus zero. what is y ?'),
    ('plus_one', 'x is n. y is x plus one. what is y ?'),
    ('plus_k', 'x is a. y is x plus b, b from two to ten, a to ten; '
               'the held-out pairs only in validation and test, the beyond pairs only in beyond'),
    ('distractors', 'plus_k with two unrelated premises'),
    ('depth_two', 'x is a. y is x plus b. z is y plus c. what is z ?'),
    ('depth_three', 'four names in one chain; test only, deeper than any trained chain'),
)
NAMES = ('x', 'y', 'z', 'w')
DISTRACTOR_NAMES = ('u', 'v')


def surface_split(surface):
    """The rule of exact.split_by_surface: sha1 modulo ten, 7/1/2."""
    import hashlib
    h = int(hashlib.sha1(surface.encode()).hexdigest(), 16) % 10
    return 'train' if h < 7 else 'validation' if h < 8 else 'test'


def chain_document(key, start, addends, *, training, rng, distractors=0, name_count=None, shuffle=True):
    """x is start; each later name is the previous plus an addend.

    The question asks for the last name. In training the addend's
    decomposition, each successor step asked as an intermediate question
    and the answer line follow, with every intermediate answer supplied.
    """
    names = NAMES[:1 + len(addends)] if name_count is None else NAMES[:name_count]
    premises = [f'{names[0]} is {number_word(start)}.']
    value, values = start, [start]
    for name, previous, addend in zip(names[1:], names, addends):
        if addend is None:
            premises.append(f'{name} is {previous}.')
            values.append(value)
            continue
        premises.append(f'{name} is {previous} plus {number_word(addend)}.')
        value += addend
        values.append(value)
    if not addends:
        value = start
    for name in DISTRACTOR_NAMES[:distractors]:
        premises.append(f'{name} is {number_word(rng.randrange(len(NUMBER_WORDS)))}.')
    if shuffle:
        rng.shuffle(premises)
    sentences = [*premises, f'what is {names[len(values) - 1]} ?']
    question, answer = len(premises), number_word(value)
    steps, questions = [], []
    if training:
        running = start
        for addend in addends:
            if addend is None:
                continue
            if addend:
                sentences.append(f'{number_word(addend)} is {number_word(addend - 1)} plus one.')
            for _ in range(addend):
                steps.append((number_word(running), 'one', number_word(running + 1)))
                sentences.append(f'what is {number_word(running)} plus one ?')
                questions.append((len(sentences) - 1, number_word(running + 1)))
                running += 1
        sentences.append(f'the answer is {answer}.')
    return ChainDocument(key, tuple(sentences), question=question, answer=answer,
                         pair=(start, *[a for a in addends if a is not None]) or None,
                         steps=tuple(steps), questions=tuple(questions))


def graded(seed):
    """Every grade's documents by split, deterministic in ``seed``."""
    import random
    rng = random.Random(seed)
    corpus = MathChainCorpus()
    out = {}
    out['counting'] = dict(train=list(corpus.counting()))
    def by_surface(name, cases):
        split = {}
        for case in cases:
            start, addends = case
            probe = chain_document('probe', start, addends, training=False, rng=rng, shuffle=False)
            part = surface_split(' '.join(probe.sentences))
            key = f'{name}:{part}:' + ':'.join(str(v) for v in (start, *[a if a is not None else 'is' for a in addends]))
            split.setdefault(part, []).append(chain_document(key, start, addends,
                training=part == 'train', rng=rng))
        return split
    top = len(NUMBER_WORDS) - 1
    out['identity'] = by_surface('identity', [(n, ()) for n in range(top + 1)])
    out['alias'] = by_surface('alias', [(n, (None,)) for n in range(top + 1)])
    out['plus_zero'] = by_surface('plus_zero', [(n, (0,)) for n in range(top + 1)])
    out['plus_one'] = by_surface('plus_one', [(n, (1,)) for n in range(top)])
    def by_pairs(name, distractors):
        split = {}
        for a in range(corpus.max_addend + 1):
            for b in range(2, corpus.max_addend + 1):
                if (a, b) in HELD_OUT_PAIRS:
                    continue
                split.setdefault('train', []).append(chain_document(f'{name}:train:{a}:{b}', a, (b,),
                    training=True, rng=rng, distractors=distractors))
        for part in ('validation', 'test'):
            split[part] = [chain_document(f'{name}:{part}:{a}:{b}', a, (b,), training=False,
                           rng=rng, distractors=distractors) for a, b in HELD_OUT_PAIRS]
        split['beyond'] = [chain_document(f'{name}:beyond:{a}:{b}', a, (b,), training=False,
                           rng=rng, distractors=distractors) for a, b in BEYOND_PAIRS]
        return split
    out['plus_k'] = by_pairs('plus_k', 0)
    out['distractors'] = by_pairs('distractors', len(DISTRACTOR_NAMES))
    out['depth_two'] = by_surface('depth_two', [(a, (b, c)) for a in range(11)
                                                for b in range(1, 6) for c in range(1, 6)])
    out['depth_three'] = dict(test=[chain_document(f'depth_three:test:{a}:{b}:{c}:{d}', a, (b, c, d),
                                    training=False, rng=rng)
                                    for a in range(6) for b in (1, 2) for c in (1, 2) for d in (1, 2)])
    return out


def freeze(seed, out_dir):
    """Write grade-<nn>.jsonl and manifest.json; a test never calls this."""
    import hashlib, json
    from dataclasses import asdict
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    documents = graded(seed)
    grades = []
    for index, (name, description) in enumerate(GRADES):
        rows, counts = [], {}
        for split in ('train', 'validation', 'test', 'beyond'):
            for document in documents[name].get(split, ()):
                row = dict(split=split, document=asdict(document))
                rows.append(json.dumps(row, sort_keys=True, separators=(',', ':')))
                counts[split] = counts.get(split, 0) + 1
        path = out_dir / f'grade-{index:02d}.jsonl'
        text = '\n'.join(rows) + '\n'
        path.write_text(text)
        grades.append(dict(grade=index, name=name, description=description, file=path.name,
                           sha256=hashlib.sha256(text.encode()).hexdigest(), rows=counts))
    manifest = dict(
        schema=1, seed=int(seed),
        command=f'python bin/math_chain_corpus.py freeze --seed {int(seed)} --out data/math',
        generator='bin/math_chain_corpus.py', number_words=list(NUMBER_WORDS),
        held_out_pairs=[list(p) for p in HELD_OUT_PAIRS], beyond_pairs=[list(p) for p in BEYOND_PAIRS],
        corrections=['1: grade order identity, alias, plus zero, plus one, longer chains',
                     '2: worked steps asked as intermediate questions with supplied answers'],
        never_direct='rows are sentences of number words; no digit, numeric code or readout',
        grades=grades)
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='Freeze the graded math corpus.')
    parser.add_argument('command', choices=['freeze'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    manifest = freeze(args.seed, args.out)
    for grade in manifest['grades']:
        print(grade['file'], grade['name'], grade['rows'])
    sys.exit(0)
