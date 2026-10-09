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

    def target(self, sentence, *, supplied=True):
        return self.answer if supplied and sentence == self.question else None


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
